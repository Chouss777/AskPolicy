"""Trusted offline hybrid retrieval over matching unpublished role bundles."""

from dataclasses import dataclass
from datetime import date
from math import isfinite

from ..auth.context import AuthenticatedContext
from ..auth.jwt import ReadRole
from ..security.acl import can_read_document
from ..security.index_selection import select_role_index
from . import role_bm25 as lexical, role_faiss as dense
from .rrf import reciprocal_rank_fusion


class HybridError(ValueError):
    """Controlled failure without evidence or underlying exception details."""


@dataclass(frozen=True, slots=True)
class HybridHit:
    child: lexical.IndexedChild
    candidate_snapshot_id: str
    active_role: ReadRole
    rrf_score: float
    bm25_rank: int | None
    faiss_rank: int | None


def _authorized(context: AuthenticatedContext, child: lexical.IndexedChild) -> bool:
    meta = child.document.metadata
    return can_read_document(context, classification=meta.classification, allowed_roles=meta.allowed_roles)


def _gate(bm25, faiss, context, expected_snapshot_id, validity_date):
    role = select_role_index(context).role
    dense.verify_matching_bundles(bm25, faiss)
    left = lexical._verified_index(bm25, role)
    right = next(i for i in faiss.indexes if i.metadata.role == role)
    if type(expected_snapshot_id) is not str or expected_snapshot_id != bm25.snapshot.candidate_snapshot_id:
        raise HybridError("SNAPSHOT_MISMATCH")
    if type(validity_date) is not date:
        raise HybridError("INVALID_VALIDITY_DATE")
    for index in (left, right):
        meta = index.metadata
        if meta.next_transition_date is not None and validity_date >= meta.next_transition_date:
            raise HybridError("STALE_INDEX")
        if validity_date != meta.validity_date:
            raise HybridError("VALIDITY_DATE_MISMATCH")
    # Compare both original catalogs before scoring. This binds complete policy
    # metadata and parent identities, beyond equality of child membership alone.
    for child_id in left.metadata.indexed_child_ids:
        child = bm25._draft.children[child_id]
        if (
            child != faiss._draft.children[child_id]
            or child.document != bm25._draft.documents[child.section.document_id]
            or child.section.document_id not in bm25.eligibility.eligible_document_ids
            or not _authorized(context, child)
        ):
            raise HybridError("CATALOG_ACCESS_MISMATCH")
    return role, left, right


def _validate_hits(hits, bundle, index, context, *, is_dense):
    expected_type = dense.DenseHit if is_dense else lexical.RoleBM25Hit
    if type(hits) is not tuple or len(hits) > 30:
        raise HybridError("HIT_INTEGRITY_FAILED")
    seen = {}
    for hit in hits:
        if type(hit) is not expected_type or type(hit.child) is not lexical.IndexedChild:
            raise HybridError("HIT_INTEGRITY_FAILED")
        child = hit.child
        if (
            type(child.child_id) is not str or child.child_id in seen
            or child.child_id not in index.metadata.indexed_child_ids
            or child != bundle._draft.children.get(child.child_id)
            or hit.candidate_snapshot_id != index.metadata.candidate_snapshot_id
            or child.section.document_id not in index.metadata.indexed_document_ids
            or not _authorized(context, child)
            or type(hit.score) is not float or not isfinite(hit.score)
        ):
            raise HybridError("HIT_INTEGRITY_FAILED")
        if is_dense:
            if hit.segment not in index.metadata.rows or hit.segment.child_id != child.child_id:
                raise HybridError("HIT_INTEGRITY_FAILED")
        elif hit.score <= 0:
            raise HybridError("HIT_INTEGRITY_FAILED")
        seen[child.child_id] = child
    return seen


def search_role_hybrid(
    bm25_bundle: lexical.RoleBM25Bundle, faiss_bundle: dense.RoleFaissBundle,
    context: AuthenticatedContext, *, query: str, expected_snapshot_id: str, validity_date: date,
) -> tuple[HybridHit, ...]:
    """Validate both scopes before scoring, then fuse ranks and reauthorize."""
    try:
        role, left_index, right_index = _gate(
            bm25_bundle, faiss_bundle, context, expected_snapshot_id, validity_date,
        )
        if type(query) is not str or not query.strip():
            raise HybridError("INVALID_QUERY")
        options = dict(query=query, expected_snapshot_id=expected_snapshot_id, validity_date=validity_date)
        left = lexical.search_role_bm25(bm25_bundle, context, **options)
        right = dense.search_role_faiss(faiss_bundle, context, **options)
        left_children = _validate_hits(left, bm25_bundle, left_index, context, is_dense=False)
        right_children = _validate_hits(right, faiss_bundle, right_index, context, is_dense=True)
        for child_id in left_children.keys() & right_children.keys():
            if left_children[child_id] != right_children[child_id]:
                raise HybridError("HIT_METADATA_CONFLICT")
        fused = reciprocal_rank_fusion(tuple(left_children), tuple(right_children))
        hits = []
        for identity in fused:
            child = bm25_bundle._draft.children[identity.child_id]
            if not _authorized(context, child):
                raise HybridError("FUSED_ACCESS_DENIED")
            hits.append(HybridHit(child, expected_snapshot_id, role, identity.score,
                                  identity.bm25_rank, identity.faiss_rank))
        return tuple(hits)
    except HybridError:
        raise
    except Exception:
        # Search engines can fail for implementation-specific reasons. Preserve
        # all-or-nothing behavior and keep those details inside this boundary.
        raise HybridError("HYBRID_RETRIEVAL_FAILED") from None
