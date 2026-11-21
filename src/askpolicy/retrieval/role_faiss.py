"""Exact dense search over four separately built, unpublished role indexes."""

from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
import json
import sqlite3
from typing import get_args

import faiss
import numpy as np

from ..auth.context import AuthenticatedContext
from ..auth.jwt import ReadRole
from ..security.acl import can_read_document
from ..security.index_selection import IndexSelectionError, select_role_index
from ..storage.snapshot import DraftSnapshotConfig, DraftSnapshotIdentity
from . import role_bm25 as lexical
from .bm25_core import read_draft_chunks
from .e5_encoder import (
    ARTIFACT_FINGERPRINT, EMBEDDING_DIMENSION, MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION,
    E5Encoder, E5EncodingError,
)
from .e5_passage_split import SPLIT_VERSION, Passage, Segment, split_passage


ROLES = get_args(ReadRole)
TOP_K = 30
BATCH_SIZE = 4


class DenseIndexError(ValueError):
    """Content-free failure at the offline dense retrieval boundary."""


def _digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class DenseConfiguration:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    artifact_fingerprint: str = ARTIFACT_FINGERPRINT
    dimension: int = EMBEDDING_DIMENSION
    normalization: str = "L2-unit-float32"
    index_type: str = "IndexFlatIP"
    faiss_version: str = faiss.__version__
    splitting: str = SPLIT_VERSION
    max_prefixed_tokens: int = MAX_INPUT_TOKENS
    batch_size: int = BATCH_SIZE
    top_k_children: int = TOP_K

    @property
    def fingerprint(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class DenseMetadata:
    candidate_snapshot_id: str
    source_manifest_hash: str
    snapshot_config_hash: str
    validity_date: date
    next_transition_date: date | None
    role: ReadRole
    configuration: DenseConfiguration
    indexed_document_ids: tuple[str, ...]
    indexed_child_ids: tuple[str, ...]
    rows: tuple[Segment, ...]
    catalog_hash: str
    vector_hash: str

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["validity_date"] = self.validity_date.isoformat()
        payload["next_transition_date"] = self.next_transition_date and self.next_transition_date.isoformat()
        return _digest(payload)


@dataclass(frozen=True, slots=True)
class _BoundFlatIndex:
    fingerprint: str
    engine: faiss.IndexFlatIP = field(repr=False)


@dataclass(frozen=True, slots=True)
class RoleFaissIndex:
    metadata: DenseMetadata
    _flat: _BoundFlatIndex = field(repr=False)


@dataclass(frozen=True, slots=True)
class RoleFaissBundle:
    indexes: tuple[RoleFaissIndex, ...]
    _draft: lexical._DraftBinding = field(repr=False)
    _encoder: E5Encoder = field(repr=False)

    @property
    def snapshot(self) -> DraftSnapshotIdentity:
        return self._draft.snapshot


@dataclass(frozen=True, slots=True)
class DenseHit:
    child: lexical.IndexedChild
    segment: Segment
    candidate_snapshot_id: str
    score: float


def _check_encoder(encoder: E5Encoder) -> None:
    if (
        type(encoder) is not E5Encoder or encoder.revision != MODEL_REVISION
        or encoder.model_id != MODEL_ID or encoder.dimension != EMBEDDING_DIMENSION
        or getattr(encoder, "artifact_fingerprint", None) != ARTIFACT_FINGERPRINT
        or encoder.max_input_tokens != MAX_INPUT_TOKENS
    ):
        raise DenseIndexError("ENCODER_BINDING_MISMATCH")


def validate_vectors(vectors: np.ndarray, count: int) -> None:
    if (
        not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32
        or vectors.shape != (count, EMBEDDING_DIMENSION) or not np.isfinite(vectors).all()
    ):
        raise DenseIndexError("INVALID_VECTORS")
    norms = np.linalg.norm(vectors, axis=1)
    if not ((norms >= 0.999) & (norms <= 1.001)).all():
        raise DenseIndexError("INVALID_VECTOR_NORM")


def _vector_hash(engine: faiss.IndexFlatIP) -> str:
    # FlatIP stores the exact float32 rows, in insertion order.
    digest = sha256()
    for start in range(0, engine.ntotal, 128):
        vectors = engine.reconstruct_n(start, min(128, engine.ntotal - start))
        digest.update(vectors.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _add_batch(engine: faiss.IndexFlatIP, encoder: E5Encoder, passages: list[Passage]) -> None:
    result = encoder.encode_passages([passage.text for passage in passages])
    if result.token_counts != tuple(p.segment.token_count for p in passages):
        raise DenseIndexError("TOKEN_COUNT_MISMATCH")
    validate_vectors(result.vectors, len(passages))
    engine.add(result.vectors)


def _catalog_hash(allowed, children) -> str:
    documents = [(key, doc.source_path, doc.source_sha256, doc.metadata.model_dump(mode="json"))
                 for key, doc in sorted(allowed.items())]
    identities = [(c.child_id, asdict(c.section), c.start) for c in children]
    if any(c.document != allowed[c.section.document_id] for c in children):
        raise DenseIndexError("INVALID_CHILD_MEMBERSHIP")
    return _digest([documents, identities])


def _build_index(
    image: sqlite3.Connection, draft: lexical._DraftBinding, role: ReadRole, encoder: E5Encoder,
) -> RoleFaissIndex:
    allowed = lexical._authorized_documents(draft, role)
    chunks = read_draft_chunks(image, allowed)
    expected = tuple(child for child in draft.children.values() if child.section.document_id in allowed)
    actual = tuple(lexical.IndexedChild(c.child_id, c.section, c.document, c.start) for c in chunks)
    if actual != expected:
        raise DenseIndexError("INVALID_CHILD_MEMBERSHIP")

    engine = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    rows = []
    batch = []
    for chunk in chunks:
        cursor = chunk.start
        for passage in split_passage(chunk, encoder.passage_token_count):
            span = passage.segment
            if (
                span.child_id != chunk.child_id or span.start != cursor
                or not cursor < span.end <= chunk.start + len(chunk.text)
                or passage.text != chunk.text[cursor - chunk.start:span.end - chunk.start]
            ):
                raise DenseIndexError("INVALID_SOURCE_SPAN")
            cursor = span.end
            rows.append(span)
            batch.append(passage)
            if len(batch) == BATCH_SIZE:
                _add_batch(engine, encoder, batch)
                batch.clear()
        if cursor != chunk.start + len(chunk.text):
            raise DenseIndexError("INCOMPLETE_SOURCE_SPAN")
    if batch:
        _add_batch(engine, encoder, batch)
    if engine.ntotal != len(rows) or len({row.segment_id for row in rows}) != len(rows):
        raise DenseIndexError("INVALID_ROW_MAPPING")
    meta = DenseMetadata(
        draft.snapshot.candidate_snapshot_id, draft.snapshot.source_manifest_hash, draft.snapshot.config_hash,
        draft.eligibility.validity_date, draft.eligibility.next_transition_date, role, DenseConfiguration(),
        tuple(sorted(allowed)), tuple(sorted(c.child_id for c in expected)), tuple(rows),
        _catalog_hash(allowed, expected), _vector_hash(engine),
    )
    return RoleFaissIndex(meta, _BoundFlatIndex(meta.fingerprint, engine))


def build_role_faiss_bundle(
    connection: sqlite3.Connection, *, manifest: bytes, snapshot_config: DraftSnapshotConfig,
    validity_date: date, encoder: E5Encoder, expected_snapshot_id: str | None = None,
) -> RoleFaissBundle:
    _check_encoder(encoder)
    try:
        with lexical.validated_role_draft(
            connection, manifest=manifest, snapshot_config=snapshot_config, validity_date=validity_date,
            expected_snapshot_id=expected_snapshot_id,
        ) as (image, draft):
            indexes = tuple(_build_index(image, draft, role, encoder) for role in ROLES)
            return RoleFaissBundle(indexes, draft, encoder)
    except (lexical.RoleBM25Error, E5EncodingError) as error:
        raise DenseIndexError(error.code) from None
    except DenseIndexError:
        raise
    except (sqlite3.Error, ValueError, RuntimeError, OverflowError):
        raise DenseIndexError("INDEX_BUILD_FAILED") from None


def _verify_bundle(bundle: RoleFaissBundle) -> dict[ReadRole, RoleFaissIndex]:
    try:
        return _validated_indexes(bundle)
    except DenseIndexError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise DenseIndexError("MALFORMED_INDEX") from None


def _validated_indexes(bundle: RoleFaissBundle) -> dict[ReadRole, RoleFaissIndex]:
    if type(bundle) is not RoleFaissBundle or type(bundle._draft) is not lexical._DraftBinding:
        raise DenseIndexError("INVALID_BUNDLE")
    _check_encoder(bundle._encoder)
    if type(bundle.indexes) is not tuple or any(
        type(index) is not RoleFaissIndex or type(index.metadata) is not DenseMetadata for index in bundle.indexes
    ):
        raise DenseIndexError("INVALID_ROLE_SET")
    indexes = {index.metadata.role: index for index in bundle.indexes}
    if len(bundle.indexes) != len(ROLES) or set(indexes) != set(ROLES):
        raise DenseIndexError("INVALID_ROLE_SET")
    draft = bundle._draft
    for role, index in indexes.items():
        meta, bound = index.metadata, index._flat
        allowed = lexical._authorized_documents(draft, role)
        children = tuple(c for c in draft.children.values() if c.section.document_id in allowed)
        if (
            meta.configuration != DenseConfiguration()
            or meta.candidate_snapshot_id != draft.snapshot.candidate_snapshot_id
            or meta.source_manifest_hash != draft.snapshot.source_manifest_hash
            or meta.snapshot_config_hash != draft.snapshot.config_hash
            or meta.validity_date != draft.eligibility.validity_date
            or meta.next_transition_date != draft.eligibility.next_transition_date
            or meta.indexed_document_ids != tuple(sorted(allowed))
            or meta.indexed_child_ids != tuple(sorted(c.child_id for c in children))
            or meta.catalog_hash != _catalog_hash(allowed, children)
            or type(bound) is not _BoundFlatIndex or bound.fingerprint != meta.fingerprint
        ):
            raise DenseIndexError("INDEX_METADATA_MISMATCH")
        engine = bound.engine
        if (
            type(engine) is not faiss.IndexFlatIP or engine.metric_type != faiss.METRIC_INNER_PRODUCT
            or engine.d != EMBEDDING_DIMENSION or engine.ntotal != len(meta.rows)
            or _vector_hash(engine) != meta.vector_hash
        ):
            raise DenseIndexError("INDEX_VECTOR_MISMATCH")
    return indexes


def verify_matching_bundles(bm25: lexical.RoleBM25Bundle, dense: RoleFaissBundle) -> None:
    """Independently validated images must describe exactly the same evidence."""
    indexes = _verify_bundle(dense)
    if (type(bm25) is not lexical.RoleBM25Bundle or type(bm25._draft) is not lexical._DraftBinding
            or bm25.snapshot != dense.snapshot):
        raise DenseIndexError("BUNDLE_SNAPSHOT_MISMATCH")
    try:
        lexical._verified_index(bm25, ROLES[0])  # Verifies every lexical role binding.
    except (AttributeError, KeyError, TypeError, ValueError):
        raise DenseIndexError("LEXICAL_BINDING_MISMATCH") from None
    for other in bm25.indexes:
        left, right = other.metadata, indexes[other.metadata.role].metadata
        if (
            left.validity_date != right.validity_date or left.next_transition_date != right.next_transition_date
            or left.indexed_document_ids != right.indexed_document_ids
            or left.indexed_child_ids != right.indexed_child_ids
        ):
            raise DenseIndexError("BUNDLE_MEMBERSHIP_MISMATCH")


def _rank_children(index: RoleFaissIndex, query: np.ndarray) -> list[tuple[int, float]]:
    rows = index.metadata.rows
    if not rows:
        return []
    # Exhaustive within this role. Includes every tied score at the cutoff and
    # lets a child with many segments consume exactly one child-budget slot.
    scores, positions = index._flat.engine.search(query, len(rows))
    if (
        not isinstance(scores, np.ndarray) or not isinstance(positions, np.ndarray)
        or scores.dtype != np.float32 or scores.shape != (1, len(rows)) or positions.shape != scores.shape
        or not np.isfinite(scores).all() or positions.dtype.kind not in "iu"
        or set(positions[0].tolist()) != set(range(len(rows)))
    ):
        raise DenseIndexError("INVALID_SEARCH_ROWS")
    ranked = sorted(zip(positions[0].tolist(), scores[0].tolist()), key=lambda hit: (
        -hit[1], rows[hit[0]].child_id, rows[hit[0]].segment_id,
    ))
    seen = set()
    distinct = []
    for position, score in ranked:
        child_id = rows[position].child_id
        if child_id not in seen:
            seen.add(child_id)
            distinct.append((position, score))
            if len(distinct) == TOP_K:
                break
    return distinct


def _can_read(context, document) -> bool:
    meta = document.metadata
    return can_read_document(context, classification=meta.classification, allowed_roles=meta.allowed_roles)


def search_role_faiss(
    bundle: RoleFaissBundle, context: AuthenticatedContext, *, query: str,
    expected_snapshot_id: str, validity_date: date,
) -> tuple[DenseHit, ...]:
    try:
        role = select_role_index(context).role
    except IndexSelectionError:
        raise DenseIndexError("UNVERIFIED_CONTEXT") from None
    index = _verify_bundle(bundle)[role]
    meta = index.metadata
    if type(expected_snapshot_id) is not str or expected_snapshot_id != meta.candidate_snapshot_id:
        raise DenseIndexError("SNAPSHOT_MISMATCH")
    if type(validity_date) is not date:
        raise DenseIndexError("INVALID_VALIDITY_DATE")
    if meta.next_transition_date and validity_date >= meta.next_transition_date:
        raise DenseIndexError("STALE_INDEX")
    if validity_date != meta.validity_date:
        raise DenseIndexError("VALIDITY_DATE_MISMATCH")
    if not all(_can_read(context, bundle._draft.documents[doc]) for doc in meta.indexed_document_ids):
        raise DenseIndexError("INDEX_ACCESS_DENIED")
    try:
        vector = bundle._encoder.encode_queries([query]).vectors
        validate_vectors(vector, 1)
        ranked = _rank_children(index, vector)
    except E5EncodingError as error:
        raise DenseIndexError(error.code) from None
    except RuntimeError:
        raise DenseIndexError("INDEX_SEARCH_FAILED") from None
    hits = []
    for position, score in ranked:
        segment = meta.rows[position]
        child = bundle._draft.children[segment.child_id]
        if (
            child.child_id != segment.child_id or child.section.document_id not in meta.indexed_document_ids
            or child.document != bundle._draft.documents[child.section.document_id]
            or not _can_read(context, child.document)
        ):
            raise DenseIndexError("HIT_INTEGRITY_FAILED")
        hits.append(DenseHit(child, segment, meta.candidate_snapshot_id, score))
    return tuple(hits)
