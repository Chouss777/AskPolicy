"""Internal exact-text boundary for an already verified published release.

The server supplies the trusted root and date. Hits are checked against both
role catalogs, never treated as a list of document-read permissions. Duplicate
children are rejected. No active pointer, draft, cache or network is consulted.
"""

from contextlib import closing
from dataclasses import dataclass, field
from datetime import date
import json
from math import isfinite
from pathlib import Path
import sqlite3

from ..auth.context import AuthenticatedContext
from ..auth.jwt import ReadRole
from ..ingestion.metadata import PolicyMetadata, parse_policy_metadata
from ..security.acl import can_read_document
from ..storage import index_store as store
from ..storage.eligibility import select_policy_versions
from .bm25_core import DraftDocument, SectionIdentity
from .hybrid import HybridHit, _gate
from .role_bm25 import IndexedChild


class EvidenceHydrationError(ValueError):
    """Stable content-free failure, with no underlying exception attached."""


@dataclass(frozen=True, slots=True)
class AuthorizedPassage:
    release_id: str
    snapshot_id: str
    active_role: ReadRole
    child_id: str
    parent_id: str
    document_id: str
    start: int
    end: int
    text: str = field(repr=False)


def _checked_hits(handle, context, hits, validity_date):
    if (type(handle) is not store.PublishedSnapshot or handle.status != "PUBLISHED"
        or not store.valid_id(handle.release_id) or not store.valid_id(handle.manifest_hash)
        or type(hits) is not tuple or len(hits) > 20):
        raise ValueError
    snapshot = handle.bm25.snapshot.candidate_snapshot_id
    if not store.valid_id(snapshot):
        raise ValueError
    role, index, _ = _gate(handle.bm25, handle.faiss, context, snapshot, validity_date)
    seen = set()
    for hit in hits:
        if type(hit) is not HybridHit or type(hit.child) is not IndexedChild:
            raise ValueError
        child = hit.child
        if (type(child.section) is not SectionIdentity or type(child.document) is not DraftDocument
            or type(child.document.metadata) is not PolicyMetadata or type(child.start) is not int
            or type(hit.active_role) is not str or type(hit.candidate_snapshot_id) is not str):
            raise ValueError
        ranks = (hit.bm25_rank, hit.faiss_rank)
        if (type(child.child_id) is not str or child.child_id in seen
            or child.child_id not in index.metadata.indexed_child_ids
            or child != handle.bm25._draft.children.get(child.child_id)
            or child != handle.faiss._draft.children.get(child.child_id)
            or hit.candidate_snapshot_id != snapshot or hit.active_role != role
            or not any(rank is not None for rank in ranks)
            or any(rank is not None and (type(rank) is not int or not 1 <= rank <= 30) for rank in ranks)
            or type(hit.rrf_score) is not float or not isfinite(hit.rrf_score)
            or hit.rrf_score != sum(1 / (60 + rank) for rank in ranks if rank is not None)
            or not _authorized(context, child)):
            raise ValueError
        seen.add(child.child_id)
    return role, snapshot


def _authorized(context, child):
    meta = child.document.metadata
    return can_read_document(context, classification=meta.classification, allowed_roles=meta.allowed_roles)


def _verified_bytes(handle, release_root):
    directory = store.safe_path(Path(release_root) / "releases" / handle.release_id)
    if not directory.is_dir():
        raise ValueError
    raw_manifest = store.safe_path(directory / "manifest.json").read_bytes()
    record = store.read_json(raw_manifest)
    if (store.digest(raw_manifest) != handle.manifest_hash or type(record) is not dict
        or type(record.get("format")) is not int or record["format"] != 1
        or record.get("artifact_paths") != store.artifact_paths()
        or type(record.get("artifacts")) is not dict
        or set(record["artifacts"]) != set(store.artifact_paths())
        or not all(store.valid_id(value) for value in record["artifacts"].values())
        or store._identity(record) != handle.release_id):
        raise ValueError
    binding = {k: v for k, v in record.items() if k not in ("artifacts", "artifact_paths")}
    if store.canonical(binding) != store.canonical(store._binding(handle.bm25, handle.faiss)):
        raise ValueError
    path = store.safe_path(directory / "sqlite/corpus.sqlite")
    # Publication produces a standalone rollback-journal image. Sidecars are
    # never opened or trusted, including a journal left by an outside writer.
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = store.safe_path(path.with_name(path.name + suffix))
        if sidecar.exists():
            raise ValueError
    raw = path.read_bytes()  # The only read of the published SQLite artifact.
    if (store.digest(raw) != record["artifacts"]["sqlite"]
        or len(raw) < 100 or raw[:16] != b"SQLite format 3\x00" or raw[18:20] != b"\x01\x01"):
        raise ValueError
    return raw


def _readonly(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if (action == sqlite3.SQLITE_READ and database == "main" and trigger is None
        and table in {"draft_documents", "draft_parents", "draft_children"}):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _requested_documents(db, ids):
    if not ids:
        return {}
    fields = tuple(PolicyMetadata.model_fields)
    placeholders = ", ".join("?" for _ in ids)
    rows = db.execute(
        "SELECT " + ", ".join((*fields, "source_path", "source_sha256"))
        + f" FROM main.draft_documents WHERE document_id IN ({placeholders})", ids,
    ).fetchall()
    documents = {}
    for row in rows:
        metadata = dict(zip(fields, row[:-2], strict=True))
        roles = json.loads(metadata["allowed_roles"])
        if type(roles) is not list:
            raise ValueError
        metadata["allowed_roles"] = tuple(roles)
        for key in ("effective_date", "expires_at"):
            if metadata[key] is not None:
                metadata[key] = date.fromisoformat(metadata[key])
        meta = PolicyMetadata.model_validate(metadata)
        if meta.document_id in documents:
            raise ValueError
        documents[meta.document_id] = DraftDocument(meta, row[-2], row[-1])
    if set(documents) != set(ids):
        raise ValueError
    return documents


def _selected_rows(db, handle, context, hits, validity_date):
    # Revalidate the complete pinned metadata/version graph without a global
    # SQL read. Only requested owning documents are selected from the image.
    pinned = handle.bm25._draft.documents
    eligibility = select_policy_versions([d.metadata for d in pinned.values()], validity_date=validity_date)
    if eligibility != handle.bm25.eligibility:
        raise ValueError
    ids = tuple(sorted({hit.child.section.document_id for hit in hits}))
    documents = _requested_documents(db, ids)
    if any(document != pinned.get(key) for key, document in documents.items()):
        raise ValueError
    checked = []
    for hit in hits:
        child = hit.child
        document_id = child.section.document_id
        if (document_id not in eligibility.eligible_document_ids or documents.get(document_id) != child.document
            or not _authorized(context, child)):
            raise ValueError
        rows = db.execute(
            "SELECT c.child_id, c.document_id, c.parent_id, c.start, c.end, "
            "p.parent_id, p.document_id, p.heading, p.start, p.end "
            "FROM main.draft_children c JOIN main.draft_parents p "
            "ON p.parent_id = c.parent_id AND p.document_id = c.document_id "
            "WHERE c.child_id = ? AND c.document_id = ?", (child.child_id, document_id),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError
        row = rows[0]
        cid, did, pid, start, end, parent_id, parent_doc, heading, pstart, pend = row
        if (cid != child.child_id or did != document_id or pid != child.section.parent_id
            or parent_id != pid or parent_doc != did or heading != child.section.heading
            or any(type(value) is not int for value in (start, end, pstart, pend))
            or start != child.start or not 0 <= pstart <= start < end <= pend):
            raise ValueError
        checked.append((hit, row))
    return checked


def _hydrate(handle, context, hits, release_root, validity_date):
    role, snapshot = _checked_hits(handle, context, hits, validity_date)
    raw = _verified_bytes(handle, release_root)
    # deserialize copies the exact checked immutable bytes into a private
    # in-memory image. No filename is reopened, and no disk sidecars are used.
    # Unsupported SQLite builds fail closed rather than falling back to disk.
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(raw)
        db.execute("PRAGMA query_only = ON")
        db.execute("PRAGMA trusted_schema = OFF")
        db.set_authorizer(_readonly)
        checked = _selected_rows(db, handle, context, hits, validity_date)
        result = []
        for hit, row in checked:
            cid, did, pid, start, end, _, _, _, pstart, pend = row
            rows = db.execute(
                "SELECT c.text, p.text, d.canonical_source FROM main.draft_children c "
                "JOIN main.draft_parents p ON p.parent_id = c.parent_id AND p.document_id = c.document_id "
                "JOIN main.draft_documents d ON d.document_id = c.document_id "
                "WHERE c.child_id = ? AND c.document_id = ? AND c.parent_id = ?", (cid, did, pid),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError
            text, parent, source = rows[0]
            if (any(type(value) is not str for value in (text, parent, source))
                or "\r" in source or end - start != len(text) or pend - pstart != len(parent)
                or end > len(source) or pend > len(source) or source[start:end] != text
                or source[pstart:pend] != parent or parent[start-pstart:end-pstart] != text
                or store.digest(source.encode("utf-8")) != hit.child.document.source_sha256
                or parse_policy_metadata(source, source_path=hit.child.document.source_path) != hit.child.document.metadata
                or not _authorized(context, hit.child)):
                raise ValueError
            result.append(AuthorizedPassage(handle.release_id, snapshot, role, cid, pid, did, start, end, text))
        return tuple(result)


def hydrate_published_hits(
    handle: store.PublishedSnapshot, context: AuthenticatedContext, hits: tuple[HybridHit, ...], *,
    release_root: Path, validity_date: date,
) -> tuple[AuthorizedPassage, ...]:
    """Hydrate up to 20 catalog-verified hits, in order, or return no passages.

    Only trusted server code calls this API. A hit's ranking is structurally
    checked, not a cryptographic receipt proving a previous search invocation.
    Authorization always derives from the verified context and pinned catalog.
    Public response serialization must never include these internal passages.
    """
    try:
        return _hydrate(handle, context, hits, release_root, validity_date)
    except Exception:
        # Leave the handler before raising so the original engine exception
        # and its sensitive frame locals are not attached as __context__.
        pass
    # Callers must not enable traceback-local dumps of their own arguments.
    raise EvidenceHydrationError("EVIDENCE_HYDRATION_FAILED")
