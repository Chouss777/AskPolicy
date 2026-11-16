"""Calculate content-addressed identities for unpublished SQLite drafts."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import sqlite3
from typing import Any, Literal

from ..ingestion.chunking import ChunkExtractionError, _CHUNKING_FORMAT, extract_policy_chunks
from ..ingestion.coordinator import (
    ManifestIngestionError,
    _ManifestDocument,
    _check_inventory,
    _read_manifest,
    _validate_supersession,
)
from ..ingestion.metadata import MetadataValidationError, PolicyMetadata


class SnapshotIdentityError(ValueError):
    """A content-free failure identified by a stable code and stage."""

    def __init__(self, code: str, *, stage: str, detail_code: str | None = None) -> None:
        self.code = code
        self.stage = stage
        self.detail_code = detail_code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class DraftSnapshotConfig:
    """Versioned fingerprint settings. Identity encoding selects the final framing."""

    snapshot_format_version: int = 1
    source_canonicalization: str = "utf8-lf-v1"
    chunking_format: str = _CHUNKING_FORMAT
    component_encoding: str = "canonical-json-v1"
    identity_encoding: Literal["json-object-v1", "json-array-v1"] = "json-object-v1"


@dataclass(frozen=True, slots=True)
class DraftSnapshotIdentity:
    candidate_snapshot_id: str
    content_hash: str
    config_hash: str
    source_manifest_hash: str
    document_count: int
    parent_count: int
    child_count: int
    status: Literal["UNPUBLISHED"] = field(init=False, default="UNPUBLISHED")


_METADATA_FIELDS = tuple(PolicyMetadata.model_fields)
_TABLES = (
    ("draft_documents", {*_METADATA_FIELDS, "source_path", "source_sha256", "canonical_source", "body_start"},
     "document_id COLLATE BINARY"),
    ("draft_parents", {"parent_id", "document_id", "ancestor_id", "heading", "level", "heading_path", "start", "end", "text"},
     "document_id COLLATE BINARY, start"),
    ("draft_children", {"child_id", "document_id", "parent_id", "kind", "start", "end", "text"},
     "document_id COLLATE BINARY, start"),
)


def _validate_config(config: DraftSnapshotConfig) -> None:
    if type(config) is not DraftSnapshotConfig or type(config.snapshot_format_version) is not int:
        raise SnapshotIdentityError("INVALID_SNAPSHOT_CONFIG", stage="configuration")
    if config.snapshot_format_version != 1:
        raise SnapshotIdentityError("UNSUPPORTED_SNAPSHOT_FORMAT", stage="configuration")
    if (
        config.source_canonicalization != "utf8-lf-v1"
        or config.chunking_format != _CHUNKING_FORMAT
        or config.component_encoding != "canonical-json-v1"
        or config.identity_encoding not in ("json-object-v1", "json-array-v1")
    ):
        raise SnapshotIdentityError("INVALID_SNAPSHOT_CONFIG", stage="configuration")


def _manifest_documents(raw: bytes) -> dict[str, _ManifestDocument]:
    if not isinstance(raw, bytes):
        raise SnapshotIdentityError("INVALID_MANIFEST", stage="manifest")
    try:
        documents = _check_inventory(_read_manifest(raw))
        _validate_supersession(documents)
    except ManifestIngestionError as error:
        code = error.code if error.stage == "manifest" else "MANIFEST_MISMATCH"
        raise SnapshotIdentityError(
            code, stage="manifest", detail_code=error.detail_code or error.code,
        ) from None
    return documents


def _read_draft(connection: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    try:
        if connection.in_transaction:
            raise SnapshotIdentityError("TRANSACTION_ACTIVE", stage="database")
        cursor = connection.cursor()
        cursor.row_factory = None
        try:
            if cursor.execute("PRAGMA read_uncommitted").fetchone() != (0,):
                raise SnapshotIdentityError("UNSAFE_READ_ISOLATION", stage="database")
            cursor.execute("BEGIN")
            try:
                # The first read pins the view for all three tables. Qualify
                # main so temporary tables cannot shadow committed records.
                tables = {
                    name for name, in cursor.execute(
                        "SELECT name FROM main.sqlite_schema WHERE type = 'table'"
                    )
                }
                draft = {}
                for table, expected_columns, order in _TABLES:
                    if table not in tables:
                        raise SnapshotIdentityError("INVALID_DRAFT_SCHEMA", stage="database")
                    # All table names and SQL fragments are constants in _TABLES.
                    columns = {
                        row[1] for row in cursor.execute(f"PRAGMA main.table_xinfo({table})")
                    }
                    if columns != expected_columns:
                        raise SnapshotIdentityError("INVALID_DRAFT_SCHEMA", stage="database")
                    cursor.execute(f"SELECT * FROM main.{table} ORDER BY {order}")
                    names = [column[0] for column in cursor.description]
                    draft[table] = [dict(zip(names, row, strict=True)) for row in cursor]
                return draft
            finally:
                connection.rollback()
        finally:
            cursor.close()
    except sqlite3.Error:
        raise SnapshotIdentityError("DRAFT_READ_FAILED", stage="database") from None


def _string_array(raw: object) -> list[str]:
    if isinstance(raw, str):
        try:
            values = json.loads(raw)
        except (ValueError, RecursionError):
            pass
        else:
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                return values
    raise SnapshotIdentityError("DRAFT_INTEGRITY_FAILED", stage="records")


def _validate_records(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    if len(actual) < len(expected):
        raise SnapshotIdentityError("INCOMPLETE_DRAFT", stage="records")
    if actual != expected:
        raise SnapshotIdentityError("DRAFT_INTEGRITY_FAILED", stage="records")
    # Numeric equality alone would treat SQLite REAL 1.0 as INTEGER 1.
    for stored, replayed in zip(actual, expected, strict=True):
        if any(type(stored[key]) is not type(value) for key, value in replayed.items()):
            raise SnapshotIdentityError("DRAFT_INTEGRITY_FAILED", stage="records")


def _validate_draft(
    draft: dict[str, list[dict[str, Any]]], manifest: dict[str, _ManifestDocument],
) -> None:
    documents = draft["draft_documents"]
    ids = {row["document_id"] for row in documents}
    if len(ids) != len(documents) or ids - manifest.keys():
        raise SnapshotIdentityError("DRAFT_INTEGRITY_FAILED", stage="documents")
    if ids != manifest.keys():
        raise SnapshotIdentityError("INCOMPLETE_DRAFT", stage="documents")

    parents = []
    children = []
    for row in documents:
        approved = manifest[row["document_id"]]
        if row["source_path"] != approved.source_path:
            raise SnapshotIdentityError("MANIFEST_MISMATCH", stage="documents")
        source = row["canonical_source"]
        if not isinstance(source, str) or "\r" in source:
            raise SnapshotIdentityError("DRAFT_INTEGRITY_FAILED", stage="documents")
        try:
            extraction = extract_policy_chunks(source, source_path=approved.source_path)
        except (MetadataValidationError, ChunkExtractionError) as error:
            raise SnapshotIdentityError("DRAFT_INTEGRITY_FAILED", stage="documents", detail_code=error.code) from None
        if (
            extraction.source_sha256 != row["source_sha256"]
            or type(row["body_start"]) is not int
            or extraction.body_start != row["body_start"]
        ):
            raise SnapshotIdentityError("DRAFT_INTEGRITY_FAILED", stage="documents")

        row["allowed_roles"] = _string_array(row["allowed_roles"])
        stored = {name: row[name] for name in _METADATA_FIELDS}
        if (
            extraction.source_sha256 != approved.content_sha256
            or extraction.metadata.model_dump(mode="json") != approved.metadata
            or type(stored["version"]) is not int
            or stored != approved.metadata
        ):
            raise SnapshotIdentityError("MANIFEST_MISMATCH", stage="documents")

        parents.extend(
            {
                "parent_id": parent.parent_id, "document_id": parent.document_id,
                "ancestor_id": parent.ancestor_id, "heading": parent.heading, "level": parent.level,
                "heading_path": list(parent.heading_path), "start": parent.start,
                "end": parent.end, "text": parent.text,
            }
            for parent in extraction.parents
        )
        children.extend(
            {
                "child_id": child.child_id, "document_id": child.document_id,
                "parent_id": child.parent_id, "kind": child.kind, "start": child.start,
                "end": child.end, "text": child.text,
            }
            for child in extraction.children
        )

    for parent in draft["draft_parents"]:
        parent["heading_path"] = _string_array(parent["heading_path"])
    # Exact replay comparisons check identities, ancestry, relationship scope,
    # source slices and completeness without trusting the database constraints.
    _validate_records(draft["draft_parents"], parents)
    _validate_records(draft["draft_children"], children)


def _digest(domain: str, payload: object) -> str:
    try:
        encoded = json.dumps(
            ["askpolicy/draft-snapshot/" + domain, payload],
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()
    except (TypeError, ValueError):
        raise SnapshotIdentityError("SNAPSHOT_HASH_FAILED", stage="hashing") from None


def compute_draft_snapshot_identity(
    connection: sqlite3.Connection, *, manifest: bytes, config: DraftSnapshotConfig,
) -> DraftSnapshotIdentity:
    """Fingerprint a complete committed draft against trusted manifest bytes.

    The caller supplies the approved manifest and owns the connection. Require
    an idle connection with committed-read isolation. Copy all rows within a
    single SQLite read transaction, then validate and hash the copied view.

    Components hash domain-tagged UTF-8 JSON with sorted object keys, compact
    separators, preserved Unicode and canonical arrays. Documents are ordered
    by ID and sections/chunks by document and source offset. The candidate ID
    frames the format version and the three digests in an object or a
    versioned positional array, as selected by identity_encoding.
    """
    _validate_config(config)
    approved = _manifest_documents(manifest)
    draft = _read_draft(connection)
    _validate_draft(draft, approved)
    counts = {
        "document_count": len(draft["draft_documents"]),
        "parent_count": len(draft["draft_parents"]),
        "child_count": len(draft["draft_children"]),
    }
    content_hash = _digest("content/v1", {"tables": draft, "counts": counts})
    config_hash = _digest("config/v1", asdict(config))
    source_manifest_hash = _digest("manifest/v1", {"documents": [
        {"source_path": doc.source_path, "content_sha256": doc.content_sha256, **doc.metadata}
        for doc in approved.values()
    ]})
    version = config.snapshot_format_version
    if config.identity_encoding == "json-object-v1":
        payload: object = {
            "version": version, "content_hash": content_hash, "config_hash": config_hash,
            "source_manifest_hash": source_manifest_hash,
        }
    else:
        payload = [version, content_hash, config_hash, source_manifest_hash]
    candidate_snapshot_id = _digest(f"identity/{config.identity_encoding}", payload)
    return DraftSnapshotIdentity(
        candidate_snapshot_id, content_hash, config_hash, source_manifest_hash, **counts,
    )
