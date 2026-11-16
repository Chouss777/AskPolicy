"""Transactional SQLite storage for unpublished policy extractions."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import sqlite3

from ..ingestion.chunking import (
    ChildChunk,
    ChunkExtractionError,
    ParentSection,
    PolicyExtraction,
    extract_policy_chunks,
    validate_parent_child,
)
from ..ingestion.metadata import MetadataValidationError, PolicyMetadata


class DraftPersistenceError(ValueError):
    """Persistence failure with a stable, content-free error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class DraftWriteSummary:
    documents: int
    parents: int
    children: int


# Version, language, source path, classification and roles are inherited by
# joining on document_id. A child's heading path is inherited from its parent.
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS draft_documents (
        document_id TEXT NOT NULL PRIMARY KEY,
        policy_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (typeof(version) = 'integer' AND version > 0),
        language TEXT NOT NULL CHECK (language IN ('fr', 'en', 'ar')),
        title TEXT NOT NULL,
        department TEXT NOT NULL,
        classification TEXT NOT NULL CHECK (classification IN ('PUBLIC', 'INTERNAL', 'RESTRICTED')),
        allowed_roles TEXT NOT NULL CHECK (
            json_valid(allowed_roles) AND json_type(allowed_roles) = 'array'
            AND json_array_length(allowed_roles) BETWEEN 1 AND 4
        ),
        effective_date TEXT NOT NULL,
        expires_at TEXT,
        supersedes TEXT,
        approval_status TEXT NOT NULL CHECK (approval_status IN ('approved', 'not_approved')),
        source_type TEXT NOT NULL,
        source_license TEXT NOT NULL,
        source_path TEXT NOT NULL UNIQUE,
        source_sha256 TEXT NOT NULL CHECK (
            length(source_sha256) = 64 AND source_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        canonical_source TEXT NOT NULL,
        body_start INTEGER NOT NULL CHECK (
            typeof(body_start) = 'integer' AND body_start > 0 AND body_start < length(canonical_source)
        ),
        UNIQUE (policy_id, language, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS draft_parents (
        parent_id TEXT NOT NULL PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES draft_documents(document_id),
        ancestor_id TEXT CHECK (ancestor_id <> parent_id),
        heading TEXT,
        level INTEGER NOT NULL CHECK (level BETWEEN 0 AND 6),
        heading_path TEXT NOT NULL CHECK (json_valid(heading_path) AND json_type(heading_path) = 'array'),
        start INTEGER NOT NULL CHECK (typeof(start) = 'integer' AND start >= 0),
        end INTEGER NOT NULL CHECK (typeof(end) = 'integer' AND end > start),
        text TEXT NOT NULL CHECK (length(text) = end - start),
        UNIQUE (document_id, start),
        UNIQUE (document_id, parent_id),
        FOREIGN KEY (document_id, ancestor_id) REFERENCES draft_parents(document_id, parent_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS draft_children (
        child_id TEXT NOT NULL PRIMARY KEY,
        document_id TEXT NOT NULL,
        parent_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (
            kind IN ('heading', 'paragraph', 'procedure', 'list', 'table', 'code', 'thematic_break')
        ),
        start INTEGER NOT NULL CHECK (typeof(start) = 'integer' AND start >= 0),
        end INTEGER NOT NULL CHECK (typeof(end) = 'integer' AND end > start),
        text TEXT NOT NULL CHECK (length(text) = end - start),
        UNIQUE (document_id, start),
        FOREIGN KEY (document_id, parent_id) REFERENCES draft_parents(document_id, parent_id)
    )
    """,
)


def _require_idle(connection: sqlite3.Connection) -> None:
    if connection.in_transaction:
        raise DraftPersistenceError("TRANSACTION_ACTIVE")


@contextmanager
def _write_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Cursor]:
    _require_idle(connection)
    cursor = connection.cursor()
    # Internal queries must work with caller-owned row factories.
    cursor.row_factory = None
    try:
        cursor.execute("PRAGMA foreign_keys = ON")
        if cursor.execute("PRAGMA foreign_keys").fetchone() != (1,):
            raise DraftPersistenceError("FOREIGN_KEYS_DISABLED")
        cursor.execute("BEGIN IMMEDIATE")
        try:
            yield cursor
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    raise DraftPersistenceError("SQLITE_ROLLBACK_FAILED") from None
            raise
    finally:
        cursor.close()


def initialize_draft_schema(connection: sqlite3.Connection) -> None:
    """Create the draft tables atomically on an idle, caller-owned connection.

    Repeated initialization preserves existing rows. Foreign keys remain enabled
    on the connection. Initialization rejects an active caller transaction.
    """
    try:
        with _write_transaction(connection) as cursor:
            for statement in _SCHEMA:
                cursor.execute(statement)
    except sqlite3.Error:
        raise DraftPersistenceError("SCHEMA_INITIALIZATION_FAILED") from None


def _unique_id(identity: object, seen: set[str], code: str) -> None:
    if not isinstance(identity, str) or not identity:
        raise DraftPersistenceError("INVALID_EXTRACTION")
    if identity in seen:
        raise DraftPersistenceError(code)
    seen.add(identity)


def _validate_extractions(extractions: Sequence[PolicyExtraction]) -> list[PolicyExtraction]:
    if not isinstance(extractions, Sequence) or isinstance(extractions, (str, bytes, bytearray)):
        raise DraftPersistenceError("INVALID_INPUT")
    if not extractions:
        raise DraftPersistenceError("EMPTY_DRAFT")

    verified = []
    documents: set[str] = set()
    records: set[str] = set()
    for extraction in extractions:
        if (
            not isinstance(extraction, PolicyExtraction)
            or not isinstance(extraction.metadata, PolicyMetadata)
            or not isinstance(extraction.parents, tuple)
            or not isinstance(extraction.children, tuple)
            or not isinstance(extraction.canonical_source, str)
            or not isinstance(extraction.source_path, str)
        ):
            raise DraftPersistenceError("INVALID_INPUT")

        _unique_id(getattr(extraction.metadata, "document_id", None), documents, "DUPLICATE_DOCUMENT_ID")
        for parent in extraction.parents:
            if not isinstance(parent, ParentSection):
                raise DraftPersistenceError("INVALID_INPUT")
            _unique_id(parent.parent_id, records, "DUPLICATE_PARENT_ID")
        for child in extraction.children:
            if not isinstance(child, ChildChunk):
                raise DraftPersistenceError("INVALID_INPUT")
            _unique_id(child.child_id, records, "DUPLICATE_CHILD_ID")
        verified.append(_validate_source(extraction))
    return sorted(verified, key=lambda result: result.metadata.document_id)


def _validate_source(extraction: PolicyExtraction) -> PolicyExtraction:
    source = extraction.canonical_source
    if "\r" in source:
        raise DraftPersistenceError("INVALID_CANONICAL_SOURCE")
    try:
        digest = sha256(source.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        raise DraftPersistenceError("INVALID_CANONICAL_SOURCE") from None
    if extraction.source_sha256 != digest:
        raise DraftPersistenceError("INVALID_SOURCE_HASH")

    # Replay the production extractor to verify metadata, ancestry, spans,
    # completeness and deterministic IDs against the exact canonical source.
    try:
        expected = extract_policy_chunks(source, source_path=extraction.source_path)
    except (MetadataValidationError, ChunkExtractionError) as error:
        raise DraftPersistenceError(error.code) from None
    if type(getattr(extraction.metadata, "version", None)) is not int or extraction.metadata != expected.metadata:
        raise DraftPersistenceError("METADATA_MISMATCH")
    if expected.metadata.version > 2**63 - 1:
        raise DraftPersistenceError("INVALID_SQLITE_VERSION")
    if type(extraction.body_start) is not int or extraction.body_start != expected.body_start:
        raise DraftPersistenceError("INVALID_SOURCE_SPAN")

    parents: dict[str, ParentSection] = {}
    for record in (*extraction.parents, *extraction.children):
        if type(record.version) is not int:
            raise DraftPersistenceError("INVALID_EXTRACTION")
        if not (
            type(record.start) is int
            and type(record.end) is int
            and extraction.body_start <= record.start < record.end <= len(source)
            and source[record.start:record.end] == record.text
        ):
            raise DraftPersistenceError("INVALID_SOURCE_SPAN")
        if isinstance(record, ParentSection):
            if type(record.level) is not int:
                raise DraftPersistenceError("INVALID_PARENT_SECTION")
            parents[record.parent_id] = record
        else:
            parent = parents.get(record.parent_id) if isinstance(record.parent_id, str) else None
            if parent is None:
                raise DraftPersistenceError("INVALID_PARENT_CHILD")
            try:
                validate_parent_child(parent, record)
            except ChunkExtractionError as error:
                raise DraftPersistenceError(error.code) from None
    if extraction.parents != expected.parents:
        raise DraftPersistenceError("INVALID_PARENT_SECTION")
    if extraction.children != expected.children:
        raise DraftPersistenceError("INVALID_CHILD_CHUNK")
    return expected


def _json(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _insert_extraction(cursor: sqlite3.Cursor, extraction: PolicyExtraction) -> None:
    metadata = extraction.metadata
    cursor.execute(
        """
        INSERT INTO draft_documents (
            document_id, policy_id, version, language, title, department, classification, allowed_roles,
            effective_date, expires_at, supersedes, approval_status, source_type, source_license,
            source_path, source_sha256, canonical_source, body_start
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata.document_id, metadata.policy_id, metadata.version, metadata.language,
            metadata.title, metadata.department, metadata.classification, _json(metadata.allowed_roles),
            metadata.effective_date.isoformat(),
            metadata.expires_at.isoformat() if metadata.expires_at is not None else None,
            metadata.supersedes, metadata.approval_status, metadata.source_type, metadata.source_license,
            extraction.source_path, extraction.source_sha256, extraction.canonical_source, extraction.body_start,
        ),
    )
    cursor.executemany(
        """
        INSERT INTO draft_parents (parent_id, document_id, ancestor_id, heading, level, heading_path, start, end, text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (parent.parent_id, parent.document_id, parent.ancestor_id, parent.heading, parent.level,
             _json(parent.heading_path), parent.start, parent.end, parent.text)
            for parent in extraction.parents
        ),
    )
    cursor.executemany(
        """
        INSERT INTO draft_children (child_id, document_id, parent_id, kind, start, end, text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (child.child_id, child.document_id, child.parent_id, child.kind, child.start, child.end, child.text)
            for child in extraction.children
        ),
    )


def _counts(cursor: sqlite3.Cursor) -> DraftWriteSummary:
    row = cursor.execute(
        """SELECT (SELECT count(*) FROM draft_documents),
                  (SELECT count(*) FROM draft_parents),
                  (SELECT count(*) FROM draft_children)"""
    ).fetchone()
    return DraftWriteSummary(*row)


def persist_draft_extractions(
    connection: sqlite3.Connection, extractions: Sequence[PolicyExtraction]
) -> DraftWriteSummary:
    """Validate and commit a batch to an initialized, empty draft store.

    An active caller transaction or a nonempty draft is refused. The caller
    owns the connection. A summary is returned only after a successful commit.
    Supersession links are stored as metadata for later corpus validation.
    """
    try:
        _require_idle(connection)
        ordered = _validate_extractions(extractions)
        with _write_transaction(connection) as cursor:
            if _counts(cursor) != DraftWriteSummary(0, 0, 0):
                raise DraftPersistenceError("DRAFT_NOT_EMPTY")
            for extraction in ordered:
                _insert_extraction(cursor, extraction)
            summary = _counts(cursor)
            if summary != DraftWriteSummary(
                len(ordered), sum(len(item.parents) for item in ordered),
                sum(len(item.children) for item in ordered),
            ):
                raise DraftPersistenceError("DRAFT_COUNT_MISMATCH")
    except sqlite3.Error:
        raise DraftPersistenceError("SQLITE_WRITE_FAILED") from None
    return summary
