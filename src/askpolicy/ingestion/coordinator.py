"""Validate the full approved source corpus and import one unpublished draft."""

from dataclasses import dataclass
from hashlib import sha256
import json
from os import stat_result
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any

from .chunking import ChunkExtractionError, extract_policy_chunks
from .metadata import MetadataValidationError, PolicyMetadata, parse_policy_metadata
from ..storage.sqlite import (
    DraftPersistenceError,
    DraftWriteSummary,
    initialize_draft_schema,
    persist_draft_extractions,
)


class ManifestIngestionError(ValueError):
    """A content-free failure with a stable code, stage, and optional detail code."""

    def __init__(self, code: str, *, stage: str, detail_code: str | None = None) -> None:
        self.code = code
        self.stage = stage
        self.detail_code = detail_code
        super().__init__(code if detail_code is None else f"{code}: {detail_code}")


# docs/decisions.md §3: the full scope is independent of manifest contents.
_ACTIVE_FAMILIES = (
    ("ACCESS-001", 2, ("fr", "en")),
    ("DATA-001", 2, ("fr", "en", "ar")),
    ("EXPENSES-001", 2, ("fr", "en", "ar")),
    ("INCIDENT-001", 1, ("fr", "en")),
    ("LEAVE-001", 2, ("fr", "en", "ar")),
    ("PROCUREMENT-001", 1, ("fr", "en")),
    ("REMOTE-001", 2, ("fr", "en", "ar")),
    ("TRAVEL-001", 1, ("fr", "en")),
)
_HISTORICAL_FAMILIES = (
    ("ACCESS-001", "fr"),
    ("DATA-001", "fr"),
    ("EXPENSES-001", "fr"),
    ("LEAVE-001", "fr"),
    ("REMOTE-001", "en"),
    ("REMOTE-001", "fr"),
)
_APPROVED_SLOTS = frozenset(
    (family, language, version)
    for family, version, languages in _ACTIVE_FAMILIES
    for language in languages
) | frozenset((family, language, 1) for family, language in _HISTORICAL_FAMILIES)


def _document_id(family: str, language: str, version: int) -> str:
    return f"{family}-{language.upper()}-V{version}"


_APPROVED_LINKS = {
    _document_id(family, language, 2): _document_id(family, language, 1)
    for family, language in _HISTORICAL_FAMILIES
}
_METADATA_FIELDS = frozenset(PolicyMetadata.model_fields)
_RECORD_FIELDS = _METADATA_FIELDS | {"source_path", "content_sha256"}
_OPTIONAL_TEXT = {"supersedes", "expires_at"}
_REQUIRED_TEXT = _RECORD_FIELDS - _OPTIONAL_TEXT - {"version", "allowed_roles"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class _ManifestDocument:
    source_path: str
    content_sha256: str
    metadata: dict[str, Any]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestIngestionError("DUPLICATE_MANIFEST_FIELD", stage="manifest")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ManifestIngestionError("INVALID_MANIFEST", stage="manifest")


def _read_manifest(raw: bytes) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
    except ManifestIngestionError:
        raise
    except (ValueError, RecursionError):
        raise ManifestIngestionError("INVALID_MANIFEST", stage="manifest") from None

    if not isinstance(manifest, dict) or manifest.keys() != {"documents"}:
        raise ManifestIngestionError("INVALID_MANIFEST", stage="manifest")
    records = manifest["documents"]
    if not isinstance(records, list) or not records:
        raise ManifestIngestionError("INVALID_MANIFEST", stage="manifest")
    for record in records:
        if not isinstance(record, dict) or record.keys() != _RECORD_FIELDS:
            raise ManifestIngestionError("INVALID_MANIFEST", stage="manifest")
        roles = record["allowed_roles"]
        if (
            any(not isinstance(record[key], str) for key in _REQUIRED_TEXT)
            or any(record[key] is not None and not isinstance(record[key], str) for key in _OPTIONAL_TEXT)
            or type(record["version"]) is not int
            or not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) for role in roles)
            or not _SHA256.fullmatch(record["content_sha256"])
        ):
            raise ManifestIngestionError("INVALID_MANIFEST", stage="manifest")
    return records


def _check_inventory(records: list[dict[str, Any]]) -> dict[str, _ManifestDocument]:
    documents: dict[str, _ManifestDocument] = {}
    paths: set[str] = set()
    slots: set[tuple[str, str, int]] = set()
    for record in records:
        family, language, version = record["policy_id"], record["language"], record["version"]
        slot = (family, language, version)
        document_id = record["document_id"]
        path = record["source_path"]
        if document_id in documents:
            raise ManifestIngestionError("DUPLICATE_DOCUMENT_ID", stage="inventory")
        if path in paths:
            raise ManifestIngestionError("DUPLICATE_SOURCE_PATH", stage="inventory")
        if slot not in _APPROVED_SLOTS or slot in slots or document_id != _document_id(*slot):
            raise ManifestIngestionError("INVALID_INVENTORY", stage="inventory")
        if path != f"data/policies/{language}/{document_id}.md":
            raise ManifestIngestionError("INVALID_SOURCE_PATH", stage="inventory")

        slots.add(slot)
        paths.add(path)
        documents[document_id] = _ManifestDocument(
            path, record["content_sha256"], {key: record[key] for key in _METADATA_FIELDS},
        )

    if slots != _APPROVED_SLOTS:
        raise ManifestIngestionError("INVALID_INVENTORY", stage="inventory")
    return dict(sorted(documents.items()))


def _file_stat(path: Path, *, stage: str, missing: str, unreadable: str) -> stat_result:
    try:
        info = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        raise ManifestIngestionError(missing, stage=stage) from None
    except OSError:
        raise ManifestIngestionError(unreadable, stage=stage) from None
    # Windows junctions and other reparse points can redirect directory traversal.
    if stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ManifestIngestionError("INVALID_SOURCE_PATH", stage=stage)
    return info


def _read_file(root: Path, parts: tuple[str, ...], *, stage: str) -> bytes:
    missing = "MISSING_MANIFEST" if stage == "manifest" else "MISSING_SOURCE"
    unreadable = "MANIFEST_READ_FAILED" if stage == "manifest" else "SOURCE_READ_FAILED"
    path = root
    for index, part in enumerate(parts):
        path /= part
        info = _file_stat(path, stage=stage, missing=missing, unreadable=unreadable)
        valid_kind = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
        if not valid_kind(info.st_mode):
            raise ManifestIngestionError("INVALID_SOURCE_PATH", stage=stage)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise ManifestIngestionError(missing, stage=stage) from None
    except OSError:
        raise ManifestIngestionError(unreadable, stage=stage) from None


def _check_source_tree(root: Path, expected_paths: set[str]) -> None:
    data = root / "data"
    info = _file_stat(data, stage="source", missing="MISSING_SOURCE", unreadable="SOURCE_READ_FAILED")
    if not stat.S_ISDIR(info.st_mode):
        raise ManifestIngestionError("INVALID_SOURCE_PATH", stage="source")
    base = data / "policies"
    pending = [base]
    found: set[str] = set()
    while pending:
        path = pending.pop()
        info = _file_stat(path, stage="source", missing="MISSING_SOURCE", unreadable="SOURCE_READ_FAILED")
        if stat.S_ISDIR(info.st_mode):
            try:
                pending.extend(sorted(path.iterdir(), reverse=True))
            except OSError:
                raise ManifestIngestionError("SOURCE_READ_FAILED", stage="source") from None
        else:
            if path == base:
                raise ManifestIngestionError("INVALID_SOURCE_PATH", stage="source")
            relative = path.relative_to(root).as_posix()
            if not stat.S_ISREG(info.st_mode) or relative not in expected_paths:
                raise ManifestIngestionError("UNEXPECTED_SOURCE", stage="source")
            found.add(relative)
    if found != expected_paths:
        raise ManifestIngestionError("MISSING_SOURCE", stage="source")


def _verified_source(root: Path, document: _ManifestDocument) -> str:
    raw = _read_file(root, tuple(document.source_path.split("/")), stage="source")
    try:
        source = raw.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        raise ManifestIngestionError("INVALID_SOURCE_ENCODING", stage="source") from None
    if "\r" in source:
        raise ManifestIngestionError("INVALID_SOURCE_NEWLINES", stage="source")
    if sha256(source.encode("utf-8")).hexdigest() != document.content_sha256:
        raise ManifestIngestionError("SOURCE_HASH_MISMATCH", stage="source")
    return source


def _graph_error(detail_code: str) -> ManifestIngestionError:
    return ManifestIngestionError("INVALID_SUPERSESSION_GRAPH", stage="supersession", detail_code=detail_code)


def _validate_supersession(documents: dict[str, _ManifestDocument]) -> None:
    links = {
        document_id: doc.metadata["supersedes"]
        for document_id, doc in documents.items()
        if doc.metadata["supersedes"] is not None
    }
    for successor, predecessor in links.items():
        if predecessor not in documents:
            raise _graph_error("MISSING_PREDECESSOR")
        if successor == predecessor:
            raise _graph_error("SELF_SUPERSESSION")
    if len(set(links.values())) != len(links):
        raise _graph_error("BRANCHING_SUCCESSORS")

    for successor, predecessor in links.items():
        seen = {successor}
        cursor = predecessor
        while cursor in links:
            if cursor in seen:
                raise _graph_error("CYCLE")
            seen.add(cursor)
            cursor = links[cursor]

        newer, older = documents[successor].metadata, documents[predecessor].metadata
        if newer["policy_id"] != older["policy_id"]:
            raise _graph_error("WRONG_PREDECESSOR_FAMILY")
        if newer["language"] != older["language"]:
            raise _graph_error("WRONG_PREDECESSOR_LANGUAGE")
        if older["version"] >= newer["version"]:
            raise _graph_error("PREDECESSOR_NOT_EARLIER")
    if links != _APPROVED_LINKS:
        raise _graph_error("UNAPPROVED_LINKS")


def ingest_manifest_to_draft(repository_root: Path, connection: sqlite3.Connection) -> DraftWriteSummary:
    """Commit the approved manifest in document-ID order to an empty draft store.

    The caller owns the connection and provides the trusted repository root.
    Manifest, source, and graph checks report their own codes. Source metadata
    failures retain the validator's code with stage 'metadata'. Extraction and
    storage failures use EXTRACTION_FAILED and DRAFT_PERSISTENCE_FAILED, with
    the underlying code available as detail_code.
    """
    raw_manifest = _read_file(repository_root, ("data", "corpus_manifest.yaml"), stage="manifest")
    documents = _check_inventory(_read_manifest(raw_manifest))
    _check_source_tree(repository_root, {document.source_path for document in documents.values()})
    sources = {key: _verified_source(repository_root, doc) for key, doc in documents.items()}

    link_mismatch = False
    for key, doc in documents.items():
        try:
            metadata = parse_policy_metadata(sources[key], source_path=doc.source_path)
        except MetadataValidationError as error:
            raise ManifestIngestionError(error.code, stage="metadata") from None
        actual = metadata.model_dump(mode="json")
        if any(actual[field] != doc.metadata[field] for field in _METADATA_FIELDS - {"supersedes"}):
            raise ManifestIngestionError("METADATA_MISMATCH", stage="metadata")
        link_mismatch |= actual["supersedes"] != doc.metadata["supersedes"]

    # Report an invalid declared graph before reporting disagreement on a link.
    _validate_supersession(documents)
    if link_mismatch:
        raise ManifestIngestionError("METADATA_MISMATCH", stage="metadata")

    try:
        batch = tuple(
            extract_policy_chunks(sources[key], source_path=document.source_path)
            for key, document in documents.items()
        )
    except (ChunkExtractionError, MetadataValidationError) as error:
        raise ManifestIngestionError("EXTRACTION_FAILED", stage="extraction", detail_code=error.code) from None

    try:
        initialize_draft_schema(connection)
        return persist_draft_extractions(connection, batch)
    except DraftPersistenceError as error:
        raise ManifestIngestionError(
            "DRAFT_PERSISTENCE_FAILED", stage="storage", detail_code=error.code,
        ) from None
