"""Build and search role-scoped BM25S indexes for an unpublished SQLite draft."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
import sqlite3
from types import MappingProxyType
from typing import Literal, get_args

from ..auth.context import AuthenticatedContext
from ..auth.jwt import ReadRole
from ..ingestion.metadata import Classification, PolicyMetadata
from ..security.acl import can_read_document
from ..security.index_selection import IndexSelectionError, select_role_index
from ..storage.eligibility import PolicyEligibilityError, PolicyEligibilityResult, select_policy_versions
from ..storage.snapshot import (
    DraftSnapshotConfig,
    DraftSnapshotIdentity,
    SnapshotIdentityError,
    compute_draft_snapshot_identity,
)
from .bm25_core import (
    BM25_OPTIONS,
    TOKENIZATION,
    BM25Options,
    DraftDocument,
    EvidenceChunk,
    SectionIdentity,
    TokenizationOptions,
    build_text_index,
    ranked_positions,
    read_draft_chunks,
    read_draft_documents,
)


TOP_K = 30
_READ_ROLES: tuple[ReadRole, ...] = get_args(ReadRole)
_CLASSIFICATIONS = frozenset(get_args(Classification))


class RoleBM25Error(ValueError):
    """Controlled failure without evidence or credentials in its message."""

    def __init__(self, code: str, *, detail_code: str | None = None) -> None:
        self.code = code
        self.detail_code = detail_code
        super().__init__(code)


def _fingerprint(domain: str, payload: object) -> str:
    encoded = json.dumps(
        ["askpolicy/role-bm25/" + domain, payload],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RoleBM25Configuration:
    bm25s_version: str
    numpy_version: str
    bm25s: BM25Options = BM25_OPTIONS
    tokenization: TokenizationOptions = TOKENIZATION
    top_k: int = TOP_K
    indexed_text: Literal["exact_stored_child_text"] = "exact_stored_child_text"
    index_order: tuple[str, ...] = ("document_id", "child_start", "child_id")
    ranking_order: tuple[str, ...] = ("score_descending", "child_id_ascending")
    finite_positive_scores_only: bool = True
    empty_token_indexed: bool = False

    @property
    def fingerprint(self) -> str:
        return _fingerprint("configuration/v1", asdict(self))


@dataclass(frozen=True, slots=True)
class RoleBM25Metadata:
    candidate_snapshot_id: str
    source_manifest_hash: str
    snapshot_config_hash: str
    validity_date: date
    next_transition_date: date | None
    role: ReadRole
    configuration: RoleBM25Configuration
    indexed_document_ids: tuple[str, ...]
    indexed_child_ids: tuple[str, ...]
    snapshot_status: Literal["UNPUBLISHED"] = field(default="UNPUBLISHED", init=False)

    @property
    def indexed_document_count(self) -> int:
        return len(self.indexed_document_ids)

    @property
    def indexed_child_count(self) -> int:
        return len(self.indexed_child_ids)

    @property
    def configuration_fingerprint(self) -> str:
        return self.configuration.fingerprint

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["validity_date"] = self.validity_date.isoformat()
        payload["next_transition_date"] = self.next_transition_date.isoformat() if self.next_transition_date else None
        return _fingerprint("index/v1", payload)


@dataclass(frozen=True, slots=True)
class IndexedChild:
    child_id: str
    section: SectionIdentity
    document: DraftDocument
    start: int


@dataclass(frozen=True, slots=True)
class RoleBM25Hit:
    child: IndexedChild
    candidate_snapshot_id: str
    score: float


@dataclass(frozen=True, slots=True)
class _BoundScorer:
    metadata_fingerprint: str
    search: Callable[[str], tuple[RoleBM25Hit, ...]] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RoleBM25Index:
    metadata: RoleBM25Metadata
    _scorer: _BoundScorer = field(repr=False, compare=False)
    _engine: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _DraftBinding:
    snapshot: DraftSnapshotIdentity
    eligibility: PolicyEligibilityResult
    documents: Mapping[str, DraftDocument]
    children: Mapping[str, IndexedChild]


@dataclass(frozen=True, slots=True)
class RoleBM25Bundle:
    indexes: tuple[RoleBM25Index, ...]
    _draft: _DraftBinding = field(repr=False)

    @property
    def snapshot(self) -> DraftSnapshotIdentity:
        return self._draft.snapshot

    @property
    def eligibility(self) -> PolicyEligibilityResult:
        return self._draft.eligibility


def _reject_busy_copy(status: int, _remaining: int, _pages: int) -> None:
    if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        raise RoleBM25Error("DRAFT_COPY_FAILED")


def _read_catalog(
    connection: sqlite3.Connection, documents: Mapping[str, DraftDocument], eligibility: PolicyEligibilityResult,
) -> dict[str, IndexedChild]:
    # Keep an independent identity catalog without passage text. Role-specific
    # extraction uses its own WHERE clause before loading any child text.
    current = set(eligibility.eligible_document_ids)
    rows = connection.execute(
        "SELECT c.child_id, c.document_id, c.parent_id, p.heading, c.start "
        "FROM main.draft_children AS c JOIN main.draft_parents AS p "
        "ON p.document_id = c.document_id AND p.parent_id = c.parent_id "
        "ORDER BY c.document_id, c.start, c.child_id",
    )
    catalog = {}
    for child_id, document_id, parent_id, heading, start in rows:
        if document_id in current:
            if child_id in catalog:
                raise RoleBM25Error("INVALID_CHILD_MEMBERSHIP")
            catalog[child_id] = IndexedChild(
                child_id, SectionIdentity(document_id, parent_id, heading), documents[document_id], start,
            )
    return catalog


def _authorized_documents(draft: _DraftBinding, role: ReadRole) -> dict[str, DraftDocument]:
    return {
        document_id: draft.documents[document_id]
        for document_id in draft.eligibility.eligible_document_ids
        if draft.documents[document_id].metadata.classification in _CLASSIFICATIONS
        and role in draft.documents[document_id].metadata.allowed_roles
    }


def _index_metadata(
    draft: _DraftBinding, role: ReadRole, configuration: RoleBM25Configuration,
) -> RoleBM25Metadata:
    allowed = _authorized_documents(draft, role)
    return RoleBM25Metadata(
        candidate_snapshot_id=draft.snapshot.candidate_snapshot_id,
        source_manifest_hash=draft.snapshot.source_manifest_hash,
        snapshot_config_hash=draft.snapshot.config_hash,
        validity_date=draft.eligibility.validity_date,
        next_transition_date=draft.eligibility.next_transition_date,
        role=role,
        configuration=configuration,
        indexed_document_ids=tuple(sorted(allowed)),
        indexed_child_ids=tuple(sorted(
            child.child_id for child in draft.children.values() if child.section.document_id in allowed
        )),
    )


def _build_index(
    connection: sqlite3.Connection, draft: _DraftBinding, role: ReadRole, configuration: RoleBM25Configuration,
) -> RoleBM25Index:
    metadata = _index_metadata(draft, role, configuration)
    chunks: tuple[EvidenceChunk, ...] = read_draft_chunks(connection, _authorized_documents(draft, role))
    expected = tuple(child for child in draft.children.values() if child.section.document_id in metadata.indexed_document_ids)
    children = tuple(IndexedChild(chunk.child_id, chunk.section, chunk.document, chunk.start) for chunk in chunks)
    if children != expected:
        raise RoleBM25Error("INVALID_CHILD_MEMBERSHIP")

    model = build_text_index([chunk.text for chunk in chunks])
    return bind_text_index(metadata, children, model)


def bind_text_index(metadata: RoleBM25Metadata, children: tuple[IndexedChild, ...], model: object) -> RoleBM25Index:
    """Bind a validated loaded engine to the original ordered child catalog."""
    child_ids = tuple(child.child_id for child in children)
    snapshot_id = metadata.candidate_snapshot_id

    def search(query: str) -> tuple[RoleBM25Hit, ...]:
        return tuple(
            RoleBM25Hit(children[position], snapshot_id, score)
            for position, score in ranked_positions(model, child_ids, query, limit=metadata.configuration.top_k)
        )

    return RoleBM25Index(metadata, _BoundScorer(metadata.fingerprint, search), model)


@contextmanager
def validated_role_draft(
    connection: sqlite3.Connection, *, manifest: bytes, snapshot_config: DraftSnapshotConfig,
    validity_date: date, expected_snapshot_id: str | None = None,
) -> Iterator[tuple[sqlite3.Connection, _DraftBinding]]:
    """Bind role membership and extraction to one private SQLite image.

    The caller supplies a committed draft, approved manifest, and trusted date.
    A SQLite backup captures the image atomically even with a separate writer.
    Fingerprinting and every extraction read use that private in-memory copy.
    The source connection must be idle and stays open for the caller.
    """
    if type(validity_date) is not date:
        raise RoleBM25Error("INVALID_VALIDITY_DATE")
    with closing(sqlite3.connect(":memory:")) as image:
        try:
            if connection.in_transaction:
                raise RoleBM25Error("TRANSACTION_ACTIVE")
            try:
                connection.backup(image, name="main", progress=_reject_busy_copy)
            except sqlite3.Error:
                raise RoleBM25Error("DRAFT_COPY_FAILED") from None
            snapshot = compute_draft_snapshot_identity(image, manifest=manifest, config=snapshot_config)
            if expected_snapshot_id is not None and (
                type(expected_snapshot_id) is not str or expected_snapshot_id != snapshot.candidate_snapshot_id
            ):
                raise RoleBM25Error("SNAPSHOT_MISMATCH")
            documents = read_draft_documents(image)
            eligibility = select_policy_versions(
                [doc.metadata for doc in documents.values()], validity_date=validity_date,
            )
            catalog = _read_catalog(image, documents, eligibility)
            draft = _DraftBinding(snapshot, eligibility, MappingProxyType(documents), MappingProxyType(catalog))
        except SnapshotIdentityError as error:
            raise RoleBM25Error("INVALID_DRAFT", detail_code=error.code) from None
        except PolicyEligibilityError as error:
            raise RoleBM25Error("INVALID_ELIGIBILITY", detail_code=error.code) from None
        except RoleBM25Error:
            raise
        except (sqlite3.Error, ValueError, RuntimeError, OverflowError):
            raise RoleBM25Error("INDEX_BUILD_FAILED") from None
        yield image, draft


def build_role_bm25_bundle(
    connection: sqlite3.Connection, *, manifest: bytes, snapshot_config: DraftSnapshotConfig,
    validity_date: date, expected_snapshot_id: str | None = None,
) -> RoleBM25Bundle:
    """Build every lexical index from the same validated private draft."""
    try:
        with validated_role_draft(
            connection, manifest=manifest, snapshot_config=snapshot_config,
            validity_date=validity_date, expected_snapshot_id=expected_snapshot_id,
        ) as (image, draft):
            configuration = RoleBM25Configuration(version("bm25s"), version("numpy"))
            indexes = tuple(_build_index(image, draft, role, configuration) for role in _READ_ROLES)
            return RoleBM25Bundle(indexes, draft)
    except RoleBM25Error:
        raise
    except (sqlite3.Error, ValueError, RuntimeError, OverflowError):
        raise RoleBM25Error("INDEX_BUILD_FAILED") from None


def _verified_index(bundle: RoleBM25Bundle, role: ReadRole) -> RoleBM25Index:
    if type(bundle.indexes) is not tuple or any(
        type(index) is not RoleBM25Index or type(index.metadata) is not RoleBM25Metadata for index in bundle.indexes
    ):
        raise RoleBM25Error("INVALID_ROLE_SET")
    roles = tuple(index.metadata.role for index in bundle.indexes)
    if len(roles) != len(_READ_ROLES) or any(type(role) is not str for role in roles) or set(roles) != set(_READ_ROLES):
        raise RoleBM25Error("INVALID_ROLE_SET")
    configuration = RoleBM25Configuration(version("bm25s"), version("numpy"))
    for index in bundle.indexes:
        expected = _index_metadata(bundle._draft, index.metadata.role, configuration)
        if (
            index.metadata != expected
            or type(index._scorer) is not _BoundScorer
            or index._scorer.metadata_fingerprint != expected.fingerprint
        ):
            raise RoleBM25Error("INDEX_METADATA_MISMATCH")
    return next(index for index in bundle.indexes if index.metadata.role == role)


def search_role_bm25(
    bundle: RoleBM25Bundle, context: AuthenticatedContext, *, query: str,
    expected_snapshot_id: str, validity_date: date,
) -> tuple[RoleBM25Hit, ...]:
    """Search the single verified role's unpublished index at the exact bound date."""
    try:
        scope = select_role_index(context)
    except IndexSelectionError:
        raise RoleBM25Error("UNVERIFIED_CONTEXT") from None
    if type(bundle) is not RoleBM25Bundle or type(bundle._draft) is not _DraftBinding:
        raise RoleBM25Error("INVALID_BUNDLE")
    if type(expected_snapshot_id) is not str or expected_snapshot_id != bundle.snapshot.candidate_snapshot_id:
        raise RoleBM25Error("SNAPSHOT_MISMATCH")
    if type(validity_date) is not date:
        raise RoleBM25Error("INVALID_VALIDITY_DATE")
    eligibility = bundle.eligibility
    if eligibility.next_transition_date is not None and validity_date >= eligibility.next_transition_date:
        raise RoleBM25Error("STALE_INDEX")
    if validity_date != eligibility.validity_date:
        raise RoleBM25Error("VALIDITY_DATE_MISMATCH")
    if type(query) is not str:
        raise RoleBM25Error("INVALID_QUERY")

    index = _verified_index(bundle, scope.role)
    try:
        hits = index._scorer.search(query)
    except (ValueError, RuntimeError, OverflowError):
        raise RoleBM25Error("INDEX_SEARCH_FAILED") from None
    allowed = set(index.metadata.indexed_child_ids)
    seen = set()
    if type(hits) is not tuple or len(hits) > TOP_K:
        raise RoleBM25Error("HIT_INTEGRITY_FAILED")
    for hit in hits:
        if type(hit) is not RoleBM25Hit or type(hit.child) is not IndexedChild:
            raise RoleBM25Error("HIT_INTEGRITY_FAILED")
        child = hit.child
        if (
            type(child.child_id) is not str
            or type(child.section) is not SectionIdentity
            or type(child.document) is not DraftDocument
            or type(child.document.metadata) is not PolicyMetadata
        ):
            raise RoleBM25Error("HIT_INTEGRITY_FAILED")
        metadata = child.document.metadata
        if not can_read_document(context, classification=metadata.classification, allowed_roles=metadata.allowed_roles):
            raise RoleBM25Error("HIT_ACCESS_DENIED")
        if (
            hit.candidate_snapshot_id != expected_snapshot_id
            or child.child_id not in allowed
            or child.child_id in seen
            or child != bundle._draft.children.get(child.child_id)
            or type(hit.score) is not float
            or not isfinite(hit.score)
            or hit.score <= 0
        ):
            raise RoleBM25Error("HIT_INTEGRITY_FAILED")
        seen.add(child.child_id)
    return hits
