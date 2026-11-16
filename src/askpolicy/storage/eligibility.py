"""Select current-policy versions for an unpublished corpus."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import ValidationError

from ..ingestion.metadata import PolicyMetadata


ExclusionReason = Literal["NOT_APPROVED", "NOT_YET_EFFECTIVE", "EXPIRED", "SUPERSEDED"]


class PolicyEligibilityError(ValueError):
    """A content-free failure identified by a stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PolicyExclusion:
    document_id: str
    reasons: tuple[ExclusionReason, ...]


@dataclass(frozen=True, slots=True)
class PolicyEligibilityResult:
    validity_date: date
    eligible_document_ids: tuple[str, ...]
    exclusions: tuple[PolicyExclusion, ...]
    next_transition_date: date | None


def _index_documents(documents: Sequence[PolicyMetadata]) -> dict[str, PolicyMetadata]:
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes, bytearray)):
        raise PolicyEligibilityError("INVALID_DOCUMENTS")
    if not documents:
        raise PolicyEligibilityError("EMPTY_DOCUMENTS")

    # Guard keys used for ordering and graph checks before revalidating the
    # entire model. Pydantic's trusted construction APIs can bypass validation.
    for document in documents:
        if type(document) is not PolicyMetadata:
            raise PolicyEligibilityError("INVALID_DOCUMENT_TYPE")
        fields = vars(document)
        if (
            type(fields.get("document_id")) is not str
            or type(fields.get("policy_id")) is not str
            or type(fields.get("language")) is not str
            or type(fields.get("version")) is not int
            or "supersedes" not in fields
            or (document.supersedes is not None and type(document.supersedes) is not str)
        ):
            raise PolicyEligibilityError("INVALID_METADATA")

    indexed = {}
    slots: set[tuple[str, str, int]] = set()
    for document in sorted(documents, key=lambda item: item.document_id):
        if document.document_id in indexed:
            raise PolicyEligibilityError("DUPLICATE_DOCUMENT_ID")
        slot = (document.policy_id, document.language, document.version)
        if slot in slots:
            raise PolicyEligibilityError("DUPLICATE_VERSION_SLOT")
        slots.add(slot)
        indexed[document.document_id] = document
    return indexed


def _superseded_ids(documents: dict[str, PolicyMetadata]) -> set[str]:
    links = {
        document_id: document.supersedes
        for document_id, document in documents.items()
        if document.supersedes is not None
    }
    for successor, predecessor in links.items():
        if predecessor == successor:
            raise PolicyEligibilityError("SELF_SUPERSESSION")
        if predecessor not in documents:
            raise PolicyEligibilityError("MISSING_PREDECESSOR")
        if documents[predecessor].policy_id != documents[successor].policy_id:
            raise PolicyEligibilityError("WRONG_PREDECESSOR_FAMILY")
        if documents[predecessor].language != documents[successor].language:
            raise PolicyEligibilityError("WRONG_PREDECESSOR_LANGUAGE")

    # Check cycles before version ordering so a cycle has its own diagnostic.
    visited: set[str] = set()
    for successor in links:
        path: set[str] = set()
        cursor = successor
        while cursor in links and cursor not in visited:
            if cursor in path:
                raise PolicyEligibilityError("CYCLE")
            path.add(cursor)
            cursor = links[cursor]
        visited.update(path)

    for successor, predecessor in links.items():
        if documents[predecessor].version >= documents[successor].version:
            raise PolicyEligibilityError("PREDECESSOR_NOT_EARLIER")

    superseded = set(links.values())
    if len(superseded) != len(links):
        raise PolicyEligibilityError("BRANCHING_SUCCESSORS")
    return superseded


def select_policy_versions(
    documents: Sequence[PolicyMetadata], *, validity_date: date,
) -> PolicyEligibilityResult:
    """Select approved, effective, unexpired, structurally current versions.

    An effective date is inclusive and an expiration date is exclusive. A
    valid supersession link always excludes its predecessor, even when the
    successor is ineligible. This is current-policy selection. Historical
    as-of-date retrieval is outside its scope.

    Results are ordered by document ID. Each exclusion retains every
    applicable reason in approval, effective date, expiry, supersession
    order. The next transition is the earliest strictly future effective or
    expiration date in the input, so reevaluation may happen conservatively
    early. A caller must reevaluate when that date is reached.

    The caller supplies validated corpus metadata. Publication must bind the
    result to the exact validated snapshot identity. Read authorization is
    separate and always uses the original ACL metadata.
    """
    if type(validity_date) is not date:
        raise PolicyEligibilityError("INVALID_VALIDITY_DATE")
    indexed = _index_documents(documents)
    superseded = _superseded_ids(indexed)
    try:
        indexed = {
            document_id: PolicyMetadata.model_validate(vars(document))
            for document_id, document in indexed.items()
        }
    except ValidationError:
        raise PolicyEligibilityError("INVALID_METADATA") from None

    eligible_ids: list[str] = []
    eligible_slots: set[tuple[str, str]] = set()
    exclusions: list[PolicyExclusion] = []
    for document_id, document in indexed.items():
        reasons: list[ExclusionReason] = []
        if document.approval_status != "approved":
            reasons.append("NOT_APPROVED")
        if document.effective_date > validity_date:
            reasons.append("NOT_YET_EFFECTIVE")
        if document.expires_at is not None and validity_date >= document.expires_at:
            reasons.append("EXPIRED")
        if document_id in superseded:
            reasons.append("SUPERSEDED")
        if reasons:
            exclusions.append(PolicyExclusion(document_id, tuple(reasons)))
            continue

        slot = (document.policy_id, document.language)
        if slot in eligible_slots:
            raise PolicyEligibilityError("VERSION_CONFLICT")
        eligible_slots.add(slot)
        eligible_ids.append(document_id)

    next_transition = min(
        (
            boundary
            for document in indexed.values()
            for boundary in (document.effective_date, document.expires_at)
            if boundary is not None and boundary > validity_date
        ),
        default=None,
    )
    return PolicyEligibilityResult(validity_date, tuple(eligible_ids), tuple(exclusions), next_transition)
