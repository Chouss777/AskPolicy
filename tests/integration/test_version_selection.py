from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
from itertools import permutations
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DAY = date(2026, 9, 1)
BEFORE = DAY - timedelta(days=1)
AFTER = DAY + timedelta(days=1)
LATER = DAY + timedelta(days=3)
LINKS = {
    "ACCESS-001-FR-V2": "ACCESS-001-FR-V1",
    "DATA-001-FR-V2": "DATA-001-FR-V1",
    "EXPENSES-001-FR-V2": "EXPENSES-001-FR-V1",
    "LEAVE-001-FR-V2": "LEAVE-001-FR-V1",
    "REMOTE-001-EN-V2": "REMOTE-001-EN-V1",
    "REMOTE-001-FR-V2": "REMOTE-001-FR-V1",
}


@pytest.fixture(scope="module")
def api():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.ingestion import metadata
        from askpolicy.storage import eligibility

        yield metadata, eligibility


@pytest.fixture
def policy(api):
    metadata, _ = api

    def build(
        version=1, *, family="FIXTURE-901", language="fr", effective=DAY,
        expires=None, approval="approved", supersedes=None,
    ):
        return metadata.PolicyMetadata(
            policy_id=family, document_id=f"{family}-{language.upper()}-V{version}",
            version=version, language=language, title="Synthetic policy",
            department="HR", classification="RESTRICTED", allowed_roles=("hr",),
            effective_date=effective, expires_at=expires, supersedes=supersedes,
            approval_status=approval, source_type="synthetic_policy", source_license="project_owned",
        )

    return build


def _select(api, documents, validity_date=DAY):
    _, eligibility = api
    return eligibility.select_policy_versions(documents, validity_date=validity_date)


def _reasons(result):
    return {entry.document_id: entry.reasons for entry in result.exclusions}


def _assert_rejected(api, documents, code, validity_date=DAY):
    _, eligibility = api
    with pytest.raises(eligibility.PolicyEligibilityError) as raised:
        _select(api, documents, validity_date)
    assert raised.value.code == code
    assert str(raised.value) == code


def test_full_corpus_current_versions_at_their_effective_boundary(api):
    metadata, _ = api
    paths = sorted((ROOT / "data/policies").rglob("*.md"))
    documents = tuple(
        metadata.parse_policy_metadata(
            path.read_text(encoding="utf-8"), source_path=path.relative_to(ROOT).as_posix(),
        )
        for path in paths
    )
    before = tuple(document.model_dump() for document in documents)
    assert len(documents) == 26
    assert {doc.document_id: doc.supersedes for doc in documents if doc.supersedes is not None} == LINKS
    predecessor_ids = set(LINKS.values())
    current = tuple(doc for doc in documents if doc.document_id not in predecessor_ids)
    assert len(current) == 20
    assert all(doc.effective_date == DAY and doc.expires_at is None for doc in current)

    prior = _select(api, documents, BEFORE)
    assert prior.eligible_document_ids == ()
    assert prior.next_transition_date == DAY

    result = _select(api, documents)
    assert result.validity_date == DAY
    assert result.eligible_document_ids == tuple(sorted(doc.document_id for doc in current))
    assert tuple(entry.document_id for entry in result.exclusions) == tuple(sorted(predecessor_ids))
    assert _reasons(result) == {key: ("SUPERSEDED",) for key in predecessor_ids}
    assert result.next_transition_date is None
    assert result == _select(api, documents[::-1])
    assert result == _select(api, list(documents[::2] + documents[1::2]))
    assert tuple(document.model_dump() for document in documents) == before

    selected = {doc.document_id: doc for doc in current}
    assert {lang: sum(doc.language == lang for doc in selected.values()) for lang in ("fr", "en", "ar")} == {
        "fr": 8, "en": 8, "ar": 4,
    }
    for lang in ("FR", "EN", "AR"):
        leave = selected[f"LEAVE-001-{lang}-V2"]
        assert leave.language == lang.lower()
        assert leave.version == 2
        assert leave.allowed_roles == ("hr",)
        assert leave.classification == "RESTRICTED"


@pytest.mark.parametrize("effective, expires, approval, reasons, transition", [
    (BEFORE, None, "approved", (), None),
    (DAY, None, "approved", (), None),
    (BEFORE, AFTER, "approved", (), AFTER),
    (BEFORE, DAY, "approved", ("EXPIRED",), None),
    (AFTER, None, "approved", ("NOT_YET_EFFECTIVE",), AFTER),
    (BEFORE, None, "not_approved", ("NOT_APPROVED",), None),
    (BEFORE, DAY, "not_approved", ("NOT_APPROVED", "EXPIRED"), None),
], ids=[
    "no-expiry", "effective-today", "day-before-expiration", "expires-today",
    "future-effective", "nonapproved", "multiple-reasons",
])
def test_eligibility_and_date_bounds(api, policy, effective, expires, approval, reasons, transition):
    doc = policy(effective=effective, expires=expires, approval=approval)
    result = _select(api, [doc])
    assert result.next_transition_date == transition
    if reasons:
        assert result.eligible_document_ids == ()
        assert _reasons(result) == {doc.document_id: reasons}
    else:
        assert result.eligible_document_ids == (doc.document_id,)
        assert result.exclusions == ()


@pytest.mark.parametrize("effective, expires, approval, reasons", [
    (AFTER, None, "approved", ("NOT_YET_EFFECTIVE",)),
    (DAY, None, "not_approved", ("NOT_APPROVED",)),
    (BEFORE, DAY, "approved", ("EXPIRED",)),
], ids=["future-successor", "nonapproved-successor", "expired-successor"])
def test_ineligible_successor_never_reactivates_its_predecessor(api, policy, effective, expires, approval, reasons):
    older = policy(effective=BEFORE)
    newer = policy(2, effective=effective, expires=expires, approval=approval, supersedes=older.document_id)
    assert _select(api, [older]).eligible_document_ids == (older.document_id,)
    result = _select(api, [older, newer])
    assert result.eligible_document_ids == ()
    assert _reasons(result) == {older.document_id: ("SUPERSEDED",), newer.document_id: reasons}


def test_entire_ancestry_is_excluded_with_all_applicable_reasons(api, policy):
    first = policy(effective=AFTER, approval="not_approved")
    second = policy(2, supersedes=first.document_id)
    third = policy(3, supersedes=second.document_id)
    result = _select(api, [third, first, second])
    assert result.eligible_document_ids == (third.document_id,)
    assert _reasons(result) == {
        first.document_id: ("NOT_APPROVED", "NOT_YET_EFFECTIVE", "SUPERSEDED"),
        second.document_id: ("SUPERSEDED",),
    }
    # A conservative transition can require reevaluation for a date on
    # an ineligible document without changing the selected version.
    assert result.next_transition_date == AFTER
    assert _select(api, [first, second, third], AFTER).eligible_document_ids == result.eligible_document_ids


def test_independent_family_language_slots_and_ordered_output(api, policy):
    older = policy(effective=BEFORE)
    newer = policy(2, supersedes=older.document_id)
    arabic = policy(2, language="ar")
    english = policy(language="en", effective=AFTER)
    other_family = policy(family="FIXTURE-902")
    assert arabic.supersedes is None
    documents = (older, newer, arabic, english, other_family)
    expected_ids = (arabic.document_id, newer.document_id, other_family.document_id)
    expected_excluded = (english.document_id, older.document_id)
    baseline = _select(api, documents)
    assert baseline.eligible_document_ids == expected_ids
    assert tuple(entry.document_id for entry in baseline.exclusions) == expected_excluded
    for ordered in permutations(documents):
        assert _select(api, ordered) == baseline
    with pytest.raises(FrozenInstanceError):
        baseline.validity_date = AFTER
    with pytest.raises(FrozenInstanceError):
        baseline.exclusions[0].reasons = ()


def test_reevaluation_at_earliest_effective_and_expiration_dates(api, policy):
    expiring = policy(effective=BEFORE, expires=AFTER)
    pending = policy(family="FIXTURE-902", effective=AFTER, expires=LATER)
    documents = [pending, expiring]

    initial = _select(api, documents, DAY)
    assert initial.eligible_document_ids == (expiring.document_id,)
    assert initial.next_transition_date == AFTER

    boundary = _select(api, documents, initial.next_transition_date)
    assert boundary.eligible_document_ids == (pending.document_id,)
    assert _reasons(boundary) == {expiring.document_id: ("EXPIRED",)}
    assert boundary.next_transition_date == LATER

    final = _select(api, documents, boundary.next_transition_date)
    assert final.eligible_document_ids == ()
    assert _reasons(final) == {doc.document_id: ("EXPIRED",) for doc in documents}
    assert final.next_transition_date is None


def test_empty_input_rejected(api, policy):
    valid = [policy()]
    assert _select(api, valid).eligible_document_ids
    _assert_rejected(api, valid[:0], "EMPTY_DOCUMENTS")


def test_non_sequence_rejected(api, policy):
    valid = [policy()]
    assert _select(api, valid).eligible_document_ids
    _assert_rejected(api, iter(valid), "INVALID_DOCUMENTS")


def test_metadata_lookalike_rejected(api, policy):
    doc = policy()
    assert _select(api, [doc]).eligible_document_ids
    _assert_rejected(api, [doc.model_dump()], "INVALID_DOCUMENT_TYPE")


def test_duplicate_document_rejected(api, policy):
    doc = policy()
    assert _select(api, [doc]).eligible_document_ids
    _assert_rejected(api, [doc, doc], "DUPLICATE_DOCUMENT_ID")


def test_duplicate_slot_with_inconsistent_second_identity_rejected(api, policy):
    first = policy()
    second = policy(2, supersedes=first.document_id)
    assert _select(api, [first, second]).eligible_document_ids == (second.document_id,)
    forged = second.model_copy(update={"version": first.version})
    assert first.document_id != forged.document_id
    _assert_rejected(api, [first, forged], "DUPLICATE_VERSION_SLOT")


@pytest.mark.parametrize("update", [
    {"document_id": "FIXTURE-901-EN-V1"},
    {"approval_status": "unrecognized"},
    {"allowed_roles": ()},
], ids=["identity", "approval", "acl"])
def test_bypassed_metadata_validation_is_detected(api, policy, update):
    doc = policy()
    assert _select(api, [doc]).eligible_document_ids
    _assert_rejected(api, [doc.model_copy(update=update)], "INVALID_METADATA")


def test_dangling_predecessor_rejected(api, policy):
    first = policy()
    second = policy(2, supersedes=first.document_id)
    assert _select(api, [first, second]).eligible_document_ids == (second.document_id,)
    _assert_rejected(api, [second], "MISSING_PREDECESSOR")


@pytest.mark.parametrize("target, code", [
    ("other_family", "WRONG_PREDECESSOR_FAMILY"),
    ("other_language", "WRONG_PREDECESSOR_LANGUAGE"),
    ("later", "PREDECESSOR_NOT_EARLIER"),
    ("successor", "SELF_SUPERSESSION"),
    ("oldest", "BRANCHING_SUCCESSORS"),
])
def test_invalid_graph_edge_rejected(api, policy, target, code):
    oldest = policy()
    middle = policy(2, supersedes=oldest.document_id)
    successor = policy(3, supersedes=middle.document_id)
    later = policy(4, effective=AFTER)
    other_family = policy(family="FIXTURE-902")
    other_language = policy(language="en")
    targets = {
        "oldest": oldest, "successor": successor, "later": later,
        "other_family": other_family, "other_language": other_language,
    }
    others = [oldest, middle, later, other_family, other_language]
    assert successor.document_id in _select(api, [*others, successor]).eligible_document_ids
    # model_copy bypasses the model's per-document edge checks so the
    # selection boundary must diagnose the single forged relationship.
    broken = successor.model_copy(update={"supersedes": targets[target].document_id})
    _assert_rejected(api, [*others, broken], code)


def test_supersession_cycle_rejected(api, policy):
    first = policy()
    second = policy(2, supersedes=first.document_id)
    third = policy(3, supersedes=second.document_id)
    assert _select(api, [first, second, third]).eligible_document_ids == (third.document_id,)
    broken = first.model_copy(update={"supersedes": third.document_id})
    _assert_rejected(api, [broken, second, third], "CYCLE")


def test_overlapping_eligible_versions_rejected_without_largest_version_fallback(api, policy):
    first = policy()
    second = policy(2, effective=AFTER)
    assert _select(api, [first, second]).eligible_document_ids == (first.document_id,)
    _assert_rejected(api, [first, second], "VERSION_CONFLICT", AFTER)
    _assert_rejected(api, [second, first], "VERSION_CONFLICT", AFTER)


@pytest.mark.parametrize("invalid", [datetime(2026, 9, 1), "2026-09-01", None])
def test_validity_date_requires_a_date_only_value(api, policy, invalid):
    docs = [policy()]
    assert _select(api, docs).eligible_document_ids
    _assert_rejected(api, docs, "INVALID_VALIDITY_DATE", invalid)
