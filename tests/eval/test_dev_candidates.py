import json
import re
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DATASET = REPOSITORY_ROOT / "eval" / "datasets" / "dev_candidates.jsonl"
ANNOTATION_REVIEW = REPOSITORY_ROOT / "docs" / "annotation-review.md"
POLICY_DIRECTORY = REPOSITORY_ROOT / "data" / "policies" / "fr"
TRAVEL_DOCUMENT = "TRAVEL-001-FR-V1"
LEAVE_DOCUMENT = "LEAVE-001-FR-V2"
READ_ROLES = {"employee", "hr", "finance", "it_security"}
RECORD_FIELDS = {
    "query_id",
    "question_family_id",
    "language",
    "active_role",
    "query",
    "expected_status",
    "authorized_source_sections",
    "required_facts",
    "case_type",
    "annotation_review_status",
    "revalidation_required",
}
SOURCE_FIELDS = {"document_id", "section_heading", "relevance_grade"}


@pytest.fixture(scope="module")
def candidates():
    lines = CANDIDATE_DATASET.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    return [json.loads(line) for line in lines]


def test_candidates_have_required_fields_and_valid_labels(candidates):
    for case in candidates:
        assert isinstance(case, dict)
        assert set(case) == RECORD_FIELDS
        assert case["language"] == "fr"
        assert case["active_role"] in READ_ROLES
        assert case["annotation_review_status"] == "human_approved"
        assert isinstance(case["revalidation_required"], bool)
        assert case["expected_status"] in {"answer", "refusal"}
        assert case["case_type"] in {"answerable", "role_sensitive", "unanswerable"}

        for field in ("query_id", "question_family_id", "query"):
            assert isinstance(case[field], str) and case[field].strip(), field
        assert isinstance(case["required_facts"], list)
        assert all(
            isinstance(fact, str) and fact.strip() for fact in case["required_facts"]
        )

        sources = case["authorized_source_sections"]
        assert isinstance(sources, list)
        for source in sources:
            assert isinstance(source, dict)
            assert set(source) == SOURCE_FIELDS
            assert isinstance(source["section_heading"], str) and source["section_heading"]
            assert type(source["relevance_grade"]) is int and source["relevance_grade"] > 0


def test_candidate_query_ids_are_unique(candidates):
    query_ids = [case["query_id"] for case in candidates]
    assert len(set(query_ids)) == len(query_ids)
    assert {case["query_id"]: case["question_family_id"] for case in candidates} == {
        "dev-candidate-001": "dev-family-001",
        "dev-candidate-002": "dev-family-002",
        "dev-candidate-003": "dev-family-003",
        "dev-candidate-004": "dev-family-003",
        "dev-candidate-005": "dev-family-004",
    }


def test_answerable_executions_retain_relevance_grade_three(candidates):
    answers = [case for case in candidates if case["expected_status"] == "answer"]
    assert len(answers) == 3
    for case in answers:
        sources = case["authorized_source_sections"]
        assert sources
        assert all(source["relevance_grade"] == 3 for source in sources)


def test_annotation_review_defines_relevance_grade_three():
    review = ANNOTATION_REVIEW.read_text(encoding="utf-8")
    assert (
        "Grade 3 = directly and fully relevant evidence that explicitly supports "
        "the required answer."
    ) in review


def test_only_candidate_005_requires_revalidation(candidates):
    flagged = {case["query_id"] for case in candidates if case["revalidation_required"]}
    assert flagged == {"dev-candidate-005"}


def test_annotation_review_requires_revalidation_before_freezing():
    review = ANNOTATION_REVIEW.read_text(encoding="utf-8")
    gate = review.partition("## Revalidation gate\n")[2].split("\n## ", 1)[0]
    assert (
        "Revalidate `dev-candidate-005` after corpus expansion and before evaluation "
        "dataset freezing."
    ) in gate
    assert "Do not automatically clear `revalidation_required`." in gate


def test_answerable_cases_cover_travel_and_leave(candidates):
    answerable = [case for case in candidates if case["case_type"] == "answerable"]
    assert len(answerable) == 2
    documents_by_role = {
        case["active_role"]: {
            source["document_id"] for source in case["authorized_source_sections"]
        }
        for case in answerable
    }
    assert documents_by_role == {"employee": {TRAVEL_DOCUMENT}, "hr": {LEAVE_DOCUMENT}}
    assert all(case["expected_status"] == "answer" for case in answerable)


def test_role_sensitive_pair_shares_question_and_family(candidates):
    pair = [case for case in candidates if case["case_type"] == "role_sensitive"]
    assert len(pair) == 2
    by_role = {case["active_role"]: case for case in pair}
    assert set(by_role) == {"hr", "employee"}
    hr_case = by_role["hr"]
    employee_case = by_role["employee"]

    assert hr_case["query"] == employee_case["query"]
    assert hr_case["question_family_id"] == employee_case["question_family_id"]
    assert hr_case["expected_status"] == "answer"
    assert employee_case["expected_status"] == "refusal"
    assert {
        source["document_id"] for source in hr_case["authorized_source_sections"]
    } == {LEAVE_DOCUMENT}

    family_ids = [case["question_family_id"] for case in candidates]
    assert family_ids.count(hr_case["question_family_id"]) == 2
    assert len(set(family_ids)) == 4


def test_unanswerable_and_refused_cases_have_no_supporting_labels(candidates):
    unanswerable = [case for case in candidates if case["case_type"] == "unanswerable"]
    assert len(unanswerable) == 1
    assert unanswerable[0]["expected_status"] == "refusal"
    for case in candidates:
        if case["expected_status"] == "refusal":
            assert case["authorized_source_sections"] == []
            assert case["required_facts"] == []
        else:
            assert case["authorized_source_sections"]
            assert case["required_facts"]


def test_referenced_headings_exist_in_authorized_synthetic_policies(candidates):
    for case in candidates:
        for source in case["authorized_source_sections"]:
            document_id = source["document_id"]
            assert document_id in {TRAVEL_DOCUMENT, LEAVE_DOCUMENT}
            path = POLICY_DIRECTORY / f"{document_id}.md"
            assert path.is_file()
            lines = path.read_text(encoding="utf-8").splitlines()
            assert lines[0] == "---"
            closing = lines.index("---", 1)
            fields = dict(line.split(": ", 1) for line in lines[1:closing])
            allowed_roles = fields["allowed_roles"].removeprefix("[").removesuffix("]")
            assert case["active_role"] in allowed_roles.split(", ")
            assert fields["document_id"] == document_id
            assert fields["source_type"] == "synthetic_policy"
            assert f"## {source['section_heading']}" in lines[closing + 1:]


def test_candidate_text_uses_generic_wording_without_contact_identifiers(candidates):
    # Guards for this small synthetic seed. These do not constitute a PII detector.
    contact_identifier = re.compile(r"@|https?://|www\.|(?:\d[ .()+-]*){7}", re.IGNORECASE)
    generic_openers = {"Quel", "Qui", "Le"}
    policy_abbreviations = {"MAD", "RH"}

    for case in candidates:
        for text in [case["query"], *case["required_facts"]]:
            assert not contact_identifier.search(text)
            words = re.findall(r"[^\W\d_]+", text)
            assert words and words[0] in generic_openers
            assert all(
                word.islower() or word in policy_abbreviations for word in words[1:]
            ), f"Unexpected name or acronym in {case['query_id']}"
