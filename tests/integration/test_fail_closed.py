"""Combine the real schema, citation validator and guard with synthetic evidence.

These tests do not exercise a generation HTTP route or Ollama inference.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from security.test_refusal_indistinguishability import guarded, secured, REFUSAL


def test_fully_valid_answer_and_citations_are_preserved(guarded):
    result = guarded.guard()
    assert result.model_dump() == guarded.payload
    assert all(type(citation) is guarded.Citation for citation in result.citations)


def test_multiple_citations_and_original_order_are_preserved(guarded):
    second = guarded.passage("Another synthetic rule.", number=2)
    payload = guarded.payload | {"citations": [
        {"chunk_id": second.child_id, "quote": second.text},
        *guarded.payload["citations"],
        {"chunk_id": second.child_id, "quote": "synthetic rule"},
    ]}
    result = guarded.guard(json.dumps(payload), evidence=guarded.evidence + (second,))
    assert result.model_dump() == payload


@pytest.mark.parametrize("answer,quote", [
    ("  Une demande synthétique peut être examinée.\n", "é et e\u0301 : texte synthétique.\n"),
    ("\tA synthetic request may be reviewed.  ", "Synthetic evidence.\r\n\t"),
    ("يمكن مراجعة الطلب الاصطناعي.", "\tالنص الاصطناعي الأصلي.  "),
    ("Synthetic Unicode remains unrestricted: 😀", "An exact quotation: 😀"),
])
def test_fr_en_ar_answers_and_quotations_remain_exact(guarded, answer, quote):
    passage = guarded.passage("Prefix " + quote + " Suffix")
    payload = {"status": "answer", "answer": answer,
               "citations": [{"chunk_id": passage.child_id, "quote": quote}]}
    raw = json.dumps(payload, ensure_ascii=False)
    result = guarded.guard(raw, evidence=(passage,))
    assert result.model_dump() == payload
    assert result.citations[0].quote.encode("utf-8") == quote.encode("utf-8")


@pytest.mark.parametrize("payload", [
    REFUSAL,
    REFUSAL | {"answer": "A model-generated refusal explanation."},
    REFUSAL | {"explanation": "A restricted document exists."},
    REFUSAL | {"document_id": "SYNTHETIC-RESTRICTED-ID"},
])
def test_model_refusal_and_explanations_become_fresh_controlled_refusal(guarded, payload):
    assert guarded.guard(json.dumps(payload)).model_dump() == REFUSAL


@pytest.mark.parametrize("raw", [
    "", "{", "not JSON", "```json\n{}\n```", "{} trailing", "{}{}",
    "null", "[]", '"text"', "42", "true",
    '{"status":"answer","status":"refusal","answer":null,"citations":[]}',
    '{"status":"refusal","status":"answer","answer":"text","citations":[]}',
    '{"status":"answer","answer":NaN,"citations":[]}',
    '{"status":"answer","answer":Infinity,"citations":[]}',
])
def test_malformed_ambiguous_or_nonconforming_json_refuses(guarded, raw):
    assert guarded.guard(raw).model_dump() == REFUSAL


@pytest.mark.parametrize("text", ["\ud800", "\udfff"])
def test_unpaired_json_surrogate_escapes_are_rejected(guarded, text):
    raw = json.dumps(guarded.payload | {"answer": text})
    assert guarded.guard(raw).model_dump() == REFUSAL


def test_duplicate_nested_citation_keys_refuse_even_when_last_value_would_pass(guarded):
    chunk_id = guarded.evidence[0].child_id
    raw = ('{"status":"answer","answer":"Synthetic answer.","citations":['
           '{"chunk_id":"invented","chunk_id":' + json.dumps(chunk_id)
           + ',"quote":"permits a review."}]}')
    assert guarded.guard(raw).model_dump() == REFUSAL


@pytest.mark.parametrize("field", ["status", "answer", "citations"])
def test_missing_required_fields_refuse(guarded, field):
    payload = guarded.payload.copy()
    del payload[field]
    assert guarded.guard(json.dumps(payload)).model_dump() == REFUSAL


@pytest.mark.parametrize("changes", [
    {"status": "unknown"}, {"answer": None}, {"answer": ""}, {"answer": " \n"},
    {"answer": 1}, {"answer": True}, {"citations": []}, {"citations": None},
    {"citations": "[]"}, {"citations": [None]}, {"citations": [{}]},
    {"extra": "SYNTHETIC_RESTRICTED"},
])
def test_invalid_schema_is_never_accepted_as_a_partial_answer(guarded, changes):
    result = guarded.guard(json.dumps(guarded.payload | changes))
    assert result.model_dump() == REFUSAL


@pytest.mark.parametrize("citation", [
    {"chunk_id": "invented", "quote": "permits a review."},
    {"chunk_id": "", "quote": "permits a review."},
    {"chunk_id": "child:" + f"{1:064x}", "quote": "Permits a review."},
    {"chunk_id": "child:" + f"{1:064x}", "quote": "permits  a review."},
    {"chunk_id": "child:" + f"{1:064x}", "quote": " \n"},
    {"chunk_id": "child:" + f"{1:064x}", "quote": 123},
    {"chunk_id": "child:" + f"{1:064x}", "quote": "permits a review.", "active_role": "hr"},
])
def test_invalid_schema_or_mechanical_citation_refuses(guarded, citation):
    payload = guarded.payload | {"citations": [citation]}
    assert guarded.guard(json.dumps(payload)).model_dump() == REFUSAL


def test_authorized_but_unsupplied_evidence_is_out_of_set(guarded):
    other = guarded.passage("Other synthetic evidence.", number=2)
    payload = guarded.payload | {"citations": [{"chunk_id": other.child_id, "quote": other.text}]}
    raw = json.dumps(payload)
    assert guarded.guard(raw, evidence=(other,)).status == "answer"
    assert guarded.guard(raw).model_dump() == REFUSAL


@pytest.mark.parametrize("defect", [
    "empty", "duplicate", "stale_snapshot", "wrong_release", "wrong_role",
    "bad_structure", "bad_offsets", "uncited_invalid",
])
def test_evidence_validation_failure_refuses_whole_answer(guarded, defect):
    passage = guarded.evidence[0]
    evidence = (passage,)
    if defect == "empty":
        evidence = ()
    elif defect == "duplicate":
        evidence *= 2
    elif defect == "stale_snapshot":
        evidence = (replace(passage, snapshot_id="c" * 64),)
    elif defect == "wrong_release":
        evidence = (replace(passage, release_id="d" * 64),)
    elif defect == "wrong_role":
        evidence = (replace(passage, active_role="hr"),)
    elif defect == "bad_structure":
        evidence = (passage.child_id,)
    elif defect == "bad_offsets":
        evidence = (replace(passage, start=-1),)
    else:
        evidence += (replace(guarded.passage(number=2), snapshot_id="c" * 64),)
    assert guarded.guard(evidence=evidence).model_dump() == REFUSAL


@pytest.mark.parametrize("field", ["release_id", "snapshot_id"])
@pytest.mark.parametrize("value", [None, "main", "f" * 64])
def test_invalid_or_mismatched_trusted_identity_refuses(guarded, field, value):
    assert guarded.guard(**{field: value}).model_dump() == REFUSAL


@pytest.mark.parametrize("bad_first", [True, False])
def test_one_invalid_citation_discards_all_previously_valid_content(guarded, bad_first):
    invalid = {"chunk_id": "invented", "quote": "SYNTHETIC_RESTRICTED_CANARY"}
    valid = guarded.payload["citations"][0]
    citations = [invalid, valid] if bad_first else [valid, invalid]
    payload = guarded.payload | {"citations": citations}
    for _ in range(3):
        assert guarded.guard(json.dumps(payload)).model_dump() == REFUSAL
    assert guarded.guard().model_dump() == guarded.payload


def test_previously_checked_output_is_revalidated_from_json_and_not_mutated(guarded):
    checked = guarded.Output.model_validate(guarded.payload)
    before = checked.model_dump()
    first = guarded.guard(checked.model_dump_json())
    second = guarded.guard(checked.model_dump_json())
    assert first.model_dump() == second.model_dump() == before
    assert first is not checked and first.citations[0] is not checked.citations[0]
    first.citations.clear()
    assert checked.model_dump() == second.model_dump() == before


@pytest.mark.parametrize("kind", ["valid_model", "constructed_model", "copied_model", "mapping", "bytes"])
def test_non_json_inputs_never_bypass_strict_revalidation(guarded, kind):
    checked = guarded.Output.model_validate(guarded.payload)
    if kind == "valid_model":
        raw = checked
    elif kind == "constructed_model":
        raw = guarded.Output.model_construct(status="answer", answer=123, citations=[])
    elif kind == "copied_model":
        raw = checked.model_copy(update={"citations": []})
    elif kind == "mapping":
        raw = guarded.payload
    else:
        raw = json.dumps(guarded.payload).encode()
    assert guarded.guard(raw).model_dump() == REFUSAL
    assert checked.model_dump() == guarded.payload


def test_refusal_results_are_fresh_and_do_not_share_citations(guarded):
    first = guarded.guard("invalid JSON")
    second = guarded.guard("invalid JSON")
    assert first is not second and first.citations is not second.citations
    first.citations.append(guarded.Citation(**guarded.payload["citations"][0]))
    assert second.model_dump() == guarded.guard("invalid JSON").model_dump() == REFUSAL


def test_guard_checks_mechanics_without_certifying_semantic_entailment(guarded):
    payload = guarded.payload | {"answer": "An unrelated synthetic assertion."}
    assert guarded.guard(json.dumps(payload)).model_dump() == payload
