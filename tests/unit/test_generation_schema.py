"""Offline structural checks; these do not certify citation security or truth."""

import json
from pathlib import Path
import runpy

from pydantic import ValidationError
import pytest


# package=false: exercise the actual src module without installing AskPolicy.
ROOT = Path(__file__).resolve().parents[2]
schemas = runpy.run_path(str(ROOT / "src/askpolicy/generation/schemas.py"))
Citation = schemas["Citation"]
GenerationOutput = schemas["GenerationOutput"]


def _answer():
    return {
        "status": "answer",
        "answer": "Synthetic staff may request a review.",
        "citations": [{"chunk_id": "synthetic evidence", "quote": "A review may be requested."}],
    }


def _refusal():
    return {"status": "refusal", "answer": None, "citations": []}


@pytest.mark.parametrize(
    "text",
    [
        "Une demande synthétique peut être examinée.",
        "A synthetic request may be reviewed.",
        "يمكن مراجعة الطلب الاصطناعي.",
    ],
    ids=["fr", "en", "ar"],
)
def test_valid_answer_in_supported_languages(text):
    result = GenerationOutput.model_validate(_answer() | {"answer": text})
    assert result.status == "answer"
    assert result.answer == text
    assert len(result.citations) == 1
    assert isinstance(result.citations[0], Citation)
    assert result.citations[0].model_dump() == _answer()["citations"][0]


def test_valid_refusal_contains_no_explanation_or_evidence():
    assert GenerationOutput.model_validate(_refusal()).model_dump() == _refusal()


def test_multiple_citations_keep_input_order():
    payload = _answer()
    payload["citations"].append({"chunk_id": "another synthetic ID", "quote": "Another statement."})
    assert GenerationOutput.model_validate(payload).model_dump() == payload


@pytest.mark.parametrize("field", ["status", "answer", "citations"])
def test_output_fields_are_required(field):
    payload = _refusal()
    del payload[field]
    with pytest.raises(ValidationError) as error:
        GenerationOutput.model_validate(payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())


@pytest.mark.parametrize("field", ["chunk_id", "quote"])
def test_citation_fields_are_required(field):
    citation = _answer()["citations"][0]
    del citation[field]
    with pytest.raises(ValidationError) as error:
        GenerationOutput.model_validate(_answer() | {"citations": [citation]})
    assert any(item["loc"] == ("citations", 0, field) for item in error.value.errors())


@pytest.mark.parametrize("field", ["explanation", "document_id", "active_role", "snapshot_id"])
def test_unknown_output_fields_cannot_carry_refusal_content_or_metadata(field):
    with pytest.raises(ValidationError) as error:
        GenerationOutput.model_validate(_refusal() | {field: "synthetic content"})
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("field", ["document_id", "allowed_roles", "source_text"])
def test_unknown_nested_citation_fields_are_forbidden(field):
    citation = _answer()["citations"][0] | {field: "synthetic content"}
    with pytest.raises(ValidationError) as error:
        GenerationOutput.model_validate(_answer() | {"citations": [citation]})
    assert error.value.errors()[0]["loc"] == ("citations", 0, field)
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("value", [0, 1.5, True, b"text", [], {}, ("text",)])
def test_answer_does_not_coerce_primitive_types(value):
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_answer() | {"answer": value})


@pytest.mark.parametrize("field", ["chunk_id", "quote"])
@pytest.mark.parametrize("value", [None, 12, 1.5, False, b"text", [], {}, ("text",)])
def test_citation_fields_do_not_coerce_primitive_types(field, value):
    citation = _answer()["citations"][0] | {field: value}
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_answer() | {"citations": [citation]})


@pytest.mark.parametrize("value", [None, 0, False, "[]", b"[]", {}, ()])
def test_citations_must_be_a_list_without_coercion(value):
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_refusal() | {"citations": value})


@pytest.mark.parametrize("value", ["", "ANSWER", "Refusal", "unknown", " answer ", None, 1, True, b"answer", [], {}])
def test_invalid_status_is_rejected(value):
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_answer() | {"status": value})


@pytest.mark.parametrize("value", [None, "", " ", "\t\n\r", "\u00a0\u2003"])
def test_answer_requires_non_whitespace_text(value):
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_answer() | {"answer": value})


@pytest.mark.parametrize("field", ["chunk_id", "quote"])
@pytest.mark.parametrize("value", ["", " ", "\n\t\r", "\u00a0\u2003"])
def test_citation_fields_require_non_whitespace_text(field, value):
    with pytest.raises(ValidationError):
        Citation.model_validate(_answer()["citations"][0] | {field: value})


def test_answer_requires_at_least_one_citation():
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_answer() | {"citations": []})


@pytest.mark.parametrize("value", ["An explanation.", "", " \n"])
def test_refusal_requires_null_answer_even_for_empty_text(value):
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_refusal() | {"answer": value})


def test_refusal_must_not_contain_citations():
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_refusal() | {"citations": _answer()["citations"]})


@pytest.mark.parametrize("value", [None, "chunk", 1, False, [], {}, {"chunk_id": "synthetic"}, {"quote": "text"}])
def test_malformed_nested_citation_objects_are_rejected(value):
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(_answer() | {"citations": [value]})


@pytest.mark.parametrize("payload", [_answer(), _refusal()])
def test_json_parsing_and_serialization_round_trip(payload):
    result = GenerationOutput.model_validate_json(json.dumps(payload, ensure_ascii=False))
    assert json.loads(result.model_dump_json()) == payload
    assert GenerationOutput.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "raw",
    [
        "{", "null", "[]", '"text"',
        '{"status":"refusal","citations":[]}',
        '{"status":"refusal","answer":null,"citations":[],"explanation":"text"}',
        '{"status":"answer","answer":12,"citations":[{"chunk_id":"id","quote":"text"}]}',
        '{"status":"answer","answer":"text","citations":[{"chunk_id":1,"quote":"text"}]}',
        '{"status":"answer","answer":"text","citations":[{"chunk_id":"id","quote":true}]}',
        '{"status":"refusal","answer":null,"citations":"[]"}',
        '{"status":"refusal","answer":"explanation","citations":[]}',
    ],
)
def test_invalid_or_nonconforming_json_is_rejected(raw):
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate_json(raw)


@pytest.mark.parametrize("quote", ["  Original text.\n\t", "L’employé : e\u0301 ≠ é.\r\n", "\tالنص الأصلي.\n", "A\u00a0B\u2003C"])
def test_quotation_and_unformatted_id_are_preserved_exactly(quote):
    citation = {"chunk_id": "  arbitrary ID / no prescribed format  ", "quote": quote}
    payload = _answer() | {"answer": "  Synthetic answer.\n", "citations": [citation]}
    result = GenerationOutput.model_validate(payload)
    assert result.model_dump() == payload
    assert result.citations[0].quote.encode("utf-8") == quote.encode("utf-8")
    restored = GenerationOutput.model_validate_json(result.model_dump_json())
    assert restored.model_dump() == payload


def test_repeated_validation_and_schema_calls_are_stable_and_lists_are_independent():
    payload = _answer()
    expected_schema = GenerationOutput.model_json_schema()
    results = [GenerationOutput.model_validate(payload) for _ in range(3)]
    for result in results:
        assert result.model_dump() == payload
        assert GenerationOutput.model_json_schema() == expected_schema
    results[0].citations.clear()
    assert len(results[1].citations) == len(results[2].citations) == len(payload["citations"]) == 1
    refusal_a = GenerationOutput.model_validate(_refusal())
    refusal_b = GenerationOutput.model_validate(_refusal())
    assert refusal_a.citations is not refusal_b.citations


@pytest.mark.parametrize("model,fields", [(Citation, {"chunk_id", "quote"}), (GenerationOutput, {"status", "answer", "citations"})])
def test_schema_has_only_required_contract_fields_and_strict_configuration(model, fields):
    schema = model.model_json_schema()
    assert set(schema["properties"]) == set(schema["required"]) == fields
    assert schema["additionalProperties"] is False
    assert model.model_config["strict"] is True
    assert model.model_config["extra"] == "forbid"


@pytest.mark.parametrize("failure", ["nested", "invariant", "extra"])
def test_validation_does_not_log_payloads_or_embed_them_in_error_strings(caplog, failure):
    sentinel = "SYNTHETIC_SENSITIVE_PAYLOAD_DO_NOT_LOG"
    payload = _answer() | {"answer": sentinel}
    if failure == "nested":
        payload["citations"][0]["quote"] = {"sensitive": sentinel}
    elif failure == "invariant":
        payload["status"] = "refusal"
    else:
        payload["unexpected"] = sentinel
    with pytest.raises(ValidationError) as error:
        GenerationOutput.model_validate(payload)
    assert sentinel not in str(error.value)
    assert not caplog.records
    # Callers must still avoid logging raw payloads or ValidationError.errors().


def test_structural_acceptance_does_not_certify_citation_security():
    payload = _answer() | {
        "citations": [{"chunk_id": "not-looked-up", "quote": "Not compared with stored evidence."}],
    }
    assert GenerationOutput.model_validate(payload).model_dump() == payload
    # Authorization, supplied-evidence membership, snapshot and quote matching
    # remain server-side responsibilities; a model's answer status grants none.
