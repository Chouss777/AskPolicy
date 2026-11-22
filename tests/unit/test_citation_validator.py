"""Synthetic mechanical checks, without corpus, model, network, or database I/O.

Passage fixtures represent trusted hydrator outputs, not proof of their provenance.
The real hydration boundary is exercised by its separate regression suites.
"""

from copy import copy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import traceback

import jwt
from pydantic import SecretStr
import pytest


ROOT = Path(__file__).resolve().parents[2]
RELEASE = "a" * 64
SNAPSHOT = "b" * 64


@pytest.fixture
def boundary(monkeypatch):
    # package=false: import src directly, without changing installed packages.
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from askpolicy.auth.context import AuthenticatedContext, authenticate_request
    from askpolicy.auth.jwt import JWTConfig, JWTVerifier
    from askpolicy.generation.schemas import Citation, GenerationOutput
    from askpolicy.retrieval.evidence_hydration import AuthorizedPassage
    from askpolicy.security import citation_validator as module

    verifier = JWTVerifier(JWTConfig(
        SecretStr("synthetic-citation-test-signing-key-1234567890"),
        "synthetic-citation-issuer", "synthetic-citation-audience", "HS256",
    ))
    now = int(datetime.now(UTC).timestamp())

    def authenticate(role="employee"):
        token = jwt.encode({
            "sub": "synthetic-citation-reviewer", "iss": verifier.config.issuer,
            "aud": verifier.config.audience, "iat": now - 60, "exp": now + 600,
            "roles": ["employee", "hr", "finance", "it_security"],
        }, verifier.config.signing_secret.get_secret_value(), algorithm="HS256")
        return authenticate_request(verifier, token, active_role=role)

    def passage(text="Synthetic evidence permits a review.", *, number=1, role="employee"):
        return AuthorizedPassage(
            release_id=RELEASE, snapshot_id=SNAPSHOT, active_role=role,
            child_id=f"child:{number:064x}", parent_id=f"parent:{number:064x}",
            document_id="FIXTURE-999-EN-V1", start=10, end=10 + len(text), text=text,
        )

    context = authenticate()
    evidence = (passage(),)
    citations = [Citation(chunk_id=evidence[0].child_id, quote="permits a review.")]

    def validate(cited=None, supplied=None, **changes):
        return module.validate_citations(
            citations if cited is None else cited, evidence if supplied is None else supplied,
            context=changes.get("context", context),
            release_id=changes.get("release_id", RELEASE),
            snapshot_id=changes.get("snapshot_id", SNAPSHOT),
        )

    return SimpleNamespace(
        module=module, Citation=Citation, Output=GenerationOutput, Passage=AuthorizedPassage,
        Context=AuthenticatedContext, authenticate=authenticate, context=context,
        passage=passage, evidence=evidence, citations=citations, validate=validate,
    )


def _rejected(boundary, *args, **kwargs):
    with pytest.raises(boundary.module.CitationValidationError) as captured:
        boundary.validate(*args, **kwargs)
    error = captured.value
    assert error.code == str(error) == "CITATION_VALIDATION_FAILED"
    assert error.__context__ is None and error.__cause__ is None
    return captured


@pytest.mark.parametrize("role", ["employee", "hr", "finance", "it_security"])
def test_valid_citation_uses_verified_active_role(boundary, role):
    evidence = (boundary.passage(role=role),)
    assert boundary.validate(supplied=evidence, context=boundary.authenticate(role)) is None


@pytest.mark.parametrize("text,quote", [
    ("  La demande synthétique peut être examinée.\n", "demande synthétique"),
    ("\tA synthetic request may be reviewed.\n", "request may be reviewed."),
    ("  يمكن مراجعة الطلب الاصطناعي.\n", "مراجعة الطلب الاصطناعي"),
    ("Ligne A\r\n\tLigne B  ", "A\r\n\tLigne B  "),
    ("Texte e\u0301 et é distincts.", "e\u0301 et é"),
])
def test_verbatim_unicode_substrings_preserve_original_text(boundary, text, quote):
    passage = boundary.passage(text)
    citation = boundary.Citation(chunk_id=passage.child_id, quote=quote)
    before = citation.model_dump()
    assert boundary.validate([citation], (passage,)) is None
    assert citation.model_dump() == before
    assert passage.text == text


def test_multiple_citations_and_exact_step118_output(boundary):
    first = boundary.passage()
    second = boundary.passage("Another synthetic rule.", number=2)
    output = boundary.Output.model_validate({
        "status": "answer", "answer": "Synthetic response.",
        "citations": [
            {"chunk_id": second.child_id, "quote": second.text},
            {"chunk_id": first.child_id, "quote": "permits a review."},
            {"chunk_id": first.child_id, "quote": "Synthetic evidence"},
        ],
    })
    before = output.model_dump()
    assert boundary.validate(output.citations, (first, second)) is None
    assert output.model_dump() == before


@pytest.mark.parametrize("chunk_id", ["fabricated", "child:" + "f" * 64, "parent:" + "1" * 64])
def test_fabricated_or_nonchild_citations_are_rejected(boundary, chunk_id):
    _rejected(boundary, [boundary.Citation(chunk_id=chunk_id, quote="permits a review.")])


def test_authorized_but_unsupplied_passage_is_out_of_set(boundary):
    other = boundary.passage("Other authorized evidence.", number=2)
    citation = boundary.Citation(chunk_id=other.child_id, quote=other.text)
    assert boundary.validate([citation], (other,)) is None
    _rejected(boundary, [citation], boundary.evidence)


@pytest.mark.parametrize("quote", [
    "Permits a review.", "permits  a review.",
    "permits\na review.", "a review. invented", "completely absent",
])
def test_modified_or_invented_quotes_are_rejected(boundary, quote):
    citation = boundary.Citation(chunk_id=boundary.evidence[0].child_id, quote=quote)
    _rejected(boundary, [citation])


@pytest.mark.parametrize("text,quote", [
    ("Only e\u0301 appears.", "é"),
    ("line one\nline two", "one\r\nline"),
    ("word gap", "word\u00a0gap"),
    ("النص الأصلي", "النص الاصلي"),
])
def test_no_unicode_newline_or_spacing_normalization(boundary, text, quote):
    passage = boundary.passage(text)
    _rejected(boundary, [boundary.Citation(chunk_id=passage.child_id, quote=quote)], (passage,))


@pytest.mark.parametrize("field,value", [
    ("active_role", "hr"), ("active_role", "unknown"), ("active_role", None),
    ("snapshot_id", "c" * 64), ("snapshot_id", "main"), ("snapshot_id", None),
    ("release_id", "d" * 64), ("release_id", "latest"), ("release_id", None),
])
@pytest.mark.parametrize("cited", [True, False])
def test_every_passage_must_match_trusted_context_even_when_uncited(boundary, field, value, cited):
    if cited:
        evidence = (replace(boundary.evidence[0], **{field: value}),)
    else:
        evidence = boundary.evidence + (replace(boundary.passage(number=2), **{field: value}),)
    _rejected(boundary, supplied=evidence)


def test_multigrant_identity_cannot_reuse_hr_evidence_in_employee_context(boundary):
    evidence = (boundary.passage(role="hr"),)
    assert boundary.validate(supplied=evidence, context=boundary.authenticate("hr")) is None
    _rejected(boundary, supplied=evidence, context=boundary.authenticate("employee"))


@pytest.mark.parametrize("kind", ["none", "role_string", "lookalike", "unregistered", "copy", "admin"])
def test_unverified_context_cannot_supply_authority(boundary, kind):
    contexts = {
        "none": None, "role_string": "employee",
        "lookalike": SimpleNamespace(active_role="employee", grants=("employee",)),
        "unregistered": boundary.Context(), "copy": copy(boundary.context),
        "admin": SimpleNamespace(capabilities=("ingestion_admin",)),
    }
    _rejected(boundary, context=contexts[kind])


@pytest.mark.parametrize("name", ["release_id", "snapshot_id"])
@pytest.mark.parametrize("value", [None, "", "main", "A" * 64, "a" * 63, 1, True])
def test_trusted_identity_parameters_must_use_existing_snapshot_id_format(boundary, name, value):
    _rejected(boundary, **{name: value})


@pytest.mark.parametrize("kind", ["identical", "text", "document", "parent", "uncited"])
def test_duplicate_or_ambiguous_evidence_identity_rejects_whole_batch(boundary, kind):
    passage = boundary.evidence[0] if kind != "uncited" else boundary.passage(number=2)
    duplicate = passage
    if kind == "text":
        duplicate = replace(passage, text="x" * len(passage.text))
    elif kind == "document":
        duplicate = replace(passage, document_id="FIXTURE-998-EN-V1")
    elif kind == "parent":
        duplicate = replace(passage, parent_id="parent:" + "f" * 64)
    evidence = (passage, duplicate) if kind != "uncited" else boundary.evidence + (passage, duplicate)
    _rejected(boundary, supplied=evidence)


@pytest.mark.parametrize("field,value", [
    ("child_id", ""), ("child_id", " \t"), ("child_id", 1), ("child_id", []),
    ("parent_id", None), ("parent_id", ""), ("parent_id", " "),
    ("document_id", None), ("document_id", ""), ("document_id", "\n"),
    ("start", True), ("start", -1), ("start", 10.0), ("start", "10"),
    ("end", False), ("end", 10), ("end", 9), ("end", 11), ("end", None),
    ("text", ""), ("text", None), ("text", b"Synthetic evidence permits a review."),
])
def test_malformed_passage_fields_fail_closed(boundary, field, value):
    _rejected(boundary, supplied=(replace(boundary.evidence[0], **{field: value}),))


@pytest.mark.parametrize("kind", ["empty", "list", "mapping", "ids", "namespace", "ranking", "subclass", "missing_field"])
def test_invalid_evidence_structures_and_raw_rankings_are_not_authorization(boundary, kind):
    passage = boundary.evidence[0]
    if kind == "empty":
        evidence = ()
    elif kind == "list":
        evidence = [passage]
    elif kind == "mapping":
        evidence = (asdict(passage),)
    elif kind == "ids":
        evidence = (passage.child_id,)
    elif kind == "namespace":
        evidence = (SimpleNamespace(**asdict(passage)),)
    elif kind == "ranking":
        evidence = (SimpleNamespace(child_id=passage.child_id, score=1.0, active_role="employee"),)
    elif kind == "subclass":
        class PretendPassage(boundary.Passage):
            pass
        evidence = (PretendPassage(**asdict(passage)),)
    else:
        evidence = (object.__new__(boundary.Passage),)
    _rejected(boundary, supplied=evidence)


@pytest.mark.parametrize("kind", ["empty", "tuple", "mapping", "namespace", "constructed", "copied", "mutated", "extra", "subclass"])
def test_invalid_or_bypassed_citation_structures_fail_closed(boundary, kind):
    valid = boundary.citations[0]
    if kind == "empty":
        citations = []
    elif kind == "tuple":
        citations = (valid,)
    elif kind == "mapping":
        citations = [valid.model_dump()]
    elif kind == "namespace":
        citations = [SimpleNamespace(**valid.model_dump())]
    elif kind == "constructed":
        citations = [boundary.Citation.model_construct(chunk_id=valid.chunk_id, quote=" \n")]
    elif kind == "copied":
        citations = [valid.model_copy(update={"quote": 42})]
    elif kind == "mutated":
        valid.quote = ""
        citations = [valid]
    elif kind == "extra":
        citations = [valid.model_copy(update={"active_role": "hr"})]
    else:
        class PretendCitation(boundary.Citation):
            pass
        citations = [PretendCitation(**valid.model_dump())]
    _rejected(boundary, citations)


@pytest.mark.parametrize("bad_first", [False, True])
def test_any_failed_citation_rejects_complete_batch_without_partial_return(boundary, bad_first):
    invalid = boundary.Citation(chunk_id="fabricated", quote="Synthetic evidence")
    citations = [invalid, *boundary.citations] if bad_first else [*boundary.citations, invalid]
    before = [item.model_dump() for item in citations]
    for _ in range(3):
        _rejected(boundary, citations)
        assert [item.model_dump() for item in citations] == before
    assert boundary.validate() is None


def test_validation_is_deterministic_order_independent_and_has_no_shared_state(boundary):
    evidence = boundary.evidence + (boundary.passage("Another synthetic rule.", number=2),)
    for _ in range(3):
        assert boundary.validate(supplied=evidence) is None
        assert boundary.validate(supplied=tuple(reversed(evidence))) is None
    _rejected(boundary, supplied=(replace(evidence[0], release_id="c" * 64),))
    assert boundary.validate(supplied=evidence) is None


def test_same_quote_in_different_supplied_chunk_does_not_validate_wrong_id(boundary):
    other = boundary.passage("A different statement.", number=2)
    citation = boundary.Citation(chunk_id=other.child_id, quote=boundary.evidence[0].text)
    _rejected(boundary, [citation], boundary.evidence + (other,))


@pytest.mark.parametrize("kind", ["unknown_id", "quote", "role", "duplicate", "malformed"])
def test_content_free_errors_have_no_payload_logs_or_exception_chain(boundary, caplog, kind):
    secret = "SYNTHETIC_RESTRICTED_CANARY"
    passage = replace(boundary.passage(secret), document_id="SYNTHETIC-RESTRICTED-DOCUMENT")
    citations = [boundary.Citation(chunk_id=passage.child_id, quote=secret)]
    evidence = (passage,)
    if kind == "unknown_id":
        citations[0].chunk_id = secret
    elif kind == "quote":
        citations[0].quote = secret + " mismatch"
    elif kind == "role":
        evidence = (replace(passage, active_role="hr"),)
    elif kind == "duplicate":
        evidence = evidence * 2
    else:
        citations[0].quote = {"sensitive": secret}
    captured = _rejected(boundary, citations, evidence)
    formatted = "".join(traceback.format_exception(captured.type, captured.value, captured.tb))
    assert secret not in formatted
    assert passage.document_id not in formatted
    assert passage.child_id not in formatted
    assert not caplog.records


def test_no_global_search_or_hydration_fallback_on_missing_evidence(boundary, monkeypatch):
    from askpolicy.retrieval import evidence_hydration, hybrid
    from askpolicy.storage import index_store

    def forbidden(*args, **kwargs):
        pytest.fail("Citation validation must not perform retrieval or hydration")

    monkeypatch.setattr(evidence_hydration, "hydrate_published_hits", forbidden)
    monkeypatch.setattr(hybrid, "search_role_hybrid", forbidden)
    monkeypatch.setattr(index_store.PublishedSnapshot, "search", forbidden)
    assert boundary.validate() is None
    _rejected(boundary, supplied=())


def test_mechanical_validation_does_not_claim_answer_truth(boundary):
    output = boundary.Output.model_validate({
        "status": "answer", "answer": "An unrelated synthetic assertion.",
        "citations": [item.model_dump() for item in boundary.citations],
    })
    assert boundary.validate(output.citations) is None
    # This is not a semantic-support or complete-generation acceptance test.
