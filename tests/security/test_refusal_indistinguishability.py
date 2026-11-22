"""Guard-level security checks; no generation HTTP or Ollama integration."""

from copy import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from security.test_auth import secured


REFUSAL = {"status": "refusal", "answer": None, "citations": []}
RELEASE = "a" * 64
SNAPSHOT = "b" * 64


@pytest.fixture
def guarded(secured):
    from askpolicy.generation.schemas import Citation, GenerationOutput
    from askpolicy.retrieval.evidence_hydration import AuthorizedPassage
    from askpolicy.security import refusal

    context = secured.authenticate(secured.sign(roles=["employee", "hr"]), "employee")

    def passage(text="Synthetic evidence permits a review.", *, number=1, role="employee"):
        # Synthetic stand-ins for trusted hydrator outputs; not provenance proof.
        return AuthorizedPassage(
            release_id=RELEASE, snapshot_id=SNAPSHOT, active_role=role,
            child_id=f"child:{number:064x}", parent_id=f"parent:{number:064x}",
            document_id="FIXTURE-999-EN-V1", start=10, end=10 + len(text), text=text,
        )

    evidence = (passage(),)
    payload = {
        "status": "answer", "answer": "A synthetic review is permitted.",
        "citations": [{"chunk_id": evidence[0].child_id, "quote": "permits a review."}],
    }

    def guard(raw=json.dumps(payload), **changes):
        return refusal.guard_generation_response(
            raw, context=changes.get("context", context),
            evidence=changes.get("evidence", evidence),
            release_id=changes.get("release_id", RELEASE),
            snapshot_id=changes.get("snapshot_id", SNAPSHOT),
            eligibility=changes.get("eligibility", refusal.EvidenceEligibility.ELIGIBLE),
        )

    return SimpleNamespace(
        module=refusal, decisions=refusal.EvidenceEligibility, context=context,
        evidence=evidence, payload=payload, guard=guard, passage=passage,
        Output=GenerationOutput, Citation=Citation, secured=secured,
    )


@pytest.mark.parametrize("decision", [
    "INSUFFICIENT_EVIDENCE", "UNAUTHORIZED", "UNAVAILABLE",
    "UNRESOLVED_CONFLICT", "UNCERTAIN_VALIDITY",
])
def test_every_server_denial_is_identical_and_prevents_answer_validation(guarded, monkeypatch, decision):
    def forbidden(*args, **kwargs):
        pytest.fail("Denied evidence must not reach answer validation")

    monkeypatch.setattr(guarded.module, "_checked_answer", forbidden)
    result = guarded.guard(eligibility=guarded.decisions[decision])
    assert result.model_dump() == REFUSAL


def test_unauthorized_and_unavailable_have_identical_public_bytes(guarded):
    restricted = replace(guarded.passage("SYNTHETIC_RESTRICTED_QUOTATION", role="hr"),
                         document_id="SYNTHETIC-RESTRICTED-ID")
    unauthorized = guarded.guard(
        evidence=(restricted,), eligibility=guarded.decisions.UNAUTHORIZED,
    )
    unavailable = guarded.guard(evidence=(), eligibility=guarded.decisions.UNAVAILABLE)
    assert unauthorized.model_dump() == unavailable.model_dump() == REFUSAL
    assert unauthorized.model_dump_json() == unavailable.model_dump_json()
    assert "SYNTHETIC" not in unauthorized.model_dump_json()


@pytest.mark.parametrize("value", [
    None, True, False, 1, "eligible", "ELIGIBLE", "unknown", {},
    {"decision": "eligible"}, SimpleNamespace(value="eligible"),
])
def test_eligibility_rejects_untyped_client_or_model_values(guarded, value):
    assert guarded.guard(eligibility=value).model_dump() == REFUSAL


@pytest.mark.parametrize("field", ["eligibility", "active_role", "allowed_roles", "release_id", "snapshot_id"])
def test_model_metadata_cannot_grant_eligibility_or_authority(guarded, field):
    payload = guarded.payload | {field: "eligible"}
    assert guarded.guard(json.dumps(payload)).model_dump() == REFUSAL
    assert guarded.guard(json.dumps(guarded.payload), eligibility="eligible").model_dump() == REFUSAL


@pytest.mark.parametrize("kind", ["none", "lookalike", "unregistered", "copied", "identity"])
@pytest.mark.parametrize("decision", ["ELIGIBLE", "UNAVAILABLE"])
def test_unverified_request_is_an_auth_error_never_a_successful_refusal(guarded, kind, decision):
    from askpolicy.auth.context import ActiveRoleError, AuthenticatedContext

    contexts = {
        "none": None, "lookalike": SimpleNamespace(active_role="employee"),
        "unregistered": AuthenticatedContext(), "copied": copy(guarded.context),
        "identity": guarded.secured.verifier.verify(guarded.secured.sign()),
    }
    with pytest.raises(ActiveRoleError, match="^UNVERIFIED_CONTEXT$"):
        guarded.guard(context=contexts[kind], eligibility=guarded.decisions[decision])


@pytest.mark.parametrize("kind", ["authentication", "role", "http401", "http403"])
def test_dependency_authentication_errors_propagate_unchanged(guarded, monkeypatch, kind):
    from fastapi import HTTPException
    from askpolicy.auth.context import ActiveRoleError
    from askpolicy.auth.jwt import AuthenticationError

    errors = {
        "authentication": AuthenticationError("INVALID_TOKEN"),
        "role": ActiveRoleError("ROLE_NOT_GRANTED"),
        "http401": HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"}),
        "http403": HTTPException(403, "Access denied"),
    }
    expected = errors[kind]

    def fail(*args, **kwargs):
        raise expected

    monkeypatch.setattr(guarded.module, "validate_citations", fail)
    with pytest.raises(type(expected)) as captured:
        guarded.guard()
    assert captured.value is expected


def test_wrong_active_role_refuses_even_for_multigrant_identity(guarded):
    hr = guarded.secured.authenticate(guarded.secured.sign(roles=["employee", "hr"]), "hr")
    evidence = (guarded.passage(role="hr"),)
    assert guarded.guard(context=hr, evidence=evidence).status == "answer"
    assert guarded.guard(evidence=evidence).model_dump() == REFUSAL


@pytest.mark.parametrize("reason", ["malformed", "schema", "citation", "unexpected", "http503"])
def test_refusals_expose_no_sensitive_payload_exception_or_log_details(guarded, monkeypatch, caplog, capsys, reason):
    from fastapi import HTTPException

    secret = "SYNTHETIC_RESTRICTED_CANARY"
    payload = guarded.payload | {"answer": secret}
    raw = json.dumps(payload)
    if reason == "malformed":
        raw += secret
    elif reason == "schema":
        payload["citations"][0]["quote"] = {"secret": secret}
        raw = json.dumps(payload)
    elif reason == "citation":
        payload["citations"][0]["quote"] = secret
        raw = json.dumps(payload)
    else:
        def fail(*args, **kwargs):
            if reason == "http503":
                raise HTTPException(503, secret)
            raise RuntimeError(secret)
        monkeypatch.setattr(guarded.module, "validate_citations", fail)
    result = guarded.guard(raw)
    assert result.model_dump() == REFUSAL
    assert secret not in result.model_dump_json()
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_no_retrieval_hydration_or_sqlite_fallback(guarded, monkeypatch):
    import sqlite3
    from askpolicy.retrieval import evidence_hydration, hybrid
    from askpolicy.storage.index_store import PublishedSnapshot

    def forbidden(*args, **kwargs):
        pytest.fail("Response guarding must not retrieve or hydrate evidence")

    monkeypatch.setattr(evidence_hydration, "hydrate_published_hits", forbidden)
    monkeypatch.setattr(hybrid, "search_role_hybrid", forbidden)
    monkeypatch.setattr(PublishedSnapshot, "search", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    assert guarded.guard().status == "answer"
    assert guarded.guard(evidence=()).model_dump() == REFUSAL
