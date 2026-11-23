"""Exact owner-decision binding; all daemon replies below are synthetic."""

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from unit.test_ollama_client import local, rejected


MODEL = "qwen3:4b-instruct"
DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
REFERENCE = "docs/decisions.md#step-125-generator-digest-approval"


@pytest.fixture
def approved(local):
    from askpolicy.generation import approved_model
    return approved_model


def test_exact_record_uses_existing_model_constant_and_dataclass(approved, local):
    record = approved.get_generator_approval()
    assert type(record) is local.module.GeneratorApproval
    assert record.model == local.module.MODEL == MODEL
    assert record.digest == DIGEST
    assert record.approval_reference == REFERENCE
    assert local.state.calls == []


def test_reference_resolves_to_the_explicit_owner_decision(approved):
    record = approved.get_generator_approval()
    path, anchor = record.approval_reference.split("#")
    decisions = (ROOT / path).read_text(encoding="utf-8")
    marker = f'<a id="{anchor}"></a>'
    assert decisions.count(marker) == 1
    decision = decisions.split(marker, 1)[1].split("\n---", 1)[0]
    assert "### Step 125 owner approval — existing generator digest" in decision
    assert "2026-11-22 (November 22, 2026)" in decision
    assert "the project owner explicitly approved this exact checkpoint" in decision
    assert f"| Model | `{MODEL}` |" in decision
    assert f"| Full SHA-256 digest | `{DIGEST}` |" in decision
    assert f"| Approval reference | `{REFERENCE}` |" in decision
    assert "`qwen3.5:4b` remains unapproved for production" in decision
    assert "tests/unit/test_model_approval.py" in decision
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert f"approved `{MODEL}` with full SHA-256 digest `{DIGEST}`" in agents
    assert "docs/decisions.md#step-125-generator-digest-approval" in agents


def test_identical_explicit_values_are_consistency_checks_only(approved):
    assert approved.get_generator_approval(
        model=MODEL, digest=DIGEST, approval_reference=REFERENCE,
    ) == approved.get_generator_approval()


@pytest.mark.parametrize("override", [
    {"digest": "f" * 64},
    {"digest": DIGEST[:12]},
    {"digest": DIGEST.upper()},
    {"digest": "sha256:" + DIGEST},
    {"digest": " " + DIGEST},
    {"digest": ""},
    {"digest": None},
    {"digest": True},
    {"digest": DIGEST.encode()},
    {"model": "qwen3.5:4b"},
    {"model": "qwen3:4b"},
    {"model": "qwen3:4b-instruct:latest"},
    {"model": ""},
    {"model": None},
    {"model": True},
    {"model": MODEL.encode()},
    {"approval_reference": "SYNTHETIC_TEST_APPROVAL_ONLY"},
    {"approval_reference": "docs/decisions.md#unknown-approval"},
    {"approval_reference": ""},
    {"approval_reference": None},
    {"approval_reference": True},
    {"approval_reference": REFERENCE.encode()},
])
def test_unauthorized_overrides_fail_before_io(approved, local, override, caplog, capsys):
    rejected(local, lambda: approved.get_generator_approval(**override), "OLLAMA_MODEL_NOT_APPROVED")
    assert local.state.calls == []
    assert caplog.records == []
    assert capsys.readouterr() == ("", "")


def test_environment_cannot_replace_the_owner_record(approved, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("OLLAMA_DIGEST", "f" * 64)
    monkeypatch.setenv("OLLAMA_APPROVAL_REFERENCE", "unapproved-environment-value")
    record = approved.get_generator_approval()
    assert (record.model, record.digest, record.approval_reference) == (MODEL, DIGEST, REFERENCE)


@pytest.mark.parametrize("field,value", [
    ("model", "qwen3.5:4b"), ("digest", "f" * 64), ("approval_reference", "unapproved"),
])
def test_approval_record_is_immutable(approved, field, value):
    record = approved.get_generator_approval()
    with pytest.raises(FrozenInstanceError):
        setattr(record, field, value)


def test_configuration_does_not_change_the_clients_default_deny(approved, local):
    approved.get_generator_approval()
    client = local.module.OllamaClient("http://127.0.0.1:11434")
    rejected(local, client.check_readiness, "OLLAMA_MODEL_NOT_APPROVED")
    rejected(local, lambda: client.generate("Synthetic system", "Synthetic question"),
             "OLLAMA_MODEL_NOT_APPROVED")
    assert local.state.calls == []


def test_trusted_configuration_works_with_existing_readiness_without_inference(approved, local):
    local.state.replies = local.ready()
    client = local.module.OllamaClient(
        "http://127.0.0.1:11434", approval=approved.get_generator_approval(),
    )
    ready = client.check_readiness()
    assert (ready.model, ready.digest) == (MODEL, DIGEST)
    assert [(call["method"], call["path"]) for call in local.state.calls] == [
        ("GET", "/api/version"), ("GET", "/api/tags"), ("POST", "/api/show"),
    ]


@pytest.mark.parametrize("installed", ["f" * 64, DIGEST[:12], None])
def test_installed_mismatch_still_fails_with_real_approval_binding(approved, local, installed):
    tags = copy.deepcopy(local.tags)
    tags["models"][0]["digest"] = installed
    local.state.replies = local.ready(tag_data=tags)
    client = local.module.OllamaClient(
        "http://127.0.0.1:11434", approval=approved.get_generator_approval(),
    )
    rejected(local, client.check_readiness, "OLLAMA_DIGEST_MISMATCH")
    assert [call["path"] for call in local.state.calls] == ["/api/version", "/api/tags"]


def test_missing_model_is_not_replaced_or_approved(approved, local):
    local.state.replies = local.ready(tag_data={"models": []})
    client = local.module.OllamaClient(
        "http://127.0.0.1:11434", approval=approved.get_generator_approval(),
    )
    rejected(local, client.check_readiness, "OLLAMA_MODEL_MISSING")
    assert [call["path"] for call in local.state.calls] == ["/api/version", "/api/tags"]
