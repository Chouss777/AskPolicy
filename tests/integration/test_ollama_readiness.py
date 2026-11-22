"""Readiness/inference-boundary integration with synthetic approved records and mocked local HTTP."""
import copy
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unit.test_ollama_client import local, Reply, rejected, PIN


def test_healthy_daemon_and_explicit_approval_do_not_load_or_generate(local):
    local.state.replies = local.ready()
    ready = local.client.check_readiness()
    assert ready.daemon_version == "0.40.17"
    assert ready.model == "qwen3:4b-instruct" and ready.digest == PIN
    assert ready.thinking_supported is True
    assert [(c["method"], c["path"]) for c in local.state.calls] == [
        ("GET", "/api/version"), ("GET", "/api/tags"), ("POST", "/api/show"),
    ]
    assert local.state.replies == []


@pytest.mark.parametrize("data,code", [
    ({"models": []}, "OLLAMA_MODEL_MISSING"),
    ({"models": [{"name": "qwen3.5:4b", "model": "qwen3.5:4b", "digest": PIN}]}, "OLLAMA_MODEL_MISSING"),
    ({"models": [{"name": "qwen3:4b-instruct", "model": "qwen3:4b-instruct", "digest": PIN[:12]}]}, "OLLAMA_DIGEST_MISMATCH"),
    ({"models": [{"name": "qwen3:4b-instruct", "model": "qwen3:4b-instruct", "digest": "f"*64}]}, "OLLAMA_DIGEST_MISMATCH"),
    ({"models": [{"name": "qwen3:4b-instruct", "model": "qwen3.5:4b", "digest": PIN}]}, "OLLAMA_MODEL_MISMATCH"),
    ({"models": {}}, "OLLAMA_PROTOCOL_ERROR"),
    ({"models": [None]}, "OLLAMA_PROTOCOL_ERROR"),
])
def test_installed_metadata_failures_stop_before_show_or_chat(local, data, code):
    local.state.replies = local.ready(tag_data=data)
    rejected(local, local.client.check_readiness, code)
    assert [c["path"] for c in local.state.calls] == ["/api/version", "/api/tags"]


def test_duplicate_model_entries_are_ambiguous(local):
    tags = copy.deepcopy(local.tags)
    tags["models"] *= 2
    local.state.replies = local.ready(tag_data=tags)
    rejected(local, local.client.check_readiness, "OLLAMA_MODEL_MISMATCH")


@pytest.mark.parametrize("version", [{}, {"version": ""}, {"version": 1}, {"version": "v"*129}])
def test_daemon_version_must_be_valid_bounded_metadata(local, version):
    local.state.replies = local.ready(version_data=version)
    rejected(local, local.client.check_readiness, "OLLAMA_PROTOCOL_ERROR")
    assert len(local.state.calls) == 1


@pytest.mark.parametrize("defect", ["architecture", "tokenizer", "context", "bool_context", "system", "capabilities", "thinking_false_unsupported", "cloud_host", "cloud_model", "not_gguf"])
def test_unsupported_runtime_profile_fails_before_inference(local, defect):
    show = copy.deepcopy(local.show)
    if defect == "architecture":
        show["model_info"]["general.architecture"] = "qwen35"
    elif defect == "tokenizer":
        show["model_info"]["tokenizer.ggml.model"] = "unknown"
    elif defect == "context":
        show["model_info"]["qwen3.context_length"] = 2048
    elif defect == "bool_context":
        show["model_info"]["qwen3.context_length"] = True
    elif defect == "system":
        show["system"] = "An unbudgeted hidden prompt."
    elif defect == "capabilities":
        del show["capabilities"]
    elif defect == "thinking_false_unsupported":
        show["thinking"] = {"values": [True]}
    elif defect == "cloud_host":
        show["remote_host"] = "https://cloud.example"
    elif defect == "cloud_model":
        show["remote_model"] = "synthetic-cloud-model"
    else:
        show["details"]["format"] = "remote"
    local.state.replies = local.ready(show_data=show)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_MODEL_UNSUPPORTED")
    assert "/api/chat" not in [c["path"] for c in local.state.calls]


@pytest.mark.parametrize("field", ["remote_host", "remote_model"])
def test_cloud_alias_in_tags_is_rejected_before_further_requests(local, field):
    tags = copy.deepcopy(local.tags)
    tags["models"][0][field] = "synthetic-remote-reference"
    local.state.replies = local.ready(tag_data=tags)
    rejected(local, local.client.check_readiness, "OLLAMA_MODEL_UNSUPPORTED")
    assert [c["path"] for c in local.state.calls] == ["/api/version", "/api/tags"]


def test_supported_thinking_metadata_sends_explicit_false(local):
    show = copy.deepcopy(local.show)
    show["thinking"] = {"values": [False, True], "default": True}
    local.generation(show_data=show)
    assert local.client.generate("Synthetic system", "Synthetic evidence") == local.text
    import json
    assert json.loads(local.state.calls[3]["body"])["think"] is False


def test_changed_tag_after_generation_never_releases_answer(local):
    tags = copy.deepcopy(local.tags)
    tags["models"][0]["digest"] = "e"*64
    local.generation(after_tags=tags)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_DIGEST_MISMATCH")
    assert [c["path"] for c in local.state.calls] == [
        "/api/version", "/api/tags", "/api/show", "/api/chat", "/api/tags",
    ]


@pytest.mark.parametrize("defect", ["missing", "digest", "short_digest", "model", "context", "duplicate"])
def test_loaded_identity_must_match_the_fresh_pin_and_context(local, defect):
    resident = copy.deepcopy(local.loaded)
    if defect == "missing":
        resident["models"] = []
    elif defect == "duplicate":
        resident["models"] *= 2
    else:
        field, value = {
            "digest": ("digest", "f"*64), "short_digest": ("digest", PIN[:12]),
            "model": ("model", "qwen3.5:4b"), "context": ("context_length", 8192),
        }[defect]
        resident["models"][0][field] = value
    local.generation(resident=resident)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_DIGEST_MISMATCH")


def test_readiness_success_is_not_reused_as_a_cached_permission(local):
    local.state.replies = local.ready()
    local.client.check_readiness()
    tags = copy.deepcopy(local.tags)
    tags["models"][0]["digest"] = "d"*64
    local.state.replies = local.ready(tag_data=tags)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_DIGEST_MISMATCH")
    assert "/api/chat" not in [c["path"] for c in local.state.calls]


def test_readiness_connection_failure_is_named_and_fail_closed(local):
    local.state.request_error = ConnectionRefusedError("synthetic local daemon unavailable")
    rejected(local, local.client.check_readiness, "OLLAMA_DAEMON_UNAVAILABLE")
    assert len(local.state.calls) == 1

