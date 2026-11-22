"""Offline Ollama client contracts; approvals and HTTP replies are synthetic fixtures."""
import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import pytest


ROOT = Path(__file__).resolve().parents[2]
PIN = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"


class Reply:
    def __init__(self, value=None, *, raw=None, status=200, headers=None, read_error=None):
        self.raw = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode() if raw is None else raw
        self.status = status
        self.headers = {"Content-Type": "application/json", "Content-Length": str(len(self.raw))}
        if headers is not None:
            self.headers.update(headers)
        self.read_error = read_error
        self.reads = []
        self.closed = False
        self.hook = None

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, size):
        self.reads.append(size)
        if self.hook:
            self.hook()
        if self.read_error:
            raise self.read_error
        return self.raw[:size]

    def close(self):
        self.closed = True


@pytest.fixture
def local(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from askpolicy.generation import ollama_client as module

    state = SimpleNamespace(replies=[], calls=[], connections=[], timers=[], now=0.0,
                            request_error=None, expire=False)
    monkeypatch.setattr(module, "monotonic", lambda: state.now)

    class Sock:
        def __init__(self):
            self.timeouts = []
            self.shutdowns = []

        def settimeout(self, value):
            self.timeouts.append(value)

        def shutdown(self, how):
            self.shutdowns.append(how)

    class Connection:
        def __init__(self, host, port, timeout):
            self.host, self.port, self.timeout = host, port, timeout
            self.sock = Sock()
            self.debug = None
            self.closed = False
            state.connections.append(self)

        def set_debuglevel(self, level):
            self.debug = level

        def request(self, method, path, body=None, headers=None):
            state.calls.append({"method": method, "path": path, "body": body, "headers": headers,
                                "host": self.host, "port": self.port})
            if state.request_error:
                raise state.request_error

        def getresponse(self):
            assert state.replies, "Unexpected HTTP request or retry"
            reply = state.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        def close(self):
            self.closed = True

    class Timer:
        def __init__(self, interval, function):
            self.interval, self.function = interval, function
            self.started = self.cancelled = False
            state.timers.append(self)

        def start(self):
            self.started = True
            if state.expire:
                self.function()

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(module, "HTTPConnection", Connection)
    monkeypatch.setattr(module, "Timer", Timer)
    approval = module.GeneratorApproval(module.MODEL, PIN, "SYNTHETIC_TEST_APPROVAL_ONLY")
    client = module.OllamaClient("http://127.0.0.1:11434", approval=approval)
    tags = {"models": [{"name": module.MODEL, "model": module.MODEL, "digest": PIN}]}
    show = {"capabilities": ["completion", "thinking"],
            "details": {"format": "gguf", "family": "qwen3"},
            "model_info": {"general.architecture": "qwen3", "tokenizer.ggml.model": "gpt2",
                           "qwen3.context_length": 262144}}
    text = json.dumps({"status": "answer", "answer": "Synthetic answer.",
                       "citations": [{"chunk_id": "synthetic-not-authorized-here", "quote": "Synthetic quote."}]})
    response = {"model": module.MODEL, "done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": text},
                "prompt_eval_count": 100, "eval_count": 30}
    loaded = {"models": [{"name": module.MODEL, "model": module.MODEL, "digest": PIN, "context_length": 4096}]}

    def ready(*, tag_data=None, show_data=None, version_data=None):
        return [Reply({"version": "0.40.17"} if version_data is None else version_data),
                Reply(copy.deepcopy(tags) if tag_data is None else tag_data),
                Reply(copy.deepcopy(show) if show_data is None else show_data)]

    def generation(*, result=None, after_tags=None, resident=None, show_data=None):
        state.replies = ready(show_data=show_data) + [
            Reply(copy.deepcopy(response) if result is None else result),
            Reply(copy.deepcopy(tags) if after_tags is None else after_tags),
            Reply(copy.deepcopy(loaded) if resident is None else resident),
        ]
        return state.replies

    return SimpleNamespace(module=module, state=state, client=client, approval=approval, ready=ready,
                           generation=generation, tags=tags, show=show, response=response,
                           loaded=loaded, text=text, Reply=Reply)


def rejected(local, action, code):
    with pytest.raises(local.module.OllamaClientError) as caught:
        action()
    assert caught.value.code == str(caught.value) == code
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    return caught


@pytest.mark.parametrize("url,host", [
    ("http://127.0.0.1:11434", "127.0.0.1"),
    ("http://localhost:11434/", "127.0.0.1"),
    ("http://[::1]:11434", "::1"),
])
def test_explicit_loopback_endpoints_only(local, url, host):
    local.state.replies = local.ready()
    ready = local.module.OllamaClient(url, approval=local.approval).check_readiness()
    assert ready.model == local.module.MODEL and ready.digest == PIN
    assert all(c["host"] == host and c["port"] == 11434 for c in local.state.calls)


@pytest.mark.parametrize("url", [
    None, b"http://127.0.0.1:11434", "", "https://localhost:11434",
    "http://ollama.com:11434", "http://192.168.0.1:11434", "http://0.0.0.0:11434",
    "http://169.254.169.254:11434", "http://localhost.evil:11434",
    "http://localhost.:11434", "http://localhost:8000", "http://localhost",
    "http://localhost:0", "http://localhost:99999", "http://localhost:11434/api",
    "http://user:password@localhost:11434", "http://localhost:11434?token=secret",
    "http://localhost:11434#secret", "http://local\nhost:11434",
    "http://127.0.0.1:11434\\evil", "http://2130706433:11434",
])
def test_unsafe_or_ambiguous_endpoints_fail_before_io(local, url):
    rejected(local, lambda: local.module.OllamaClient(url, approval=local.approval), "OLLAMA_CONFIG_INVALID")
    assert local.state.calls == []


@pytest.mark.parametrize("digest", [None, "", PIN[:12], PIN.upper(), "a"*63, "g"*64, True])
def test_approval_requires_a_complete_exact_digest(local, digest):
    rejected(local, lambda: local.module.GeneratorApproval(local.module.MODEL, digest, "synthetic-owner-record"),
             "OLLAMA_MODEL_NOT_APPROVED")
    assert local.state.calls == []


@pytest.mark.parametrize("reference", [None, "", " \t", True, b"record", "r"*1025])
def test_approval_requires_an_explicit_record_reference(local, reference):
    rejected(local, lambda: local.module.GeneratorApproval(local.module.MODEL, PIN, reference),
             "OLLAMA_MODEL_NOT_APPROVED")


def test_observed_digest_never_creates_default_approval(local):
    local.state.replies = local.ready()
    client = local.module.OllamaClient("http://localhost:11434")
    rejected(local, client.check_readiness, "OLLAMA_MODEL_NOT_APPROVED")
    rejected(local, lambda: client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_MODEL_NOT_APPROVED")
    assert local.state.calls == []


@pytest.mark.parametrize("record", [True, {"model": "qwen3:4b-instruct", "digest": PIN},
                                   SimpleNamespace(model="qwen3:4b-instruct", digest=PIN, approval_reference="claimed")])
def test_client_metadata_and_boolean_flags_are_not_approval_records(local, record):
    rejected(local, lambda: local.module.OllamaClient("http://localhost:11434", approval=record),
             "OLLAMA_MODEL_NOT_APPROVED")


@pytest.mark.parametrize("model", ["qwen3.5:4b", "qwen3:latest", "other", None])
def test_wrong_generator_is_not_selectable(local, model):
    rejected(local, lambda: local.module.GeneratorApproval(model, PIN, "synthetic-record"),
             "OLLAMA_MODEL_NOT_APPROVED")
    assert local.state.calls == []


def test_constructor_has_no_io_and_configuration_is_immutable(local):
    assert local.state.calls == local.state.connections == []
    with pytest.raises(FrozenInstanceError):
        local.client.base_url = "http://other:11434"
    with pytest.raises(FrozenInstanceError):
        local.approval.digest = "f"*64


def test_bypassed_approval_construction_is_rechecked(local):
    bad = object.__new__(local.module.GeneratorApproval)
    object.__setattr__(bad, "model", local.module.MODEL)
    object.__setattr__(bad, "digest", PIN)
    object.__setattr__(bad, "approval_reference", "")
    rejected(local, lambda: local.module.OllamaClient("http://localhost:11434", approval=bad),
             "OLLAMA_MODEL_NOT_APPROVED")


def test_generation_uses_exact_structured_request_and_fresh_identity_checks(local):
    replies = local.generation()
    system, user = "  Synthetic system.\n", "\tDonnées synthétiques : نص اصطناعي.\r\n"
    assert local.client.generate(system, user) == local.text
    assert [c["path"] for c in local.state.calls] == [
        "/api/version", "/api/tags", "/api/show", "/api/chat", "/api/tags", "/api/ps",
    ]
    chat = local.state.calls[3]
    body = json.loads(chat["body"])
    assert body == {
        "model": "qwen3:4b-instruct",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": local.module.GenerationOutput.model_json_schema(),
        "options": {"num_ctx": 4096, "num_predict": 512, "temperature": 0},
        "stream": False, "keep_alive": "5m", "think": False,
    }
    assert list(body["format"]["properties"]) == ["status", "answer", "citations"]
    assert all(c.timeout == 90 and c.debug == 0 and c.closed for c in local.state.connections)
    assert all(t.interval == 90 and t.started and t.cancelled for t in local.state.timers)
    assert all(r.closed and r.reads == [local.module.MAX_RESPONSE_BYTES+1] for r in replies)
    # The synthetic ID is not a permission or provenance receipt. Client success
    # does not replace the citation validator or controlled response guard.


def test_think_false_is_omitted_only_when_metadata_reports_no_thinking_capability(local):
    show = copy.deepcopy(local.show)
    show["capabilities"] = ["completion"]
    local.generation(show_data=show)
    assert local.client.generate("Synthetic system", "Synthetic evidence") == local.text
    assert "think" not in json.loads(local.state.calls[3]["body"])


@pytest.mark.parametrize("prompt", [None, 1, b"text", "", " \n", "\ud800"])
def test_invalid_prompt_inputs_do_not_reach_http(local, prompt):
    rejected(local, lambda: local.client.generate("Synthetic system", prompt), "OLLAMA_REQUEST_INVALID")
    assert local.state.calls == []


@pytest.mark.parametrize("prompt", ["x"*20000, "ع"*2000, "\n"+"x"*3000])
def test_oversized_evidence_is_rejected_not_truncated(local, prompt):
    rejected(local, lambda: local.client.generate("Synthetic system", prompt), "OLLAMA_REQUEST_TOO_LARGE")
    assert local.state.calls == []


@pytest.mark.parametrize("raw", [
    "not JSON", "{}", '{"status":"answer","answer":null,"citations":[]}',
    '{"status":"answer","answer":"text","citations":[]}',
    '{"status":"refusal","answer":"explanation","citations":[]}',
    '{"status":"refusal","answer":null,"citations":[],"explanation":"secret"}',
    '{"status":"refusal","status":"refusal","answer":null,"citations":[]}',
    '{"status":"answer","answer":NaN,"citations":[]}',
])
def test_invalid_generation_json_or_schema_is_rejected(local, raw):
    response = copy.deepcopy(local.response)
    response["message"]["content"] = raw
    local.generation(result=response)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_GENERATION_INVALID")
    assert "/api/ps" not in [c["path"] for c in local.state.calls]


def test_valid_model_refusal_remains_raw_untrusted_json(local):
    raw = '{ "status": "refusal", "answer": null, "citations": [] }\n'
    result = copy.deepcopy(local.response)
    result["message"]["content"] = raw
    local.generation(result=result)
    assert local.client.generate("Synthetic system", "No supporting fact.") == raw


@pytest.mark.parametrize("defect", ["model", "done", "reason", "role", "prompt_count", "output_count", "thinking", "tools"])
def test_invalid_response_envelopes_fail_closed(local, defect):
    result = copy.deepcopy(local.response)
    if defect == "model":
        result["model"] = "qwen3.5:4b"
    elif defect == "done":
        result["done"] = 1
    elif defect == "reason":
        del result["done_reason"]
    elif defect == "role":
        result["message"]["role"] = "tool"
    elif defect == "prompt_count":
        result["prompt_eval_count"] = 4000
    elif defect == "output_count":
        result["eval_count"] = 513
    elif defect == "thinking":
        result["message"]["thinking"] = "SYNTHETIC_REASONING"
    else:
        result["message"]["tool_calls"] = [{"function": "not_authorized"}]
    local.generation(result=result)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"),
             "OLLAMA_MODEL_MISMATCH" if defect == "model" else "OLLAMA_PROTOCOL_ERROR")


def test_output_token_limit_is_rejected_even_when_json_happens_to_be_valid(local):
    result = copy.deepcopy(local.response)
    result["done_reason"] = "length"
    local.generation(result=result)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_OUTPUT_LIMIT")


def test_cached_prompt_count_zero_is_not_mistaken_for_invalid_content(local):
    result = copy.deepcopy(local.response)
    result["prompt_eval_count"] = 0
    local.generation(result=result)
    assert local.client.generate("Synthetic system", "Synthetic evidence") == local.text


@pytest.mark.parametrize("field,value", [("prompt_eval_count", -1), ("prompt_eval_count", True), ("eval_count", 0), ("eval_count", False)])
def test_invalid_or_coerced_token_counts_are_rejected(local, field, value):
    result = copy.deepcopy(local.response)
    result[field] = value
    local.generation(result=result)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_PROTOCOL_ERROR")


def test_oversized_generated_content_is_rejected(local):
    result = copy.deepcopy(local.response)
    result["message"]["content"] = "x"*(local.module.MAX_GENERATED_BYTES+1)
    local.generation(result=result)
    rejected(local, lambda: local.client.generate("Synthetic system", "Synthetic evidence"), "OLLAMA_RESPONSE_TOO_LARGE")


@pytest.mark.parametrize("headers", [
    {"Content-Length": "-1"}, {"Content-Length": "1.2"}, {"Content-Length": " 10"},
    {"Content-Length": "0"}, {"Content-Type": "text/html"},
])
def test_invalid_http_framing_is_rejected(local, headers):
    local.state.replies = [Reply({"version": "0.40.17"}, headers=headers)]
    rejected(local, local.client.check_readiness, "OLLAMA_PROTOCOL_ERROR")


def test_large_declared_response_is_rejected_without_reading_body(local):
    reply = Reply({}, headers={"Content-Length": str(local.module.MAX_RESPONSE_BYTES+1)})
    local.state.replies = [reply]
    rejected(local, local.client.check_readiness, "OLLAMA_RESPONSE_TOO_LARGE")
    assert reply.reads == [] and reply.closed


def test_unknown_length_response_is_bounded_by_actual_bytes(local):
    reply = Reply(raw=b"x"*(local.module.MAX_RESPONSE_BYTES+10), headers={"Content-Length": None})
    local.state.replies = [reply]
    rejected(local, local.client.check_readiness, "OLLAMA_RESPONSE_TOO_LARGE")
    assert reply.reads == [local.module.MAX_RESPONSE_BYTES+1]


@pytest.mark.parametrize("raw", [b"\xff", b"null", b"[]", b'{"version":"a","version":"b"}', b'{"version":NaN}'])
def test_invalid_metadata_json_is_rejected(local, raw):
    local.state.replies = [Reply(raw=raw)]
    rejected(local, local.client.check_readiness, "OLLAMA_PROTOCOL_ERROR")


@pytest.mark.parametrize("error,code", [
    (TimeoutError("SYNTHETIC_PRIVATE_DETAIL"), "OLLAMA_TIMEOUT"),
    (ConnectionRefusedError("SYNTHETIC_PRIVATE_DETAIL"), "OLLAMA_DAEMON_UNAVAILABLE"),
    (RuntimeError("SYNTHETIC_PRIVATE_DETAIL"), "OLLAMA_PROTOCOL_ERROR"),
])
def test_transport_failures_are_content_free_and_not_retried(local, error, code):
    local.state.request_error = error
    caught = rejected(local, local.client.check_readiness, code)
    assert len(local.state.calls) == 1
    assert local.state.connections[0].closed and local.state.timers[0].cancelled
    formatted = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert "SYNTHETIC_PRIVATE_DETAIL" not in formatted


def test_watchdog_shutdown_enforces_deadline_without_waiting(local):
    local.state.expire = True
    rejected(local, local.client.check_readiness, "OLLAMA_TIMEOUT")
    assert local.state.calls == []
    assert local.state.connections[0].sock.shutdowns
    assert local.state.connections[0].closed


def test_all_subrequests_share_the_same_90_second_operation_budget(local):
    replies = local.generation()
    def spend():
        local.state.now += 10
    for reply in replies:
        reply.hook = spend
    assert local.client.generate("Synthetic system", "Synthetic evidence") == local.text
    assert [c.timeout for c in local.state.connections] == [90, 80, 70, 60, 50, 40]


def test_elapsed_deadline_is_not_reset_or_accepted(local):
    replies = local.ready()
    replies[0].hook = lambda: setattr(local.state, "now", 91)
    local.state.replies = replies
    rejected(local, local.client.check_readiness, "OLLAMA_TIMEOUT")
    assert len(local.state.calls) == 1


def test_no_proxy_environment_or_redirect_fallback(local, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://cloud.example:8080")
    monkeypatch.setenv("OLLAMA_HOST", "https://cloud.example")
    local.state.replies = [Reply({}, status=302, headers={"Location": "https://cloud.example"})]
    rejected(local, local.client.check_readiness, "OLLAMA_HTTP_ERROR")
    assert len(local.state.calls) == 1 and local.state.calls[0]["host"] == "127.0.0.1"
    assert local.state.calls[0]["headers"] == {"Accept": "application/json", "Content-Type": "application/json"}


def test_sensitive_prompts_answers_approval_refs_and_errors_are_never_logged(local, caplog, capsys):
    sentinel = "SYNTHETIC_SENSITIVE_PAYLOAD_DO_NOT_LOG"
    approval = local.module.GeneratorApproval(local.module.MODEL, PIN, sentinel)
    client = local.module.OllamaClient("http://localhost:11434", approval=approval)
    result = copy.deepcopy(local.response)
    result["message"]["content"] = sentinel
    local.generation(result=result)
    caught = rejected(local, lambda: client.generate(sentinel, sentinel), "OLLAMA_GENERATION_INVALID")
    assert sentinel not in repr(approval) + repr(client)
    assert sentinel not in "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_repeated_failures_do_not_cache_readiness_or_change_inputs(local):
    for _ in range(3):
        local.state.replies = local.ready(tag_data={"models": []})
        rejected(local, local.client.check_readiness, "OLLAMA_MODEL_MISSING")
    assert [c["path"] for c in local.state.calls] == ["/api/version", "/api/tags"]*3

