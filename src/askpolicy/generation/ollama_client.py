"""Bounded transport for the approved local generator; no RAG orchestration.

A GeneratorApproval must come from an explicit owner-approved record supplied by
trusted server configuration. This module cannot authenticate that external
decision: constructing a record is not itself approval. Never derive it from
Ollama metadata, client/model input, a Boolean flag, or an observed digest.

Returned text remains untrusted. Structural validation establishes neither
citation integrity, authorization nor semantic support; the caller must use the
existing citation validator and server-controlled response guard. Do not log
prompts, answers, approval records, error details or traceback locals.
"""

from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPException
import ipaddress
import json
import re
import socket
from threading import Event, Timer
from time import monotonic
from urllib.parse import urlsplit

from .schemas import GenerationOutput


MODEL = "qwen3:4b-instruct"
NUM_CTX = 4096
NUM_PREDICT = 512
TIMEOUT_SECONDS = 90
MAX_REQUEST_BYTES = 16_384
MAX_RESPONSE_BYTES = 262_144
MAX_GENERATED_BYTES = 65_536
TEMPLATE_TOKEN_RESERVE = 1024
_ERRORS = frozenset({
    "OLLAMA_CONFIG_INVALID", "OLLAMA_MODEL_NOT_APPROVED", "OLLAMA_DAEMON_UNAVAILABLE",
    "OLLAMA_MODEL_MISSING", "OLLAMA_DIGEST_MISMATCH", "OLLAMA_PROTOCOL_ERROR",
    "OLLAMA_HTTP_ERROR", "OLLAMA_REQUEST_INVALID", "OLLAMA_REQUEST_TOO_LARGE",
    "OLLAMA_RESPONSE_TOO_LARGE", "OLLAMA_TIMEOUT", "OLLAMA_GENERATION_INVALID",
    "OLLAMA_OUTPUT_LIMIT", "OLLAMA_MODEL_UNSUPPORTED", "OLLAMA_MODEL_MISMATCH",
})


class OllamaClientError(ValueError):
    """A stable content-free code, with no transport/payload exception attached."""

    def __init__(self, code: str) -> None:
        self.code = code if type(code) is str and code in _ERRORS else "OLLAMA_PROTOCOL_ERROR"
        super().__init__(self.code)


def _safe(action):
    code = "OLLAMA_PROTOCOL_ERROR"
    try:
        return action()
    except OllamaClientError as error:
        code = error.code
    except (TimeoutError, socket.timeout):
        code = "OLLAMA_TIMEOUT"
    except (OSError, HTTPException):
        code = "OLLAMA_DAEMON_UNAVAILABLE"
    except Exception:
        pass
    # Raise outside the handler, removing sensitive nested exception frames.
    raise OllamaClientError(code)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _approval(record):
    if (
        type(record) is not GeneratorApproval or type(record.model) is not str
        or record.model != MODEL or not _digest(record.digest)
        or type(record.approval_reference) is not str or len(record.approval_reference) > 1024
        or not record.approval_reference.strip()
    ):
        raise OllamaClientError("OLLAMA_MODEL_NOT_APPROVED")
    return record


@dataclass(frozen=True, slots=True)
class GeneratorApproval:
    """Trusted configuration's explicit model/digest/owner-decision binding.

    approval_reference identifies the real external approval record. It is not
    verified by this transport; the trusted caller must supply genuine approval.
    No observed digest is built in, and no default approval is provided.
    """

    model: str = field(repr=False)
    digest: str = field(repr=False)
    approval_reference: str = field(repr=False)

    def __post_init__(self):
        _safe(lambda: _approval(self))


@dataclass(frozen=True, slots=True)
class OllamaReadiness:
    daemon_version: str
    model: str
    digest: str
    thinking_supported: bool


def _endpoint(value):
    if type(value) is not str or not value or any(c.isspace() for c in value) or "\\" in value:
        raise OllamaClientError("OLLAMA_CONFIG_INVALID")
    try:
        url = urlsplit(value)
        host = url.hostname
        if (
            url.scheme != "http" or url.port != 11434 or url.path not in ("", "/")
            or url.username is not None or url.password is not None
            or "?" in value or "#" in value or host is None
        ):
            raise ValueError
        if host == "localhost":
            # Avoid DNS/proxy rebinding: this local alias always uses IPv4 loopback.
            host = "127.0.0.1"
        elif not ipaddress.ip_address(host).is_loopback:
            raise ValueError
        return host, 11434
    except Exception:
        pass
    raise OllamaClientError("OLLAMA_CONFIG_INVALID")


def _remaining(deadline):
    seconds = deadline - monotonic()
    if seconds <= 0:
        raise OllamaClientError("OLLAMA_TIMEOUT")
    return seconds


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
        result[key] = value
    return result


def _reject_constant(_):
    raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _encode(value):
    # Preserve the actual GenerationOutput schema property order on the wire.
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_REQUEST_BYTES:
        raise OllamaClientError("OLLAMA_REQUEST_TOO_LARGE")
    return raw


@dataclass(frozen=True, slots=True)
class OllamaClient:
    """Explicit loopback endpoint and trusted approval; no I/O during construction.

    Only HTTP loopback at the locked port 11434 is supported. Approval is absent
    by default, leaving readiness and generation closed. There is no readiness
    cache, inference cache, environment lookup or automatic provisioning.
    """

    base_url: str = field(repr=False)
    approval: GeneratorApproval | None = field(default=None, repr=False)

    def __post_init__(self):
        _safe(lambda: _endpoint(self.base_url))
        if self.approval is not None:
            _safe(lambda: _approval(self.approval))

    def _http(self, method, path, body, deadline):
        host, port = _endpoint(self.base_url)
        raw = None if body is None else _encode(body)
        connection = HTTPConnection(host, port, timeout=_remaining(deadline))
        expired = Event()

        def expire():
            expired.set()
            # Socket shutdown interrupts blocked reads, including slow-drip
            # bodies; ordinary close alone may leave a makefile read blocked.
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass

        timer = Timer(_remaining(deadline), expire)
        timer.daemon = True
        response = None
        try:
            connection.set_debuglevel(0)
            timer.start()
            if expired.is_set():
                raise OllamaClientError("OLLAMA_TIMEOUT")
            connection.request(method, path, body=raw, headers={"Accept": "application/json", "Content-Type": "application/json"})
            if connection.sock is not None:
                connection.sock.settimeout(_remaining(deadline))
            response = connection.getresponse()
            if type(response.status) is not int or response.status != 200:
                raise OllamaClientError("OLLAMA_HTTP_ERROR")
            content_type = response.getheader("Content-Type")
            if type(content_type) is not str or content_type.split(";", 1)[0].strip().lower() != "application/json":
                raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
            length = response.getheader("Content-Length")
            if length is not None:
                if type(length) is not str or not re.fullmatch(r"[0-9]+", length):
                    raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
                if int(length) > MAX_RESPONSE_BYTES:
                    raise OllamaClientError("OLLAMA_RESPONSE_TOO_LARGE")
            if connection.sock is not None:
                connection.sock.settimeout(_remaining(deadline))
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if expired.is_set():
                raise OllamaClientError("OLLAMA_TIMEOUT")
            _remaining(deadline)
            if type(data) is not bytes:
                raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
            if len(data) > MAX_RESPONSE_BYTES:
                raise OllamaClientError("OLLAMA_RESPONSE_TOO_LARGE")
            if length is not None and len(data) != int(length):
                raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
            result = _json(data.decode("utf-8", errors="strict"))
            if type(result) is not dict:
                raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
            _remaining(deadline)
            return result
        except Exception:
            if expired.is_set():
                raise OllamaClientError("OLLAMA_TIMEOUT")
            raise
        finally:
            timer.cancel()
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            try:
                connection.close()
            except Exception:
                pass

    def _installed(self, approval, deadline):
        data = self._http("GET", "/api/tags", None, deadline)
        models = data.get("models")
        if type(models) is not list or any(type(m) is not dict for m in models):
            raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
        selected = [m for m in models if m.get("name") == MODEL]
        if not selected:
            raise OllamaClientError("OLLAMA_MODEL_MISSING")
        if len(selected) != 1 or selected[0].get("model") != MODEL:
            raise OllamaClientError("OLLAMA_MODEL_MISMATCH")
        if any(selected[0].get(key, "") != "" for key in ("remote_host", "remote_model")):
            raise OllamaClientError("OLLAMA_MODEL_UNSUPPORTED")
        digest = selected[0].get("digest")
        if not _digest(digest) or digest != approval.digest:
            raise OllamaClientError("OLLAMA_DIGEST_MISMATCH")

    def _readiness(self, approval, deadline):
        data = self._http("GET", "/api/version", None, deadline)
        version = data.get("version")
        if type(version) is not str or not version.strip() or len(version) > 128:
            raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
        self._installed(approval, deadline)
        show = self._http("POST", "/api/show", {"model": MODEL}, deadline)
        caps, info, details = show.get("capabilities"), show.get("model_info"), show.get("details")
        if (
            type(caps) is not list or any(type(c) is not str for c in caps) or "completion" not in caps
            or type(details) is not dict or details.get("format") != "gguf" or details.get("family") != "qwen3"
            or any(show.get(key, "") != "" for key in ("remote_host", "remote_model"))
            or type(info) is not dict or info.get("general.architecture") != "qwen3"
            or info.get("tokenizer.ggml.model") != "gpt2"
            or type(info.get("qwen3.context_length")) is not int or info["qwen3.context_length"] < NUM_CTX
            or show.get("system", "") != ""
        ):
            raise OllamaClientError("OLLAMA_MODEL_UNSUPPORTED")
        supported = "thinking" in caps
        thinking = show.get("thinking")
        if supported and thinking is not None:
            if type(thinking) is not dict or type(thinking.get("values")) is not list or not any(v is False for v in thinking["values"]):
                raise OllamaClientError("OLLAMA_MODEL_UNSUPPORTED")
        return OllamaReadiness(version, MODEL, approval.digest, supported)

    def check_readiness(self) -> OllamaReadiness:
        """Check daemon, exact approved installed digest and supported profile.

        This never loads a model or creates an approval record.
        """
        return _safe(lambda: self._readiness(_approval(self.approval), monotonic() + TIMEOUT_SECONDS))

    def _generate(self, system_prompt, user_prompt):
        approval = _approval(self.approval)
        deadline = monotonic() + TIMEOUT_SECONDS
        for value in (system_prompt, user_prompt):
            if type(value) is not str:
                raise OllamaClientError("OLLAMA_REQUEST_INVALID")
            if len(value) > MAX_REQUEST_BYTES:
                raise OllamaClientError("OLLAMA_REQUEST_TOO_LARGE")
            if not value.strip():
                raise OllamaClientError("OLLAMA_REQUEST_INVALID")
        try:
            prompt_bytes = len(system_prompt.encode("utf-8")) + len(user_prompt.encode("utf-8"))
        except UnicodeError:
            raise OllamaClientError("OLLAMA_REQUEST_INVALID")
        schema = GenerationOutput.model_json_schema()
        # Supported GPT2 byte-level BPE cannot need more content tokens than UTF-8
        # bytes. Include the whole schema plus a conservative 1024-token reserve
        # for the approved Qwen two-message template/special tokens. No chopping,
        # normalization, summarization or evidence removal is performed.
        if prompt_bytes + len(_encode(schema)) + TEMPLATE_TOKEN_RESERVE + NUM_PREDICT > NUM_CTX:
            raise OllamaClientError("OLLAMA_REQUEST_TOO_LARGE")
        body = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "format": schema,
            "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": 0},
            "stream": False, "keep_alive": "5m",
        }
        _encode(body)
        ready = self._readiness(approval, deadline)
        if ready.thinking_supported:
            body["think"] = False
        data = self._http("POST", "/api/chat", body, deadline)
        if data.get("model") != MODEL:
            raise OllamaClientError("OLLAMA_MODEL_MISMATCH")
        if data.get("done_reason") == "length":
            raise OllamaClientError("OLLAMA_OUTPUT_LIMIT")
        message = data.get("message")
        if (
            data.get("done") is not True or data.get("done_reason") != "stop"
            or type(message) is not dict or message.get("role") != "assistant"
            or type(data.get("prompt_eval_count")) is not int or not 0 <= data["prompt_eval_count"] <= NUM_CTX - NUM_PREDICT
            or type(data.get("eval_count")) is not int or not 0 < data["eval_count"] <= NUM_PREDICT
            or message.get("thinking", "") != "" or message.get("tool_calls") or message.get("images")
        ):
            raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
        raw = message.get("content")
        if type(raw) is not str or not raw.strip():
            raise OllamaClientError("OLLAMA_GENERATION_INVALID")
        if len(raw.encode("utf-8")) > MAX_GENERATED_BYTES:
            raise OllamaClientError("OLLAMA_RESPONSE_TOO_LARGE")
        try:
            _json(raw)
            GenerationOutput.model_validate_json(raw)
        except Exception:
            raise OllamaClientError("OLLAMA_GENERATION_INVALID")
        # A mutable tag must not silently substitute another installed/loaded
        # model. Verify both reported identities again before releasing text.
        self._installed(approval, deadline)
        models = self._http("GET", "/api/ps", None, deadline).get("models")
        if type(models) is not list or any(type(m) is not dict for m in models):
            raise OllamaClientError("OLLAMA_PROTOCOL_ERROR")
        loaded = [m for m in models if m.get("name") == MODEL]
        if (
            len(loaded) != 1 or loaded[0].get("model") != MODEL
            or not _digest(loaded[0].get("digest")) or loaded[0]["digest"] != approval.digest
            or type(loaded[0].get("context_length")) is not int or loaded[0]["context_length"] != NUM_CTX
        ):
            raise OllamaClientError("OLLAMA_DIGEST_MISMATCH")
        _remaining(deadline)
        return raw

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return structurally checked but still untrusted JSON, unchanged.

        The 90-second deadline covers readiness, inference and identity rechecks.
        Byte limits and a conservative profile-specific context bound reject
        oversized inputs before inference. Never use this text as an authorized
        answer without the existing server response guard and citation checks.
        """
        return _safe(lambda: self._generate(system_prompt, user_prompt))

