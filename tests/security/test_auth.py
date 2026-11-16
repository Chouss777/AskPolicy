from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
from secrets import token_urlsafe
from types import SimpleNamespace
from typing import NamedTuple

from fastapi import Depends, FastAPI
import jwt
from jwt.utils import base64url_encode
from pydantic import SecretStr
import pytest


ROOT = Path(__file__).resolve().parents[2]
ISSUER = "askpolicy-test-issuer"
AUDIENCE = "askpolicy-test-api"
SUBJECT = "synthetic-employee"


class Response(NamedTuple):
    status: int
    headers: dict[bytes, bytes]
    body: bytes

    def json(self):
        return json.loads(self.body)


def _request(app, headers, *, query=b"", body=b""):
    async def dispatch():
        messages = []
        request_delivered = False

        async def receive():
            nonlocal request_delivered
            if request_delivered:
                return {"type": "http.disconnect"}
            request_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/protected", "raw_path": b"/protected", "root_path": "",
            "query_string": query,
            "headers": [(name.lower().encode("ascii"), value.encode("latin-1")) for name, value in headers],
        }
        await app(scope, receive, send)
        starts = [message for message in messages if message["type"] == "http.response.start"]
        chunks = [message for message in messages if message["type"] == "http.response.body"]
        assert len(starts) == 1 and chunks and not chunks[-1].get("more_body", False)
        start = starts[0]
        return Response(start["status"], dict(start["headers"]), b"".join(chunk["body"] for chunk in chunks))

    # This in-process app and both ASGI callbacks complete without external
    # I/O. Advance the coroutine directly so the harness needs no sockets.
    with closing(dispatch()) as operation:
        try:
            operation.send(None)
        except StopIteration as completed:
            return completed.value
    raise AssertionError("The test ASGI app unexpectedly suspended")


@pytest.fixture
def secured(monkeypatch):
    """Exercise real signing and the reusable dependency on a synthetic route."""
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from askpolicy.auth import context, jwt as tokens
    from askpolicy.security import acl, index_selection

    key = token_urlsafe(48)
    config = tokens.JWTConfig(SecretStr(key), ISSUER, AUDIENCE, "HS256")
    verifier = tokens.JWTVerifier(config)
    now = int(datetime.now(UTC).timestamp())

    def claims(**overrides):
        return {
            "sub": SUBJECT, "iss": ISSUER, "aud": AUDIENCE, "iat": now - 60, "exp": now + 900,
            "roles": ["employee"], **overrides,
        }

    def sign(payload=None, *, algorithm="HS256", signing_key=key, **overrides):
        try:
            return jwt.encode(claims(**overrides) if payload is None else payload, signing_key, algorithm=algorithm)
        except (TypeError, ValueError, jwt.PyJWTError):
            raise AssertionError("Synthetic JWT signing failed") from None

    app = FastAPI()
    dependency = context.make_auth_dependency(verifier)

    @app.post("/protected")
    async def protected(request_context: context.AuthenticatedContext = Depends(dependency)):
        return {
            "subject": request_context.subject,
            "grants": request_context.grants,
            "active_role": request_context.active_role,
            "capabilities": request_context.capabilities,
            "hr_allowed": acl.can_read_document(request_context, classification="RESTRICTED", allowed_roles=("hr",)),
            "index_role": index_selection.select_role_index(request_context).role,
        }

    def request(token, role="employee", *, headers=(), query=b"", body=b""):
        supplied = list(headers)
        if token is not None:
            supplied.append(("authorization", f"Bearer {token}"))
        if role is not None:
            supplied.append(("x-active-role", role))
        return _request(app, supplied, query=query, body=body)

    def authenticate(token, role):
        return context.authenticate_request(verifier, token, active_role=role)

    return SimpleNamespace(
        key=key, now=now, config=config, verifier=verifier, claims=claims, sign=sign, app=app,
        request=request, authenticate=authenticate, tokens=tokens, context=context, acl=acl, index=index_selection,
    )


def _assert_unauthorized(response):
    assert response.status == 401
    assert response.headers[b"www-authenticate"] == b"Bearer"
    assert response.json() == {"detail": "Authentication required"}


@pytest.mark.parametrize("headers", [
    [],
    [("authorization", "Bearer")],
    [("authorization", "Bearer ")],
    [("authorization", "Basic malformed")],
    [("authorization", "Bearer token trailing")],
])
def test_missing_or_malformed_bearer_header(secured, headers):
    _assert_unauthorized(secured.request(None, headers=headers))


def test_duplicate_authorization_headers_denied_even_when_both_are_valid(secured):
    token = secured.sign()
    _assert_unauthorized(secured.request(token, headers=[("Authorization", f"Bearer {token}")]))


@pytest.mark.parametrize("token", ["bad", "bad.encoding.signature", "not.valid!.compact"])
def test_malformed_jwt(secured, token):
    _assert_unauthorized(secured.request(token))


def test_expired_jwt(secured):
    token = secured.sign(iat=secured.now - 600, exp=secured.now - 300)
    _assert_unauthorized(secured.request(token))


def test_forged_signature(secured):
    _assert_unauthorized(secured.request(secured.sign(signing_key=token_urlsafe(48))))


@pytest.mark.parametrize("override", [
    {"iss": "different-issuer"}, {"aud": "different-audience"}, {"aud": [AUDIENCE]},
], ids=["wrong-issuer", "wrong-audience", "audience-array"])
def test_issuer_and_audience_are_single_expected_strings(secured, override):
    _assert_unauthorized(secured.request(secured.sign(**override)))


def test_signed_issuer_array_is_rejected(secured):
    raw = json.dumps(secured.claims(iss=[ISSUER])).encode("utf-8")
    token = jwt.PyJWS().encode(raw, secured.key, algorithm="HS256")
    _assert_unauthorized(secured.request(token))
    with pytest.raises(secured.tokens.AuthenticationError, match="^INVALID_CLAIMS$"):
        secured.verifier.verify(token)


@pytest.mark.parametrize("claim", ["sub", "iss", "aud", "iat", "exp", "roles"])
def test_missing_required_claim(secured, claim):
    payload = secured.claims()
    del payload[claim]
    _assert_unauthorized(secured.request(secured.sign(payload)))


def test_other_signed_algorithm_is_rejected(secured):
    token = secured.sign(algorithm="HS512", signing_key=token_urlsafe(80))
    assert jwt.get_unverified_header(token)["alg"] == "HS512"
    _assert_unauthorized(secured.request(token))


def test_unsigned_jwt_is_rejected(secured):
    token = secured.sign(algorithm="none", signing_key=None)
    assert jwt.get_unverified_header(token)["alg"] == "none"
    _assert_unauthorized(secured.request(token))


@pytest.mark.parametrize("field", ["nbf", "iat"])
def test_future_time_claim_is_rejected(secured, field):
    _assert_unauthorized(secured.request(secured.sign(**{field: secured.now + 300})))


@pytest.mark.parametrize("field", ["iat", "exp", "nbf"])
@pytest.mark.parametrize("value", ["1794000000", 1794000000.0, True, None, [], {}],
                         ids=["numeric-string", "float", "bool", "null", "list", "mapping"])
def test_time_claim_types_are_strict(secured, field, value):
    token = secured.sign(**{field: value})
    _assert_unauthorized(secured.request(token))
    with pytest.raises(secured.tokens.AuthenticationError, match="^INVALID_CLAIMS$"):
        secured.verifier.verify(token)


@pytest.mark.parametrize("claim", ["exp", "nbf"])
def test_inconsistent_time_windows_rejected(secured, claim):
    override = {"exp": secured.now - 60} if claim == "exp" else {"nbf": secured.now + 900}
    _assert_unauthorized(secured.request(secured.sign(**override)))


def test_nbf_is_inclusive_and_exp_is_exclusive_at_utc_boundary(secured, monkeypatch):
    real_datetime = datetime
    boundary = secured.now

    class FixedClock:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return real_datetime.fromtimestamp(boundary, tz=tz)

    monkeypatch.setattr(secured.tokens, "datetime", FixedClock)
    token = secured.sign(iat=boundary, nbf=boundary, exp=boundary + 1)
    assert secured.request(token).status == 200
    boundary += 1
    _assert_unauthorized(secured.request(token))


@pytest.mark.parametrize("override", [
    {"roles": ["owner"]}, {"roles": ["employee", "employee"]}, {"roles": ["ingestion_admin"]},
    {"roles": "employee"}, {"roles": {"employee": True}}, {"roles": [2]}, {"roles": None},
    {"capabilities": ["publisher"]}, {"capabilities": ["employee"]},
    {"capabilities": ["ingestion_admin", "ingestion_admin"]}, {"capabilities": "ingestion_admin"},
    {"capabilities": None}, {"capabilities": [False]},
    {"sub": ""}, {"sub": " "}, {"sub": {"name": SUBJECT}}, {"sub": "synthetic\nsubject"},
], ids=[
    "unknown-role", "duplicate-role", "admin-as-role", "roles-string", "roles-mapping", "role-number", "roles-null",
    "unknown-capability", "role-as-capability", "duplicate-capability", "capabilities-string", "capabilities-null",
    "capability-bool", "empty-sub", "blank-sub", "sub-mapping", "sub-control",
])
def test_invalid_identity_or_grant_claims(secured, override):
    _assert_unauthorized(secured.request(secured.sign(**override)))


def test_signed_payload_must_be_json_object(secured):
    token = jwt.PyJWS().encode(b'["employee"]', secured.key, algorithm="HS256")
    _assert_unauthorized(secured.request(token))


def test_duplicate_claim_names_are_rejected_even_with_a_valid_signature(secured):
    raw = json.dumps(secured.claims()).removesuffix("}") + ', "roles":["employee"]}'
    token = jwt.PyJWS().encode(raw.encode("utf-8"), secured.key, algorithm="HS256")
    _assert_unauthorized(secured.request(token))


def test_duplicate_protected_headers_are_rejected_even_with_a_valid_signature(secured):
    header = base64url_encode(b'{"alg":"none","alg":"HS256"}')
    payload = base64url_encode(json.dumps(secured.claims()).encode("utf-8"))
    message = b".".join((header, payload))
    algorithm = jwt.get_algorithm_by_name("HS256")
    signature = algorithm.sign(message, algorithm.prepare_key(secured.key))
    token = b".".join((message, base64url_encode(signature))).decode("ascii")
    _assert_unauthorized(secured.request(token))


def test_valid_signed_token_returns_verified_context(secured):
    token = secured.sign(roles=["employee", "finance"], nbf=secured.now - 1)
    response = secured.request(token)
    assert response.status == 200
    assert response.json() == {
        "subject": SUBJECT, "grants": ["employee", "finance"], "active_role": "employee",
        "capabilities": [], "hr_allowed": False, "index_role": "employee",
    }
    context = secured.authenticate(token, "finance")
    assert context.grants == ("employee", "finance")
    assert context.active_role == "finance"
    with pytest.raises(AttributeError):
        context.active_role = "hr"


def test_failures_do_not_expose_credentials_claims_or_query(secured, caplog):
    credentials = {
        "subject": "synthetic-sensitive-subject",
        "password": token_urlsafe(36),
        "query": "synthetic confidential question",
        "policy": "synthetic restricted source text",
    }
    token = secured.sign(sub=credentials["subject"], expired_password=credentials["password"],
                         excerpt=credentials["policy"], iat=secured.now - 300, exp=secured.now - 10)
    response = secured.request(token, body=json.dumps({"query": credentials["query"]}).encode("utf-8"))
    _assert_unauthorized(response)
    with pytest.raises(secured.tokens.AuthenticationError) as raised:
        secured.verifier.verify(token)
    assert raised.value.code == str(raised.value) == "TOKEN_EXPIRED"
    assert raised.value.__cause__ is None
    exposed = response.body.decode("utf-8") + str(raised.value) + caplog.text + repr(secured.config)
    assert not any(value in exposed for value in (token, secured.key, *credentials.values()))


@pytest.mark.parametrize("field, replacement", [
    ("signing_secret", SecretStr("")), ("signing_secret", SecretStr("short")),
    ("signing_secret", SecretStr("placeholder")), ("signing_secret", SecretStr("x" * 32)),
    ("signing_secret", SecretStr("replace-me-with-a-longer-random-key")),
    ("signing_secret", None), ("issuer", "placeholder"), ("audience", " "),
    ("algorithm", "HS512"), ("algorithm", "none"),
], ids=[
    "empty-key", "short-key", "placeholder-key", "invalid-key", "long-placeholder-key",
    "missing-key", "placeholder-issuer", "blank-audience", "wrong-algorithm", "none",
])
def test_configuration_rejects_invalid_values(secured, field, replacement):
    settings = {
        "signing_secret": SecretStr(secured.key), "issuer": ISSUER, "audience": AUDIENCE, "algorithm": "HS256",
    }
    settings[field] = replacement
    with pytest.raises(secured.tokens.AuthenticationError) as raised:
        secured.tokens.JWTConfig(**settings)
    assert raised.value.code == str(raised.value) == "INVALID_JWT_CONFIG"


def test_dependency_refuses_missing_verifier_configuration(secured):
    with pytest.raises(secured.tokens.AuthenticationError, match="^INVALID_JWT_CONFIG$"):
        secured.tokens.JWTVerifier(None)
    with pytest.raises(secured.tokens.AuthenticationError, match="^INVALID_JWT_CONFIG$"):
        secured.context.make_auth_dependency(None)
