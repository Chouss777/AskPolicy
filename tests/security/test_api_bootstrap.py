import json
import os
from pathlib import Path
from secrets import token_urlsafe
import socket
import sqlite3
import subprocess
import sys

from fastapi import FastAPI
import pytest
import uvicorn

from .test_auth import _assert_unauthorized, _request, AUDIENCE, ISSUER, ROOT, secured


JWT_NAMES = ("JWT_SECRET", "JWT_ISSUER", "JWT_AUDIENCE")


@pytest.fixture
def api(secured):
    from askpolicy.api import app

    return app


@pytest.fixture
def runtime_environment(monkeypatch, secured):
    for name, value in zip(JWT_NAMES, (secured.key, ISSUER, AUDIENCE), strict=True):
        monkeypatch.setenv(name, value)


@pytest.fixture
def app(api, runtime_environment):
    return api.create_app()


def _http(app, token=None, role=None, *, headers=(), path="/auth/verify", method="GET", query=b"", body=b""):
    supplied = list(headers)
    if token is not None:
        supplied.append(("authorization", f"Bearer {token}"))
    if role is not None:
        supplied.append(("x-active-role", role))

    async def diagnostic_app(scope, receive, send):
        await app(
            {**scope, "method": method, "path": path, "raw_path": path.encode("ascii")},
            receive,
            send,
        )

    return _request(diagnostic_app, supplied, query=query, body=body)


def _assert_forbidden(response):
    assert response.status == 403
    assert response.json() == {"detail": "Access denied"}


def _assert_verified(response, role):
    assert response.status == 200
    assert response.json() == {"authenticated": True, "active_role": role}
    assert response.headers[b"cache-control"] == b"no-store"


def test_factory_constructs_app_and_registers_real_dependency(api, secured, monkeypatch):
    for name in JWT_NAMES:
        monkeypatch.delenv(name, raising=False)
    registered = []
    verifiers = []
    construct_verifier = api.JWTVerifier
    def observe_verifier(config):
        verifier = construct_verifier(config)
        verifiers.append(verifier)
        return verifier
    create_dependency = secured.context.make_auth_dependency

    def observe_dependency(verifier):
        assert type(verifier) is secured.tokens.JWTVerifier
        assert verifier.config == secured.config
        dependency = create_dependency(verifier)
        registered.append(dependency)
        return dependency

    monkeypatch.setattr(api, "make_auth_dependency", observe_dependency)
    monkeypatch.setattr(api, "JWTVerifier", observe_verifier)
    app = api.create_app(jwt_config=secured.config)
    assert isinstance(app, FastAPI)
    assert len(app.routes) == 2
    routes = {(route.path, frozenset(route.methods)):route for route in app.routes}
    assert set(routes) == {("/auth/verify",frozenset({"GET"})),
                          ("/retrieval/search",frozenset({"POST"}))}
    dependency, = registered
    verifier, = verifiers
    assert verifier.config == secured.config
    auth_route = routes[("/auth/verify",frozenset({"GET"}))]
    retrieval_route = routes[("/retrieval/search",frozenset({"POST"}))]
    assert [item.call for item in auth_route.dependant.dependencies] == [dependency]
    assert retrieval_route.dependant.dependencies[0].call is dependency
    query_dependency = retrieval_route.dependant.dependencies[1]
    assert [item.call for item in query_dependency.dependencies] == [dependency]
    _assert_verified(_http(app, secured.sign(), "employee"), "employee")
    from integration.test_retrieval_api import request_api
    unavailable = request_api(app, secured.sign(), "employee")
    assert unavailable.status == 503
    assert unavailable.json() == {"detail":"Retrieval unavailable"}
    assert unavailable.headers[b"cache-control"] == b"no-store"
    _assert_unauthorized(request_api(app, None, "employee"))
    _assert_forbidden(request_api(app, secured.sign(), "hr"))


@pytest.mark.parametrize("missing", JWT_NAMES)
def test_missing_runtime_configuration_prevents_any_app_construction(
    api, secured, runtime_environment, monkeypatch, missing,
):
    monkeypatch.delenv(missing)

    def unexpected_app(*args, **kwargs):
        pytest.fail("FastAPI was constructed before JWT config validation")

    monkeypatch.setattr(api, "FastAPI", unexpected_app)
    with pytest.raises(secured.tokens.AuthenticationError, match="^INVALID_JWT_CONFIG$"):
        api.create_app()


@pytest.mark.parametrize("key", ["", "short", "placeholder"])
def test_invalid_runtime_secret_prevents_startup(api, secured, runtime_environment, monkeypatch, key):
    monkeypatch.setenv("JWT_SECRET", key)
    with pytest.raises(secured.tokens.AuthenticationError, match="^INVALID_JWT_CONFIG$"):
        api.create_app()


@pytest.mark.parametrize("wrong_config", [{}, "trusted-only", 123])
def test_invalid_injected_type_never_falls_back_to_environment(api, secured, runtime_environment, wrong_config):
    with pytest.raises(secured.tokens.AuthenticationError, match="^INVALID_JWT_CONFIG$"):
        api.create_app(jwt_config=wrong_config)


@pytest.mark.parametrize("role", ["employee", "hr"])
def test_production_endpoint_returns_only_verified_diagnostic_fields(app, secured, role):
    token = secured.sign(roles=[role], capabilities=["ingestion_admin"])
    _assert_verified(_http(app, token, role), role)


@pytest.mark.parametrize("headers", [
    (), (("authorization", "Basic unexpected"),), (("authorization", "Bearer "),),
    (("authorization", "Bearer bad.jwt"),),
], ids=["no-authorization", "wrong-scheme", "empty-bearer", "malformed-jwt"])
def test_missing_or_malformed_authentication_returns_401(app, headers):
    _assert_unauthorized(_http(app, role="employee", headers=headers))


def test_expired_token_returns_401(app, secured):
    token = secured.sign(iat=secured.now - 600, exp=secured.now - 300)
    _assert_unauthorized(_http(app, token, "employee"))


def test_forged_signature_returns_401(app, secured):
    _assert_unauthorized(_http(app, secured.sign(signing_key=token_urlsafe(48)), "employee"))


@pytest.mark.parametrize("override", [{"iss": "other-issuer"}, {"aud": "other-api"}], ids=["issuer", "audience"])
def test_wrong_issuer_or_audience_returns_401(app, secured, override):
    _assert_unauthorized(_http(app, secured.sign(**override), "employee"))


@pytest.mark.parametrize("missing", ["sub", "iss", "aud", "iat", "exp", "roles"])
def test_missing_required_claim_returns_401(app, secured, missing):
    claims = secured.claims()
    del claims[missing]
    _assert_unauthorized(_http(app, secured.sign(claims), "employee"))


def test_disallowed_signature_algorithm_returns_401(app, secured):
    token = secured.sign(algorithm="HS512", signing_key=token_urlsafe(80))
    _assert_unauthorized(_http(app, token, "employee"))


@pytest.mark.parametrize("role", [None, "unknown", "hr"], ids=["missing", "unknown", "ungranted"])
def test_invalid_active_role_returns_403(app, secured, role):
    _assert_forbidden(_http(app, secured.sign(), role))


@pytest.mark.parametrize("second", ["employee", "hr"])
def test_duplicate_active_role_headers_return_403(app, secured, second):
    token = secured.sign(roles=["employee", "hr"])
    _assert_forbidden(_http(app, token, "employee", headers=[("X-Active-Role", second)]))


def test_each_request_reverifies_one_active_role_with_no_state_leak(app, secured):
    token = secured.sign(roles=["employee", "hr"])
    _assert_verified(_http(app, token, "hr"), "hr")
    _assert_verified(_http(app, token, "employee"), "employee")
    _assert_forbidden(_http(app, token))
    _assert_forbidden(_http(app, secured.sign(), "hr"))
    _assert_unauthorized(_http(app, role="hr"))
    _assert_verified(_http(app, token, "hr"), "hr")


def test_admin_only_identity_has_no_active_read_role(app, secured):
    token = secured.sign(roles=[], capabilities=["ingestion_admin"])
    _assert_forbidden(_http(app, token, "employee"))
    _assert_forbidden(_http(app, token, "ingestion_admin"))


def test_query_and_body_role_claims_are_ignored(app, secured):
    token = secured.sign(roles=["employee", "hr"])
    query = b"active_role=hr&role=hr"
    body = b'{"active_role":"hr","role":"hr"}'
    headers = (("content-type", "application/json"),)
    _assert_verified(_http(app, token, "employee", query=query, body=body, headers=headers), "employee")
    _assert_forbidden(_http(app, token, query=query, body=body, headers=headers))


def test_responses_and_logs_omit_credentials_and_private_claims(app, secured, caplog, capsys):
    subject = "synthetic-private-identity"
    note = "synthetic confidential payload"
    token = secured.sign(sub=subject, note=note)
    forged = secured.sign(sub=subject, note=note, signing_key=token_urlsafe(48))
    accepted = _http(app, token, "employee")
    forbidden = _http(app, token, "hr")
    rejected = _http(app, forged, "employee")
    _assert_verified(accepted, "employee")
    _assert_forbidden(forbidden)
    _assert_unauthorized(rejected)
    captured = capsys.readouterr()
    bodies = b"".join(response.body for response in (accepted, forbidden, rejected))
    emitted = bodies.decode("utf-8") + caplog.text + captured.out + captured.err
    assert not any(value in emitted for value in (token, forged, secured.key, subject, note))


def test_bootstrap_and_request_do_not_read_corpus_databases_or_indexes(api, secured, runtime_environment, monkeypatch):
    def unexpected_io(*args, **kwargs):
        pytest.fail("Security bootstrap attempted filesystem, database, or network I/O")

    token = secured.sign()
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", unexpected_io)
        patch.setattr(os, "open", unexpected_io)
        patch.setattr(Path, "open", unexpected_io)
        patch.setattr(Path, "iterdir", unexpected_io)
        patch.setattr(sqlite3, "connect", unexpected_io)
        patch.setattr(socket, "create_connection", unexpected_io)
        response = _http(api.create_app(), token, "employee")
    _assert_verified(response, "employee")


def test_factory_module_can_be_imported_cold_without_credentials_or_services():
    script = """
import sys

sys.path.insert(0, sys.argv[1])
from askpolicy.api.app import create_app
from askpolicy.auth.jwt import AuthenticationError

assert callable(create_app)
assert not any(name in sys.modules for name in (
    "bm25s", "faiss", "askpolicy.storage", "askpolicy.retrieval", "askpolicy.generation",
))
try:
    create_app()
except AuthenticationError as error:
    assert error.code == "INVALID_JWT_CONFIG"
else:
    raise AssertionError("The factory accepted missing JWT configuration")
"""
    environment = {name: value for name, value in os.environ.items() if name.upper() not in JWT_NAMES}
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(ROOT / "src")],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_documented_uvicorn_factory_command_loads_asgi_app_without_listening(
    secured, runtime_environment, monkeypatch,
):
    loaded = []

    def load_only(server):
        assert server.config.factory is True
        assert server.config.access_log is False
        server.config.load()
        loaded.append(server.config.loaded_app)
        server.started = True

    def unexpected_listen(*args, **kwargs):
        pytest.fail("The Uvicorn entry point test attempted to bind a socket")

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(uvicorn.Config, "configure_logging", lambda self: None)
    monkeypatch.setattr(uvicorn.Server, "run", load_only)
    monkeypatch.setattr(socket.socket, "bind", unexpected_listen)
    uvicorn.main.main(
        args=["askpolicy.api.app:create_app", "--factory", "--app-dir", "src", "--no-access-log"],
        standalone_mode=False,
    )
    server_app, = loaded
    _assert_unauthorized(_http(server_app, role="employee"))
    _assert_verified(_http(server_app, secured.sign(), "employee"), "employee")


@pytest.mark.parametrize("path", ["/", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json", "/auth/verify/"])
def test_no_alternate_unprotected_diagnostic_or_documentation_route(app, path):
    response = _http(app, path=path)
    assert response.status == 404
    assert response.json() == {"detail": "Not Found"}


def test_diagnostic_has_no_unprotected_post_route(app):
    response = _http(app, method="POST", body=json.dumps({"active_role": "hr"}).encode("utf-8"))
    assert response.status == 405
