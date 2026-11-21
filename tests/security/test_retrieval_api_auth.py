"""Real signed JWTs must pass before query parsing, model loading or storage."""

from secrets import token_urlsafe

import pytest

from integration.test_retrieval_api import request_api
from .test_auth import secured


@pytest.mark.parametrize("defect,status", [
    ("missing",401),("expired",401),("forged",401),("issuer",401),("audience",401),
    ("malformed",401),("duplicate-auth",401),("missing-role",403),("ungranted",403),
    ("unknown",403),("duplicate-role",403),("admin",403),
])
def test_auth_failure_precedes_all_retrieval(secured,monkeypatch,caplog,defect,status):
    from askpolicy.api.app import create_app
    from askpolicy.api.retrieval_runtime import RetrievalRuntime
    from askpolicy.storage import publication

    def forbidden(*args,**kwargs):
        pytest.fail("rejected authentication reached retrieval")
    monkeypatch.setattr(RetrievalRuntime,"search",forbidden)
    monkeypatch.setattr(RetrievalRuntime,"_load_encoder",forbidden)
    monkeypatch.setattr(publication,"load_active_snapshot",forbidden)
    app=create_app(jwt_config=secured.config)
    token,role,headers=secured.sign(),"employee",[]
    if defect=="missing":
        token=None
    elif defect=="expired":
        token=secured.sign(exp=secured.now-1,iat=secured.now-100)
    elif defect=="forged":
        token=secured.sign(signing_key=token_urlsafe(48))
    elif defect in ("issuer","audience"):
        token=secured.sign(**{"iss" if defect=="issuer" else "aud":"wrong"})
    elif defect=="malformed":
        token=None
        headers=[("authorization","Basic private")]
    elif defect=="duplicate-auth":
        headers=[("authorization","Bearer "+token)]
    elif defect=="missing-role":
        role=None
    elif defect=="ungranted":
        role="hr"
    elif defect=="unknown":
        role="owner"
    elif defect=="duplicate-role":
        headers=[("x-active-role","employee")]
    else:
        token=secured.sign(roles=[],capabilities=["ingestion_admin"])
    # Invalid JSON further proves auth completes before payload parsing.
    response=request_api(app,token,role,headers=headers,raw=b"{")
    assert response.status==status
    assert response.json()=={"detail":"Authentication required" if status==401 else "Access denied"}
    assert response.headers[b"cache-control"]==b"no-store"
    if status==401:
        assert response.headers[b"www-authenticate"]==b"Bearer"
    assert not any(secret in response.body.decode()+caplog.text for secret in (secured.key,token or "unused-secret"))


def test_unconfigured_authenticated_request_is_503(secured,monkeypatch):
    from askpolicy.api.app import create_app
    for key in ("ASKPOLICY_RELEASE_ROOT","ASKPOLICY_E5_DIR","ASKPOLICY_MANIFEST_PATH"):
        monkeypatch.delenv(key,raising=False)
    response=request_api(create_app(jwt_config=secured.config),secured.sign(),"employee")
    assert response.status==503 and response.json()=={"detail":"Retrieval unavailable"}
