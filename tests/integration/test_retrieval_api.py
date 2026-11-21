"""HTTP contracts over real published SQLite/BM25S/FAISS with fixture embeddings."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from integration.test_embedding_revision import dense_env, DAY
from security.test_auth import Response, ROOT, secured


def request_api(app, token=None, role=None, *, payload=None, raw=None, headers=(), query=b"",
                path="/retrieval/search", method="POST"):
    supplied = [(key.lower().encode(), value.encode("latin1")) for key,value in headers]
    supplied.append((b"content-type",b"application/json"))
    if token is not None:
        supplied.append((b"authorization",("Bearer "+token).encode()))
    if role is not None:
        supplied.append((b"x-active-role",role.encode()))
    body = json.dumps({"query":"travel hébergement déplacement"} if payload is None else payload).encode() if raw is None else raw

    async def dispatch():
        messages = []
        delivered = False
        async def receive():
            nonlocal delivered
            if delivered:
                return {"type":"http.disconnect"}
            delivered = True
            return {"type":"http.request","body":body,"more_body":False}
        async def send(message):
            messages.append(message)
        await app({"type":"http","asgi":{"version":"3.0","spec_version":"2.4"},"http_version":"1.1",
                   "method":method,"scheme":"http","path":path,"raw_path":path.encode(),
                   "root_path":"","query_string":query,"headers":supplied},receive,send)
        start, = [m for m in messages if m["type"]=="http.response.start"]
        chunks = [m["body"] for m in messages if m["type"]=="http.response.body"]
        return Response(start["status"],dict(start["headers"]),b"".join(chunks))
    return asyncio.run(dispatch())


@pytest.fixture(scope="module")
def api_release(dense_env, tmp_path_factory):
    from askpolicy.storage.publication import publish_snapshot
    root = tmp_path_factory.mktemp("api-release")
    opts = dense_env.options | {"encoder":dense_env.encoder}
    handle = publish_snapshot(dense_env.db,root,**opts)
    return root,handle,opts


@pytest.fixture
def api_env(api_release, dense_env, secured, monkeypatch):
    from askpolicy.api.app import create_app
    from askpolicy.api.retrieval_runtime import RetrievalConfig
    from askpolicy.retrieval import e5_encoder

    root, handle, opts = api_release
    loads = []
    clocks = []
    token_calls = []
    def tokenize(text, **kw):
        token_calls.append((text,kw))
        return {"input_ids":[1]*(len(text.split())+2)}
    monkeypatch.setattr(dense_env.encoder,"_tokenizer",tokenize,raising=False)
    def load(path):
        loads.append(path)
        return dense_env.encoder
    monkeypatch.setattr(e5_encoder,"E5Encoder",load)
    config = RetrievalConfig(release_root=root,e5_dir=root/"model",manifest_path=ROOT/"data/corpus_manifest.yaml")
    def clock():
        clocks.append(DAY)
        return DAY
    app = create_app(jwt_config=secured.config,retrieval_config=config,clock=clock)
    return SimpleNamespace(app=app,config=config,clock=clock,loads=loads,clocks=clocks,tokens=token_calls,
                           root=root,handle=handle,options=opts,encoder=dense_env.encoder)


def test_response_contract_identity_order_and_reused_encoder(api_env, secured):
    from askpolicy.api.retrieval import SearchResponse

    env = api_env
    token = secured.sign()
    first = request_api(env.app,token,"employee")
    second = request_api(env.app,token,"employee")
    assert first.status == second.status == 200 and first.json()==second.json()
    result = SearchResponse.model_validate_json(first.body)
    assert result.release_id == env.handle.release_id
    assert result.snapshot_id == env.handle.faiss.snapshot.candidate_snapshot_id
    assert result.active_role == "employee" and len(result.results)==20
    assert len({h.child_id for h in result.results})==20
    assert result.results == tuple(sorted(result.results,key=lambda h:(-h.rrf_score,h.child_id)))
    assert set(first.json())=={"release_id","snapshot_id","active_role","results"}
    assert all(set(h)=={"child_id","parent_id","document_id","rrf_score","bm25_rank","faiss_rank"}
               for h in first.json()["results"])
    assert first.headers[b"cache-control"]==b"no-store"
    assert len(env.loads)==1 and env.clocks==[DAY,DAY]
    assert all(kw==dict(add_special_tokens=True,padding=False,truncation=False,verbose=False)
               for _,kw in env.tokens)


@pytest.mark.parametrize("payload", [{},{"query":""},{"query":"  "},{"query":1},{"query":None},
                                    {"query":["hello"]},{"query":"a"*4097},["query"],"query"])
def test_invalid_query_rejected_before_clock_or_model(api_env,secured,payload):
    env=api_env
    result=request_api(env.app,secured.sign(),"employee",payload=payload)
    assert result.status==422 and result.json()=={"detail":"Invalid query"}
    assert result.headers[b"cache-control"]==b"no-store"
    assert not env.loads and not env.clocks


@pytest.mark.parametrize("field", ["role","active_role","validity_date","snapshot_id","release_id","document_ids","index_name"])
def test_request_cannot_override_scope(api_env,secured,field):
    result=request_api(api_env.app,secured.sign(),"employee",payload={"query":"travel",field:"hr"})
    assert result.status==422 and not api_env.loads and not api_env.clocks


@pytest.mark.parametrize("raw", [b"{",b"",b"x"*32769])
def test_malformed_or_oversized_body_is_generic_422(api_env,secured,raw):
    response=request_api(api_env.app,secured.sign(),"employee",raw=raw)
    assert response.status==422 and response.json()=={"detail":"Invalid query"}
    assert not api_env.loads


def test_token_limit_has_no_truncation_or_search(api_env,secured,monkeypatch):
    from askpolicy.storage import publication
    def forbidden(*args,**kwargs):
        pytest.fail("overlength query reached publication")
    monkeypatch.setattr(publication,"load_active_snapshot",forbidden)
    response=request_api(api_env.app,secured.sign(),"employee",payload={"query":"word "*510})
    assert response.status==422 and response.json()=={"detail":"Invalid query"}


def test_legitimate_empty_response(api_env,secured,monkeypatch):
    from askpolicy.storage.index_store import PublishedSnapshot
    monkeypatch.setattr(PublishedSnapshot,"search",lambda *a,**kw:())
    response=request_api(api_env.app,secured.sign(),"employee")
    assert response.status==200 and response.json()["results"]==[]


def test_auth_verify_stays_available_without_retrieval(secured,monkeypatch):
    from askpolicy.api.app import create_app
    for name in ("ASKPOLICY_RELEASE_ROOT","ASKPOLICY_E5_DIR","ASKPOLICY_MANIFEST_PATH"):
        monkeypatch.delenv(name,raising=False)
    app=create_app(jwt_config=secured.config)
    auth=request_api(app,secured.sign(),"employee",path="/auth/verify",method="GET")
    assert auth.status==200 and auth.json()=={"authenticated":True,"active_role":"employee"}
    assert request_api(app,secured.sign(),"employee").status==503


def test_one_verifier_per_request_and_ordered_storage_gate(api_env,secured,monkeypatch):
    from askpolicy.auth.jwt import JWTVerifier
    from askpolicy.storage import publication
    from askpolicy.retrieval import role_bm25,role_faiss
    events=[]
    verifiers=[]
    verify=JWTVerifier.verify
    def auth(self,*args,**kw):
        result=verify(self,*args,**kw)
        events.append("auth")
        verifiers.append(self)
        return result
    monkeypatch.setattr(JWTVerifier,"verify",auth)
    for module,name,event in ((publication,"load_active_snapshot","load"),
                              (role_bm25,"search_role_bm25","bm25"),
                              (role_faiss,"search_role_faiss","faiss")):
        original=getattr(module,name)
        def observe(*args,_fn=original,_event=event,**kw):
            events.append(_event)
            return _fn(*args,**kw)
        monkeypatch.setattr(module,name,observe)
    response=request_api(api_env.app,secured.sign(),"employee")
    assert response.status==200 and events==["auth","load","bm25","faiss"]
    response=request_api(api_env.app,secured.sign(),"employee",path="/auth/verify",method="GET")
    assert response.status==200 and len(verifiers)==2 and verifiers[0] is verifiers[1]
