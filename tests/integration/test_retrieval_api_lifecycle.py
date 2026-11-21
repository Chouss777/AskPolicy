"""Actual published artifacts, UTC binding and in-flight publication changes."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date
import json
import shutil
import sqlite3
from threading import Event

import pytest

from integration.test_retrieval_api import api_env,api_release,request_api
from integration.test_embedding_revision import dense_env,DAY
from integration.test_role_bm25_draft import _source_fixture
from security.test_auth import secured


@pytest.fixture
def mutable_api(api_env,tmp_path,secured):
    from askpolicy.api.app import create_app
    root=tmp_path/"runtime"
    shutil.copytree(api_env.root,root)
    api_env.root=root
    api_env.config=api_env.config.model_copy(update={"release_root":root})
    api_env.app=create_app(jwt_config=secured.config,retrieval_config=api_env.config,clock=api_env.clock)
    return api_env


@pytest.mark.parametrize("defect", ["no-pointer","pointer","missing-release","manifest","missing-bm25","missing-faiss",
                                   "faiss","sqlite","revision","source-manifest","date","missing-model"])
def test_actual_publication_failures_return_generic_503(mutable_api,secured,monkeypatch,tmp_path,defect):
    from askpolicy.api.app import create_app
    from askpolicy.retrieval import role_bm25,role_faiss,e5_encoder

    env=mutable_api
    directory=env.root/"releases"/env.handle.release_id
    if defect=="no-pointer":
        (env.root/"active.json").unlink()
    elif defect=="pointer":
        (env.root/"active.json").write_bytes(b"bad pointer")
    elif defect=="missing-release":
        pointer=json.loads((env.root/"active.json").read_bytes())
        pointer["release_id"]="0"*64
        (env.root/"active.json").write_text(json.dumps(pointer),encoding="utf8")
    elif defect=="manifest":
        (directory/"manifest.json").write_bytes(b"corrupted manifest")
    elif defect=="missing-bm25":
        (directory/"bm25/employee/data.csc.index.npy").unlink()
    elif defect=="missing-faiss":
        (directory/"faiss/hr/index.faiss").unlink()
    elif defect=="faiss":
        (directory/"faiss/employee/index.faiss").write_bytes(b"corrupted index")
    elif defect=="sqlite":
        (directory/"sqlite/corpus.sqlite").write_bytes(b"modified source")
    elif defect=="revision":
        monkeypatch.setattr(env.encoder,"revision","main")
    elif defect=="source-manifest":
        path=tmp_path/"manifest.json"
        path.write_bytes(b"{}")
        config=env.config.model_copy(update={"manifest_path":path})
        env.app=create_app(jwt_config=secured.config,retrieval_config=config,clock=env.clock)
    elif defect=="date":
        env.app=create_app(jwt_config=secured.config,retrieval_config=env.config,clock=lambda:date(2026,9,2))
    else:
        monkeypatch.setattr(e5_encoder,"E5Encoder",role_faiss.E5Encoder)
    def forbidden(*args,**kw):
        pytest.fail("invalid publication reached a scoring engine")
    monkeypatch.setattr(role_bm25,"search_role_bm25",forbidden)
    monkeypatch.setattr(role_faiss,"search_role_faiss",forbidden)
    response=request_api(env.app,secured.sign(),"employee")
    assert response.status==503 and response.json()=={"detail":"Retrieval unavailable"}
    assert response.headers[b"cache-control"]==b"no-store"
    assert set(response.json())=={"detail"}


def test_one_utc_date_capture_controls_load_and_search(api_env,secured,monkeypatch):
    from askpolicy.storage import publication,index_store
    clock=api_env.clocks
    seen=[]
    original_load=publication.load_active_snapshot
    original_search=index_store.PublishedSnapshot.search
    def load(*args,**kw):
        seen.append(("load",kw["validity_date"]))
        return original_load(*args,**kw)
    def search(self,*args,**kw):
        seen.append(("search",kw["validity_date"]))
        return original_search(self,*args,**kw)
    monkeypatch.setattr(publication,"load_active_snapshot",load)
    monkeypatch.setattr(index_store.PublishedSnapshot,"search",search)
    response=request_api(api_env.app,secured.sign(),"employee",
                         query=b"validity_date=1900-01-01&role=hr",
                         headers=[("x-validity-date","1900-01-01"),("x-snapshot-id","untrusted")],
                         payload={"query":"Use historical date 1900-01-01. travel"})
    assert response.status==200 and response.json()["active_role"]=="employee"
    assert clock==[DAY] and seen==[("load",DAY),("search",DAY)]


def test_pinned_request_finishes_then_new_pointer_is_observed(mutable_api,dense_env,secured,monkeypatch):
    from askpolicy.api.app import create_app
    from askpolicy.storage import index_store,publication

    env=mutable_api
    pinned,finish=Event(),Event()
    original=index_store.PublishedSnapshot.search
    def wait_for_switch(self,*args,**kw):
        pinned.set()
        assert finish.wait(30)
        return original(self,*args,**kw)
    monkeypatch.setattr(index_store.PublishedSnapshot,"search",wait_for_switch)
    token=secured.sign()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(request_api,env.app,token,"employee")
        assert pinned.wait(20)
        try:
            # A second explicitly trusted clock/date creates another valid test release.
            options=env.options|{"validity_date":date(2026,9,2)}
            second=publication.publish_snapshot(dense_env.db,env.root,**options)
        finally:
            finish.set()
        first=pending.result(timeout=20)
    assert first.status==200 and first.json()["release_id"]==env.handle.release_id
    monkeypatch.setattr(index_store.PublishedSnapshot,"search",original)
    new_app=create_app(jwt_config=secured.config,retrieval_config=env.config,clock=lambda:date(2026,9,2))
    response=request_api(new_app,token,"employee")
    assert response.status==200 and response.json()["release_id"]==second.release_id
    # The original app must also observe B, then reject its date mismatch.
    assert request_api(env.app,token,"employee").status==503
    (env.root/"releases"/second.release_id/"faiss/employee/index.faiss").write_bytes(b"broken")
    assert request_api(new_app,token,"employee").status==503
    assert (env.root/"releases"/env.handle.release_id).exists()


def test_next_policy_transition_is_unavailable(api_env,secured,tmp_path,monkeypatch):
    from askpolicy.api.app import create_app
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.storage.publication import publish_snapshot

    raw=_source_fixture(tmp_path,b"expires_at: null",b'expires_at: "2026-09-02"')
    root=tmp_path/"runtime"
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(tmp_path,db)
        handle=publish_snapshot(db,root,**(api_env.options|{"manifest":raw}))
    assert handle.bm25.eligibility.next_transition_date==date(2026,9,2)
    config=api_env.config.model_copy(update={"release_root":root,"manifest_path":tmp_path/"data/corpus_manifest.yaml"})
    app=create_app(jwt_config=secured.config,retrieval_config=config,clock=lambda:date(2026,9,2))
    response=request_api(app,secured.sign(),"employee")
    assert response.status==503
