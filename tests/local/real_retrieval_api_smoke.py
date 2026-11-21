"""Explicit authenticated HTTP smoke against real persisted E5/FAISS artifacts."""

from datetime import date
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from time import perf_counter
from uuid import uuid4

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests"))
from integration.test_retrieval_api import request_api
from security.test_auth import secured


def test_real_retrieval_api_smoke(secured,monkeypatch):
    from askpolicy.api.app import create_app
    from askpolicy.api.retrieval_runtime import RetrievalConfig,RetrievalRuntime

    torch.set_num_threads(4)
    source=Path(os.environ["ASKPOLICY_API_RELEASE_ROOT"])
    root=ROOT/"runtime"/("step98-"+uuid4().hex)
    shutil.copytree(source,root)
    pointer=json.loads((root/"active.json").read_bytes())
    manifest=json.loads((root/"releases"/pointer["release_id"]/"manifest.json").read_bytes())
    config=RetrievalConfig(release_root=root,e5_dir=Path(os.environ["ASKPOLICY_E5_DIR"]),
                           manifest_path=ROOT/"data/corpus_manifest.yaml")
    day=[date(2026,9,1)]
    loads=[]
    original=RetrievalRuntime._load_encoder
    def observe(self):
        if self._encoder is None:
            loads.append(True)
        return original(self)
    monkeypatch.setattr(RetrievalRuntime,"_load_encoder",observe)
    app=create_app(jwt_config=secured.config,retrieval_config=config,clock=lambda:day[0])
    records=[]
    def execute(label,token,role,expected):
        start=perf_counter()
        response=request_api(app,token,role,payload={"query":
            "registre des congés et soldes des absences" if role=="hr"
            else "Quel est le plafond pour une nuit d’hôtel en déplacement ?"})
        seconds=perf_counter()-start
        assert response.status==expected and response.headers[b"cache-control"]==b"no-store"
        data=response.json()
        result=dict(case=label,status=response.status,seconds=seconds)
        if expected==200:
            assert data["release_id"]==pointer["release_id"]
            assert data["snapshot_id"]==manifest["snapshot"]["candidate_snapshot_id"]
            assert data["active_role"]==role and len(data["results"])==20
            assert len({h["child_id"] for h in data["results"]})==20
            connection=sqlite3.connect(root/"releases"/pointer["release_id"]/"sqlite/corpus.sqlite")
            try:
                grants={doc:json.loads(roles) for doc,roles in connection.execute(
                    "SELECT document_id,allowed_roles FROM draft_documents")}
            finally:
                connection.close()
            assert all(role in grants[h["document_id"]] for h in data["results"])
            if role=="hr":
                assert any(h["document_id"].startswith("LEAVE-") for h in data["results"])
            else:
                assert all(not h["document_id"].startswith(("LEAVE-","ACCESS-")) for h in data["results"])
            result.update(role=role,release=data["release_id"],snapshot=data["snapshot_id"],
                          candidates=len(data["results"]),acl_violations=0)
        else:
            assert set(data)=={"detail"}
        records.append(result)
    execute("missing-jwt",None,"employee",401)
    execute("ungranted-role",secured.sign(),"hr",403)
    assert not loads
    execute("cold-employee",secured.sign(),"employee",200)
    execute("warm-hr",secured.sign(roles=["hr"]),"hr",200)
    execute("warm-employee",secured.sign(),"employee",200)
    assert len(loads)==1
    day[0]=date(2026,9,2)
    execute("wrong-trusted-date",secured.sign(),"employee",503)
    day[0]=date(2026,9,1)
    (root/"active.json").write_bytes(b"corrupted pointer")
    execute("corrupted-pointer",secured.sign(),"employee",503)
    print(json.dumps(dict(root=str(root),model_loads=len(loads),cases=records),sort_keys=True))
