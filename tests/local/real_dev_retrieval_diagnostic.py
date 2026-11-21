"""Explicit real E5 three-mode development diagnostic over the full corpus."""

from contextlib import closing
from datetime import UTC,datetime
from pathlib import Path
import json
import os
import sqlite3
import sys
import subprocess
from hashlib import sha256
from time import perf_counter
from uuid import uuid4

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests"))
sys.path.insert(0,str(ROOT/"eval"))
from security.test_auth import secured
from unit.test_e5_prefix import _working_set_mib


def test_real_dev_retrieval_diagnostic(secured):
    import dev_retrieval_diagnostic as diagnostic
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval import role_bm25,role_faiss
    from askpolicy.retrieval.e5_encoder import E5Encoder
    from askpolicy.storage.snapshot import DraftSnapshotConfig

    torch.set_num_threads(4)
    started=perf_counter()
    encoder=E5Encoder(Path(os.environ["ASKPOLICY_E5_DIR"]))
    model_seconds=perf_counter()-started
    manifest=(ROOT/"data/corpus_manifest.yaml").read_bytes()
    raw=(ROOT/"eval/datasets/dev_seed_five.jsonl").read_bytes()
    started=perf_counter()
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(ROOT,db)
        options=dict(manifest=manifest,snapshot_config=DraftSnapshotConfig(),validity_date=diagnostic.DAY)
        bm25=role_bm25.build_role_bm25_bundle(db,**options)
        faiss=role_faiss.build_role_faiss_bundle(db,**options,encoder=encoder)
        build_seconds=perf_counter()-started
        report=diagnostic.evaluate(db,manifest=manifest,raw_dataset=raw,bm25=bm25,faiss=faiss,
            context_for_role=lambda role:secured.authenticate(secured.sign(roles=[role]),role),
            execution_kind="real_e5_cpu")
    report["measurements"].update(model_loading_seconds=model_seconds,index_building_seconds=build_seconds,
        process_ram=_working_set_mib(),git_commit=subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        lockfile_sha256=sha256((ROOT/"uv.lock").read_bytes()).hexdigest(),
        command="uv run --locked --no-sync python -m pytest tests/local/real_dev_retrieval_diagnostic.py -v -s",
        api_overhead_measured=False,random_seed=None,evaluation_utc=datetime.now(UTC).isoformat(),
        evaluator_source_sha256={name:sha256((ROOT/"eval"/name).read_bytes()).hexdigest()
                                 for name in ("retrieval_metrics.py","dev_retrieval_diagnostic.py")})
    path=ROOT/"runtime/reports"/("step100_seed_rerun_"+uuid4().hex+".json")
    diagnostic.write_report(report,path,repository_root=ROOT)
    assert report["deterministic"]["evaluated_count"]==5
    assert all(data["unauthorized_hit_count"]==0 for data in report["deterministic"]["modes"].values())
    print(json.dumps(dict(path=str(path),identity=report["identity_sha256"],
        snapshot=report["deterministic"]["snapshot"]["candidate_snapshot_id"],
        dataset=report["deterministic"]["dataset_sha256"],
        metrics={mode:data["metrics"] for mode,data in report["deterministic"]["modes"].items()},
        measurements=report["measurements"]),sort_keys=True))
