"""Explicit real CPU experiment. Run separately from deterministic CI."""

from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
sys.path.insert(0,str(ROOT / "tests"))
sys.path.insert(0,str(ROOT / "eval"))
from security.test_auth import secured
from unit.test_e5_prefix import _working_set_mib


def test_real_dev_four_way_diagnostic(secured):
    import dev_four_way_diagnostic as diagnostic
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval.e5_encoder import E5Encoder
    from askpolicy.retrieval.cross_encoder import CrossEncoder, verify_checkpoint
    from askpolicy.storage import publication
    from askpolicy.storage.snapshot import DraftSnapshotConfig

    torch.set_num_threads(4)
    def no_network(*args,**kwargs):
        raise AssertionError("UNEXPECTED_NETWORK_ACCESS")
    started = perf_counter()
    with patch.object(socket.socket,"connect",no_network),patch.object(socket,"getaddrinfo",no_network):
        tick = perf_counter()
        e5 = E5Encoder(Path(os.environ["ASKPOLICY_E5_DIR"]))
        e5_loading_seconds = perf_counter()-tick
        tick = perf_counter()
        checkpoint = Path(os.environ["ASKPOLICY_RERANKER_DIR"])
        cross_encoder = CrossEncoder(checkpoint)
        cross_loading_seconds = perf_counter()-tick
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in cross_encoder._model.parameters())
        assert not cross_encoder._model.training and cross_encoder._tokenizer.is_fast
        root = ROOT / "runtime" / ("step107-published-" + uuid4().hex)
        options = dict(manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
                       snapshot_config=DraftSnapshotConfig(),validity_date=diagnostic.DAY,encoder=e5)
        raw = (ROOT / diagnostic.base.DATASET_PATH).read_bytes()
        cases = diagnostic.base.read_cases(raw)
        assert len(cases) == 5
        stage_seconds = []
        stage = publication.stage_snapshot
        def timed_stage(*args,**kwargs):
            tick = perf_counter()
            value = stage(*args,**kwargs)
            stage_seconds.append(perf_counter()-tick)
            return value
        with closing(sqlite3.connect(":memory:")) as draft:
            tick = perf_counter()
            ingest_manifest_to_draft(ROOT,draft)
            ingestion_seconds = perf_counter()-tick
            tick = perf_counter()
            with patch.object(publication,"stage_snapshot",timed_stage):
                published = publication.publish_snapshot(draft,root,**options)
            publish_seconds = perf_counter()-tick
        assert len(stage_seconds) == 1
        tick = perf_counter()
        handle = publication.load_active_snapshot(root,**options)
        reload_seconds = perf_counter()-tick
        assert handle.release_id == published.release_id and handle.manifest_hash == published.manifest_hash

        forward = cross_encoder._model.forward
        calls = []
        def observe(**batch):
            assert torch.is_inference_mode_enabled()
            assert batch["input_ids"].shape[1] <= 512
            calls.append(dict(pairs=int(batch["input_ids"].shape[0]),
                              padded_tokens=int(batch["input_ids"].shape[1])))
            return forward(**batch)
        with patch.object(cross_encoder._model,"forward",observe):
            report = diagnostic.evaluate(
                handle,release_root=root,manifest=options["manifest"],raw_dataset=raw,
                context_for_role=lambda role:secured.authenticate(secured.sign(roles=[role]),role),
                encoder=cross_encoder,execution_kind="real_cpu_float32",
            )
        core = report["deterministic"]
        assert len(calls) == 26 and calls[0]["pairs"] == 1
        assert sum(c["pairs"] for c in calls[1:]) == 100
        assert core["reranker_failures"] == core["unauthorized_hit_count"] == 0
        assert core["evaluated_count"] == 5 and core["positive_metric_denominator"] == 3
        token_counts = [h["token_count"] for c in core["modes"]["hybrid_rerank"]["cases"] for h in c["ranking"]]
        assert len(token_counts) == 100 and max(token_counts) <= 512
        session = dict(
            e5_loading_seconds=e5_loading_seconds,cross_encoder_loading_seconds=cross_loading_seconds,
            loading_condition="fresh_model_objects_in_this_process; operating_system_file_cache_not_flushed",
            measured_condition="one_sample_per_query_after_one_E5_query_and_one_CrossEncoder_pair_warmup",
            ingestion_seconds=ingestion_seconds,index_build_and_artifact_staging_seconds=stage_seconds[0],
            publication_verification_and_reload_seconds=publish_seconds-stage_seconds[0]+reload_seconds,
            published_reload_seconds=reload_seconds,publish_wall_seconds=publish_seconds,
            note="publish_wall overlaps staging; disjoint preparation total uses ingestion+staging+publication_verification_and_reload",
            memory=_working_set_mib(),network_requests=0,device="cpu",dtype="float32",
            warmup_forward_calls=1,measured_forward_calls=len(calls)-1,
            warmup_pair_count=1,measured_pair_count=sum(c["pairs"] for c in calls[1:]),
            forward_batch_observations=calls,token_minimum=min(token_counts),token_maximum=max(token_counts),
            overlength_cases=0,scored_overlength_pairs=0,
            cross_encoder_artifacts=verify_checkpoint(checkpoint),
            git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            lockfile_sha256=sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
            evaluator_source_sha256={name:sha256((ROOT / "eval" / name).read_bytes()).hexdigest()
                                     for name in ("dev_four_way_diagnostic.py","dev_retrieval_diagnostic.py","retrieval_metrics.py")},
            command="uv run --locked --no-sync python -m pytest tests/local/real_dev_four_way_diagnostic.py -v -s",
            evaluation_utc=datetime.now(UTC).isoformat(),random_seed=None,
            session_total_seconds=perf_counter()-started,
            limitations=[
                "Five approved French executions, four question families, three positive cases.",
                "Fifteen development annotations remain missing. Held-out cases were not read.",
                "Five samples per mode, one measured execution per query, exploratory offline CPU latency.",
                "Warm-up is one E5 query and one synthetic Cross-Encoder pair without gold.",
                "No production API, generation, refusal accuracy or final ablation acceptance measured.",
                "Raw logits are ranking scores, not probabilities or calibrated confidence.",
                "Whole-request token overflow rejection is authorized only for this offline experiment.",
            ],
        )
        report["measurements"]["session"] = session
        diagnostic.validate_report(report)
        target = ROOT / diagnostic.REPORT_PATH
        diagnostic.write_report(report,target,repository_root=ROOT)
        print(json.dumps(dict(report_path=str(target),metrics={m:d["metrics"] for m,d in core["modes"].items()},
            gold_ranks={m:[dict(query_id=c["query_id"],gold_ranks=c["gold_ranks"]) for c in d["cases"]]
                        for m,d in core["modes"].items()},comparisons=core["comparisons"],
            release_id=handle.release_id,snapshot_id=core["snapshot"]["candidate_snapshot_id"],
            measurements=report["measurements"]),sort_keys=True))
