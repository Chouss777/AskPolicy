"""Deterministic Step 112 contracts; fixture vectors/logits, no checkpoint load."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests"))
from eval.test_dev_four_way_diagnostic import evaluation, session, release, local, dense_env, secured

spec = importlib.util.spec_from_file_location("step112_runner",ROOT/"tests/local/real_dev_20_four_way_diagnostic.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def complete(evaluation):
    report = evaluation[2]()
    report["measurements"]["step112_summary"] = runner.derive_summary(report)
    cases = runner.read_inputs()[2]
    runner.validate_benchmark_report(report,cases)
    return report,cases


def test_exact_approved_input_and_counts_without_seed_or_heldout_reads(monkeypatch):
    reads = []
    original = Path.read_bytes
    def observe(path):
        reads.append(path.relative_to(ROOT).as_posix())
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",observe)
    raw,approval_raw,cases=runner.read_inputs()
    assert reads==["eval/datasets/dev_candidates.jsonl","eval/datasets/dev_annotation_approval.json"]
    assert sha256(raw).hexdigest()==runner.DATASET_SHA
    assert json.loads(approval_raw)["canonical_dataset_sha256"]==runner.DATASET_SHA
    assert len(cases)==20 and all(c.annotation_review_status=="human_approved" and not c.revalidation_required for c in cases)


def test_four_mode_identity_denominators_pairs_and_no_pruning(complete):
    report,cases=complete
    core=report["deterministic"]
    assert tuple(core["modes"])==("bm25","faiss","hybrid","hybrid_rerank")
    assert core["status"]=="DEVELOPMENT_COVERAGE_COMPLETE"
    assert core["positive_metric_denominator"]==14 and core["no_positive_count"]==6
    assert core["held_out_evaluated_count"]==0
    assert core["full_ablation_complete"] is False and core["production_reranker_enabled"] is False
    for mode,data in core["modes"].items():
        assert data["dataset_sha256"]==runner.DATASET_SHA
        assert data["snapshot_id"]==core["snapshot"]["candidate_snapshot_id"]
        assert data["published_release_id"]==core["published_release_id"]
        assert len(data["cases"])==20
        assert sum(r["metrics"] is None for r in data["cases"])==6
        assert data["metrics"]["eligible_positive_count"]==14
        assert sum(len(r["gold_sections"]) for r in data["cases"])==50
        assert [(p["positive_query_id"],p["negative_query_id"]) for p in data["role_pairs"]]==runner.PAIR_IDS
        assert all(p["negative_restricted_section_hits"]==0 for p in data["role_pairs"])
    for left,right in zip(core["modes"]["hybrid"]["cases"],core["modes"]["hybrid_rerank"]["cases"],strict=True):
        assert {(h["child_id"],h["document_id"],h["parent_id"]) for h in left["ranking"]}=={
            (h["child_id"],h["document_id"],h["parent_id"]) for h in right["ranking"]}


def test_language_role_type_aggregation_and_metric_ceilings(complete):
    report,_=complete
    s=report["measurements"]["step112_summary"]
    for language,total,positives in (("fr",8,6),("en",6,4),("ar",6,4)):
        for mode,data in report["deterministic"]["modes"].items():
            group=s["breakdowns"]["languages"][language][mode]
            assert (group["executions"],group["positive_gold_count"],group["no_positive_count"])==(total,positives,total-positives)
            rows=[r for r in data["cases"] if r["language"]==language]
            for metric in runner.METRIC_NAMES:
                assert group["metrics"][metric]==sum(r["metrics"][metric] for r in rows if r["metrics"] is not None)/positives
    assert {role:groups["hybrid"]["executions"] for role,groups in s["breakdowns"]["roles"].items()}=={
        "employee":10,"hr":3,"finance":3,"it_security":4}
    assert {kind:groups["hybrid"]["executions"] for kind,groups in s["breakdowns"]["case_types"].items()}=={
        "answerable":11,"unanswerable":3,"role_sensitive":6}
    low={r["query_id"]:(r["gold_section_count"],r["theoretical_maximum_recall_at_5"])
         for r in s["gold_section_ceilings"] if r["theoretical_maximum_recall_at_5"]<1}
    assert low=={"dev-candidate-010":(6,5/6),"dev-candidate-012":(6,5/6),
                 "dev-candidate-014":(9,5/9),"dev-candidate-017":(6,5/6)}


def test_paired_differences_rank_summary_and_latency_are_actual(complete):
    report,_=complete
    core=report["deterministic"]; s=report["measurements"]["step112_summary"]
    assert len(s["paired_positive_cases"])==14 and s["paired_full_sample_mean_available"]
    for row in s["paired_positive_cases"]:
        a=next(r for r in core["modes"]["hybrid"]["cases"] if r["query_id"]==row["query_id"])
        b=next(r for r in core["modes"]["hybrid_rerank"]["cases"] if r["query_id"]==row["query_id"])
        assert row["differences"]=={k:b["metrics"][k]-a["metrics"][k] for k in runner.METRIC_NAMES}
        assert row["before_first_relevant_rank"]==min((g["first_child_rank"] for g in a["gold_ranks"] if g["first_child_rank"] is not None),default=None)
    for outcomes in s["per_metric_outcomes"].values():
        assert sum(outcomes.values())==14 and outcomes["unavailable"]==0
    assert sum(s["pareto_outcomes"].values())==14
    for mode,t in s["latency"].items():
        actual=report["measurements"]["modes"][mode]
        assert t["sample_count"]==20
        assert t["total_seconds"]==sum(r["seconds"] for r in actual["per_case_seconds"])
        assert t["p50_seconds"]==actual["p50_seconds"] and t["p95_seconds"]==actual["p95_seconds"]
    assert len(s["reranking_overhead"]["per_case_seconds"])==20


@pytest.mark.parametrize("defect",["missing_model","wrong_snapshot","wrong_dataset","null_metrics","timing","full_ablation","production","gold_pruning","derived_summary","role_pair","sensitive_field","sensitive_value"])
def test_invalid_reports_rejected_even_after_rehash(complete,defect):
    report,cases=complete
    core=report["deterministic"]
    if defect=="missing_model": del core["models"]["cross_encoder"]
    elif defect=="wrong_snapshot": core["modes"]["faiss"]["snapshot_id"]="0"*64
    elif defect=="wrong_dataset": core["dataset_sha256"]="0"*64
    elif defect=="null_metrics": core["modes"]["bm25"]["cases"][3]["metrics"]={"recall_at_5":1.0}
    elif defect=="timing": report["measurements"]["modes"]["hybrid_rerank"]["p95_seconds"]=-1.0
    elif defect=="full_ablation": core["full_ablation_complete"]=True
    elif defect=="production": core["production_reranker_enabled"]=True
    elif defect=="gold_pruning": core["modes"]["hybrid_rerank"]["cases"][9]["gold_sections"].pop()
    elif defect=="derived_summary": report["measurements"]["step112_summary"]["gold_section_ceilings"][0]["theoretical_maximum_recall_at_5"]=0.1
    elif defect=="role_pair": core["modes"]["hybrid"]["role_pairs"].pop()
    elif defect=="sensitive_field": report["measurements"]["passage"]="protected marker"
    else: report["measurements"]["accidental_payload"]=cases[0].query
    if defect not in ("missing_model",):
        report["identity_sha256"]=sha256(runner.canonical(core)).hexdigest()
    with pytest.raises((ValueError,KeyError)):
        runner.validate_benchmark_report(report,cases)


@pytest.mark.parametrize("defect",["missing","invalid","unauthorized"])
def test_gold_failure_prevents_any_forward(evaluation,defect):
    session,raw,run=evaluation
    records=[json.loads(l) for l in raw.splitlines()]
    if defect=="missing": records[0]["authorized_source_sections"]=[]
    elif defect=="invalid": records[0]["authorized_source_sections"][0]["section_heading"]="nonexistent"
    else: records[1]["active_role"]="employee"
    with pytest.raises(runner.diagnostic.FourWayDiagnosticError):
        run(raw=b"\n".join(json.dumps(r,ensure_ascii=False).encode() for r in records))
    assert not session.local.model.calls


def test_catalog_guard_rejects_wrong_role_and_cross_snapshot(evaluation):
    session,_,run=evaluation
    report=run()
    cases=runner.read_inputs()[2]
    # Production contexts are freshly verified; no role strings are accepted.
    context={r:session.context for r in ("employee","hr","finance","it_security")}
    with pytest.raises(ValueError,match="ROLE_MISMATCH"):
        runner.check_evidence(report,session.handle,context,cases)
    changed=deepcopy(report)
    changed["deterministic"]["modes"]["bm25"]["snapshot_id"]="0"*64
    with pytest.raises(ValueError,match="SNAPSHOT_MISMATCH"):
        runner.check_evidence(changed,session.handle,context,cases)


@pytest.mark.parametrize("document_id,negative_index",[("LEAVE-001-EN-V2",3),("ACCESS-001-FR-V2",17),("LEAVE-001-FR-V1",3)])
def test_catalog_guard_rejects_other_language_protected_and_superseded_sources(evaluation,secured,dense_env,document_id,negative_index):
    session,_,run=evaluation
    report=run()
    cases=runner.read_inputs()[2]
    contexts={r:secured.authenticate(secured.sign(roles=[r]),r) for r in {c.active_role for c in cases}}
    runner.check_evidence(report,session.handle,contexts,cases)
    # Superseded children correctly do not exist in the active catalog. Take
    # only their real identities from the separate all-26-document fixture.
    identity=dense_env.db.execute("SELECT child_id,document_id,parent_id FROM draft_children WHERE document_id=? LIMIT 1",(document_id,)).fetchone()
    assert identity is not None
    hit=report["deterministic"]["modes"]["hybrid"]["cases"][negative_index]["ranking"][0]
    hit.update(child_id=identity[0],document_id=identity[1],parent_id=identity[2])
    with pytest.raises(ValueError,match="UNAUTHORIZED_RETURNED_EVIDENCE|INVALID_RETURNED_IDENTITY"):
        runner.check_evidence(report,session.handle,contexts,cases)


def test_failed_positive_query_keeps_full_denominator_and_null_summary(evaluation,monkeypatch):
    original=runner.diagnostic.rerank_published_hits
    calls=[]
    def overflow(*args,**kwargs):
        calls.append(1)
        if len(calls)==1: raise runner.diagnostic.OfflineRerankError("TOKEN_LIMIT")
        return original(*args,**kwargs)
    monkeypatch.setattr(runner.diagnostic,"rerank_published_hits",overflow)
    report=evaluation[2]()
    report["measurements"]["step112_summary"]=runner.derive_summary(report)
    runner.validate_benchmark_report(report,runner.read_inputs()[2])
    s=report["measurements"]["step112_summary"]
    assert report["deterministic"]["modes"]["hybrid_rerank"]["metrics"] is None
    assert report["deterministic"]["modes"]["hybrid_rerank"]["positive_metric_denominator"]==14
    assert not s["paired_full_sample_mean_available"]
    assert s["macro_differences"]["hybrid"] is None
    assert s["breakdowns"]["languages"]["fr"]["hybrid_rerank"]["metrics"] is None
    assert s["per_metric_outcomes"]["recall_at_5"]["unavailable"]==1
    assert s["failures"][0]["query_id"]=="dev-candidate-001"
    assert s["failures"][0]["metrics"] is None and s["failures"][0]["seconds"]>=0


def test_new_report_is_atomic_allowlisted_and_no_heldout_or_seed_access(complete,tmp_path,monkeypatch):
    report,cases=complete
    def forbidden(*args,**kwargs): raise AssertionError("UNEXPECTED_DATASET_ACCESS")
    monkeypatch.setattr(Path,"read_bytes",forbidden)
    target=runner.write_benchmark_report(report,cases,repository_root=tmp_path)
    assert target==tmp_path/runner.REPORT_PATH
    # Inspect this newly written artifact without opening any dataset.
    with target.open("rb") as stream: before=stream.read()
    # Upstream canonical JSON converts configuration tuples to JSON arrays.
    # Require exact canonical bytes and independently validate the readback.
    assert before==runner.canonical(report)+b"\n"
    restored=json.loads(before)
    assert restored==json.loads(runner.canonical(report))
    runner.validate_benchmark_report(restored,cases)
    for name in ("runtime/reports/step107_dev_four_way_diagnostic.json","eval/report.json"):
        with pytest.raises(ValueError,match="UNSAFE_BENCHMARK_PATH"):
            runner.write_benchmark_report(report,cases,repository_root=tmp_path,output_path=tmp_path/name)
    def interrupted(*args): raise OSError("interrupted")
    monkeypatch.setattr(runner.os,"replace",interrupted)
    with pytest.raises(OSError):
        runner.write_benchmark_report(report,cases,repository_root=tmp_path)
    with target.open("rb") as stream: assert stream.read()==before
    assert not list(target.parent.glob(".step112-*"))

