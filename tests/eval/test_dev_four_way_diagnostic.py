"""Controlled four-mode comparison contracts with fixture embeddings and logits."""

from copy import deepcopy
from hashlib import sha256
import json
from math import log2
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "eval"))
sys.path.insert(0,str(ROOT / "tests"))
from unit.test_offline_reranking import session
from unit.test_evidence_hydration import release
from unit.test_cross_encoder_checkpoint import local
from integration.test_embedding_revision import dense_env
from security.test_auth import secured
import dev_four_way_diagnostic as diagnostic


@pytest.fixture
def evaluation(session,secured):
    raw = (ROOT / diagnostic.base.DATASET_PATH).read_bytes()
    def run(**changes):
        return diagnostic.evaluate(
            changes.get("handle",session.handle),release_root=session.root,
            manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
            raw_dataset=changes.get("raw",raw),
            context_for_role=changes.get("provider",lambda role:secured.authenticate(secured.sign(roles=[role]),role)),
            encoder=session.encoder,execution_kind="fixture_embeddings_and_logits",
        )
    return session,raw,run


def rehash(report):
    report["identity_sha256"] = sha256(diagnostic.canonical(report["deterministic"])).hexdigest()


def test_identical_four_mode_contract_and_fourteen_positive_denominator(evaluation):
    session,raw,run = evaluation
    report = run()
    core = report["deterministic"]
    assert tuple(core["modes"]) == diagnostic.MODES
    assert core["evaluated_count"] == 20 and core["positive_metric_denominator"] == 14
    assert core["no_positive_count"] == 6 and core["question_family_count"] == 17
    assert core["required_development_count"] == 20 and core["required_held_out_count"] == 60
    assert core["annotation_gap"] == 0 and core["held_out_evaluated_count"] == 0
    assert core["status"] == "DEVELOPMENT_COVERAGE_COMPLETE"
    assert core["validation_status"] == "VALIDATED_OFFLINE_DIAGNOSTIC"
    assert not core["full_ablation_complete"] and not core["production_reranker_enabled"]
    assert core["dataset_sha256"] == sha256(raw).hexdigest()
    assert core["languages"] == {"fr":8,"en":6,"ar":6}
    assert core["roles"] == {"employee":10,"hr":3,"finance":3,"it_security":4}
    expected_ids = [c.query_id for c in diagnostic.base.read_cases(raw)]
    for mode,data in core["modes"].items():
        assert data["snapshot_id"] == session.handle.bm25.snapshot.candidate_snapshot_id
        assert data["published_release_id"] == session.handle.release_id
        assert data["dataset_sha256"] == core["dataset_sha256"]
        assert [c["query_id"] for c in data["cases"]] == expected_ids
        assert [c["metrics"] is None for c in data["cases"]] == [not c.authorized_source_sections for c in diagnostic.base.read_cases(raw)]
        assert data["metrics"]["eligible_positive_count"] == 14
        assert data["metrics"]["excluded_no_positive_count"] == 6
        assert data["unauthorized_hit_count"] == 0
        assert data["role_pairs"][0]["negative_restricted_section_hits"] == 0
        assert len(data["role_pairs"]) == 3
        assert all(p["negative_restricted_section_hits"] == 0 for p in data["role_pairs"])
        timing = report["measurements"]["modes"][mode]
        assert timing["sample_count"] == 20
        assert timing["interpretation"] == "exploratory_20_query_samples_not_production_api_latency"
    left, right = (core["modes"][m]["cases"] for m in ("hybrid","hybrid_rerank"))
    for before,after in zip(left,right,strict=True):
        assert before["gold_sections"] == after["gold_sections"]
        assert {h["child_id"] for h in before["ranking"]} == {h["child_id"] for h in after["ranking"]}
        assert len(before["ranking"]) == len(after["ranking"]) == 20
    assert core["models"]["cross_encoder"]["revision"] == diagnostic.MODEL_REVISION


def test_section_metrics_duplicate_gain_and_original_child_positions():
    first,second,irrelevant = ("DOC","p1"),("DOC","p2"),("DOC","p3")
    scores = diagnostic.section_metrics([first,first,irrelevant,second],{first:3,second:2})
    assert scores["recall_at_5"] == 1.0 and scores["mrr_at_10"] == 1.0
    assert scores["ndcg_at_10"] == pytest.approx((7+3/log2(5))/(7+3/log2(3)))
    assert diagnostic.section_metrics([irrelevant]*5+[first],{first:3})["recall_at_5"] == 0
    assert diagnostic.section_metrics([irrelevant]*5+[first],{first:3})["mrr_at_10"] == 1/6
    assert diagnostic.section_metrics([first],{}) is None


def test_paired_deltas_and_macro_recomputed_without_improvement_assumption(evaluation):
    report = evaluation[2]()
    core = report["deterministic"]
    modes = core["modes"]
    paired = core["comparisons"]["paired_hybrid_differences"]
    for delta,before,after in zip(paired,modes["hybrid"]["cases"],modes["hybrid_rerank"]["cases"],strict=True):
        if before["metrics"] is None:
            assert delta["differences"] is None
        else:
            assert delta["differences"] == {
                name:after["metrics"][name]-before["metrics"][name] for name in diagnostic.METRIC_NAMES}
    for mode in ("bm25","faiss","hybrid"):
        assert core["comparisons"]["rerank_minus_baseline_macro"][mode] == {
            name:modes["hybrid_rerank"]["metrics"][name]-modes[mode]["metrics"][name]
            for name in diagnostic.METRIC_NAMES}


@pytest.mark.parametrize("defect",["missing","empty","unreviewed","section","grade"])
def test_missing_or_invalid_labels_prevent_model_forward(evaluation,defect):
    session,raw,run = evaluation
    records = [json.loads(line) for line in raw.splitlines()]
    if defect == "missing":
        del records[0]["authorized_source_sections"]
    elif defect == "empty":
        records[0]["authorized_source_sections"] = []
    elif defect == "unreviewed":
        records[0]["annotation_review_status"] = "pending_human_review"
    elif defect == "section":
        records[0]["authorized_source_sections"][0]["section_heading"] = "nonexistent"
    else:
        records[0]["authorized_source_sections"][0]["relevance_grade"] = 0
    changed = b"\n".join(json.dumps(r,ensure_ascii=False).encode() for r in records)
    with pytest.raises(diagnostic.FourWayDiagnosticError):
        run(raw=changed)
    assert not session.local.model.calls


@pytest.mark.parametrize("defect",[
    "mode","snapshot","release","dataset","gold","metrics","macro","denominator","candidate","order",
    "revision","budget","coverage","full_claim","unauthorized","delta","rank","score","tokens","text",
    "extra","latency","stage","validation",
])
def test_report_schema_and_binding_reject_tampering(evaluation,defect):
    report = evaluation[2]()
    core = report["deterministic"]
    fourth = core["modes"]["hybrid_rerank"]
    row = fourth["cases"][0]
    if defect == "mode":
        del core["modes"]["bm25"]
    elif defect in ("snapshot","release","dataset"):
        key = {"snapshot":"snapshot_id","release":"published_release_id","dataset":"dataset_sha256"}[defect]
        fourth[key] = "0"*64
    elif defect == "gold":
        row["gold_sections"] = []
    elif defect == "metrics":
        row["metrics"]["mrr_at_10"] = -1.0
    elif defect == "macro":
        fourth["metrics"]["ndcg_at_10"] = -1.0
    elif defect == "denominator":
        core["positive_metric_denominator"] = 5
    elif defect == "candidate":
        row["ranking"][0]["child_id"] = "fabricated"
    elif defect == "order":
        row["ranking"].reverse()
        for rank,hit in enumerate(row["ranking"],1):
            hit["rank"] = rank
    elif defect == "revision":
        core["models"]["cross_encoder"]["revision"] = "main"
    elif defect == "budget":
        core["configurations"]["hybrid"]["rrf_constant"] = 10
    elif defect == "coverage":
        core["annotation_gap"] = 15
    elif defect == "full_claim":
        core["full_ablation_complete"] = True
    elif defect == "unauthorized":
        fourth["unauthorized_hit_count"] = 1
    elif defect == "delta":
        core["comparisons"]["rerank_minus_baseline_macro"]["hybrid"]["recall_at_5"] = -100.0
    elif defect == "rank":
        row["ranking"][0]["parent_id"] = "wrong"
    elif defect == "score":
        row["ranking"][0]["raw_score"] = float("inf")
    elif defect == "tokens":
        row["ranking"][0]["token_count"] = 513
    elif defect == "text":
        row["text"] = "protected-marker"
    elif defect == "extra":
        core["deployment_approved"] = True
    elif defect == "latency":
        report["measurements"]["modes"]["hybrid_rerank"]["p95_seconds"] = -1.0
    elif defect == "validation":
        core["validation_status"] = "PRODUCTION_APPROVED"
    else:
        report["measurements"]["modes"]["hybrid_rerank"]["per_case_seconds"][0]["stages"]["scoring_seconds"] = -1.0
    if defect != "score":
        rehash(report)
    with pytest.raises(diagnostic.FourWayDiagnosticError):
        diagnostic.validate_report(report)


def test_overflow_failed_case_has_no_fourth_metrics_or_comparable_macro(evaluation,monkeypatch):
    original = diagnostic.rerank_published_hits
    calls = []
    def overflow(*args,**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise diagnostic.OfflineRerankError("TOKEN_LIMIT")
        return original(*args,**kwargs)
    monkeypatch.setattr(diagnostic,"rerank_published_hits",overflow)
    report = evaluation[2]()
    core = report["deterministic"]
    fourth = core["modes"]["hybrid_rerank"]
    assert core["reranker_failures"] == 1
    assert fourth["cases"][0]["status"] == "FAILED_TOKEN_LIMIT"
    assert fourth["cases"][0]["metrics"] is None and fourth["cases"][0]["ranking"] == []
    assert fourth["metrics"] is None and fourth["completed_positive_count"] == 13
    assert core["comparisons"]["paired_hybrid_differences"][0]["differences"] is None
    assert core["comparisons"]["rerank_minus_baseline_macro"]["hybrid"] is None
    assert core["modes"]["hybrid"]["metrics"]["eligible_positive_count"] == 14


def test_wrong_active_role_provider_is_rejected_before_warmup(evaluation):
    session,_,run = evaluation
    with pytest.raises(diagnostic.FourWayDiagnosticError):
        run(provider=lambda role:session.context)
    assert not session.local.model.calls


def test_model_failure_aborts_without_fake_fourth_scores(evaluation,monkeypatch):
    def fail(*args,**kwargs):
        raise diagnostic.OfflineRerankError("OFFLINE_RERANK_FAILED")
    monkeypatch.setattr(diagnostic,"rerank_published_hits",fail)
    with pytest.raises(diagnostic.FourWayDiagnosticError):
        evaluation[2]()


def test_historical_five_case_coverage_remains_incomplete(evaluation):
    raw=(ROOT/"eval/datasets/dev_seed_five.jsonl").read_bytes()
    report=evaluation[2](raw=raw)
    core=report["deterministic"]
    assert core["status"] == "INCOMPLETE_DEVELOPMENT_COVERAGE"
    assert core["evaluated_count"] == 5 and core["annotation_gap"] == 15
    assert core["positive_metric_denominator"] == 3 and core["no_positive_count"] == 2
    assert core["held_out_evaluated_count"] == 0
    assert core["full_ablation_complete"] is False and core["production_reranker_enabled"] is False
    assert report["measurements"]["modes"]["hybrid_rerank"]["interpretation"] == "exploratory_5_query_samples_not_production_api_latency"


def test_atomic_safe_report_and_no_raw_text_or_heldout_access(evaluation,tmp_path,monkeypatch):
    session,raw,run = evaluation
    accessed = []
    original_read = Path.read_bytes
    def observe(path):
        accessed.append(path)
        if "held" in str(path).lower():
            raise AssertionError("HELD_OUT_ACCESS")
        return original_read(path)
    monkeypatch.setattr(Path,"read_bytes",observe)
    report = run()
    target = tmp_path / diagnostic.REPORT_PATH
    diagnostic.write_report(report,target,repository_root=tmp_path)
    restored = json.loads(target.read_bytes())
    diagnostic.validate_report(restored)
    text = target.read_text(encoding="utf-8")
    assert all(c.query not in text for c in diagnostic.base.read_cases(raw))
    assert not any("held" in str(p).lower() for p in accessed)
    before = target.read_bytes()
    def interrupted(*args):
        raise OSError("interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(diagnostic.os,"replace",interrupted)
        with pytest.raises(OSError):
            diagnostic.write_report(report,target,repository_root=tmp_path)
    assert target.read_bytes() == before
    assert not list(target.parent.glob(".step107-*"))
    for path in (tmp_path/"eval/report.json",tmp_path/"runtime/reports/step100_dev_retrieval_diagnostic.json"):
        with pytest.raises(diagnostic.FourWayDiagnosticError):
            diagnostic.write_report(report,path,repository_root=tmp_path)
