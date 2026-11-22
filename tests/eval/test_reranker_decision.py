"""Step 116 owner decision and actual non-reranked production path.

Standard tests use committed contracts and fixture embeddings, never ignored
benchmark reports, held-out cases, checkpoint files or real-model inference.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
for directory in ("src", "eval", "tests"):
    sys.path.insert(0, str(ROOT / directory))

from integration.test_retrieval_api import api_env, api_release, request_api
from integration.test_embedding_revision import dense_env
from security.test_auth import secured
import dev_four_way_diagnostic as diagnostic
import dev_retrieval_architecture_review as review
from askpolicy.retrieval import cross_encoder, offline_reranking, role_bm25, role_faiss, rrf


@pytest.fixture(scope="module")
def contracts():
    document = (ROOT / "docs/decisions.md").read_text(encoding="utf-8")
    start = document.index("### Step 116 owner decision — v1 retrieval baseline\n")
    end = document.index("### Exact model versions\n", start)
    section = document[start:end]
    blocks = re.findall(r"```json\n(.*?)\n```", section, re.DOTALL)
    assert len(blocks) == 1, "One unambiguous owner decision record is required"
    return document, section, json.loads(blocks[0])


def test_explicit_owner_approval_retains_hybrid_and_disables_production_ce(contracts):
    _, _, record = contracts
    assert record["decision_id"] == "step116_v1_retrieval"
    assert record["decision_date"] == "2026-11-22"
    assert record["owner_approved"] is True
    assert record["approval_source"] == "explicit_project_owner_step116_instruction"
    assert record["deployment_baseline"] == "hybrid"
    assert record["production_reranker_enabled"] is False
    assert record["retrieval_configuration_change_approved"] is False


def test_all_four_evaluation_modes_and_locked_budgets_preserved(contracts):
    _, _, record = contracts
    assert tuple(record["evaluation_modes"]) == diagnostic.MODES == (
        "bm25", "faiss", "hybrid", "hybrid_rerank")
    assert record["budgets"] == dict(bm25_top_k=30, faiss_top_k=30, rrf_k=60,
        hybrid_top_k=20, rerank_top_k=20, generation_evidence_max=5)
    budgets = record["budgets"]
    assert role_bm25.RoleBM25Configuration("fixture", "fixture").top_k == budgets["bm25_top_k"]
    assert role_faiss.DenseConfiguration().top_k_children == budgets["faiss_top_k"]
    assert rrf.INPUT_LIMIT == 30 and rrf.OUTPUT_LIMIT == budgets["hybrid_top_k"]
    assert rrf.RRF_K == budgets["rrf_k"]
    assert callable(diagnostic.rerank_published_hits), "Fourth evaluation configuration remains available"


def test_post_release_deferral_does_not_authorize_architecture_change_or_fine_tuning(contracts):
    document, section, record = contracts
    for field in ("faiss_only_selection", "embedding_fine_tuning"):
        assert record[field] == "DEFERRED_UNTIL_AFTER_V1_PORTFOLIO_RELEASE"
    assert record["post_release_changes_require_separate_approval"] is True
    assert "deferral, not advance approval" in section
    assert "separate approval under §18" in section
    assert "No fine-tuning;" in document.split("## 22. Explicitly Out of Scope\n", 1)[1]


def quality_rows(section):
    rows = {}
    for label in ("BM25", "FAISS", "Hybrid", "Hybrid + CE"):
        line, = [line for line in section.splitlines() if line.startswith("| " + label + " |")]
        rows[label] = [float(v.strip()) for v in line.split("|")[2:-1]]
    return rows


def test_recorded_metrics_and_latency_match_verified_historical_evidence(contracts):
    _, section, _ = contracts
    assert quality_rows(section) == {
        "BM25": [0.5476190476190476, 0.9523809523809523, 0.6104075558218088,
                 0.0034319000005780254, 0.004305120000935859],
        "FAISS": [0.5892857142857142, 1.0, 0.7219831671154645,
                  0.021969399998852168, 0.026169804999699408],
        "Hybrid": [0.5357142857142856, 0.9642857142857143, 0.6287460933335888,
                   0.04606510000121489, 0.049169684998560115],
        "Hybrid + CE": [0.6924603174603174, 0.869047619047619, 0.719221936220623,
                        0.48695730000144977, 0.5658439600001657],
    }


def test_faiss_advantage_is_acknowledged_and_rationale_does_not_claim_quality_win(contracts):
    _, section, _ = contracts
    rows = quality_rows(section)
    assert all(a > b for a, b in zip(rows["FAISS"][:3], rows["Hybrid"][:3], strict=True))
    assert "FAISS outperformed Hybrid on all three macro quality metrics" in section
    for rationale in ("architecture continuity", "tested RBAC boundaries", "reproducibility", "completion of v1"):
        assert rationale in section
    assert "not a claim that Hybrid won the benchmark or that FAISS is inherently less secure" in section


def test_reranker_gains_regressions_and_p95_are_explicitly_recorded(contracts):
    _, section, _ = contracts
    rows = quality_rows(section)
    for index in range(3):
        assert str(abs(rows["Hybrid + CE"][index] - rows["Hybrid"][index])) in section
    assert rows["Hybrid + CE"][1] < rows["Hybrid"][1]
    assert rows["Hybrid + CE"][4] > rows["Hybrid"][4]
    assert "does not approve Cross-Encoder activation" in section
    assert "dev-candidate-009" in section and "dev-candidate-010" in section
    assert "original rank-1 gold child actually moved to rank 4" in section
    assert "20 observations per mode" in section and "OS file cache not flushed" in section
    assert "not production API or end-to-end generation latency" in section


def test_provenance_is_bound_without_requiring_private_reports_in_ci(contracts):
    _, _, record = contracts
    assert record["step112_report_sha256"] == review.SOURCE_SHA
    assert record["step114_analysis_sha256"] == "f0aca0f80e2dc874a618abf737595383e095797b7aec559dd941168245c70d90"
    assert record["step114_packet_sha256"] == "44d62451a57fb75a3f2096cad521f9d9ff81a17519c91cfdc41712b6f5186fe3"
    assert record["dataset_sha256"] == sha256((ROOT / "eval/datasets/dev_candidates.jsonl").read_bytes()).hexdigest()
    smoke = json.loads((ROOT / "eval/benchmarks/bm25_public_smoke.json").read_bytes())
    assert record["snapshot_id"] == smoke["candidate_draft_snapshot_id"]
    assert smoke["draft_counts"]["documents"] >= 20
    assert smoke["macro_recall_at_5"] > 0 and smoke["evaluated_positive_case_count"] == 1


def test_unmet_frozen_ablation_gate_is_not_replaced_by_development_diagnostics(contracts):
    document, section, record = contracts
    assert record["development_diagnostic_complete"] is True
    assert record["contractual_frozen_ablation_complete"] is False
    assert record["phase5_status"] == "OPEN"
    assert "| Retrieval ablation | Four configs run on frozen benchmark with machine-readable `eval/report.json`. | `tests/eval/test_ablation.py` |" in document
    assert "Exit: role isolation passes, 4-way ablation recorded, inclusion decision justified on dev metrics." in document
    assert "No retrieval architecture" in section and "sequencing tension" in section
    assert "Step 116 approves neither a waiver nor such an amendment" in section
    assert "Do not advance to Phase 6 until the actual Phase 5 gates pass" in section
    # This historical audit does not assert that future approved work must keep
    # the ablation files absent. Their actual absence is inspected in Step 116.


@pytest.mark.parametrize("role", ["employee", "hr", "finance", "it_security"])
def test_production_http_uses_both_role_engines_rrf_and_no_cross_encoder(
    contracts, api_env, secured, monkeypatch, role,
):
    assert contracts[2]["production_reranker_enabled"] is False
    calls = []
    for mode, module, name in (("bm25", role_bm25, "search_role_bm25"),
                               ("faiss", role_faiss, "search_role_faiss")):
        original = getattr(module, name)
        def observed(bundle, context, _mode=mode, _original=original, **options):
            hits = _original(bundle, context, **options)
            calls.append((_mode, context.active_role, hits))
            return hits
        monkeypatch.setattr(module, name, observed)

    def forbidden(*args, **kwargs):
        raise AssertionError("PRODUCTION_RERANKING_NOT_APPROVED")
    monkeypatch.setattr(cross_encoder.CrossEncoder, "__init__", forbidden)
    monkeypatch.setattr(cross_encoder.CrossEncoder, "score_pairs", forbidden)
    monkeypatch.setattr(offline_reranking, "rerank_published_hits", forbidden)
    monkeypatch.setattr(diagnostic, "rerank_published_hits", forbidden)

    response = request_api(api_env.app, secured.sign(roles=[role]), role,
                           payload={"query": "travel congés expenses access"})
    assert response.status == 200
    assert [(mode, active) for mode, active, _ in calls] == [("bm25", role), ("faiss", role)]
    expected = rrf.reciprocal_rank_fusion(
        [h.child.child_id for h in calls[0][2]], [h.child.child_id for h in calls[1][2]])
    results = response.json()["results"]
    assert [(h["child_id"], h["rrf_score"]) for h in results] == [(h.child_id, h.score) for h in expected]
    assert all(role in api_env.handle.bm25._draft.documents[h["document_id"]].metadata.allowed_roles for h in results)
    assert response.json()["active_role"] == role
    assert response.headers[b"cache-control"] == b"no-store"
