"""Synthetic recorded reports only: no model load, search or private report input."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"eval"))
import dev_retrieval_architecture_review as review
from askpolicy.retrieval.e5_encoder import MODEL_ID, MODEL_REVISION, ARTIFACT_FINGERPRINT


def digest(value):
    return sha256(value.encode()).hexdigest()


def refresh(report):
    core=report["deterministic"]
    core["comparisons"]=review.diagnostic._deltas(core["modes"])
    report["identity_sha256"]=sha256(review.canonical(core)).hexdigest()
    report["measurements"]["step112_summary"]=review.runner.derive_summary(report)


@pytest.fixture(scope="module")
def synthetic():
    """Construct recorded lists and timings arithmetically, never run retrieval.

    real_cpu_float32 is a schema field under test, not a claim that these
    fabricated fixture observations came from a real execution.
    """
    cases=review.runner.read_inputs()[2]
    provenance=dict(runner_source_sha256=digest("runner"),lockfile_sha256=digest("lock"),
        dataset_sha256=review.runner.DATASET_SHA,approval_manifest_sha256=digest("approval"),
        evaluator_source_sha256={n:digest(n) for n in review.EVALUATORS})
    modes={m:dict(cases=[]) for m in review.MODES}
    gold_by_id={}
    for case in cases:
        gold={(g.document_id,digest(g.document_id+g.section_heading)):g.relevance_grade
              for g in case.authorized_source_sections}
        gold_by_id[case.query_id]=gold
        identities=[]
        for d,p in gold:
            for i in range(2):
                identities.append(dict(child_id=digest(p+str(i)),document_id=d,parent_id=p))
        for i in range(30-len(identities)):
            identities.append(dict(child_id=digest(case.query_id+str(i)),
                document_id="TRAVEL-001-EN-V1",parent_id=digest("non-gold"+str(i))))
        def ranking(hits, scores=None, ce=False):
            return [dict(**h,rank=i,raw_score=float(31-i) if scores is None else scores[h["child_id"]],
                         token_count=20 if ce else None) for i,h in enumerate(hits,1)]
        dense=ranking(identities)
        lexical=ranking(identities[::2]+identities[1::2])
        fused=review.reciprocal_rank_fusion([h["child_id"] for h in lexical],[h["child_id"] for h in dense])
        mapping={h["child_id"]:{k:h[k] for k in ("child_id","document_id","parent_id")} for h in dense}
        hybrid=ranking([mapping[h.child_id] for h in fused],{h.child_id:h.score for h in fused})
        order=[mapping[h["child_id"]] for h in hybrid]
        if case.query_id in ("dev-candidate-009","dev-candidate-010"):
            nongold=[h for h in order if review._key(h) not in gold][:2]
            order=nongold+[h for h in order if h not in nongold]
        reranked=ranking(order,ce=True)
        for mode,ranked in zip(review.MODES,(lexical,dense,hybrid,reranked),strict=True):
            modes[mode]["cases"].append(review.diagnostic._row(case,ranked,gold))
    snapshot_id,release_id,fingerprint=digest("snapshot"),digest("release"),digest("manifest")
    for mode,data in modes.items():
        data.update(snapshot_id=snapshot_id,published_release_id=release_id,
            dataset_sha256=review.runner.DATASET_SHA,corpus_manifest_fingerprint=fingerprint,
            metrics=review.macro_metrics([r["metrics"] for r in data["cases"]]),
            positive_metric_denominator=14,completed_positive_count=14,
            returned_hit_count=sum(len(r["ranking"]) for r in data["cases"]),unauthorized_hit_count=0,
            violating_query_count=0,role_pairs=review.diagnostic.base._role_pairs(cases,gold_by_id,data["cases"]))
    core=dict(evaluation_kind="development_four_way_diagnostic",status="DEVELOPMENT_COVERAGE_COMPLETE",
        required_development_count=20,required_held_out_count=60,held_out_evaluated_count=0,evaluated_count=20,
        annotation_gap=0,positive_metric_denominator=14,no_positive_count=6,question_family_count=17,
        roles=dict(employee=10,hr=3,finance=3,it_security=4),languages=dict(fr=8,en=6,ar=6),
        snapshot=dict(candidate_snapshot_id=snapshot_id,source_manifest_hash=fingerprint,document_count=26,
                      child_count=387,parent_count=161,status="UNPUBLISHED"),
        published_release_id=release_id,published_manifest_sha256=digest("published manifest"),
        corpus_manifest_fingerprint=fingerprint,source_manifest_sha256=digest("source manifest"),
        dataset_sha256=review.runner.DATASET_SHA,validity_date="2026-09-01",execution_kind="real_cpu_float32",
        models=dict(e5=dict(model_id=MODEL_ID,revision=MODEL_REVISION,artifact_fingerprint=ARTIFACT_FINGERPRINT),
            cross_encoder=dict(model_id=review.diagnostic.MODEL_ID,revision=review.diagnostic.MODEL_REVISION,
                               artifact_fingerprint=review.diagnostic.ARTIFACT_FINGERPRINT)),
        configurations=dict(bm25=dict(top_k=30),faiss=dict(top_k_children=30),
            hybrid=dict(rrf_constant=60,bm25_top_k=30,faiss_top_k=30,top_k=20),
            hybrid_rerank=dict(input_top_k=20,joint_token_limit=512,overflow="reject_entire_request",
                              order=["raw_logit_descending","child_id_ascending"])),
        modes=modes,comparisons={},reranker_failures=0,unauthorized_hit_count=0,
        production_reranker_enabled=False,full_ablation_complete=False,validation_status="VALIDATED_OFFLINE_DIAGNOSTIC")
    timings={}
    for index,mode in enumerate(review.MODES):
        times=[]
        for i,case in enumerate(cases):
            value=float(index+1)+i/100
            row=dict(query_id=case.query_id,seconds=value)
            if mode=="hybrid_rerank":
                row.update(failure_code=None,stages=dict(hybrid_seconds=3.0+i/100,
                    hydration_seconds=0.1,token_validation_seconds=0.1,scoring_seconds=0.2,
                    ranking_seconds=0.1,helper_total_seconds=0.75,other_seconds=0.25))
            times.append(row)
        timings[mode]=review.diagnostic._timing(times)
    report=dict(deterministic=core,identity_sha256="",measurements=dict(modes=timings,
        hardware=dict(fixture="synthetic recorded report; no inference"),api_overhead_measured=False,
        session=dict(**deepcopy(provenance),git_commit=review.runner.BASELINE,device="cpu",dtype="float32",
            network_requests=0,memory=dict(peak_mib=100.0),benchmark_order=dict(
                query_ids=[c.query_id for c in cases],modes=list(review.MODES)))))
    refresh(report)
    review.validate_source(report,cases,provenance)
    return report,cases,provenance


@pytest.fixture
def evidence(synthetic):
    return deepcopy(synthetic)


def run(evidence):
    report,cases,provenance=evidence
    return review.analyze(report,cases,source_sha256=sha256(review.canonical(report)).hexdigest(),
                          current_commit=review.BASELINE,provenance=provenance)


def test_missing_and_altered_source_reject_without_private_report(tmp_path):
    with pytest.raises(review.ReviewError,match="MISSING_SOURCE_REPORT"):
        review.load_source(tmp_path)
    path=tmp_path/review.SOURCE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}")
    with pytest.raises(review.ReviewError,match="ALTERED_SOURCE_REPORT"):
        review.load_source(tmp_path)
    report,actual=review.load_source(tmp_path,expected_sha256=sha256(b"{}").hexdigest())
    assert report=={} and actual==sha256(b"{}").hexdigest()


@pytest.mark.parametrize("defect",[
    "dataset","historical_commit","runner_hash","evaluator_hash","lock_hash","approval_hash",
    "snapshot","snapshot_manifest","model","pair","unauthorized","metrics","macro","delta",
    "latency_count","invented_observation","negative_metrics","production","gold_grade","english_identity",
    "sensitive_field","query_value","jwt_value","timing_reuse","run_order",
])
def test_tampering_rejected_even_after_rehash(evidence,defect):
    report,cases,provenance=evidence
    core=report["deterministic"]; measured=report["measurements"]; session=measured["session"]
    if defect=="dataset": core["dataset_sha256"]="0"*64
    elif defect=="historical_commit": session["git_commit"]=review.BASELINE
    elif defect=="runner_hash": session["runner_source_sha256"]="0"*64
    elif defect=="evaluator_hash": session["evaluator_source_sha256"]["retrieval_metrics.py"]="0"*64
    elif defect=="lock_hash": session["lockfile_sha256"]="0"*64
    elif defect=="approval_hash": session["approval_manifest_sha256"]="0"*64
    elif defect=="snapshot": core["modes"]["faiss"]["snapshot_id"]="0"*64
    elif defect=="snapshot_manifest": core["snapshot"]["source_manifest_hash"]="0"*64
    elif defect=="model": core["models"]["cross_encoder"]["revision"]="main"
    elif defect=="pair": core["modes"]["hybrid"]["role_pairs"].pop()
    elif defect=="unauthorized": core["modes"]["faiss"]["cases"][3]["unauthorized_hit_count"]=1
    elif defect=="metrics": core["modes"]["faiss"]["cases"][0]["metrics"]["recall_at_5"]=0.01
    elif defect=="macro": core["modes"]["faiss"]["metrics"]["mrr_at_10"]=0.01
    elif defect=="delta": core["comparisons"]["paired_hybrid_differences"][0]["differences"]["mrr_at_10"]=0.01
    elif defect=="latency_count": measured["modes"]["faiss"]["sample_count"]=21
    elif defect=="invented_observation": measured["modes"]["faiss"]["per_case_seconds"].append(dict(query_id="invented",seconds=0.1))
    elif defect=="negative_metrics": core["modes"]["faiss"]["cases"][3]["metrics"]={m:0.0 for m in review.METRIC_NAMES}
    elif defect=="production": core["production_reranker_enabled"]=True
    elif defect=="gold_grade": core["modes"]["faiss"]["cases"][0]["gold_sections"][0]["relevance_grade"]=1
    elif defect=="english_identity": core["modes"]["hybrid_rerank"]["cases"][8]["query_id"]="dev-candidate-010"
    elif defect=="sensitive_field": measured["passage"]="protected synthetic marker"
    elif defect=="query_value": measured["accidental_payload"]=cases[0].query
    elif defect=="jwt_value": measured["accidental_payload"]="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.signature"
    elif defect=="timing_reuse": measured["modes"]["hybrid_rerank"]["per_case_seconds"][0]["stages"]["hybrid_seconds"]+=0.1
    elif defect=="run_order": session["benchmark_order"]["modes"].reverse()
    report["identity_sha256"]=sha256(review.canonical(core)).hexdigest()
    with pytest.raises((ValueError,KeyError)):
        review.validate_source(report,cases,provenance)


def test_paired_delta_tampering_detected_by_writer(evidence,tmp_path):
    a=run(evidence)
    a["per_query_comparisons"][0]["hybrid_minus_faiss"]["recall_at_5"]=999
    with pytest.raises(review.ReviewError,match="INVALID_ANALYSIS_DERIVATION"):
        review.write_outputs(a,*evidence[:2],provenance=evidence[2],repository_root=tmp_path)


def test_rrf_full_reconstruction_and_tie_order(evidence):
    a=run(evidence)
    assert len(a["rrf_reconstruction"])==20
    for r in a["rrf_reconstruction"]:
        assert r["exact_match"] and r["k"]==60 and r["hybrid_count"]==20
        assert r["candidates"]==sorted(r["candidates"],key=lambda h:(-h["rrf_score"],h["child_id"]))
        for h in r["candidates"]:
            assert h["rrf_score"]==sum(0.0 if h[m+"_rank"] is None else 1/(60+h[m+"_rank"]) for m in ("bm25","faiss"))
    hits=review.reciprocal_rank_fusion(["b","a"],["a","b"])
    assert [h.child_id for h in hits]==["a","b"] and hits[0].score==hits[1].score


def test_wrong_rrf_score_rejected_after_valid_metric_rehash(evidence):
    report,cases,provenance=evidence
    report["deterministic"]["modes"]["hybrid"]["cases"][0]["ranking"][0]["raw_score"]+=0.0001
    refresh(report)
    with pytest.raises(review.ReviewError,match="RRF_RECONSTRUCTION_MISMATCH"):
        run(evidence)


def test_zero_pair_counters_cannot_hide_actual_restricted_membership(evidence):
    report,cases,provenance=evidence
    for mode in review.MODES:
        rows=report["deterministic"]["modes"][mode]["cases"]
        positive=rows[2]["ranking"][0]
        negative=rows[3]["ranking"][0]
        for field in ("child_id","document_id","parent_id"):
            negative[field]=positive[field]
    refresh(report)
    # The upstream schema accepts the recorded zero counters; the independent
    # membership check must reject the actual positive section in the pair.
    review.runner.validate_benchmark_report(report,cases)
    with pytest.raises(review.ReviewError,match="UNAUTHORIZED_PAIR_EVIDENCE"):
        review.validate_source(report,cases,provenance)


def test_worktree_and_committed_provenance_must_agree(tmp_path,monkeypatch):
    paths=[review.runner.RUNNER_PATH,Path("uv.lock"),review.diagnostic.base.DATASET_PATH,
           review.APPROVAL_PATH,*(Path("eval")/n for n in review.EVALUATORS)]
    for p in paths:
        target=tmp_path/p
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.as_posix().encode())
    def committed(args,**kwargs):
        return args[-1].removeprefix("HEAD:").encode()
    monkeypatch.setattr(review.subprocess,"check_output",committed)
    actual=review.trusted_provenance(tmp_path)
    assert actual["runner_source_sha256"]==digest(review.runner.RUNNER_PATH.as_posix())
    (tmp_path/"uv.lock").write_bytes(b"changed")
    with pytest.raises(review.ReviewError,match="WORKTREE_PROVENANCE_MISMATCH"):
        review.trusted_provenance(tmp_path)


def test_duplicate_sections_and_ceiling_are_counted_without_relabelling(evidence):
    a=run(evidence)
    assert a["gold_reference_count"]==50
    assert {c["query_id"]:c["gold_section_count"] for c in a["gold_ceilings"]}=={
        "dev-candidate-010":6,"dev-candidate-012":6,"dev-candidate-014":9,"dev-candidate-017":6}
    for c in a["gold_ceilings"]:
        assert c["theoretical_recall_at_5_ceiling"]==5/c["gold_section_count"]
        for mode,v in c["modes"].items():
            assert v["gap_to_ceiling"]==c["theoretical_recall_at_5_ceiling"]-v["actual_recall_at_5"]
    accounting=a["per_query_comparisons"][0]["accounting"]["faiss"]
    assert accounting["duplicate_child_ids"]==0 and accounting["top5_repeated_section_slots"]>0
    assert len(accounting["gold_top5"])==1
    assert accounting["repeated_parent_sections"][0]["child_count"]>=2


def test_english_identity_logits_and_exact_deltas(evidence):
    a=run(evidence)
    assert [r["query_id"] for r in a["english_regressions"]]==["dev-candidate-009","dev-candidate-010"]
    for detail in a["english_regressions"]:
        assert len(detail["hybrid_top10"])==len(detail["reranked_top10"])==10
        assert detail["differences"]==review._delta(detail["before_metrics"],detail["after_metrics"])
        assert detail["before_first_relevant_rank"]==1 and detail["after_first_relevant_rank"]==3
        assert len(detail["non_gold_above_previous_first_relevant"])==2
        assert all(h["relevance_grade"]==0 for h in detail["non_gold_above_previous_first_relevant"])
        logits={h["child_id"]:h["raw_score"] for h in evidence[0]["deterministic"]["modes"]["hybrid_rerank"]["cases"]
                if h["query_id"]==detail["query_id"] for h in h["ranking"]}
        assert all(h["recorded_ce_raw_logit"]==logits[h["child_id"]] for h in detail["hybrid_top10"])


def test_latency_pairing_denominators_and_nonadditive_stages(evidence):
    a=run(evidence)
    for value in a["latency"].values():
        assert value["sample_count"]==20
    for name,value in a["paired_latency_overhead"].items():
        assert value["sample_count"]==len(value["per_case_seconds"])==20
        expected=1.0 if name in ("hybrid_minus_faiss","hybrid_rerank_minus_hybrid") else 2.0
        assert all(r["seconds"]==pytest.approx(expected) for r in value["per_case_seconds"])
        assert value["total_seconds"]==pytest.approx(20*expected)
    assert a["fourth_mode_stages"]["helper_unattributed_seconds"]["total_seconds"]==pytest.approx(5.0)


def test_null_negatives_language_denominators_and_no_automatic_decision(evidence):
    a=run(evidence)
    assert len(a["no_positive_query_ids"])==6 and a["security_execution_count"]==20
    assert a["decision_status"]=="AWAITING_OWNER_APPROVAL"
    assert not any(a[k] for k in ("owner_decision_made","production_reranker_enabled","retrieval_behavior_changed","full_ablation_complete","held_out_evaluated_count"))
    assert [o["option"] for o in a["decision_options"]]==["A","B","C"]
    for lang,count in (("fr",6),("en",4),("ar",4)):
        assert a["language_breakdowns"][lang]["positive_count"]==count
        assert all(v["eligible_positive_count"]==count for v in a["language_breakdowns"][lang]["quality"].values())
    for mode in review.MODES:
        assert sum(r["metrics"] is None for r in evidence[0]["deterministic"]["modes"][mode]["cases"])==6


def test_source_loading_reads_only_allowlisted_report(tmp_path,monkeypatch):
    path=tmp_path/review.SOURCE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}")
    original=Path.read_bytes
    reads=[]
    def observe(p):
        reads.append(p)
        assert p==path
        return original(p)
    monkeypatch.setattr(Path,"read_bytes",observe)
    review.load_source(tmp_path,expected_sha256=sha256(b"{}").hexdigest())
    assert reads==[path]


def test_analysis_does_not_read_datasets_search_or_load_checkpoints(evidence,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError("UNEXPECTED_IO_OR_EXECUTION")
    monkeypatch.setattr(Path,"read_bytes",forbidden)
    monkeypatch.setattr(review.diagnostic,"evaluate",forbidden)
    monkeypatch.setattr(review.diagnostic.role_bm25,"search_role_bm25",forbidden)
    monkeypatch.setattr(review.diagnostic.role_faiss,"search_role_faiss",forbidden)
    monkeypatch.setattr(review.diagnostic.hybrid,"search_role_hybrid",forbidden)
    from askpolicy.retrieval.e5_encoder import E5Encoder
    from askpolicy.retrieval.cross_encoder import CrossEncoder
    monkeypatch.setattr(E5Encoder,"__init__",forbidden)
    monkeypatch.setattr(CrossEncoder,"__init__",forbidden)
    a=run(evidence)
    encoded=review.canonical(a).decode()
    assert all(c.query not in encoded and all(f not in encoded for f in c.required_facts) for c in evidence[1])
    assert "eyJ" not in encoded


def test_atomic_safe_allowlisted_paths_and_determinism(evidence,tmp_path,monkeypatch):
    a=run(evidence)
    args=dict(provenance=evidence[2],repository_root=tmp_path)
    paths=review.write_outputs(a,*evidence[:2],**args)
    before=[p.read_bytes() for p in paths]
    assert paths==(tmp_path/review.JSON_PATH,tmp_path/review.PACKET_PATH)
    assert json.loads(before[0])==a
    assert before[1].decode()==review.render_packet(a)
    assert before[1].decode().count("\n## ")==13
    review.write_outputs(run(evidence),*evidence[:2],**args)
    assert [p.read_bytes() for p in paths]==before
    for bad in (review.SOURCE_PATH,Path("eval/report.json"),Path("runtime/reports/../bad.json")):
        with pytest.raises(review.ReviewError,match="UNSAFE_OUTPUT_PATH"):
            review.write_outputs(a,*evidence[:2],**args,json_path=tmp_path/bad)
    with pytest.raises(review.ReviewError,match="UNSAFE_OUTPUT_PATH"):
        review.write_outputs(a,*evidence[:2],**args,packet_path=tmp_path/"wrong.md")
    def interrupted(*args): raise OSError("interrupted")
    monkeypatch.setattr(review.os,"replace",interrupted)
    with pytest.raises(OSError): review.write_outputs(a,*evidence[:2],**args)
    assert [p.read_bytes() for p in paths]==before
    assert not list(paths[0].parent.glob(".step114-*"))


def test_staged_readback_failure_does_not_replace_outputs(evidence,tmp_path,monkeypatch):
    a=run(evidence)
    args=dict(provenance=evidence[2],repository_root=tmp_path)
    paths=review.write_outputs(a,*evidence[:2],**args)
    before=[p.read_bytes() for p in paths]
    original=Path.read_bytes
    def corrupted(p):
        return b"invalid" if p.name.startswith(".step114-") else original(p)
    monkeypatch.setattr(Path,"read_bytes",corrupted)
    with pytest.raises(review.ReviewError,match="INVALID_STAGED_READBACK"):
        review.write_outputs(a,*evidence[:2],**args)
    assert [p.read_bytes() for p in paths]==before
    assert not list(paths[0].parent.glob(".step114-*"))
