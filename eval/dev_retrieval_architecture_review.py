"""Step 114: post-hoc arithmetic over recorded development evidence only.

No search, checkpoint construction, inference, tuning, or held-out input occurs
here. The CLI binds the exact historical report; pure functions also accept
explicit trusted provenance for synthetic report tests.
"""

from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
for directory in ("src", "eval", "tests"):
    sys.path.insert(0, str(ROOT / directory))

from askpolicy.retrieval.rrf import reciprocal_rank_fusion
from askpolicy.storage.index_store import canonical, read_json, safe_path
from retrieval_metrics import METRIC_NAMES, macro_metrics, percentile, section_metrics

_spec = importlib.util.spec_from_file_location(
    "architecture_step112_validator", ROOT / "tests/local/real_dev_20_four_way_diagnostic.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
diagnostic = runner.diagnostic
MODES = diagnostic.MODES
BASELINE = "be9b5e31b52e03875a9e0592f594cfaa19393155"
SOURCE_SHA = "ad38924f4cd46f28c95d891060a135756a6add125377d97c0fdb867a94b8eec0"
SOURCE_PATH = runner.REPORT_PATH
JSON_PATH = Path("runtime/reports/step114_architecture_analysis.json")
PACKET_PATH = Path("runtime/reports/step114_owner_decision_packet.md")
APPROVAL_PATH = Path("eval/datasets/dev_annotation_approval.json")
EVALUATORS = ("dev_four_way_diagnostic.py", "dev_retrieval_diagnostic.py", "retrieval_metrics.py")
MISSING_GATES = [
    "Explicit owner quality/latency decision before production reranker activation (decisions §2).",
    "Separate approved production overflow policy and production integration/security acceptance tests.",
    "Any FAISS-only proposal requires separate architecture approval and contract/test review (§18).",
    "Frozen held-out four-way ablation and later full evaluation; development results do not close that gate.",
    "Generation, semantic citation support, refusal accuracy and end-to-end/API latency remain unmeasured here.",
    "Phase 5 acceptance and retrieval-readiness review remain open pending human review.",
]
LIMITATIONS = [
    "FR 6, EN 4, AR 4 positive cases: exploratory, no statistical language-superiority claim.",
    "20 warm observations per mode; fixed query/mode order; OS file cache not flushed.",
    "Offline CPU retrieval, not production API or end-to-end generation latency.",
    "Stage percentiles are not additive; helper aggregate overlaps its component timers.",
    "Rank arithmetic establishes movement, not lexical noise, semantic mismatch or language bias as causes.",
    "Non-gold means unlabelled as relevant for this case; it does not establish semantic irrelevance universally.",
    "Cross-Encoder raw logits are ranking scores, not probabilities; reasons for individual scores are unknown.",
    "Security is bounded historical report evidence, not a new runtime audit or universal security proof.",
    "Report hash authenticates this reviewed local artifact, not an independent attestation of the historical run.",
]


class ReviewError(ValueError):
    """Content-free rejection; never echo untrusted input or source passages."""


def _require(condition, code):
    if not condition:
        raise ReviewError(code)


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def trusted_provenance(root=ROOT):
    """Compare execution-byte hashes with both working and committed inputs."""
    paths = [runner.RUNNER_PATH, Path("uv.lock"), diagnostic.base.DATASET_PATH,
             APPROVAL_PATH, *(Path("eval") / name for name in EVALUATORS)]
    hashes = {}
    for path in paths:
        digest = file_digest(root / path)
        committed = subprocess.check_output(["git", "show", "HEAD:" + path.as_posix()], cwd=root)
        _require(digest == sha256(committed).hexdigest(), "WORKTREE_PROVENANCE_MISMATCH")
        hashes[path.as_posix()] = digest
    return dict(runner_source_sha256=hashes[runner.RUNNER_PATH.as_posix()],
                lockfile_sha256=hashes["uv.lock"],
                dataset_sha256=hashes[diagnostic.base.DATASET_PATH.as_posix()],
                approval_manifest_sha256=hashes[APPROVAL_PATH.as_posix()],
                evaluator_source_sha256={n: hashes["eval/" + n] for n in EVALUATORS})


def load_source(root=ROOT, *, expected_sha256=SOURCE_SHA):
    path = safe_path(root / SOURCE_PATH)
    _require(path.is_file(), "MISSING_SOURCE_REPORT")
    raw = path.read_bytes()
    _require(sha256(raw).hexdigest() == expected_sha256, "ALTERED_SOURCE_REPORT")
    try:
        return read_json(raw), sha256(raw).hexdigest()
    except (ValueError, UnicodeError):
        raise ReviewError("INVALID_SOURCE_JSON") from None


def validate_source(report, cases, provenance):
    """Reuse the immutable Step 112 validator, then check execution provenance."""
    try:
        runner.validate_benchmark_report(report, cases)
        core, measured = report["deterministic"], report["measurements"]
        session = measured["session"]
        _require(session["git_commit"] == runner.BASELINE, "WRONG_HISTORICAL_COMMIT")
        _require(all(session[k] == v for k, v in provenance.items()), "PROVENANCE_MISMATCH")
        _require(provenance["dataset_sha256"] == core["dataset_sha256"] == runner.DATASET_SHA,
                 "DATASET_MISMATCH")
        _require(core["execution_kind"] == "real_cpu_float32" and session["device"] == "cpu"
                 and session["dtype"] == "float32" and session["network_requests"] == 0,
                 "WRONG_EXECUTION_KIND")
        _require(core["snapshot"]["source_manifest_hash"] == core["corpus_manifest_fingerprint"],
                 "SNAPSHOT_BINDING_MISMATCH")
        ids = [c.query_id for c in cases]
        _require(session["benchmark_order"] == dict(query_ids=ids, modes=list(MODES)), "WRONG_RUN_ORDER")
        for mode in MODES:
            rows = core["modes"][mode]["cases"]
            recomputed = [section_metrics([_key(h) for h in r["ranking"]], _gold(r)) for r in rows]
            _require(all(r["status"] == "OK" for r in rows), "FAILED_SOURCE_CASE")
            _require(recomputed == [r["metrics"] for r in rows]
                     and macro_metrics(recomputed) == core["modes"][mode]["metrics"], "METRIC_MISMATCH")
            # Recompute pair evidence rather than trusting a zero-valued counter.
            by_id = {r["query_id"]: r for r in rows}
            for positive, negative in runner.PAIR_IDS:
                _require(not any(_key(h) in _gold(by_id[positive])
                                 for h in by_id[negative]["ranking"]), "UNAUTHORIZED_PAIR_EVIDENCE")
        for h, ce in zip(measured["modes"]["hybrid"]["per_case_seconds"],
                         measured["modes"]["hybrid_rerank"]["per_case_seconds"], strict=True):
            _require(h["seconds"] == ce["stages"]["hybrid_seconds"], "HYBRID_TIMING_MISMATCH")
        _require(sum(len(r["gold_sections"]) for r in core["modes"]["hybrid"]["cases"]) == 50,
                 "GOLD_COUNT_MISMATCH")
        _safe_content(report, cases)
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("INVALID_SOURCE_EVIDENCE") from None


def _safe_content(value, cases):
    diagnostic._no_sensitive_fields(value)
    def strings(item):
        if isinstance(item, dict):
            return [s for v in item.values() for s in strings(v)]
        if isinstance(item, (list, tuple)):
            return [s for v in item for s in strings(v)]
        return [item] if isinstance(item, str) else []
    values = strings(value)
    _require(not any(re.search(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", v)
                     for v in values), "CREDENTIAL_PAYLOAD")
    _require(not any(c.query in v or any(f in v for f in c.required_facts)
                     for c in cases for v in values), "ANNOTATION_PAYLOAD")


def _key(hit):
    return hit["document_id"], hit["parent_id"]


def _gold(row):
    return {_key(g): g["relevance_grade"] for g in row["gold_sections"]}


def _language(document_id):
    return document_id.rsplit("-", 2)[-2].lower()


def _family(document_id):
    return document_id.rsplit("-", 2)[0]


def _delta(a, b):
    return None if a is None or b is None else {m: b[m] - a[m] for m in METRIC_NAMES}


def _outcome(delta):
    if all(v == 0 for v in delta.values()):
        return "tied"
    if all(v >= 0 for v in delta.values()):
        return "improved"
    if all(v <= 0 for v in delta.values()):
        return "worsened"
    return "mixed"


def section_accounting(row):
    gold, ranked = _gold(row), row["ranking"]
    sections = [_key(h) for h in ranked]
    counts = Counter(sections)
    return dict(
        first_relevant_child_rank=next((h["rank"] for h in ranked if _key(h) in gold), None),
        gold_top5=[g for g in row["gold_sections"] if _key(g) in sections[:5]],
        gold_top10=[g for g in row["gold_sections"] if _key(g) in sections[:10]],
        gold_absent=[g for g in row["gold_sections"] if _key(g) not in sections],
        duplicate_child_ids=len(ranked)-len({h["child_id"] for h in ranked}),
        repeated_parent_sections=[dict(document_id=d, parent_id=p, child_count=n,
            ranks=[h["rank"] for h in ranked if _key(h) == (d,p)], relevance_grade=gold.get((d,p),0))
            for (d,p),n in sorted(counts.items()) if n > 1],
        top5_repeated_section_slots=len(sections[:5])-len(set(sections[:5])),
        top10_repeated_section_slots=len(sections[:10])-len(set(sections[:10])),
        cross_language_gold_top5=[g for g in row["gold_sections"]
                                 if _key(g) in sections[:5] and _language(g["document_id"]) != row["language"]],
        language_versions_by_family={family:sorted({_language(h["document_id"]) for h in ranked
                                      if _family(h["document_id"]) == family})
                                    for family in sorted({_family(h["document_id"]) for h in ranked})},
    )


def reconstruct_rrf(bm25, faiss, hybrid):
    """Record arithmetic for the whole source union, including top-20 exclusions."""
    fused = reciprocal_rank_fusion([h["child_id"] for h in bm25["ranking"]],
                                   [h["child_id"] for h in faiss["ranking"]])
    _require([(h.child_id,h.score) for h in fused] ==
             [(h["child_id"],h["raw_score"]) for h in hybrid["ranking"]], "RRF_RECONSTRUCTION_MISMATCH")
    sources = {}
    for mode, row in (("bm25", bm25), ("faiss", faiss)):
        for hit in row["ranking"]:
            entry = sources.setdefault(hit["child_id"], dict(child_id=hit["child_id"],
                document_id=hit["document_id"],parent_id=hit["parent_id"],bm25_rank=None,faiss_rank=None))
            _require(_key(entry) == _key(hit), "CHILD_IDENTITY_CONFLICT")
            entry[mode+"_rank"] = hit["rank"]
    hybrid_ranks = {h["child_id"]:h["rank"] for h in hybrid["ranking"]}
    for hit in hybrid["ranking"]:
        _require(_key(hit) == _key(sources[hit["child_id"]]), "CHILD_IDENTITY_CONFLICT")
    gold = _gold(hybrid)
    for child, entry in sources.items():
        for mode in ("bm25", "faiss"):
            rank = entry[mode+"_rank"]
            entry[mode+"_contribution"] = 0.0 if rank is None else 1/(60+rank)
        entry["rrf_score"] = entry["bm25_contribution"] + entry["faiss_contribution"]
        entry["hybrid_rank"] = hybrid_ranks.get(child)
        entry["source_systems"] = [m for m in ("bm25","faiss") if entry[m+"_rank"] is not None]
        entry["hybrid_minus_faiss_rank"] = (None if entry["faiss_rank"] is None or child not in hybrid_ranks
                                                else hybrid_ranks[child]-entry["faiss_rank"])
        entry["relevance_grade"] = gold.get(_key(entry),0)
    return dict(query_id=hybrid["query_id"], exact_match=True,
                order="score_descending_child_id_ascending", k=60,
                bm25_input_count=len(bm25["ranking"]),faiss_input_count=len(faiss["ranking"]),
                hybrid_count=len(fused), candidates=sorted(sources.values(),key=lambda h:(-h["rrf_score"],h["child_id"])))


def _stats(values):
    return dict(sample_count=len(values),total_seconds=sum(values),
                p50_seconds=percentile(values,50),p95_seconds=percentile(values,95))


def _english(before, after):
    gold = _gold(before)
    original = next(h for h in before["ranking"] if _key(h) in gold)
    ranks = {h["child_id"]:h["rank"] for h in after["ranking"]}
    logits = {h["child_id"]:h["raw_score"] for h in after["ranking"]}
    def annotated(rows):
        return [dict(**h,relevance_grade=gold.get(_key(h),0),
                     recorded_ce_raw_logit=logits[h["child_id"]]) for h in rows[:10]]
    ahead = []
    for hit in after["ranking"][:ranks[original["child_id"]]-1]:
        if _key(hit) in gold:
            continue
        same_family = _family(hit["document_id"]) == _family(original["document_id"])
        ahead.append(dict(**hit,relevance_grade=0,
            different_language=_language(hit["document_id"]) != before["language"],
            other_policy_family=not same_family,
            same_family_non_gold_section=same_family,
            duplicate_section_child=sum(_key(h)==_key(hit) for h in after["ranking"])>1,
            interpretation="ID/label classification only; semantic relatedness and logit cause not established"))
    return dict(query_id=before["query_id"],language=before["language"],gold_sections=before["gold_sections"],
        hybrid_top10=annotated(before["ranking"]),reranked_top10=annotated(after["ranking"]),
        before_first_relevant_rank=section_accounting(before)["first_relevant_child_rank"],
        after_first_relevant_rank=section_accounting(after)["first_relevant_child_rank"],
        differences=_delta(before["metrics"],after["metrics"]),before_metrics=before["metrics"],
        after_metrics=after["metrics"],non_gold_above_previous_first_relevant=ahead)


def analyze(report, cases, *, source_sha256, current_commit, provenance):
    validate_source(report,cases,provenance)
    core, measured = report["deterministic"],report["measurements"]
    rows = {mode:{r["query_id"]:r for r in core["modes"][mode]["cases"]} for mode in MODES}
    quality = {m:macro_metrics([section_metrics([_key(h) for h in r["ranking"]],_gold(r))
                              for r in rows[m].values()]) for m in MODES}
    rrf, comparisons, english, ceilings = [],[],[],[]
    case_map = {c.query_id:c for c in cases}
    for qid, h in rows["hybrid"].items():
        f,b,ce = (rows[m][qid] for m in ("faiss","bm25","hybrid_rerank"))
        reconstruction = reconstruct_rrf(b,f,h)
        rrf.append(reconstruction)
        if not h["gold_sections"]:
            continue
        accounting = {m:section_accounting(rows[m][qid]) for m in MODES}
        relevant = [c for c in reconstruction["candidates"] if c["relevance_grade"] > 0]
        categories = dict(
            A_faiss_relevant_child_absent_hybrid=any(c["faiss_rank"] is not None and c["hybrid_rank"] is None for c in relevant),
            B_relevant_child_demoted=any((c["hybrid_minus_faiss_rank"] or 0)>0 for c in relevant),
            C_relevant_parent_multiple_children=any(p["relevance_grade"]>0 for m in ("faiss","hybrid")
                for p in accounting[m]["repeated_parent_sections"]),
            D_multiple_language_versions=any(len(langs)>1 for m in ("faiss","hybrid")
                for langs in accounting[m]["language_versions_by_family"].values()),
            E_more_than_five_gold=len(h["gold_sections"])>5)
        comparisons.append(dict(query_id=qid,language=h["language"],gold_section_count=len(h["gold_sections"]),
            metrics={m:rows[m][qid]["metrics"] for m in MODES},accounting=accounting,
            hybrid_minus_faiss=_delta(f["metrics"],h["metrics"]),
            rerank_minus_hybrid=_delta(h["metrics"],ce["metrics"]),
            relevant_child_movements=relevant,categories=categories,
            approved_source_labels=[g.model_dump() for g in case_map[qid].authorized_source_sections]))
        if qid in ("dev-candidate-009","dev-candidate-010"):
            english.append(_english(h,ce))
        n = len(h["gold_sections"])
        if n>5:
            ceiling=5/n
            ceilings.append(dict(query_id=qid,gold_section_count=n,theoretical_recall_at_5_ceiling=ceiling,
                modes={m:dict(actual_recall_at_5=rows[m][qid]["metrics"]["recall_at_5"],
                              gap_to_ceiling=ceiling-rows[m][qid]["metrics"]["recall_at_5"],
                              repeated_top5_slots=accounting[m]["top5_repeated_section_slots"],
                              cross_language_gold_top5=accounting[m]["cross_language_gold_top5"]) for m in MODES}))
    languages={}
    for language in ("fr","en","ar"):
        selected=[c for c in comparisons if c["language"]==language]
        languages[language]=dict(positive_count=len(selected),
            quality={m:macro_metrics([r["metrics"] for r in rows[m].values() if r["language"]==language]) for m in MODES},
            comparison_outcomes={name:dict(Counter(_outcome(c[name]) for c in selected))
                                 for name in ("hybrid_minus_faiss","rerank_minus_hybrid")},
            metric_outcomes={name:{metric:dict(Counter("improved" if c[name][metric]>0 else
                "worsened" if c[name][metric]<0 else "tied" for c in selected)) for metric in METRIC_NAMES}
                for name in ("hybrid_minus_faiss","rerank_minus_hybrid")},
            many_gold_query_ids=[c["query_id"] for c in selected if c["gold_section_count"]>5],
            cross_language_top5_gold_matches={m:sum(len(c["accounting"][m]["cross_language_gold_top5"]) for c in selected) for m in MODES},
            repeated_top5_section_slots={m:sum(c["accounting"][m]["top5_repeated_section_slots"] for c in selected) for m in MODES})
    latency={m:_stats([r["seconds"] for r in measured["modes"][m]["per_case_seconds"]]) for m in MODES}
    overhead={}
    for a,b in (("faiss","hybrid"),("hybrid","hybrid_rerank"),("faiss","hybrid_rerank")):
        values=[y["seconds"]-x["seconds"] for x,y in zip(measured["modes"][a]["per_case_seconds"],
                measured["modes"][b]["per_case_seconds"],strict=True)]
        overhead[b+"_minus_"+a]=dict(**_stats(values),per_case_seconds=[dict(query_id=c.query_id,seconds=v)
            for c,v in zip(cases,values,strict=True)],total_latency_ratio=latency[b]["total_seconds"]/latency[a]["total_seconds"])
    summary=runner.derive_summary(report)
    options=[]
    for option,mode,disposition,strength,regression,approval in (
        ("A","hybrid","Retain current non-reranked Hybrid RRF baseline",
         "Preserves approved lexical+dense architecture and conditional reranker gate",
         "Lower development quality than FAISS on all three macro metrics",
         "Human review of this packet; no production behavior change in this task"),
        ("B","hybrid_rerank","Consider future separately approved Cross-Encoder activation",
         "Measured development Recall@5 gain versus Hybrid; per-language outcome counts below",
         "Two first-relevant-rank regressions; lower MRR; substantial CPU overhead",
         "Separate owner quality/latency and overflow approvals, API integration and security tests"),
        ("C","faiss","Consider future FAISS-only architecture proposal",
         "Highest measured development MRR/nDCG, faster than Hybrid",
         "Lower Recall@5 than CE; removes approved lexical fusion from deployed path",
         "Separate §18 architecture proposal, rationale, acceptance/test diff and explicit approval")):
        options.append(dict(option=option,mode=mode,disposition=disposition,quality=quality[mode],latency=latency[mode],
            strengths=strength,regressions=regression,security="Zero unauthorized hits in recorded execution; no new runtime audit",
            uncertainty="Small development set, fixed timing order; no held-out/generalization proof",required_approval_or_tests=approval))
    result=dict(analysis_kind="development_architecture_decision_review",decision_status="AWAITING_OWNER_APPROVAL",
        source_report_sha256=source_sha256,historical_execution_commit=measured["session"]["git_commit"],
        current_analysis_commit=current_commit,verified_provenance=provenance,
        dataset_sha256=core["dataset_sha256"],snapshot=core["snapshot"],published_release_id=core["published_release_id"],
        published_manifest_sha256=core["published_manifest_sha256"],models=core["models"],
        source_validation="STEP112_VALIDATOR_AND_INDEPENDENT_ARITHMETIC_PASSED",
        quality=quality,macro_differences={b+"_minus_"+a:_delta(quality[a],quality[b])
            for a,b in (("faiss","hybrid"),("hybrid","hybrid_rerank"),("faiss","hybrid_rerank"))},
        positive_metric_denominator=14,no_positive_count=6,security_execution_count=20,gold_reference_count=50,
        no_positive_query_ids=[c.query_id for c in cases if not c.authorized_source_sections],
        per_query_comparisons=comparisons,rrf_reconstruction=rrf,english_regressions=english,
        category_occurrences={key:[c["query_id"] for c in comparisons if c["categories"][key]]
                              for key in comparisons[0]["categories"]},
        language_breakdowns=languages,gold_ceilings=ceilings,latency=latency,paired_latency_overhead=overhead,
        fourth_mode_stages=summary["fourth_mode_stages"],hardware=measured["hardware"],memory=measured["session"]["memory"],
        security=dict(unauthorized_hit_counts=summary["unauthorized_hit_counts"],role_pairs=summary["role_pairs"],
                      runtime_authorization_reaudited=False),
        decision_options=options,recommendation="DO_NOT_ACTIVATE_CROSS_ENCODER; retain current Hybrid RRF and all four evaluation configurations",
        missing_contractual_gates=MISSING_GATES,limitations=LIMITATIONS,
        production_reranker_enabled=False,retrieval_behavior_changed=False,full_ablation_complete=False,
        held_out_evaluated_count=0,owner_decision_made=False)
    _safe_content(result,cases)
    return result


def _table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join("---" for _ in headers)+" |",
                      *("| "+" | ".join(str(v) for v in row)+" |" for row in rows)])


def render_packet(a):
    """Private owner packet: identifiers and annotations, never source bodies."""
    def block(value):
        return "\n```json\n"+json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n```\n"
    def identity(h):
        return h["document_id"]+" / "+h["parent_id"]+" / "+h["child_id"]
    comparisons=a["per_query_comparisons"]
    case_table=_table(["Case","Language","Gold N","First FAISS→H","Gold top5 F→H","Gold top10 F→H",
                       "Absent F→H","ΔRecall","ΔMRR","ΔnDCG"],[
        [c["query_id"],c["language"],c["gold_section_count"],
         str(c["accounting"]["faiss"]["first_relevant_child_rank"])+"→"+str(c["accounting"]["hybrid"]["first_relevant_child_rank"]),
         *[str(len(c["accounting"]["faiss"][key]))+"→"+str(len(c["accounting"]["hybrid"][key]))
           for key in ("gold_top5","gold_top10","gold_absent")],
         *[c["hybrid_minus_faiss"][metric] for metric in METRIC_NAMES]] for c in comparisons])
    movements=[]
    for c in comparisons:
        changed=[h for h in c["relevant_child_movements"]
                 if h["faiss_rank"] != h["hybrid_rank"]]
        movements.append("### "+c["query_id"]+" — relevant child movements\n"+
            (_table(["Document / parent / child","BM25 rank","FAISS rank","H rank","BM25 contribution",
                     "FAISS contribution","RRF score","H−F rank","Systems"],[[identity(h),h["bm25_rank"],
                    h["faiss_rank"],h["hybrid_rank"],h["bm25_contribution"],h["faiss_contribution"],h["rrf_score"],
                    h["hybrid_minus_faiss_rank"],", ".join(h["source_systems"])] for h in changed])
             if changed else "No relevant-child rank movement.")+"\n\n"+
            "Duplicate parent sections and their exact child positions (zero duplicate child IDs):"+
            block({m:c["accounting"][m]["repeated_parent_sections"] for m in ("faiss","hybrid")})+
            "Gold section membership (distinct parent identities):"+
            block({m:{k:c["accounting"][m][k] for k in ("gold_top5","gold_top10","gold_absent")}
                   for m in ("faiss","hybrid")}))
    english=[]
    for e in a["english_regressions"]:
        text="### "+e["query_id"]+"\n"+block({k:e[k] for k in (
            "before_first_relevant_rank","after_first_relevant_rank","before_metrics","after_metrics","differences","gold_sections")})
        for label,key in (("Hybrid top 10","hybrid_top10"),("Cross-Encoder top 10","reranked_top10")):
            text+="\n"+label+"\n\n"+_table(["Rank","Document / parent / child","Gold grade","RRF/logit raw score","Recorded CE logit"],
                [[h["rank"],identity(h),h["relevance_grade"],h["raw_score"],h["recorded_ce_raw_logit"]] for h in e[key]])+"\n"
        english.append(text+"\nNon-gold promotions above the previous first relevant child:"+
                       block(e["non_gold_above_previous_first_relevant"]))
    language_table=_table(["Language","Positive N","Mode",*METRIC_NAMES],[
        [lang,group["positive_count"],m,*[group["quality"][m][k] for k in METRIC_NAMES]]
        for lang,group in a["language_breakdowns"].items() for m in MODES])
    ceiling_table=_table(["Case","Gold N","Recall ceiling",*MODES],[
        [c["query_id"],c["gold_section_count"],c["theoretical_recall_at_5_ceiling"],
         *[str(c["modes"][m]["actual_recall_at_5"])+" (gap "+str(c["modes"][m]["gap_to_ceiling"])+")" for m in MODES]]
        for c in a["gold_ceilings"]])
    parts=["# Step 114 — Development retrieval architecture decision packet",
        "## 1. Executive findings\nRecommendation only. AWAITING_OWNER_APPROVAL. No production behavior change. "
        "FAISS leads MRR/nDCG; CE leads Recall@5 but regresses two English first-relevant ranks and adds measured CPU cost.",
        "## 2. Exact evidence/provenance"+block({k:a[k] for k in ("source_report_sha256","historical_execution_commit",
        "current_analysis_commit","verified_provenance","dataset_sha256","snapshot","published_release_id","published_manifest_sha256","models","source_validation")}),
        "## 3. Four-mode quality\n14 positive executions; six no-positive executions excluded. All 50 approved gold references retained.\n"+
        _table(["Mode",*METRIC_NAMES],[[m,*[a["quality"][m][k] for k in METRIC_NAMES]] for m in MODES])+block(a["macro_differences"]),
        "## 4. Four-mode latency\nSeconds; 20 observations per mode.\n"+
        _table(["Mode","p50","p95","Total"],[[m,*[a["latency"][m][k] for k in ("p50_seconds","p95_seconds","total_seconds")]] for m in MODES])+
        block(dict(paired_overhead=a["paired_latency_overhead"],stages=a["fourth_mode_stages"],hardware=a["hardware"],memory=a["memory"])),
        "## 5. FAISS vs Hybrid root-cause evidence and limitations\nAll 20 RRF rankings match the pinned k=60, input 30/30, output 20 implementation, including child-ID tie breaking. "
        "Positive rank deltas mean demotion. Missing Hybrid rank means outside its top 20. Duplicate child IDs are rejected; repeated parent sections consume positions and gain once. "
        "Different language counterparts are distinct approved sections, not duplicate parent IDs. Arithmetic is established; semantic causes remain unproven.\n"+
        case_table+"\n\n"+block(a["category_occurrences"])+"\n\n"+"\n\n".join(movements)+
        "\n\nThe companion step114_architecture_analysis.json contains the complete ranked source union, "
        "including non-gold promotions, one-system versus two-system candidates, tie ordering, approved section labels "
        "and every reconstructed contribution for all 20 executions.",
        "## 6. Detailed cases 009/010\nComplete top-10 identities before/after; grades zero denote non-gold for this case. "
        "Raw CE logits are copied by child identity onto both lists for comparison; Hybrid raw_score remains RRF. "
        "ID-based same-family classification does not prove semantic relatedness.\n"+"\n\n".join(english),
        "## 7. FR/EN/AR analysis\nFR 6 positives, EN 4, AR 4. No statistical superiority inference. "
        "Outcomes are Pareto counts across all three metrics; mixed outcomes are separate.\n"+language_table+
        block({lang:{k:v for k,v in group.items() if k != "quality"} for lang,group in a["language_breakdowns"].items()}),
        "## 8. Gold-set metric ceilings\nOfficial Recall@5 is unchanged. Max=min(5,N)/N; multilingual gold is retained, "
        "and repeated children consume slots. No alternative contractual metric is introduced.\n"+ceiling_table+
        block(a["gold_ceilings"]),
        "## 9. Reranker quality/latency trade-off\n"+
        _table(["Option","Disposition","Strengths","Regressions","Security","Uncertainty","Approval/tests"],
               [[o[k] for k in ("option","disposition","strengths","regressions","security","uncertainty","required_approval_or_tests")]
                for o in a["decision_options"]]),
        "## 10. Security findings\nHistorical bounded evidence includes all 20 executions and three pairs in each mode. "
        "No new authentication, ACL, snapshot or production security certification is inferred.\n"+block(a["security"]),
        "## 11. Recommendation\n"+a["recommendation"]+". This packet makes no owner decision and enables no reranker.",
        "## 12. Explicit owner decisions still required\nReview option A as the interim disposition. "
        "Option B requires separate quality/latency, production overflow and integration approval. "
        "Option C requires a separate §18 architecture decision and contract/test review. Preserve all four ablation configurations.",
        "## 13. Unresolved limitations and later acceptance gates\n"+
        "\n".join("- "+s for s in a["limitations"]+a["missing_contractual_gates"])]
    return "\n\n".join(parts)+"\n"


def write_outputs(analysis, report, cases, *, provenance, repository_root=ROOT, json_path=None, packet_path=None):
    """Validate derivation, stage and verify both artifacts, then atomically replace each."""
    expected=analyze(report,cases,source_sha256=analysis["source_report_sha256"],
                     current_commit=analysis["current_analysis_commit"],provenance=provenance)
    _require(canonical(analysis)==canonical(expected), "INVALID_ANALYSIS_DERIVATION")
    targets=[safe_path(repository_root/JSON_PATH),safe_path(repository_root/PACKET_PATH)]
    for requested,target in zip((json_path,packet_path),targets,strict=True):
        _require(requested is None or safe_path(requested)==target,"UNSAFE_OUTPUT_PATH")
    payloads=[canonical(analysis)+b"\n",render_packet(analysis).encode("utf-8")]
    _require(read_json(payloads[0])==json.loads(canonical(expected)),"INVALID_JSON_READBACK")
    temporary=[]
    try:
        for target,payload in zip(targets,payloads,strict=True):
            target.parent.mkdir(parents=True,exist_ok=True)
            staging=target.parent/(".step114-"+uuid4().hex)
            temporary.append(staging)
            with staging.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _require(staging.read_bytes()==payload,"INVALID_STAGED_READBACK")
        for staging,target,payload in zip(temporary,targets,payloads,strict=True):
            os.replace(staging,target)
            _require(target.read_bytes()==payload,"INVALID_OUTPUT_READBACK")
    finally:
        for staging in temporary:
            staging.unlink(missing_ok=True)
    return tuple(targets)


def main():
    _require(subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==BASELINE,
             "WRONG_ANALYSIS_BASELINE")
    report,digest=load_source()
    _,_,cases=runner.read_inputs()
    provenance=trusted_provenance()
    analysis=analyze(report,cases,source_sha256=digest,current_commit=BASELINE,provenance=provenance)
    paths=write_outputs(analysis,report,cases,provenance=provenance)
    print(json.dumps(dict(source_sha256=digest,rrf_matches=len(analysis["rrf_reconstruction"]),
        positive_cases=len(analysis["per_query_comparisons"]),decision_status=analysis["decision_status"],
        outputs=[str(p) for p in paths]),sort_keys=True))


if __name__ == "__main__":
    main()
