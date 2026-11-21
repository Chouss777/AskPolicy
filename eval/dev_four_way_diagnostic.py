"""Exploratory four-mode development comparison, never a release ablation."""

from collections import Counter
from contextlib import closing
from dataclasses import asdict
from datetime import date
from hashlib import sha256
from math import isfinite, isclose
import os
from pathlib import Path
import sqlite3
from time import perf_counter
from uuid import uuid4

from askpolicy.retrieval import role_bm25, role_faiss, hybrid
from askpolicy.retrieval.cross_encoder import MODEL_ID, MODEL_REVISION, ARTIFACT_FINGERPRINT
from askpolicy.retrieval.evidence_hydration import _verified_bytes
from askpolicy.retrieval.offline_reranking import rerank_published_hits, OfflineRerankError, _check_encoder
from askpolicy.security.index_selection import select_role_index
from askpolicy.storage.index_store import PublishedSnapshot, canonical, safe_path

import dev_retrieval_diagnostic as base
from retrieval_metrics import METRIC_NAMES, macro_metrics, percentile, section_metrics

MODES = ("bm25", "faiss", "hybrid", "hybrid_rerank")
REPORT_PATH = Path("runtime/reports/step107_dev_four_way_diagnostic.json")
DAY = base.DAY
CORE_FIELDS = set("""evaluation_kind status required_development_count required_held_out_count
held_out_evaluated_count evaluated_count annotation_gap positive_metric_denominator no_positive_count
question_family_count roles languages snapshot published_release_id published_manifest_sha256
corpus_manifest_fingerprint source_manifest_sha256 dataset_sha256 validity_date execution_kind models
configurations modes comparisons reranker_failures unauthorized_hit_count production_reranker_enabled
full_ablation_complete validation_status""".split())
ROW_FIELDS = set("""query_id question_family_id active_role language case_type gold_sections gold_ranks
status failure_code metrics exclusion_reason ranking unauthorized_hit_count""".split())


class FourWayDiagnosticError(ValueError):
    """Fixed diagnostic failure without protected source exception details."""


def _row(case, ranking, gold, status="OK"):
    sections = [(h["document_id"], h["parent_id"]) for h in ranking]
    return dict(
        query_id=case.query_id, question_family_id=case.question_family_id,
        active_role=case.active_role, language=case.language, case_type=case.case_type,
        gold_sections=[dict(document_id=d, parent_id=p, relevance_grade=g) for (d,p),g in sorted(gold.items())],
        gold_ranks=[dict(document_id=d, parent_id=p,
                        first_child_rank=next((i for i,s in enumerate(sections,1) if s == (d,p)),None))
                    for d,p in sorted(gold)],
        status=status, failure_code=None if status == "OK" else "TOKEN_LIMIT",
        metrics=section_metrics(sections,gold) if status == "OK" else None,
        exclusion_reason=("TOKEN_LIMIT" if status != "OK" else None if gold else "NO_POSITIVE_GOLD"),
        ranking=ranking, unauthorized_hit_count=0,
    )


def _ranking(hits, case, context, handle, mode):
    rows = base._checked_ranking(hits, case, context, handle.bm25,
                                 handle.bm25.snapshot.candidate_snapshot_id,
                                 "hybrid" if mode in ("hybrid", "hybrid_rerank") else mode)
    for row, hit in zip(rows, hits, strict=True):
        score = hit.rrf_score if mode == "hybrid" else hit.score
        if type(score) is not float or not isfinite(score):
            raise ValueError
        row.update(raw_score=score, token_count=hit.token_count if mode == "hybrid_rerank" else None)
    return rows


def _deltas(modes):
    hybrid = modes["hybrid"]["cases"]
    reranked = modes["hybrid_rerank"]["cases"]
    paired = []
    for before, after in zip(hybrid,reranked,strict=True):
        paired.append(dict(query_id=before["query_id"], differences={
            name:after["metrics"][name]-before["metrics"][name] for name in METRIC_NAMES
        } if before["metrics"] is not None and after["metrics"] is not None else None))
    result = {}
    for mode in ("bm25","faiss","hybrid"):
        left, right = modes[mode]["metrics"], modes["hybrid_rerank"]["metrics"]
        result[mode] = ({name:right[name]-left[name] for name in METRIC_NAMES}
                        if left is not None and right is not None else None)
    return dict(paired_hybrid_differences=paired, rerank_minus_baseline_macro=result)


def _timing(rows):
    seconds = [r["seconds"] for r in rows]
    return dict(sample_count=len(rows), per_case_seconds=rows,
                p50_seconds=percentile(seconds,50), p95_seconds=percentile(seconds,95),
                percentile_method="linear-(n-1)", state="warm_after_explicit_model_warmups",
                interpretation="exploratory_five_query_samples_not_production_api_latency")


def _evaluate(handle, *, release_root, manifest, raw_dataset, context_for_role, encoder,
              validity_date, execution_kind):
    if (type(handle) is not PublishedSnapshot or type(validity_date) is not date or validity_date != DAY
        or execution_kind not in ("fixture_embeddings_and_logits","real_cpu_float32")):
        raise ValueError
    _check_encoder(encoder)
    cases = base.read_cases(raw_dataset)
    contexts = {c.query_id:context_for_role(c.active_role) for c in cases}
    for case in cases:
        if select_role_index(contexts[case.query_id]).role != case.active_role:
            raise ValueError
        hybrid._gate(handle.bm25,handle.faiss,contexts[case.query_id],
                     handle.bm25.snapshot.candidate_snapshot_id,validity_date)
    # Gold resolution uses the exact checksum-verified release image and the
    # approved parser/section resolver. No live draft or held-out file is read.
    with closing(sqlite3.connect(":memory:")) as image:
        image.deserialize(_verified_bytes(handle,release_root))
        image.execute("PRAGMA query_only = ON")
        gold = base._gold(image,cases,handle.bm25,contexts)
    # Validate the supplied canonical manifest against the handle without
    # trusting a caller-provided fingerprint string.
    from askpolicy.storage.snapshot import _manifest_documents, _digest
    documents = _manifest_documents(manifest)
    fingerprint = _digest("manifest/v1", {"documents":[
        {"source_path":d.source_path,"content_sha256":d.content_sha256,**d.metadata} for d in documents.values()
    ]})
    if fingerprint != handle.bm25.snapshot.source_manifest_hash:
        raise ValueError
    tick = perf_counter()
    handle.faiss._encoder.encode_queries([cases[0].query])
    e5_warmup = perf_counter()-tick
    tick = perf_counter()
    encoder.score_pairs([("warmup","warmup")])
    cross_warmup = perf_counter()-tick
    rows = {mode:[] for mode in MODES}
    times = {mode:[] for mode in MODES}
    for case in cases:
        context = contexts[case.query_id]
        options = dict(query=case.query, expected_snapshot_id=handle.bm25.snapshot.candidate_snapshot_id,
                       validity_date=validity_date)
        for mode, search, bundle in (("bm25",role_bm25.search_role_bm25,handle.bm25),
                                     ("faiss",role_faiss.search_role_faiss,handle.faiss)):
            tick = perf_counter()
            hits = search(bundle,context,**options)
            seconds = perf_counter()-tick
            rows[mode].append(_row(case,_ranking(hits,case,context,handle,mode),gold[case.query_id]))
            times[mode].append(dict(query_id=case.query_id,seconds=seconds))
        hybrid_started = perf_counter()
        original = handle.search(context,query=case.query,validity_date=validity_date)
        hybrid_seconds = perf_counter()-hybrid_started
        # Exactly this tuple is the fourth mode's sole candidate input.
        failed = False
        try:
            reranked = rerank_published_hits(handle,context,case.query,original,release_root=release_root,
                                             validity_date=validity_date,encoder=encoder)
        except OfflineRerankError as error:
            if error.code != "TOKEN_LIMIT":
                raise
            failed = True
        end_to_end = perf_counter()-hybrid_started
        rows["hybrid"].append(_row(case,_ranking(original,case,context,handle,"hybrid"),gold[case.query_id]))
        times["hybrid"].append(dict(query_id=case.query_id,seconds=hybrid_seconds))
        if failed:
            rows["hybrid_rerank"].append(_row(case,[],gold[case.query_id],"FAILED_TOKEN_LIMIT"))
            times["hybrid_rerank"].append(dict(query_id=case.query_id,seconds=end_to_end,
                stages=None, failure_code="TOKEN_LIMIT"))
        else:
            ranking = _ranking(reranked.hits,case,context,handle,"hybrid_rerank")
            if {h.child.child_id for h in original} != {r["child_id"] for r in ranking}:
                raise ValueError
            rows["hybrid_rerank"].append(_row(case,ranking,gold[case.query_id]))
            stages = asdict(reranked.timings)
            stages["hybrid_seconds"] = hybrid_seconds
            stages["other_seconds"] = end_to_end - hybrid_seconds - stages["total_seconds"]
            stages["helper_total_seconds"] = stages.pop("total_seconds")
            times["hybrid_rerank"].append(dict(query_id=case.query_id,seconds=end_to_end,
                                              stages=stages,failure_code=None))
    dataset_hash = sha256(raw_dataset).hexdigest()
    modes = {}
    positive_count = sum(bool(g) for g in gold.values())
    for mode in MODES:
        completed = sum(bool(r["gold_sections"]) and r["status"] == "OK" for r in rows[mode])
        modes[mode] = dict(
            snapshot_id=handle.bm25.snapshot.candidate_snapshot_id,published_release_id=handle.release_id,
            dataset_sha256=dataset_hash,corpus_manifest_fingerprint=fingerprint,
            cases=rows[mode],
            metrics=macro_metrics([r["metrics"] for r in rows[mode]]) if completed == positive_count else None,
            positive_metric_denominator=positive_count,completed_positive_count=completed,
            returned_hit_count=sum(len(r["ranking"]) for r in rows[mode]),
            unauthorized_hit_count=0,violating_query_count=0,
            role_pairs=base._role_pairs(cases,gold,rows[mode]),
        )
    core = dict(
        evaluation_kind="development_four_way_diagnostic",status="INCOMPLETE_DEVELOPMENT_COVERAGE",
        validation_status="VALIDATED_OFFLINE_DIAGNOSTIC",
        required_development_count=20,required_held_out_count=60,held_out_evaluated_count=0,
        evaluated_count=len(cases),annotation_gap=20-len(cases),positive_metric_denominator=positive_count,
        no_positive_count=len(cases)-positive_count,question_family_count=len({c.question_family_id for c in cases}),
        roles=dict(Counter(c.active_role for c in cases)),languages=dict(Counter(c.language for c in cases)),
        snapshot=asdict(handle.bm25.snapshot),published_release_id=handle.release_id,
        published_manifest_sha256=handle.manifest_hash,corpus_manifest_fingerprint=fingerprint,
        source_manifest_sha256=sha256(manifest).hexdigest(),dataset_sha256=dataset_hash,
        validity_date=validity_date.isoformat(),execution_kind=execution_kind,
        models=dict(e5=dict(model_id=handle.faiss._encoder.model_id,revision=handle.faiss._encoder.revision,
                            artifact_fingerprint=handle.faiss._encoder.artifact_fingerprint),
                    cross_encoder=dict(model_id=MODEL_ID,revision=MODEL_REVISION,artifact_fingerprint=ARTIFACT_FINGERPRINT)),
        configurations=dict(bm25=asdict(handle.bm25.indexes[0].metadata.configuration),
                            faiss=asdict(handle.faiss.indexes[0].metadata.configuration),
                            hybrid=dict(rrf_constant=60,bm25_top_k=30,faiss_top_k=30,top_k=20),
                            hybrid_rerank=dict(input_top_k=20,joint_token_limit=512,overflow="reject_entire_request",
                                               order=["raw_logit_descending","child_id_ascending"])),
        modes=modes,comparisons=_deltas(modes),
        reranker_failures=sum(r["status"] != "OK" for r in rows["hybrid_rerank"]),
        unauthorized_hit_count=0,production_reranker_enabled=False,full_ablation_complete=False,
    )
    report = dict(deterministic=core,identity_sha256=sha256(canonical(core)).hexdigest(),
                  measurements=dict(hardware=base.hardware(),e5_warmup_seconds=e5_warmup,
                      cross_encoder_warmup_seconds=cross_warmup,
                      warmup_pair_count=1,modes={m:_timing(times[m]) for m in MODES},
                      scoring_timer_scope="score_pairs includes repeated token validation and padding",
                      api_overhead_measured=False))
    validate_report(report)
    return report


def evaluate(handle: PublishedSnapshot, *, release_root: Path, manifest: bytes, raw_dataset: bytes,
             context_for_role, encoder, validity_date: date=DAY, execution_kind: str):
    try:
        return _evaluate(handle,release_root=release_root,manifest=manifest,raw_dataset=raw_dataset,
                         context_for_role=context_for_role,encoder=encoder,validity_date=validity_date,
                         execution_kind=execution_kind)
    except Exception:
        pass
    raise FourWayDiagnosticError("FOUR_WAY_DIAGNOSTIC_FAILED")


def validate_report(report):
    try:
        core = report["deterministic"]
        if (set(report) != {"deterministic","identity_sha256","measurements"} or set(core) != CORE_FIELDS
            or core["evaluation_kind"] != "development_four_way_diagnostic"
            or core["status"] != "INCOMPLETE_DEVELOPMENT_COVERAGE"
            or core["validation_status"] != "VALIDATED_OFFLINE_DIAGNOSTIC"
            or set(core["modes"]) != set(MODES) or core["required_development_count"] != 20
            or core["required_held_out_count"] != 60 or core["held_out_evaluated_count"] != 0
            or core["annotation_gap"] != 20-core["evaluated_count"]
            or core["full_ablation_complete"] is not False or core["production_reranker_enabled"] is not False
            or core["unauthorized_hit_count"] != 0 or core["validity_date"] != DAY.isoformat()
            or report["identity_sha256"] != sha256(canonical(core)).hexdigest()
            or core["models"]["cross_encoder"] != dict(model_id=MODEL_ID,revision=MODEL_REVISION,
                                                       artifact_fingerprint=ARTIFACT_FINGERPRINT)):
            raise ValueError
        from askpolicy.retrieval.e5_encoder import MODEL_ID as E5_ID, MODEL_REVISION as E5_REV, ARTIFACT_FINGERPRINT as E5_FP
        if core["models"]["e5"] != dict(model_id=E5_ID,revision=E5_REV,artifact_fingerprint=E5_FP):
            raise ValueError
        reference = core["modes"]["hybrid"]["cases"]
        if (core["configurations"]["hybrid"] != dict(rrf_constant=60,bm25_top_k=30,faiss_top_k=30,top_k=20)
            or core["configurations"]["hybrid_rerank"] != dict(input_top_k=20,joint_token_limit=512,
                overflow="reject_entire_request",order=["raw_logit_descending","child_id_ascending"])
            or core["configurations"]["bm25"]["top_k"] != 30
            or core["configurations"]["faiss"]["top_k_children"] != 30
            or not 1 <= core["evaluated_count"] <= 20
            or core["execution_kind"] not in ("fixture_embeddings_and_logits","real_cpu_float32")):
            raise ValueError
        ids = [c["query_id"] for c in reference]
        if len(ids) != core["evaluated_count"] or len(set(ids)) != len(ids):
            raise ValueError
        positives = sum(bool(c["gold_sections"]) for c in reference)
        if positives != core["positive_metric_denominator"] or len(ids)-positives != core["no_positive_count"]:
            raise ValueError
        if len({c["question_family_id"] for c in reference}) != core["question_family_count"]:
            raise ValueError
        for mode,data in core["modes"].items():
            if (len(data["cases"]) != len(ids) or data["snapshot_id"] != core["snapshot"]["candidate_snapshot_id"]
                or data["published_release_id"] != core["published_release_id"]
                or data["dataset_sha256"] != core["dataset_sha256"]
                or data["corpus_manifest_fingerprint"] != core["corpus_manifest_fingerprint"]
                or data["unauthorized_hit_count"] != 0 or data["violating_query_count"] != 0
                or data["positive_metric_denominator"] != positives):
                raise ValueError
            completed = 0
            for row,expected in zip(data["cases"],reference,strict=True):
                if set(row) != ROW_FIELDS or any(row[k] != expected[k] for k in (
                    "query_id","question_family_id","active_role","language","case_type","gold_sections")):
                    raise ValueError
                ranking = row["ranking"]
                gold = {(g["document_id"],g["parent_id"]):g["relevance_grade"] for g in row["gold_sections"]}
                if len(gold) != len(row["gold_sections"]) or any(type(g) is not int or g <= 0 for g in gold.values()):
                    raise ValueError
                if (len(ranking) > (20 if mode.startswith("hybrid") else 30)
                    or len({h["child_id"] for h in ranking}) != len(ranking)
                    or [h["rank"] for h in ranking] != list(range(1,len(ranking)+1))
                    or row["unauthorized_hit_count"] != 0):
                    raise ValueError
                sections = [(h["document_id"],h["parent_id"]) for h in ranking]
                if row["status"] == "OK":
                    if row["metrics"] != section_metrics(sections,gold) or row["failure_code"] is not None:
                        raise ValueError
                    completed += bool(gold)
                elif (mode != "hybrid_rerank" or row["status"] != "FAILED_TOKEN_LIMIT" or ranking
                      or row["metrics"] is not None or row["failure_code"] != "TOKEN_LIMIT"):
                    raise ValueError
                if row["gold_ranks"] != [dict(document_id=d,parent_id=p,
                    first_child_rank=next((i for i,s in enumerate(sections,1) if s == (d,p)),None)) for d,p in sorted(gold)]:
                    raise ValueError
                if row["exclusion_reason"] != ("TOKEN_LIMIT" if row["status"] != "OK"
                                               else None if gold else "NO_POSITIVE_GOLD"):
                    raise ValueError
                if mode == "hybrid_rerank" and row["status"] == "OK":
                    mapping = lambda rs:{h["child_id"]:(h["document_id"],h["parent_id"]) for h in rs}
                    if mapping(ranking) != mapping(expected["ranking"]):
                        raise ValueError
                    if ranking != sorted(ranking,key=lambda h:(-h["raw_score"],h["child_id"])):
                        raise ValueError
                for hit in ranking:
                    if (set(hit) != {"rank","child_id","document_id","parent_id","raw_score","token_count"}
                        or any(type(hit[k]) is not str or not hit[k] for k in ("child_id","document_id","parent_id"))
                        or type(hit["raw_score"]) is not float or not isfinite(hit["raw_score"])
                        or (mode == "hybrid_rerank" and (type(hit["token_count"]) is not int
                                                       or not 1 <= hit["token_count"] <= 512))
                        or (mode != "hybrid_rerank" and hit["token_count"] is not None)):
                        raise ValueError
            expected_macro = macro_metrics([r["metrics"] for r in data["cases"]]) if completed == positives else None
            if data["metrics"] != expected_macro or data["completed_positive_count"] != completed:
                raise ValueError
            if data["returned_hit_count"] != sum(len(row["ranking"]) for row in data["cases"]):
                raise ValueError
            measured = report["measurements"]["modes"][mode]
            durations = measured["per_case_seconds"]
            if measured["sample_count"] != len(ids) or [r["query_id"] for r in durations] != ids:
                raise ValueError
            if mode == "hybrid_rerank":
                for duration,row in zip(durations,data["cases"],strict=True):
                    stages = duration["stages"]
                    if row["status"] != "OK":
                        if stages is not None or duration["failure_code"] != "TOKEN_LIMIT":
                            raise ValueError
                        continue
                    names = {"hydration_seconds","token_validation_seconds","scoring_seconds","ranking_seconds",
                             "helper_total_seconds","hybrid_seconds","other_seconds"}
                    if (duration["failure_code"] is not None or set(stages) != names
                        or any(type(v) is not float or not isfinite(v) or v < 0 for v in stages.values())
                        or sum(stages[k] for k in ("hydration_seconds","token_validation_seconds",
                            "scoring_seconds","ranking_seconds")) > stages["helper_total_seconds"]
                        or not isclose(duration["seconds"],stages["helper_total_seconds"]+
                                       stages["hybrid_seconds"]+stages["other_seconds"],abs_tol=1e-8)):
                        raise ValueError
            seconds = [r["seconds"] for r in durations]
            if (any(type(s) is not float or not isfinite(s) or s < 0 for s in seconds)
                or measured["p50_seconds"] != percentile(seconds,50)
                or measured["p95_seconds"] != percentile(seconds,95)):
                raise ValueError
        if (core["comparisons"] != _deltas(core["modes"])
            or core["reranker_failures"] != sum(r["status"] != "OK" for r in core["modes"]["hybrid_rerank"]["cases"])):
            raise ValueError
        _no_sensitive_fields(report)
    except Exception:
        pass
    else:
        return
    raise FourWayDiagnosticError("INVALID_FOUR_WAY_REPORT")


def _no_sensitive_fields(value):
    if type(value) is dict:
        if set(value) & {"text","query","prompt","jwt","token","password","passage","canonical_source","required_facts"}:
            raise ValueError
        for item in value.values():
            _no_sensitive_fields(item)
    elif type(value) in (list,tuple):
        for item in value:
            _no_sensitive_fields(item)


def write_report(report, output_path: Path, *, repository_root: Path):
    validate_report(report)
    output = safe_path(output_path)
    if output != safe_path(repository_root / REPORT_PATH):
        raise FourWayDiagnosticError("UNSAFE_REPORT_PATH")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.parent / (".step107-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(report)+b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,output)
    finally:
        temporary.unlink(missing_ok=True)
