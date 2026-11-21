"""Explicit Step 112 CPU benchmark; ordinary CI uses deterministic doubles."""

from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
import json
from math import isclose, isfinite
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
for directory in ("src", "tests", "eval"):
    sys.path.insert(0, str(ROOT / directory))
import dev_four_way_diagnostic as diagnostic
from retrieval_metrics import METRIC_NAMES, macro_metrics, percentile
from askpolicy.security.acl import can_read_document
from askpolicy.security.index_selection import select_role_index
from askpolicy.storage.index_store import canonical, safe_path
from security.test_auth import secured
from unit.test_e5_prefix import _working_set_mib

BASELINE = "a9ca231409446cfa74fb380fdbbbad82948d5d39"
DATASET_SHA = "d3280f8570ca81f2f1b2c300e9e2c1385a1d5113f07d7f42d1f286331551dc5b"
REPORT_PATH = Path("runtime/reports/step112_dev_20_four_way_diagnostic.json")
RUNNER_PATH = Path("tests/local/real_dev_20_four_way_diagnostic.py")
COMMAND = "uv run --locked --no-sync python -m pytest tests/local/real_dev_20_four_way_diagnostic.py -v -s"
PAIR_IDS = [("dev-candidate-003", "dev-candidate-004"),
            ("dev-candidate-017", "dev-candidate-018"),
            ("dev-candidate-019", "dev-candidate-020")]


def read_inputs(repository_root=ROOT):
    raw = (repository_root / diagnostic.base.DATASET_PATH).read_bytes()
    approval_raw = (repository_root / "eval/datasets/dev_annotation_approval.json").read_bytes()
    approval = json.loads(approval_raw)
    if sha256(raw).hexdigest() != DATASET_SHA or approval["canonical_dataset_sha256"] != DATASET_SHA:
        raise ValueError("INVALID_APPROVED_DATASET")
    cases = diagnostic.base.read_cases(raw)
    if ([c.query_id for c in cases] != [f"dev-candidate-{i:03}" for i in range(1, 21)]
        or Counter(c.language for c in cases) != {"fr":8,"en":6,"ar":6}
        or Counter(c.active_role for c in cases) != {"employee":10,"hr":3,"finance":3,"it_security":4}
        or Counter(c.case_type for c in cases) != {"answerable":11,"unanswerable":3,"role_sensitive":6}
        or len({c.question_family_id for c in cases}) != 17
        or sum(bool(c.authorized_source_sections) for c in cases) != 14
        or sum(len(c.authorized_source_sections) for c in cases) != 50):
        raise ValueError("INVALID_APPROVED_INVENTORY")
    hashes = {c.query_id:sha256(json.dumps(c.model_dump(),ensure_ascii=False,
              separators=(",",":"),allow_nan=False).encode()).hexdigest() for c in cases}
    if (approval["canonical_record_sha256"] != hashes
        or approval["approval_source"] != "explicit_project_owner_chat_decision"
        or approval["trusted_validity_date"] != diagnostic.DAY.isoformat()):
        raise ValueError("INVALID_ANNOTATION_APPROVAL")
    return raw, approval_raw, cases


def _aggregate(rows):
    positives = sum(bool(r["gold_sections"]) for r in rows)
    failed = sum(bool(r["gold_sections"]) and r["status"] != "OK" for r in rows)
    return dict(executions=len(rows), positive_gold_count=positives,
                no_positive_count=len(rows)-positives, failed_positive_count=failed,
                metrics=None if failed else macro_metrics([r["metrics"] for r in rows]))


def derive_summary(report):
    """Derived reporting only. All rankings and metric formulas stay upstream."""
    core, measurements = report["deterministic"], report["measurements"]
    modes = core["modes"]
    reference = modes["hybrid"]["cases"]
    breakdowns = {}
    for field, label in (("language","languages"),("active_role","roles"),("case_type","case_types")):
        breakdowns[label] = {value:{mode:_aggregate([r for r in data["cases"] if r[field] == value])
            for mode,data in modes.items()} for value in sorted({r[field] for r in reference})}
    paired, ceilings = [], []
    counts = {metric:dict(improved=0,tied=0,worsened=0,unavailable=0) for metric in METRIC_NAMES}
    pareto = dict(improved=0,tied=0,worsened=0,mixed=0,unavailable=0)
    first_rank = lambda row:min((g["first_child_rank"] for g in row["gold_ranks"]
                                if g["first_child_rank"] is not None),default=None)
    for before,after in zip(reference,modes["hybrid_rerank"]["cases"],strict=True):
        if not before["gold_sections"]:
            continue
        size = len(before["gold_sections"])
        ceilings.append(dict(query_id=before["query_id"],language=before["language"],
            gold_section_count=size,theoretical_maximum_recall_at_5=min(5,size)/size))
        delta = (None if after["metrics"] is None else
                 {m:after["metrics"][m]-before["metrics"][m] for m in METRIC_NAMES})
        if delta is None:
            outcome = "unavailable"
        elif all(v == 0 for v in delta.values()):
            outcome = "tied"
        elif all(v >= 0 for v in delta.values()):
            outcome = "improved"
        elif all(v <= 0 for v in delta.values()):
            outcome = "worsened"
        else:
            outcome = "mixed"
        pareto[outcome] += 1
        for metric in METRIC_NAMES:
            result = ("unavailable" if delta is None else "improved" if delta[metric]>0
                      else "worsened" if delta[metric]<0 else "tied")
            counts[metric][result] += 1
        paired.append(dict(query_id=before["query_id"],language=before["language"],
            before_first_relevant_rank=first_rank(before),after_first_relevant_rank=first_rank(after),
            differences=delta,outcome=outcome))
    latency = {}
    for mode in diagnostic.MODES:
        timing = measurements["modes"][mode]
        seconds = [r["seconds"] for r in timing["per_case_seconds"]]
        latency[mode] = dict(sample_count=len(seconds),total_seconds=sum(seconds),
            p50_seconds=percentile(seconds,50),p95_seconds=percentile(seconds,95))
    stages = {}
    successful = [r["stages"] for r in measurements["modes"]["hybrid_rerank"]["per_case_seconds"]
                  if r["stages"] is not None]
    if successful:
        for name in sorted(successful[0]):
            values = [r[name] for r in successful]
            stages[name] = dict(sample_count=len(values),total_seconds=sum(values),
                p50_seconds=percentile(values,50),p95_seconds=percentile(values,95))
        values = [r["helper_total_seconds"]-sum(r[k] for k in (
            "hydration_seconds","token_validation_seconds","scoring_seconds","ranking_seconds"))
                  for r in successful]
        stages["helper_unattributed_seconds"] = dict(sample_count=len(values),total_seconds=sum(values),
            p50_seconds=percentile(values,50),p95_seconds=percentile(values,95))
    differences = [b["seconds"]-a["seconds"] for a,b in zip(
        measurements["modes"]["hybrid"]["per_case_seconds"],
        measurements["modes"]["hybrid_rerank"]["per_case_seconds"],strict=True)]
    overhead = dict(per_case_seconds=[dict(query_id=r["query_id"],seconds=s) for r,s in zip(reference,differences,strict=True)],
        total_seconds=sum(differences),p50_seconds=percentile(differences,50),
        p95_seconds=percentile(differences,95),
        total_latency_ratio=latency["hybrid_rerank"]["total_seconds"]/latency["hybrid"]["total_seconds"])
    failures = [dict(query_id=r["query_id"],status=r["status"],failure_code=r["failure_code"],
                     seconds=t["seconds"],metrics=r["metrics"])
        for r,t in zip(modes["hybrid_rerank"]["cases"],measurements["modes"]["hybrid_rerank"]["per_case_seconds"],strict=True)
        if r["status"] != "OK"]
    return dict(breakdowns=breakdowns,paired_positive_cases=paired,per_metric_outcomes=counts,
        pareto_outcomes=pareto,paired_full_sample_mean_available=not any(
            r["differences"] is None for r in paired),
        macro_differences=core["comparisons"]["rerank_minus_baseline_macro"],
        all_mode_quality={m:d["metrics"] for m,d in modes.items()},
        gold_section_ceilings=ceilings,latency=latency,reranking_overhead=overhead,
        fourth_mode_stages=stages,failures=failures,
        role_pairs={m:d["role_pairs"] for m,d in modes.items()},
        unauthorized_hit_counts={m:d["unauthorized_hit_count"] for m,d in modes.items()})


def validate_benchmark_report(report, cases):
    diagnostic.validate_report(report)
    core = report["deterministic"]
    if (core["dataset_sha256"] != DATASET_SHA or core["evaluated_count"] != 20
        or core["positive_metric_denominator"] != 14 or core["no_positive_count"] != 6
        or core["question_family_count"] != 17 or core["status"] != "DEVELOPMENT_COVERAGE_COMPLETE"
        or core["languages"] != {"fr":8,"en":6,"ar":6}
        or core["roles"] != {"employee":10,"hr":3,"finance":3,"it_security":4}):
        raise ValueError("INVALID_BENCHMARK_COVERAGE")
    for data in core["modes"].values():
        if [(p["positive_query_id"],p["negative_query_id"]) for p in data["role_pairs"]] != PAIR_IDS:
            raise ValueError("INVALID_ROLE_PAIRS")
        if any(p["negative_restricted_section_hits"] != 0 for p in data["role_pairs"]):
            raise ValueError("UNAUTHORIZED_PAIR_EVIDENCE")
        for row,case in zip(data["cases"],cases,strict=True):
            if (row["query_id"] != case.query_id or row["question_family_id"] != case.question_family_id
                or row["language"] != case.language or row["active_role"] != case.active_role
                or row["case_type"] != case.case_type
                or Counter((g["document_id"],g["relevance_grade"]) for g in row["gold_sections"]) !=
                   Counter((g.document_id,g.relevance_grade) for g in case.authorized_source_sections)):
                raise ValueError("GOLD_OR_CASE_CHANGED")
    summary = report["measurements"].get("step112_summary")
    if summary != derive_summary(report):
        raise ValueError("INVALID_DERIVED_SUMMARY")
    # Existing validator checks field names. Also reject canonical query/fact
    # payloads anywhere in report string values, including optional metadata.
    def strings(value):
        if isinstance(value,dict):
            return [s for v in value.values() for s in strings(v)]
        if isinstance(value,(list,tuple)):
            return [s for v in value for s in strings(v)]
        return [value] if isinstance(value,str) else []
    values = strings(report)
    if any(c.query in v or any(f in v for f in c.required_facts) for c in cases for v in values):
        raise ValueError("SENSITIVE_ANNOTATION_PAYLOAD")


def check_evidence(report, handle, contexts, cases):
    """Recheck validated catalog identities/ACLs; never read live source text."""
    current = set(handle.bm25.eligibility.eligible_document_ids)
    children = handle.bm25._draft.children
    for data in report["deterministic"]["modes"].values():
        if data["snapshot_id"] != handle.bm25.snapshot.candidate_snapshot_id or data["published_release_id"] != handle.release_id:
            raise ValueError("SNAPSHOT_MISMATCH")
        for row,case in zip(data["cases"],cases,strict=True):
            context = contexts[case.active_role]
            if select_role_index(context).role != row["active_role"]:
                raise ValueError("ROLE_MISMATCH")
            expected = {}
            for label in case.authorized_source_sections:
                sections = {c.section.parent_id for c in children.values()
                            if c.section.document_id == label.document_id and c.section.heading == label.section_heading}
                if len(sections) != 1:
                    raise ValueError("INVALID_GOLD_SECTION")
                expected[(label.document_id,next(iter(sections)))] = label.relevance_grade
            if {(g["document_id"],g["parent_id"]):g["relevance_grade"] for g in row["gold_sections"]} != expected:
                raise ValueError("GOLD_SECTION_CHANGED")
            for hit in row["ranking"]:
                child = children.get(hit["child_id"])
                if (child is None or child.section.document_id != hit["document_id"]
                    or child.section.parent_id != hit["parent_id"] or hit["document_id"] not in current):
                    raise ValueError("INVALID_RETURNED_IDENTITY")
                meta = child.document.metadata
                if not can_read_document(context,classification=meta.classification,allowed_roles=meta.allowed_roles):
                    raise ValueError("UNAUTHORIZED_RETURNED_EVIDENCE")


def write_benchmark_report(report, cases, *, repository_root=ROOT, output_path=None):
    """Use upstream canonical encoding/validation at the new allowlisted path."""
    validate_benchmark_report(report,cases)
    target = safe_path(repository_root / REPORT_PATH)
    if output_path is not None and safe_path(output_path) != target:
        raise ValueError("UNSAFE_BENCHMARK_PATH")
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.parent / (".step112-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(report)+b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def test_real_dev_20_four_way_diagnostic(secured):
    import torch
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval.e5_encoder import E5Encoder
    from askpolicy.retrieval.cross_encoder import CrossEncoder
    from askpolicy.storage import publication, index_store
    from askpolicy.storage.snapshot import DraftSnapshotConfig

    raw, approval_raw, cases = read_inputs()
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert head == BASELINE
    torch.set_num_threads(4)
    def no_network(*args,**kwargs):
        raise AssertionError("UNEXPECTED_NETWORK_ACCESS")
    session_started = perf_counter()
    with patch.object(socket.socket,"connect",no_network), patch.object(socket.socket,"connect_ex",no_network), patch.object(socket,"getaddrinfo",no_network):
        tick = perf_counter()
        e5 = E5Encoder(Path(os.environ["ASKPOLICY_E5_DIR"]))
        e5_load = perf_counter()-tick
        tick = perf_counter()
        cross = CrossEncoder(Path(os.environ["ASKPOLICY_RERANKER_DIR"]))
        cross_load = perf_counter()-tick
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in cross._model.parameters())
        assert not cross._model.training and cross._tokenizer.is_fast
        root = ROOT / "runtime" / ("step112-published-" + uuid4().hex)
        options = dict(manifest=(ROOT/"data/corpus_manifest.yaml").read_bytes(),
            snapshot_config=DraftSnapshotConfig(),validity_date=diagnostic.DAY,encoder=e5)
        stage_times, index_build_times = [], []
        stage = publication.stage_snapshot
        bm_build = index_store.lexical.build_role_bm25_bundle
        fa_build = index_store.dense.build_role_faiss_bundle
        def timed_build(kind, build, *args, **kwargs):
            tick = perf_counter()
            value = build(*args,**kwargs)
            index_build_times.append(dict(kind=kind,seconds=perf_counter()-tick))
            return value
        def timed_bm(*args,**kwargs):
            return timed_build("bm25",bm_build,*args,**kwargs)
        def timed_fa(*args,**kwargs):
            return timed_build("faiss",fa_build,*args,**kwargs)
        def timed_stage(*args,**kwargs):
            tick = perf_counter()
            with patch.object(index_store.lexical,"build_role_bm25_bundle",timed_bm), patch.object(index_store.dense,"build_role_faiss_bundle",timed_fa):
                value = stage(*args,**kwargs)
            stage_times.append(perf_counter()-tick)
            return value
        with closing(sqlite3.connect(":memory:")) as draft:
            tick = perf_counter()
            counts = ingest_manifest_to_draft(ROOT,draft)
            ingestion = perf_counter()-tick
            assert counts.documents == 26
            tick = perf_counter()
            with patch.object(publication,"stage_snapshot",timed_stage):
                published = publication.publish_snapshot(draft,root,**options)
            publish_wall = perf_counter()-tick
        assert len(stage_times) == 1
        assert [r["kind"] for r in index_build_times] == ["bm25","faiss"]
        tick = perf_counter()
        handle = publication.load_active_snapshot(root,**options)
        reload_seconds = perf_counter()-tick
        assert handle.release_id == published.release_id and handle.manifest_hash == published.manifest_hash
        contexts = {role:secured.authenticate(secured.sign(roles=[role]),role)
                    for role in sorted({c.active_role for c in cases})}
        forward_calls, token_calls, rerank_calls = [], [], []
        active = dict(query_id=None)
        original_forward = cross._model.forward
        def observe_forward(**batch):
            assert torch.is_inference_mode_enabled() and not cross._model.training
            assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in cross._model.parameters())
            assert all(v.device.type == "cpu" for v in batch.values())
            shape = batch["input_ids"].shape
            assert 1 <= shape[0] <= 4 and 1 <= shape[1] <= 512
            forward_calls.append(dict(query_id=active["query_id"],pairs=int(shape[0]),
                padded_tokens=int(shape[1]),token_counts=[int(v) for v in batch["attention_mask"].sum(dim=1).tolist()]))
            return original_forward(**batch)
        tokenizer_type = type(cross._tokenizer)
        original_tokenize = tokenizer_type.__call__
        def observe_tokens(instance,*args,**kwargs):
            encoded = original_tokenize(instance,*args,**kwargs)
            if instance is cross._tokenizer:
                assert kwargs.get("truncation") is False and kwargs.get("padding") is False
                token_calls.append(dict(query_id=active["query_id"],
                    token_counts=[len(ids) for ids in encoded["input_ids"]]))
            return encoded
        original_rerank = diagnostic.rerank_published_hits
        def observe_rerank(*args,**kwargs):
            case = cases[len(rerank_calls)]
            active["query_id"] = case.query_id
            rerank_calls.append(case.query_id)
            try:
                return original_rerank(*args,**kwargs)
            finally:
                active["query_id"] = None
        with patch.object(cross._model,"forward",observe_forward), patch.object(tokenizer_type,"__call__",observe_tokens), patch.object(diagnostic,"rerank_published_hits",observe_rerank):
            report = diagnostic.evaluate(handle,release_root=root,manifest=options["manifest"],raw_dataset=raw,
                context_for_role=contexts.__getitem__,encoder=cross,execution_kind="real_cpu_float32")
        assert rerank_calls == [c.query_id for c in cases]
        warm = [c for c in forward_calls if c["query_id"] is None]
        measured = [c for c in forward_calls if c["query_id"] is not None]
        assert len(warm) == 1 and warm[0]["pairs"] == 1
        core = report["deterministic"]
        by_id = {r["query_id"]:r for r in core["modes"]["hybrid_rerank"]["cases"]}
        expected_pairs = sum(len(r["ranking"]) for r in by_id.values() if r["status"] == "OK")
        assert sum(c["pairs"] for c in measured) == expected_pairs
        for qid,row in by_id.items():
            scored = [c for c in measured if c["query_id"] == qid]
            assert sum(c["pairs"] for c in scored) == (len(row["ranking"]) if row["status"] == "OK" else 0)
        check_evidence(report,handle,contexts,cases)
        token_audit = []
        hybrid_rows = core["modes"]["hybrid"]["cases"]
        for case,hybrid_row in zip(cases,hybrid_rows,strict=True):
            size = len(hybrid_row["ranking"])
            observed = [n for call in token_calls if call["query_id"] == case.query_id for n in call["token_counts"]][:size]
            row = by_id[case.query_id]
            token_audit.append(dict(query_id=case.query_id,candidate_count=size,
                first_prevalidation_token_counts=observed,unobserved_after_fail_closed=max(0,size-len(observed)),
                observed_overlength_pairs=sum(n>512 for n in observed),status=row["status"]))
            if row["status"] == "OK":
                assert len(observed) == size and all(0<n<=512 for n in observed)
        report["measurements"]["step112_summary"] = derive_summary(report)
        report["measurements"]["session"] = dict(
            git_commit=head,dataset_sha256=sha256(raw).hexdigest(),
            published_runtime_root=root.relative_to(ROOT).as_posix(),
            approval_manifest_sha256=sha256(approval_raw).hexdigest(),
            lockfile_sha256=sha256((ROOT/"uv.lock").read_bytes()).hexdigest(),
            runner_source_sha256=sha256((ROOT/RUNNER_PATH).read_bytes()).hexdigest(),
            evaluator_source_sha256={name:sha256((ROOT/"eval"/name).read_bytes()).hexdigest()
                for name in ("dev_four_way_diagnostic.py","dev_retrieval_diagnostic.py","retrieval_metrics.py")},
            evaluation_utc=datetime.now(UTC).isoformat(),command=COMMAND,
            e5_loading_seconds=e5_load,cross_encoder_loading_seconds=cross_load,
            ingestion_seconds=ingestion,index_build_and_artifact_staging_seconds=stage_times[0],
            index_build_observations=index_build_times,
            index_build_seconds=sum(r["seconds"] for r in index_build_times),
            artifact_staging_residual_seconds=stage_times[0]-sum(r["seconds"] for r in index_build_times),
            publication_verification_seconds=publish_wall-stage_times[0],
            snapshot_reload_seconds=reload_seconds,publish_wall_seconds=publish_wall,
            publication_timing_note="stage time includes artifact serialization; publication residual includes verification/finalization/pointer publication; publish_wall overlaps both",
            memory=_working_set_mib(),network_requests=0,device="cpu",dtype="float32",
            warmup_forward_calls=len(warm),warmup_pair_count=sum(c["pairs"] for c in warm),
            measured_forward_calls=len(measured),measured_pair_count=sum(c["pairs"] for c in measured),
            forward_batch_observations=forward_calls,token_audit=token_audit,
            tokenization_observation_count=len(token_calls),
            successful_rerank_queries=sum(r["status"]=="OK" for r in by_id.values()),
            token_limit_failures=sum(r["status"]=="FAILED_TOKEN_LIMIT" for r in by_id.values()),
            actually_scored_pairs=expected_pairs,
            total_padded_tokens=sum(c["pairs"]*c["padded_tokens"] for c in measured),
            session_total_seconds=perf_counter()-session_started,
            benchmark_order=dict(query_ids=[c.query_id for c in cases],modes=list(diagnostic.MODES)),
            limitations=[
                "One warm measured sample per query/mode; small development language groups, no production API latency.",
                "Fixed per-query order BM25, FAISS, Hybrid, rerank; Hybrid is reused exactly for mode four.",
                "Fresh model objects, OS file cache not flushed. Sequential order and runtime caches may bias timing.",
                "Scoring timer includes repeated token validation, padding and inference; instrumentation overhead is included.",
                "All approved multilingual gold sections retained; more than five gold sections limits Recall@5.",
                "Failures preserve the full positive denominator and unavailable comparable metrics.",
                "Token observations may repeat during validation; failed-query tails may be unobserved and never scored.",
                "No generation, refusal accuracy, held-out evaluation, production activation or contractual full-ablation completion.",
            ])
        validate_benchmark_report(report,cases)
        target = write_benchmark_report(report,cases)
        print(json.dumps(dict(report_path=str(target),release_id=handle.release_id,
            snapshot_id=core["snapshot"]["candidate_snapshot_id"],metrics={m:d["metrics"] for m,d in core["modes"].items()},
            reranker_failures=core["reranker_failures"],unauthorized_hit_count=core["unauthorized_hit_count"]),sort_keys=True))

