"""Offline development-only three-way evaluation with verified role contexts."""

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
import os
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

from askpolicy.retrieval import bm25_smoke as seed, role_bm25, role_faiss, hybrid
from askpolicy.security.acl import can_read_document
from askpolicy.security.index_selection import select_role_index
from askpolicy.storage.index_store import canonical, safe_path
from askpolicy.storage.snapshot import DraftSnapshotConfig

from retrieval_metrics import macro_metrics, percentile, section_metrics

MODES = ("bm25", "faiss", "hybrid")
DAY = date(2026, 9, 1)
DATASET_PATH = Path("eval/datasets/dev_candidates.jsonl")


class DiagnosticError(ValueError):
    """Content-free dataset, identity or evaluation failure."""


class DiagnosticSecurityError(DiagnosticError):
    def __init__(self, count: int = 0):
        self.unauthorized_hit_count = count
        self.violating_query_count = int(count > 0)
        super().__init__("UNAUTHORIZED_EVIDENCE")


def coverage_status(count):
    if type(count) is not int or not 1 <= count <= 20:
        raise DiagnosticError("INVALID_DEVELOPMENT_COUNT")
    return ("DEVELOPMENT_COVERAGE_COMPLETE" if count == 20
            else "INCOMPLETE_DEVELOPMENT_COVERAGE")


def read_cases(raw: bytes):
    try:
        cases = seed._read_cases(raw)
        if not cases or len(cases) > 20:
            raise ValueError()
        families = defaultdict(list)
        for case in cases:
            if any(not value.strip() for value in (case.query_id, case.question_family_id, case.query)):
                raise ValueError()
            if case.case_type == "answerable" and case.expected_status != "answer":
                raise ValueError()
            if case.case_type == "unanswerable" and case.expected_status != "refusal":
                raise ValueError()
            families[case.question_family_id].append(case)
        for family in families.values():
            if any(c.case_type == "role_sensitive" for c in family):
                if (len(family) != 2 or {c.case_type for c in family} != {"role_sensitive"}
                    or len({c.query for c in family}) != 1 or len({c.language for c in family}) != 1
                    or len({c.active_role for c in family}) != 2
                    or {c.expected_status for c in family} != {"answer", "refusal"}):
                    raise ValueError()
            elif len(family) != 1:
                raise ValueError()
        return cases
    except (ValueError, TypeError, UnicodeError):
        raise DiagnosticError("INVALID_DEVELOPMENT_DATASET") from None


def _gold(connection, cases, bundle, contexts):
    result = {}
    current = set(bundle.eligibility.eligible_document_ids)
    for case in cases:
        grades = {}
        for label in case.authorized_source_sections:
            document = bundle._draft.documents.get(label.document_id)
            if document is None or label.document_id not in current:
                raise DiagnosticError("INVALID_GOLD_DOCUMENT")
            meta = document.metadata
            if not can_read_document(contexts[case.query_id], classification=meta.classification,
                                     allowed_roles=meta.allowed_roles):
                raise DiagnosticSecurityError()
            section = seed._resolve_section(connection, label)
            key = (section.document_id, section.parent_id)
            if key in grades:
                raise DiagnosticError("DUPLICATE_GOLD_SECTION")
            grades[key] = label.relevance_grade
        result[case.query_id] = grades
    return result


def hardware():
    import torch
    return dict(os=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), python=platform.python_version(),
                torch_threads=torch.get_num_threads(),
                versions={name:version(name) for name in ("bm25s","faiss-cpu","torch","transformers","numpy")})


def evaluate(connection, *, manifest: bytes, raw_dataset: bytes, bm25, faiss, context_for_role,
             validity_date: date = DAY, execution_kind: str):
    """Validate the entire run before warmup, then evaluate exactly three modes.

    context_for_role is trusted runner infrastructure that signs and verifies
    synthetic JWTs through production authentication. Each returned handle is
    independently checked against the case's approved active role.
    """
    try:
        if validity_date != DAY or type(validity_date) is not date:
            raise DiagnosticError("INVALID_DIAGNOSTIC_DATE")
        if execution_kind not in ("real_e5_cpu", "fixture_embeddings"):
            raise DiagnosticError("INVALID_EXECUTION_KIND")
        role_faiss.verify_matching_bundles(bm25, faiss)
        with role_bm25.validated_role_draft(
            connection, manifest=manifest, snapshot_config=DraftSnapshotConfig(), validity_date=validity_date,
            expected_snapshot_id=bm25.snapshot.candidate_snapshot_id,
        ) as (image, binding):
            snapshot = binding.snapshot
            if snapshot != bm25.snapshot:
                raise DiagnosticError("SNAPSHOT_MISMATCH")
            cases = read_cases(raw_dataset)
            contexts = {}
            for case in cases:
                context = context_for_role(case.active_role)
                if select_role_index(context).role != case.active_role:
                    raise DiagnosticSecurityError()
                hybrid._gate(bm25, faiss, context, snapshot.candidate_snapshot_id, validity_date)
                contexts[case.query_id] = context
            gold = _gold(image, cases, bm25, contexts)
        # One explicitly reported untimed query warmup, with no retrieval or
        # fabricated independent sample. Every measured query is encoded afresh.
        started = perf_counter()
        faiss._encoder.encode_queries([cases[0].query])
        warmup_seconds = perf_counter() - started
        modes = {}
        timings = {}
        for mode in MODES:
            rows = []
            durations = []
            for case in cases:
                context = contexts[case.query_id]
                started = perf_counter()
                options = dict(query=case.query, expected_snapshot_id=snapshot.candidate_snapshot_id,
                               validity_date=validity_date)
                if mode == "bm25":
                    hits = role_bm25.search_role_bm25(bm25, context, **options)
                elif mode == "faiss":
                    hits = role_faiss.search_role_faiss(faiss, context, **options)
                else:
                    hits = hybrid.search_role_hybrid(bm25, faiss, context, **options)
                durations.append(perf_counter() - started)
                ranking = _checked_ranking(hits, case, context, bm25, snapshot.candidate_snapshot_id, mode)
                sections = [(h["document_id"], h["parent_id"]) for h in ranking]
                rows.append(dict(query_id=case.query_id, question_family_id=case.question_family_id,
                    active_role=case.active_role, language=case.language, case_type=case.case_type,
                    gold_sections=[dict(document_id=d, parent_id=p, relevance_grade=g)
                                   for (d,p),g in sorted(gold[case.query_id].items())],
                    metrics=section_metrics(sections, gold[case.query_id]), ranking=ranking,
                    exclusion_reason=None if gold[case.query_id] else "NO_POSITIVE_GOLD",
                    unauthorized_hit_count=0))
            returned_count = sum(len(row["ranking"]) for row in rows)
            modes[mode] = dict(
                retrieval_configuration=(asdict(bm25.indexes[0].metadata.configuration) if mode == "bm25"
                    else asdict(faiss.indexes[0].metadata.configuration) if mode == "faiss"
                    else dict(k=60,top_k=20,bm25_top_k=30,faiss_top_k=30)),
                snapshot_id=snapshot.candidate_snapshot_id, dataset_sha256=sha256(raw_dataset).hexdigest(),
                corpus_manifest_fingerprint=snapshot.source_manifest_hash,
                model_revision=faiss._encoder.revision, cases=rows,
                metrics=macro_metrics([row["metrics"] for row in rows]),
                returned_hit_count=returned_count,
                unauthorized_hit_count=0, violating_query_count=0,
                unauthorized_retrieval_rate=0.0 if returned_count else None,
                role_pairs=_role_pairs(cases, gold, rows),
            )
            timings[mode] = dict(sample_count=len(durations), state="warm_after_one_encoder_warmup",
                per_case_seconds=[dict(query_id=c.query_id, seconds=s) for c,s in zip(cases,durations,strict=True)],
                total_seconds=sum(durations), p50_seconds=percentile(durations,50),
                p95_seconds=percentile(durations,95), percentile_method="linear-(n-1)",
                interpretation="exploratory_small_sample_not_production_latency")
        core = dict(evaluation_kind="development_three_way_diagnostic", status=coverage_status(len(cases)),
            authorized_full_count=80, required_development_count=20, required_held_out_count=60,
            held_out_evaluated_count=0,
            evaluated_count=len(cases), annotation_gap=20-len(cases),
            positive_metric_denominator=sum(bool(g) for g in gold.values()),
            no_positive_count=sum(not g for g in gold.values()),
            question_family_count=len({c.question_family_id for c in cases}),
            languages=dict(Counter(c.language for c in cases)), roles=dict(Counter(c.active_role for c in cases)),
            case_types=dict(Counter(c.case_type for c in cases)),
            snapshot=asdict(snapshot), dataset_sha256=sha256(raw_dataset).hexdigest(),
            source_manifest_sha256=sha256(manifest).hexdigest(), validity_date=validity_date.isoformat(),
            configurations=dict(bm25=asdict(bm25.indexes[0].metadata.configuration),
                faiss=asdict(faiss.indexes[0].metadata.configuration), rrf=dict(k=60,top_k=20)),
            execution_kind=execution_kind, modes=modes,
            fourth_configuration="PENDING_OWNER_APPROVED_REVISION_ARTIFACTS_AND_CPU_SMOKE",
            metric_definition="section pairs; gain=2^grade-1; discount=log2(rank+1); repeated section gain once")
        report = dict(deterministic=core, identity_sha256=sha256(canonical(core)).hexdigest(),
                      measurements=dict(hardware=hardware(), encoder_warmup_seconds=warmup_seconds, modes=timings))
        validate_report(report)
        return report
    except DiagnosticError:
        raise
    except Exception:
        raise DiagnosticError("DIAGNOSTIC_FAILED") from None


def _checked_ranking(hits, case, context, bundle, snapshot_id, mode):
    if type(hits) is not tuple or len(hits) > (20 if mode == "hybrid" else 30):
        raise DiagnosticError("INVALID_RANKING")
    result = []
    seen = set()
    for rank, hit in enumerate(hits,1):
        child = hit.child
        meta = child.document.metadata
        if not can_read_document(context,classification=meta.classification,allowed_roles=meta.allowed_roles):
            raise DiagnosticSecurityError(1)
        if (child != bundle._draft.children.get(child.child_id) or child.child_id in seen
            or child.section.document_id not in bundle.eligibility.eligible_document_ids
            or hit.candidate_snapshot_id != snapshot_id):
            raise DiagnosticError("INVALID_HIT")
        seen.add(child.child_id)
        result.append(dict(rank=rank, child_id=child.child_id, document_id=child.section.document_id,
                           parent_id=child.section.parent_id))
    return result


def _role_pairs(cases, gold, rows):
    indexed = {row["query_id"]:row for row in rows}
    pairs = []
    for positive in cases:
        if positive.case_type != "role_sensitive" or positive.expected_status != "answer":
            continue
        negative, = [c for c in cases if c.question_family_id == positive.question_family_id
                     and c.expected_status == "refusal"]
        forbidden = set(gold[positive.query_id])
        count = sum((hit["document_id"],hit["parent_id"]) in forbidden
                    for hit in indexed[negative.query_id]["ranking"])
        if count:
            raise DiagnosticSecurityError(count)
        pairs.append(dict(question_family_id=positive.question_family_id,
            positive_query_id=positive.query_id, negative_query_id=negative.query_id,
            negative_restricted_section_hits=count))
    return pairs


def validate_report(report):
    try:
        core = report["deterministic"]
        if (core["evaluation_kind"] != "development_three_way_diagnostic"
            or core["status"] != coverage_status(core["evaluated_count"])
            or core["annotation_gap"] != 20-core["evaluated_count"]
            or set(core["modes"]) != set(MODES)
            or core["required_development_count"] != 20 or core["required_held_out_count"] != 60
            or core["authorized_full_count"] != 80
            or report["identity_sha256"] != sha256(canonical(core)).hexdigest()):
            raise ValueError()
        for mode, data in core["modes"].items():
            if (len(data["cases"]) != core["evaluated_count"] or data["snapshot_id"] != core["snapshot"]["candidate_snapshot_id"]
                or data["dataset_sha256"] != core["dataset_sha256"]
                or data["corpus_manifest_fingerprint"] != core["snapshot"]["source_manifest_hash"]
                or data["model_revision"] != core["configurations"]["faiss"]["revision"]
                or data["metrics"] != macro_metrics([c["metrics"] for c in data["cases"]])
                or data["unauthorized_hit_count"] != 0 or data["violating_query_count"] != 0
                or report["measurements"]["modes"][mode]["sample_count"] != core["evaluated_count"]):
                raise ValueError()
            if (core["positive_metric_denominator"] != sum(bool(c["gold_sections"]) for c in data["cases"])
                or core["no_positive_count"] != sum(not c["gold_sections"] for c in data["cases"])):
                raise ValueError()
            ids = [row["query_id"] for row in data["cases"]]
            if len(set(ids)) != len(ids) or ids != sorted(ids):
                raise ValueError()
            for row in data["cases"]:
                ranking = row["ranking"]
                gold = {(g["document_id"],g["parent_id"]):g["relevance_grade"] for g in row["gold_sections"]}
                if (len(gold) != len(row["gold_sections"]) or len(ranking) > (20 if mode == "hybrid" else 30)
                    or len({h["child_id"] for h in ranking}) != len(ranking)
                    or [h["rank"] for h in ranking] != list(range(1,len(ranking)+1))
                    or row["metrics"] != section_metrics([(h["document_id"],h["parent_id"]) for h in ranking],gold)
                    or row["exclusion_reason"] != (None if gold else "NO_POSITIVE_GOLD")
                    or row["unauthorized_hit_count"] != 0):
                    raise ValueError()
            measured = report["measurements"]["modes"][mode]
            if [row["query_id"] for row in measured["per_case_seconds"]] != ids:
                raise ValueError()
            seconds = [row["seconds"] for row in measured["per_case_seconds"]]
            if (any(type(s) is not float or not isfinite(s) or s < 0 for s in seconds)
                or measured["p50_seconds"] != percentile(seconds,50)
                or measured["p95_seconds"] != percentile(seconds,95)):
                raise ValueError()
        if len({tuple(c["query_id"] for c in data["cases"]) for data in core["modes"].values()}) != 1:
            raise ValueError()
    except Exception:
        raise DiagnosticError("INVALID_REPORT") from None


def write_report(report, output_path: Path, *, repository_root: Path):
    """Write only an explicitly selected ignored runtime report, atomically."""
    validate_report(report)
    output = safe_path(output_path)
    allowed = safe_path(repository_root / "runtime/reports")
    if not output.is_relative_to(allowed) or output.suffix != ".json":
        raise DiagnosticError("UNSAFE_REPORT_PATH")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.parent / (".report-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(report) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,output)
    finally:
        temporary.unlink(missing_ok=True)
