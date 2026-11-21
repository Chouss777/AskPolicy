"""Development evaluation contracts with real indexes and fixture embeddings."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from integration.test_embedding_revision import dense_env,DAY
from security.test_auth import ROOT,secured

sys.path.insert(0,str(ROOT/"eval"))
sys.path.insert(0,str(ROOT/"src"))
import dev_retrieval_diagnostic as diagnostic


@pytest.fixture
def evaluation(dense_env,secured):
    return SimpleNamespace(env=dense_env,raw=(ROOT/diagnostic.DATASET_PATH).read_bytes(),
        provider=lambda role:secured.authenticate(secured.sign(roles=[role]),role))


def run(evaluation,*,raw=None,bm25=None,faiss=None,provider=None):
    e=evaluation
    return diagnostic.evaluate(e.env.db,manifest=e.env.options["manifest"],raw_dataset=e.raw if raw is None else raw,
        bm25=e.env.bm25 if bm25 is None else bm25,faiss=e.env.bundle if faiss is None else faiss,
        context_for_role=e.provider if provider is None else provider,execution_kind="fixture_embeddings")


def encoded(records):
    return b"\n".join(json.dumps(r,ensure_ascii=False).encode() for r in records)+b"\n"


def test_all_twenty_records_resolve_exact_sections_and_three_modes(evaluation,tmp_path):
    first=run(evaluation)
    second=run(evaluation)
    assert first["deterministic"]==second["deterministic"]
    assert first["identity_sha256"]==second["identity_sha256"]
    core=first["deterministic"]
    assert core["status"]=="DEVELOPMENT_COVERAGE_COMPLETE"
    assert core["evaluated_count"]==20 and core["question_family_count"]==17 and core["annotation_gap"]==0
    assert core["positive_metric_denominator"]==14 and core["no_positive_count"]==6
    assert core["required_development_count"]==20 and core["required_held_out_count"]==60
    assert core["dataset_sha256"]==sha256(evaluation.raw).hexdigest()
    assert core["languages"]=={"fr":8,"en":6,"ar":6} and core["roles"]=={"employee":10,"hr":3,"finance":3,"it_security":4}
    assert set(core["modes"])=={"bm25","faiss","hybrid"}
    assert core["configurations"]["rrf"]=={"k":60,"top_k":20}
    for mode,data in core["modes"].items():
        assert data["snapshot_id"]==evaluation.env.bundle.snapshot.candidate_snapshot_id
        assert data["metrics"]["eligible_positive_count"]==14
        assert data["metrics"]["excluded_no_positive_count"]==6
        assert [row["metrics"] is None for row in data["cases"]]==[not c.authorized_source_sections for c in diagnostic.read_cases(evaluation.raw)]
        for row in data["cases"]:
            for gold in row["gold_sections"]:
                assert evaluation.env.db.execute(
                    "SELECT count(*) FROM draft_parents WHERE document_id=? AND parent_id=?",
                    (gold["document_id"],gold["parent_id"])).fetchone()==(1,)
        assert data["role_pairs"][0]["negative_restricted_section_hits"]==0
        assert first["measurements"]["modes"][mode]["sample_count"]==20
    target=tmp_path/"runtime/reports/diagnostic.json"
    diagnostic.write_report(first,target,repository_root=tmp_path)
    restored=json.loads(target.read_bytes())
    diagnostic.validate_report(restored)
    assert restored["identity_sha256"]==first["identity_sha256"]
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.write_report(first,ROOT/"eval/benchmarks/bm25_public_smoke.json",repository_root=ROOT)


@pytest.mark.parametrize("defect",["json","duplicate-id","approval","revalidation","missing","role","language",
                                   "grade","family-query","family-role","expected","blank-id","duplicate-key"])
def test_strict_dataset_rejections(evaluation,defect):
    records=[json.loads(line) for line in evaluation.raw.splitlines()]
    if defect=="json":
        raw=b"not json"
    elif defect=="duplicate-key":
        raw=evaluation.raw.replace(b'"language":"fr"',b'"language":"fr","language":"en"',1)
    else:
        if defect=="duplicate-id":
            records[1]["query_id"]=records[0]["query_id"]
        elif defect=="missing":
            del records[0]["required_facts"]
        elif defect=="grade":
            records[0]["authorized_source_sections"][0]["relevance_grade"]=True
        elif defect=="family-query":
            records[3]["query"]="different question"
        elif defect=="family-role":
            records[3]["active_role"]="hr"
        elif defect=="blank-id":
            records[0]["query_id"]=" "
        else:
            field,value={"approval":("annotation_review_status","pending_human_review"),
                         "revalidation":("revalidation_required",True),"role":("active_role","owner"),
                         "language":("language","de"),"expected":("expected_status","refusal")}[defect]
            records[0][field]=value
        raw=encoded(records)
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.read_cases(raw)


@pytest.mark.parametrize("defect",["document","heading","superseded","duplicate-section","unauthorized"])
def test_gold_rejections_before_retrieval(evaluation,monkeypatch,defect):
    records=[json.loads(line) for line in evaluation.raw.splitlines()]
    label=records[0]["authorized_source_sections"][0]
    if defect=="document":
        label["document_id"]="MISSING-001-FR-V1"
    elif defect=="heading":
        label["section_heading"]="missing heading"
    elif defect=="superseded":
        label["document_id"]="LEAVE-001-FR-V1"
    elif defect=="duplicate-section":
        records[0]["authorized_source_sections"].append(deepcopy(label))
    else:
        records[1]["active_role"]="employee"
    def forbidden(*args,**kw):
        pytest.fail("invalid gold reached retrieval")
    monkeypatch.setattr(diagnostic.role_bm25,"search_role_bm25",forbidden)
    monkeypatch.setattr(diagnostic.role_faiss,"search_role_faiss",forbidden)
    with pytest.raises(diagnostic.DiagnosticError) as raised:
        run(evaluation,raw=encoded(records))
    if defect=="unauthorized":
        assert raised.value.unauthorized_hit_count==raised.value.violating_query_count==0


def test_ambiguous_gold_heading_uses_exact_sql_resolution(evaluation):
    cases=diagnostic.read_cases(evaluation.raw)
    contexts={case.query_id:evaluation.provider(case.active_role) for case in cases}
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE draft_parents(document_id, parent_id, heading)")
        label=cases[0].authorized_source_sections[0]
        db.executemany("INSERT INTO draft_parents VALUES(?,?,?)",
                       [(label.document_id,"parent-a",label.section_heading),
                        (label.document_id,"parent-b",label.section_heading)])
        with pytest.raises(ValueError,match="GOLD_SECTION_AMBIGUOUS"):
            diagnostic._gold(db,cases,evaluation.env.bm25,contexts)


def test_report_schema_tampering_rejected(evaluation):
    report=run(evaluation)
    report["deterministic"]["evaluated_count"]=5
    with pytest.raises(diagnostic.DiagnosticError,match="INVALID_REPORT"):
        diagnostic.validate_report(report)
    report=run(evaluation)
    data=report["deterministic"]["modes"]["bm25"]
    data["cases"][0]["metrics"]["recall_at_5"]=0.123456
    data["metrics"]=diagnostic.macro_metrics([row["metrics"] for row in data["cases"]])
    report["identity_sha256"]=sha256(diagnostic.canonical(report["deterministic"])).hexdigest()
    with pytest.raises(diagnostic.DiagnosticError,match="INVALID_REPORT"):
        diagnostic.validate_report(report)


def test_evaluator_does_not_read_held_out_or_other_dataset_files(evaluation,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail("the evaluator accessed a dataset path outside its supplied development bytes")
    monkeypatch.setattr(Path,"read_bytes",forbidden)
    report=run(evaluation)
    assert report["deterministic"]["held_out_evaluated_count"]==0
    assert diagnostic.DATASET_PATH==Path("eval/datasets/dev_candidates.jsonl")


def test_gold_resolution_uses_private_validated_sqlite_image(evaluation,monkeypatch):
    original=diagnostic._gold
    images=[]
    def observe(image,*args):
        assert image is not evaluation.env.db and not image.in_transaction
        images.append(image)
        return original(image,*args)
    monkeypatch.setattr(diagnostic,"_gold",observe)
    before=evaluation.env.db.execute("SELECT count(*) FROM draft_children").fetchone()
    report=run(evaluation)
    assert len(images)==1 and report["deterministic"]["snapshot"]["child_count"]==before[0]
    assert evaluation.env.db.execute("SELECT count(*) FROM draft_children").fetchone()==before


def test_historical_seed_retains_incomplete_coverage_and_three_positives(evaluation):
    raw=(ROOT/"eval/datasets/dev_seed_five.jsonl").read_bytes()
    report=run(evaluation,raw=raw)
    core=report["deterministic"]
    assert core["status"]=="INCOMPLETE_DEVELOPMENT_COVERAGE"
    assert core["evaluated_count"]==5 and core["annotation_gap"]==15
    assert core["positive_metric_denominator"]==3 and core["no_positive_count"]==2
    for data in core["modes"].values():
        assert data["metrics"]["eligible_positive_count"]==3
