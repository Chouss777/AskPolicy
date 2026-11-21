"""Owner-approved development conversion: bytes, provenance and source ACLs."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from security.test_auth import secured
from askpolicy.retrieval import bm25_smoke as seed
from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
from askpolicy.ingestion.chunking import extract_policy_chunks
from askpolicy.security.acl import can_read_document
from askpolicy.storage.eligibility import select_policy_versions
import dev_retrieval_diagnostic as diagnostic
from retrieval_metrics import section_metrics

ORIGINAL = "dce7427c1e096b5d0cd81f6b928883a0ef2c707f0cfa6dce7826d66f9aff916c"
FINAL = "d3280f8570ca81f2f1b2c300e9e2c1385a1d5113f07d7f42d1f286331551dc5b"
PROPOSAL = "6cf013bfd4b1d16108f03d2f1808e7a3dc771201694a3adf69a49ef3048f893c"
REVIEW = "452cead212fce6db1efed682187b400d9681ae6ea8e5e0d898093ad0c4e9b09a"
FIELDS = set("query_id question_family_id language active_role query expected_status authorized_source_sections required_facts case_type annotation_review_status revalidation_required".split())


@pytest.fixture(scope="module")
def dataset():
    raw = (ROOT / diagnostic.DATASET_PATH).read_bytes()
    fixture = (ROOT / seed.DATASET).read_bytes()
    approval = json.loads((ROOT / "eval/datasets/dev_annotation_approval.json").read_bytes())
    return raw, fixture, approval, diagnostic.read_cases(raw)


def test_original_five_bytes_and_immutable_fixture(dataset):
    raw, fixture, _, _ = dataset
    assert seed.DATASET == Path("eval/datasets/dev_seed_five.jsonl")
    assert sha256(fixture).hexdigest() == ORIGINAL
    assert raw.startswith(fixture)
    assert b"".join(raw.splitlines(keepends=True)[:5]) == fixture
    assert len(fixture.splitlines()) == 5
    assert sha256(raw).hexdigest() == FINAL
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")


def test_all_twenty_strict_records_and_approved_flags(dataset):
    raw, _, _, cases = dataset
    assert len(cases) == 20
    assert [c.query_id for c in cases] == [f"dev-candidate-{i:03}" for i in range(1, 21)]
    assert [c.query_id for c in cases[5:]] == [f"dev-candidate-{i:03}" for i in range(6, 21)]
    for record in raw.splitlines():
        value = json.loads(record)
        assert set(value) == FIELDS
        parsed = seed._DevelopmentCase.model_validate(value)
        assert parsed.annotation_review_status == "human_approved"
        assert parsed.revalidation_required is False


def test_exact_owner_approved_inventory(dataset):
    _, _, approval, cases = dataset
    assert Counter(c.language for c in cases) == {"fr": 8, "en": 6, "ar": 6}
    assert Counter(c.active_role for c in cases) == {"employee": 10, "hr": 3, "finance": 3, "it_security": 4}
    assert Counter(c.case_type for c in cases) == {"answerable": 11, "unanswerable": 3, "role_sensitive": 6}
    assert sum(bool(c.authorized_source_sections) for c in cases) == 14
    assert sum(not c.authorized_source_sections for c in cases) == 6
    assert len({c.question_family_id for c in cases}) == 17
    assert sum(len(c.authorized_source_sections) for c in cases[5:]) == 47
    assert sum(len(c.authorized_source_sections) for c in cases) == 50
    assert approval["counts"] == dict(executions=20, positive_gold_executions=14,
        no_positive_executions=6, case_types=dict(answerable=11,unanswerable=3,role_sensitive=6),
        languages=dict(fr=8,en=6,ar=6), roles=dict(employee=10,hr=3,finance=3,it_security=4),
        question_families=17,role_sensitive_pairs=3,new_positive_source_references=47,
        total_positive_source_references=50)


@pytest.mark.parametrize("positive,negative", [(3,4),(17,18),(19,20)])
def test_complete_pairs_stay_in_development(dataset, positive, negative):
    cases = dataset[3]
    first, second = cases[positive-1], cases[negative-1]
    assert first.question_family_id == second.question_family_id
    assert first.query == second.query and first.language == second.language
    assert first.active_role != second.active_role
    assert first.case_type == second.case_type == "role_sensitive"
    assert first.expected_status == "answer" and second.expected_status == "refusal"
    assert first.authorized_source_sections and first.required_facts
    assert second.authorized_source_sections == second.required_facts == []
    assert sum(c.question_family_id == first.question_family_id for c in cases) == 2


def test_case_013_exact_gold_and_contextual_correction(dataset):
    _, _, approval, cases = dataset
    case = cases[12]
    actual = [s.model_dump() for s in case.authorized_source_sections]
    expected = [dict(document_id=f"EXPENSES-001-{lang}-V2",section_heading=heading,relevance_grade=3)
                for lang,heading in (("AR","4. المراجعة والموافقة"),
                    ("EN","4. Review and approval"),("FR","4. Vérification et approbation"))]
    assert actual == expected
    correction = approval["case_013_correction"]
    assert correction["retained_positive_gold"] == expected
    assert correction["contextual_references"] == [
        dict(document_id=f"EXPENSES-001-{lang}-V2",section_heading=heading,relevance_grade=2)
        for lang,heading in (("AR","1. الهدف ونطاق التطبيق"),("EN","1. Purpose and scope"),
                            ("FR","1. Objet et champ d’application"))]
    assert correction["contextual_treatment"] == "CONTEXTUAL_ONLY_NOT_POSITIVE_GOLD"
    assert correction["other_canonical_fields_unchanged"] is True
    assert not any(s.section_heading.startswith("1. ") for s in case.authorized_source_sections)


def test_approval_manifest_hashes_and_explicit_owner_provenance(dataset):
    raw, fixture, approval, cases = dataset
    assert approval["format"] == "askpolicy_development_annotation_approval" and approval["version"] == 1
    assert approval["baseline_commit"] == "5ca36a4880452a23e0f6c18545bd83254e66bcdf"
    assert approval["original_five_case_sha256"] == approval["seed_fixture_sha256"] == sha256(fixture).hexdigest() == ORIGINAL
    assert approval["canonical_dataset_sha256"] == sha256(raw).hexdigest() == FINAL
    assert approval["step109_proposal_sha256"] == PROPOSAL
    assert approval["step109_review_packet_sha256"] == REVIEW
    assert approval["approval_source"] == "explicit_project_owner_chat_decision"
    assert approval["approval_date"] == "2026-11-21" and approval["trusted_validity_date"] == "2026-09-01"
    assert approval["approved_new_ids"] == [c.query_id for c in cases[5:]]
    assert approval["held_out_labels_untouched"] is True
    assert approval["model_deployment_approved"] is False
    actual = {c.query_id: sha256(json.dumps(c.model_dump(),ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest() for c in cases}
    assert approval["canonical_record_sha256"] == actual


def test_all_gold_exact_unique_current_authorized_and_facts_source_backed(dataset, secured):
    cases = dataset[3]
    with sqlite3.connect(":memory:") as db:
        counts = ingest_manifest_to_draft(ROOT, db)
        assert counts.documents == 26
        docs = seed._read_documents(db)
        eligibility = select_policy_versions([d.metadata for d in docs.values()],validity_date=diagnostic.DAY)
        assert len(eligibility.eligible_document_ids) == 20 and len(eligibility.exclusions) == 6
        excluded = {e.document_id for e in eligibility.exclusions}
        for case in cases:
            context = secured.authenticate(secured.sign(roles=[case.active_role]),case.active_role)
            texts = []
            for label in case.authorized_source_sections:
                assert label.document_id in eligibility.eligible_document_ids
                assert label.document_id not in excluded
                meta = docs[label.document_id].metadata
                assert can_read_document(context,classification=meta.classification,allowed_roles=meta.allowed_roles)
                assert label.relevance_grade in (2,3)
                identity = seed._resolve_section(db,label)
                matches = db.execute("SELECT parent_id,text FROM draft_parents WHERE document_id=? AND heading=?",(label.document_id,label.section_heading)).fetchall()
                assert len(matches)==1 and matches[0][0]==identity.parent_id
                path=ROOT/f"data/policies/{meta.language}/{meta.document_id}.md"
                extraction=extract_policy_chunks(path.read_text(encoding="utf-8"),source_path=path.relative_to(ROOT).as_posix())
                assert any(p.parent_id==identity.parent_id and p.text==matches[0][1] for p in extraction.parents)
                texts.append(matches[0][1])
            if case.expected_status == "refusal":
                assert case.authorized_source_sections == case.required_facts == []
            else:
                assert texts and case.required_facts
                # Exact text is evidence of source consistency, not semantic certification.
                for fact in case.required_facts:
                    reviewed_seed_excerpt = {
                        "dev-candidate-001": "Plafond de 850 MAD par nuit",
                        "dev-candidate-002": "au moins cinq jours ouvrés avant le départ",
                    }.get(case.query_id, fact)
                    assert any(reviewed_seed_excerpt in text for text in texts)


def test_section_recall_keeps_large_gold_sets(dataset):
    cases = dataset[3]
    large = [c for c in cases if len(c.authorized_source_sections)>5]
    assert {c.query_id for c in large} == {"dev-candidate-010","dev-candidate-012","dev-candidate-014","dev-candidate-017"}
    for c in large:
        gold = {(s.document_id,s.section_heading):s.relevance_grade for s in c.authorized_source_sections}
        assert len(gold)==len(c.authorized_source_sections)
        assert section_metrics(list(gold)[:5],gold)["recall_at_5"] == 5/len(gold)
    assert section_metrics([], {}) is None


def test_seed_public_smoke_retains_exact_benchmark(dataset):
    report = seed.evaluate_public_smoke(ROOT)
    assert report["dataset_sha256"] == ORIGINAL
    assert report["evaluated_positive_case_count"] == 1
    assert report["evaluated_query_ids"] == ["dev-candidate-001"]
    assert report["indexed_document_ids"] == ["TRAVEL-001-EN-V1","TRAVEL-001-FR-V1"]
    assert report["excluded_case_count"] == 4
    assert report == json.loads((ROOT/"eval/benchmarks/bm25_public_smoke.json").read_bytes())


def test_approval_and_cases_have_no_held_out_path_references(dataset):
    raw, _, approval, _ = dataset
    def values(value):
        if isinstance(value,dict):
            return [s for v in value.values() for s in values(v)]
        if isinstance(value,list):
            return [s for v in value for s in values(v)]
        return [value] if isinstance(value,str) else []
    assert not any("held" in s.casefold() for s in values(approval)+values(json.loads(b"["+b",".join(raw.splitlines())+b"]")))


@pytest.mark.parametrize("count,status",[(5,"INCOMPLETE_DEVELOPMENT_COVERAGE"),(20,"DEVELOPMENT_COVERAGE_COMPLETE")])
def test_coverage_status(count,status):
    assert diagnostic.coverage_status(count)==status


@pytest.mark.parametrize("count",[0,21,True,20.0])
def test_invalid_coverage_counts_rejected(count):
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.coverage_status(count)


def test_strict_parser_rejects_more_than_twenty(dataset):
    records=[json.loads(l) for l in dataset[0].splitlines()]
    extra=dict(records[0],query_id="extra",question_family_id="extra")
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.read_cases(b"\n".join(json.dumps(c).encode() for c in records+[extra]))


def test_explicit_lf_attributes_preserve_fresh_checkout_with_autocrlf(tmp_path):
    paths = (
        ".gitattributes",
        "eval/datasets/dev_candidates.jsonl",
        "eval/datasets/dev_seed_five.jsonl",
        "eval/datasets/dev_annotation_approval.json",
    )
    expected_rules = "".join(f"{path} text eol=lf\n" for path in paths).encode("utf-8")
    assert (ROOT / ".gitattributes").read_bytes() == expected_rules
    originals = {path: (ROOT / path).read_bytes() for path in paths}
    assert all(b"\r" not in raw and raw.endswith(b"\n") for raw in originals.values())

    repository = tmp_path / "isolated-repository"
    checkout = tmp_path / "fresh-checkout"
    repository.mkdir()
    checkout.mkdir()
    # Git writes only to this disposable repository and its fresh output.
    # No commit or operation against the user's index/worktree is required.
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_CONFIG", "GIT_CONFIG_COUNT"):
        env.pop(key, None)
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_ATTR_NOSYSTEM="1")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], env=env)

    git("init", "--quiet")
    git("config", "core.autocrlf", "true")
    assert git("config", "--get", "core.autocrlf").strip() == b"true"
    for path, raw in originals.items():
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (repository / "control.txt").write_bytes(b"first\nsecond\n")
    attributes = git("check-attr", "text", "eol", "--", *paths).decode("utf-8")
    assert attributes.splitlines() == [
        line for path in paths for line in (f"{path}: text: set", f"{path}: eol: lf")
    ]
    git("add", "--", *paths, "control.txt")
    git("checkout-index", "--all", "--prefix=" + checkout.as_posix() + "/")
    assert (checkout / "control.txt").read_bytes() == b"first\r\nsecond\r\n"
    for path, original in originals.items():
        restored = (checkout / path).read_bytes()
        assert restored == original
        assert sha256(restored).digest() == sha256(original).digest()
    assert sha256((checkout / paths[1]).read_bytes()).hexdigest() == FINAL
    assert sha256((checkout / paths[2]).read_bytes()).hexdigest() == ORIGINAL
    seed_bytes = (checkout / paths[2]).read_bytes()
    assert b"".join((checkout / paths[1]).read_bytes().splitlines(keepends=True)[:5]) == seed_bytes
