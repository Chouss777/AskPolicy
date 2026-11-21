from contextlib import closing
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
from secrets import token_urlsafe
import shutil
import sqlite3
import unicodedata

from pydantic import SecretStr
import pytest


ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("eval/benchmarks/bm25_public_smoke.json")
TRAVEL = {"TRAVEL-001-EN-V1", "TRAVEL-001-FR-V1"}
EMPLOYEE = "employee"
DAY = date(2026, 9, 1)


@pytest.fixture(scope="module")
def smoke():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.retrieval import bm25_smoke

        yield bm25_smoke


@pytest.fixture(scope="module")
def draft(smoke):
    with closing(sqlite3.connect(":memory:")) as connection:
        manifest = (ROOT / smoke.MANIFEST).read_bytes()
        counts = smoke.ingest_manifest_to_draft(ROOT, connection)
        identity = smoke.compute_draft_snapshot_identity(
            connection, manifest=manifest, config=smoke.DraftSnapshotConfig(),
        )
        documents = smoke._read_documents(connection)
        eligibility = smoke.select_policy_versions([doc.metadata for doc in documents.values()], validity_date=DAY)
        yield connection, counts, identity, documents, eligibility


@pytest.fixture(scope="module")
def measured(smoke):
    return smoke.evaluate_public_smoke(ROOT)


def test_full_validated_corpus_and_unpublished_identity(smoke, draft, measured):
    _, counts, identity, documents, eligibility = draft
    assert (counts.documents, counts.parents, counts.children) == (26, 161, 387)
    assert measured["draft_counts"] == {"documents": 26, "parents": 161, "children": 387}
    assert measured["snapshot_status"] == identity.status == "UNPUBLISHED"
    assert measured["candidate_draft_snapshot_id"] == identity.candidate_snapshot_id
    assert measured["canonical_source_manifest_sha256"] == identity.source_manifest_hash
    assert measured["draft_snapshot_config_hash"] == identity.config_hash
    assert measured["source_manifest_sha256"] == sha256((ROOT / smoke.MANIFEST).read_bytes()).hexdigest()
    assert measured["validity_date"] == eligibility.validity_date.isoformat() == "2026-09-01"
    assert len(eligibility.eligible_document_ids) == 20
    historical = {doc.metadata.supersedes for doc in documents.values() if doc.metadata.supersedes is not None}
    assert len(historical) == 6
    assert not (historical & set(eligibility.eligible_document_ids))


def test_public_index_only_tokenizes_exact_stored_travel_children(smoke, draft, monkeypatch):
    connection, _, _, documents, eligibility = draft
    captured_text = []
    token_batches = []
    tokenize = smoke.bm25s.tokenize
    index = smoke.bm25s.BM25.index

    def observe_tokenize(texts, **options):
        captured_text.append(list(texts))
        return tokenize(texts, **options)

    def observe_index(retriever, tokens, **options):
        token_batches.append(tokens)
        return index(retriever, tokens, **options)

    monkeypatch.setattr(smoke.bm25s, "tokenize", observe_tokenize)
    monkeypatch.setattr(smoke.bm25s.BM25, "index", observe_index)
    report = smoke.evaluate_public_smoke(ROOT)
    public = smoke._load_public_chunks(connection, documents, eligibility, EMPLOYEE)
    actual_rows = connection.execute(
        "SELECT child_id, document_id, parent_id, start, text FROM draft_children "
        "WHERE document_id IN (?, ?) ORDER BY document_id, start, child_id",
        sorted(TRAVEL),
    ).fetchall()
    assert [(c.child_id, c.section.document_id, c.section.parent_id, c.start, c.text) for c in public] == actual_rows
    assert all(c.document == documents[c.section.document_id] for c in public)
    assert all(c.document.metadata.classification == "PUBLIC" and EMPLOYEE in c.document.metadata.allowed_roles for c in public)
    assert set(report["indexed_document_ids"]) == TRAVEL
    assert report["indexed_document_count"] == len(TRAVEL)
    assert report["indexed_child_count"] == len(actual_rows)
    historical_ids = {entry.document_id for entry in eligibility.exclusions}
    leave_and_history = {child_id for child_id, document_id in connection.execute(
        "SELECT child_id, document_id FROM draft_children"
    ) if document_id.startswith("LEAVE-") or document_id in historical_ids}
    assert leave_and_history
    assert not (leave_and_history & {c.child_id for c in public})
    expected_texts = [unicodedata.normalize("NFC", row[-1]) for row in actual_rows]
    seed = json.loads((ROOT / smoke.DATASET).read_bytes().splitlines()[0])
    assert captured_text == [expected_texts, [unicodedata.normalize("NFC", seed["query"])]]
    assert token_batches == [tokenize(
        expected_texts, lower=True, token_pattern=r"(?u)\b\w\w+\b",
        stopwords=[], stemmer=None, return_ids=False, show_progress=False,
    )]


def test_exact_seed_and_denominator(smoke, draft, measured):
    connection, _, _, documents, eligibility = draft
    cases = smoke._read_cases((ROOT / smoke.DATASET).read_bytes())
    case, gold, excluded = smoke._select_case(connection, cases, documents, eligibility)
    assert (case.query_id, case.language, case.active_role, case.expected_status) == (
        "dev-candidate-001", "fr", "employee", "answer",
    )
    assert all(
        candidate.active_role == "hr"
        and documents[candidate.authorized_source_sections[0].document_id].metadata.classification == "RESTRICTED"
        for candidate in cases[1:3]
    )
    assert [candidate.expected_status for candidate in cases[3:]] == ["refusal", "refusal"]
    assert excluded == [
        {"query_id": "dev-candidate-002", "reason": "GOLD_NOT_PUBLIC"},
        {"query_id": "dev-candidate-003", "reason": "GOLD_NOT_PUBLIC"},
        {"query_id": "dev-candidate-004", "reason": "REFUSAL_NO_POSITIVE_GOLD"},
        {"query_id": "dev-candidate-005", "reason": "REFUSAL_NO_POSITIVE_GOLD"},
    ]
    assert measured["evaluated_positive_case_count"] == 1
    assert measured["evaluated_query_ids"] == [case.query_id]
    assert measured["excluded_cases"] == excluded
    section, = gold
    assert (section.document_id, section.heading) == ("TRAVEL-001-FR-V1", "4. Transport et hébergement")
    assert connection.execute(
        "SELECT document_id, heading FROM draft_parents WHERE parent_id = ?", (section.parent_id,),
    ).fetchone() == (section.document_id, section.heading)
    assert measured["cases"][0]["gold_section_count"] == 1
    assert measured["cases"][0]["gold_sections"] == [{
        "document_id": section.document_id, "parent_id": section.parent_id, "section_heading": section.heading,
    }]


def _fixture_extraction(family, *, version=1, changes=None, paragraphs=1):
    from askpolicy.ingestion.chunking import extract_policy_chunks

    document_id = f"{family}-FR-V{version}"
    metadata = {
        "policy_id": family, "document_id": document_id, "version": version, "language": "fr",
        "title": "Synthetic pressure test", "department": "Operations", "classification": "PUBLIC",
        "allowed_roles": ["employee"], "effective_date": "2026-08-01", "expires_at": None, "supersedes": None,
        "approval_status": "approved", "source_type": "synthetic_policy", "source_license": "project_owned",
        **(changes or {}),
    }
    frontmatter = "".join(f"{name}: {json.dumps(value)}\n" for name, value in metadata.items())
    text = "---\n" + frontmatter + "---\n# Synthetic heading\n\n" + "capsule evidence\n\n" * paragraphs
    return extract_policy_chunks(text, source_path=f"data/policies/fr/{document_id}.md")


def _fixture_draft(smoke, connection, extractions):
    from askpolicy.storage.sqlite import initialize_draft_schema, persist_draft_extractions

    initialize_draft_schema(connection)
    persist_draft_extractions(connection, extractions)
    documents = smoke._read_documents(connection)
    eligibility = smoke.select_policy_versions([doc.metadata for doc in documents.values()], validity_date=DAY)
    return documents, eligibility


@pytest.mark.parametrize("changes, reason", [
    ({"effective_date": "2026-09-02"}, "NOT_YET_EFFECTIVE"),
    ({"expires_at": "2026-09-01"}, "EXPIRED"),
    ({"approval_status": "not_approved"}, "NOT_APPROVED"),
    ({"classification": "RESTRICTED"}, None),
    ({"allowed_roles": ["hr"]}, None),
], ids=["future", "expired", "unapproved", "restricted", "ungranted"])
def test_noncandidate_to_public_index_rejected_before_tokenization(smoke, changes, reason, monkeypatch):
    keep = _fixture_extraction("FIXTURE-901")
    exclude = _fixture_extraction("FIXTURE-902", changes=changes)
    with closing(sqlite3.connect(":memory:")) as connection:
        docs, eligibility = _fixture_draft(smoke, connection, [keep, exclude])
        bad_id = exclude.metadata.document_id
        if reason is not None:
            assert {entry.document_id: entry.reasons for entry in eligibility.exclusions}[bad_id] == (reason,)
        public = smoke._load_public_chunks(connection, docs, eligibility, EMPLOYEE)
        assert {item.section.document_id for item in public} == {keep.metadata.document_id}
        actual = smoke.rank_public_chunks(public, query="capsule", eligibility=eligibility, role=EMPLOYEE)
        assert {hit.section.document_id for hit in actual} == {keep.metadata.document_id}

        child = exclude.children[0]
        forbidden = smoke.EvidenceChunk(
            child.child_id, smoke.SectionIdentity(bad_id, child.parent_id, "Synthetic heading"),
            docs[bad_id], child.start, child.text,
        )

        def unexpected_tokenization(*args, **kwargs):
            pytest.fail("out-of-scope input reached tokenization")

        monkeypatch.setattr(smoke.bm25s, "tokenize", unexpected_tokenization)
        with pytest.raises(smoke.SmokeEvaluationError, match="^OUT_OF_SCOPE_INDEX_INPUT$"):
            smoke.rank_public_chunks((*public, forbidden), query="capsule", eligibility=eligibility, role=EMPLOYEE)


def test_superseded_public_child_never_enters_the_index(smoke):
    old = _fixture_extraction("FIXTURE-901")
    new = _fixture_extraction("FIXTURE-901", version=2, changes={"supersedes": old.metadata.document_id})
    with closing(sqlite3.connect(":memory:")) as connection:
        docs, eligibility = _fixture_draft(smoke, connection, [old, new])
        assert eligibility.exclusions[0].reasons == ("SUPERSEDED",)
        public = smoke._load_public_chunks(connection, docs, eligibility, EMPLOYEE)
        assert {item.section.document_id for item in public} == {new.metadata.document_id}
        result = smoke.rank_public_chunks(public, query="capsule", eligibility=eligibility, role=EMPLOYEE)
        assert {hit.section.document_id for hit in result} == {new.metadata.document_id}


def test_real_bm25s_ties_use_chunk_id_and_truncate_at_five(smoke):
    doc = _fixture_extraction("FIXTURE-901", paragraphs=7)
    with closing(sqlite3.connect(":memory:")) as connection:
        docs, eligibility = _fixture_draft(smoke, connection, [doc])
        evidence = smoke._load_public_chunks(connection, docs, eligibility, EMPLOYEE)
    expected = sorted(chunk.child_id for chunk in evidence if chunk.text.startswith("capsule"))
    assert len(expected) == 7
    rank = smoke.rank_public_chunks(evidence, query="capsule", eligibility=eligibility, role=EMPLOYEE)
    assert [hit.child_id for hit in rank] == expected[:5]
    assert len({hit.score for hit in rank}) == 1 and rank[0].score > 0
    reversed_rank = smoke.rank_public_chunks(evidence[::-1], query="capsule", eligibility=eligibility, role=EMPLOYEE)
    assert reversed_rank == rank
    assert smoke.recall_at_5([rank[0].section], rank) == smoke.recall_at_5([rank[0].section], reversed_rank)


def test_no_bm25s_overlap_means_no_hits_and_zero_recall(smoke, draft):
    connection, _, _, documents, eligibility = draft
    evidence = smoke._load_public_chunks(connection, documents, eligibility, EMPLOYEE)
    rank = smoke.rank_public_chunks(evidence, query="zzznomatchxxx", eligibility=eligibility, role=EMPLOYEE)
    assert rank == ()
    assert smoke.recall_at_5([evidence[0].section], rank) == 0.0
    assert not smoke.recall_at_5([evidence[0].section], rank) > smoke.READINESS_THRESHOLD


def test_recall_credits_distinct_document_parent_pairs_within_rank_five(smoke):
    gold_a = smoke.SectionIdentity("A", "a", "Section")
    gold_b = smoke.SectionIdentity("B", "b", "Section")
    wrong_doc = smoke.SectionIdentity("C", "a", "Section")
    wrong_parent = smoke.SectionIdentity("B", "other", "Section")
    hit_a = smoke.RankedChunk("duplicate", gold_a, 10.0)
    hit_b = smoke.RankedChunk("match-b", gold_b, 1.0)
    ranking = [
        hit_a, hit_a, smoke.RankedChunk("same-parent", gold_a, 9.0),
        smoke.RankedChunk("wrong-doc", wrong_doc, 8.0), smoke.RankedChunk("wrong-parent", wrong_parent, 7.0), hit_b,
    ]
    assert smoke.recall_at_5([gold_a, gold_b, gold_a], ranking) == 0.5
    assert smoke.recall_at_5([gold_b], ranking) == 0.0
    assert smoke.recall_at_5([gold_a, gold_b], [hit_b, *ranking]) == 1.0
    with pytest.raises(smoke.SmokeEvaluationError, match="^NO_POSITIVE_GOLD$"):
        smoke.recall_at_5([], ranking)


@pytest.mark.parametrize("heading", ["", "4. Transport et hebergement"], ids=["empty-heading", "nonexact-heading"])
def test_missing_or_invalid_gold_reference_is_rejected(smoke, draft, heading):
    connection, _, _, documents, eligibility = draft
    rows = [json.loads(line) for line in (ROOT / smoke.DATASET).read_bytes().splitlines()]
    rows[0]["authorized_source_sections"][0]["section_heading"] = heading
    raw = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
    if not heading:
        with pytest.raises(smoke.SmokeEvaluationError, match="^INVALID_SEED_DATASET$"):
            smoke._read_cases(raw)
    else:
        with pytest.raises(smoke.SmokeEvaluationError, match="^GOLD_SECTION_MISSING$"):
            smoke._select_case(connection, smoke._read_cases(raw), documents, eligibility)


def test_gold_heading_in_another_document_does_not_resolve(smoke, draft):
    connection, _, _, documents, eligibility = draft
    rows = [json.loads(line) for line in (ROOT / smoke.DATASET).read_bytes().splitlines()]
    rows[0]["authorized_source_sections"][0]["document_id"] = "TRAVEL-001-EN-V1"
    raw = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
    with pytest.raises(smoke.SmokeEvaluationError, match="^GOLD_SECTION_MISSING$"):
        smoke._select_case(connection, smoke._read_cases(raw), documents, eligibility)


def test_ambiguous_heading_is_rejected(smoke):
    extraction = _fixture_extraction("FIXTURE-901")
    from askpolicy.ingestion.chunking import extract_policy_chunks

    source = extraction.canonical_source + "# Synthetic heading\n\nSeparate text.\n"
    ambiguous = extract_policy_chunks(source, source_path=extraction.source_path)
    label = smoke._GoldLabel(document_id=ambiguous.metadata.document_id, section_heading="Synthetic heading", relevance_grade=3)
    with closing(sqlite3.connect(":memory:")) as connection:
        _fixture_draft(smoke, connection, [ambiguous])
        with pytest.raises(smoke.SmokeEvaluationError, match="^GOLD_SECTION_AMBIGUOUS$"):
            smoke._resolve_section(connection, label)


def test_no_eligible_positive_seed_is_an_error(smoke, draft):
    connection, _, _, documents, eligibility = draft
    cases = smoke._read_cases((ROOT / smoke.DATASET).read_bytes())
    assert cases[0].query_id == "dev-candidate-001"
    with pytest.raises(smoke.SmokeEvaluationError, match="^NO_ELIGIBLE_POSITIVE_CASES$"):
        smoke._select_case(connection, cases[1:], documents, eligibility)


def test_reordered_approved_corpus_gives_same_snapshot_ranking_and_metric(smoke, measured, tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    (tmp_path / smoke.DATASET).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / smoke.DATASET, tmp_path / smoke.DATASET)
    manifest = json.loads((ROOT / smoke.MANIFEST).read_bytes())
    manifest["documents"].reverse()
    (tmp_path / smoke.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    reordered = smoke.evaluate_public_smoke(tmp_path)
    assert reordered.pop("source_manifest_sha256") != measured["source_manifest_sha256"]
    assert reordered == {key: value for key, value in measured.items() if key != "source_manifest_sha256"}


def test_report_measurement_gate_and_artifact_match(smoke, measured, tmp_path):
    case, = measured["cases"]
    gold = {(section["document_id"], section["parent_id"]) for section in case["gold_sections"]}
    hits = {(hit["document_id"], hit["parent_id"]) for hit in case["ranking"]}
    assert len(gold) == case["gold_section_count"] == 1
    assert len(case["ranking"]) <= 5
    assert [hit["rank"] for hit in case["ranking"]] == list(range(1, len(case["ranking"]) + 1))
    assert len({hit["child_id"] for hit in case["ranking"]}) == len(case["ranking"])
    expected = len(gold & hits) / len(gold)
    assert case["recall_at_5"] == measured["macro_recall_at_5"] == expected
    assert measured["readiness_gate"] == {"operator": ">", "threshold": 0.0, "passed": expected > 0.0}
    artifact = tmp_path / "report.json"
    artifact.write_bytes(smoke.encode_report(measured))
    assert json.loads(artifact.read_bytes()) == measured
    if (ROOT / REPORT).is_file():
        assert (ROOT / REPORT).read_bytes() == artifact.read_bytes()


def test_report_omits_queries_required_facts_and_restricted_passages(smoke, draft, measured):
    connection, _, _, _, _ = draft
    raw = smoke.encode_report(measured).decode("utf-8")
    cases = smoke._read_cases((ROOT / smoke.DATASET).read_bytes())
    assert all(case.query not in raw and all(fact not in raw for fact in case.required_facts) for case in cases)
    assert "LEAVE-001" not in raw
    restricted = connection.execute(
        "SELECT c.text FROM draft_children AS c JOIN draft_documents AS d USING (document_id) "
        "WHERE d.classification = 'RESTRICTED' AND c.kind <> 'heading' AND length(c.text) > 60",
    ).fetchall()
    assert restricted and all(json.dumps(text.strip(), ensure_ascii=False)[1:-1] not in raw for text, in restricted)
    assert not re.search(r'"(?:text|canonical_source)"\s*:', raw)


def test_offline_index_cannot_grant_read_access_or_add_api_routes(smoke, draft, measured):
    from askpolicy.api.app import create_app
    from askpolicy.auth.jwt import JWTConfig
    from askpolicy.security.acl import can_read_document

    _, _, _, documents, _ = draft
    metadata = documents[measured["indexed_document_ids"][0]].metadata
    assert not can_read_document(None, classification=metadata.classification, allowed_roles=metadata.allowed_roles)
    assert not can_read_document(EMPLOYEE, classification=metadata.classification, allowed_roles=metadata.allowed_roles)
    app = create_app(jwt_config=JWTConfig(SecretStr(token_urlsafe(48)), "smoke-test-issuer", "smoke-test-api", "HS256"))
    assert len(app.routes) == 2
    routes = {(route.path, frozenset(route.methods)): route for route in app.routes}
    assert set(routes) == {
        ("/auth/verify", frozenset({"GET"})),
        ("/retrieval/search", frozenset({"POST"})),
    }
    auth_route = routes[("/auth/verify", frozenset({"GET"}))]
    retrieval_route = routes[("/retrieval/search", frozenset({"POST"}))]
    dependency, = auth_route.dependant.dependencies
    assert retrieval_route.dependant.dependencies[0].call is dependency.call
    query_dependency = retrieval_route.dependant.dependencies[1]
    assert [item.call for item in query_dependency.dependencies] == [dependency.call]
