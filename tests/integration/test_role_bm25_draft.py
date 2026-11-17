from contextlib import closing
from dataclasses import FrozenInstanceError
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
import unicodedata

import pytest

from security.test_auth import secured


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("data/corpus_manifest.yaml")
DAY = date(2026, 9, 1)
TOMORROW = date(2026, 9, 2)
TRAVEL = "TRAVEL-001-EN-V1"
COUNTS = {"employee": 15, "hr": 18, "finance": 15, "it_security": 17}


@pytest.fixture(scope="module")
def role_bm25():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.retrieval import role_bm25 as module

        yield module


@pytest.fixture(scope="module")
def approved_draft(role_bm25):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    with closing(sqlite3.connect(":memory:")) as connection:
        ingest_manifest_to_draft(ROOT, connection)
        yield connection, (ROOT / MANIFEST).read_bytes()


@pytest.fixture
def draft(approved_draft):
    source, manifest = approved_draft
    with closing(sqlite3.connect(":memory:")) as connection:
        source.backup(connection)
        yield connection, manifest


def _build(module, connection, manifest, *, day=DAY, **options):
    return module.build_role_bm25_bundle(
        connection, manifest=manifest, snapshot_config=module.DraftSnapshotConfig(),
        validity_date=day, **options,
    )


def _metadata(bundle):
    return {index.metadata.role: index.metadata for index in bundle.indexes}


def _source_fixture(tmp_path, before, after):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    manifest_path = tmp_path / MANIFEST
    manifest = json.loads(manifest_path.read_bytes())
    record = next(record for record in manifest["documents"] if record["document_id"] == TRAVEL)
    path = tmp_path / record["source_path"]
    source = path.read_bytes().replace(b"\r\n", b"\n")
    assert source.count(before) == 1
    source = source.replace(before, after, 1)
    path.write_bytes(source)
    record["content_sha256"] = sha256(source).hexdigest()
    for field in ("allowed_roles", "approval_status", "effective_date", "expires_at"):
        if before.startswith(f"{field}: ".encode("ascii")):
            value = after.decode("utf-8").split(": ", 1)[1]
            record[field] = {
                "approved": "approved", "not_approved": "not_approved", "[hr]": ["hr"],
            }.get(value, value.strip('"'))
    raw = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    manifest_path.write_bytes(raw)
    return raw


def test_role_membership_matches_validated_sqlite_and_ingests_full_corpus(role_bm25, approved_draft, monkeypatch):
    from askpolicy.retrieval import bm25_core

    connection, manifest = approved_draft
    expected_snapshot = role_bm25.compute_draft_snapshot_identity(
        connection, manifest=manifest, config=role_bm25.DraftSnapshotConfig(),
    )
    rows = connection.execute(
        "SELECT c.child_id, d.document_id, d.allowed_roles, c.text FROM draft_children c "
        "JOIN draft_documents d USING (document_id) WHERE d.document_id NOT IN "
        "(SELECT supersedes FROM draft_documents WHERE supersedes IS NOT NULL) "
        "ORDER BY d.document_id, c.start, c.child_id",
    ).fetchall()
    tokenize = bm25_core.bm25s.tokenize
    captured = []

    def observe_tokenization(texts, **options):
        captured.append(list(texts))
        return tokenize(texts, **options)

    monkeypatch.setattr(bm25_core.bm25s, "tokenize", observe_tokenization)
    bundle = _build(role_bm25, connection, manifest, expected_snapshot_id=expected_snapshot.candidate_snapshot_id)
    assert bundle.snapshot == expected_snapshot
    assert bundle.snapshot.status == "UNPUBLISHED"
    assert (bundle.snapshot.document_count, bundle.snapshot.parent_count, bundle.snapshot.child_count) == (26, 161, 387)
    assert len(bundle.eligibility.eligible_document_ids) == 20 and len(bundle.eligibility.exclusions) == 6
    assert set(_metadata(bundle)) == set(COUNTS)
    assert len(captured) == 4
    excluded = {entry.document_id for entry in bundle.eligibility.exclusions}
    for position, index in enumerate(bundle.indexes):
        meta = index.metadata
        allowed = [row for row in rows if meta.role in json.loads(row[2])]
        assert meta.indexed_document_ids == tuple(sorted({row[1] for row in allowed}))
        assert meta.indexed_document_count == COUNTS[meta.role]
        assert meta.indexed_child_ids == tuple(sorted(row[0] for row in allowed))
        assert meta.indexed_child_count == len(allowed)
        assert len(set(meta.indexed_child_ids)) == len(allowed)
        assert not (excluded & set(meta.indexed_document_ids))
        assert meta.candidate_snapshot_id == bundle.snapshot.candidate_snapshot_id
        assert meta.source_manifest_hash == bundle.snapshot.source_manifest_hash
        assert meta.snapshot_config_hash == bundle.snapshot.config_hash
        assert meta.validity_date == DAY and meta.next_transition_date is None
        # Compare exact stored input without printing any policy text on failure.
        same_input = captured[position] == [unicodedata.normalize("NFC", row[3]) for row in allowed]
        assert same_input
    assert len({index.metadata.configuration_fingerprint for index in bundle.indexes}) == 1
    assert len({index.metadata.fingerprint for index in bundle.indexes}) == 4
    with pytest.raises(FrozenInstanceError):
        bundle.indexes[0].metadata.role = "global"
    for role in ("employee", "finance"):
        assert all(not doc.startswith(("LEAVE-", "ACCESS-")) for doc in _metadata(bundle)[role].indexed_document_ids)
    assert sum(doc.startswith("LEAVE-") for doc in _metadata(bundle)["hr"].indexed_document_ids) == 3
    assert sum(doc.startswith("ACCESS-") for doc in _metadata(bundle)["it_security"].indexed_document_ids) == 2


def test_two_clean_database_paths_and_row_orders_have_same_metadata(role_bm25, tmp_path):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    manifest = (ROOT / MANIFEST).read_bytes()
    data = json.loads(manifest)
    data["documents"].reverse()
    revised = json.dumps(data).encode("utf-8")
    alternate = tmp_path / "other"
    alternate.mkdir()
    with closing(sqlite3.connect(tmp_path / "first.sqlite")) as first, closing(sqlite3.connect(alternate / "other.db")) as second:
        ingest_manifest_to_draft(ROOT, first)
        ingest_manifest_to_draft(ROOT, second)
        for table in ("draft_documents", "draft_parents", "draft_children"):
            second.execute(f"UPDATE {table} SET rowid = -rowid")
        second.commit()
        second.execute("PRAGMA reverse_unordered_selects = ON")
        original = _build(role_bm25, first, manifest)
        reordered = _build(role_bm25, second, revised)
    assert original.snapshot == reordered.snapshot
    assert _metadata(original) == _metadata(reordered)
    assert tuple(index.metadata.fingerprint for index in original.indexes) == tuple(
        index.metadata.fingerprint for index in reordered.indexes
    )


def test_validation_and_extraction_use_the_same_private_image(role_bm25, tmp_path, monkeypatch):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    path = tmp_path / "source.sqlite"
    manifest = (ROOT / MANIFEST).read_bytes()
    with closing(sqlite3.connect(path)) as writer, closing(sqlite3.connect(path)) as caller:
        assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        ingest_manifest_to_draft(ROOT, writer)
        baseline = _build(role_bm25, caller, manifest)
        child_id, original = writer.execute(
            "SELECT child_id, text FROM draft_children WHERE document_id = ? ORDER BY start LIMIT 1", (TRAVEL,),
        ).fetchone()
        fingerprint = role_bm25.compute_draft_snapshot_identity
        read_chunks = role_bm25.read_draft_chunks
        images = []

        def mutate_caller_database_after_validation(image, **kwargs):
            assert image is not caller
            images.append(image)
            result = fingerprint(image, **kwargs)
            assert writer.execute(
                "UPDATE draft_children SET text = ? WHERE child_id = ?", ("Z" + original[1:], child_id),
            ).rowcount == 1
            writer.commit()
            return result

        def observe_read(image, documents):
            assert image is images[0]
            return read_chunks(image, documents)

        monkeypatch.setattr(role_bm25, "compute_draft_snapshot_identity", mutate_caller_database_after_validation)
        monkeypatch.setattr(role_bm25, "read_draft_chunks", observe_read)
        result = _build(role_bm25, caller, manifest)
        assert result.snapshot == baseline.snapshot and _metadata(result) == _metadata(baseline)
        assert len(images) == 1
        assert not caller.in_transaction
        assert caller.execute("SELECT text FROM draft_children WHERE child_id = ?", (child_id,)).fetchone()[0] != original
    with pytest.raises(sqlite3.ProgrammingError):
        images[0].execute("SELECT 1")


@pytest.mark.parametrize("before, after, build_day, reason", [
    (b'effective_date: "2026-09-01"', b'effective_date: "2026-09-02"', DAY, "NOT_YET_EFFECTIVE"),
    (b"expires_at: null", b'expires_at: "2026-09-02"', TOMORROW, "EXPIRED"),
    (b"approval_status: approved", b"approval_status: not_approved", DAY, "NOT_APPROVED"),
    (b"allowed_roles: [employee, hr, finance, it_security]", b"allowed_roles: [hr]", DAY, None),
], ids=["future", "expired", "unapproved", "public-without-employee-grant"])
def test_policy_eligibility_and_explicit_grants_drive_index_membership(
    role_bm25, tmp_path, before, after, build_day, reason,
):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    manifest = _source_fixture(tmp_path, before, after)
    with closing(sqlite3.connect(":memory:")) as connection:
        ingest_manifest_to_draft(tmp_path, connection)
        bundle = _build(role_bm25, connection, manifest, day=build_day)
    if reason is None:
        assert TRAVEL in bundle.eligibility.eligible_document_ids
        assert [meta.role for meta in _metadata(bundle).values() if TRAVEL in meta.indexed_document_ids] == ["hr"]
    else:
        assert {entry.document_id: entry.reasons for entry in bundle.eligibility.exclusions}[TRAVEL] == (reason,)
        assert all(TRAVEL not in index.metadata.indexed_document_ids for index in bundle.indexes)


@pytest.mark.parametrize("sql, detail", [
    ("UPDATE draft_documents SET allowed_roles = '[\"unknown\"]' WHERE document_id = ?", "MANIFEST_MISMATCH"),
    ("ALTER TABLE draft_documents RENAME COLUMN allowed_roles TO missing_roles", "INVALID_DRAFT_SCHEMA"),
    ("UPDATE draft_documents SET source_sha256 = '" + "0" * 64 + "' WHERE document_id = ?", "DRAFT_INTEGRITY_FAILED"),
    ("DELETE FROM draft_children WHERE child_id = (SELECT child_id FROM draft_children LIMIT 1)", "INCOMPLETE_DRAFT"),
])
def test_invalid_draft_aborts_before_any_index_build(role_bm25, draft, sql, detail, monkeypatch):
    connection, manifest = draft
    connection.execute(sql, (TRAVEL,) if "?" in sql else ())
    connection.commit()

    def unexpected_index(*args):
        pytest.fail("invalid SQLite data reached BM25S")

    monkeypatch.setattr(role_bm25, "build_text_index", unexpected_index)
    with pytest.raises(role_bm25.RoleBM25Error, match="^INVALID_DRAFT$") as raised:
        _build(role_bm25, connection, manifest)
    assert raised.value.detail_code == detail
    assert connection.execute("SELECT 1").fetchone() == (1,)


def test_manifest_hash_and_expected_snapshot_mismatches_fail(role_bm25, draft, monkeypatch):
    connection, manifest = draft
    changed = json.loads(manifest)
    changed["documents"][0]["content_sha256"] = "0" * 64

    def unexpected_index(*args):
        pytest.fail("snapshot identity mismatch reached BM25S")

    monkeypatch.setattr(role_bm25, "build_text_index", unexpected_index)
    with pytest.raises(role_bm25.RoleBM25Error, match="^INVALID_DRAFT$") as raised:
        _build(role_bm25, connection, json.dumps(changed).encode("utf-8"))
    assert raised.value.detail_code == "MANIFEST_MISMATCH"
    with pytest.raises(role_bm25.RoleBM25Error, match="^SNAPSHOT_MISMATCH$"):
        _build(role_bm25, connection, manifest, expected_snapshot_id="0" * 64)


def test_active_transaction_is_preserved(role_bm25, draft):
    connection, manifest = draft
    connection.execute("BEGIN")
    connection.execute("UPDATE draft_documents SET title = ? WHERE document_id = ?", ("Synthetic pending change", TRAVEL))
    with pytest.raises(role_bm25.RoleBM25Error, match="^TRANSACTION_ACTIVE$"):
        _build(role_bm25, connection, manifest)
    assert connection.in_transaction
    assert connection.execute("SELECT title FROM draft_documents WHERE document_id = ?", (TRAVEL,)).fetchone() == (
        "Synthetic pending change",
    )
    connection.rollback()


def test_duplicate_positional_child_is_rejected_and_partial_build_returns_nothing(role_bm25, draft, monkeypatch):
    connection, manifest = draft
    read_chunks = role_bm25.read_draft_chunks
    stage_calls = []

    def duplicate_on_second_role(image, allowed):
        chunks = read_chunks(image, allowed)
        stage_calls.append(len(chunks))
        return chunks if len(stage_calls) == 1 else (*chunks, chunks[0])

    monkeypatch.setattr(role_bm25, "read_draft_chunks", duplicate_on_second_role)
    result = None
    with pytest.raises(role_bm25.RoleBM25Error, match="^INVALID_CHILD_MEMBERSHIP$"):
        result = _build(role_bm25, connection, manifest)
    assert result is None and len(stage_calls) == 2 and all(stage_calls)


def test_real_top_30_positive_scores_tie_order_and_exact_date(role_bm25, tmp_path, secured):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    manifest = _source_fixture(tmp_path, b"## 6. Exceptions and escalation", b"## 6. Exceptions and escalation" + b"\n\nfixturecapsule" * 34)
    with closing(sqlite3.connect(":memory:")) as connection:
        ingest_manifest_to_draft(tmp_path, connection)
        bundle = _build(role_bm25, connection, manifest)
        expected = connection.execute(
            "SELECT child_id FROM draft_children WHERE document_id = ? AND trim(text, char(10)) = 'fixturecapsule' "
            "ORDER BY child_id", (TRAVEL,),
        ).fetchall()
    assert len(expected) == 34
    context = secured.authenticate(secured.sign(), "employee")
    bound = {"expected_snapshot_id": bundle.snapshot.candidate_snapshot_id, "validity_date": DAY}
    hits = role_bm25.search_role_bm25(bundle, context, query="fixturecapsule", **bound)
    assert [hit.child.child_id for hit in hits] == [child_id for child_id, in expected[:30]]
    assert len({hit.score for hit in hits}) == 1 and hits[0].score > 0
    assert role_bm25.search_role_bm25(bundle, context, query="zzznomatchxxx", **bound) == ()
    with pytest.raises(role_bm25.RoleBM25Error, match="^VALIDITY_DATE_MISMATCH$"):
        role_bm25.search_role_bm25(bundle, context, query="fixturecapsule", **{**bound, "validity_date": TOMORROW})


def test_nonfinite_bm25s_scores_do_not_enter_a_ranking(role_bm25, monkeypatch):
    from askpolicy.retrieval import bm25_core

    child_ids = ("a", "b", "c", "d")
    model = bm25_core.build_text_index(["capsule"] * len(child_ids))
    baseline = bm25_core.ranked_positions(model, child_ids, "capsule", limit=30)
    assert len(baseline) == 4
    get_scores = bm25_core.bm25s.BM25.get_scores

    def corrupted_scores(engine, tokens):
        scores = get_scores(engine, tokens)
        scores[:3] = [float("nan"), float("inf"), float("-inf")]
        return scores

    monkeypatch.setattr(bm25_core.bm25s.BM25, "get_scores", corrupted_scores)
    assert bm25_core.ranked_positions(model, child_ids, "capsule", limit=30) == (baseline[3],)


def test_actual_policy_transition_blocks_queries_at_boundary(role_bm25, tmp_path, secured):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    manifest = _source_fixture(tmp_path, b"expires_at: null", b'expires_at: "2026-09-02"')
    with closing(sqlite3.connect(":memory:")) as connection:
        ingest_manifest_to_draft(tmp_path, connection)
        bundle = _build(role_bm25, connection, manifest)
    assert all(index.metadata.next_transition_date == TOMORROW for index in bundle.indexes)
    context = secured.authenticate(secured.sign(), "employee")
    kwargs = {"expected_snapshot_id": bundle.snapshot.candidate_snapshot_id, "query": "travel"}
    assert role_bm25.search_role_bm25(bundle, context, validity_date=DAY, **kwargs)
    with pytest.raises(role_bm25.RoleBM25Error, match="^STALE_INDEX$"):
        role_bm25.search_role_bm25(bundle, context, validity_date=TOMORROW, **kwargs)
