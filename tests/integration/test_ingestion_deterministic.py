from contextlib import closing
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path("data/corpus_manifest.yaml")
TABLES = ("draft_documents", "draft_parents", "draft_children")
TRAVEL = "TRAVEL-001-FR-V1"
LEAVE = "LEAVE-001-FR-V2"
LINKS = {
    "ACCESS-001-FR-V2": "ACCESS-001-FR-V1",
    "DATA-001-FR-V2": "DATA-001-FR-V1",
    "EXPENSES-001-FR-V2": "EXPENSES-001-FR-V1",
    LEAVE: "LEAVE-001-FR-V1",
    "REMOTE-001-EN-V2": "REMOTE-001-EN-V1",
    "REMOTE-001-FR-V2": "REMOTE-001-FR-V1",
}
FIRST_CHILD = "(SELECT child_id FROM draft_children ORDER BY document_id, start LIMIT 1)"
CHANGE_CHILD_TEXT = (
    "UPDATE draft_children SET text = char(88 + (substr(text, 1, 1) = 'X')) || substr(text, 2) "
    f"WHERE child_id = {FIRST_CHILD}"
)


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.ingestion import coordinator
        from askpolicy.storage import snapshot, sqlite

        yield coordinator, snapshot, sqlite


@pytest.fixture(scope="module")
def manifest():
    return (ROOT / MANIFEST_PATH).read_bytes()


@pytest.fixture(scope="module")
def valid_draft(modules, manifest):
    coordinator, snapshot, _ = modules
    with closing(sqlite3.connect(":memory:")) as connection:
        coordinator.ingest_manifest_to_draft(ROOT, connection)
        identity = _compute(snapshot, connection, manifest)
        assert (identity.document_count, identity.parent_count, identity.child_count) == (26, 161, 387)
        yield connection, identity


@pytest.fixture
def draft(valid_draft):
    source, _ = valid_draft
    with closing(sqlite3.connect(":memory:")) as connection:
        source.backup(connection)
        yield connection


def _compute(snapshot, connection, manifest):
    return snapshot.compute_draft_snapshot_identity(
        connection, manifest=manifest, config=snapshot.DraftSnapshotConfig(),
    )


def _assert_rejected(snapshot, connection, manifest, code, stage):
    before = list(connection.iterdump())
    changes = connection.total_changes
    with pytest.raises(snapshot.SnapshotIdentityError) as raised:
        _compute(snapshot, connection, manifest)
    assert raised.value.code == code
    assert raised.value.stage == stage
    assert str(raised.value) == code
    assert not connection.in_transaction
    assert connection.total_changes == changes
    assert list(connection.iterdump()) == before
    assert connection.execute("SELECT 1").fetchone() == (1,)


def _canonical_digest(domain, payload):
    serialized = json.dumps(
        [f"askpolicy/draft-snapshot/{domain}", payload],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def test_complete_draft_fingerprints_every_logical_record(modules, valid_draft, manifest):
    _, snapshot, _ = modules
    connection, identity = valid_draft
    assert identity.status == "UNPUBLISHED"
    connection.row_factory = sqlite3.Row
    try:
        records = {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY {ordering}")]
            for table, ordering in zip(TABLES, ("document_id", "document_id, start", "document_id, start"), strict=True)
        }
        before = list(connection.iterdump())
        changes = connection.total_changes
        assert _compute(snapshot, connection, manifest) == identity
        assert _compute(snapshot, connection, manifest) == identity
        assert not connection.in_transaction
        assert connection.row_factory is sqlite3.Row
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert connection.total_changes == changes
        assert list(connection.iterdump()) == before
    finally:
        connection.row_factory = None

    documents, parents, children = (records[table] for table in TABLES)
    for document in documents:
        document["allowed_roles"] = json.loads(document["allowed_roles"])
    for parent in parents:
        parent["heading_path"] = json.loads(parent["heading_path"])

    approved = sorted(json.loads(manifest)["documents"], key=lambda document: document["document_id"])
    assert all(len(record) == 16 for record in approved)
    assert {document["document_id"]: document["supersedes"] for document in documents if document["supersedes"]} == LINKS
    ids = {document["document_id"] for document in documents}
    assert ids == {document["document_id"] for document in approved}
    assert len(ids - set(LINKS.values())) == 20
    assert set(LINKS.values()) <= ids
    stored = {document["document_id"]: document for document in documents}
    assert stored[TRAVEL]["allowed_roles"] == ["employee", "hr", "finance", "it_security"]
    assert stored[LEAVE]["allowed_roles"] == ["hr"]
    assert "é" in stored[TRAVEL]["canonical_source"]
    assert any(
        stored[chunk["document_id"]]["language"] == "ar" and re.search(r"[\u0600-\u06ff]", chunk["text"])
        for chunk in children
    )
    assert len({parent["parent_id"] for parent in parents}) == 161
    assert len({child["child_id"] for child in children}) == 387

    counts = {"document_count": 26, "parent_count": 161, "child_count": 387}
    assert identity.content_hash == _canonical_digest("content/v1", {"tables": records, "counts": counts})
    assert identity.source_manifest_hash == _canonical_digest("manifest/v1", {"documents": approved})
    assert identity.config_hash == _canonical_digest("config/v1", {
        "snapshot_format_version": 1,
        "source_canonicalization": "utf8-lf-v1",
        "chunking_format": "structural-blocks-v1",
        "component_encoding": "canonical-json-v1",
        "identity_encoding": "json-object-v1",
    })
    assert identity.candidate_snapshot_id == _canonical_digest("identity/json-object-v1", {
        "version": 1,
        "content_hash": identity.content_hash,
        "config_hash": identity.config_hash,
        "source_manifest_hash": identity.source_manifest_hash,
    })
    for value in (identity.content_hash, identity.config_hash, identity.source_manifest_hash, identity.candidate_snapshot_id):
        assert re.fullmatch(r"[0-9a-f]{64}", value)


_CLEAN_BUILD = """
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
from askpolicy.storage.snapshot import DraftSnapshotConfig, compute_draft_snapshot_identity

with closing(sqlite3.connect(sys.argv[2])) as connection:
    ingest_manifest_to_draft(root, connection)
    identity = compute_draft_snapshot_identity(
        connection, manifest=(root / "data/corpus_manifest.yaml").read_bytes(), config=DraftSnapshotConfig(),
    )
print(json.dumps(asdict(identity), sort_keys=True))
"""


def test_independent_clean_builds_in_separate_processes(valid_draft, tmp_path):
    _, baseline = valid_draft
    builds = []
    for dirname, filename, seed in (("first", "corpus.db", "101"), ("second", "other.sqlite", "202")):
        directory = tmp_path / dirname
        directory.mkdir()
        path = directory / filename
        assert not path.exists()
        result = subprocess.run(
            [sys.executable, "-c", _CLEAN_BUILD, str(ROOT), str(path)],
            cwd=directory,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=seed),
            check=True,
            capture_output=True,
            text=True,
        )
        builds.append(json.loads(result.stdout))
    assert builds[0] == builds[1] == asdict(baseline)
    for build_number, result in enumerate(builds, 1):
        print(f"Clean build {build_number} {json.dumps(result, sort_keys=True)}")


def _copy_corpus(tmp_path, manifest):
    checkout = tmp_path / "checkout"
    shutil.copytree(ROOT / "data/policies", checkout / "data/policies")
    (checkout / MANIFEST_PATH).write_bytes(manifest)
    return checkout


def test_crlf_checkout_and_reformatted_reordered_manifest(modules, valid_draft, manifest, tmp_path):
    coordinator, snapshot, _ = modules
    _, baseline = valid_draft
    records = list(reversed(json.loads(manifest)["documents"]))
    reordered = json.dumps({"documents": records}, sort_keys=True, ensure_ascii=True).encode("utf-8")
    checkout = _copy_corpus(tmp_path, reordered)
    for record in records:
        path = checkout / record["source_path"]
        source = path.read_bytes()
        crlf = source.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        assert b"\r\n" in crlf
        path.write_bytes(crlf)

    with closing(sqlite3.connect(":memory:")) as connection:
        coordinator.ingest_manifest_to_draft(checkout, connection)
        assert _compute(snapshot, connection, reordered) == baseline


def test_reversed_physical_row_order_preserves_identity(modules, valid_draft, draft, manifest):
    _, snapshot, _ = modules
    _, baseline = valid_draft
    for table in TABLES:
        original = draft.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        draft.execute(f"UPDATE {table} SET rowid = -rowid")
        assert draft.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() == original[::-1]
    draft.commit()
    draft.execute("PRAGMA reverse_unordered_selects = ON")
    assert _compute(snapshot, draft, manifest) == baseline


def test_supported_configuration_changes_only_config_hash_and_candidate_id(modules, valid_draft, manifest):
    _, snapshot, _ = modules
    connection, baseline = valid_draft
    config = snapshot.DraftSnapshotConfig(identity_encoding="json-array-v1")
    alternative = snapshot.compute_draft_snapshot_identity(connection, manifest=manifest, config=config)
    assert alternative.config_hash == _canonical_digest("config/v1", asdict(config))
    assert alternative.config_hash != baseline.config_hash
    assert alternative.candidate_snapshot_id != baseline.candidate_snapshot_id
    assert alternative.content_hash == baseline.content_hash
    assert alternative.source_manifest_hash == baseline.source_manifest_hash
    assert alternative.candidate_snapshot_id == _canonical_digest("identity/json-array-v1", [
        1, alternative.content_hash, alternative.config_hash, alternative.source_manifest_hash,
    ])
    assert snapshot.compute_draft_snapshot_identity(connection, manifest=manifest, config=replace(config)) == alternative


def test_coordinated_source_and_manifest_acl_revision_changes_identity(modules, valid_draft, manifest, tmp_path):
    coordinator, snapshot, _ = modules
    _, baseline = valid_draft
    checkout = _copy_corpus(tmp_path, manifest)
    approved = json.loads(manifest)
    record = next(document for document in approved["documents"] if document["document_id"] == TRAVEL)
    path = checkout / record["source_path"]
    original = path.read_bytes().replace(b"\r\n", b"\n")
    clause = b"allowed_roles: [employee, hr, finance, it_security]"
    assert original.count(clause) == 1
    changed = original.replace(clause, b"allowed_roles: [employee]")
    path.write_bytes(changed)
    record["allowed_roles"] = ["employee"]
    record["content_sha256"] = sha256(changed).hexdigest()
    revised = json.dumps(approved, ensure_ascii=False).encode("utf-8")
    (checkout / MANIFEST_PATH).write_bytes(revised)
    with closing(sqlite3.connect(":memory:")) as connection:
        coordinator.ingest_manifest_to_draft(checkout, connection)
        identity = _compute(snapshot, connection, revised)
    assert identity.content_hash != baseline.content_hash
    assert identity.source_manifest_hash != baseline.source_manifest_hash
    assert identity.candidate_snapshot_id != baseline.candidate_snapshot_id
    assert identity.config_hash == baseline.config_hash


@pytest.mark.parametrize("initialize, code, stage", [
    (False, "INVALID_DRAFT_SCHEMA", "database"),
    (True, "INCOMPLETE_DRAFT", "documents"),
], ids=["no-schema", "no-corpus"])
def test_empty_draft_is_rejected(modules, manifest, initialize, code, stage):
    _, snapshot, sqlite = modules
    with closing(sqlite3.connect(":memory:")) as connection:
        if initialize:
            sqlite.initialize_draft_schema(connection)
        _assert_rejected(snapshot, connection, manifest, code, stage)


def test_missing_required_table_is_rejected(modules, draft, manifest):
    _, snapshot, _ = modules
    draft.execute("DROP TABLE draft_children")
    _assert_rejected(snapshot, draft, manifest, "INVALID_DRAFT_SCHEMA", "database")


@pytest.mark.parametrize("statement, code, stage", [
    pytest.param(f"DELETE FROM draft_documents WHERE document_id = '{TRAVEL}'",
                 "INCOMPLETE_DRAFT", "documents", id="missing-document"),
    pytest.param("DELETE FROM draft_parents WHERE rowid = 1",
                 "INCOMPLETE_DRAFT", "records", id="missing-parent"),
    pytest.param(f"DELETE FROM draft_children WHERE child_id = {FIRST_CHILD}",
                 "INCOMPLETE_DRAFT", "records", id="missing-child"),
    pytest.param(f"UPDATE draft_documents SET source_sha256 = '{'0' * 64}' WHERE document_id = '{TRAVEL}'",
                 "DRAFT_INTEGRITY_FAILED", "documents", id="corrupt-hash"),
    pytest.param(f"UPDATE draft_documents SET canonical_source = canonical_source || char(10) WHERE document_id = '{TRAVEL}'",
                 "DRAFT_INTEGRITY_FAILED", "documents", id="changed-source"),
    pytest.param(f"""UPDATE draft_documents SET allowed_roles = '["finance"]' WHERE document_id = '{TRAVEL}'""",
                 "MANIFEST_MISMATCH", "documents", id="altered-roles"),
    pytest.param(f"""UPDATE draft_documents SET allowed_roles = '["hr","employee"]' WHERE document_id = '{LEAVE}'""",
                 "MANIFEST_MISMATCH", "documents", id="broadened-restricted-acl"),
    pytest.param(f"UPDATE draft_documents SET classification = 'INTERNAL' WHERE document_id = '{LEAVE}'",
                 "MANIFEST_MISMATCH", "documents", id="changed-classification"),
    pytest.param(f"UPDATE draft_documents SET version = 3 WHERE document_id = '{TRAVEL}'",
                 "MANIFEST_MISMATCH", "documents", id="changed-version"),
    pytest.param(f"UPDATE draft_documents SET language = 'ar' WHERE document_id = '{TRAVEL}'",
                 "MANIFEST_MISMATCH", "documents", id="changed-language"),
    pytest.param(f"UPDATE draft_documents SET supersedes = 'LEAVE-001-FR-V9' WHERE document_id = '{LEAVE}'",
                 "MANIFEST_MISMATCH", "documents", id="broken-predecessor"),
    pytest.param(f"UPDATE draft_children SET parent_id = 'missing' WHERE child_id = {FIRST_CHILD}",
                 "DRAFT_INTEGRITY_FAILED", "records", id="orphan-child"),
    pytest.param(
        "UPDATE draft_children SET parent_id = (SELECT draft_parents.parent_id FROM draft_parents "
        "WHERE draft_parents.document_id = draft_children.document_id ORDER BY start DESC LIMIT 1) "
        f"WHERE child_id = {FIRST_CHILD}",
        "DRAFT_INTEGRITY_FAILED", "records", id="wrong-existing-parent",
    ),
    pytest.param(f"UPDATE draft_children SET start = start + 100000, end = end + 100000 WHERE child_id = {FIRST_CHILD}",
                 "DRAFT_INTEGRITY_FAILED", "records", id="offset-out-of-source"),
    pytest.param(CHANGE_CHILD_TEXT, "DRAFT_INTEGRITY_FAILED", "records", id="changed-chunk-text"),
    pytest.param(f"UPDATE draft_documents SET source_path = 'different.md' WHERE document_id = '{TRAVEL}'",
                 "MANIFEST_MISMATCH", "documents", id="different-path"),
    pytest.param(f"UPDATE draft_documents SET document_id = 'TRAVEL-001-FR-V9' WHERE document_id = '{TRAVEL}'",
                 "DRAFT_INTEGRITY_FAILED", "documents", id="unexpected-document"),
])
def test_sqlite_tampering_is_rejected(modules, draft, manifest, statement, code, stage):
    _, snapshot, _ = modules
    draft.execute("PRAGMA foreign_keys = OFF")
    before = list(draft.iterdump())
    assert draft.execute(statement).rowcount == 1
    draft.commit()
    assert list(draft.iterdump()) != before
    _assert_rejected(snapshot, draft, manifest, code, stage)


def test_duplicate_document_is_rejected_even_without_primary_key(modules, draft, manifest):
    _, snapshot, _ = modules
    draft.executescript("""
        PRAGMA foreign_keys = OFF;
        CREATE TABLE unconstrained AS SELECT * FROM draft_documents;
        DROP TABLE draft_documents;
        ALTER TABLE unconstrained RENAME TO draft_documents;
        INSERT INTO draft_documents SELECT * FROM draft_documents LIMIT 1;
    """)
    assert draft.execute("SELECT count(*), count(DISTINCT document_id) FROM draft_documents").fetchone() == (27, 26)
    _assert_rejected(snapshot, draft, manifest, "DRAFT_INTEGRITY_FAILED", "documents")


def test_rehashed_source_tampering_still_disagrees_with_trusted_manifest(modules, draft, manifest):
    _, snapshot, _ = modules
    source, = draft.execute("SELECT canonical_source FROM draft_documents WHERE document_id = ?", (TRAVEL,)).fetchone()
    source += "\n"
    assert draft.execute(
        "UPDATE draft_documents SET canonical_source = ?, source_sha256 = ? WHERE document_id = ?",
        (source, sha256(source.encode("utf-8")).hexdigest(), TRAVEL),
    ).rowcount == 1
    draft.commit()
    _assert_rejected(snapshot, draft, manifest, "MANIFEST_MISMATCH", "documents")


@pytest.mark.parametrize("document_id, field, value, stage", [
    (TRAVEL, "content_sha256", "0" * 64, "documents"),
    (TRAVEL, "allowed_roles", ["hr", "employee", "finance", "it_security"], "documents"),
    (LEAVE, "supersedes", "LEAVE-001-FR-V9", "manifest"),
], ids=["source-digest", "ordered-metadata", "invalid-graph"])
def test_manifest_disagreement_is_rejected(modules, draft, manifest, document_id, field, value, stage):
    _, snapshot, _ = modules
    changed = json.loads(manifest)
    record = next(record for record in changed["documents"] if record["document_id"] == document_id)
    assert record[field] != value
    record[field] = value
    _assert_rejected(snapshot, draft, json.dumps(changed).encode("utf-8"), "MANIFEST_MISMATCH", stage)


@pytest.mark.parametrize("malformed, code", [
    (b'{"documents":', "INVALID_MANIFEST"),
    (b'{"documents":[{}]}', "INVALID_MANIFEST"),
    (b'{"documents":[{"title":"a","title":"b"}]}', "DUPLICATE_MANIFEST_FIELD"),
    (b'\xff', "INVALID_MANIFEST"),
], ids=["syntax", "shape", "duplicate-key", "utf8"])
def test_invalid_manifest_is_rejected(modules, draft, malformed, code):
    _, snapshot, _ = modules
    _assert_rejected(snapshot, draft, malformed, code, "manifest")


@pytest.mark.parametrize("settings, code", [
    ({"snapshot_format_version": True}, "INVALID_SNAPSHOT_CONFIG"),
    ({"source_canonicalization": "preserve-newlines"}, "INVALID_SNAPSHOT_CONFIG"),
    ({"chunking_format": "structural-blocks-v9"}, "INVALID_SNAPSHOT_CONFIG"),
    ({"component_encoding": "unknown"}, "INVALID_SNAPSHOT_CONFIG"),
    ({"identity_encoding": "unknown"}, "INVALID_SNAPSHOT_CONFIG"),
    ({"snapshot_format_version": 2}, "UNSUPPORTED_SNAPSHOT_FORMAT"),
])
def test_invalid_configuration_is_rejected(modules, draft, manifest, settings, code):
    _, snapshot, _ = modules
    config = snapshot.DraftSnapshotConfig(**settings)
    with pytest.raises(snapshot.SnapshotIdentityError) as raised:
        snapshot.compute_draft_snapshot_identity(draft, manifest=manifest, config=config)
    assert raised.value.code == code
    assert raised.value.stage == "configuration"
    assert not draft.in_transaction


def test_active_caller_transaction_is_preserved(modules, draft, manifest):
    _, snapshot, _ = modules
    assert draft.execute(CHANGE_CHILD_TEXT).rowcount == 1
    active_state = list(draft.iterdump())
    assert draft.in_transaction
    with pytest.raises(snapshot.SnapshotIdentityError) as raised:
        _compute(snapshot, draft, manifest)
    assert raised.value.code == "TRANSACTION_ACTIVE"
    assert raised.value.stage == "database"
    assert draft.in_transaction
    assert list(draft.iterdump()) == active_state
    draft.rollback()
    assert not draft.in_transaction


def test_read_uncommitted_is_rejected_without_changing_pragmas(modules, draft, manifest):
    _, snapshot, _ = modules
    draft.execute("PRAGMA read_uncommitted = ON")
    _assert_rejected(snapshot, draft, manifest, "UNSAFE_READ_ISOLATION", "database")
    assert draft.execute("PRAGMA read_uncommitted").fetchone() == (1,)


def test_concurrent_commit_cannot_mix_corpus_rows(modules, valid_draft, manifest, tmp_path):
    coordinator, snapshot, _ = modules
    _, baseline = valid_draft
    path = tmp_path / "concurrent.sqlite"

    class InterleavingCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            if sql.startswith("SELECT * FROM main.draft_documents"):
                self.connection.after_document_read()
            return result

    class InterleavingConnection(sqlite3.Connection):
        def cursor(self, factory=InterleavingCursor):
            return super().cursor(factory)

    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        coordinator.ingest_manifest_to_draft(ROOT, writer)
        with closing(sqlite3.connect(path, factory=InterleavingConnection)) as reader:
            commits = []

            def commit_change():
                assert reader.in_transaction
                assert writer.execute(CHANGE_CHILD_TEXT).rowcount == 1
                writer.commit()
                commits.append(True)

            reader.after_document_read = commit_change
            assert _compute(snapshot, reader, manifest) == baseline
            assert commits == [True]
            assert not reader.in_transaction
            reader.after_document_read = lambda: None
            _assert_rejected(snapshot, reader, manifest, "DRAFT_INTEGRITY_FAILED", "records")
