from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "data/corpus_manifest.yaml"
TABLES = ("draft_documents", "draft_parents", "draft_children")
TARGET = "TRAVEL-001-FR-V1"
REMOTE = "REMOTE-001-FR-V2"
LINKS = {
    "ACCESS-001-FR-V2": "ACCESS-001-FR-V1",
    "DATA-001-FR-V2": "DATA-001-FR-V1",
    "EXPENSES-001-FR-V2": "EXPENSES-001-FR-V1",
    "LEAVE-001-FR-V2": "LEAVE-001-FR-V1",
    "REMOTE-001-EN-V2": "REMOTE-001-EN-V1",
    "REMOTE-001-FR-V2": "REMOTE-001-FR-V1",
}


@pytest.fixture(scope="module")
def coordinator():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.ingestion import coordinator as module

        yield module


@pytest.fixture(scope="module")
def approved_sources(coordinator, tmp_path_factory):
    fixture = tmp_path_factory.mktemp("approved_sources")
    (fixture / "data").mkdir()
    shutil.copyfile(ROOT / MANIFEST, fixture / MANIFEST)
    shutil.copytree(ROOT / "data/policies", fixture / "data/policies")
    with closing(sqlite3.connect(":memory:")) as connection:
        assert coordinator.ingest_manifest_to_draft(fixture, connection).documents == 26
    return fixture


@pytest.fixture
def repository(approved_sources, tmp_path):
    return shutil.copytree(approved_sources, tmp_path / "checkout")


@pytest.fixture
def database():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        yield connection


def _manifest(root):
    return json.loads((root / MANIFEST).read_text(encoding="utf-8"))


def _write_manifest(root, manifest):
    (root / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def _document(manifest, document_id):
    return next(row for row in manifest["documents"] if row["document_id"] == document_id)


def _source_path(root, document_id):
    return root / _document(_manifest(root), document_id)["source_path"]


def _replace_source(root, document_id, before, after):
    manifest = _manifest(root)
    record = _document(manifest, document_id)
    path = root / record["source_path"]
    source = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert source.count(before) == 1
    raw = source.replace(before, after, 1).encode("utf-8")
    path.write_bytes(raw)
    record["content_sha256"] = sha256(raw).hexdigest()
    _write_manifest(root, manifest)


def _assert_ingestion_error(coordinator, root, connection, code, stage, detail=None):
    with pytest.raises(coordinator.ManifestIngestionError) as failure:
        coordinator.ingest_manifest_to_draft(root, connection)
    assert failure.value.code == code
    assert failure.value.stage == stage
    assert failure.value.detail_code == detail
    assert str(failure.value) == (code if detail is None else f"{code}: {detail}")


def _assert_no_schema(connection):
    assert not connection.in_transaction
    assert connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == []


def _logical_rows(connection):
    # Table names and ORDER BY clauses are fixed by the production draft schema.
    return tuple(
        tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"))
        for table, order in zip(TABLES, ("document_id", "document_id, start", "document_id, start"))
    )


def test_ingests_the_complete_corpus_in_one_commit(coordinator, repository, tmp_path, monkeypatch):
    expected_rows = _manifest(repository)["documents"]
    expected = {
        row["document_id"]: coordinator.extract_policy_chunks(
            (repository / row["source_path"]).read_bytes().decode("utf-8"),
            source_path=row["source_path"],
        )
        for row in expected_rows
    }
    calls = []
    initialize = coordinator.initialize_draft_schema
    persist = coordinator.persist_draft_extractions

    def observe_initialize(connection):
        calls.append("initialize")
        initialize(connection)

    def observe_persist(connection, batch):
        calls.append("persist")
        assert [item.metadata.document_id for item in batch] == sorted(expected)
        assert {item.metadata.document_id: item for item in batch} == expected
        return persist(connection, batch)

    monkeypatch.setattr(coordinator, "initialize_draft_schema", observe_initialize)
    monkeypatch.setattr(coordinator, "persist_draft_extractions", observe_persist)

    path = tmp_path / "draft.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        result = coordinator.ingest_manifest_to_draft(repository, connection)
        assert calls == ["initialize", "persist"]
        assert result == coordinator.DraftWriteSummary(
            documents=26,
            parents=sum(len(item.parents) for item in expected.values()),
            children=sum(len(item.children) for item in expected.values()),
        )
        assert not connection.in_transaction
        assert connection.execute("SELECT 1").fetchone() == (1,)

    with closing(sqlite3.connect(path)) as reader:
        reader.row_factory = sqlite3.Row
        stored = {row["document_id"]: dict(row) for row in reader.execute("SELECT * FROM draft_documents")}
        assert stored.keys() == expected.keys()
        assert {key: row["supersedes"] for key, row in stored.items() if row["supersedes"]} == LINKS
        assert set(LINKS.values()) <= stored.keys()
        assert {row["source_path"] for row in stored.values()} == {row["source_path"] for row in expected_rows}

        for key, row in stored.items():
            item = expected[key]
            assert row.pop("canonical_source") == item.canonical_source
            assert row.pop("source_sha256") == item.source_sha256
            assert row.pop("source_path") == item.source_path
            assert row.pop("body_start") == item.body_start
            row["allowed_roles"] = json.loads(row["allowed_roles"])
            assert row == item.metadata.model_dump(mode="json")

        for table, relation, id_field in (
            ("draft_parents", "parents", "parent_id"),
            ("draft_children", "children", "child_id"),
        ):
            chunks = {
                getattr(record, id_field): record
                for item in expected.values()
                for record in getattr(item, relation)
            }
            rows = reader.execute(f"SELECT * FROM {table}").fetchall()
            assert len(rows) == len(chunks)
            for row in rows:
                record = chunks.pop(row[id_field])
                assert dict(row) == {
                    key: json.dumps(record.heading_path, ensure_ascii=False, separators=(",", ":"))
                    if key == "heading_path" else getattr(record, key)
                    for key in row.keys()
                }
            assert not chunks

        assert stored["LEAVE-001-FR-V2"]["allowed_roles"] == ["hr"]
        assert stored["ACCESS-001-EN-V2"]["allowed_roles"] == ["it_security"]
        assert "déplacements" in stored[TARGET]["title"]
        assert stored["LEAVE-001-AR-V2"]["title"] == "سياسة الإجازات والغياب"
        assert reader.execute("PRAGMA foreign_key_check").fetchall() == []
        assert {row[0] for row in reader.execute("SELECT name FROM sqlite_master WHERE type = 'table'")} == set(TABLES)
        for table in TABLES:
            names = {row[1] for row in reader.execute(f"PRAGMA table_info({table})")}
            assert names.isdisjoint({"snapshot_id", "published", "retrieval_ready", "searchable"})


def test_reordered_manifest_and_crlf_sources_produce_identical_rows(
    coordinator, approved_sources, repository, database
):
    manifest = _manifest(repository)
    manifest["documents"].reverse()
    _write_manifest(repository, manifest)
    for record in manifest["documents"]:
        path = repository / record["source_path"]
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        path.write_bytes(raw.replace(b"\n", b"\r\n"))

    with closing(sqlite3.connect(":memory:")) as original:
        expected = coordinator.ingest_manifest_to_draft(approved_sources, original)
        assert coordinator.ingest_manifest_to_draft(repository, database) == expected
        assert _logical_rows(original) == _logical_rows(database)


@pytest.mark.parametrize(
    ("defect", "code"),
    [
        ("syntax", "INVALID_MANIFEST"),
        ("general-yaml", "INVALID_MANIFEST"),
        ("non-finite", "INVALID_MANIFEST"),
        ("duplicate-root-key", "DUPLICATE_MANIFEST_FIELD"),
        ("duplicate-nested-key", "DUPLICATE_MANIFEST_FIELD"),
    ],
)
def test_malformed_manifest_fails_before_schema_initialization(coordinator, repository, database, defect, code):
    path = repository / MANIFEST
    original = path.read_text(encoding="utf-8")
    invalid = {
        "syntax": original.rstrip()[:-1],
        "general-yaml": "documents:\n" + json.dumps(_manifest(repository)["documents"]),
        "non-finite": original.replace('"version": 2', '"version": NaN', 1),
        "duplicate-root-key": original.replace('"documents":', '"documents": [], "documents":', 1),
        "duplicate-nested-key": original.replace('"version": 2', '"version": {"key": 1, "key": 2}', 1),
    }[defect]
    assert invalid != original
    path.write_text(invalid, encoding="utf-8")
    _assert_ingestion_error(coordinator, repository, database, code, "manifest")
    _assert_no_schema(database)


def test_missing_manifest_is_rejected(coordinator, repository, database):
    (repository / MANIFEST).unlink()
    _assert_ingestion_error(coordinator, repository, database, "MISSING_MANIFEST", "manifest")
    _assert_no_schema(database)


@pytest.mark.parametrize(
    ("defect", "code", "stage"),
    [
        ("unknown-root-key", "INVALID_MANIFEST", "manifest"),
        ("missing-documents", "INVALID_MANIFEST", "manifest"),
        ("empty-documents", "INVALID_MANIFEST", "manifest"),
        ("unknown-record-key", "INVALID_MANIFEST", "manifest"),
        ("missing-required-field", "INVALID_MANIFEST", "manifest"),
        ("bool-version", "INVALID_MANIFEST", "manifest"),
        ("numeric-role", "INVALID_MANIFEST", "manifest"),
        ("invalid-digest", "INVALID_MANIFEST", "manifest"),
        ("duplicate-id", "DUPLICATE_DOCUMENT_ID", "inventory"),
        ("duplicate-path", "DUPLICATE_SOURCE_PATH", "inventory"),
        ("duplicate-slot", "INVALID_INVENTORY", "inventory"),
        ("omitted-document", "INVALID_INVENTORY", "inventory"),
        ("invented-id", "INVALID_INVENTORY", "inventory"),
    ],
)
def test_invalid_manifest_schema_or_inventory(coordinator, repository, database, defect, code, stage):
    manifest = _manifest(repository)
    first, second, *remaining = manifest["documents"]
    changes = {
        "unknown-root-key": manifest | {"extra": True},
        "missing-documents": {},
        "empty-documents": {"documents": []},
        "omitted-document": {"documents": [second, *remaining]},
    }
    invalid_first = {
        "unknown-record-key": first | {"extra": True},
        "missing-required-field": {key: value for key, value in first.items() if key != "expires_at"},
        "bool-version": first | {"version": True},
        "numeric-role": first | {"allowed_roles": [1]},
        "invalid-digest": first | {"content_sha256": "x" * 64},
        "invented-id": first | {"document_id": "OTHER-001-EN-V2"},
    }
    invalid_second = {
        "duplicate-id": second | {"document_id": first["document_id"]},
        "duplicate-path": second | {"source_path": first["source_path"]},
        "duplicate-slot": second | {key: first[key] for key in ("policy_id", "language", "version")},
    }
    if defect not in changes:
        changes[defect] = {"documents": [
            invalid_first.get(defect, first), invalid_second.get(defect, second), *remaining,
        ]}
    _write_manifest(repository, changes[defect])
    _assert_ingestion_error(coordinator, repository, database, code, stage)
    _assert_no_schema(database)


@pytest.mark.parametrize(
    "path",
    [
        "../external/TRAVEL-001-FR-V1.md",
        "/external/TRAVEL-001-FR-V1.md",
        "C:/external/TRAVEL-001-FR-V1.md",
        r"data\policies\fr\TRAVEL-001-FR-V1.md",
        "data/policies/en/TRAVEL-001-FR-V1.md",
        "data/policies/fr/OTHER.md",
        "data/policies/fr/TRAVEL-001-FR-V1.txt",
    ],
    ids=["traversal", "absolute-posix", "absolute-windows", "backslash",
         "wrong-language-path", "wrong-filename", "wrong-extension"],
)
def test_invalid_paths_are_rejected_before_source_access(coordinator, repository, database, path, monkeypatch):
    manifest = _manifest(repository)
    _document(manifest, TARGET)["source_path"] = path
    _write_manifest(repository, manifest)
    read_bytes = Path.read_bytes

    def only_manifest(self):
        assert self == repository / MANIFEST
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", only_manifest)
    _assert_ingestion_error(coordinator, repository, database, "INVALID_SOURCE_PATH", "inventory")
    _assert_no_schema(database)


@pytest.mark.parametrize(
    ("defect", "code"),
    [
        ("missing-file", "MISSING_SOURCE"),
        ("unlisted-policy", "UNEXPECTED_SOURCE"),
        ("unreadable", "SOURCE_READ_FAILED"),
        ("changed-body", "SOURCE_HASH_MISMATCH"),
        ("invalid-utf8", "INVALID_SOURCE_ENCODING"),
        ("standalone-cr", "INVALID_SOURCE_NEWLINES"),
    ],
)
def test_source_validation_rejects_incomplete_or_changed_corpus(
    coordinator, repository, database, monkeypatch, defect, code
):
    path = _source_path(repository, TARGET)
    original = path.read_bytes()
    if defect == "missing-file":
        path.unlink()
    elif defect == "unlisted-policy":
        nested = path.parent / "nested"
        nested.mkdir()
        (nested / "unlisted.md").write_bytes(original)
    elif defect == "unreadable":
        read_bytes = Path.read_bytes

        def deny_target(self):
            if self == path:
                raise PermissionError("synthetic read failure")
            return read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", deny_target)
    else:
        suffix = {"changed-body": b"\nbody change", "invalid-utf8": b"\xff", "standalone-cr": b"\rbody"}[defect]
        path.write_bytes(original + suffix)
    _assert_ingestion_error(coordinator, repository, database, code, "source")
    _assert_no_schema(database)


def test_symlinked_source_directory_cannot_escape_root(coordinator, repository, database, tmp_path, monkeypatch):
    directory = repository / "data/policies/fr"
    outside = tmp_path / "outside"
    directory.rename(outside)
    if os.name == "nt":
        from _winapi import CreateJunction

        try:
            CreateJunction(str(outside), str(directory))
        except PermissionError:
            # Restricted Windows runners can deny junction creation. Inject
            # the same reparse-point metadata at the filesystem boundary.
            original_stat = Path.lstat

            def report_junction(self):
                if self == directory:
                    return SimpleNamespace(
                        st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                    )
                return original_stat(self)

            monkeypatch.setattr(Path, "lstat", report_junction)
    else:
        directory.symlink_to(outside, target_is_directory=True)

    read_bytes = Path.read_bytes

    def reject_redirected_reads(self):
        assert not self.is_relative_to(directory)
        return read_bytes(self)

    original_iterdir = Path.iterdir

    def reject_redirected_traversal(self):
        assert not self.is_relative_to(directory)
        return original_iterdir(self)

    monkeypatch.setattr(Path, "read_bytes", reject_redirected_reads)
    monkeypatch.setattr(Path, "iterdir", reject_redirected_traversal)
    _assert_ingestion_error(coordinator, repository, database, "INVALID_SOURCE_PATH", "source")
    _assert_no_schema(database)


def test_symlink_source_file_is_rejected_without_opening_it(coordinator, repository, database, monkeypatch):
    path = _source_path(repository, TARGET)
    original_stat = Path.lstat
    original_read = Path.read_bytes

    # Model a symlink at the filesystem boundary even on Windows hosts where
    # creating file symlinks requires a privilege unavailable to the runner.
    def report_symlink(self):
        if self == path:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return original_stat(self)

    def reject_target_read(self):
        assert self != path
        return original_read(self)

    monkeypatch.setattr(Path, "lstat", report_symlink)
    monkeypatch.setattr(Path, "read_bytes", reject_target_read)
    _assert_ingestion_error(coordinator, repository, database, "INVALID_SOURCE_PATH", "source")
    _assert_no_schema(database)


@pytest.mark.parametrize(
    ("document_id", "before", "after", "code", "stage", "detail"),
    [
        (TARGET, "allowed_roles: [employee, hr, finance, it_security]\n", "",
         "MISSING_ALLOWED_ROLES", "metadata", None),
        (TARGET, 'effective_date: "2026-09-01"\n', "", "MISSING_EFFECTIVE_DATE", "metadata", None),
        (TARGET, "allowed_roles: [employee, hr, finance, it_security]", "allowed_roles: [employee]",
         "METADATA_MISMATCH", "metadata", None),
        (TARGET, "allowed_roles: [employee, hr, finance, it_security]", "allowed_roles: [hr, employee, finance, it_security]",
         "METADATA_MISMATCH", "metadata", None),
        (TARGET, "classification: PUBLIC", "classification: INTERNAL",
         "METADATA_MISMATCH", "metadata", None),
        (TARGET, "expires_at: null", 'expires_at: "2028-01-01"', "METADATA_MISMATCH", "metadata", None),
        (REMOTE, "supersedes: REMOTE-001-FR-V1", "supersedes: null", "METADATA_MISMATCH", "metadata", None),
        (TARGET, "classification: PUBLIC", "classification: &tag PUBLIC", "MALFORMED_FRONT_MATTER", "metadata", None),
        (TARGET, "## 1.", "~~~\n## 1.", "EXTRACTION_FAILED", "extraction", "MALFORMED_STRUCTURE"),
    ],
    ids=["missing-roles", "missing-effective", "acl", "role-order", "classification",
         "expiration", "supersedes", "malformed-front-matter", "malformed-body"],
)
def test_hash_matching_sources_still_require_valid_matching_metadata_and_chunks(
    coordinator, repository, database, document_id, before, after, code, stage, detail
):
    _replace_source(repository, document_id, before, after)
    _assert_ingestion_error(coordinator, repository, database, code, stage, detail)
    _assert_no_schema(database)


@pytest.mark.parametrize(
    ("changes", "detail"),
    [
        ({REMOTE: "REMOTE-001-FR-V0"}, "MISSING_PREDECESSOR"),
        ({REMOTE: REMOTE}, "SELF_SUPERSESSION"),
        ({REMOTE: "TRAVEL-001-FR-V1"}, "WRONG_PREDECESSOR_FAMILY"),
        ({REMOTE: "REMOTE-001-EN-V2"}, "WRONG_PREDECESSOR_LANGUAGE"),
        ({REMOTE: None, "REMOTE-001-FR-V1": REMOTE}, "PREDECESSOR_NOT_EARLIER"),
        ({"REMOTE-001-FR-V1": REMOTE}, "CYCLE"),
        ({"REMOTE-001-AR-V2": "REMOTE-001-FR-V1"}, "BRANCHING_SUCCESSORS"),
        ({REMOTE: None}, "UNAPPROVED_LINKS"),
    ],
    ids=["missing", "self-or-same-version", "wrong-family", "wrong-language", "later", "cycle",
         "branching", "removed-link"],
)
def test_manifest_graph_rejects_invalid_predecessors(coordinator, repository, database, changes, detail):
    manifest = _manifest(repository)
    for document_id, predecessor in changes.items():
        _document(manifest, document_id)["supersedes"] = predecessor
    _write_manifest(repository, manifest)
    _assert_ingestion_error(coordinator, repository, database, "INVALID_SUPERSESSION_GRAPH", "supersession", detail)
    _assert_no_schema(database)


def test_invalid_corpus_and_second_write_preserve_existing_draft(
    coordinator, approved_sources, repository, database
):
    coordinator.ingest_manifest_to_draft(approved_sources, database)
    before = _logical_rows(database)
    _source_path(repository, TARGET).unlink()
    _assert_ingestion_error(coordinator, repository, database, "MISSING_SOURCE", "source")
    assert _logical_rows(database) == before
    _assert_ingestion_error(coordinator, approved_sources, database, "DRAFT_PERSISTENCE_FAILED",
                            "storage", "DRAFT_NOT_EMPTY")
    assert _logical_rows(database) == before
    assert not database.in_transaction


def test_sqlite_failure_after_successful_inserts_leaves_no_draft_rows(
    coordinator, repository, database
):
    coordinator.initialize_draft_schema(database)
    database.execute(
        """
        CREATE TEMP TRIGGER fail_after_prior_document
        BEFORE INSERT ON draft_documents
        WHEN EXISTS (SELECT 1 FROM draft_children)
        BEGIN
            SELECT RAISE(ABORT, 'synthetic trigger failure');
        END
        """
    )
    _assert_ingestion_error(coordinator, repository, database, "DRAFT_PERSISTENCE_FAILED",
                            "storage", "SQLITE_WRITE_FAILED")
    assert _logical_rows(database) == ((), (), ())
    assert not database.in_transaction

    database.execute("DROP TRIGGER fail_after_prior_document")
    assert coordinator.ingest_manifest_to_draft(repository, database).documents == 26
