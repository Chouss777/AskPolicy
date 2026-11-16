from contextlib import closing
from dataclasses import fields, replace
import json
from pathlib import Path
import sqlite3

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "data/corpus_manifest.yaml").read_text(encoding="utf-8"))["documents"]
TABLES = ("draft_documents", "draft_parents", "draft_children")

SOURCE = """---
policy_id: DEMO-999
document_id: DEMO-999-FR-V2
version: 2
language: fr
title: Congés d’essai
department: HR
classification: RESTRICTED
allowed_roles: [hr]
effective_date: "2026-09-01"
expires_at: null
supersedes: DEMO-999-FR-V1
approval_status: approved
source_type: synthetic_policy
source_license: project_owned
---

# Congés
Café et بيانات.

## Détails
Approbation préalable.
"""
SOURCE_PATH = "data/policies/fr/DEMO-999-FR-V2.md"


@pytest.fixture(scope="module")
def modules():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.ingestion.chunking import extract_policy_chunks
        from askpolicy.storage import sqlite as storage

        yield storage, extract_policy_chunks


@pytest.fixture
def database(modules):
    storage, _ = modules
    with closing(sqlite3.connect(":memory:")) as connection:
        storage.initialize_draft_schema(connection)
        connection.row_factory = sqlite3.Row
        yield connection


@pytest.fixture
def extraction(modules):
    _, extract = modules
    return extract(SOURCE, source_path=SOURCE_PATH)


@pytest.fixture(scope="module")
def corpus(modules):
    _, extract = modules
    return tuple(
        extract((ROOT / row["source_path"]).read_bytes().decode("utf-8"), source_path=row["source_path"])
        for row in MANIFEST
    )


def _logical_contents(connection):
    # Table names and ordering columns are fixed schema identifiers.
    return tuple(
        tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"))
        for table, order in zip(TABLES, ("document_id", "document_id, start", "document_id, start"))
    )


def _assert_exact_records(connection, extraction):
    document_id = extraction.metadata.document_id
    row = dict(connection.execute(
        "SELECT * FROM draft_documents WHERE document_id = ?", (document_id,)
    ).fetchone())
    assert row.pop("canonical_source") == extraction.canonical_source
    assert row.pop("source_sha256") == extraction.source_sha256
    assert row.pop("source_path") == extraction.source_path
    assert row.pop("body_start") == extraction.body_start
    row["allowed_roles"] = json.loads(row["allowed_roles"])
    assert row == extraction.metadata.model_dump(mode="json")

    parents = connection.execute(
        """
        SELECT p.*, d.policy_id, d.version, d.language, d.source_path, d.classification, d.allowed_roles
        FROM draft_parents AS p JOIN draft_documents AS d USING (document_id)
        WHERE p.document_id = ? ORDER BY p.start
        """,
        (document_id,),
    ).fetchall()
    children = connection.execute(
        """
        SELECT c.*, p.heading_path, d.policy_id, d.version, d.language, d.source_path,
               d.classification, d.allowed_roles
        FROM draft_children AS c
        JOIN draft_parents AS p USING (document_id, parent_id)
        JOIN draft_documents AS d USING (document_id)
        WHERE c.document_id = ? ORDER BY c.start
        """,
        (document_id,),
    ).fetchall()
    for rows, records in ((parents, extraction.parents), (children, extraction.children)):
        assert len(rows) == len(records)
        for stored, expected in zip(rows, records, strict=True):
            actual = dict(stored)
            for field in ("allowed_roles", "heading_path"):
                actual[field] = tuple(json.loads(actual[field]))
            assert actual == {field.name: getattr(expected, field.name) for field in fields(expected)}
            assert extraction.canonical_source[stored["start"]:stored["end"]] == stored["text"]


def _assert_summary(storage, summary, extractions):
    assert summary == storage.DraftWriteSummary(
        documents=len(extractions),
        parents=sum(len(item.parents) for item in extractions),
        children=sum(len(item.children) for item in extractions),
    )


def _assert_failure(storage, connection, extractions, code):
    with pytest.raises(storage.DraftPersistenceError) as error:
        storage.persist_draft_extractions(connection, extractions)
    assert error.value.code == code
    assert str(error.value) == code


def test_schema_is_idempotent_enables_foreign_keys_and_keeps_existing_rows(modules, database, extraction):
    storage, _ = modules
    assert database.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert tuple(row[0] for row in database.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )) == tuple(sorted(TABLES))
    storage.persist_draft_extractions(database, [extraction])
    before = _logical_contents(database)

    storage.initialize_draft_schema(database)
    storage.initialize_draft_schema(database)
    assert _logical_contents(database) == before
    assert not database.in_transaction
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []


def test_file_backed_write_is_committed_and_preserves_a_complete_record(modules, tmp_path, extraction):
    storage, _ = modules
    path = tmp_path / "draft.sqlite"
    with closing(sqlite3.connect(path)) as writer:
        storage.initialize_draft_schema(writer)
        summary = storage.persist_draft_extractions(writer, [extraction])

        _assert_summary(storage, summary, [extraction])
        assert not writer.in_transaction
        assert writer.execute("SELECT count(*) FROM draft_documents").fetchone()[0] == 1

    with closing(sqlite3.connect(path)) as reader:
        reader.row_factory = sqlite3.Row
        _assert_exact_records(reader, extraction)


def test_full_corpus_round_trips_and_is_independent_of_input_order(modules, database, corpus):
    storage, _ = modules
    assert len(corpus) == 26
    summary = storage.persist_draft_extractions(database, corpus)
    _assert_summary(storage, summary, corpus)
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []
    assert {row[0] for row in database.execute("SELECT document_id FROM draft_documents")} == {
        row["document_id"] for row in MANIFEST
    }
    for item in corpus:
        _assert_exact_records(database, item)
    records = {
        row["document_id"]: (row["version"], row["classification"], json.loads(row["allowed_roles"]), row["supersedes"])
        for row in database.execute("SELECT document_id, version, classification, allowed_roles, supersedes FROM draft_documents")
    }
    assert records["LEAVE-001-FR-V1"] == (1, "RESTRICTED", ["hr"], None)
    assert records["LEAVE-001-FR-V2"] == (2, "RESTRICTED", ["hr"], "LEAVE-001-FR-V1")
    assert records["ACCESS-001-EN-V2"][1:3] == ("RESTRICTED", ["it_security"])
    assert records["TRAVEL-001-FR-V1"][1:3] == ("PUBLIC", ["employee", "hr", "finance", "it_security"])

    with closing(sqlite3.connect(":memory:", isolation_level=None)) as second:
        storage.initialize_draft_schema(second)
        assert storage.persist_draft_extractions(second, tuple(reversed(corpus))) == summary
        assert _logical_contents(second) == _logical_contents(database)


def test_sql_like_text_is_stored_literally_and_public_keeps_its_explicit_role(modules, database):
    storage, extract = modules
    text = "O’Brien'); DROP TABLE draft_documents; -- بيانات"
    source = SOURCE.replace("Café et بيانات.", text).replace("RESTRICTED", "PUBLIC")
    source = source.replace("expires_at: null", 'expires_at: "2028-01-01"')
    item = extract(source.replace("\n", "\r\n"), source_path=SOURCE_PATH)
    _assert_summary(storage, storage.persist_draft_extractions(database, [item]), [item])
    _assert_exact_records(database, item)
    assert database.execute("SELECT allowed_roles FROM draft_documents").fetchone()[0] == '["hr"]'
    assert database.execute("SELECT text FROM draft_children WHERE kind = 'paragraph' ORDER BY start").fetchone()[0] == text + "\n"


def test_draft_schema_has_no_published_identity(modules, database, extraction):
    storage, _ = modules
    summary = storage.persist_draft_extractions(database, [extraction])
    assert {field.name for field in fields(summary)} == {"documents", "parents", "children"}
    for table in TABLES:
        assert "snapshot_id" not in {row[1] for row in database.execute(f"PRAGMA table_info({table})")}


@pytest.mark.parametrize("operation", ["initialize", "persist"])
def test_caller_transaction_is_rejected_without_commit_or_rollback(modules, database, extraction, operation):
    storage, _ = modules
    database.execute("CREATE TABLE caller_work (value INTEGER)")
    database.execute("INSERT INTO caller_work VALUES (1)")
    assert database.in_transaction
    if operation == "persist":
        _assert_failure(storage, database, [extraction], "TRANSACTION_ACTIVE")
    else:
        with pytest.raises(storage.DraftPersistenceError) as error:
            storage.initialize_draft_schema(database)
        assert error.value.code == "TRANSACTION_ACTIVE"
    assert database.in_transaction
    assert database.execute("SELECT value FROM caller_work").fetchone()[0] == 1
    database.rollback()
    assert database.execute("SELECT count(*) FROM caller_work").fetchone()[0] == 0
    assert _logical_contents(database) == ((), (), ())


def test_repeated_and_conflicting_writes_preserve_committed_state(modules, database, extraction):
    storage, extract = modules
    storage.persist_draft_extractions(database, [extraction])
    before = _logical_contents(database)
    conflict = extract(SOURCE.replace("Café", "Thé"), source_path=SOURCE_PATH)
    for item in (extraction, conflict):
        _assert_failure(storage, database, [item], "DRAFT_NOT_EMPTY")
        assert not database.in_transaction
        assert _logical_contents(database) == before


def _corrupt(record, **updates):
    """Create a malformed frozen record without relying on constructor rejection."""
    copied = replace(record)
    for name, value in updates.items():
        object.__setattr__(copied, name, value)
    return copied


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("parent_id", "parent:absent", "INVALID_PARENT_CHILD"),
        ("document_id", "OTHER-999-FR-V2", "INVALID_PARENT_CHILD"),
        ("version", 1, "INVALID_PARENT_CHILD"),
        ("language", "en", "INVALID_PARENT_CHILD"),
        ("classification", "PUBLIC", "INVALID_PARENT_CHILD"),
        ("allowed_roles", ("hr", "employee"), "INVALID_PARENT_CHILD"),
        ("start", -1, "INVALID_SOURCE_SPAN"),
        ("text", "different", "INVALID_SOURCE_SPAN"),
        ("child_id", "child:forged", "INVALID_CHILD_CHUNK"),
    ],
    ids=["orphan", "cross-document", "cross-version", "cross-language", "classification",
         "acl", "span", "text", "identity"],
)
def test_invalid_child_is_rejected_before_any_batch_rows_are_written(
    modules, database, extraction, field, value, code
):
    storage, extract = modules
    other = extract(SOURCE.replace("DEMO-999", "BEFORE-999"), source_path=SOURCE_PATH.replace("DEMO", "BEFORE"))
    child = _corrupt(extraction.children[0], **{field: value})
    altered = _corrupt(extraction, children=(child, *extraction.children[1:]))

    _assert_failure(storage, database, [other, altered], code)
    assert _logical_contents(database) == ((), (), ())
    _assert_summary(storage, storage.persist_draft_extractions(database, [other, extraction]), [other, extraction])


@pytest.mark.parametrize(
    ("defect", "code"),
    [
        ("duplicate-document", "DUPLICATE_DOCUMENT_ID"),
        ("duplicate-parent", "DUPLICATE_PARENT_ID"),
        ("duplicate-child", "DUPLICATE_CHILD_ID"),
        ("missing-child", "INVALID_CHILD_CHUNK"),
        ("bad-ancestor", "INVALID_PARENT_SECTION"),
        ("bad-source-hash", "INVALID_SOURCE_HASH"),
        ("bad-path", "INVALID_SOURCE_PATH"),
        ("changed-metadata", "METADATA_MISMATCH"),
        ("cross-document-ancestor", "INVALID_PARENT_SECTION"),
    ],
)
def test_malformed_extraction_does_not_persist(modules, database, extraction, defect, code):
    storage, extract = modules
    candidate = [extraction]
    if defect == "duplicate-document":
        candidate.append(extraction)
    elif defect == "duplicate-parent":
        candidate = [_corrupt(extraction, parents=(*extraction.parents, extraction.parents[0]))]
    elif defect == "duplicate-child":
        candidate = [_corrupt(extraction, children=(*extraction.children, extraction.children[0]))]
    elif defect == "missing-child":
        candidate = [_corrupt(extraction, children=extraction.children[:-1])]
    elif defect in {"bad-ancestor", "cross-document-ancestor"}:
        ancestor_id = "parent:absent"
        if defect == "cross-document-ancestor":
            other = extract(SOURCE.replace("DEMO-999", "OTHER-999"), source_path=SOURCE_PATH.replace("DEMO", "OTHER"))
            candidate.insert(0, other)
            ancestor_id = other.parents[0].parent_id
        parent = replace(extraction.parents[-1], ancestor_id=ancestor_id)
        candidate[-1] = _corrupt(extraction, parents=(*extraction.parents[:-1], parent))
    elif defect == "bad-source-hash":
        candidate = [replace(extraction, source_sha256="0" * 64)]
    elif defect == "bad-path":
        candidate = [_corrupt(extraction, source_path=f"../{SOURCE_PATH}")]
    else:
        metadata = extraction.metadata.model_copy(update={"title": "Different title"})
        candidate = [replace(extraction, metadata=metadata)]

    _assert_failure(storage, database, candidate, code)
    assert _logical_contents(database) == ((), (), ())
    _assert_summary(storage, storage.persist_draft_extractions(database, [extraction]), [extraction])


@pytest.mark.parametrize(
    ("defect", "code"),
    [("empty", "EMPTY_DRAFT"), ("bad-item", "INVALID_INPUT"), ("not-sequence", "INVALID_INPUT")],
)
def test_invalid_input_type_or_empty_batch(modules, database, extraction, defect, code):
    storage, _ = modules
    invalid = {"empty": [], "bad-item": [extraction, object()], "not-sequence": iter([extraction])}
    _assert_failure(storage, database, invalid[defect], code)
    assert _logical_contents(database) == ((), (), ())
    _assert_summary(storage, storage.persist_draft_extractions(database, [extraction]), [extraction])


def test_version_outside_sqlite_integer_range_fails_cleanly(modules, database, extraction):
    storage, extract = modules
    version = str(2**63)
    source = SOURCE.replace("FR-V2", f"FR-V{version}").replace("version: 2", f"version: {version}")
    oversized = extract(source, source_path=SOURCE_PATH.replace("FR-V2", f"FR-V{version}"))

    _assert_failure(storage, database, [oversized], "INVALID_SQLITE_VERSION")
    assert _logical_contents(database) == ((), (), ())
    _assert_summary(storage, storage.persist_draft_extractions(database, [extraction]), [extraction])


def test_foreign_keys_reject_cross_document_and_orphaned_links(modules, database, extraction):
    storage, extract = modules
    other = extract(SOURCE.replace("DEMO-999", "OTHER-999"), source_path=SOURCE_PATH.replace("DEMO", "OTHER"))
    storage.persist_draft_extractions(database, [extraction, other])
    before = _logical_contents(database)

    for parent_id in (other.parents[0].parent_id, "parent:absent"):
        with pytest.raises(sqlite3.IntegrityError) as error:
            database.execute("UPDATE draft_children SET parent_id = ? WHERE child_id = ?",
                             (parent_id, extraction.children[0].child_id))
        assert error.value.sqlite_errorname == "SQLITE_CONSTRAINT_FOREIGNKEY"
        database.rollback()
        assert _logical_contents(database) == before


@pytest.mark.parametrize(
    ("statement", "error_name"),
    [
        ("UPDATE draft_documents SET allowed_roles = '[]'", "SQLITE_CONSTRAINT_CHECK"),
        ("UPDATE draft_documents SET classification = NULL", "SQLITE_CONSTRAINT_NOTNULL"),
        ("UPDATE draft_children SET end = start", "SQLITE_CONSTRAINT_CHECK"),
        ("UPDATE draft_parents SET start = -1", "SQLITE_CONSTRAINT_CHECK"),
    ],
)
def test_sql_constraints_reject_missing_security_and_invalid_spans(
    modules, database, extraction, statement, error_name
):
    storage, _ = modules
    storage.persist_draft_extractions(database, [extraction])
    before = _logical_contents(database)
    with pytest.raises(sqlite3.IntegrityError) as error:
        database.execute(statement)
    assert error.value.sqlite_errorname == error_name
    database.rollback()
    assert _logical_contents(database) == before


def test_mid_write_sqlite_failure_rolls_back_documents_parents_and_children(modules, database, extraction):
    storage, extract = modules
    later = extract(SOURCE.replace("DEMO-999", "LATER-999"), source_path=SOURCE_PATH.replace("DEMO", "LATER"))
    database.execute(
        """
        CREATE TEMP TRIGGER abort_second_document
        BEFORE INSERT ON draft_children
        WHEN (SELECT count(*) FROM draft_documents) = 2
             AND (SELECT count(*) FROM draft_parents) > 1
             AND EXISTS (SELECT 1 FROM draft_children WHERE document_id <> NEW.document_id)
        BEGIN
            SELECT RAISE(ABORT, 'synthetic trigger failure');
        END
        """
    )
    observed = []
    database.set_trace_callback(
        lambda statement: observed.append(statement) if statement in {"BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"} else None
    )
    try:
        _assert_failure(storage, database, [later, extraction], "SQLITE_WRITE_FAILED")
    finally:
        database.set_trace_callback(None)
    assert "BEGIN IMMEDIATE" in observed and "ROLLBACK" in observed
    assert "COMMIT" not in observed
    assert _logical_contents(database) == ((), (), ())
    assert not database.in_transaction

    database.execute("DROP TRIGGER abort_second_document")
    _assert_summary(storage, storage.persist_draft_extractions(database, [later, extraction]), [later, extraction])


def test_failed_commit_returns_no_summary_and_rolls_back(modules, database, extraction):
    storage, _ = modules
    database.executescript(
        """
        CREATE TABLE deferred_parent (id INTEGER PRIMARY KEY);
        CREATE TABLE deferred_child (
            parent_id INTEGER REFERENCES deferred_parent(id) DEFERRABLE INITIALLY DEFERRED
        );
        CREATE TRIGGER defer_invalid_reference AFTER INSERT ON draft_children
        BEGIN
            INSERT INTO deferred_child VALUES (1);
        END;
        """
    )
    observed = []
    database.set_trace_callback(
        lambda statement: observed.append(statement) if statement in {"BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"} else None
    )
    try:
        _assert_failure(storage, database, [extraction], "SQLITE_WRITE_FAILED")
    finally:
        database.set_trace_callback(None)
    assert observed.index("COMMIT") < observed.index("ROLLBACK")
    assert database.execute("SELECT count(*) FROM deferred_child").fetchone()[0] == 0
    assert _logical_contents(database) == ((), (), ())
    assert not database.in_transaction

    database.execute("INSERT INTO deferred_parent VALUES (1)")
    database.commit()
    _assert_summary(storage, storage.persist_draft_extractions(database, [extraction]), [extraction])
