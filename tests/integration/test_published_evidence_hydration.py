"""Real ingestion/publication and source integrity, with fixture embeddings."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from integration.test_embedding_revision import dense_env, DAY
from integration.test_snapshot_consistency import store, published
from security.test_auth import secured


@pytest.fixture
def session(published, secured):
    from askpolicy.retrieval.evidence_hydration import hydrate_published_hits, EvidenceHydrationError
    from types import SimpleNamespace
    root, handle, opts = published
    context = secured.authenticate(secured.sign(roles=["employee", "hr"]), "hr")
    hits = handle.search(context, query="congés registre des soldes", validity_date=DAY)
    return SimpleNamespace(root=root, handle=handle, opts=opts, context=context, hits=hits,
                           hydrate=hydrate_published_hits, error=EvidenceHydrationError)


def hydrate(session, **changes):
    return session.hydrate(
        changes.get("handle", session.handle), session.context, session.hits,
        release_root=session.root, validity_date=DAY,
    )


def test_real_search_top_twenty_exact_sources_and_determinism(session):
    passages = hydrate(session)
    assert len(passages) == len({p.child_id for p in passages}) == 20
    assert passages == hydrate(session)
    path = session.root / "releases" / session.handle.release_id / "sqlite/corpus.sqlite"
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)) as db:
        for passage, hit in zip(passages, session.hits, strict=True):
            child = db.execute("SELECT document_id, parent_id, start, end, text FROM draft_children WHERE child_id = ?",
                               (passage.child_id,)).fetchone()
            assert child == (passage.document_id, passage.parent_id, passage.start, passage.end, passage.text)
            parent_doc, start, end, parent = db.execute(
                "SELECT document_id, start, end, text FROM draft_parents WHERE parent_id = ?",
                (passage.parent_id,),
            ).fetchone()
            source, = db.execute("SELECT canonical_source FROM draft_documents WHERE document_id = ?",
                                 (passage.document_id,)).fetchone()
            assert parent_doc == passage.document_id
            assert start <= passage.start < passage.end <= end
            assert parent[passage.start-start:passage.end-start] == passage.text
            assert source[passage.start:passage.end] == passage.text
            assert passage.snapshot_id == session.handle.bm25.snapshot.candidate_snapshot_id
            assert passage.release_id == session.handle.release_id
            assert passage.active_role in hit.child.document.metadata.allowed_roles


def test_pointer_switch_during_hydration_keeps_original_release(session, dense_env, store, monkeypatch):
    from askpolicy.retrieval import evidence_hydration as module
    from integration.test_role_bm25_draft import _source_fixture
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    original = hydrate(session)
    changed_source = "850 MAD".encode()
    # A genuinely different approved fixture release B, built through the same
    # production ingestion, index validation and atomic publication flow.
    corpus_root = session.root / "changed-input"
    manifest = _source_fixture(corpus_root, changed_source, b"851 MAD")
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(corpus_root, db)
        options = session.opts | {"manifest": manifest}
        previous = module._verified_bytes
        published_ids = []
        def switch_then_read(handle, root):
            next_handle = store[1].publish_snapshot(db, root, **options)
            published_ids.append(next_handle.release_id)
            return previous(handle, root)
        monkeypatch.setattr(module, "_verified_bytes", switch_then_read)
        assert hydrate(session) == original
    assert published_ids and published_ids[0] != session.handle.release_id
    active = store[0].read_json((session.root / "active.json").read_bytes())
    assert active["release_id"] == published_ids[0]


@pytest.mark.parametrize("pointer", [b"invalid", b'{"release_id":"wrong"}'])
def test_never_rereads_active_pointer(session, pointer):
    before = hydrate(session)
    (session.root / "active.json").write_bytes(pointer)
    assert hydrate(session) == before


@pytest.mark.parametrize("defect", [
    "sqlite_hash", "truncated", "sqlite_missing", "manifest_tamper", "manifest_missing",
    "release_missing", "release_is_file", "child_text", "parent_association", "missing_child",
    "manifest_hash", "release_id", "artifact_paths", "artifact_set", "binding",
    "wal", "shm", "journal",
])
def test_artifact_tamper_and_malformed_release_fail_closed(session, store, defect):
    directory = session.root / "releases" / session.handle.release_id
    sqlite_path = directory / "sqlite/corpus.sqlite"
    manifest_path = directory / "manifest.json"
    handle = session.handle
    if defect in {"child_text", "parent_association", "missing_child"}:
        with closing(sqlite3.connect(sqlite_path)) as db:
            cid = session.hits[-1].child.child_id
            if defect == "child_text":
                db.execute("PRAGMA ignore_check_constraints = ON")
                db.execute("UPDATE draft_children SET text = 'tampered' WHERE child_id = ?", (cid,))
            elif defect == "parent_association":
                db.execute("UPDATE draft_children SET parent_id = 'wrong' WHERE child_id = ?", (cid,))
            else:
                db.execute("DELETE FROM draft_children WHERE child_id = ?", (cid,))
            db.commit()
    elif defect == "sqlite_hash":
        sqlite_path.write_bytes(sqlite_path.read_bytes() + b"tamper")
    elif defect == "truncated":
        sqlite_path.write_bytes(sqlite_path.read_bytes()[:100])
    elif defect == "sqlite_missing":
        sqlite_path.unlink()
    elif defect == "manifest_tamper":
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    elif defect == "manifest_missing":
        manifest_path.unlink()
    elif defect in {"release_missing", "release_is_file"}:
        directory.rename(directory.with_name("retained-for-test"))
        if defect == "release_is_file":
            directory.write_bytes(b"invalid directory")
    elif defect == "manifest_hash":
        handle = replace(handle, manifest_hash="0"*64)
    elif defect == "release_id":
        handle = replace(handle, release_id="0"*64)
    elif defect in {"artifact_paths", "artifact_set", "binding"}:
        record = store[0].read_json(manifest_path.read_bytes())
        if defect == "artifact_paths":
            record["artifact_paths"]["sqlite"] = "../unverified.sqlite"
        elif defect == "artifact_set":
            del record["artifacts"]["faiss.hr"]
        else:
            record["snapshot"]["candidate_snapshot_id"] = "0"*64
        raw = store[0].canonical(record)
        new_id = store[0]._identity(record)
        revised = directory.with_name(new_id)
        directory.rename(revised)
        (revised / "manifest.json").write_bytes(raw)
        handle = replace(handle, release_id=new_id, manifest_hash=store[0].digest(raw))
    else:
        sqlite_path.with_name(sqlite_path.name + "-" + defect).write_bytes(b"unverified sidecar")
    with pytest.raises(session.error, match="^EVIDENCE_HYDRATION_FAILED$"):
        hydrate(session, handle=handle)


def test_no_hash_then_reopen_race(session, monkeypatch):
    expected = hydrate(session)
    path = session.root / "releases" / session.handle.release_id / "sqlite/corpus.sqlite"
    original = Path.read_bytes
    reads = []
    def replace_after_read(current):
        raw = original(current)
        if current == path:
            reads.append(1)
            current.write_bytes(b"replacement after verified read")
        return raw
    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    assert hydrate(session) == expected
    assert reads == [1]


def test_reparse_point_rejected(session):
    from types import SimpleNamespace
    from unittest.mock import patch
    with patch("pathlib.Path.lstat", return_value=SimpleNamespace(st_mode=0, st_file_attributes=1024)):
        with pytest.raises(session.error):
            hydrate(session)


def test_time_transition_rejects_pinned_release_before_text(session, dense_env, store, monkeypatch):
    from integration.test_role_bm25_draft import _source_fixture
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval import evidence_hydration as module
    source = session.root / "expiring-input"
    manifest = _source_fixture(source, b"expires_at: null", b'expires_at: "2026-09-02"')
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(source, db)
        handle = store[1].publish_snapshot(db, session.root, **(session.opts | {"manifest": manifest}))
    assert handle.bm25.eligibility.next_transition_date == DAY.replace(day=2)
    hits = handle.search(session.context, query="travel policy", validity_date=DAY)
    reads = []
    monkeypatch.setattr(module, "_verified_bytes", lambda *args: reads.append(1))
    with pytest.raises(session.error):
        session.hydrate(handle, session.context, hits, release_root=session.root, validity_date=DAY.replace(day=2))
    assert reads == []
