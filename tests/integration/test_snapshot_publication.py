"""Atomic publication, interruption, single-writer and reader isolation."""

from datetime import date
import json
from pathlib import Path
from threading import Event, Thread

import pytest

from integration.test_snapshot_consistency import store, published
from integration.test_embedding_revision import dense_env
from security.test_auth import secured
from integration.test_role_bm25_draft import _source_fixture


@pytest.mark.parametrize("stage", ["partial", "corrupt", "pointer"])
def test_failed_publication_retains_previous_release(published, store, dense_env, monkeypatch, stage):
    root, old, opts = published
    publication = store[1]
    old_pointer = (root / "active.json").read_bytes()
    original = publication.stage_snapshot
    if stage == "pointer":
        def fail(*args):
            raise OSError("synthetic failure")
        monkeypatch.setattr(publication, "_switch_pointer", fail)
    else:
        def fail(connection, directory, **kwargs):
            if stage == "partial":
                directory.mkdir(parents=True)
                (directory / "partial").write_bytes(b"incomplete")
                raise OSError("synthetic interruption")
            release = original(connection, directory, **kwargs)
            (directory / "faiss/hr/index.faiss").write_bytes(b"broken")
            return release
        monkeypatch.setattr(publication, "stage_snapshot", fail)
    with pytest.raises(store[0].IndexStoreError):
        publication.publish_snapshot(dense_env.db, root, **opts)
    assert (root / "active.json").read_bytes() == old_pointer
    assert publication.load_active_snapshot(root, **opts).release_id == old.release_id
    assert not (root / ".writer.lock").exists()


def test_reader_pins_release_during_switch_and_explicit_rollback(published, store, dense_env, secured, monkeypatch):
    root, first, opts = published
    publication = store[1]
    context = secured.authenticate(secured.sign(), "employee")
    before = first.search(context, query="travel", validity_date=opts["validity_date"])
    second_opts = opts | {"validity_date": date(2026,9,2)}
    replace_pointer = publication.os.replace
    observed = []
    def observe(source, target):
        assert Path(source).parent == Path(target).parent
        # New release is finalized while the old pointer still loads completely.
        observed.append(publication.load_active_snapshot(root, **opts).release_id)
        replace_pointer(source, target)
    monkeypatch.setattr(publication.os, "replace", observe)
    second = publication.publish_snapshot(dense_env.db, root, **second_opts)
    assert observed == [first.release_id] and second.release_id != first.release_id
    assert first.search(context, query="travel", validity_date=opts["validity_date"]) == before
    assert publication.load_active_snapshot(root, **second_opts).release_id == second.release_id
    assert (root / "releases" / first.release_id).exists()
    monkeypatch.setattr(publication.os, "replace", replace_pointer)
    rolled_back = publication.publish_snapshot(dense_env.db, root, **opts)
    assert rolled_back.release_id == first.release_id
    assert publication.load_active_snapshot(root, **opts).release_id == first.release_id


def test_concurrent_writer_rejected_while_existing_reader_operates(published, store, dense_env):
    root, old, opts = published
    entered, finish = Event(), Event()
    errors = []
    def owner():
        try:
            with store[1]._writer(root):
                entered.set()
                assert finish.wait(30)
        except BaseException as exc:
            errors.append(exc)
    worker = Thread(target=owner)
    worker.start()
    assert entered.wait(10)
    try:
        with pytest.raises(store[0].IndexStoreError, match="^PUBLICATION_BUSY$"):
            store[1].publish_snapshot(dense_env.db, root, **opts)
        assert store[1].load_active_snapshot(root, **opts).release_id == old.release_id
    finally:
        finish.set()
        worker.join(10)
    assert not worker.is_alive() and not errors


@pytest.mark.parametrize("pointer", [None, b"invalid", b'{"release_id":"../escape"}',
                                    b'{"format":1,"release_id":"global","manifest_hash":"zero"}'])
def test_missing_or_malformed_pointer_has_no_fallback(published, store, pointer):
    root, old, opts = published
    path = root / "active.json"
    if pointer is None:
        path.unlink()
    else:
        path.write_bytes(pointer)
    with pytest.raises(store[0].IndexStoreError):
        store[1].load_active_snapshot(root, **opts)
    assert (root / "releases" / old.release_id).exists()


def test_missing_active_directory_does_not_load_previous(published, store):
    root, old, opts = published
    (root / "active.json").write_text(json.dumps(dict(format=1,release_id="0"*64,manifest_hash="0"*64)), encoding="utf8")
    with pytest.raises(store[0].IndexStoreError):
        store[1].load_active_snapshot(root, **opts)


def test_boolean_format_is_not_a_valid_publication_version(published, store):
    root, old, opts = published
    path = root / "active.json"
    pointer = json.loads(path.read_bytes())
    pointer["format"] = True
    path.write_text(json.dumps(pointer), encoding="utf8")
    with pytest.raises(store[0].IndexStoreError, match="^INVALID_ACTIVE_POINTER$"):
        store[1].load_active_snapshot(root, **opts)


def test_expiration_blocks_pinned_reader_and_reload(dense_env, store, tmp_path, secured):
    from contextlib import closing
    import sqlite3
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    manifest = _source_fixture(tmp_path, b"expires_at: null", b'expires_at: "2026-09-02"')
    opts = dense_env.options | {"manifest":manifest,"encoder":dense_env.encoder}
    root = tmp_path / "runtime"
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(tmp_path, db)
        handle = store[1].publish_snapshot(db, root, **opts)
    assert handle.faiss._draft.eligibility.next_transition_date == date(2026,9,2)
    with pytest.raises(ValueError, match="^STALE_INDEX$"):
        handle.search(secured.authenticate(secured.sign(),"employee"),query="travel",validity_date=date(2026,9,2))
    with pytest.raises(store[0].IndexStoreError):
        store[1].load_active_snapshot(root, **(opts | {"validity_date":date(2026,9,2)}))
