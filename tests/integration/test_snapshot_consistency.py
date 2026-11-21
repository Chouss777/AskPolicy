"""Persisted source and index binding checks with fixture embeddings."""

from dataclasses import replace
import json

import pytest

from integration.test_embedding_revision import dense_env, DAY
from security.test_auth import secured


@pytest.fixture
def store(dense_env):
    from askpolicy.storage import index_store, publication
    return index_store, publication


@pytest.fixture
def published(dense_env, store, tmp_path):
    index_store, publication = store
    opts = dense_env.options | {"encoder": dense_env.encoder}
    root = tmp_path / "runtime"
    handle = publication.publish_snapshot(dense_env.db, root, **opts)
    return root, handle, opts


def test_sqlite_and_all_eight_indexes_match_trusted_inputs(published, store):
    root, handle, opts = published
    loaded = store[1].load_active_snapshot(root, **opts)
    assert loaded.release_id == handle.release_id and loaded.status == "PUBLISHED"
    assert loaded.faiss.snapshot.status == "UNPUBLISHED"
    assert loaded.bm25.snapshot == loaded.faiss.snapshot
    assert loaded.bm25.snapshot.document_count == 26
    assert len(loaded.bm25.indexes) == len(loaded.faiss.indexes) == 4
    assert loaded.faiss._encoder is opts["encoder"]
    counts = {"employee": (15,229), "hr": (18,271), "finance": (15,229), "it_security": (17,259)}
    for left, right in zip(loaded.bm25.indexes, loaded.faiss.indexes, strict=True):
        assert left.metadata.indexed_child_ids == right.metadata.indexed_child_ids
        assert (len(left.metadata.indexed_document_ids), len(left.metadata.indexed_child_ids)) == counts[left.metadata.role]
    assert len(list((root / "releases" / handle.release_id).rglob("index.faiss"))) == 4


@pytest.mark.parametrize("component", ["sqlite", "bm25.employee.data.csc.index.npy", "faiss.hr",
                                     "bm25.finance.params.index.json", "faiss.it_security"])
@pytest.mark.parametrize("operation", ["missing", "corrupt"])
def test_missing_or_corrupted_components_fail_closed(published, store, component, operation):
    root, handle, opts = published
    path = root / "releases" / handle.release_id / store[0].artifact_paths()[component]
    if operation == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupted artifact")
    with pytest.raises(store[0].IndexStoreError):
        store[1].load_active_snapshot(root, **opts)


@pytest.mark.parametrize("defect", ["manifest", "config", "date", "model"])
def test_trusted_input_mismatch_rejected(published, store, monkeypatch, defect):
    root, handle, opts = published
    opts = dict(opts)
    if defect == "manifest":
        opts["manifest"] = b"{}"
    elif defect == "config":
        opts["snapshot_config"] = replace(opts["snapshot_config"], identity_encoding="json-array-v1")
    elif defect == "date":
        opts["validity_date"] = DAY.replace(day=2)
    else:
        monkeypatch.setattr(opts["encoder"], "revision", "main")
    with pytest.raises(store[0].IndexStoreError):
        store[1].load_active_snapshot(root, **opts)


def test_symlink_escape_and_duplicate_json_rejected(published, store, tmp_path):
    root, handle, opts = published
    assert store[0].read_json(b'{"one":1}') == {"one":1}
    with pytest.raises(store[0].IndexStoreError):
        store[0].read_json(b'{"one":1,"one":2}')
    # Check the reparse-point guard without requiring Windows link privileges.
    from unittest.mock import patch
    from types import SimpleNamespace
    with patch("pathlib.Path.lstat", return_value=SimpleNamespace(st_mode=0, st_file_attributes=1024)):
        with pytest.raises(store[0].IndexStoreError, match="^UNSAFE_PATH$"):
            store[0].safe_path(root)


@pytest.mark.parametrize("defect", ["lexical", "vector-count", "mapping", "roles", "unsafe-path", "revision"])
def test_rehashed_artifacts_still_require_trusted_source_and_configuration(published, store, defect):
    import faiss
    root, handle, opts = published
    index_store, publication = store
    directory = root / "releases" / handle.release_id
    record = index_store.read_json((directory / "manifest.json").read_bytes())
    if defect == "lexical":
        slot = "bm25.employee.params.index.json"
        path = directory / index_store.artifact_paths()[slot]
        params = json.loads(path.read_bytes())
        params["k1"] = 99
        path.write_bytes(index_store.canonical(params))
        record["artifacts"][slot] = index_store.digest(path.read_bytes())
    elif defect == "vector-count":
        slot = "faiss.employee"
        path = directory / index_store.artifact_paths()[slot]
        faiss.write_index(faiss.IndexFlatIP(384), str(path))
        record["artifacts"][slot] = index_store.digest(path.read_bytes())
    elif defect == "mapping":
        record["dense"][0]["rows"].reverse()
    elif defect == "roles":
        record["lexical"][0]["role"] = "hr"
    elif defect == "unsafe-path":
        record["artifact_paths"]["sqlite"] = "../outside.sqlite"
    else:
        record["dense"][0]["configuration"]["revision"] = "main"
    new_id = index_store._identity(record)
    revised = root / "releases" / new_id
    directory.rename(revised)
    raw = index_store.canonical(record)
    (revised / "manifest.json").write_bytes(raw)
    (root / "active.json").write_bytes(index_store.canonical(
        dict(format=1, release_id=new_id, manifest_hash=index_store.digest(raw))))
    with pytest.raises(index_store.IndexStoreError):
        publication.load_active_snapshot(root, **opts)
