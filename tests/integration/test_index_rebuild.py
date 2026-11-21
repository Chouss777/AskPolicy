"""Explicit rebuild, safe serialization and actual loaded-engine scoring."""

from contextlib import closing
from dataclasses import FrozenInstanceError
from hashlib import sha256
import sqlite3

import numpy as np
import pytest

from integration.test_snapshot_consistency import store, published
from integration.test_embedding_revision import dense_env
from security.test_auth import secured


def test_rebuild_from_persisted_sqlite_has_identical_index_metadata(published, store, tmp_path, secured):
    root, first, opts = published
    path = root / "releases" / first.release_id / "sqlite/corpus.sqlite"
    with closing(sqlite3.connect(path)) as db:
        release = store[0].stage_snapshot(db, tmp_path / "rebuilt", **opts)
    rebuilt = store[0].load_snapshot(tmp_path / "rebuilt", expected_release_id=release, **opts)
    assert [i.metadata for i in first.faiss.indexes] == [i.metadata for i in rebuilt.faiss.indexes]
    assert [i.metadata for i in first.bm25.indexes] == [i.metadata for i in rebuilt.bm25.indexes]
    for index in rebuilt.faiss.indexes:
        vectors = index._flat.engine.reconstruct_n(0, index._flat.engine.ntotal)
        assert sha256(vectors.astype("<f4",copy=False).tobytes()).hexdigest() == index.metadata.vector_hash
        assert len(index.metadata.rows) == index._flat.engine.ntotal
        assert index.metadata.configuration.revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    context = secured.authenticate(secured.sign(), "employee")
    hits = rebuilt.search(context, query="travel", validity_date=opts["validity_date"])
    assert hits == first.search(context, query="travel", validity_date=opts["validity_date"])
    assert len(hits) == 20 and all(h.active_role == "employee" for h in hits)
    with pytest.raises(FrozenInstanceError):
        rebuilt.release_id = "0"*64


def test_restart_load_does_not_encode_passages_or_stage(published, store, monkeypatch, secured):
    root, original, opts = published
    def forbidden(*args, **kwargs):
        pytest.fail("active read attempted staging or passage inference")
    monkeypatch.setattr(opts["encoder"], "encode_passages", forbidden)
    monkeypatch.setattr(store[1], "stage_snapshot", forbidden)
    loaded = store[1].load_active_snapshot(root, **opts)
    assert loaded is not original and loaded.bm25 is not original.bm25
    assert loaded.bm25.indexes[0]._engine is not original.bm25.indexes[0]._engine
    hits = loaded.search(secured.authenticate(secured.sign(), "employee"), query="travel",
                         validity_date=opts["validity_date"])
    assert len(hits) == 20


@pytest.mark.parametrize("defect", ["source", "schema", "child", "transaction"])
def test_invalid_draft_cannot_be_staged(dense_env, store, tmp_path, defect):
    env = dense_env
    with closing(sqlite3.connect(":memory:")) as db:
        env.db.backup(db)
        if defect == "source":
            db.execute("UPDATE draft_documents SET canonical_source = canonical_source || 'x'")
        elif defect == "schema":
            db.execute("ALTER TABLE draft_children RENAME COLUMN text TO broken")
        elif defect == "child":
            db.execute("DELETE FROM draft_children WHERE rowid=(SELECT min(rowid) FROM draft_children)")
        else:
            db.execute("BEGIN")
        if defect != "transaction":
            db.commit()
        with pytest.raises(store[0].IndexStoreError):
            store[0].stage_snapshot(db, tmp_path / "invalid", **(env.options | {"encoder":env.encoder}))


def test_serialization_uses_safe_numpy_without_pickle(published, store):
    root, handle, opts = published
    path = root / "releases" / handle.release_id
    arrays = list(path.rglob("*.npy"))
    assert len(arrays) == 12
    for artifact in arrays:
        assert np.load(artifact, allow_pickle=False).dtype.kind in "ifu"
    assert not list(path.rglob("*.pkl"))


def test_lexical_serialization_is_stable_across_process_hash_seeds(store, tmp_path):
    import os
    import subprocess
    import sys
    from security.test_auth import ROOT
    script = """
import json, sys
import bm25s
from pathlib import Path
from askpolicy.retrieval.bm25_core import build_text_index
from askpolicy.storage.index_store import _save_lexical
path = Path(sys.argv[1])
model = build_text_index(["travel lodging employee", "leave hr approval", "employee travel approval"])
_save_lexical(model, path)
loaded = bm25s.BM25.load(path, allow_pickle=False, load_corpus=False, show_progress=False, verbosity=0)
for term in ("travel", "approval", "hr"):
    assert (model.get_scores([term]) == loaded.get_scores([term])).all()
"""
    paths = []
    for seed in ("17", "91"):
        directory = tmp_path / seed
        child = subprocess.run([sys.executable, "-c", script, str(directory)],
            env={**os.environ, "PYTHONHASHSEED":seed, "PYTHONPATH":str(ROOT/"src")},
            capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
        paths.append(directory)
    assert {p.name:p.read_bytes() for p in paths[0].iterdir()} == {
        p.name:p.read_bytes() for p in paths[1].iterdir()}
