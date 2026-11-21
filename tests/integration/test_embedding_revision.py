"""Offline contract tests using deterministic fixture vectors and genuine FAISS."""

from contextlib import closing
from dataclasses import replace
from datetime import date
from hashlib import sha256
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from integration.test_role_bm25_draft import _source_fixture
from security.test_auth import ROOT, secured


DAY = date(2026, 9, 1)
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


@pytest.fixture(scope="module")
def dense_env():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
        from askpolicy.retrieval import role_bm25, role_faiss as dense
        from askpolicy.retrieval.e5_encoder import EmbeddingBatch

        # Only the explicit local runner exercises real weights. The fixture
        # ties a repeatable unit vector to the exact input bytes for ACL tests.
        encoder = object.__new__(dense.E5Encoder)
        encoder.artifact_fingerprint = dense.ARTIFACT_FINGERPRINT
        encoder.passage_token_count = lambda text: 3 + len(text)
        batches = []

        def encode(texts):
            vectors = []
            for text in texts:
                seed = int.from_bytes(sha256(text.encode()).digest()[:8], "big")
                vector = np.random.default_rng(seed).standard_normal(384, dtype=np.float32)
                vectors.append(vector / np.linalg.norm(vector))
            return EmbeddingBatch(np.stack(vectors), tuple(encoder.passage_token_count(t) for t in texts))

        def passages(texts):
            batches.append(tuple(sha256(t.encode()).digest() for t in texts))
            return encode(texts)

        encoder.encode_queries = encode
        encoder.encode_passages = passages
        opts = dict(manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
                    snapshot_config=dense.DraftSnapshotConfig(), validity_date=DAY)
        with closing(sqlite3.connect(":memory:")) as db:
            ingest_manifest_to_draft(ROOT, db)
            bundle = dense.build_role_faiss_bundle(db, encoder=encoder, **opts)
            bm25 = role_bm25.build_role_bm25_bundle(db, **opts)
            yield SimpleNamespace(module=dense, encoder=encoder, bundle=bundle, db=db,
                                  bm25=bm25, options=opts, batches=tuple(batches))


def search(env, context, query, *, bundle=None, **opts):
    binding = {"expected_snapshot_id": env.bundle.snapshot.candidate_snapshot_id, "validity_date": DAY}
    return env.module.search_role_faiss(env.bundle if bundle is None else bundle, context,
                                        query=query, **(binding | opts))


def test_real_loader_rejects_wrong_missing_or_mismatched_local_artifacts(dense_env, tmp_path):
    from askpolicy.retrieval.e5_encoder import E5Encoder, E5EncodingError

    path = tmp_path / REVISION
    assert E5Encoder.revision == REVISION
    with pytest.raises(E5EncodingError, match="^MODEL_REVISION_MISMATCH$"):
        E5Encoder(path, revision="main")
    with pytest.raises(E5EncodingError, match="^MODEL_FILES_MISSING$"):
        E5Encoder(path)
    path.mkdir()
    (path / "config.json").write_bytes(b"modified")
    with pytest.raises(E5EncodingError, match="^MODEL_ARTIFACT_MISMATCH$"):
        E5Encoder(path)


def test_independent_private_images_have_identical_role_memberships(dense_env):
    env = dense_env
    env.module.verify_matching_bundles(env.bm25, env.bundle)
    assert env.bm25.snapshot == env.bundle.snapshot
    assert env.bm25.snapshot is not env.bundle.snapshot
    assert env.bundle.snapshot.status == "UNPUBLISHED"
    for index in env.bundle.indexes:
        meta = index.metadata
        other = next(x.metadata for x in env.bm25.indexes if x.metadata.role == meta.role)
        assert meta.indexed_document_ids == other.indexed_document_ids
        assert meta.indexed_child_ids == other.indexed_child_ids
        assert meta.configuration.revision == REVISION == env.encoder.revision
        assert meta.configuration.artifact_fingerprint == env.encoder.artifact_fingerprint


def test_distinct_valid_snapshots_and_dates_cannot_be_combined(dense_env):
    env = dense_env
    lexical = env.module.lexical
    for change, code in [
        (dict(snapshot_config=env.module.DraftSnapshotConfig(identity_encoding="json-array-v1")),
         "BUNDLE_SNAPSHOT_MISMATCH"),
        (dict(validity_date=date(2026, 9, 2)), "BUNDLE_MEMBERSHIP_MISMATCH"),
    ]:
        other = lexical.build_role_bm25_bundle(env.db, **(env.options | change))
        with pytest.raises(env.module.DenseIndexError, match=f"^{code}$"):
            env.module.verify_matching_bundles(other, env.bundle)


@pytest.mark.parametrize("field,value", [
    ("revision", "main"), ("artifact_fingerprint", "0" * 64), ("dimension", 383),
])
def test_changed_query_encoder_fails_before_inference(dense_env, secured, monkeypatch, field, value):
    env = dense_env
    monkeypatch.setattr(env.encoder, field, value)
    monkeypatch.setattr(env.encoder, "encode_queries", lambda _: pytest.fail("unexpected inference"))
    with pytest.raises(env.module.DenseIndexError, match="^ENCODER_BINDING_MISMATCH$"):
        search(env, secured.authenticate(secured.sign(), "employee"), "test")


@pytest.mark.parametrize("field", ["source_manifest_hash", "snapshot_config_hash", "candidate_snapshot_id",
                                  "indexed_child_ids", "indexed_document_ids", "configuration"])
def test_tampered_index_rejected_before_search_or_pairing(dense_env, secured, monkeypatch, field):
    env = dense_env
    index, *others = env.bundle.indexes
    value = getattr(index.metadata, field)
    if field == "configuration":
        value = replace(value, revision="unapproved")
    elif isinstance(value, tuple):
        value = value[1:]
    else:
        value = "0" * 64
    broken = replace(env.bundle, indexes=(replace(index, metadata=replace(index.metadata, **{field: value})), *others))
    monkeypatch.setattr(env.encoder, "encode_queries", lambda _: pytest.fail("unexpected inference"))
    with pytest.raises(env.module.DenseIndexError, match="^INDEX_METADATA_MISMATCH$"):
        search(env, secured.authenticate(secured.sign(), "employee"), "test", bundle=broken)
    with pytest.raises(env.module.DenseIndexError):
        env.module.verify_matching_bundles(env.bm25, broken)


@pytest.mark.parametrize("opts,code", [
    ({"expected_snapshot_id": "0" * 64}, "SNAPSHOT_MISMATCH"),
    ({"validity_date": date(2026, 8, 31)}, "VALIDITY_DATE_MISMATCH"),
])
def test_exact_request_binding_before_inference(dense_env, secured, monkeypatch, opts, code):
    monkeypatch.setattr(dense_env.encoder, "encode_queries", lambda _: pytest.fail("unexpected inference"))
    with pytest.raises(dense_env.module.DenseIndexError, match=f"^{code}$"):
        search(dense_env, secured.authenticate(secured.sign(), "employee"), "test", **opts)


def test_expiration_boundary_denied_before_query(dense_env, tmp_path, secured, monkeypatch):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    env = dense_env
    manifest = _source_fixture(tmp_path, b"expires_at: null", b'expires_at: "2026-09-02"')
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(tmp_path, db)
        bundle = env.module.build_role_faiss_bundle(db, **(env.options | {"manifest": manifest}), encoder=env.encoder)
    tomorrow = date(2026, 9, 2)
    assert all(i.metadata.next_transition_date == tomorrow for i in bundle.indexes)
    monkeypatch.setattr(env.encoder, "encode_queries", lambda _: pytest.fail("unexpected inference"))
    with pytest.raises(env.module.DenseIndexError, match="^STALE_INDEX$"):
        search(env, secured.authenticate(secured.sign(), "employee"), "test", bundle=bundle,
               expected_snapshot_id=bundle.snapshot.candidate_snapshot_id, validity_date=tomorrow)


def test_gap_or_overlap_in_derived_span_rejected_at_build(dense_env, monkeypatch):
    env = dense_env
    split = env.module.split_passage

    def shifted(chunk, count):
        for p in split(chunk, count):
            yield replace(p, segment=replace(p.segment, start=p.segment.start + 1))

    monkeypatch.setattr(env.module, "split_passage", shifted)
    with pytest.raises(env.module.DenseIndexError, match="^INVALID_SOURCE_SPAN$"):
        env.module.build_role_faiss_bundle(env.db, encoder=env.encoder, **env.options)
