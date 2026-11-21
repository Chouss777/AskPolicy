"""Validated SQLite, real BM25S/FAISS and deterministic fixture embeddings."""

from contextlib import closing
from dataclasses import replace
from datetime import date
import sqlite3

import faiss
import pytest

from integration.test_embedding_revision import dense_env, DAY
from integration.test_role_bm25_draft import _source_fixture
from security.test_auth import ROOT, secured


@pytest.fixture
def hybrid(dense_env):
    from askpolicy.retrieval import hybrid
    return hybrid


def run(env, module, context, *, bm25=None, dense=None, **opts):
    options = dict(query="travel hébergement déplacement", expected_snapshot_id=env.bundle.snapshot.candidate_snapshot_id,
                   validity_date=DAY)
    return module.search_role_hybrid(env.bm25 if bm25 is None else bm25, env.bundle if dense is None else dense,
                                     context, **(options | opts))


def test_matching_real_engines_budgets_rank_math_and_determinism(dense_env, hybrid, secured, monkeypatch):
    env = dense_env
    ctx = secured.authenticate(secured.sign(), "employee")
    query, = env.db.execute(
        "SELECT text FROM draft_children WHERE document_id='TRAVEL-001-FR-V1' "
        "AND length(text) BETWEEN 80 AND 400 ORDER BY length(text) DESC LIMIT 1"
    ).fetchone()
    calls = []
    for module, name in ((hybrid.lexical, "search_role_bm25"), (hybrid.dense, "search_role_faiss")):
        original = getattr(module, name)
        def observe(*args, _fn=original, _name=name, **kwargs):
            hits = _fn(*args, **kwargs)
            calls.append((_name, hits))
            return hits
        monkeypatch.setattr(module, name, observe)
    hits = run(env, hybrid, ctx, query=query)
    assert len(hits) == len({h.child.child_id for h in hits}) == 20
    assert [c[0] for c in calls] == ["search_role_bm25", "search_role_faiss"]
    left, right = [c[1] for c in calls]
    assert 0 < len(left) <= 30 and len(right) == 30
    assert any(h.bm25_rank is not None and h.faiss_rank is not None for h in hits)
    assert hits == run(env, hybrid, ctx, query=query)
    assert hits == tuple(sorted(hits, key=lambda h: (-h.rrf_score, h.child.child_id)))
    for h in hits:
        score = sum(1 / (60 + rank) for rank in (h.bm25_rank, h.faiss_rank) if rank is not None)
        assert h.rrf_score == score
    assert env.bundle.snapshot.status == "UNPUBLISHED"


@pytest.mark.parametrize("defect", ["snapshot", "missing-bm25", "missing-dense"])
def test_separate_valid_snapshot_or_absent_bundle_rejected(dense_env, hybrid, secured, monkeypatch, defect):
    env = dense_env
    bm25, dense = env.bm25, env.bundle
    if defect == "snapshot":
        opts = env.options | {"snapshot_config": hybrid.dense.DraftSnapshotConfig(identity_encoding="json-array-v1")}
        bm25 = hybrid.lexical.build_role_bm25_bundle(env.db, **opts)
        assert bm25.snapshot != dense.snapshot
    elif defect == "missing-bm25":
        bm25 = None
    else:
        dense = None
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible bundles reached search")
    monkeypatch.setattr(hybrid.lexical, "search_role_bm25", forbidden)
    monkeypatch.setattr(hybrid.dense, "search_role_faiss", forbidden)
    monkeypatch.setattr(env.encoder, "encode_queries", forbidden)
    with pytest.raises(hybrid.HybridError):
        hybrid.search_role_hybrid(bm25, dense, secured.authenticate(secured.sign(), "employee"), query="travel",
            expected_snapshot_id=env.bundle.snapshot.candidate_snapshot_id, validity_date=DAY)


@pytest.mark.parametrize("defect", ["snapshot", "manifest", "config", "date", "missing-bm25", "missing-dense",
                                   "missing-role", "swap", "revision", "membership", "vector", "mapping", "lexical"])
def test_gate_prevents_both_engines_and_encoding(dense_env, hybrid, secured, monkeypatch, defect):
    env = dense_env
    bm, dense = env.bm25, env.bundle
    opts = {}
    first, *others = dense.indexes
    if defect in ("snapshot", "date"):
        opts = {"expected_snapshot_id": "0" * 64} if defect == "snapshot" else {"validity_date": date(2026, 9, 2)}
    elif defect in ("missing-bm25", "missing-dense"):
        bm = object() if defect == "missing-bm25" else bm
        dense = object() if defect == "missing-dense" else dense
    elif defect == "missing-role":
        dense = replace(dense, indexes=tuple(others))
    elif defect == "swap":
        dense = replace(dense, indexes=(replace(first, _flat=others[0]._flat), *others))
    elif defect == "vector":
        engine = faiss.clone_index(first._flat.engine)
        engine.reset()
        dense = replace(dense, indexes=(replace(first, _flat=replace(first._flat, engine=engine)), *others))
    elif defect == "lexical":
        index, *rest = bm.indexes
        bm = replace(bm, indexes=(replace(index, metadata=replace(index.metadata, source_manifest_hash="0"*64)), *rest))
    else:
        fields = {"manifest": {"source_manifest_hash": "0"*64}, "config": {"snapshot_config_hash": "0"*64},
                  "revision": {"configuration": replace(first.metadata.configuration, revision="main")},
                  "membership": {"indexed_child_ids": first.metadata.indexed_child_ids[1:]},
                  "mapping": {"rows": first.metadata.rows[::-1]}}
        dense = replace(dense, indexes=(replace(first, metadata=replace(first.metadata, **fields[defect])), *others))
    def forbidden(*args, **kwargs):
        pytest.fail("failed gate reached search or query encoding")
    monkeypatch.setattr(hybrid.lexical, "search_role_bm25", forbidden)
    monkeypatch.setattr(hybrid.dense, "search_role_faiss", forbidden)
    monkeypatch.setattr(env.encoder, "encode_queries", forbidden)
    with pytest.raises(hybrid.HybridError):
        run(env, hybrid, secured.authenticate(secured.sign(), "employee"), bm25=bm, dense=dense, **opts)


def test_actual_stale_transition_fails_before_scoring(dense_env, hybrid, tmp_path, secured, monkeypatch):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft

    env = dense_env
    manifest = _source_fixture(tmp_path, b"expires_at: null", b'expires_at: "2026-09-02"')
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(tmp_path, db)
        opts = env.options | {"manifest": manifest}
        bm = hybrid.lexical.build_role_bm25_bundle(db, **opts)
        dense = hybrid.dense.build_role_faiss_bundle(db, encoder=env.encoder, **opts)
    def forbidden(*args, **kwargs):
        pytest.fail("stale gate reached engine")
    monkeypatch.setattr(hybrid.lexical, "search_role_bm25", forbidden)
    monkeypatch.setattr(hybrid.dense, "search_role_faiss", forbidden)
    monkeypatch.setattr(env.encoder, "encode_queries", forbidden)
    with pytest.raises(hybrid.HybridError, match="^STALE_INDEX$"):
        run(env, hybrid, secured.authenticate(secured.sign(), "employee"), bm25=bm, dense=dense,
            expected_snapshot_id=dense.snapshot.candidate_snapshot_id, validity_date=date(2026, 9, 2))


@pytest.mark.parametrize("empty", ["lexical", "dense", "both"])
def test_valid_empty_rankings_are_allowed(dense_env, hybrid, secured, monkeypatch, empty):
    env = dense_env
    context = secured.authenticate(secured.sign(), "employee")
    options = dict(query="travel hébergement déplacement",
                   expected_snapshot_id=env.bundle.snapshot.candidate_snapshot_id, validity_date=DAY)
    left = hybrid.lexical.search_role_bm25(env.bm25, context, **options)
    right = hybrid.dense.search_role_faiss(env.bundle, context, **options)
    if empty in ("lexical", "both"):
        monkeypatch.setattr(hybrid.lexical, "search_role_bm25", lambda *a, **kw: ())
    if empty in ("dense", "both"):
        monkeypatch.setattr(hybrid.dense, "search_role_faiss", lambda *a, **kw: ())
    hits = run(env, hybrid, context)
    expected = hybrid.reciprocal_rank_fusion(
        [] if empty in ("lexical", "both") else [h.child.child_id for h in left],
        [] if empty in ("dense", "both") else [h.child.child_id for h in right],
    )
    assert [(h.child.child_id, h.rrf_score) for h in hits] == [(h.child_id, h.score) for h in expected]


@pytest.mark.parametrize("engine", ["lexical", "dense"])
def test_engine_exception_returns_zero_evidence(dense_env, hybrid, secured, monkeypatch, caplog, engine):
    module = getattr(hybrid, engine)
    name = "search_role_bm25" if engine == "lexical" else "search_role_faiss"
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic-private-detail")
    monkeypatch.setattr(module, name, broken)
    with pytest.raises(hybrid.HybridError, match="^HYBRID_RETRIEVAL_FAILED$") as exc:
        run(dense_env, hybrid, secured.authenticate(secured.sign(), "employee"))
    assert "synthetic-private-detail" not in str(exc.value) + caplog.text


def test_raw_score_scale_does_not_change_fusion(dense_env, hybrid, secured, monkeypatch):
    env = dense_env
    ctx = secured.authenticate(secured.sign(), "employee")
    before = run(env, hybrid, ctx)
    for module, name in ((hybrid.lexical, "search_role_bm25"), (hybrid.dense, "search_role_faiss")):
        fn = getattr(module, name)
        def rescale(*args, _fn=fn, **kwargs):
            return tuple(replace(h, score=h.score * 1000) for h in _fn(*args, **kwargs))
        monkeypatch.setattr(module, name, rescale)
    assert run(env, hybrid, ctx) == before
