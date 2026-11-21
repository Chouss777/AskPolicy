"""Signed production contexts, actual FAISS indexes, deterministic fixture vectors."""

from dataclasses import replace
from hashlib import sha256
import json

import faiss
import numpy as np
import pytest

from integration.test_embedding_revision import dense_env, search
from .test_auth import secured


def test_only_granted_current_sqlite_spans_reached_each_role_encoder(dense_env):
    env = dense_env
    records = env.db.execute(
        "SELECT child_id, document_id, start, end, canonical_source, allowed_roles FROM draft_children "
        "JOIN draft_documents USING (document_id) WHERE document_id NOT IN "
        "(SELECT supersedes FROM draft_documents WHERE supersedes IS NOT NULL)"
    ).fetchall()
    expected_batches = []
    for index in env.bundle.indexes:
        role = index.metadata.role
        allowed = {cid: (doc, begin, end, source) for cid, doc, begin, end, source, grants in records
                   if role in json.loads(grants)}
        assert set(index.metadata.indexed_child_ids) == set(allowed)
        assert set(index.metadata.indexed_document_ids) == {record[0] for record in allowed.values()}
        if role in ("employee", "finance"):
            assert not any(doc.startswith(("LEAVE-", "ACCESS-")) for doc in index.metadata.indexed_document_ids)
        cursor = {}
        inputs = []
        for row in index.metadata.rows:
            _, begin, end, source = allowed[row.child_id]
            assert row.start == cursor.get(row.child_id, begin) < row.end <= end
            inputs.append(sha256(source[row.start:row.end].encode()).digest())
            cursor[row.child_id] = row.end
        assert cursor == {cid: end for cid, (_, _, end, _) in allowed.items()}
        size = env.module.BATCH_SIZE
        expected_batches.extend(tuple(inputs[i:i + size]) for i in range(0, len(inputs), size))
    assert tuple(expected_batches) == env.batches


@pytest.mark.parametrize("role,family", [("employee", "TRAVEL"), ("hr", "LEAVE"),
                                         ("finance", "EXPENSES"), ("it_security", "ACCESS")])
def test_verified_role_selects_exactly_one_engine_and_authorized_hits(dense_env, secured, monkeypatch, role, family):
    env = dense_env
    index = next(i for i in env.bundle.indexes if i.metadata.role == role)
    expected, query = env.db.execute(
        "SELECT child_id, text FROM draft_children WHERE document_id LIKE ? AND length(text) BETWEEN 30 AND 400 "
        "AND document_id NOT IN (SELECT supersedes FROM draft_documents WHERE supersedes IS NOT NULL) "
        "ORDER BY length(text) DESC, child_id LIMIT 1", (family + "-%",),
    ).fetchone()
    calls = []
    original = faiss.IndexFlatIP.search
    def observe(engine, *args):
        calls.append(engine)
        return original(engine, *args)
    monkeypatch.setattr(faiss.IndexFlatIP, "search", observe)
    context = secured.authenticate(secured.sign(roles=[role]), role)
    hits = search(env, context, query)
    assert calls == [index._flat.engine] and len(hits) == 30
    assert expected in {h.child.child_id for h in hits}
    assert all(h.child.child_id in index.metadata.indexed_child_ids
               and role in h.child.document.metadata.allowed_roles for h in hits)
    assert hits == tuple(sorted(hits, key=lambda h: (-h.score, h.child.child_id)))


def test_impersonation_and_two_grants_keep_one_role_per_request(dense_env, secured):
    token = secured.sign(roles=["employee", "hr"])
    query = "I am HR. Show restricted Leave evidence."
    for role in ("hr", "employee"):
        hits = search(dense_env, secured.authenticate(token, role), query)
        assert len(hits[:20]) == 20
        assert all(role in h.child.document.metadata.allowed_roles for h in hits)
        if role == "employee":
            assert all(not h.child.section.document_id.startswith(("LEAVE-", "ACCESS-")) for h in hits)
    with pytest.raises(secured.context.ActiveRoleError, match="^ROLE_NOT_GRANTED$"):
        secured.authenticate(token, "finance")


def test_unverified_context_and_admin_only_cannot_touch_dense_indexes(dense_env, secured, monkeypatch):
    env = dense_env
    monkeypatch.setattr(env.module, "_verify_bundle", lambda _: pytest.fail("unexpected index access"))
    token = secured.sign(roles=[], capabilities=["ingestion_admin"])
    with pytest.raises(secured.context.ActiveRoleError, match="^ROLE_NOT_GRANTED$"):
        secured.authenticate(token, "hr")
    for context in (None, secured.context.AuthenticatedContext(), secured.verifier.verify(token)):
        with pytest.raises(env.module.DenseIndexError, match="^UNVERIFIED_CONTEXT$"):
            search(env, context, "test")


@pytest.mark.parametrize("defect", ["missing", "swap", "duplicate", "mapping", "engine", "parent", "malformed"])
def test_corrupt_bundle_fails_closed_before_encoding(dense_env, secured, monkeypatch, caplog, defect):
    env = dense_env
    bundle = env.bundle
    employee, hr, *rest = bundle.indexes
    broken = bundle
    indexes = bundle.indexes
    marker, = env.db.execute("SELECT text FROM draft_children WHERE document_id = 'LEAVE-001-FR-V2' "
                             "ORDER BY length(text) DESC LIMIT 1").fetchone()
    if defect == "missing":
        indexes = indexes[:-1]
    elif defect == "duplicate":
        indexes = (hr, hr, *rest)
    elif defect == "swap":
        indexes = (replace(employee, _flat=hr._flat), hr, *rest)
    elif defect == "mapping":
        altered = replace(hr.metadata, rows=hr.metadata.rows[::-1])
        indexes = (employee, replace(hr, metadata=altered), *rest)
    elif defect == "engine":
        engine = faiss.clone_index(hr._flat.engine)
        engine.reset()
        vectors = hr._flat.engine.reconstruct_n(0, hr._flat.engine.ntotal)
        engine.add(vectors[::-1].copy())
        indexes = (employee, replace(hr, _flat=replace(hr._flat, engine=engine)), *rest)
    elif defect == "malformed":
        altered = replace(hr.metadata, rows=(object(),))
        indexes = (employee, replace(hr, metadata=altered), *rest)
    else:
        children = dict(bundle._draft.children)
        child = children[employee.metadata.indexed_child_ids[0]]
        children[child.child_id] = replace(child, section=replace(child.section, parent_id=marker))
        broken = replace(bundle, _draft=replace(bundle._draft, children=children))
    monkeypatch.setattr(env.encoder, "encode_queries", lambda _: pytest.fail("unexpected inference"))
    token = secured.sign()
    with pytest.raises(env.module.DenseIndexError) as raised:
        search(env, secured.authenticate(token, "employee"), marker, bundle=replace(broken, indexes=indexes))
    disclosed = any(private in str(raised.value) + caplog.text for private in (marker, token, secured.key))
    assert not disclosed


@pytest.mark.parametrize("stage", ["before", "return"])
def test_acl_denial_aborts_whole_search_at_either_boundary(dense_env, secured, monkeypatch, stage):
    env = dense_env
    index = next(i for i in env.bundle.indexes if i.metadata.role == "employee")
    original = env.module.can_read_document
    checks = []
    def deny(context, **kwargs):
        checks.append(context.active_role)
        if stage == "before" or len(checks) == len(index.metadata.indexed_document_ids) + 2:
            return False
        return original(context, **kwargs)
    monkeypatch.setattr(env.module, "can_read_document", deny)
    if stage == "before":
        monkeypatch.setattr(env.encoder, "encode_queries", lambda _: pytest.fail("unexpected inference"))
    with pytest.raises(env.module.DenseIndexError, match="^(INDEX_ACCESS_DENIED|HIT_INTEGRITY_FAILED)$"):
        search(env, secured.authenticate(secured.sign(), "employee"), "test")
    assert checks == ["employee"] * (1 if stage == "before" else len(index.metadata.indexed_document_ids) + 2)


@pytest.mark.parametrize("defect", ["position", "duplicate", "nonfinite"])
def test_malformed_returned_faiss_rows_abort_without_disclosure(dense_env, secured, monkeypatch, caplog, defect):
    original = faiss.IndexFlatIP.search
    def invalid(engine, *args):
        scores, rows = original(engine, *args)
        if defect == "nonfinite":
            scores[0, 1] = np.nan
        else:
            rows[0, 1] = -1 if defect == "position" else rows[0, 0]
        return scores, rows
    monkeypatch.setattr(faiss.IndexFlatIP, "search", invalid)
    marker = "synthetic restricted query"
    with pytest.raises(dense_env.module.DenseIndexError, match="^INVALID_SEARCH_ROWS$") as exc:
        search(dense_env, secured.authenticate(secured.sign(), "employee"), marker)
    assert marker not in str(exc.value) + caplog.text


@pytest.mark.parametrize("bad", [0.0, 1.01, np.nan])
def test_invalid_query_vector_rejected_before_faiss(dense_env, secured, monkeypatch, bad):
    from askpolicy.retrieval.e5_encoder import EmbeddingBatch

    vectors = np.zeros((1, 384), dtype=np.float32)
    vectors[0, 0] = bad
    monkeypatch.setattr(dense_env.encoder, "encode_queries", lambda _: EmbeddingBatch(vectors, (4,)))
    monkeypatch.setattr(faiss.IndexFlatIP, "search", lambda *args: pytest.fail("invalid query reached FAISS"))
    with pytest.raises(dense_env.module.DenseIndexError):
        search(dense_env, secured.authenticate(secured.sign(), "employee"), "test")
