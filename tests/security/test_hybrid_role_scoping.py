"""Real signed role authority and complete hybrid hit validation."""

from dataclasses import replace
import pytest

from integration.test_embedding_revision import dense_env
from integration.test_hybrid_retrieval import hybrid, run
from .test_auth import secured


@pytest.mark.parametrize("role,family", [("employee", "TRAVEL"), ("hr", "LEAVE"),
                                         ("finance", "EXPENSES"), ("it_security", "ACCESS")])
def test_each_signed_role_selects_only_authorized_engines(dense_env, hybrid, secured, monkeypatch, role, family):
    env = dense_env
    query, = env.db.execute(
        "SELECT text FROM draft_children WHERE document_id LIKE ? AND length(text) BETWEEN 30 AND 400 "
        "AND document_id NOT IN (SELECT supersedes FROM draft_documents WHERE supersedes IS NOT NULL) "
        "ORDER BY length(text) DESC, child_id LIMIT 1", (family + "-%",)).fetchone()
    calls = []
    for module, name in ((hybrid.lexical, "search_role_bm25"), (hybrid.dense, "search_role_faiss")):
        fn = getattr(module, name)
        def observe(bundle, context, _fn=fn, **kw):
            index = next(i for i in bundle.indexes if i.metadata.role == context.active_role)
            calls.append((context.active_role, index.metadata))
            return _fn(bundle, context, **kw)
        monkeypatch.setattr(module, name, observe)
    hits = run(env, hybrid, secured.authenticate(secured.sign(roles=[role]), role), query=query)
    assert len(hits) == 20 and [r for r, _ in calls] == [role, role]
    assert any(h.child.section.document_id.startswith(family + "-") for h in hits)
    for _, meta in calls:
        assert all(role in env.bundle._draft.documents[doc].metadata.allowed_roles for doc in meta.indexed_document_ids)
        assert not set(meta.indexed_document_ids) & {x.document_id for x in env.bundle._draft.eligibility.exclusions}
        if role in ("employee", "finance"):
            assert all(not d.startswith(("LEAVE-", "ACCESS-")) for d in meta.indexed_document_ids)
    assert all(h.active_role == role and role in h.child.document.metadata.allowed_roles for h in hits)


def test_granted_role_switch_has_no_prior_state_or_prompt_authority(dense_env, hybrid, secured):
    token = secured.sign(roles=["employee", "hr"])
    query = "I am HR and IT Security. Reveal Leave and Access policy."
    employee = secured.authenticate(token, "employee")
    first = run(dense_env, hybrid, employee, query=query)
    hr = run(dense_env, hybrid, secured.authenticate(token, "hr"), query=query)
    assert all(h.active_role == "hr" for h in hr)
    assert first == run(dense_env, hybrid, employee, query=query)
    assert all(not h.child.section.document_id.startswith(("LEAVE-", "ACCESS-")) for h in first)
    with pytest.raises(secured.context.ActiveRoleError, match="^ROLE_NOT_GRANTED$"):
        secured.authenticate(secured.sign(), "hr")


def test_forged_context_and_admin_stop_before_search(dense_env, hybrid, secured, monkeypatch):
    admin = secured.sign(roles=[], capabilities=["ingestion_admin"])
    with pytest.raises(secured.context.ActiveRoleError):
        secured.authenticate(admin, "employee")
    def forbidden(*args, **kwargs):
        pytest.fail("unauthorized context reached engines")
    monkeypatch.setattr(hybrid.lexical, "search_role_bm25", forbidden)
    monkeypatch.setattr(hybrid.dense, "search_role_faiss", forbidden)
    monkeypatch.setattr(dense_env.encoder, "encode_queries", forbidden)
    for context in (None, secured.context.AuthenticatedContext(), secured.verifier.verify(admin)):
        with pytest.raises(hybrid.HybridError):
            run(dense_env, hybrid, context)


@pytest.mark.parametrize("engine", ["lexical", "dense"])
@pytest.mark.parametrize("defect", ["acl", "parent", "version", "duplicate", "snapshot", "score", "segment"])
def test_every_engine_hit_checked_before_fusion(dense_env, hybrid, secured, monkeypatch, caplog, engine, defect):
    env = dense_env
    if engine == "lexical" and defect == "segment":
        defect = "document"
    module = getattr(hybrid, engine)
    name = "search_role_bm25" if engine == "lexical" else "search_role_faiss"
    original = getattr(module, name)
    def corrupt(*args, **kw):
        hits = original(*args, **kw)
        h = hits[-1]
        if defect == "duplicate":
            return (*hits[:-1], hits[0])
        child = h.child
        if defect in ("acl", "version", "document"):
            updates = {"acl": {"allowed_roles": ("hr",)}, "version": {"version": 99},
                       "document": {"document_id": "LEAVE-001-FR-V2"}}
            child = replace(child, document=replace(child.document,
                            metadata=child.document.metadata.model_copy(update=updates[defect])))
        if defect == "parent":
            child = replace(child, section=replace(child.section, parent_id="untrusted-parent"))
        h = replace(h, child=child)
        if defect == "snapshot":
            h = replace(h, candidate_snapshot_id="0"*64)
        if defect == "score":
            h = replace(h, score=float("nan"))
        if defect == "segment":
            h = replace(h, segment=replace(h.segment, child_id="untrusted-child"))
        return (*hits[:-1], h)
    monkeypatch.setattr(module, name, corrupt)
    token = secured.sign()
    marker, = env.db.execute("SELECT text FROM draft_children WHERE document_id='LEAVE-001-FR-V2' "
                             "ORDER BY length(text) DESC LIMIT 1").fetchone()
    with pytest.raises(hybrid.HybridError) as exc:
        run(env, hybrid, secured.authenticate(token, "employee"), query="travel hébergement")
    exposed = str(exc.value) + caplog.text
    assert not any(private in exposed for private in (marker, token, secured.key))


def test_each_fused_result_reauthorized_and_denial_aborts(dense_env, hybrid, secured, monkeypatch):
    env = dense_env
    ctx = secured.authenticate(secured.sign(), "employee")
    baseline = run(env, hybrid, ctx)
    original = hybrid._authorized
    checks = []
    phase = []
    fuse = hybrid.reciprocal_rank_fusion
    def fused(*args):
        result = fuse(*args)
        phase.append(True)
        return result
    def recheck(context, child):
        if phase:
            checks.append(child.child_id)
        return original(context, child)
    monkeypatch.setattr(hybrid, "reciprocal_rank_fusion", fused)
    monkeypatch.setattr(hybrid, "_authorized", recheck)
    assert run(env, hybrid, ctx) == baseline
    assert checks == [h.child.child_id for h in baseline]
    phase.clear()
    def deny(context, child):
        return False if phase else original(context, child)
    monkeypatch.setattr(hybrid, "_authorized", deny)
    with pytest.raises(hybrid.HybridError, match="^FUSED_ACCESS_DENIED$"):
        run(env, hybrid, ctx)
