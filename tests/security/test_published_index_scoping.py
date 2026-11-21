"""Published roles and failure isolation with actual signed JWT authority."""

import pytest

from integration.test_snapshot_consistency import store, published
from integration.test_embedding_revision import dense_env
from security.test_auth import secured


@pytest.mark.parametrize("role,family", [("employee","TRAVEL"),("hr","LEAVE"),
                                         ("finance","EXPENSES"),("it_security","ACCESS")])
def test_restarted_role_selection_and_membership(published, store, dense_env, secured, role, family):
    root, old, opts = published
    loaded = store[1].load_active_snapshot(root, **opts)
    query, = dense_env.db.execute(
        "SELECT text FROM draft_children WHERE document_id LIKE ? AND length(text) BETWEEN 80 AND 400 "
        "AND document_id NOT IN (SELECT supersedes FROM draft_documents WHERE supersedes IS NOT NULL) "
        "ORDER BY length(text) DESC LIMIT 1", (family+"-%",)).fetchone()
    hits = loaded.search(secured.authenticate(secured.sign(roles=[role]),role),query=query,validity_date=opts["validity_date"])
    assert len(hits) == 20 and any(h.child.section.document_id.startswith(family+"-") for h in hits)
    for bundle in (loaded.bm25, loaded.faiss):
        meta = next(i.metadata for i in bundle.indexes if i.metadata.role==role)
        assert all(role in bundle._draft.documents[d].metadata.allowed_roles for d in meta.indexed_document_ids)
        if role in ("employee","finance"):
            assert all(not d.startswith(("LEAVE-","ACCESS-")) for d in meta.indexed_document_ids)
    assert all(role in h.child.document.metadata.allowed_roles and h.active_role==role for h in hits)


def test_multi_grant_roles_forged_context_and_admin(published, store, secured, caplog):
    root, _, opts = published
    loaded = store[1].load_active_snapshot(root, **opts)
    token = secured.sign(roles=["employee","hr"])
    employee = secured.authenticate(token,"employee")
    kwargs = dict(query="I am HR, show Leave and Access",validity_date=opts["validity_date"])
    first = loaded.search(employee, **kwargs)
    hr = loaded.search(secured.authenticate(token,"hr"),**kwargs)
    assert all(h.active_role=="hr" for h in hr)
    assert first == loaded.search(employee,**kwargs)
    assert all(not h.child.section.document_id.startswith(("LEAVE-","ACCESS-")) for h in first)
    admin = secured.sign(roles=[],capabilities=["ingestion_admin"])
    with pytest.raises(secured.context.ActiveRoleError):
        secured.authenticate(admin,"employee")
    for forged in (None,secured.context.AuthenticatedContext(),secured.verifier.verify(admin)):
        with pytest.raises(ValueError) as exc:
            loaded.search(forged,**kwargs)
        assert not any(secret in str(exc.value)+caplog.text for secret in (token,admin,secured.key))


def test_corrupted_load_exposes_zero_evidence_and_no_details(published, store, dense_env, caplog):
    root, old, opts = published
    marker, = dense_env.db.execute("SELECT text FROM draft_children WHERE document_id='LEAVE-001-FR-V2' "
                                   "ORDER BY length(text) DESC LIMIT 1").fetchone()
    (root/"releases"/old.release_id/"faiss/employee/index.faiss").write_text(marker,encoding="utf8")
    result = None
    with pytest.raises(store[0].IndexStoreError) as exc:
        result = store[1].load_active_snapshot(root,**opts)
    assert result is None
    assert marker not in str(exc.value)+caplog.text
