"""JWT, publication and source isolation for experimental reranking."""

from copy import copy
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from unit.test_offline_reranking import session
from unit.test_evidence_hydration import release
from unit.test_cross_encoder_checkpoint import local
from integration.test_embedding_revision import dense_env, DAY
from security.test_auth import secured


@pytest.mark.parametrize("role,query",[
    ("employee","travel hébergement"),("hr","registre congés"),("finance","frais remboursement"),
    ("it_security","contrôle accès"),
])
def test_four_signed_roles_preserve_exact_authorized_set(session,secured,role,query):
    context = secured.authenticate(secured.sign(roles=[role]),role)
    hits = session.handle.search(context,query=query,validity_date=DAY)
    result = session.run(hits,context=context,query=query)
    assert len(result.hits) == 20 and {h.original for h in result.hits} == set(hits)
    assert all(h.active_role == role and role in h.child.document.metadata.allowed_roles for h in result.hits)
    assert sum(session.local.model.calls) == 20


def test_hr_evidence_cannot_follow_multigrant_employee_switch(session,secured):
    token = secured.sign(roles=["employee","hr"])
    hr, employee = (secured.authenticate(token,role) for role in ("hr","employee"))
    hits = session.handle.search(hr,query="registre congés justificatifs",validity_date=DAY)
    restricted = tuple(h for h in hits if h.child.section.document_id.startswith("LEAVE-"))
    assert restricted
    with pytest.raises(session.module.OfflineRerankError):
        session.run(restricted,context=employee)
    with pytest.raises(session.module.OfflineRerankError):
        session.run(tuple(replace(h,active_role="employee") for h in restricted),context=employee)
    assert not session.local.model.calls
    assert session.run(restricted,context=hr).hits


@pytest.mark.parametrize("kind",["none","lookalike","copied","admin"])
def test_unverified_contexts_and_admin_never_reach_forward(session,secured,kind):
    options = dict(none=None,lookalike=SimpleNamespace(active_role="hr"),copied=copy(session.context),
        admin=secured.verifier.verify(secured.sign(roles=[],capabilities=["ingestion_admin"])))
    with pytest.raises(session.module.OfflineRerankError):
        session.run(context=options[kind])
    assert not session.local.model.calls


def test_ungranted_and_expired_signed_tokens_rejected_before_experiment(session,secured):
    with pytest.raises(secured.context.ActiveRoleError):
        secured.authenticate(secured.sign(roles=["employee"]),"hr")
    with pytest.raises(secured.tokens.AuthenticationError):
        secured.authenticate(secured.sign(roles=["hr"],iat=secured.now-600,exp=secured.now-300),"hr")
    assert not session.local.model.calls


@pytest.mark.parametrize("defect",[
    "injected_id","snapshot","role","parent","revoked","expired","superseded","classification","grants",
    "duplicate","wrong_release","wrong_date","sqlite","manifest",
])
def test_security_failures_prevent_any_candidate_forward(session,defect):
    hit = session.hits[0]
    changes, hits = {}, session.hits
    if defect == "wrong_release":
        changes["handle"] = replace(session.handle,release_id="0"*64)
    elif defect == "wrong_date":
        changes["validity_date"] = date(2026,9,2)
    elif defect in ("sqlite","manifest"):
        directory = session.root / "releases" / session.handle.release_id
        path = directory / ("sqlite/corpus.sqlite" if defect == "sqlite" else "manifest.json")
        # The shared fixture is restored even if the rejection assertion fails.
        raw = path.read_bytes()
        try:
            path.write_bytes(b"corrupt-publication")
            with pytest.raises(session.module.OfflineRerankError):
                session.run()
            assert not session.local.model.calls
        finally:
            path.write_bytes(raw)
        return
    elif defect == "duplicate":
        hits = hits[:1]*2
    else:
        child = hit.child
        if defect == "injected_id":
            hit = replace(hit,child=replace(child,child_id="fabricated"))
        elif defect == "snapshot":
            hit = replace(hit,candidate_snapshot_id="0"*64)
        elif defect == "role":
            hit = replace(hit,active_role="employee")
        elif defect == "parent":
            hit = replace(hit,child=replace(child,section=replace(child.section,parent_id="wrong-parent")))
        elif defect == "superseded":
            old = session.handle.bm25._draft.documents["LEAVE-001-FR-V1"]
            hit = replace(hit,child=replace(child,document=old,
                                           section=replace(child.section,document_id=old.metadata.document_id)))
        else:
            update = {"revoked":{"approval_status":"not_approved"},"expired":{"expires_at":DAY},
                      "classification":{"classification":"UNKNOWN"},
                      "grants":{"allowed_roles":("unknown",)}}[defect]
            meta = child.document.metadata.model_copy(update=update)
            hit = replace(hit,child=replace(child,document=replace(child.document,metadata=meta)))
        hits = (hit,)
    with pytest.raises(session.module.OfflineRerankError):
        session.run(hits,**changes)
    assert not session.local.model.calls


def test_authorization_rechecked_after_scoring_not_inferred_from_logits(session,monkeypatch):
    original = session.encoder.score_pairs
    def revoke_after_scoring(pairs):
        result = original(pairs)
        monkeypatch.setattr(session.module,"can_read_document",lambda *args,**kwargs:False)
        return result
    monkeypatch.setattr(session.encoder,"score_pairs",revoke_after_scoring)
    with pytest.raises(session.module.OfflineRerankError):
        session.run()
    assert sum(session.local.model.calls) == 20
