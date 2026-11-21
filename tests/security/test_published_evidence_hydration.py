"""Authenticated role isolation at the internal published-text boundary."""

from dataclasses import replace
from types import SimpleNamespace
import traceback

import pytest

from integration.test_embedding_revision import dense_env, DAY
from unit.test_evidence_hydration import release
from security.test_auth import secured


@pytest.fixture
def boundary(release):
    from askpolicy.retrieval import evidence_hydration as module
    root, handle = release
    def hydrate(context, hits):
        return module.hydrate_published_hits(handle, context, hits, release_root=root, validity_date=DAY)
    return module, handle, hydrate


@pytest.mark.parametrize("role,query", [
    ("employee", "travel hébergement"), ("hr", "registre congés justificatifs"),
    ("finance", "frais remboursement"), ("it_security", "contrôle accès habilitations"),
])
def test_all_four_roles_receive_only_their_authorized_evidence(boundary, secured, role, query):
    module, handle, hydrate = boundary
    context = secured.authenticate(secured.sign(roles=[role]), role)
    hits = handle.search(context, query=query, validity_date=DAY)
    passages = hydrate(context, hits)
    assert len(passages) == 20
    for hit, passage in zip(hits, passages, strict=True):
        assert module._authorized(context, hit.child)
        assert passage.active_role == role and role in hit.child.document.metadata.allowed_roles
        assert passage.child_id in next(i.metadata.indexed_child_ids for i in handle.bm25.indexes
                                        if i.metadata.role == role)
        assert passage.document_id in handle.bm25.eligibility.eligible_document_ids


def test_multigrant_role_switch_does_not_reuse_hr_hits(boundary, secured):
    module, handle, hydrate = boundary
    token = secured.sign(roles=["employee", "hr"])
    hr = secured.authenticate(token, "hr")
    employee = secured.authenticate(token, "employee")
    hits = handle.search(hr, query="registre congés justificatifs", validity_date=DAY)
    private = tuple(h for h in hits if h.child.section.document_id.startswith("LEAVE-"))
    assert private and hydrate(hr, private)
    with pytest.raises(module.EvidenceHydrationError):
        hydrate(employee, private)
    # Even a forged employee role on an HR hit confers no access.
    with pytest.raises(module.EvidenceHydrationError):
        hydrate(employee, tuple(replace(h, active_role="employee") for h in private))
    own = handle.search(employee, query="registre congés justificatifs", validity_date=DAY)
    assert not any(p.document_id.startswith("LEAVE-") for p in hydrate(employee, own))
    assert hydrate(hr, private) == hydrate(secured.authenticate(token, "hr"), private)


@pytest.mark.parametrize("context_kind", ["none", "namespace", "forged", "admin"])
@pytest.mark.parametrize("empty", [False, True])
def test_unverified_and_admin_only_denied_even_without_hits(boundary, secured, monkeypatch, context_kind, empty):
    module, handle, hydrate = boundary
    hr = secured.authenticate(secured.sign(roles=["hr"]), "hr")
    hits = handle.search(hr, query="congés", validity_date=DAY)
    contexts = {
        "none": None, "namespace": SimpleNamespace(active_role="hr"),
        "forged": secured.context.AuthenticatedContext(),
        "admin": secured.verifier.verify(secured.sign(roles=[], capabilities=["ingestion_admin"])),
    }
    reads = []
    monkeypatch.setattr(module, "_verified_bytes", lambda *args: reads.append(1))
    with pytest.raises(module.EvidenceHydrationError):
        hydrate(contexts[context_kind], () if empty else hits)
    assert reads == []


@pytest.mark.parametrize("defect", [
    "snapshot", "role", "fabricated", "superseded", "unknown_grant", "empty_grant",
    "duplicate_grant", "classification", "unauthorized", "parent", "document",
])
def test_malformed_or_unauthorized_child_denied_before_text(boundary, secured, monkeypatch, caplog, defect):
    module, handle, hydrate = boundary
    context = secured.authenticate(secured.sign(roles=["employee", "hr"]), "employee")
    hits = handle.search(context, query="travel policy", validity_date=DAY)
    hit = hits[0]
    child = hit.child
    if defect == "snapshot":
        hit = replace(hit, candidate_snapshot_id="0"*64)
    elif defect == "role":
        hit = replace(hit, active_role="hr")
    elif defect == "fabricated":
        hit = replace(hit, child=replace(child, child_id="invented"))
    elif defect == "superseded":
        old = next(d for d in handle.bm25._draft.documents.values() if d.metadata.document_id == "LEAVE-001-FR-V1")
        hit = replace(hit, child=replace(child, document=old,
                                        section=replace(child.section, document_id=old.metadata.document_id)))
    elif defect == "parent":
        hit = replace(hit, child=replace(child, section=replace(child.section, parent_id="wrong")))
    elif defect == "document":
        hit = replace(hit, child=replace(child, section=replace(child.section, document_id="LEAVE-001-FR-V2")))
    else:
        values = {
            "unknown_grant": {"allowed_roles": ("unknown",)}, "empty_grant": {"allowed_roles": ()},
            "duplicate_grant": {"allowed_roles": ("employee", "employee")},
            "classification": {"classification": "UNKNOWN"}, "unauthorized": {"allowed_roles": ("hr",)},
        }[defect]
        values["title"] = "sensitive-canary"
        meta = child.document.metadata.model_copy(update=values)
        hit = replace(hit, child=replace(child, document=replace(child.document, metadata=meta)))
    reads = []
    monkeypatch.setattr(module, "_verified_bytes", lambda *args: reads.append(1))
    with pytest.raises(module.EvidenceHydrationError) as captured:
        hydrate(context, (hit,))
    assert reads == []
    formatted = "".join(traceback.format_exception(captured.type, captured.value, captured.tb))
    assert "sensitive-canary" not in formatted + caplog.text
    assert "EVIDENCE_HYDRATION_FAILED" == str(captured.value)


def test_public_api_registration_and_payload_contract_unchanged(boundary, secured):
    from askpolicy.api.app import create_app
    app = create_app(jwt_config=secured.config)
    assert {(r.path, frozenset(r.methods)) for r in app.routes} == {
        ("/auth/verify", frozenset({"GET"})), ("/retrieval/search", frozenset({"POST"})),
    }
