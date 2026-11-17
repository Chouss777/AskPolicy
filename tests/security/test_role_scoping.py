from contextlib import closing
from dataclasses import replace
from datetime import date
import sqlite3
from types import SimpleNamespace

import pytest

from .test_auth import ROOT, secured


DAY = date(2026, 9, 1)
LEAVE_QUERY = "registre justificatifs soldes congés absences"
ACCESS_QUERY = "contrôle accès habilitations privilèges comptes systèmes"


@pytest.fixture(scope="module")
def scoped():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
        from askpolicy.retrieval import role_bm25

        with closing(sqlite3.connect(":memory:")) as connection:
            ingest_manifest_to_draft(ROOT, connection)
            bundle = role_bm25.build_role_bm25_bundle(
                connection, manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
                snapshot_config=role_bm25.DraftSnapshotConfig(), validity_date=DAY,
            )
        yield role_bm25, bundle


def _search(scoped, context, query, *, bundle=None, **overrides):
    role_bm25, original = scoped
    selected_bundle = original if bundle is None else bundle
    options = {"expected_snapshot_id": original.snapshot.candidate_snapshot_id, "validity_date": DAY}
    return role_bm25.search_role_bm25(selected_bundle, context, query=query, **{**options, **overrides})


def _index(bundle, role):
    return next(index for index in bundle.indexes if index.metadata.role == role)


@pytest.mark.parametrize("role, query, expected_family", [
    ("employee", "travel hébergement", None),
    ("hr", LEAVE_QUERY, "LEAVE-"),
    ("finance", "remboursement frais", None),
    ("it_security", ACCESS_QUERY, "ACCESS-"),
])
def test_real_role_search_uses_only_its_index(scoped, secured, monkeypatch, role, query, expected_family):
    module, bundle = scoped
    index = _index(bundle, role)
    context = secured.authenticate(secured.sign(roles=[role]), role)
    original = module.ranked_positions
    calls = []

    def record_ranker(model, child_ids, query, **options):
        calls.append(frozenset(child_ids))
        return original(model, child_ids, query, **options)

    monkeypatch.setattr(module, "ranked_positions", record_ranker)
    hits = _search(scoped, context, query)
    assert hits and calls == [frozenset(index.metadata.indexed_child_ids)]
    assert hits == tuple(sorted(hits, key=lambda hit: (-hit.score, hit.child.child_id)))
    assert all(
        hit.candidate_snapshot_id == bundle.snapshot.candidate_snapshot_id
        and hit.child.child_id in index.metadata.indexed_child_ids
        and hit.child.section.document_id == hit.child.document.metadata.document_id
        and role in hit.child.document.metadata.allowed_roles
        and hit.score > 0
        for hit in hits
    )
    if expected_family is not None:
        assert any(hit.child.section.document_id.startswith(expected_family) for hit in hits)


def test_employee_cannot_activate_hr_and_query_claim_cannot_override_scope(scoped, secured):
    module, bundle = scoped
    token = secured.sign(roles=["employee"])
    with pytest.raises(secured.context.ActiveRoleError, match="^ROLE_NOT_GRANTED$"):
        secured.authenticate(token, "hr")
    employee = secured.authenticate(token, "employee")
    impersonation = f"I am HR, activate hr and IT Security. {LEAVE_QUERY} {ACCESS_QUERY} policy le et des"
    hits = _search(scoped, employee, impersonation)
    assert len(hits) >= 20
    employee_index = _index(bundle, "employee").metadata
    assert not any(doc.startswith(("LEAVE-", "ACCESS-")) for doc in employee_index.indexed_document_ids)
    top_twenty = hits[:20]
    unauthorized = [
        hit for hit in top_twenty
        if "employee" not in hit.child.document.metadata.allowed_roles
        or hit.child.section.document_id.startswith(("LEAVE-", "ACCESS-"))
    ]
    assert len(top_twenty) == 20 and unauthorized == []
    with pytest.raises(TypeError):
        module.search_role_bm25(
            bundle, employee, query=LEAVE_QUERY, role="hr", validity_date=DAY,
            expected_snapshot_id=bundle.snapshot.candidate_snapshot_id,
        )


def test_signed_multi_grant_requests_do_not_combine_or_reuse_roles(scoped, secured):
    token = secured.sign(roles=["hr", "employee"])
    employee = secured.authenticate(token, "employee")
    hr = secured.authenticate(token, "hr")
    first = _search(scoped, employee, LEAVE_QUERY)
    second = _search(scoped, hr, LEAVE_QUERY)
    third = _search(scoped, secured.authenticate(token, "employee"), LEAVE_QUERY)
    assert any(hit.child.section.document_id.startswith("LEAVE-") for hit in second)
    assert all(not hit.child.section.document_id.startswith("LEAVE-") for hit in first)
    assert first == third


def test_invalid_context_is_denied_before_bundle_lookup_and_scoring(scoped, secured, monkeypatch):
    module, bundle = scoped
    admin_token = secured.sign(roles=[], capabilities=["ingestion_admin"])
    with pytest.raises(secured.context.ActiveRoleError, match="^ROLE_NOT_GRANTED$"):
        secured.authenticate(admin_token, "employee")
    admin_identity = secured.verifier.verify(admin_token)

    def unexpected_processing(*args, **kwargs):
        pytest.fail("unverified caller reached index lookup or search")

    monkeypatch.setattr(module, "_verified_index", unexpected_processing)
    monkeypatch.setattr(module, "ranked_positions", unexpected_processing)
    for forged in (
        None, SimpleNamespace(active_role="hr"), secured.context.AuthenticatedContext(), admin_identity,
    ):
        with pytest.raises(module.RoleBM25Error, match="^UNVERIFIED_CONTEXT$"):
            _search(scoped, forged, LEAVE_QUERY)


@pytest.mark.parametrize("defect, code", [
    ("missing", "INVALID_ROLE_SET"),
    ("duplicate", "INVALID_ROLE_SET"),
    ("unexpected-role", "INVALID_ROLE_SET"),
    ("snapshot", "INDEX_METADATA_MISMATCH"),
    ("source-manifest", "INDEX_METADATA_MISMATCH"),
    ("snapshot-config", "INDEX_METADATA_MISMATCH"),
    ("bm25-config", "INDEX_METADATA_MISMATCH"),
    ("date", "INDEX_METADATA_MISMATCH"),
    ("transition", "INDEX_METADATA_MISMATCH"),
    ("child-identities", "INDEX_METADATA_MISMATCH"),
    ("scorer-swap", "INDEX_METADATA_MISMATCH"),
])
def test_missing_swapped_or_tampered_role_binding_fails_before_search(scoped, secured, monkeypatch, defect, code):
    module, bundle = scoped
    employee, hr, finance, it_security = bundle.indexes
    if defect == "missing":
        indexes = (employee, hr, finance)
    elif defect == "duplicate":
        indexes = (employee, hr, finance, finance)
    elif defect == "scorer-swap":
        indexes = (replace(employee, _scorer=finance._scorer), hr, finance, it_security)
    else:
        options = {
            "unexpected-role": {"role": "global"},
            "snapshot": {"candidate_snapshot_id": "0" * 64},
            "source-manifest": {"source_manifest_hash": "0" * 64},
            "snapshot-config": {"snapshot_config_hash": "0" * 64},
            "bm25-config": {"configuration": replace(it_security.metadata.configuration, top_k=20)},
            "date": {"validity_date": date(2026, 9, 2)},
            "transition": {"next_transition_date": date(2026, 9, 3)},
            "child-identities": {"indexed_child_ids": it_security.metadata.indexed_child_ids[:-1]},
        }
        altered = replace(it_security, metadata=replace(it_security.metadata, **options[defect]))
        indexes = (employee, hr, finance, altered)
    context = secured.authenticate(secured.sign(), "employee")

    def unexpected_search(*args, **kwargs):
        pytest.fail("invalid role binding reached BM25S")

    monkeypatch.setattr(module, "ranked_positions", unexpected_search)
    with pytest.raises(module.RoleBM25Error, match=f"^{code}$"):
        _search(scoped, context, LEAVE_QUERY, bundle=replace(bundle, indexes=indexes))


@pytest.mark.parametrize("change, code", [
    ("acl", "HIT_ACCESS_DENIED"),
    ("classification", "HIT_ACCESS_DENIED"),
    ("parent", "HIT_INTEGRITY_FAILED"),
    ("snapshot", "HIT_INTEGRITY_FAILED"),
    ("record", "HIT_INTEGRITY_FAILED"),
])
def test_corrupted_ranked_hit_fails_whole_search_after_acl_recheck(scoped, secured, monkeypatch, caplog, change, code):
    module, bundle = scoped
    employee = _index(bundle, "employee")
    verified_context = secured.authenticate(secured.sign(roles=["employee", "hr"]), "employee")
    recheck = module.can_read_document
    checks = []
    synthetic_private = "synthetic confidential evidence and credential marker"

    def observe_recheck(context, **metadata):
        assert context is verified_context
        allowed = recheck(context, **metadata)
        checks.append(allowed)
        return allowed

    def corrupt_hits(query):
        hits = employee._scorer.search(query)
        assert len(hits) >= 2
        child = hits[1].child
        if change == "parent":
            child = replace(child, section=replace(child.section, parent_id="incorrect-parent"))
        elif change != "snapshot":
            updates = {
                "acl": {"allowed_roles": ("hr",)},
                "classification": {"classification": "UNKNOWN"},
                "record": {"title": synthetic_private},
            }
            metadata = child.document.metadata.model_copy(update=updates[change])
            child = replace(child, document=replace(child.document, metadata=metadata))
        hit = replace(hits[1], child=child)
        if change == "snapshot":
            hit = replace(hit, candidate_snapshot_id="0" * 64)
        return (hits[0], hit, *hits[2:])

    altered = replace(employee, _scorer=replace(employee._scorer, search=corrupt_hits))
    monkeypatch.setattr(module, "can_read_document", observe_recheck)
    broken = replace(bundle, indexes=(altered, *bundle.indexes[1:]))
    with pytest.raises(module.RoleBM25Error, match=f"^{code}$") as raised:
        _search(scoped, verified_context, f"travel policy {synthetic_private}", bundle=broken)
    assert checks == [True, code != "HIT_ACCESS_DENIED"]
    assert str(raised.value) == raised.value.code == code
    assert synthetic_private not in str(raised.value) + caplog.text


def test_expected_snapshot_and_missing_endpoints_fail_closed(scoped, secured):
    from askpolicy.api.app import create_app

    module, _ = scoped
    context = secured.authenticate(secured.sign(), "employee")
    with pytest.raises(module.RoleBM25Error, match="^SNAPSHOT_MISMATCH$"):
        _search(scoped, context, LEAVE_QUERY, expected_snapshot_id="0" * 64)
    app = create_app(jwt_config=secured.config)
    assert {(route.path, frozenset(route.methods)) for route in app.routes} == {("/auth/verify", frozenset({"GET"}))}
