"""Deterministic boundary checks with fixture vectors, no model checkpoint."""

from contextlib import closing
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.test_embedding_revision import dense_env, DAY
from security.test_auth import secured


@pytest.fixture(scope="module")
def release(dense_env, tmp_path_factory):
    from askpolicy.storage import publication
    root = tmp_path_factory.mktemp("hydration-unit")
    opts = dense_env.options | {"encoder": dense_env.encoder}
    return root, publication.publish_snapshot(dense_env.db, root, **opts)


@pytest.fixture
def case(release, secured):
    from askpolicy.retrieval import evidence_hydration as module
    root, handle = release
    context = secured.authenticate(secured.sign(roles=["employee", "hr"]), "hr")
    hits = handle.search(context, query="congés registre hébergement العمل عن بعد", validity_date=DAY)
    assert len(hits) == 20
    def hydrate(selected=hits, **changes):
        return module.hydrate_published_hits(
            changes.get("handle", handle), changes.get("context", context), selected,
            release_root=changes.get("release_root", root), validity_date=changes.get("validity_date", DAY),
        )
    return SimpleNamespace(root=root, handle=handle, context=context, hits=hits, hydrate=hydrate, module=module)


@pytest.mark.parametrize("count", [0, 1, 20])
def test_bounded_tuple_and_original_order(case, count):
    hits = tuple(reversed(case.hits[:count]))
    passages = case.hydrate(hits)
    assert type(passages) is tuple and len(passages) == count
    assert [p.child_id for p in passages] == [h.child.child_id for h in hits]
    assert all(p.release_id == case.handle.release_id and p.active_role == "hr"
               and p.snapshot_id == h.candidate_snapshot_id for p, h in zip(passages, hits, strict=True))


def test_immutable_and_private_repr(case):
    passage, = case.hydrate(case.hits[:1])
    with pytest.raises(FrozenInstanceError):
        passage.text = "changed"
    assert passage.text and passage.text not in repr(passage)
    assert not hasattr(passage, "__dict__")


@pytest.mark.parametrize("language,query", [("fr", "congés registre"), ("ar", "العمل عن بعد")])
def test_original_unicode_and_half_open_character_offsets(case, language, query):
    hits = case.handle.search(case.context, query=query, validity_date=DAY)
    selected = tuple(h for h in hits if h.child.document.metadata.language == language)
    assert selected
    passages = case.hydrate(selected)
    path = case.root / "releases" / case.handle.release_id / "sqlite/corpus.sqlite"
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)) as db:
        for passage in passages:
            source, = db.execute("SELECT canonical_source FROM draft_documents WHERE document_id = ?",
                                 (passage.document_id,)).fetchone()
            assert source[passage.start:passage.end] == passage.text
            assert passage.end - passage.start == len(passage.text)
    assert any(len(p.text.encode("utf-8")) > len(p.text) for p in passages)


@pytest.mark.parametrize("defect", [
    "too_many", "duplicate", "list", "ids", "snapshot", "role", "child", "start", "metadata",
    "rank", "score", "handle", "state", "release", "manifest_hash", "date", "date_type", "bundles",
])
def test_invalid_requests_rejected_before_artifact_read(case, monkeypatch, defect):
    hits, changes = case.hits, {}
    hit = hits[0]
    if defect == "too_many":
        hits = hits + hits[:1]
    elif defect == "duplicate":
        hits = hits[:1] * 2
    elif defect == "list":
        hits = list(hits)
    elif defect == "ids":
        hits = (hit.child.child_id,)
    elif defect in {"snapshot", "role", "child", "start", "metadata", "rank", "score"}:
        if defect == "snapshot":
            hit = replace(hit, candidate_snapshot_id="0"*64)
        elif defect == "role":
            hit = replace(hit, active_role="employee")
        elif defect == "child":
            hit = replace(hit, child=replace(hit.child, child_id="fabricated"))
        elif defect == "start":
            hit = replace(hit, child=replace(hit.child, start=hit.child.start+1))
        elif defect == "metadata":
            meta = hit.child.document.metadata.model_copy(update={"version": 99})
            hit = replace(hit, child=replace(hit.child, document=replace(hit.child.document, metadata=meta)))
        elif defect == "rank":
            hit = replace(hit, bm25_rank=True)
        else:
            hit = replace(hit, rrf_score=float("nan"))
        hits = (hit,)
    elif defect == "handle":
        changes["handle"] = object()
    elif defect == "state":
        handle = replace(case.handle)
        object.__setattr__(handle, "status", "UNPUBLISHED")
        changes["handle"] = handle
    elif defect == "release":
        changes["handle"] = replace(case.handle, release_id="../escape")
    elif defect == "manifest_hash":
        changes["handle"] = replace(case.handle, manifest_hash="main")
    elif defect == "bundles":
        draft = case.handle.faiss._draft
        altered = replace(draft, snapshot=replace(draft.snapshot, candidate_snapshot_id="0"*64))
        changes["handle"] = replace(case.handle, faiss=replace(case.handle.faiss, _draft=altered))
    elif defect == "date":
        changes["validity_date"] = DAY.replace(day=2)
    else:
        changes["validity_date"] = DAY.isoformat()
    reads = []
    monkeypatch.setattr(case.module, "_verified_bytes", lambda *args: reads.append(1))
    with pytest.raises(case.module.EvidenceHydrationError, match="^EVIDENCE_HYDRATION_FAILED$"):
        case.hydrate(hits, **changes)
    assert reads == []


@pytest.mark.parametrize("defect", [
    "missing_text", "child_text", "parent_text", "parent_id", "missing_child", "offset",
    "source", "source_hash", "source_path", "version", "classification", "grants", "heading", "parent_bounds", "role_mapping",
])
def test_selected_row_defenses_and_no_partial_result(case, monkeypatch, defect):
    # Exercise row validation independently of the outer SHA gate. Every
    # successful path uses a real publication. This private test image is
    # deliberately corrupted after its outer artifact check.
    raw = case.module._verified_bytes(case.handle, case.root)
    hit = case.hits[-1]
    cid, did = hit.child.child_id, hit.child.section.document_id
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(raw)
        db.execute("PRAGMA ignore_check_constraints = ON")
        if defect == "missing_text":
            db.execute("UPDATE draft_children SET text = '' WHERE child_id = ?", (cid,))
        elif defect == "child_text":
            db.execute("UPDATE draft_children SET text = ? WHERE child_id = ?", ("sensitive-canary", cid))
        elif defect == "parent_text":
            db.execute("UPDATE draft_parents SET text = '' WHERE parent_id = ?", (hit.child.section.parent_id,))
        elif defect == "parent_id":
            db.execute("UPDATE draft_children SET parent_id = 'wrong' WHERE child_id = ?", (cid,))
        elif defect == "parent_bounds":
            db.execute("UPDATE draft_parents SET start = ? WHERE parent_id = ?",
                       (hit.child.start + 1, hit.child.section.parent_id))
        elif defect == "missing_child":
            db.execute("DELETE FROM draft_children WHERE child_id = ?", (cid,))
        elif defect == "offset":
            db.execute("UPDATE draft_children SET end = end + 1 WHERE child_id = ?", (cid,))
        elif defect == "heading":
            db.execute("UPDATE draft_parents SET heading = 'wrong' WHERE parent_id = ?", (hit.child.section.parent_id,))
        elif defect == "source":
            db.execute("UPDATE draft_documents SET canonical_source = canonical_source || 'x' WHERE document_id = ?", (did,))
        elif defect == "role_mapping":
            db.execute("UPDATE draft_documents SET allowed_roles = ? WHERE document_id = ?", ('{"hr":true}', did))
        else:
            column, value = {
                "source_hash": ("source_sha256", "0"*64), "source_path": ("source_path", "wrong"),
                "version": ("version", 99), "classification": ("classification", "UNKNOWN"),
                "grants": ("allowed_roles", '["unknown"]'),
            }[defect]
            db.execute(f"UPDATE draft_documents SET {column} = ? WHERE document_id = ?", (value, did))
        db.commit()
        altered = db.serialize()
    monkeypatch.setattr(case.module, "_verified_bytes", lambda *args: altered)
    with pytest.raises(case.module.EvidenceHydrationError) as captured:
        case.hydrate()
    assert str(captured.value) == "EVIDENCE_HYDRATION_FAILED"


def test_readonly_private_image_denies_writes_and_functions(case):
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(case.module._verified_bytes(case.handle, case.root))
        db.execute("PRAGMA query_only = ON")
        db.set_authorizer(case.module._readonly)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("DELETE FROM draft_children")
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("SELECT load_extension('anything')")


def test_exception_traceback_is_content_free(case, monkeypatch):
    import traceback
    def fail(*args):
        raise ValueError("sensitive-canary")
    monkeypatch.setattr(case.module, "_selected_rows", fail)
    with pytest.raises(case.module.EvidenceHydrationError) as captured:
        case.hydrate()
    formatted = "".join(traceback.format_exception(captured.type, captured.value, captured.tb))
    assert "sensitive-canary" not in formatted
    assert captured.value.__context__ is None
    assert all(h.child.section.heading not in formatted for h in case.hits if h.child.section.heading)


def test_metadata_sql_selects_only_requested_owning_documents(case):
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(case.module._verified_bytes(case.handle, case.root))
        statements = []
        db.set_trace_callback(statements.append)
        rows = case.module._selected_rows(db, case.handle, case.context, case.hits[:1], DAY)
    assert len(rows) == 1
    metadata = [sql for sql in statements if "FROM main.draft_documents" in sql]
    assert len(metadata) == 1
    assert metadata[0].endswith("WHERE document_id IN ('" + case.hits[0].child.section.document_id + "')")
    assert all("canonical_source" not in sql and "c.text" not in sql and "p.text" not in sql for sql in statements)
