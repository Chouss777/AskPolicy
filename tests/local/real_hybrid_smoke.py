"""Explicit full-corpus hybrid test using the provisioned E5 CPU checkpoint."""

from contextlib import closing
from datetime import date
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from security.test_auth import secured


def test_real_hybrid_smoke(secured, monkeypatch):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval import hybrid, role_bm25, role_faiss
    from askpolicy.retrieval.e5_encoder import E5Encoder

    torch.set_num_threads(4)
    encoder = E5Encoder(Path(os.environ["ASKPOLICY_E5_DIR"]))
    opts = dict(manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
                snapshot_config=role_faiss.DraftSnapshotConfig(), validity_date=date(2026, 9, 1))
    start = perf_counter()
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(ROOT, db)
        lexical = role_bm25.build_role_bm25_bundle(db, **opts)
        dense = role_faiss.build_role_faiss_bundle(db, encoder=encoder, **opts)
    build_seconds = perf_counter() - start
    assert lexical.snapshot == dense.snapshot and dense.snapshot.document_count == 26
    role_faiss.verify_matching_bundles(lexical, dense)
    calls = []
    for module, name in ((role_bm25, "search_role_bm25"), (role_faiss, "search_role_faiss")):
        original = getattr(module, name)
        def observe(*args, _fn=original, _name=name, **kw):
            hits = _fn(*args, **kw)
            calls.append((_name, len(hits)))
            return hits
        monkeypatch.setattr(module, name, observe)
    records = []
    for role, query, family in [
        ("employee", "Quel est le plafond pour une nuit d’hôtel en déplacement ?", "TRAVEL-"),
        ("hr", "registre des congés et soldes des absences", "LEAVE-"),
    ]:
        context = secured.authenticate(secured.sign(roles=[role]), role)
        calls.clear()
        started = perf_counter()
        hits = hybrid.search_role_hybrid(lexical, dense, context, query=query,
                         expected_snapshot_id=dense.snapshot.candidate_snapshot_id, validity_date=opts["validity_date"])
        elapsed = perf_counter() - started
        assert len(hits) == len({h.child.child_id for h in hits}) == 20
        assert [name for name, _ in calls] == ["search_role_bm25", "search_role_faiss"]
        assert 0 < calls[0][1] <= 30 and calls[1][1] == 30
        assert all(role in h.child.document.metadata.allowed_roles for h in hits)
        assert any(h.child.section.document_id.startswith(family) for h in hits)
        assert hits == hybrid.search_role_hybrid(lexical, dense, context, query=query,
                         expected_snapshot_id=dense.snapshot.candidate_snapshot_id, validity_date=opts["validity_date"])
        records.append(dict(role=role, query=query, bm25=calls[0][1], faiss=calls[1][1], hybrid=len(hits),
                            child_ids=[h.child.child_id for h in hits], seconds=elapsed, acl_violations=0))
    print(json.dumps(dict(snapshot=dense.snapshot.candidate_snapshot_id,
                          manifest=dense.snapshot.source_manifest_hash, revision=encoder.revision,
                          status=dense.snapshot.status, build_seconds=build_seconds, queries=records),
                     ensure_ascii=True, sort_keys=True))
