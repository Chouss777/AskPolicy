"""Explicit pytest run with ASKPOLICY_E5_DIR pointing at the provisioned checkpoint."""

from collections import Counter
from contextlib import closing
from datetime import date
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from security.test_auth import secured
from unit.test_e5_prefix import _working_set_mib


def test_real_dense_smoke(secured):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval import role_bm25, role_faiss as dense
    from askpolicy.retrieval.bm25_core import EvidenceChunk
    from askpolicy.retrieval.e5_encoder import E5Encoder

    torch.set_num_threads(4)
    assert torch.version.cuda is None
    encoder = E5Encoder(Path(os.environ["ASKPOLICY_E5_DIR"]))
    day = date(2026, 9, 1)
    options = dict(manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
                   snapshot_config=dense.DraftSnapshotConfig(), validity_date=day)
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(ROOT, db)
        bm25 = role_bm25.build_role_bm25_bundle(db, **options)
        started = perf_counter()
        bundle = dense.build_role_faiss_bundle(db, encoder=encoder, **options)
        elapsed = perf_counter() - started
        sources = {cid: (doc, start, end, source, json.loads(grants)) for cid, doc, start, end, source, grants
                   in db.execute("SELECT child_id, document_id, start, end, canonical_source, allowed_roles "
                                 "FROM draft_children JOIN draft_documents USING(document_id)")}
    dense.verify_matching_bundles(bm25, bundle)
    snapshot = bundle.snapshot
    assert snapshot.candidate_snapshot_id == "ddd98004d41bc5fcd9021078c036e3b07cbe94066f528a9bbbc44eef45453794"
    assert (snapshot.status, snapshot.document_count, snapshot.child_count) == ("UNPUBLISHED", 26, 387)
    report = {}
    for index in bundle.indexes:
        meta = index.metadata
        role = meta.role
        vectors = index._flat.engine.reconstruct_n(0, len(meta.rows))
        dense.validate_vectors(vectors, len(meta.rows))
        norms = np.linalg.norm(vectors, axis=1)
        cursor = {}
        for row in meta.rows:
            doc, start, end, source, grants = sources[row.child_id]
            assert role in grants and doc in meta.indexed_document_ids
            assert cursor.get(row.child_id, start) == row.start < row.end <= end
            assert encoder.passage_token_count(source[row.start:row.end]) == row.token_count <= 512
            cursor[row.child_id] = row.end
        assert set(cursor) == set(meta.indexed_child_ids)
        assert all(stop == sources[cid][2] for cid, stop in cursor.items())
        expected_counts = {"employee": (15, 229), "hr": (18, 271), "finance": (15, 229), "it_security": (17, 259)}
        assert (len(meta.indexed_document_ids), len(cursor)) == expected_counts[role]
        queries = {"employee": "Quel est le plafond pour une nuit d’hôtel en déplacement ?",
                   "hr": "registre des congés et soldes des absences",
                   "finance": "Which expenses can be reimbursed?", "it_security": "contrôle des accès et habilitations"}
        context = secured.authenticate(secured.sign(roles=[role]), role)
        hits = dense.search_role_faiss(bundle, context, query=queries[role],
                                       expected_snapshot_id=snapshot.candidate_snapshot_id, validity_date=day)
        assert len({hit.child.child_id for hit in hits}) == len(hits) == 30
        for hit in hits:
            doc, _, _, _, grants = sources[hit.child.child_id]
            assert role in grants and doc == hit.child.section.document_id
        assert any(h.child.section.document_id.startswith({"hr": "LEAVE-", "it_security": "ACCESS-",
                   "employee": "TRAVEL-", "finance": "EXPENSE"}[role]) for h in hits)
        per_child = Counter(row.child_id for row in meta.rows)
        report[role] = dict(documents=len(meta.indexed_document_ids), children=len(cursor), vectors=len(meta.rows),
                            split_children=sum(n > 1 for n in per_child.values()), fingerprint=meta.fingerprint,
                            norms=[float(norms.min()), float(norms.max())], hits=len(hits), unauthorized=0)
    # The fixed corpus need not contain an overlength child. Exercise that
    # path separately, using the same tokenizer and encoder on synthetic text.
    child = bundle._draft.children[meta.indexed_child_ids[0]]
    text = ("\tÉcole e\u0301، سفر 🌍!\n" * 120) + " fin.  "
    example = EvidenceChunk(child.child_id, child.section, child.document, child.start, text)
    pieces = list(dense.split_passage(example, encoder.passage_token_count))
    assert encoder.passage_token_count(text) > 512 and len(pieces) > 1
    assert "".join(p.text for p in pieces) == text
    for p in pieces:
        assert p.text == text[p.segment.start - child.start:p.segment.end - child.start]
        encoded = encoder.encode_passages([p.text])
        assert encoded.token_counts == (p.segment.token_count,) and p.segment.token_count <= 512
        dense.validate_vectors(encoded.vectors, 1)
    print(json.dumps(dict(snapshot=snapshot.candidate_snapshot_id, manifest=snapshot.source_manifest_hash,
                          snapshot_config=snapshot.config_hash, dense_config=meta.configuration.fingerprint,
                          artifacts=encoder.artifact_fingerprint, revision=encoder.revision,
                          synthetic_split_token_counts=[p.segment.token_count for p in pieces],
                          faiss_version=meta.configuration.faiss_version, indexing_seconds=elapsed,
                          ram=_working_set_mib(), roles=report), sort_keys=True))
