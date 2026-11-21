"""Explicit real E5 persistence round trip, with artifacts under ignored runtime."""

from contextlib import closing
from datetime import date
import gc
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from security.test_auth import secured
from unit.test_e5_prefix import _working_set_mib


def test_real_snapshot_roundtrip(secured):
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval.e5_encoder import E5Encoder
    from askpolicy.storage import index_store, publication
    from askpolicy.storage.snapshot import DraftSnapshotConfig

    torch.set_num_threads(4)
    encoder = E5Encoder(Path(os.environ["ASKPOLICY_E5_DIR"]))
    root = ROOT / "runtime" / ("step96-" + uuid4().hex)
    opts = dict(manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
                snapshot_config=DraftSnapshotConfig(),validity_date=date(2026,9,1),encoder=encoder)
    started = perf_counter()
    with closing(sqlite3.connect(":memory:")) as db:
        ingest_manifest_to_draft(ROOT,db)
        handle = publication.publish_snapshot(db,root,**opts)
    publish_seconds = perf_counter()-started
    release_id = handle.release_id
    original_fingerprints = [i.metadata.fingerprint for i in handle.faiss.indexes]
    del handle
    gc.collect()
    started = perf_counter()
    handle = publication.load_active_snapshot(root,**opts)
    load_seconds = perf_counter()-started
    assert handle.release_id == release_id
    assert handle.faiss.snapshot.document_count == 26 and handle.faiss.snapshot.status=="UNPUBLISHED"
    assert original_fingerprints == [i.metadata.fingerprint for i in handle.faiss.indexes]
    results=[]
    for role,query,family in [
        ("employee","Quel est le plafond pour une nuit d’hôtel en déplacement ?","TRAVEL-"),
        ("hr","registre des congés et soldes des absences","LEAVE-"),
    ]:
        started=perf_counter()
        hits=handle.search(secured.authenticate(secured.sign(roles=[role]),role),
                           query=query,validity_date=opts["validity_date"])
        seconds=perf_counter()-started
        assert len(hits)==len({h.child.child_id for h in hits})==20
        assert all(role in h.child.document.metadata.allowed_roles for h in hits)
        assert any(h.child.section.document_id.startswith(family) for h in hits)
        results.append(dict(role=role,query=query,hits=len(hits),violations=0,seconds=seconds,
                            first_child=hits[0].child.child_id))
    paths=index_store.artifact_paths()
    sizes={slot:(root/"releases"/release_id/path).stat().st_size for slot,path in paths.items()}
    counts={i.metadata.role:dict(documents=len(i.metadata.indexed_document_ids),
                                 children=len(i.metadata.indexed_child_ids),vectors=len(i.metadata.rows))
            for i in handle.faiss.indexes}
    report = dict(root=str(root),release=release_id,snapshot=handle.faiss.snapshot.candidate_snapshot_id,
                  manifest_hash=handle.manifest_hash,revision=encoder.revision,
                  artifact_fingerprint=encoder.artifact_fingerprint,counts=counts,
                  index_fingerprints=original_fingerprints,publish_seconds=publish_seconds,
                  load_seconds=load_seconds,ram=_working_set_mib(),sizes=sizes,queries=results)
    # Confirm process-independent reload after releasing the parent's models
    # and index bundles. The child uses the same signed production JWT fixture.
    del handle
    del opts["encoder"]
    del encoder
    gc.collect()
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--reload", str(root), os.environ["ASKPOLICY_E5_DIR"]],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    fresh = json.loads(child.stdout.strip().splitlines()[-1])
    assert fresh["pid"] != os.getpid() and fresh["release"] == release_id
    assert fresh["counts"] == [20,20] and fresh["violations"] == 0
    report["fresh_process"] = fresh
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from askpolicy.retrieval.e5_encoder import E5Encoder
    from askpolicy.storage.publication import load_active_snapshot
    from askpolicy.storage.snapshot import DraftSnapshotConfig

    assert sys.argv[1] == "--reload"
    torch.set_num_threads(4)
    local_encoder = E5Encoder(Path(sys.argv[3]))
    started = perf_counter()
    loaded = load_active_snapshot(
        Path(sys.argv[2]), manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
        snapshot_config=DraftSnapshotConfig(),validity_date=date(2026,9,1),encoder=local_encoder,
    )
    counts = []
    with pytest.MonkeyPatch.context() as patch:
        identity = secured.__wrapped__(patch)
        for role, query in (("employee","travel hébergement déplacement"),("hr","registre des congés soldes")):
            hits = loaded.search(identity.authenticate(identity.sign(roles=[role]),role),
                                 query=query,validity_date=date(2026,9,1))
            assert len(hits) == len({h.child.child_id for h in hits}) == 20
            assert all(role in h.child.document.metadata.allowed_roles for h in hits)
            counts.append(len(hits))
    print(json.dumps(dict(pid=os.getpid(),release=loaded.release_id,counts=counts,violations=0,
                          load_and_search_seconds=perf_counter()-started)))
