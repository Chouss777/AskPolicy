"""Explicit offline real-E5 hydration and full-child pair-length diagnostic.

Run separately. Ordinary test discovery does not load model weights. The
ignored runtime report contains identifiers and counts, never query/passage
text, scores, labels or a retrieval-quality conclusion.
"""

from collections import Counter
from contextlib import closing
from datetime import date
import json
import os
from pathlib import Path
import platform
import socket
import sqlite3
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "eval"))

from security.test_auth import secured
from unit.test_e5_prefix import _working_set_mib


def test_real_published_evidence_hydration_smoke(secured):
    import torch
    from askpolicy.ingestion.coordinator import ingest_manifest_to_draft
    from askpolicy.retrieval.cross_encoder import CrossEncoder, CrossEncoderError
    from askpolicy.retrieval.e5_encoder import E5Encoder
    from askpolicy.retrieval.evidence_hydration import hydrate_published_hits
    from askpolicy.security.acl import can_read_document
    from askpolicy.storage import index_store, publication
    from askpolicy.storage.snapshot import DraftSnapshotConfig
    from dev_retrieval_diagnostic import read_cases

    started = perf_counter()
    torch.set_num_threads(4)
    raw_cases = (ROOT / "eval/datasets/dev_candidates.jsonl").read_bytes()
    cases = read_cases(raw_cases)
    assert len(cases) == 5
    assert all(c.annotation_review_status == "human_approved" and not c.revalidation_required for c in cases)
    day = date(2026, 9, 1)
    runtime = ROOT / "runtime" / ("step105-" + uuid4().hex)
    runtime.mkdir(parents=True)
    timings = {}
    def no_network(*args, **kwargs):
        raise AssertionError("UNEXPECTED_NETWORK_ACCESS")
    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "getaddrinfo", no_network):
        tick = perf_counter()
        encoder = E5Encoder(Path(os.environ["ASKPOLICY_E5_DIR"]))
        timings["e5_load_seconds"] = perf_counter() - tick
        options = dict(manifest=(ROOT / "data/corpus_manifest.yaml").read_bytes(),
                       snapshot_config=DraftSnapshotConfig(), validity_date=day, encoder=encoder)
        tick = perf_counter()
        with closing(sqlite3.connect(":memory:")) as draft:
            ingest_manifest_to_draft(ROOT, draft)
            published = publication.publish_snapshot(draft, runtime, **options)
        handle = publication.load_active_snapshot(runtime, **options)
        assert handle.release_id == published.release_id and handle.manifest_hash == published.manifest_hash
        timings["ingest_publish_reload_seconds"] = perf_counter() - tick
        tick = perf_counter()
        cross_encoder = CrossEncoder(Path(os.environ["ASKPOLICY_RERANKER_DIR"]))
        timings["cross_encoder_load_seconds"] = perf_counter() - tick
        assert cross_encoder._tokenizer.is_fast
        all_counts, per_case, coverage = [], [], Counter()
        unauthorized = 0
        overflow_probe = None
        # Protect against accidentally enabling reranking during this audit.
        with patch.object(cross_encoder._model, "forward", side_effect=AssertionError("AUDIT_REACHED_INFERENCE")):
            for case in cases:
                context = secured.authenticate(secured.sign(roles=[case.active_role]), case.active_role)
                tick = perf_counter()
                hits = handle.search(context, query=case.query, validity_date=day)
                search_seconds = perf_counter() - tick
                assert len(hits) == 20
                tick = perf_counter()
                passages = hydrate_published_hits(handle, context, hits, release_root=runtime, validity_date=day)
                hydration_seconds = perf_counter() - tick
                assert len(passages) == 20
                # Independent selected-span check against the same checked
                # published image, without loading unrequested child texts.
                raw = (runtime / "releases" / handle.release_id / "sqlite/corpus.sqlite").read_bytes()
                record = index_store.read_json(
                    (runtime / "releases" / handle.release_id / "manifest.json").read_bytes())
                assert index_store.digest(raw) == record["artifacts"]["sqlite"]
                with closing(sqlite3.connect(":memory:")) as check:
                    check.deserialize(raw)
                    check.execute("PRAGMA query_only = ON")
                    for passage, hit in zip(passages, hits, strict=True):
                        source, text = check.execute(
                            "SELECT d.canonical_source, c.text FROM draft_children c JOIN draft_documents d "
                            "ON d.document_id = c.document_id WHERE c.child_id = ? AND c.document_id = ?",
                            (passage.child_id, passage.document_id),
                        ).fetchone()
                        assert source[passage.start:passage.end] == passage.text == text
                        assert passage.active_role == case.active_role
                        assert passage.release_id == handle.release_id
                        assert passage.snapshot_id == handle.bm25.snapshot.candidate_snapshot_id
                        allowed = can_read_document(context,
                            classification=hit.child.document.metadata.classification,
                            allowed_roles=hit.child.document.metadata.allowed_roles)
                        unauthorized += int(not allowed)
                        assert allowed
                        coverage[hit.child.document.metadata.language] += 1
                tick = perf_counter()
                with cross_encoder._lock:
                    encoded = cross_encoder._tokenizer(
                        [case.query] * len(passages), [p.text for p in passages],
                        add_special_tokens=True, padding=False, truncation=False,
                        return_attention_mask=True, verbose=False,
                    )
                counts = [len(ids) for ids in encoded["input_ids"]]
                assert len(counts) == len(passages) and min(counts) > 0
                assert all(len(mask) == length for mask, length in
                           zip(encoded["attention_mask"], counts, strict=True))
                token_seconds = perf_counter() - tick
                overflow = sum(n > 512 for n in counts)
                if overflow_probe is None:
                    overflow_probe = next(((case.query, p.text) for p, n in zip(passages, counts, strict=True)
                                           if n > 512), None)
                all_counts.extend(counts)
                per_case.append(dict(query_id=case.query_id, active_role=case.active_role,
                    language=case.language, pairs=len(counts), overlength_pairs=overflow,
                    minimum=min(counts), median=median(counts), maximum=max(counts),
                    search_seconds=search_seconds, hydration_seconds=hydration_seconds,
                    tokenization_seconds=token_seconds))
            overflow_rejected = 0
            if overflow_probe is not None:
                with pytest.raises(CrossEncoderError, match="^PAIR_TOO_LONG$"):
                    cross_encoder.score_pairs([overflow_probe])
                overflow_rejected = 1
        assert unauthorized == 0 and len(all_counts) == 100
        report = dict(
            diagnostic="step105_full_child_joint_token_audit",
            dataset_sha256=index_store.digest(raw_cases), trusted_validity_date=day.isoformat(),
            release_id=handle.release_id, snapshot_id=handle.bm25.snapshot.candidate_snapshot_id,
            published_manifest_sha256=handle.manifest_hash,
            e5_revision=encoder.revision, e5_artifact_fingerprint=encoder.artifact_fingerprint,
            cross_encoder_revision=cross_encoder.revision,
            cross_encoder_artifact_fingerprint=cross_encoder.artifact_fingerprint,
            evaluated_cases=len(cases), total_pairs=len(all_counts),
            overlength_pairs=sum(n > 512 for n in all_counts),
            overlength_percentage=100 * sum(n > 512 for n in all_counts) / len(all_counts),
            cases_with_overflow=sum(c["overlength_pairs"] > 0 for c in per_case),
            token_counts=dict(minimum=min(all_counts), median=median(all_counts), maximum=max(all_counts)),
            roles=sorted({c.active_role for c in cases}), query_languages=sorted({c.language for c in cases}),
            passage_language_counts=dict(sorted(coverage.items())), cases=per_case,
            unauthorized_hits=unauthorized, failure_count=0, network_requests=0,
            reranker_forward_calls=0, overlength_preinference_rejections=overflow_rejected,
            device="cpu", dtype="float32", hardware=dict(platform=platform.platform(),
                processor=platform.processor(), logical_cpus=os.cpu_count(), torch_threads=torch.get_num_threads()),
            memory=_working_set_mib(), timings=timings, total_seconds=perf_counter()-started,
            limitations=[
                "Five approved French development executions and two active roles only.",
                "Fixture tests cover all four roles. This diagnostic does not certify other queries or languages.",
                "Joint-token counts include special tokens and the full original query and child.",
                "No production overflow policy selected. No splitting, trimming, aggregation or dropped candidates.",
                "Tokenizer diagnostic only. No reranker inference, quality or refusal accuracy measured.",
            ],
        )
        target = runtime / "overflow-audit.json"
        target.write_bytes(index_store.canonical(report) + b"\n")
        print(json.dumps(dict(report_path=str(target), **report), sort_keys=True))
