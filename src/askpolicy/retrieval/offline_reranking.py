"""Experimental internal reranking. Production hybrid retrieval never calls it."""

from dataclasses import dataclass, field
from datetime import date
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch

from ..auth.context import AuthenticatedContext
from ..security.acl import can_read_document
from ..storage.index_store import PublishedSnapshot
from .cross_encoder import (
    ARTIFACT_FINGERPRINT, MODEL_ID, MODEL_REVISION, CrossEncoder, CrossEncoderError, PairScores,
)
from .evidence_hydration import AuthorizedPassage, hydrate_published_hits
from .hybrid import HybridHit, _gate


class OfflineRerankError(ValueError):
    """A fixed content-free code, without attached source exception frames."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RerankedHit:
    original: HybridHit = field(repr=False)
    release_id: str
    score: float
    token_count: int

    @property
    def child(self):
        return self.original.child

    @property
    def candidate_snapshot_id(self):
        return self.original.candidate_snapshot_id

    @property
    def active_role(self):
        return self.original.active_role


@dataclass(frozen=True, slots=True)
class RerankTimings:
    hydration_seconds: float
    token_validation_seconds: float
    scoring_seconds: float
    ranking_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class OfflineRerankResult:
    hits: tuple[RerankedHit, ...]
    timings: RerankTimings


def _check_encoder(encoder):
    # The existing constructor owns artifact verification. Trusted local code
    # supplies its loaded instance, never an HTTP/client-selected scorer.
    if (type(encoder) is not CrossEncoder or encoder.model_id != MODEL_ID
        or encoder.revision != MODEL_REVISION or encoder.artifact_fingerprint != ARTIFACT_FINGERPRINT
        or not encoder._tokenizer.is_fast or encoder._tokenizer.model_max_length != 512
        or encoder._model.training
        or not all(p.device.type == "cpu" and p.dtype == torch.float32 for p in encoder._model.parameters())):
        raise ValueError


def _passage_matches(passage, hit, handle, role):
    return (
        type(passage) is AuthorizedPassage
        and passage.release_id == handle.release_id
        and passage.snapshot_id == hit.candidate_snapshot_id
        and passage.active_role == role == hit.active_role
        and passage.child_id == hit.child.child_id
        and passage.parent_id == hit.child.section.parent_id
        and passage.document_id == hit.child.section.document_id
        and type(passage.start) is int and type(passage.end) is int
        and passage.start == hit.child.start
        and type(passage.text) is str and bool(passage.text)
        and 0 <= passage.start < passage.end
        and passage.end - passage.start == len(passage.text)
    )


def _run(handle, context, query, hits, release_root, validity_date, encoder):
    started = perf_counter()
    _check_encoder(encoder)
    if type(query) is not str or not query.strip():
        raise ValueError
    tick = perf_counter()
    passages = hydrate_published_hits(handle, context, hits, release_root=release_root, validity_date=validity_date)
    role, _, _ = _gate(handle.bm25, handle.faiss, context,
                       handle.bm25.snapshot.candidate_snapshot_id, validity_date)
    if (type(passages) is not tuple or len(passages) != len(hits)
        or not all(_passage_matches(p, h, handle, role) for p, h in zip(passages, hits, strict=True))):
        raise ValueError
    hydration_seconds = perf_counter() - tick
    pairs = tuple((query, p.text) for p in passages)
    tick = perf_counter()
    counts = []
    # Delegate joint encoding to the approved scorer. All pairs are checked
    # before scoring any candidate. Its score_pairs independently repeats
    # validation before its first forward pass, including unchanged counts.
    with encoder._lock:
        for start in range(0, len(pairs), 4):
            _, measured = encoder._tokens(pairs[start:start+4])
            counts.extend(measured)
    if any(type(n) is not int or not 1 <= n <= 512 for n in counts) or len(counts) != len(hits):
        raise ValueError
    token_seconds = perf_counter() - tick
    tick = perf_counter()
    scored = encoder.score_pairs(pairs) if pairs else PairScores((), ())
    scoring_seconds = perf_counter() - tick
    if (type(scored) is not PairScores or type(scored.scores) is not tuple
        or type(scored.token_counts) is not tuple or scored.token_counts != tuple(counts)
        or len(scored.scores) != len(hits)
        or any(type(s) is not float or not isfinite(s) for s in scored.scores)):
        raise ValueError
    tick = perf_counter()
    _check_encoder(encoder)
    _gate(handle.bm25, handle.faiss, context, handle.bm25.snapshot.candidate_snapshot_id, validity_date)
    for hit in hits:
        meta = hit.child.document.metadata
        if (hit.child != handle.bm25._draft.children.get(hit.child.child_id)
            or not can_read_document(context, classification=meta.classification, allowed_roles=meta.allowed_roles)):
            raise ValueError
    ranked = tuple(sorted(
        (RerankedHit(hit, handle.release_id, score, count)
         for hit, score, count in zip(hits, scored.scores, counts, strict=True)),
        key=lambda h: (-h.score, h.child.child_id),
    ))
    ranking_seconds = perf_counter() - tick
    return OfflineRerankResult(ranked, RerankTimings(
        hydration_seconds, token_seconds, scoring_seconds, ranking_seconds, perf_counter()-started,
    ))


def rerank_published_hits(
    handle: PublishedSnapshot, context: AuthenticatedContext, query: str, hits: tuple[HybridHit, ...], *,
    release_root: Path, validity_date: date, encoder: CrossEncoder,
) -> OfflineRerankResult:
    """Score every authorized full child or fail the entire experimental call.

    Stage timers are disjoint. Scoring includes score_pairs' own repeated
    tokenization and batch padding. No production overflow decision is made.
    Inputs are trusted internal search results, with catalog/ACL verification
    delegated to the secure hydrator. No search, caching or source persistence.
    """
    code = "OFFLINE_RERANK_FAILED"
    try:
        return _run(handle, context, query, hits, release_root, validity_date, encoder)
    except CrossEncoderError as error:
        if error.code == "PAIR_TOO_LONG":
            code = "TOKEN_LIMIT"
    except Exception:
        pass
    raise OfflineRerankError(code)
