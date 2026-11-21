"""Pure rank-only reciprocal rank fusion over source child identities."""

from collections.abc import Sequence
from dataclasses import dataclass

RRF_K = 60
INPUT_LIMIT = 30
OUTPUT_LIMIT = 20


class RRFError(ValueError):
    """Content-free rejection of a malformed ranking."""


@dataclass(frozen=True, slots=True)
class FusedIdentity:
    child_id: str
    score: float
    bm25_rank: int | None
    faiss_rank: int | None


def _ranks(ranking: Sequence[str]) -> dict[str, int]:
    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes, bytearray)):
        raise RRFError("INVALID_RANKING")
    if len(ranking) > INPUT_LIMIT:
        raise RRFError("RANKING_TOO_LONG")
    if any(type(child) is not str or not child.strip() for child in ranking):
        raise RRFError("INVALID_CHILD_ID")
    if len(set(ranking)) != len(ranking):
        raise RRFError("DUPLICATE_CHILD_ID")
    return {child: rank for rank, child in enumerate(ranking, 1)}


def reciprocal_rank_fusion(bm25: Sequence[str], faiss: Sequence[str]) -> tuple[FusedIdentity, ...]:
    left, right = _ranks(bm25), _ranks(faiss)
    fused = []
    for child in left.keys() | right.keys():
        score = sum(1 / (RRF_K + ranks[child]) for ranks in (left, right) if child in ranks)
        fused.append(FusedIdentity(child, score, left.get(child), right.get(child)))
    return tuple(sorted(fused, key=lambda hit: (-hit.score, hit.child_id))[:OUTPUT_LIMIT])
