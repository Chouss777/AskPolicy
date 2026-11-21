"""Section-level retrieval metrics over ranked source children.

The child ranking retains its original positions. Repeated sections consume
positions but receive graded gain only on their first occurrence. nDCG uses
gain 2**grade - 1 and discount log2(rank + 1), with ideal unique gold sections.
No-positive executions return None and are excluded from macro denominators.
"""

from collections.abc import Mapping, Sequence
from math import log2

Section = tuple[str, str]
METRIC_NAMES = ("recall_at_5", "mrr_at_10", "ndcg_at_10")


def _section(value: Section) -> None:
    if type(value) is not tuple or len(value) != 2 or any(type(v) is not str or not v.strip() for v in value):
        raise ValueError("INVALID_SECTION")


def section_metrics(ranking: Sequence[Section], gold: Mapping[Section, int]) -> dict[str, float] | None:
    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)) or not isinstance(gold, Mapping):
        raise ValueError("INVALID_METRIC_INPUT")
    for section in ranking:
        _section(section)
    for section, grade in gold.items():
        _section(section)
        if type(grade) is not int or grade <= 0:
            raise ValueError("INVALID_GRADE")
    if not gold:
        return None
    recall = len(set(ranking[:5]) & gold.keys()) / len(gold)
    reciprocal = next((1 / rank for rank, section in enumerate(ranking[:10], 1) if section in gold), 0.0)
    seen = set()
    dcg = 0.0
    for rank, section in enumerate(ranking[:10], 1):
        if section not in seen and section in gold:
            dcg += (2**gold[section] - 1) / log2(rank + 1)
        seen.add(section)
    ideal = sum((2**grade - 1) / log2(rank + 1)
                for rank, grade in enumerate(sorted(gold.values(), reverse=True)[:10], 1))
    return dict(recall_at_5=recall, mrr_at_10=reciprocal, ndcg_at_10=dcg / ideal)


def macro_metrics(cases: Sequence[dict[str, float] | None]) -> dict:
    eligible = [case for case in cases if case is not None]
    return dict(
        eligible_positive_count=len(eligible), excluded_no_positive_count=len(cases) - len(eligible),
        **{name: sum(case[name] for case in eligible) / len(eligible) if eligible else None
           for name in METRIC_NAMES},
    )


def percentile(samples: Sequence[float], percentage: float) -> float:
    """Linear interpolation at (n - 1) * percentage / 100."""
    if not samples or not 0 <= percentage <= 100:
        raise ValueError("INVALID_PERCENTILE_INPUT")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentage / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
