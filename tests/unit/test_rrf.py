"""Acceptance checks for rank-only RRF."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest


@pytest.fixture
def rrf(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    from askpolicy.retrieval import rrf
    return rrf


def test_exact_formula_and_order_without_input_mutation(rrf):
    left, right = ["A", "B", "C"], ["B", "D", "A"]
    result = rrf.reciprocal_rank_fusion(left, right)
    assert rrf.RRF_K == 60
    assert [h.child_id for h in result] == ["B", "A", "D", "C"]
    expected = {"A": (1/61 + 1/63, 1, 3), "B": (1/62 + 1/61, 2, 1),
                "C": (1/63, 3, None), "D": (1/62, None, 2)}
    for h in result:
        score, a, b = expected[h.child_id]
        assert h.score == pytest.approx(score, abs=1e-16)
        assert (h.bm25_rank, h.faiss_rank) == (a, b)
    assert left == ["A", "B", "C"] and right == ["B", "D", "A"]
    assert rrf.reciprocal_rank_fusion(right, left)[0].child_id == "B"
    assert rrf.reciprocal_rank_fusion(["B", "A"], [])[0].child_id == "B"
    with pytest.raises(FrozenInstanceError):
        result[0].score = 0


@pytest.mark.parametrize("left,right,ids", [
    ([], [], []), ([], ["B", "A"], ["B", "A"]), (["A"], [], ["A"]),
    (["Z"], ["A"], ["A", "Z"]),
])
def test_empty_and_ties(rrf, left, right, ids):
    assert [h.child_id for h in rrf.reciprocal_rank_fusion(left, right)] == ids


@pytest.mark.parametrize("bad", [None, "ABC", b"ABC", {"A": 1}, {"A"}, iter(["A"]),
                                     ["A", "A"], [""], [" "], [1], [None], ["A"] * 31,
                                     [str(i) for i in range(31)]])
def test_invalid_input_rejected_in_either_ranking(rrf, bad):
    for a, b in ((bad, []), ([], bad)):
        with pytest.raises(rrf.RRFError):
            rrf.reciprocal_rank_fusion(a, b)


def test_budget_exact_scores_reproducibility_and_small_differences(rrf):
    ids = [f"child-{i:02d}" for i in range(30)]
    for _ in range(3):
        fused = rrf.reciprocal_rank_fusion(ids, ids)
        assert len(fused) == 20 and [h.child_id for h in fused] == ids[:20]
        assert all(h.score == 2 / (61 + i) for i, h in enumerate(fused))
    # A single late rank difference must survive without rounding.
    fused = rrf.reciprocal_rank_fusion(ids, list(reversed(ids)))
    assert fused[0].score > fused[2].score
