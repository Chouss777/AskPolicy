"""Exact metric mathematics independent of embedding models."""

from math import log2
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"eval"))
from retrieval_metrics import section_metrics,macro_metrics,percentile

A=("doc-a","parent-a")
B=("doc-b","parent-b")
X=("other","parent-x")


def test_exact_graded_metrics_and_repeated_parent():
    score=section_metrics([X,A,A,B],{A:3,B:1})
    ideal=7+1/log2(3)
    assert score["recall_at_5"]==1
    assert score["mrr_at_10"]==0.5
    assert score["ndcg_at_10"]==pytest.approx((7/log2(3)+1/log2(5))/ideal,abs=1e-15)
    assert section_metrics([A,A,A],{A:3})["ndcg_at_10"]==1


@pytest.mark.parametrize("ranking,recall,mrr",[
    ([],0,0),([X]*4+[A],1,1/5),([X]*5+[A],0,1/6),
    ([X]*9+[A],0,1/10),([X]*10+[A],0,0),
])
def test_child_rank_cutoffs(ranking,recall,mrr):
    score=section_metrics(ranking,{A:3})
    assert score["recall_at_5"]==recall and score["mrr_at_10"]==mrr
    if not mrr:
        assert score["ndcg_at_10"]==0


def test_macro_excludes_no_positive_cases_and_does_not_credit_non_gold():
    perfect=section_metrics([A],{A:3})
    missing=section_metrics([X],{A:3})
    excluded=section_metrics([A],{})
    assert excluded is None
    macro=macro_metrics([perfect,missing,excluded,excluded])
    assert macro==dict(eligible_positive_count=2,excluded_no_positive_count=2,
                      recall_at_5=0.5,mrr_at_10=0.5,ndcg_at_10=0.5)
    assert macro_metrics([None])["recall_at_5"] is None
    assert section_metrics([A],{A:3,B:2})["recall_at_5"]==0.5
    assert section_metrics([A],{A:3})==section_metrics([A],{A:3})


@pytest.mark.parametrize("grade",[0,-1,True,1.5,"3"])
def test_invalid_grades(grade):
    with pytest.raises(ValueError):
        section_metrics([A],{A:grade})


def test_quantiles_are_exploratory_linear_interpolation():
    assert percentile([1,2,3,4,5],50)==3
    assert percentile([1,2,3,4,5],95)==pytest.approx(4.8)
