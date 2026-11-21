"""Bounded fixture scoring and failure containment, independent of real weights."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from test_cross_encoder_checkpoint import local,module
from integration.test_embedding_revision import dense_env
from security.test_auth import secured


@pytest.mark.parametrize("batch_size",[1,2,4])
def test_order_one_score_each_and_bounded_batches(local,batch_size):
    encoder = module.CrossEncoder(local.path,batch_size=batch_size)
    pairs = [("q" * size,"synthetic passage") for size in range(1,10)]
    result = encoder.score_pairs(pairs)
    assert result.scores == tuple(float(size) for size in range(1,10))
    assert len(result.token_counts) == len(pairs)
    assert max(local.model.calls) <= batch_size
    assert encoder.score_pairs(pairs) == result
    assert all(call[2] == dict(add_special_tokens=True,padding=False,truncation=False,
                              return_attention_mask=True,verbose=False) for call in local.tokenizer.calls)
    with pytest.raises(FrozenInstanceError):
        result.scores = ()


@pytest.mark.parametrize("bad",[None,[], "pair", [("query",)], [("query","")], [(1,"passage")],
                                   [("query","passage","extra")], [{"query":"passage"}]])
def test_invalid_inputs(local,bad):
    encoder = module.CrossEncoder(local.path)
    with pytest.raises(module.CrossEncoderError,match="^INVALID_PAIRS$"):
        encoder.score_pairs(bad)
    assert not local.model.calls


@pytest.mark.parametrize("size",[0,5,True,"4"])
def test_invalid_batch_size_cannot_reach_inference(local,size):
    with pytest.raises(module.CrossEncoderError,match="^INVALID_BATCH_SIZE$"):
        module.CrossEncoder(local.path,batch_size=size)
    encoder = module.CrossEncoder(local.path)
    encoder.batch_size = size
    with pytest.raises(module.CrossEncoderError,match="^INVALID_BATCH_SIZE$"):
        encoder.score_pairs([("query","passage")])
    assert not local.model.calls


def test_exact_pair_boundary_and_later_overflow_prevents_all_inference(local):
    encoder = module.CrossEncoder(local.path,batch_size=1)
    valid = ("query"," ".join(["word"] * 507))
    assert encoder.score_pairs([valid]).token_counts == (512,)
    local.model.calls.clear()
    invalid = (valid[0],valid[1] + " word")
    with pytest.raises(module.CrossEncoderError,match="^PAIR_TOO_LONG$"):
        encoder.score_pairs([("short","valid"),invalid])
    assert not local.model.calls


@pytest.mark.parametrize("value",[float("nan"),float("inf"),float("-inf")])
def test_nonfinite_logits_rejected(local,value):
    encoder = module.CrossEncoder(local.path)
    local.model.output = torch.tensor([[value]],dtype=torch.float32)
    with pytest.raises(module.CrossEncoderError,match="^INVALID_LOGITS$"):
        encoder.score_pairs([("query","passage")])


@pytest.mark.parametrize("shape,dtype",[((1,2),torch.float32),((1,1),torch.float64),((2,1),torch.float32)])
def test_malformed_logits_rejected(local,shape,dtype):
    encoder = module.CrossEncoder(local.path)
    local.model.output = torch.ones(shape,dtype=dtype)
    with pytest.raises(module.CrossEncoderError,match="^INVALID_LOGITS$"):
        encoder.score_pairs([("query","passage")])


def test_failure_messages_never_include_passages(local,monkeypatch,caplog,capsys):
    encoder = module.CrossEncoder(local.path)
    marker = "synthetic-private-question-and-passage"
    def fail(*args,**kwargs):
        raise ValueError(marker)
    monkeypatch.setattr(encoder,"_tokenizer",fail)
    with pytest.raises(module.CrossEncoderError,match="^SCORING_FAILED$") as raised:
        encoder.score_pairs([(marker,marker)])
    out,err = capsys.readouterr()
    assert marker not in str(raised.value) + caplog.text + out + err


def test_existing_hybrid_never_invokes_cross_encoder(dense_env,secured,monkeypatch):
    from askpolicy.retrieval.hybrid import search_role_hybrid
    def forbidden(*args,**kwargs):
        pytest.fail("production hybrid activated reranking")
    monkeypatch.setattr(module.CrossEncoder,"score_pairs",forbidden)
    bundle = dense_env.bundle
    hits = search_role_hybrid(dense_env.bm25,bundle,secured.authenticate(secured.sign(),"employee"),
                              query="travel",expected_snapshot_id=bundle.snapshot.candidate_snapshot_id,
                              validity_date=dense_env.options["validity_date"])
    assert len(hits) == 20
