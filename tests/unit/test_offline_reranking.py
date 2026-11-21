"""Deterministic reranking contracts using the existing verified-loader doubles."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unit.test_evidence_hydration import release
from integration.test_embedding_revision import dense_env, DAY
from unit.test_cross_encoder_checkpoint import local
from security.test_auth import secured


@pytest.fixture
def session(release, secured, local):
    from askpolicy.retrieval import offline_reranking as module
    root, handle = release
    context = secured.authenticate(secured.sign(roles=["employee","hr"]), "hr")
    hits = handle.search(context,query="congés registre",validity_date=DAY)
    encoder = module.CrossEncoder(local.path)
    def run(candidates=hits, **changes):
        return module.rerank_published_hits(
            changes.get("handle",handle),changes.get("context",context),changes.get("query","question exacte"),
            candidates,release_root=root,validity_date=changes.get("validity_date",DAY),
            encoder=changes.get("encoder",encoder),
        )
    return SimpleNamespace(root=root,handle=handle,context=context,hits=hits,encoder=encoder,
                           local=local,module=module,run=run)


@pytest.mark.parametrize("count",[0,1,20])
def test_exact_candidate_set_scores_and_immutable_output(session,count):
    selected = session.hits[:count]
    result = session.run(selected)
    assert len(result.hits) == count
    assert {h.original for h in result.hits} == set(selected)
    assert all(h.release_id == session.handle.release_id and 1 <= h.token_count <= 512 for h in result.hits)
    assert result.hits == tuple(sorted(result.hits,key=lambda h:(-h.score,h.child.child_id)))
    with pytest.raises(FrozenInstanceError):
        result.hits = ()
    if result.hits:
        with pytest.raises(FrozenInstanceError):
            result.hits[0].score = 0.0
    assert not session.local.model.calls if count == 0 else sum(session.local.model.calls) == count
    assert result.timings.total_seconds >= sum((result.timings.hydration_seconds,result.timings.token_validation_seconds,
                                               result.timings.scoring_seconds,result.timings.ranking_seconds))


def test_input_order_pairs_exact_text_and_equal_logit_tie(session,monkeypatch):
    selected = tuple(reversed(session.hits))
    passages = session.module.hydrate_published_hits(session.handle,session.context,selected,
                                                    release_root=session.root,validity_date=DAY)
    seen = []
    original = session.encoder.score_pairs
    def capture(pairs):
        seen.extend(pairs)
        return original(pairs)
    monkeypatch.setattr(session.encoder,"score_pairs",capture)
    result = session.run(selected,query="  question exacte non modifiée  ")
    assert seen == [("  question exacte non modifiée  ",p.text) for p in passages]
    assert [h.child.child_id for h in result.hits] == sorted(h.child.child_id for h in selected)
    assert not any(p.text in repr(result) for p in passages)
    assert all(call[2]["truncation"] is False for call in session.local.tokenizer.calls)


def test_descending_negative_logits_no_secondary_rank_criterion(session,monkeypatch):
    scores = tuple(-float(i % 3 + 1) for i in range(20))
    def score(pairs):
        counts = session.encoder._tokens(pairs)[1]
        return session.module.PairScores(scores,counts)
    monkeypatch.setattr(session.encoder,"score_pairs",score)
    result = session.run()
    expected = sorted(zip(session.hits,scores),key=lambda pair:(-pair[1],pair[0].child.child_id))
    assert [(h.original,h.score) for h in result.hits] == expected


@pytest.mark.parametrize("defect",["over20","duplicate","ids","list","query"])
def test_malformed_input_never_reaches_forward(session,defect):
    hits = session.hits
    changes = {}
    if defect == "over20":
        hits += hits[:1]
    elif defect == "duplicate":
        hits = hits[:1]*2
    elif defect == "ids":
        hits = (hits[0].child.child_id,)
    elif defect == "list":
        hits = list(hits)
    else:
        changes["query"] = " "
    with pytest.raises(session.module.OfflineRerankError):
        session.run(hits,**changes)
    assert not session.local.model.calls


@pytest.mark.parametrize("defect",["nan","inf","short","long","token_count","scalar_type"])
def test_no_partial_or_fallback_on_bad_scores(session,monkeypatch,defect):
    def score(pairs):
        counts = session.encoder._tokens(pairs)[1]
        scores = [1.0]*len(pairs)
        if defect == "nan":
            scores[-1] = float("nan")
        elif defect == "inf":
            scores[-1] = float("inf")
        elif defect == "short":
            scores.pop()
        elif defect == "long":
            scores.append(0.0)
        elif defect == "scalar_type":
            scores[-1] = True
        else:
            counts = tuple(n+1 for n in counts)
        return session.module.PairScores(tuple(scores),counts)
    monkeypatch.setattr(session.encoder,"score_pairs",score)
    with pytest.raises(session.module.OfflineRerankError,match="^OFFLINE_RERANK_FAILED$"):
        session.run()


def test_late_overflow_rejects_all_candidates_before_forward(session):
    with pytest.raises(session.module.OfflineRerankError,match="^TOKEN_LIMIT$"):
        session.run(query=" ".join(["question"]*512))
    assert not session.local.model.calls
    assert all(call[2]["truncation"] is False for call in session.local.tokenizer.calls)


def test_last_candidate_batch_overflow_prevents_all_forward_calls(session,monkeypatch):
    original = session.encoder._tokens
    checked = []
    def late_overflow(pairs):
        checked.append(len(pairs))
        if len(checked) == 5:
            raise session.module.CrossEncoderError("PAIR_TOO_LONG")
        return original(pairs)
    monkeypatch.setattr(session.encoder,"_tokens",late_overflow)
    with pytest.raises(session.module.OfflineRerankError,match="^TOKEN_LIMIT$"):
        session.run()
    assert checked == [4,4,4,4,4] and not session.local.model.calls


@pytest.mark.parametrize("defect",["reorder","short","identity","span","role","release"])
def test_hydrated_identity_mismatch_prevents_scoring(session,monkeypatch,defect):
    original = session.module.hydrate_published_hits
    def corrupt(*args,**kwargs):
        rows = original(*args,**kwargs)
        if defect == "reorder":
            return tuple(reversed(rows))
        if defect == "short":
            return rows[:-1]
        changes = {"identity":{"child_id":"fabricated"},"span":{"end":rows[0].end+1},
                   "role":{"active_role":"employee"},"release":{"release_id":"0"*64}}[defect]
        return (replace(rows[0],**changes),*rows[1:])
    monkeypatch.setattr(session.module,"hydrate_published_hits",corrupt)
    with pytest.raises(session.module.OfflineRerankError):
        session.run()
    assert not session.local.model.calls


@pytest.mark.parametrize("defect",["revision","fingerprint","type","missing_model","training"])
def test_wrong_or_incomplete_encoder_rejected(session,defect):
    encoder = session.encoder
    if defect == "type":
        encoder = SimpleNamespace(score_pairs=encoder.score_pairs)
    elif defect == "missing_model":
        encoder._model = None
    elif defect == "training":
        encoder._model.training = True
    else:
        setattr(encoder,"revision" if defect == "revision" else "artifact_fingerprint","unapproved")
    with pytest.raises(session.module.OfflineRerankError):
        session.run(encoder=encoder)
    assert not session.local.model.calls


def test_missing_checkpoint_fails_before_model_load(session):
    (session.local.path / "model.safetensors").unlink()
    before = list(session.local.loads)
    with pytest.raises(session.module.CrossEncoderError):
        session.module.CrossEncoder(session.local.path)
    assert session.local.loads == before and not session.local.model.calls


def test_engine_error_has_no_text_or_chained_traceback(session,monkeypatch,caplog):
    marker = "protected-question-and-passage-marker"
    def fail(pairs):
        raise ValueError(marker)
    monkeypatch.setattr(session.encoder,"score_pairs",fail)
    with pytest.raises(session.module.OfflineRerankError) as raised:
        session.run(query=marker)
    formatted = "".join(traceback.format_exception(raised.type,raised.value,raised.tb))
    assert marker not in formatted + caplog.text
    assert raised.value.__context__ is None
