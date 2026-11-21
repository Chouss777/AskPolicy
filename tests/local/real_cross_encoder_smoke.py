"""Explicit real-checkpoint CPU smoke. Synthetic pairs carry no quality labels."""

import json
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
sys.path.insert(0,str(ROOT / "tests"))
from unit.test_e5_prefix import _working_set_mib
from askpolicy.retrieval.cross_encoder import (
    CrossEncoder, CrossEncoderError, MODEL_REVISION, verify_checkpoint,
)


def test_real_cross_encoder_smoke():
    directory = Path(os.environ["ASKPOLICY_RERANKER_DIR"])
    torch.set_num_threads(4)
    pairs = [
        ("Quel est le plafond d’hébergement ?", "Le plafond synthétique est de 850 MAD par nuit."),
        ("Quel est le plafond d’hébergement ?", "La cafétéria propose du thé."),
        ("How do I request planned leave?", "Send the requested dates to human resources."),
        ("How do I request planned leave?", "The office printer uses blue paper."),
        ("كيف أطلب العمل عن بعد؟", "قدّم طلب العمل عن بعد إلى مديرك."),
        ("كيف أطلب العمل عن بعد؟", "تتوفر مياه الشرب في قاعة الاستراحة."),
    ]
    calls = []
    def no_network(*args,**kwargs):
        raise AssertionError("Unexpected network access")
    with patch.object(socket.socket,"connect",no_network), patch.object(socket,"getaddrinfo",no_network):
        start = perf_counter()
        encoder = CrossEncoder(directory)
        loading_seconds = perf_counter() - start
        assert encoder._model.config.architectures == ["XLMRobertaForSequenceClassification"]
        assert type(encoder._model).__name__ == "XLMRobertaForSequenceClassification"
        assert not encoder._model.training
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in encoder._model.parameters())
        forward = encoder._model.forward
        def observe(**batch):
            calls.append(len(batch["input_ids"]))
            return forward(**batch)
        with patch.object(encoder._model,"forward",observe):
            start = perf_counter()
            first = encoder.score_pairs(pairs)
            first_scoring_seconds = perf_counter() - start
            assert calls == [4,2]
            warm = []
            for _ in range(3):
                start = perf_counter()
                repeated = encoder.score_pairs(pairs)
                warm.append(perf_counter() - start)
                torch.testing.assert_close(torch.tensor(repeated.scores),torch.tensor(first.scores),rtol=1e-5,atol=1e-6)
            for size in (1,2):
                encoder.batch_size = size
                repeated = encoder.score_pairs(pairs)
                torch.testing.assert_close(torch.tensor(repeated.scores),torch.tensor(first.scores),rtol=1e-5,atol=1e-6)
            encoder.batch_size = 4
        assert len(first.scores) == len(pairs) == 6 and all(torch.isfinite(torch.tensor(first.scores)))
        options = dict(add_special_tokens=True,truncation=False,verbose=False)
        base = len(encoder._tokenizer("test","test",**options)["input_ids"])
        passage = " ".join(["test"] * (513-base))
        assert len(encoder._tokenizer("test",passage,**options)["input_ids"]) == 512
        boundary = encoder.score_pairs([("test",passage)])
        assert boundary.token_counts == (512,)
        overflow = passage + " test"
        assert len(encoder._tokenizer("test",overflow,**options)["input_ids"]) == 513
        with patch.object(encoder._model,"forward",side_effect=AssertionError("Overflow reached inference")):
            try:
                encoder.score_pairs([("test",overflow)])
            except CrossEncoderError as error:
                assert error.code == "PAIR_TOO_LONG"
            else:
                raise AssertionError("Missing overflow rejection")
        with TemporaryDirectory(prefix="askpolicy-reranker-smoke-") as temporary:
            test_dir = Path(temporary) / MODEL_REVISION
            test_dir.mkdir()
            try:
                CrossEncoder(test_dir)
            except CrossEncoderError as error:
                assert error.code == "MODEL_FILES_MISSING"
            else:
                raise AssertionError("Missing artifact accepted")
            for name in verify_checkpoint(directory):
                if name == "config.json":
                    raw = (directory / name).read_bytes()
                    changed = raw.replace(b"xlm-roberta",b"xlm-robertb",1)
                    assert changed != raw and len(changed) == len(raw)
                    (test_dir / name).write_bytes(changed)
                else:
                    os.link(directory / name,test_dir / name)
            try:
                CrossEncoder(test_dir)
            except CrossEncoderError as error:
                assert error.code == "MODEL_ARTIFACT_HASH_MISMATCH"
            else:
                raise AssertionError("Modified artifact accepted")
        hashes = verify_checkpoint(directory)
    print(json.dumps(dict(revision=MODEL_REVISION,architecture=type(encoder._model).__name__,
        artifact_fingerprint=encoder.artifact_fingerprint,device="cpu",dtype="float32",pair_count=6,
        languages=["fr","en","ar"],scores=first.scores,token_counts=first.token_counts,
        loading_seconds=loading_seconds,first_scoring_seconds=first_scoring_seconds,
        warm_batch_seconds=warm,repeat_rtol=1e-5,repeat_atol=1e-6,batch_sizes_tested=[1,2,4],
        boundary_accepted=512,boundary_rejected=513,network_requests=0,
        memory=_working_set_mib(),artifacts=hashes),sort_keys=True))
