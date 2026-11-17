"""Deterministic units and an explicit local smoke runner for the pinned model.

Run the real check with Python and this file's path, followed by the verified
revision directory. Normal pytest collection runs the model-free fixtures.
"""

from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


@pytest.fixture(scope="module")
def encoder_module():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        from askpolicy.retrieval import e5_encoder

        yield e5_encoder


class FixtureTokenizer:
    is_fast = True
    model_max_length = 512

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, **options):
        self.calls.append((inputs, options))
        tokens = [[1] * (len(text.split()) + 2) for text in inputs]
        return {"input_ids": tokens, "attention_mask": tokens}

    def pad(self, encoded, *, padding, return_tensors, verbose):
        assert padding and return_tensors == "pt" and not verbose
        size = max(map(len, encoded["input_ids"]))
        return {
            key: torch.tensor([row + [0] * (size - len(row)) for row in rows])
            for key, rows in encoded.items()
        }


class FixtureModel:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="bert", hidden_size=384, num_hidden_layers=12, max_position_embeddings=512,
        )
        self.device_options = None
        self.evaluating = False
        self.calls = 0
        self.hidden = None

    def to(self, **options):
        self.device_options = options
        return self

    def eval(self):
        self.evaluating = True
        return self

    def __call__(self, *, input_ids, attention_mask):
        assert self.evaluating and torch.is_inference_mode_enabled()
        assert input_ids.device.type == attention_mask.device.type == "cpu"
        self.calls += 1
        hidden = self.hidden
        if hidden is None:
            hidden = torch.ones((*input_ids.shape, 384), dtype=torch.float64)
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def local(encoder_module, tmp_path, monkeypatch):
    module = encoder_module
    path = tmp_path / REVISION
    path.mkdir()
    hashes = {}
    for name in module._ARTIFACT_SHA256:
        blob = name.encode("utf-8")
        (path / name).write_bytes(blob)
        hashes[name] = sha256(blob).hexdigest()
    monkeypatch.setattr(module, "_ARTIFACT_SHA256", hashes)
    tokenizer, model = FixtureTokenizer(), FixtureModel()
    loads = []

    def load_tokenizer(snapshot, **options):
        loads.append(("tokenizer", snapshot, options))
        return tokenizer

    def load_model(snapshot, **options):
        loads.append(("model", snapshot, options))
        return model

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(module.AutoModel, "from_pretrained", load_model)
    return SimpleNamespace(module=module, path=path, tokenizer=tokenizer, model=model, loads=loads)


def test_verified_load_is_local_safetensors_only_and_cpu(local):
    encoder = local.module.E5Encoder(local.path)
    assert encoder.model_id == "intfloat/multilingual-e5-small" and encoder.revision == REVISION
    assert local.loads == [
        ("tokenizer", local.path, {"use_fast": True, "local_files_only": True, "trust_remote_code": False}),
        ("model", local.path, {
            "local_files_only": True, "trust_remote_code": False, "use_safetensors": True, "add_pooling_layer": False,
        }),
    ]
    assert local.model.device_options == {"device": "cpu", "dtype": torch.float32}
    assert local.model.evaluating


@pytest.mark.parametrize("prefix, method", [("query: ", "encode_queries"), ("passage: ", "encode_passages")])
def test_prefix_precedes_tokenization_for_french_english_and_arabic(local, prefix, method):
    inputs = ["travail à distance", "Remote work policy", "سياسة العمل عن بعد"]
    encoder = local.module.E5Encoder(local.path)
    result = getattr(encoder, method)(inputs)
    captured, options = local.tokenizer.calls[0]
    assert captured == [prefix + text for text in inputs]
    assert options == {
        "add_special_tokens": True, "padding": False, "truncation": False,
        "return_attention_mask": True, "verbose": False,
    }
    assert result.token_counts == tuple(len(text.split()) + 3 for text in inputs)
    assert result.vectors.shape == (3, 384) and result.vectors.dtype == np.float32
    assert np.isfinite(result.vectors).all()
    np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=1), 1, rtol=0, atol=0.001)
    assert {field.name for field in fields(result)} == {"vectors", "token_counts"}


@pytest.mark.parametrize("bad", [[], [""], [" "], [None], [True], "unbatched", b"bytes"])
def test_empty_or_invalid_input_never_reaches_tokenizer(local, bad):
    encoder = local.module.E5Encoder(local.path)
    with pytest.raises(local.module.E5EncodingError, match="^INVALID_INPUT$"):
        encoder.encode_queries(bad)
    assert not local.tokenizer.calls and local.model.calls == 0


def test_valid_token_limit_includes_prefix_and_special_tokens(local):
    encoder = local.module.E5Encoder(local.path)
    valid = " ".join(["word"] * 509)
    result = encoder.encode_passages([valid])
    assert result.token_counts == (512,)
    assert local.model.calls == 1


def test_overlength_is_rejected_before_forward_without_truncation(local):
    encoder = local.module.E5Encoder(local.path)
    overlength = " ".join(["word"] * 510)
    with pytest.raises(local.module.E5EncodingError, match="^INPUT_TOO_LONG$"):
        encoder.encode_queries([overlength])
    assert local.tokenizer.calls[0][1]["truncation"] is False
    assert local.model.calls == 0


@pytest.mark.parametrize("defect,code", [
    ("revision", "MODEL_REVISION_MISMATCH"), ("directory", "MODEL_REVISION_MISMATCH"),
    ("missing", "MODEL_FILES_MISSING"), ("checksum", "MODEL_ARTIFACT_MISMATCH"),
])
def test_wrong_revision_missing_files_and_tampering_are_denied_before_model_load(local, defect, code):
    path, revision = local.path, REVISION
    if defect == "revision":
        revision = "main"
    elif defect == "directory":
        path = path.with_name("main")
        local.path.rename(path)
    elif defect == "missing":
        (path / "model.safetensors").unlink()
    else:
        (path / "tokenizer.json").write_bytes(b"unapproved tokenizer")
    with pytest.raises(local.module.E5EncodingError, match=f"^{code}$"):
        local.module.E5Encoder(path, revision=revision)
    assert not local.loads


def test_unsupported_model_configuration_is_rejected(local):
    local.model.config.hidden_size = 768
    with pytest.raises(local.module.E5EncodingError, match="^UNSUPPORTED_MODEL_CONFIGURATION$"):
        local.module.E5Encoder(local.path)
    assert not local.model.evaluating


def test_attention_mask_mean_ignores_padded_hidden_states(local):
    encoder = local.module.E5Encoder(local.path)
    hidden = torch.zeros((2, 6, 384), dtype=torch.float64)
    hidden[0, :4, 0] = 3
    hidden[0, :4, 1] = 4
    hidden[0, 4:] = float("nan")
    hidden[1, :, 2] = 2
    local.model.hidden = hidden
    result = encoder.encode_queries(["short", "three words here"])
    assert result.token_counts == (4, 6)
    expected = np.zeros((2, 384), dtype=np.float32)
    expected[0, :2] = (0.6, 0.8)
    expected[1, 2] = 1
    np.testing.assert_allclose(result.vectors, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("dimension,value,code", [
    (383, 1.0, "INVALID_EMBEDDING_DIMENSION"),
    (384, 0.0, "ZERO_EMBEDDING_NORM"),
    (384, float("nan"), "NONFINITE_EMBEDDING"),
    (384, float("inf"), "NONFINITE_EMBEDDING"),
])
def test_invalid_embeddings_fail_closed(local, dimension, value, code):
    encoder = local.module.E5Encoder(local.path)
    local.model.hidden = torch.full((1, 4, dimension), value)
    with pytest.raises(local.module.E5EncodingError, match=f"^{code}$"):
        encoder.encode_queries(["short"])


def test_error_does_not_leak_text_tokens_or_secrets(local, monkeypatch, caplog, capsys):
    encoder = local.module.E5Encoder(local.path)
    sensitive = "synthetic-private-policy-and-secret"

    def broken_tokenizer(*args, **kwargs):
        raise ValueError(sensitive)

    monkeypatch.setattr(encoder, "_tokenizer", broken_tokenizer)
    with pytest.raises(local.module.E5EncodingError, match="^ENCODING_FAILED$") as raised:
        encoder.encode_passages([sensitive])
    assert raised.value.__suppress_context__
    out, err = capsys.readouterr()
    assert sensitive not in str(raised.value) + caplog.text + out + err


def _working_set_mib():
    import ctypes
    from ctypes import wintypes
    import sys

    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
            *[(name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged", "paged", "peak_nonpaged",
                "nonpaged", "pagefile", "peak_pagefile",
            )],
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    get_memory.restype = wintypes.BOOL
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        return None
    return {
        "working_set_mib": round(counters.working_set / 2**20, 2),
        "peak_working_set_mib": round(counters.peak_working_set / 2**20, 2),
    }


def run_provisioned_smoke(snapshot_dir: Path) -> dict:
    """Exercise the real pinned model, tokenizer, CPU forward pass, and token limit."""
    import socket
    import sys
    from statistics import median
    from time import perf_counter
    from unittest.mock import patch

    sys.path.insert(0, str(ROOT / "src"))
    from askpolicy.retrieval.e5_encoder import E5Encoder, E5EncodingError

    torch.set_num_threads(4)
    assert torch.version.cuda is None
    rtol, atol = 1e-5, 1e-6

    def unexpected_network(*args, **kwargs):
        raise AssertionError("The local smoke test attempted a network connection")

    with patch.object(socket.socket, "connect", unexpected_network), patch.object(socket, "getaddrinfo", unexpected_network):
        start = perf_counter()
        encoder = E5Encoder(snapshot_dir)
        load_seconds = perf_counter() - start
        start = perf_counter()
        query = encoder.encode_queries(["travail à distance"])
        passage = encoder.encode_passages(["Politique du travail à distance"])
        first_pair_seconds = perf_counter() - start
        query_norm = float(np.linalg.norm(query.vectors[0]))
        passage_norm = float(np.linalg.norm(passage.vectors[0]))
        for result, norm in ((query, query_norm), (passage, passage_norm)):
            assert result.vectors.dtype == np.float32 and result.vectors.shape == (1, 384)
            assert np.isfinite(result.vectors).all() and 0.999 <= norm <= 1.001
            assert len(result.token_counts) == 1 and 0 < result.token_counts[0] < 512

        warm_pair = []
        for _ in range(3):
            start = perf_counter()
            query_again = encoder.encode_queries(["travail à distance"])
            passage_again = encoder.encode_passages(["Politique du travail à distance"])
            warm_pair.append(perf_counter() - start)
            np.testing.assert_allclose(query_again.vectors, query.vectors, rtol=rtol, atol=atol)
            np.testing.assert_allclose(passage_again.vectors, passage.vectors, rtol=rtol, atol=atol)

        options = {"add_special_tokens": True, "truncation": False, "verbose": False}
        query_tokens = encoder._tokenizer("query: travail à distance", **options)["input_ids"]
        passage_tokens = encoder._tokenizer("passage: Politique du travail à distance", **options)["input_ids"]
        assert query.token_counts == (len(query_tokens),) and passage.token_counts == (len(passage_tokens),)
        base_ids = encoder._tokenizer("query: test", **options)["input_ids"]
        boundary_text = " ".join(["test"] * (513 - len(base_ids)))
        boundary = encoder.encode_queries([boundary_text])
        assert boundary.token_counts == (512,) and boundary.vectors.shape == (1, 384)
        try:
            encoder.encode_queries([boundary_text + " test"])
        except E5EncodingError as error:
            assert error.code == "INPUT_TOO_LONG"
        else:
            raise AssertionError("The real model overlength check did not reject its input")

    return {
        "model_id": encoder.model_id, "revision": encoder.revision, "device": "cpu",
        "torch_threads": torch.get_num_threads(), "shape_per_input": list(query.vectors.shape),
        "dtype": str(query.vectors.dtype), "all_finite": True,
        "query_tokens": query.token_counts[0], "passage_tokens": passage.token_counts[0],
        "query_l2_norm": query_norm, "passage_l2_norm": passage_norm,
        "token_limit": {"accepted": 512, "rejected": 513}, "repeat_rtol": rtol, "repeat_atol": atol,
        "load_seconds": load_seconds, "first_pair_seconds": first_pair_seconds,
        "median_warm_pair_seconds": median(warm_pair), "process_ram": _working_set_mib(),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the real offline E5 CPU smoke test")
    parser.add_argument("snapshot_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_provisioned_smoke(args.snapshot_dir), indent=2))
