"""Encode queries and passages with the approved local E5 checkpoint."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import file_digest, sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIMENSION = 384
MAX_INPUT_TOKENS = 512
_ARTIFACT_SHA256 = MappingProxyType({
    "config.json": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    "model.safetensors": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    "special_tokens_map.json": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
})


ARTIFACT_FINGERPRINT = sha256(json.dumps(
    dict(_ARTIFACT_SHA256), sort_keys=True, separators=(",", ":"),
).encode("ascii")).hexdigest()


class E5EncodingError(ValueError):
    """Controlled model or input failure that never includes source text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: NDArray[np.float32] = field(repr=False)
    token_counts: tuple[int, ...]


def _verify_snapshot(snapshot: Path, revision: str) -> None:
    if revision != MODEL_REVISION or snapshot.name != MODEL_REVISION:
        raise E5EncodingError("MODEL_REVISION_MISMATCH")
    try:
        for filename, expected in _ARTIFACT_SHA256.items():
            with (snapshot / filename).open("rb") as artifact:
                if file_digest(artifact, "sha256").hexdigest() != expected:
                    raise E5EncodingError("MODEL_ARTIFACT_MISMATCH")
    except FileNotFoundError:
        raise E5EncodingError("MODEL_FILES_MISSING") from None
    except OSError:
        raise E5EncodingError("MODEL_FILES_UNREADABLE") from None


def _pool_and_normalize(hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[-1] != EMBEDDING_DIMENSION:
        raise E5EncodingError("INVALID_EMBEDDING_DIMENSION")
    if hidden.shape[:2] != attention.shape:
        raise E5EncodingError("INVALID_ENCODER_OUTPUT")
    if hidden.device.type != "cpu":
        raise E5EncodingError("NON_CPU_ENCODER_OUTPUT")

    included = attention.to(torch.bool)
    lengths = included.sum(dim=1, keepdim=True)
    if not torch.all(lengths > 0):
        raise E5EncodingError("INVALID_ATTENTION_MASK")
    pooled = hidden.to(torch.float32).masked_fill(~included.unsqueeze(-1), 0.0).sum(dim=1) / lengths
    if not torch.isfinite(pooled).all():
        raise E5EncodingError("NONFINITE_EMBEDDING")
    norms = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise E5EncodingError("NONFINITE_EMBEDDING")
    if not torch.all(norms > 0):
        raise E5EncodingError("ZERO_EMBEDDING_NORM")
    normalized = pooled / norms
    unit_norms = torch.linalg.vector_norm(normalized, dim=1)
    if not torch.all((unit_norms >= 0.999) & (unit_norms <= 1.001)):
        raise E5EncodingError("INVALID_NORMALIZED_NORM")
    return normalized


class E5Encoder:
    """Load a checksum-verified revision directory and encode batches on CPU.

    The tokenizer counts prefixed text with special tokens before inference.
    The dense index builder measures original source slices with this same
    tokenizer before submitting bounded batches for passage encoding.
    """

    model_id = MODEL_ID
    revision = MODEL_REVISION
    dimension = EMBEDDING_DIMENSION
    max_input_tokens = MAX_INPUT_TOKENS

    def __init__(self, snapshot_dir: Path, *, revision: str = MODEL_REVISION) -> None:
        snapshot = Path(snapshot_dir)
        _verify_snapshot(snapshot, revision)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot, use_fast=True, local_files_only=True, trust_remote_code=False,
            )
            model = AutoModel.from_pretrained(
                snapshot, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, add_pooling_layer=False,
            )
            if (
                not tokenizer.is_fast
                or tokenizer.model_max_length != MAX_INPUT_TOKENS
                or model.config.model_type != "bert"
                or model.config.hidden_size != EMBEDDING_DIMENSION
                or model.config.num_hidden_layers != 12
                or model.config.max_position_embeddings != MAX_INPUT_TOKENS
            ):
                raise E5EncodingError("UNSUPPORTED_MODEL_CONFIGURATION")
            model.to(device="cpu", dtype=torch.float32)
            model.eval()
        except E5EncodingError:
            raise
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            raise E5EncodingError("MODEL_LOAD_FAILED") from None
        self._tokenizer = tokenizer
        self._model = model
        self.artifact_fingerprint = ARTIFACT_FINGERPRINT

    def passage_token_count(self, text: str) -> int:
        """Measure an exact source slice with the same prefix and special tokens as inference."""
        try:
            ids = self._tokenizer(
                "passage: " + text, add_special_tokens=True, padding=False,
                truncation=False, verbose=False,
            )["input_ids"]
            if not ids:
                raise E5EncodingError("INVALID_TOKENIZER_OUTPUT")
            return len(ids)
        except (KeyError, RuntimeError, TypeError, ValueError):
            raise E5EncodingError("INVALID_TOKENIZER_OUTPUT") from None

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self._encode(texts, prefix="query: ")

    def encode_passages(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self._encode(texts, prefix="passage: ")

    def _encode(self, texts: Sequence[str], *, prefix: Literal["query: ", "passage: "]) -> EmbeddingBatch:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes, bytearray)) or not texts:
            raise E5EncodingError("INVALID_INPUT")
        if any(type(text) is not str or not text.strip() for text in texts):
            raise E5EncodingError("INVALID_INPUT")
        inputs = [prefix + text for text in texts]

        try:
            encoded = self._tokenizer(
                inputs, add_special_tokens=True, padding=False, truncation=False,
                return_attention_mask=True, verbose=False,
            )
            counts = tuple(len(tokens) for tokens in encoded["input_ids"])
            if len(counts) != len(inputs) or not all(counts):
                raise E5EncodingError("INVALID_TOKENIZER_OUTPUT")
            if max(counts) > MAX_INPUT_TOKENS:
                raise E5EncodingError("INPUT_TOO_LONG")

            batch = self._tokenizer.pad(encoded, padding=True, return_tensors="pt", verbose=False)
            ids = batch["input_ids"].to("cpu")
            mask = batch["attention_mask"].to("cpu")
            if (
                ids.shape != (len(counts), max(counts))
                or mask.shape != ids.shape
                or tuple(mask.sum(dim=1).tolist()) != counts
            ):
                raise E5EncodingError("INVALID_TOKENIZER_OUTPUT")
            with torch.inference_mode():
                output = self._model(input_ids=ids, attention_mask=mask)
                vectors = _pool_and_normalize(output.last_hidden_state, mask)
                return EmbeddingBatch(vectors.contiguous().numpy(), counts)
        except E5EncodingError:
            raise
        except (AttributeError, KeyError, OverflowError, RuntimeError, TypeError, ValueError):
            raise E5EncodingError("ENCODING_FAILED") from None
