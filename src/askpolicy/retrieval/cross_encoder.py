"""Checksum-verified local Cross-Encoder scoring for the Step 103 CPU smoke."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
import stat
from threading import Lock
from types import MappingProxyType

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MAX_PAIR_TOKENS = 512
ARTIFACTS = MappingProxyType({
    "model.safetensors": (470592698, "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4"),
    "tokenizer.json": (17082660, "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"),
    "config.json": (891, "cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77"),
    "tokenizer_config.json": (435, "e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e"),
    "special_tokens_map.json": (239, "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d"),
})
ARTIFACT_FINGERPRINT = sha256(json.dumps(
    dict(ARTIFACTS), sort_keys=True, separators=(",", ":"),
).encode("ascii")).hexdigest()


class CrossEncoderError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    for item in (*reversed(path.parents), path):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
                raise CrossEncoderError("UNSAFE_MODEL_PATH")
    return path


def verify_checkpoint(directory: Path, *, revision: str = MODEL_REVISION) -> dict:
    """Verify all allowed bytes before Transformers sees any local artifact."""
    try:
        directory = safe_path(directory)
        if revision != MODEL_REVISION or directory.name != MODEL_REVISION:
            raise CrossEncoderError("MODEL_REVISION_MISMATCH")
        if not directory.is_dir():
            raise CrossEncoderError("MODEL_FILES_MISSING")
        entries = {entry.name for entry in directory.iterdir()}
        if not set(ARTIFACTS) <= entries:
            raise CrossEncoderError("MODEL_FILES_MISSING")
        if entries != set(ARTIFACTS):
            raise CrossEncoderError("UNEXPECTED_MODEL_ARTIFACT")
        records = {}
        for name, (size, expected) in ARTIFACTS.items():
            path = safe_path(directory / name)
            if not path.is_file() or path.stat().st_size != size:
                raise CrossEncoderError("MODEL_ARTIFACT_SIZE_MISMATCH")
            with path.open("rb") as stream:
                actual = file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise CrossEncoderError("MODEL_ARTIFACT_HASH_MISMATCH")
            records[name] = dict(bytes=size, sha256=actual)
        return records
    except CrossEncoderError:
        raise
    except OSError:
        raise CrossEncoderError("MODEL_FILES_UNREADABLE") from None


@dataclass(frozen=True, slots=True)
class PairScores:
    scores: tuple[float, ...]
    token_counts: tuple[int, ...]


class CrossEncoder:
    """One local CPU model, serialized tokenizer access, no inference cache.

    Batches contain one to four explicit pairs. All pairs pass token validation
    before the first forward pass. No ranking, splitting or aggregation occurs.
    """

    model_id = MODEL_ID
    revision = MODEL_REVISION
    artifact_fingerprint = ARTIFACT_FINGERPRINT

    def __init__(self, directory: Path, *, revision: str = MODEL_REVISION, batch_size: int = 4):
        if type(batch_size) is not int or not 1 <= batch_size <= 4:
            raise CrossEncoderError("INVALID_BATCH_SIZE")
        verify_checkpoint(directory, revision=revision)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                directory, use_fast=True, local_files_only=True, trust_remote_code=False,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            )
            config = model.config
            if (
                not tokenizer.is_fast or tokenizer.model_max_length != MAX_PAIR_TOKENS
                or config.model_type != "xlm-roberta"
                or config.architectures != ["XLMRobertaForSequenceClassification"]
                or config.hidden_size != 384 or config.num_hidden_layers != 12 or config.num_labels != 1
                or config.max_position_embeddings != 514
            ):
                raise CrossEncoderError("UNEXPECTED_MODEL_CONFIGURATION")
            model.to(device="cpu", dtype=torch.float32)
            model.eval()
            if not all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()):
                raise CrossEncoderError("INVALID_MODEL_DEVICE_OR_DTYPE")
        except CrossEncoderError:
            raise
        except Exception:
            raise CrossEncoderError("MODEL_LOAD_FAILED") from None
        self._tokenizer = tokenizer
        self._model = model
        self.batch_size = batch_size
        self._lock = Lock()

    def _tokens(self, pairs):
        encoded = self._tokenizer(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            add_special_tokens=True, padding=False, truncation=False,
            return_attention_mask=True, verbose=False,
        )
        counts = tuple(len(ids) for ids in encoded["input_ids"])
        if len(counts) != len(pairs) or not all(counts):
            raise CrossEncoderError("INVALID_TOKENIZER_OUTPUT")
        if max(counts) > MAX_PAIR_TOKENS:
            raise CrossEncoderError("PAIR_TOO_LONG")
        return encoded, counts

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> PairScores:
        if (
            not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)) or not pairs
            or any(type(pair) not in (tuple, list) or len(pair) != 2
                   or any(type(text) is not str or not text.strip() for text in pair) for pair in pairs)
        ):
            raise CrossEncoderError("INVALID_PAIRS")
        # Snapshot caller inputs so later mutation cannot alter the checked pairs.
        pairs = tuple(tuple(pair) for pair in pairs)
        try:
            with self._lock:
                batch_size = self.batch_size
                if type(batch_size) is not int or not 1 <= batch_size <= 4:
                    raise CrossEncoderError("INVALID_BATCH_SIZE")
                counts = []
                for start in range(0, len(pairs), batch_size):
                    _, measured = self._tokens(pairs[start:start + batch_size])
                    counts.extend(measured)
                scores = []
                for start in range(0, len(pairs), batch_size):
                    encoded, measured = self._tokens(pairs[start:start + batch_size])
                    if measured != tuple(counts[start:start + batch_size]):
                        raise CrossEncoderError("TOKEN_COUNT_MISMATCH")
                    batch = self._tokenizer.pad(encoded, padding=True, return_tensors="pt", verbose=False)
                    batch = {name: value.to("cpu") for name, value in batch.items()}
                    mask = batch["attention_mask"]
                    if (batch["input_ids"].shape != (len(measured), max(measured))
                        or mask.shape != batch["input_ids"].shape
                        or tuple(mask.sum(dim=1).tolist()) != measured):
                        raise CrossEncoderError("INVALID_TOKENIZER_OUTPUT")
                    with torch.inference_mode():
                        logits = self._model(**batch).logits
                    if (logits.shape != (len(measured), 1) or logits.dtype != torch.float32
                        or logits.device.type != "cpu" or not torch.isfinite(logits).all()):
                        raise CrossEncoderError("INVALID_LOGITS")
                    scores.extend(float(score) for score in logits[:, 0].tolist())
                return PairScores(tuple(scores), tuple(counts))
        except CrossEncoderError:
            raise
        except Exception:
            raise CrossEncoderError("SCORING_FAILED") from None
