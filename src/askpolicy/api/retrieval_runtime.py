"""Trusted process configuration and serialized blocking retrieval execution.

Heavy model and storage imports remain lazy so authentication bootstrap performs
no corpus, index or model I/O.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
import os
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, ConfigDict, field_validator


class RetrievalUnavailable(ValueError):
    """Content-free retrieval availability failure."""


class QueryLimitError(ValueError):
    """Controlled user input limit failure."""


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    release_root: Path
    e5_dir: Path
    manifest_path: Path

    @field_validator("release_root", "e5_dir", "manifest_path", mode="before")
    @classmethod
    def absolute_path(cls, value):
        if type(value) is str and value.strip() == value and value:
            value = Path(value)
        if not isinstance(value, Path) or not value.is_absolute() or ".." in value.parts:
            raise ValueError("INVALID_RETRIEVAL_PATH")
        return value


def load_retrieval_config(environ: Mapping[str, str] | None = None) -> RetrievalConfig | None:
    source = os.environ if environ is None else environ
    names = {"release_root":"ASKPOLICY_RELEASE_ROOT", "e5_dir":"ASKPOLICY_E5_DIR",
             "manifest_path":"ASKPOLICY_MANIFEST_PATH"}
    values = {field:source.get(name) for field,name in names.items()}
    if not any(value is not None for value in values.values()):
        return None
    return RetrievalConfig.model_validate(values)


def utc_today() -> date:
    return datetime.now(UTC).date()


class RetrievalRuntime:
    def __init__(self, config: RetrievalConfig | None, *, clock: Callable[[], date] = utc_today) -> None:
        self.config = config
        self.clock = clock
        self._encoder = None
        self._lock = Lock()

    def _load_encoder(self):
        from ..retrieval.e5_encoder import E5Encoder
        from ..storage.index_store import safe_path

        if self._encoder is None:
            self._encoder = E5Encoder(safe_path(self.config.e5_dir))
        return self._encoder

    def search(self, context, query: str):
        from .retrieval import SearchResponse, SearchResult

        try:
            # The lock also protects fast-tokenizer state and CPU inference.
            # No role, handle or user evidence is cached between requests.
            with self._lock:
                if type(self.config) is not RetrievalConfig:
                    raise RetrievalUnavailable("RETRIEVAL_UNAVAILABLE")
                day = self.clock()
                if type(day) is not date:
                    raise RetrievalUnavailable("RETRIEVAL_UNAVAILABLE")
                encoder = self._load_encoder()
                tokens = encoder._tokenizer(
                    "query: " + query, add_special_tokens=True, padding=False,
                    truncation=False, verbose=False,
                )["input_ids"]
                if len(tokens) > encoder.max_input_tokens:
                    raise QueryLimitError("QUERY_TOO_LONG")
                from ..storage.index_store import safe_path
                from ..storage.publication import load_active_snapshot
                from ..storage.snapshot import DraftSnapshotConfig

                manifest = safe_path(self.config.manifest_path).read_bytes()
                handle = load_active_snapshot(
                    safe_path(self.config.release_root), manifest=manifest,
                    snapshot_config=DraftSnapshotConfig(), validity_date=day, encoder=encoder,
                )
                if handle.status != "PUBLISHED":
                    raise RetrievalUnavailable("RETRIEVAL_UNAVAILABLE")
                hits = handle.search(context, query=query, validity_date=day)
                if any(h.active_role != context.active_role or
                       h.candidate_snapshot_id != handle.faiss.snapshot.candidate_snapshot_id for h in hits):
                    raise RetrievalUnavailable("RETRIEVAL_UNAVAILABLE")
                return SearchResponse(
                    release_id=handle.release_id, snapshot_id=handle.faiss.snapshot.candidate_snapshot_id,
                    active_role=context.active_role,
                    results=tuple(SearchResult(
                        child_id=h.child.child_id, parent_id=h.child.section.parent_id,
                        document_id=h.child.section.document_id, rrf_score=h.rrf_score,
                        bm25_rank=h.bm25_rank, faiss_rank=h.faiss_rank,
                    ) for h in hits),
                )
        except QueryLimitError:
            raise
        except Exception:
            raise RetrievalUnavailable("RETRIEVAL_UNAVAILABLE") from None
