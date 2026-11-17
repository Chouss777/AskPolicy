"""Shared deterministic BM25S scoring and validated-draft evidence records."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
import json
from math import isfinite
import sqlite3
import unicodedata

import bm25s

from ..ingestion.metadata import PolicyMetadata


class BM25CoreError(ValueError):
    """Content-free failure of shared lexical processing."""


@dataclass(frozen=True, slots=True)
class BM25Options:
    method: str = "lucene"
    idf_method: str = "lucene"
    k1: float = 1.5
    b: float = 0.75
    dtype: str = "float32"
    int_dtype: str = "int32"
    backend: str = "numpy"
    csc_backend: str = "numpy"


@dataclass(frozen=True, slots=True)
class TokenizationOptions:
    normalization: str = "NFC"
    unicode_version: str = unicodedata.unidata_version
    lowercase: bool = True
    pattern: str = r"(?u)\b\w\w+\b"
    stopwords: tuple[str, ...] = ()
    stemmer: None = None


BM25_OPTIONS = BM25Options()
TOKENIZATION = TokenizationOptions()


@dataclass(frozen=True, slots=True)
class DraftDocument:
    metadata: PolicyMetadata = field(repr=False)
    source_path: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class SectionIdentity:
    document_id: str
    parent_id: str
    heading: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    child_id: str
    section: SectionIdentity
    document: DraftDocument
    start: int
    text: str = field(repr=False)


def read_draft_documents(connection: sqlite3.Connection) -> dict[str, DraftDocument]:
    """Read typed document metadata from a private, already validated draft."""
    fields = tuple(PolicyMetadata.model_fields)
    rows = connection.execute(
        "SELECT " + ", ".join((*fields, "source_path", "source_sha256"))
        + " FROM main.draft_documents ORDER BY document_id",
    )
    documents = {}
    for row in rows:
        metadata = dict(zip(fields, row[:-2], strict=True))
        try:
            metadata["allowed_roles"] = tuple(json.loads(metadata["allowed_roles"]))
            for name in ("effective_date", "expires_at"):
                if metadata[name] is not None:
                    metadata[name] = date.fromisoformat(metadata[name])
            validated = PolicyMetadata.model_validate(metadata)
        except (TypeError, ValueError):
            raise BM25CoreError("INVALID_STORED_METADATA") from None
        documents[validated.document_id] = DraftDocument(validated, row[-2], row[-1])
    return documents


def read_draft_chunks(
    connection: sqlite3.Connection, allowed_documents: Mapping[str, DraftDocument],
) -> tuple[EvidenceChunk, ...]:
    """Select only the supplied authorized document IDs before loading text."""
    if not allowed_documents:
        return ()
    parameters = tuple(sorted(allowed_documents))
    placeholders = ", ".join("?" for _ in parameters)
    rows = connection.execute(
        "SELECT c.child_id, c.document_id, c.parent_id, p.heading, c.start, c.text "
        "FROM main.draft_children AS c JOIN main.draft_parents AS p "
        "ON p.document_id = c.document_id AND p.parent_id = c.parent_id "
        f"WHERE c.document_id IN ({placeholders}) ORDER BY c.document_id, c.start, c.child_id",
        parameters,
    )
    return tuple(
        EvidenceChunk(child_id, SectionIdentity(document_id, parent_id, heading),
                      allowed_documents[document_id], start, text)
        for child_id, document_id, parent_id, heading, start, text in rows
    )


def chunk_order(chunk: EvidenceChunk) -> tuple[str, int, str]:
    return chunk.section.document_id, chunk.start, chunk.child_id


def tokenize(texts: Sequence[str]) -> list[list[str]]:
    return bm25s.tokenize(
        [unicodedata.normalize(TOKENIZATION.normalization, text) for text in texts],
        lower=TOKENIZATION.lowercase, token_pattern=TOKENIZATION.pattern,
        stopwords=list(TOKENIZATION.stopwords), stemmer=TOKENIZATION.stemmer,
        return_ids=False, show_progress=False,
    )


def build_text_index(texts: Sequence[str]) -> bm25s.BM25 | None:
    """Index preselected texts in order. An empty scope has no BM25S engine."""
    if not texts:
        return None
    tokens = tokenize(texts)
    if not any(tokens):
        raise BM25CoreError("NO_INDEXABLE_TOKENS")
    index = bm25s.BM25(**asdict(BM25_OPTIONS))
    index.index(tokens, create_empty_token=False, show_progress=False)
    return index


def ranked_positions(
    index: bm25s.BM25 | None, child_ids: Sequence[str], query: str, *, limit: int,
) -> tuple[tuple[int, float], ...]:
    if index is None:
        if child_ids:
            raise BM25CoreError("INDEX_SCORE_MISMATCH")
        return ()
    tokens = tokenize([query])[0]
    if not tokens:
        return ()
    scores = index.get_scores(tokens)
    if len(scores) != len(child_ids):
        raise BM25CoreError("INDEX_SCORE_MISMATCH")
    ranked = [(position, float(score)) for position, score in enumerate(scores) if isfinite(score) and score > 0]
    ranked.sort(key=lambda hit: (-hit[1], child_ids[hit[0]]))
    return tuple(ranked[:limit])
