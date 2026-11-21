"""Partition one validated source child using absolute Unicode character offsets."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from hashlib import sha256
import json

from .bm25_core import EvidenceChunk
from .e5_encoder import MAX_INPUT_TOKENS


SPLIT_VERSION = "prefixed-count-char-bisection-whitespace-v1"


@dataclass(frozen=True, slots=True)
class Segment:
    segment_id: str
    child_id: str
    start: int
    end: int
    token_count: int


@dataclass(frozen=True, slots=True)
class Passage:
    segment: Segment
    text: str = field(repr=False)


def split_passage(chunk: EvidenceChunk, count_tokens: Callable[[str], int]) -> Iterator[Passage]:
    """Count original slices directly, avoiding decoded tokens or normalized offsets.

    Bisection finds a fitting cut, without requiring token counts to be monotonic.
    Prefer the last whitespace boundary in the latter half of that slice.
    Every emitted slice is recounted, including its prefix and special tokens.
    """
    text = chunk.text
    if type(chunk.start) is not int or chunk.start < 0 or not text.strip():
        raise ValueError("INVALID_SOURCE_SPAN")
    cursor = 0
    while cursor < len(text):
        end = len(text)
        count = count_tokens(text[cursor:end])
        if count > MAX_INPUT_TOKENS:
            low, high = cursor, end
            while high - low > 1:
                mid = (low + high) // 2
                if count_tokens(text[cursor:mid]) <= MAX_INPUT_TOKENS:
                    low = mid
                else:
                    high = mid
            end = low
            for boundary in range(low, cursor + (low - cursor) // 2, -1):
                if text[boundary - 1].isspace() and text[cursor:boundary].strip():
                    if count_tokens(text[cursor:boundary]) <= MAX_INPUT_TOKENS:
                        end = boundary
                        break
            # Attach trailing whitespace to its last substantive segment.
            if not text[end:].strip():
                end = len(text)
            count = count_tokens(text[cursor:end])
        piece = text[cursor:end]
        if not piece.strip() or type(count) is not int or not 0 < count <= MAX_INPUT_TOKENS:
            raise ValueError("INVALID_SEGMENT_BOUNDARY")
        start_offset, end_offset = chunk.start + cursor, chunk.start + end
        identity = [SPLIT_VERSION, chunk.child_id, start_offset, end_offset, sha256(piece.encode()).hexdigest()]
        digest = sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        yield Passage(Segment(digest, chunk.child_id, start_offset, end_offset, count), piece)
        cursor = end
