"""Exact FAISS mathematics and source offsets, independent of model weights."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest


@pytest.fixture
def dense(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    from askpolicy.retrieval import role_faiss
    return role_faiss


@pytest.mark.parametrize("dtype,width,component", [
    ("float64", 384, 1), ("float32", 383, 1), ("float32", 385, 1),
    ("float32", 384, np.nan), ("float32", 384, np.inf), ("float32", 384, 0),
    ("float32", 384, 0.998), ("float32", 384, 1.002),
])
def test_batch_rejects_invalid_vector(dense, dtype, width, component):
    vectors = np.zeros((2, width), dtype=dtype)
    vectors[:, 0] = [1, component]
    with pytest.raises(dense.DenseIndexError):
        dense.validate_vectors(vectors, 2)


def test_unit_interval_includes_boundaries(dense):
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[:, 0] = [0.999, 1.001]
    dense.validate_vectors(vectors, 2)


def test_exact_inner_product_scores_ties_and_distinct_child_budget(dense):
    # More than 30 rows from one child tie for first. A lower-scoring segment
    # from that same child must never replace its best evidence.
    labels = [("many", 1.0)] * 35 + [("z-tie", 0.8), ("a-tie", 0.8), ("many", -1.0)]
    labels += [(f"other-{i:02d}", -0.6) for i in reversed(range(40))]
    vectors = np.zeros((len(labels), 384), dtype=np.float32)
    rows = []
    for i, (child, score) in enumerate(labels):
        vectors[i, :2] = (score, {1.0: 0, 0.8: 0.6, -1.0: 0, -0.6: 0.8}[score])
        rows.append(dense.Segment(f"seg-{i:03d}", child, i, i + 1, 4))
    dense.validate_vectors(vectors, len(labels))
    engine = faiss.IndexFlatIP(384)
    engine.add(vectors)
    index = SimpleNamespace(metadata=SimpleNamespace(rows=rows), _flat=SimpleNamespace(engine=engine))
    hits = dense._rank_children(index, vectors[:1])
    ids = [rows[pos].child_id for pos, _ in hits]
    assert ids == ["many", "a-tie", "z-tie", *[f"other-{i:02d}" for i in range(27)]]
    assert hits[0] == (0, 1.0)
    for pos, score in hits:
        cosine = np.dot(vectors[0], vectors[pos]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[pos]))
        assert score == pytest.approx(cosine, abs=1e-6)
    assert hits[-1][1] == pytest.approx(-0.6)


def test_native_rows_without_matching_segment_positions_fail(dense):
    # The native row with the highest score has no mapped source segment.
    engine = faiss.IndexFlatIP(384)
    vectors = np.eye(2, 384, dtype=np.float32)
    engine.add(vectors)
    rows = (dense.Segment("segment-0", "child-0", 20, 25, 6),)
    index = SimpleNamespace(metadata=SimpleNamespace(rows=rows), _flat=SimpleNamespace(engine=engine))
    with pytest.raises(dense.DenseIndexError, match="^INVALID_SEARCH_ROWS$"):
        dense._rank_children(index, vectors[1:])


@pytest.mark.parametrize("text,split", [
    (" \tDéplacement، سفر!\n", False),
    ((" \tÉcole e\u0301 🎒، سفر.\n" * 90) + " fin !  ", True),
    ("密" * 1400, True),
])
def test_segments_cover_exact_original_character_spans(dense, text, split):
    from askpolicy.retrieval.bm25_core import EvidenceChunk, SectionIdentity

    chunk = EvidenceChunk("child-id", SectionIdentity("doc", "parent", None), None, 117, text)
    def count(value):
        return 3 + len(value)
    pieces = list(dense.split_passage(chunk, count))
    assert (len(pieces) > 1) is split
    assert list(dense.split_passage(chunk, count)) == pieces
    assert len({p.segment.segment_id for p in pieces}) == len(pieces)
    assert "".join(p.text for p in pieces) == text
    cursor = 117
    for passage in pieces:
        seg = passage.segment
        assert seg.child_id == chunk.child_id and cursor == seg.start < seg.end
        assert passage.text == text[seg.start - 117:seg.end - 117]
        assert seg.token_count == count(passage.text) <= 512
        cursor = seg.end
    assert cursor == 117 + len(text)
    with pytest.raises(ValueError, match="^INVALID_SOURCE_SPAN$"):
        list(dense.split_passage(replace(chunk, start=-1), count))
