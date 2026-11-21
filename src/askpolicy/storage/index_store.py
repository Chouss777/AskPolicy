"""Verified artifact storage for complete offline index releases."""

from contextlib import closing
from copy import copy
from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import stat
from tempfile import TemporaryDirectory

import bm25s
import faiss
import numpy as np

from .snapshot import DraftSnapshotConfig
from ..retrieval import role_bm25 as lexical, role_faiss as dense
from ..retrieval.e5_encoder import E5Encoder
from ..retrieval.e5_passage_split import split_passage
from ..retrieval.hybrid import search_role_hybrid


class IndexStoreError(ValueError):
    """Content-free artifact or publication failure."""


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=_json_date).encode("utf-8")


def _json_date(value):
    if type(value) is date:
        return value.isoformat()
    raise ValueError("INVALID_MANIFEST_VALUE")


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def safe_path(path: Path) -> Path:
    """Reject links and Windows reparse points in every existing component."""
    path = Path(path).absolute()
    for component in (*reversed(path.parents), path):
        if component.exists() or component.is_symlink():
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
                raise IndexStoreError("UNSAFE_PATH")
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IndexStoreError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def read_json(raw: bytes):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)


def valid_id(value: object) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


_BM_FILES = ("data.csc.index.npy", "indices.csc.index.npy", "indptr.csc.index.npy",
             "vocab.index.json", "params.index.json")


def artifact_paths() -> dict[str, str]:
    paths = {"sqlite": "sqlite/corpus.sqlite"}
    for role in dense.ROLES:
        paths["faiss." + role] = f"faiss/{role}/index.faiss"
        for name in _BM_FILES:
            paths[f"bm25.{role}.{name}"] = f"bm25/{role}/{name}"
    return paths


@dataclass(frozen=True, slots=True)
class PublishedSnapshot:
    release_id: str
    manifest_hash: str
    bm25: lexical.RoleBM25Bundle = field(repr=False)
    faiss: dense.RoleFaissBundle = field(repr=False)
    status: str = field(default="PUBLISHED", init=False)

    def search(self, context, *, query: str, validity_date: date):
        return search_role_hybrid(self.bm25, self.faiss, context, query=query,
                                  expected_snapshot_id=self.faiss.snapshot.candidate_snapshot_id,
                                  validity_date=validity_date)


def _identity(record):
    # Artifact locations are fixed by the format and excluded from identity.
    return digest(canonical({k: v for k, v in record.items() if k != "artifact_paths"}))


def _binding(bm25, faiss):
    dense.verify_matching_bundles(bm25, faiss)
    return dict(format=1, snapshot=asdict(bm25.snapshot),
                validity_date=bm25.eligibility.validity_date,
                next_transition_date=bm25.eligibility.next_transition_date,
                lexical=[asdict(i.metadata) for i in bm25.indexes],
                dense=[asdict(i.metadata) for i in faiss.indexes])


def _save_lexical(model, path: Path) -> None:
    """Canonicalize CSC token columns, whose IDs can vary with Python hash seeds."""
    vocabulary = model.vocab_dict
    if set(vocabulary.values()) != set(range(len(vocabulary))):
        raise IndexStoreError("INVALID_LEXICAL_VOCABULARY")
    saved = copy(model)
    saved.vocab_dict = {term: position for position, term in enumerate(sorted(vocabulary))}
    saved.unique_token_ids_set = set(saved.vocab_dict.values())
    pieces = []
    pointers = [0]
    for term in saved.vocab_dict:
        column = vocabulary[term]
        start, end = model.scores["indptr"][column:column + 2]
        pieces.append(slice(start, end))
        pointers.append(pointers[-1] + end - start)
    saved.scores = {
        "data": np.concatenate([model.scores["data"][part] for part in pieces]),
        "indices": np.concatenate([model.scores["indices"][part] for part in pieces]),
        "indptr": np.asarray(pointers, dtype=model.scores["indptr"].dtype),
        "num_docs": model.scores["num_docs"],
    }
    saved.save(path, allow_pickle=False, show_progress=False)


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(raw)
        target.flush()
        import os
        os.fsync(target.fileno())


def stage_snapshot(
    connection: sqlite3.Connection, directory: Path, *, manifest: bytes,
    snapshot_config: DraftSnapshotConfig, validity_date: date, encoder: E5Encoder,
) -> str:
    """Build all eight indexes from the exact validated, persisted SQLite image."""
    try:
        directory = safe_path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        sqlite_path = directory / artifact_paths()["sqlite"]
        sqlite_path.parent.mkdir()
        with lexical.validated_role_draft(connection, manifest=manifest, snapshot_config=snapshot_config,
                                          validity_date=validity_date) as (image, draft):
            with closing(sqlite3.connect(sqlite_path)) as stored:
                image.backup(stored)
        options = dict(manifest=manifest, snapshot_config=snapshot_config, validity_date=validity_date)
        with closing(sqlite3.connect(sqlite_path)) as stored:
            bm = lexical.build_role_bm25_bundle(stored, **options,
                                                expected_snapshot_id=draft.snapshot.candidate_snapshot_id)
            fa = dense.build_role_faiss_bundle(stored, **options, encoder=encoder,
                                               expected_snapshot_id=draft.snapshot.candidate_snapshot_id)
        for index in bm.indexes:
            path = directory / "bm25" / index.metadata.role
            if index._engine is None:
                raise IndexStoreError("EMPTY_LEXICAL_INDEX")
            _save_lexical(index._engine, path)
        for index in fa.indexes:
            path = directory / "faiss" / index.metadata.role / "index.faiss"
            path.parent.mkdir(parents=True)
            faiss.write_index(index._flat.engine, str(path))
        import os
        for relative in artifact_paths().values():
            with (directory / relative).open("r+b") as artifact:
                os.fsync(artifact.fileno())
        record = _binding(bm, fa)
        record["artifact_paths"] = artifact_paths()
        record["artifacts"] = {slot: digest((directory / path).read_bytes())
                               for slot, path in artifact_paths().items()}
        _write(directory / "manifest.json", canonical(record))
        return _identity(record)
    except IndexStoreError:
        raise
    except Exception:
        raise IndexStoreError("STAGING_FAILED") from None


def _load(directory, *, manifest, snapshot_config, validity_date, encoder, expected_release_id):
    directory = safe_path(directory)
    raw_manifest = (safe_path(directory / "manifest.json")).read_bytes()
    record = read_json(raw_manifest)
    if (
        type(record) is not dict or type(record.get("format")) is not int or record["format"] != 1
        or record.get("artifact_paths") != artifact_paths()
        or type(record.get("artifacts")) is not dict or set(record["artifacts"]) != set(artifact_paths())
        or not valid_id(expected_release_id) or _identity(record) != expected_release_id
    ):
        raise IndexStoreError("INVALID_RELEASE_MANIFEST")
    # Copy checked bytes into a private directory. Both native loaders and
    # SQLite operate exclusively on this checked copy, including during switches.
    with TemporaryDirectory(prefix="askpolicy-load-") as temporary:
        private = Path(temporary)
        for slot, relative in artifact_paths().items():
            raw = safe_path(directory / relative).read_bytes()
            if digest(raw) != record["artifacts"][slot]:
                raise IndexStoreError("ARTIFACT_HASH_MISMATCH")
            _write(private / relative, raw)
        with closing(sqlite3.connect(private / artifact_paths()["sqlite"])) as db:
            options = dict(manifest=manifest, snapshot_config=snapshot_config, validity_date=validity_date)
            bm = lexical.build_role_bm25_bundle(db, **options)
            # A lexical rebuild independently checks the safe serialized content.
            # The returned scorer uses BM25.load, with pickle and corpus loading disabled.
            loaded = []
            for index in bm.indexes:
                role = index.metadata.role
                check = private / "verify" / role
                _save_lexical(index._engine, check)
                for name in _BM_FILES:
                    if (check / name).read_bytes() != (private / "bm25" / role / name).read_bytes():
                        raise IndexStoreError("LEXICAL_SOURCE_MISMATCH")
                model = bm25s.BM25.load(private / "bm25" / role, allow_pickle=False, load_corpus=False,
                                       mmap=False, show_progress=False, verbosity=0)
                children = tuple(c for c in bm._draft.children.values()
                                 if c.section.document_id in index.metadata.indexed_document_ids)
                loaded.append(lexical.bind_text_index(index.metadata, children, model))
            bm = lexical.RoleBM25Bundle(tuple(loaded), bm._draft)
            dense._check_encoder(encoder)
            indexes = []
            for role in dense.ROLES:
                chunks = dense.read_draft_chunks(db, lexical._authorized_documents(bm._draft, role))
                rows = tuple(p.segment for c in chunks for p in split_passage(c, encoder.passage_token_count))
                engine = faiss.read_index(str(private / "faiss" / role / "index.faiss"))
                if type(engine) is not faiss.IndexFlatIP or engine.d != 384 or engine.ntotal != len(rows):
                    raise IndexStoreError("INVALID_DENSE_INDEX")
                dense.validate_vectors(engine.reconstruct_n(0, engine.ntotal), len(rows))
                allowed = lexical._authorized_documents(bm._draft, role)
                children = tuple(c for c in bm._draft.children.values() if c.section.document_id in allowed)
                meta = dense.DenseMetadata(
                    bm.snapshot.candidate_snapshot_id, bm.snapshot.source_manifest_hash, bm.snapshot.config_hash,
                    bm.eligibility.validity_date, bm.eligibility.next_transition_date, role, dense.DenseConfiguration(),
                    tuple(sorted(allowed)), tuple(sorted(c.child_id for c in children)), rows,
                    dense._catalog_hash(allowed, children), dense._vector_hash(engine),
                )
                indexes.append(dense.RoleFaissIndex(meta, dense._BoundFlatIndex(meta.fingerprint, engine)))
            fa = dense.RoleFaissBundle(tuple(indexes), bm._draft, encoder)
        expected = _binding(bm, fa)
        actual = {k: v for k, v in record.items() if k not in ("artifacts", "artifact_paths")}
        if canonical(expected) != canonical(actual):
            raise IndexStoreError("SOURCE_BINDING_MISMATCH")
        return PublishedSnapshot(expected_release_id, digest(raw_manifest), bm, fa)


def load_snapshot(
    directory: Path, *, manifest: bytes, snapshot_config: DraftSnapshotConfig,
    validity_date: date, encoder: E5Encoder, expected_release_id: str,
) -> PublishedSnapshot:
    """Load a complete release against trusted inputs with no partial result."""
    try:
        return _load(directory, manifest=manifest, snapshot_config=snapshot_config,
                     validity_date=validity_date, encoder=encoder, expected_release_id=expected_release_id)
    except IndexStoreError:
        raise
    except Exception:
        raise IndexStoreError("RELEASE_LOAD_FAILED") from None
