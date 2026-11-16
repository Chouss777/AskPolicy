"""Measure seed Recall@5 offline using only validated PUBLIC policy evidence."""

import argparse
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
import sys
from typing import Annotated, Literal
import unicodedata

import bm25s
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..ingestion.coordinator import ingest_manifest_to_draft
from ..ingestion.metadata import Language, PolicyMetadata, ReadRole
from ..storage.eligibility import PolicyEligibilityResult, select_policy_versions
from ..storage.snapshot import DraftSnapshotConfig, compute_draft_snapshot_identity


VALIDITY_DATE = date(2026, 9, 1)
TOP_K = 5
READINESS_THRESHOLD = 0.0
DATASET = Path("eval/datasets/dev_candidates.jsonl")
MANIFEST = Path("data/corpus_manifest.yaml")
_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
_BM25S_OPTIONS = {
    "method": "lucene", "idf_method": "lucene", "k1": 1.5, "b": 0.75,
    "dtype": "float32", "int_dtype": "int32", "backend": "numpy", "csc_backend": "numpy",
}
_SEED_IDS = frozenset(f"dev-candidate-{number:03}" for number in range(1, 6))


class SmokeEvaluationError(ValueError):
    """A controlled evaluation failure with a content-free code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _GoldLabel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)
    document_id: Annotated[str, Field(min_length=1)]
    section_heading: Annotated[str, Field(min_length=1)]
    relevance_grade: Annotated[int, Field(gt=0)]


class _DevelopmentCase(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)
    query_id: Annotated[str, Field(min_length=1)]
    question_family_id: Annotated[str, Field(min_length=1)]
    language: Language
    active_role: ReadRole
    query: Annotated[str, Field(min_length=1)]
    expected_status: Literal["answer", "refusal"]
    authorized_source_sections: list[_GoldLabel]
    required_facts: list[Annotated[str, Field(min_length=1)]]
    case_type: Literal["answerable", "role_sensitive", "unanswerable"]
    annotation_review_status: Literal["human_approved"]
    revalidation_required: bool


@dataclass(frozen=True, slots=True)
class DraftDocument:
    metadata: PolicyMetadata
    source_path: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class SectionIdentity:
    document_id: str
    parent_id: str
    heading: str | None


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    child_id: str
    section: SectionIdentity
    document: DraftDocument
    start: int
    text: str


@dataclass(frozen=True, slots=True)
class RankedChunk:
    child_id: str
    section: SectionIdentity
    score: float


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise SmokeEvaluationError("INVALID_SEED_DATASET")
        fields[key] = value
    return fields


def _read_cases(raw: bytes) -> list[_DevelopmentCase]:
    try:
        cases = [
            _DevelopmentCase.model_validate(json.loads(line.decode("utf-8"), object_pairs_hook=_unique_fields))
            for line in raw.splitlines()
        ]
    except (ValueError, RecursionError):
        raise SmokeEvaluationError("INVALID_SEED_DATASET") from None
    if len({case.query_id for case in cases}) != len(cases):
        raise SmokeEvaluationError("INVALID_SEED_DATASET")
    for case in cases:
        expected_evidence = case.expected_status == "answer"
        if (
            not case.query.strip()
            or case.revalidation_required is not False
            or bool(case.authorized_source_sections) != expected_evidence
            or bool(case.required_facts) != expected_evidence
        ):
            raise SmokeEvaluationError("INVALID_SEED_DATASET")
    return sorted(cases, key=lambda case: case.query_id)


def _read_documents(connection: sqlite3.Connection) -> dict[str, DraftDocument]:
    fields = tuple(PolicyMetadata.model_fields)
    rows = connection.execute(
        "SELECT " + ", ".join((*fields, "source_path", "source_sha256"))
        + " FROM draft_documents ORDER BY document_id",
    )
    documents = {}
    for row in rows:
        metadata = dict(zip(fields, row[:-2], strict=True))
        metadata["allowed_roles"] = tuple(json.loads(metadata["allowed_roles"]))
        for name in ("effective_date", "expires_at"):
            if metadata[name] is not None:
                metadata[name] = date.fromisoformat(metadata[name])
        try:
            validated = PolicyMetadata.model_validate(metadata)
        except ValidationError:
            raise SmokeEvaluationError("INVALID_STORED_METADATA") from None
        documents[validated.document_id] = DraftDocument(validated, row[-2], row[-1])
    return documents


def _public_documents(
    documents: dict[str, DraftDocument], eligibility: PolicyEligibilityResult, role: ReadRole,
) -> dict[str, DraftDocument]:
    return {
        document_id: documents[document_id]
        for document_id in eligibility.eligible_document_ids
        if documents[document_id].metadata.classification == "PUBLIC"
        and role in documents[document_id].metadata.allowed_roles
    }


def _resolve_section(connection: sqlite3.Connection, label: _GoldLabel) -> SectionIdentity:
    matches = connection.execute(
        "SELECT parent_id, heading FROM draft_parents WHERE document_id = ? AND heading = ?",
        (label.document_id, label.section_heading),
    ).fetchall()
    if len(matches) != 1:
        raise SmokeEvaluationError("GOLD_SECTION_MISSING" if not matches else "GOLD_SECTION_AMBIGUOUS")
    parent_id, heading = matches[0]
    return SectionIdentity(label.document_id, parent_id, heading)


def _select_case(
    connection: sqlite3.Connection, cases: Sequence[_DevelopmentCase],
    documents: dict[str, DraftDocument], eligibility: PolicyEligibilityResult,
) -> tuple[_DevelopmentCase, tuple[SectionIdentity, ...], list[dict[str, str]]]:
    selected = []
    excluded = []
    for case in cases:
        if case.expected_status == "refusal":
            reason = "REFUSAL_NO_POSITIVE_GOLD"
        else:
            gold = tuple(_resolve_section(connection, label) for label in case.authorized_source_sections)
            public = _public_documents(documents, eligibility, case.active_role)
            if any(documents[section.document_id].metadata.classification != "PUBLIC" for section in gold):
                reason = "GOLD_NOT_PUBLIC"
            elif any(section.document_id not in public for section in gold):
                reason = "GOLD_NOT_CURRENT_OR_GRANTED"
            else:
                selected.append((case, gold))
                continue
        excluded.append({"query_id": case.query_id, "reason": reason})

    if not selected:
        raise SmokeEvaluationError("NO_ELIGIBLE_POSITIVE_CASES")
    if len(selected) != 1 or {case.query_id for case in cases} != _SEED_IDS:
        raise SmokeEvaluationError("UNEXPECTED_SEED_SCOPE")
    case, gold = selected[0]
    if (case.query_id, case.language, case.active_role, len(gold)) != ("dev-candidate-001", "fr", "employee", 1):
        raise SmokeEvaluationError("UNEXPECTED_SEED_SCOPE")
    return case, gold, excluded


def _load_public_chunks(
    connection: sqlite3.Connection, documents: dict[str, DraftDocument],
    eligibility: PolicyEligibilityResult, role: ReadRole,
) -> tuple[EvidenceChunk, ...]:
    public = _public_documents(documents, eligibility, role)
    if not public:
        raise SmokeEvaluationError("NO_PUBLIC_EVIDENCE")
    parameters = tuple(public)
    placeholders = ", ".join("?" for _ in parameters)
    rows = connection.execute(
        "SELECT c.child_id, c.document_id, c.parent_id, p.heading, c.start, c.text "
        "FROM draft_children AS c JOIN draft_parents AS p "
        "ON p.document_id = c.document_id AND p.parent_id = c.parent_id "
        f"WHERE c.document_id IN ({placeholders}) ORDER BY c.document_id, c.start, c.child_id",
        parameters,
    )
    return tuple(
        EvidenceChunk(child_id, SectionIdentity(document_id, parent_id, heading), public[document_id], start, text)
        for child_id, document_id, parent_id, heading, start, text in rows
    )


def _tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(
        [unicodedata.normalize("NFC", text) for text in texts],
        lower=True, token_pattern=_TOKEN_PATTERN, stopwords=[], stemmer=None,
        return_ids=False, show_progress=False,
    )


def rank_public_chunks(
    chunks: Sequence[EvidenceChunk], *, query: str,
    eligibility: PolicyEligibilityResult, role: ReadRole,
) -> tuple[RankedChunk, ...]:
    """Rank preselected PUBLIC children by positive BM25S score and child ID.

    The caller supplies verified draft records and their eligibility result.
    Reject input outside the reviewed role's PUBLIC scope before tokenization.
    """
    ordered = sorted(chunks, key=lambda chunk: (chunk.section.document_id, chunk.start, chunk.child_id))
    current = set(eligibility.eligible_document_ids)
    if any(
        chunk.section.document_id not in current
        or chunk.section.document_id != chunk.document.metadata.document_id
        or chunk.document.metadata.classification != "PUBLIC"
        or role not in chunk.document.metadata.allowed_roles
        for chunk in ordered
    ):
        raise SmokeEvaluationError("OUT_OF_SCOPE_INDEX_INPUT")
    if len({chunk.child_id for chunk in ordered}) != len(ordered):
        raise SmokeEvaluationError("DUPLICATE_CHILD_ID")
    tokens = _tokenize([chunk.text for chunk in ordered])
    if not any(tokens):
        raise SmokeEvaluationError("NO_INDEXABLE_TOKENS")

    index = bm25s.BM25(**_BM25S_OPTIONS)
    index.index(tokens, create_empty_token=False, show_progress=False)
    query_tokens = _tokenize([query])[0]
    if not query_tokens:
        return ()
    scores = index.get_scores(query_tokens)
    candidates = [
        RankedChunk(chunk.child_id, chunk.section, float(score))
        for chunk, score in zip(ordered, scores, strict=True)
        if score > 0
    ]
    return tuple(sorted(candidates, key=lambda chunk: (-chunk.score, chunk.child_id))[:TOP_K])


def recall_at_5(gold: Sequence[SectionIdentity], ranking: Sequence[RankedChunk]) -> float:
    """Credit each exact document/parent pair once among the first five ranks."""
    expected = {(section.document_id, section.parent_id) for section in gold}
    if not expected:
        raise SmokeEvaluationError("NO_POSITIVE_GOLD")
    found = set()
    seen_children = set()
    for hit in ranking[:TOP_K]:
        if hit.child_id not in seen_children:
            seen_children.add(hit.child_id)
            found.add((hit.section.document_id, hit.section.parent_id))
    return len(expected & found) / len(expected)


def _section_record(section: SectionIdentity) -> dict[str, str | None]:
    return {"document_id": section.document_id, "parent_id": section.parent_id, "section_heading": section.heading}


def evaluate_public_smoke(repository_root: Path) -> dict[str, object]:
    """Run the fixed-date, human-approved seed evaluation on a fresh memory draft."""
    raw_dataset = (repository_root / DATASET).read_bytes()
    raw_manifest = (repository_root / MANIFEST).read_bytes()
    cases = _read_cases(raw_dataset)
    config = DraftSnapshotConfig()
    with closing(sqlite3.connect(":memory:")) as connection:
        ingest_manifest_to_draft(repository_root, connection)
        identity = compute_draft_snapshot_identity(connection, manifest=raw_manifest, config=config)
        documents = _read_documents(connection)
        eligibility = select_policy_versions(
            [document.metadata for document in documents.values()], validity_date=VALIDITY_DATE,
        )
        case, gold, excluded = _select_case(connection, cases, documents, eligibility)
        chunks = _load_public_chunks(connection, documents, eligibility, case.active_role)

    ranking = rank_public_chunks(chunks, query=case.query, eligibility=eligibility, role=case.active_role)
    score = recall_at_5(gold, ranking)
    indexed_ids = sorted({chunk.section.document_id for chunk in chunks})
    excluded_counts = {
        reason: sum(record["reason"] == reason for record in excluded)
        for reason in sorted({record["reason"] for record in excluded})
    }
    return {
        "evaluation_kind": "offline_public_smoke",
        "scope": "human_approved_seed_public_evidence",
        "validity_date": VALIDITY_DATE.isoformat(),
        "dataset_sha256": sha256(raw_dataset).hexdigest(),
        "source_manifest_sha256": sha256(raw_manifest).hexdigest(),
        "canonical_source_manifest_sha256": identity.source_manifest_hash,
        "candidate_draft_snapshot_id": identity.candidate_snapshot_id,
        "snapshot_status": identity.status,
        "draft_counts": {
            "documents": identity.document_count, "parents": identity.parent_count, "children": identity.child_count,
        },
        "draft_snapshot_config_hash": identity.config_hash,
        "bm25s_version": version("bm25s"),
        "numpy_version": version("numpy"),
        "configuration": {
            "bm25s": dict(_BM25S_OPTIONS),
            "tokenization": {
                "normalization": "NFC", "unicode_version": unicodedata.unidata_version,
                "lowercase": True, "pattern": _TOKEN_PATTERN,
                "stopwords": [], "stemmer": None,
            },
            "indexed_text": "exact_stored_child_text",
            "index_order": ["document_id", "child_start", "child_id"],
            "ranking_order": ["score_descending", "child_id_ascending"],
            "positive_scores_only": True,
            "top_k": TOP_K,
            "empty_token_indexed": False,
        },
        "indexed_document_count": len(indexed_ids),
        "indexed_document_ids": indexed_ids,
        "indexed_child_count": len(chunks),
        "evaluated_positive_case_count": 1,
        "evaluated_query_ids": [case.query_id],
        "excluded_case_count": len(excluded),
        "excluded_case_counts_by_reason": excluded_counts,
        "excluded_cases": excluded,
        "cases": [{
            "query_id": case.query_id,
            "language": case.language,
            "active_role": case.active_role,
            "gold_sections": [_section_record(section) for section in gold],
            "gold_section_count": len({(section.document_id, section.parent_id) for section in gold}),
            "ranking": [
                {"rank": rank, "child_id": hit.child_id, "score": hit.score, **_section_record(hit.section)}
                for rank, hit in enumerate(ranking, 1)
            ],
            "recall_at_5": score,
        }],
        "macro_recall_at_5": score,
        "readiness_gate": {"operator": ">", "threshold": READINESS_THRESHOLD, "passed": score > READINESS_THRESHOLD},
    }


def encode_report(report: dict[str, object]) -> bytes:
    return (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the measured JSON report to this path")
    args = parser.parse_args()
    report = evaluate_public_smoke(Path.cwd())
    payload = encode_report(report)
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    return 0 if report["macro_recall_at_5"] > READINESS_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
