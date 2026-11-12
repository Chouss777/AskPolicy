from collections import Counter
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re

import pytest

from .test_corpus_batch1 import (
    ACTIVE_SLOTS,
    HISTORICAL_IDS,
    MATRIX_PATH,
    POLICY_ROOT,
    REPOSITORY_ROOT,
    _nullable,
    _registry_rows,
    _scalar,
)
from .test_corpus_batch2 import IDENTITY_COLUMNS, PENDING_LINKS as EXPECTED_LINKS
from .test_corpus_expansion import _parse_policy


MANIFEST_PATH = REPOSITORY_ROOT / "data" / "corpus_manifest.yaml"
METADATA_FIELDS = (
    "policy_id", "document_id", "version", "language", "title", "department",
    "classification", "allowed_roles", "effective_date", "expires_at",
    "supersedes", "approval_status", "source_type", "source_license",
)
RECORD_FIELDS = {"source_path", "content_sha256", *METADATA_FIELDS}
ACTIVE_IDS = {f"{family}-{language.upper()}-V{version}": (family, language, version)
              for family, language, version in ACTIVE_SLOTS}
HISTORICAL_SLOTS = {document_id: slot for slot, document_id in HISTORICAL_IDS.items()}
EXPECTED_SLOTS = ACTIVE_IDS | HISTORICAL_SLOTS
EXPECTED_PATHS = {document_id: f"data/policies/{language}/{document_id}.md"
                  for document_id, (_, language, _) in EXPECTED_SLOTS.items()}
ALL_ROLES = ["employee", "hr", "finance", "it_security"]
FAMILY_ACLS = {
    "TRAVEL-001": ("PUBLIC", ALL_ROLES),
    "LEAVE-001": ("RESTRICTED", ["hr"]),
    "ACCESS-001": ("RESTRICTED", ["it_security"]),
    "REMOTE-001": ("INTERNAL", ALL_ROLES),
    "EXPENSES-001": ("INTERNAL", ALL_ROLES),
    "DATA-001": ("INTERNAL", ALL_ROLES),
    "INCIDENT-001": ("INTERNAL", ALL_ROLES),
    "PROCUREMENT-001": ("INTERNAL", ALL_ROLES),
}
TRAVEL_EN = "TRAVEL-001-EN-V1"
TRAVEL_EN_PATH = EXPECTED_PATHS[TRAVEL_EN]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate manifest field: {key}"
        result[key] = value
    return result


def _parse_manifest(text):
    # The checked-in manifest uses the JSON-compatible subset of YAML 1.2.
    payload = json.loads(text, object_pairs_hook=_unique_object)
    assert isinstance(payload, dict) and payload.keys() == {"documents"}, "Expected only documents"
    assert isinstance(payload["documents"], list), "Expected a documents list"
    return payload["documents"]


def _canonical_bytes(raw):
    # Hash the entire UTF-8 source, including both front-matter delimiters.
    # Convert CRLF to LF. Preserve all other bytes, including whitespace and Unicode.
    raw.decode("utf-8", errors="strict")
    canonical = raw.replace(b"\r\n", b"\n")
    assert b"\r" not in canonical, "Unexpected standalone CR in policy source"
    return canonical


def _source_record(path, raw):
    canonical = _canonical_bytes(raw)
    fields, _ = _parse_policy(canonical.decode("utf-8"), PurePosixPath(path))
    assert fields.keys() == set(METADATA_FIELDS), f"Unexpected source fields: {path}"
    assert re.fullmatch(r"[1-9][0-9]*", fields["version"]), f"Invalid source version: {path}"

    # The corpus reader validates role grants and converts block lists to flow lists.
    # Decode the validated tokens in source order to compare the full sequence.
    role_list = re.fullmatch(r"\[([^\[\]]*)\](?:[ \t]+#.*)?", fields["allowed_roles"].strip())
    assert role_list is not None, f"Invalid source roles: {path}"
    metadata = {}
    for key in METADATA_FIELDS:
        value = fields[key]
        if key == "version":
            metadata[key] = int(value)
        elif key == "allowed_roles":
            roles = role_list[1].strip().removesuffix(",")
            metadata[key] = [_scalar(role) for role in roles.split(",")]
        elif key in {"expires_at", "supersedes"}:
            metadata[key] = _nullable(value)
        else:
            metadata[key] = _scalar(value)

    return {"source_path": path, "content_sha256": sha256(canonical).hexdigest(), **metadata}


def _check_source_path(value, document_id):
    assert isinstance(value, str), f"Invalid source path: {value!r}"
    path = PurePosixPath(value)
    assert (
        len(path.parts) == 4
        and path.parts[:2] == ("data", "policies")
        and ".." not in path.parts
        and "\\" not in value
        and value == path.as_posix()
    ), f"Invalid source path: {value!r}"
    assert value == EXPECTED_PATHS[document_id], f"Source path does not match {document_id}: {value}"


def _index_records(records):
    indexed = {}
    slots = set()
    for row in records:
        assert isinstance(row, dict) and row.keys() == RECORD_FIELDS, "Unexpected manifest fields"
        document_id = row["document_id"]
        assert isinstance(document_id, str) and document_id in EXPECTED_SLOTS, (
            f"Unauthorized document ID: {document_id!r}"
        )
        assert document_id not in indexed, f"Duplicate manifest record: {document_id}"
        assert type(row["version"]) is int, f"Invalid manifest version: {document_id}"
        slot = (row["policy_id"], row["language"], row["version"])
        assert isinstance(slot[0], str) and isinstance(slot[1], str), f"Invalid policy slot: {document_id}"
        assert slot not in slots, f"Duplicate policy slot: {slot}"
        assert slot == EXPECTED_SLOTS[document_id], f"Unauthorized policy slot for {document_id}: {slot}"
        _check_source_path(row["source_path"], document_id)

        classification, roles = FAMILY_ACLS[row["policy_id"]]
        assert row["classification"] == classification, f"Incorrect classification: {document_id}"
        assert row["allowed_roles"] == roles, f"Incorrect reader ACL: {document_id}"
        assert row["source_type"] == "synthetic_policy", f"Incorrect source type: {document_id}"
        assert row["source_license"] == "project_owned", f"Incorrect source license: {document_id}"
        indexed[document_id] = row
        slots.add(slot)

    missing = EXPECTED_SLOTS.keys() - indexed.keys()
    assert not missing, f"Missing manifest documents: {sorted(missing)}"
    assert list(indexed) == sorted(indexed), "Manifest documents must be sorted by document_id"
    return indexed


def _check_predecessors(indexed):
    links = {}
    for document_id, row in indexed.items():
        predecessor = row["supersedes"]
        if predecessor is None:
            continue
        assert isinstance(predecessor, str) and predecessor in indexed, (
            f"Missing predecessor for {document_id}: {predecessor!r}"
        )
        assert predecessor != document_id, f"Self-supersession: {document_id}"
        previous = indexed[predecessor]
        assert row["policy_id"] == previous["policy_id"], f"Wrong-family predecessor: {document_id}"
        assert row["language"] == previous["language"], f"Wrong-language predecessor: {document_id}"
        links[document_id] = predecessor
    assert links == EXPECTED_LINKS, f"Unexpected supersession relationships: {links}"


def _validate_manifest(records, sources):
    paths = set(EXPECTED_PATHS.values())
    assert not (missing := paths - sources.keys()), f"Missing source files: {sorted(missing)}"
    assert not (extra := sources.keys() - paths), f"Unexpected source files: {sorted(extra)}"
    indexed = _index_records(records)
    _check_predecessors(indexed)

    for document_id, row in indexed.items():
        path = row["source_path"]
        expected = _source_record(path, sources[path])
        for key in METADATA_FIELDS:
            assert row[key] == expected[key], f"Metadata mismatch for {document_id}: {key}"
        assert row["content_sha256"] == expected["content_sha256"], (
            f"Content SHA-256 mismatch: {document_id}"
        )


@pytest.fixture(scope="module")
def manifest_text():
    assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}"
    return MANIFEST_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def records(manifest_text):
    return _parse_manifest(manifest_text)


@pytest.fixture(scope="module")
def sources():
    files = {}
    for path in POLICY_ROOT.rglob("*"):
        if path.is_file():
            assert path.resolve().is_relative_to(POLICY_ROOT.resolve()), f"External source file: {path}"
            files[path.relative_to(REPOSITORY_ROOT).as_posix()] = path.read_bytes()
    return files


@pytest.fixture(scope="module")
def verified_manifest(records, sources):
    _validate_manifest(records, sources)
    return records, sources


def test_manifest_covers_all_authorized_sources_with_exact_metadata_and_hashes(records, sources):
    _validate_manifest(records, sources)
    assert len(records) == 26
    current = [row for row in records if row["document_id"] in ACTIVE_IDS]
    historical = [row for row in records if row["document_id"] in HISTORICAL_SLOTS]
    assert Counter(row["language"] for row in current) == {"fr": 8, "en": 8, "ar": 4}
    assert Counter(row["language"] for row in historical) == {"fr": 5, "en": 1}


def test_manifest_identities_and_acls_agree_with_the_rule_matrix(verified_manifest):
    records, _ = verified_manifest
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    rows = _registry_rows(matrix, "E", "F", IDENTITY_COLUMNS)
    registry = [tuple(cell.strip("\x60") for cell in row) for row in rows]
    expected = {
        (row["document_id"], row["policy_id"], row["language"], str(row["version"]),
         row["classification"], f"[{', '.join(row['allowed_roles'])}]", row["supersedes"] or "null")
        for row in records
    }
    assert len(registry) == len(expected)
    assert set(registry) == expected


def _render_source_manifest(source_items):
    records = sorted((_source_record(path, raw) for path, raw in source_items),
                     key=lambda row: row["document_id"])
    return json.dumps({"documents": records}, ensure_ascii=False, indent=2) + "\n"


def test_manifest_is_stable_for_unchanged_sources(manifest_text, verified_manifest):
    _, sources = verified_manifest
    items = list(sources.items())
    assert _render_source_manifest(items) == _render_source_manifest(reversed(items)) == manifest_text


def _change_record(records, document_id, **updates):
    assert any(row["document_id"] == document_id for row in records)
    return [row | updates if row["document_id"] == document_id else row for row in records]


def test_hash_is_stable_across_lf_and_crlf_checkouts(verified_manifest):
    records, sources = verified_manifest
    lf = _canonical_bytes(sources[TRAVEL_EN_PATH])
    crlf = lf.replace(b"\n", b"\r\n")
    assert crlf != lf
    assert _canonical_bytes(crlf) == lf
    for raw in (lf, crlf):
        _validate_manifest(records, sources | {TRAVEL_EN_PATH: raw})


def test_hash_detects_one_character_source_body_change(verified_manifest):
    records, sources = verified_manifest
    raw = sources[TRAVEL_EN_PATH]
    before = b"\n# Business Travel Policy"
    assert raw.count(before) == 1
    changed = raw.replace(before, b"\n# Business Travel Policx", 1)
    assert len(changed) == len(raw)
    with pytest.raises(AssertionError, match=f"Content SHA-256 mismatch: {TRAVEL_EN}"):
        _validate_manifest(records, sources | {TRAVEL_EN_PATH: changed})


def test_source_metadata_change_is_detected_by_comparison_and_hash(verified_manifest):
    records, sources = verified_manifest
    raw = sources[TRAVEL_EN_PATH]
    before = b"title: Business Travel Policy"
    assert raw.count(before) == 1
    altered = sources | {TRAVEL_EN_PATH: raw.replace(before, b"title: Business Travel Policx", 1)}
    with pytest.raises(AssertionError, match=f"Metadata mismatch for {TRAVEL_EN}: title"):
        _validate_manifest(records, altered)

    changed_records = _change_record(records, TRAVEL_EN, title="Business Travel Policx")
    with pytest.raises(AssertionError, match=f"Content SHA-256 mismatch: {TRAVEL_EN}"):
        _validate_manifest(changed_records, altered)


def test_hash_rejects_invalid_utf8(verified_manifest):
    records, sources = verified_manifest
    invalid = sources | {TRAVEL_EN_PATH: sources[TRAVEL_EN_PATH] + b"\xff"}
    with pytest.raises(UnicodeDecodeError):
        _validate_manifest(records, invalid)


def test_hash_rejects_standalone_carriage_return(verified_manifest):
    records, sources = verified_manifest
    invalid = sources | {TRAVEL_EN_PATH: sources[TRAVEL_EN_PATH] + b"\r"}
    with pytest.raises(AssertionError, match="Unexpected standalone CR"):
        _validate_manifest(records, invalid)


@pytest.mark.parametrize("defect", ["changed-hash", "metadata-only-hash"])
def test_manifest_rejects_wrong_content_hash(verified_manifest, defect):
    records, sources = verified_manifest
    original = next(row["content_sha256"] for row in records if row["document_id"] == TRAVEL_EN)
    if defect == "changed-hash":
        incorrect = ("1" if original[0] == "0" else "0") + original[1:]
    else:
        front, delimiter, _ = _canonical_bytes(sources[TRAVEL_EN_PATH]).partition(b"\n---\n")
        assert delimiter
        incorrect = sha256(front + delimiter).hexdigest()
    assert incorrect != original
    changed = _change_record(records, TRAVEL_EN, content_sha256=incorrect)
    with pytest.raises(AssertionError, match=f"Content SHA-256 mismatch: {TRAVEL_EN}"):
        _validate_manifest(changed, sources)


def test_manifest_rejects_missing_source_file(verified_manifest):
    records, sources = verified_manifest
    incomplete = {path: raw for path, raw in sources.items() if path != TRAVEL_EN_PATH}
    with pytest.raises(AssertionError, match=f"Missing source files.*{re.escape(TRAVEL_EN_PATH)}"):
        _validate_manifest(records, incomplete)


def test_manifest_rejects_unexpected_source_file(verified_manifest):
    records, sources = verified_manifest
    extra = "data/policies/fr/EXTRA-001-FR-V1.md"
    with pytest.raises(AssertionError, match=f"Unexpected source files.*{re.escape(extra)}"):
        _validate_manifest(records, sources | {extra: sources[TRAVEL_EN_PATH]})


def test_manifest_rejects_duplicate_record(verified_manifest):
    records, sources = verified_manifest
    with pytest.raises(AssertionError, match=f"Duplicate manifest record: {records[0]['document_id']}"):
        _validate_manifest([*records, records[0]], sources)


def test_manifest_rejects_omitted_record(verified_manifest):
    records, sources = verified_manifest
    with pytest.raises(AssertionError, match=f"Missing manifest documents.*{records[-1]['document_id']}"):
        _validate_manifest(records[:-1], sources)


def test_manifest_rejects_unauthorized_extra_record(verified_manifest):
    records, sources = verified_manifest
    extra = records[0] | {"document_id": "TRAVEL-001-AR-V1"}
    with pytest.raises(AssertionError, match="Unauthorized document ID.*TRAVEL-001-AR-V1"):
        _validate_manifest([*records, extra], sources)


def test_manifest_rejects_missing_metadata_field(verified_manifest):
    records, sources = verified_manifest
    incomplete = records[0].copy()
    del incomplete["expires_at"]
    with pytest.raises(AssertionError, match="Unexpected manifest fields"):
        _validate_manifest([incomplete, *records[1:]], sources)


def test_manifest_rejects_unsorted_records(verified_manifest):
    records, sources = verified_manifest
    with pytest.raises(AssertionError, match="sorted by document_id"):
        _validate_manifest([records[1], records[0], *records[2:]], sources)


def test_manifest_parser_rejects_duplicate_fields(manifest_text, verified_manifest):
    assert _parse_manifest(manifest_text) == verified_manifest[0]
    field = '"version": 1,'
    assert field in manifest_text
    changed = manifest_text.replace(field, f"{field}\n      {field}", 1)
    with pytest.raises(AssertionError, match="Duplicate manifest field: version"):
        _parse_manifest(changed)


@pytest.mark.parametrize(
    ("document_id", "updates", "error"),
    [
        (TRAVEL_EN, {"title": "Wrong title"}, "Metadata mismatch.*title"),
        (TRAVEL_EN, {"version": True}, "Invalid manifest version"),
        ("TRAVEL-001-FR-V1", {"language": "en"}, "Duplicate policy slot"),
        (TRAVEL_EN, {"version": 9}, "Unauthorized policy slot"),
        (TRAVEL_EN, {"is_active": True}, "Unexpected manifest fields"),
        ("LEAVE-001-FR-V2", {"classification": "PUBLIC"}, "Incorrect classification"),
        ("LEAVE-001-FR-V2", {"allowed_roles": ["hr", "employee"]}, "Incorrect reader ACL"),
        ("ACCESS-001-FR-V2", {"allowed_roles": ["ingestion_admin"]}, "Incorrect reader ACL"),
        ("ACCESS-001-FR-V2", {"allowed_roles": []}, "Incorrect reader ACL"),
        (TRAVEL_EN, {"source_license": "unknown"}, "Incorrect source license"),
        (TRAVEL_EN, {"source_type": "real_policy"}, "Incorrect source type"),
    ],
    ids=[
        "metadata", "noninteger-version", "duplicate-slot", "unauthorized-slot",
        "unknown-field", "classification", "broadened-acl", "unknown-reader", "empty-acl",
        "license", "provenance",
    ],
)
def test_manifest_rejects_invalid_record_fields(verified_manifest, document_id, updates, error):
    records, sources = verified_manifest
    changed = _change_record(records, document_id, **updates)
    with pytest.raises(AssertionError, match=error):
        _validate_manifest(changed, sources)


@pytest.mark.parametrize(
    ("predecessor", "error"),
    [
        ("REMOTE-001-EN-V1", "Wrong-language predecessor"),
        ("DATA-001-FR-V1", "Wrong-family predecessor"),
        ("REMOTE-001-FR-V0", "Missing predecessor"),
        ("REMOTE-001-FR-V2", "Self-supersession"),
        (None, "Unexpected supersession relationships"),
    ],
    ids=["wrong-language", "wrong-family", "unknown-predecessor", "self-link", "missing-link"],
)
def test_manifest_rejects_invalid_predecessors(verified_manifest, predecessor, error):
    records, sources = verified_manifest
    changed = _change_record(records, "REMOTE-001-FR-V2", supersedes=predecessor)
    with pytest.raises(AssertionError, match=error):
        _validate_manifest(changed, sources)


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("C:/external/TRAVEL-001-EN-V1.md", "Invalid source path"),
        ("/external/TRAVEL-001-EN-V1.md", "Invalid source path"),
        ("data/policies/../TRAVEL-001-EN-V1.md", "Invalid source path"),
        ("data\\policies\\en\\TRAVEL-001-EN-V1.md", "Invalid source path"),
        ("data/policies/fr/TRAVEL-001-EN-V1.md", "Source path does not match"),
        ("data/policies/en/wrong.md", "Source path does not match"),
    ],
    ids=["windows-absolute", "posix-absolute", "traversal", "backslashes", "wrong-language", "wrong-name"],
)
def test_manifest_rejects_invalid_source_paths(verified_manifest, path, error):
    records, sources = verified_manifest
    changed = _change_record(records, TRAVEL_EN, source_path=path)
    with pytest.raises(AssertionError, match=error):
        _validate_manifest(changed, sources)
