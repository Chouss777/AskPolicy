from datetime import date
import json
from pathlib import Path
import runpy

from pydantic import ValidationError
import pytest


ROOT = Path(__file__).resolve().parents[2]
# The repository uses uv package=false, so load the actual module from src.
metadata = runpy.run_path(str(ROOT / "src/askpolicy/ingestion/metadata.py"))
parse_policy_metadata = metadata["parse_policy_metadata"]
PolicyMetadata = metadata["PolicyMetadata"]
MetadataValidationError = metadata["MetadataValidationError"]

FIELDS = {
    "policy_id": "FIXTURE-999",
    "document_id": "FIXTURE-999-EN-V2",
    "version": "2",
    "language": "en",
    "title": "Fixture policy",
    "department": "Operations",
    "classification": "INTERNAL",
    "allowed_roles": "[employee, hr]",
    "effective_date": '"2026-01-01"',
    "expires_at": "null",
    "supersedes": "null",
    "approval_status": "approved",
    "source_type": "synthetic_policy",
    "source_license": "project_owned",
}
SOURCE_PATH = "data/policies/en/FIXTURE-999-EN-V2.md"


def _source(fields):
    front_matter = "\n".join(
        f"{key}: {value}" if not value.startswith("\n") else f"{key}:{value}"
        for key, value in fields.items()
    )
    return f"---\n{front_matter}\n---\n\n# Synthetic policy\n\nBody.\n---\n"


@pytest.fixture
def valid_fields():
    fields = FIELDS.copy()
    assert isinstance(parse_policy_metadata(_source(fields), source_path=SOURCE_PATH), PolicyMetadata)
    return fields


@pytest.mark.parametrize(
    ("classification", "role_text", "expected_roles"),
    [
        ("PUBLIC", "[finance]", ("finance",)),
        ("INTERNAL", "[employee, hr]", ("employee", "hr")),
        ("RESTRICTED", "[it_security]", ("it_security",)),
    ],
)
def test_explicit_role_grants_are_independent_of_classification(
    valid_fields, classification, role_text, expected_roles
):
    source = _source(valid_fields | {"classification": classification, "allowed_roles": role_text})
    result = parse_policy_metadata(source)

    assert result.classification == classification
    assert result.allowed_roles == expected_roles


@pytest.mark.parametrize(
    "roles",
    [
        """[ 'employee', "hr", ] # readers""",
        " # readers\n  - 'employee'  # primary\n\n  # alternate\n  - \"hr\"",
        "\n- employee\n- hr",
    ],
    ids=["flow-with-comments", "indented-block", "indentless-block"],
)
def test_supported_role_list_syntax_and_crlf(valid_fields, roles):
    source = _source(valid_fields | {"allowed_roles": roles}).replace("\n", "\r\n")
    assert parse_policy_metadata(source, source_path=SOURCE_PATH).allowed_roles == ("employee", "hr")


@pytest.mark.parametrize(
    ("language", "title", "expected_title"),
    [
        ("fr", "'Politique d''accès : #  Révision' # note", "Politique d'accès : #  Révision"),
        ("ar", '"سياسة التعامل مع البيانات: # توضيح" # note', "سياسة التعامل مع البيانات: # توضيح"),
        ("en", "Policy # note", "Policy"),
        ("en", '"  Internal \\"A\\" policy  " # note', '  Internal "A" policy  '),
    ],
)
def test_unicode_and_quoting_preserve_scalar_values(valid_fields, language, title, expected_title):
    document_id = f"FIXTURE-999-{language.upper()}-V2"
    result = parse_policy_metadata(
        _source(valid_fields | {"language": language, "document_id": document_id, "title": title}),
        source_path=f"data/policies/{language}/{document_id}.md",
    )

    assert result.title == expected_title


def test_typed_metadata_supports_literal_null_and_is_frozen(valid_fields):
    result = parse_policy_metadata(_source(valid_fields))

    assert type(result.version) is int
    assert result.effective_date == date(2026, 1, 1)
    assert type(result.allowed_roles) is tuple
    assert result.expires_at is None
    assert result.supersedes is None
    assert len(type(result).model_fields) == 14
    with pytest.raises(ValidationError) as error:
        result.allowed_roles = ("it_security",)
    assert error.value.errors()[0]["type"] == "frozen_instance"


def test_earlier_predecessor_and_optional_expiration(valid_fields):
    source = _source(valid_fields | {
        "effective_date": "2024-02-29",
        "expires_at": "'2026-12-31' # last validity boundary",
        "supersedes": '"FIXTURE-999-EN-V1"',
        "approval_status": "not_approved",
    })
    result = parse_policy_metadata(source)

    assert result.effective_date == date(2024, 2, 29)
    assert result.expires_at == date(2026, 12, 31)
    assert result.supersedes == "FIXTURE-999-EN-V1"
    assert result.approval_status == "not_approved"


MANIFEST = json.loads((ROOT / "data/corpus_manifest.yaml").read_text(encoding="utf-8"))["documents"]


@pytest.mark.parametrize("record", MANIFEST, ids=[row["document_id"] for row in MANIFEST])
def test_corpus_metadata_matches_the_approved_manifest(record):
    path = record["source_path"]
    source = (ROOT / path).read_text(encoding="utf-8")
    expected = {key: value for key, value in record.items() if key not in {"source_path", "content_sha256"}}

    actual = parse_policy_metadata(source, source_path=path)
    assert actual.model_dump(mode="json") == expected


def _assert_error(code, source, *, source_path=None):
    with pytest.raises(MetadataValidationError) as error:
        parse_policy_metadata(source, source_path=source_path)
    assert error.value.code == code
    return error.value


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("---\n", "# no opening delimiter\n"),
        ("\n---\n\n#", "\n#"),
        ("title: Fixture policy", "title Fixture policy"),
        ("title: Fixture policy", "title: Fixture policy\ntitle: Duplicate"),
        ("title: Fixture policy", "title: \"unterminated"),
        ("title: Fixture policy", r'title: "invalid \q escape"'),
        ("title: Fixture policy", "title: ambiguous: value"),
        ("title: Fixture policy", "title: !custom secret"),
        ("title: Fixture policy", "title: *alias"),
        ("title: Fixture policy", "title: &anchor value"),
        ("title: Fixture policy", "title: |\n  multiline"),
        ("title: Fixture policy", "title: {nested: value}"),
        ("title: Fixture policy", "title:\n  nested: value"),
        ("title: Fixture policy", " title: Fixture policy"),
        ("title: Fixture policy", "title: Fixture\0policy"),
        ("title: Fixture policy", "title: Fixture\rpolicy"),
        ("allowed_roles: [employee, hr]", "allowed_roles: [employee, hr"),
        ("allowed_roles: [employee, hr]", "allowed_roles: [employee,, hr]"),
        ("allowed_roles: [employee, hr]", "allowed_roles:\n  - employee\n   - hr"),
        ("allowed_roles: [employee, hr]", "allowed_roles:\n\t- employee"),
        ("allowed_roles: [employee, hr]", "allowed_roles: [employee]\n  - hr"),
        ("allowed_roles: [employee, hr]", "allowed_roles:\n  - [employee]"),
    ],
    ids=[
        "opening-delimiter", "closing-delimiter", "missing-colon", "duplicate-key",
        "unterminated-quote", "invalid-escape", "ambiguous-scalar", "tag", "alias", "anchor",
        "block-scalar", "flow-mapping", "block-mapping", "mapping-indent",
        "control-character", "standalone-cr", "unclosed-list", "empty-list-item",
        "mixed-indent", "tab-indent", "mixed-list-styles", "nested-list",
    ],
)
def test_malformed_front_matter_is_rejected(valid_fields, before, after):
    source = _source(valid_fields)
    if before == "\n---\n\n#":
        # Remove all later Markdown delimiters too, leaving no closing marker.
        source = source.removesuffix("---\n")
    assert before in source
    _assert_error("MALFORMED_FRONT_MATTER", source.replace(before, after, 1))


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("allowed_roles", "MISSING_ALLOWED_ROLES"),
        ("effective_date", "MISSING_EFFECTIVE_DATE"),
        ("classification", "INVALID_METADATA"),
        ("expires_at", "INVALID_METADATA"),
        ("supersedes", "INVALID_METADATA"),
        ("source_license", "INVALID_METADATA"),
    ],
)
def test_required_keys_cannot_be_omitted(valid_fields, field, code):
    del valid_fields[field]
    _assert_error(code, _source(valid_fields))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("unexpected_key", "value", "INVALID_METADATA"),
        ("allowed_roles", "[]", "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", "null", "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", '"null"', "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", "employee", "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", "[contractor]", "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", "[employee, 'employee']", "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", "[ingestion_admin]", "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", "[true, 1]", "INVALID_ALLOWED_ROLES"),
        ("effective_date", "null", "INVALID_EFFECTIVE_DATE"),
        ("effective_date", "2026-02-29", "INVALID_EFFECTIVE_DATE"),
        ("effective_date", '"2026-1-01"', "INVALID_EFFECTIVE_DATE"),
        ("effective_date", "20260101", "INVALID_EFFECTIVE_DATE"),
        ("classification", "PRIVATE", "UNKNOWN_CLASSIFICATION"),
        ("classification", "null", "UNKNOWN_CLASSIFICATION"),
        ("classification", '""', "UNKNOWN_CLASSIFICATION"),
        ("classification", "true", "UNKNOWN_CLASSIFICATION"),
        ("classification", "1", "UNKNOWN_CLASSIFICATION"),
        ("supersedes", "FIXTURE-999-EN-V2", "INVALID_SUPERSESSION"),
        ("supersedes", "FIXTURE-999-FR-V1", "INVALID_SUPERSESSION"),
        ("supersedes", "OTHER-999-EN-V1", "INVALID_SUPERSESSION"),
        ("supersedes", "FIXTURE-999-EN-V3", "INVALID_SUPERSESSION"),
        ("supersedes", "broken-id", "INVALID_SUPERSESSION"),
        ("supersedes", '"null"', "INVALID_SUPERSESSION"),
        ("document_id", "fixture-999-en-v2", "INVALID_DOCUMENT_ID"),
        ("document_id", "FIXTURE-999-EN-V02", "INVALID_DOCUMENT_ID"),
        ("document_id", "OTHER-999-EN-V2", "INVALID_DOCUMENT_ID"),
        ("document_id", "FIXTURE-999-EN-V1", "INVALID_DOCUMENT_ID"),
        ("document_id", "FIXTURE-999-FR-V2", "INVALID_DOCUMENT_ID"),
        ("policy_id", "FIXTURE", "INVALID_DOCUMENT_ID"),
        ("language", "es", "INVALID_DOCUMENT_ID"),
        ("version", "true", "INVALID_DOCUMENT_ID"),
        ("version", '"2"', "INVALID_DOCUMENT_ID"),
        ("version", "0", "INVALID_DOCUMENT_ID"),
        ("expires_at", "2026-13-01", "INVALID_EXPIRATION_DATE"),
        ("expires_at", '"null"', "INVALID_EXPIRATION_DATE"),
        ("expires_at", "2025-12-31", "INVALID_EXPIRATION_DATE"),
        ("expires_at", "2026-01-01", "INVALID_EXPIRATION_DATE"),
        ("title", '" "', "INVALID_METADATA"),
        ("department", "null", "INVALID_METADATA"),
        ("source_type", "false", "INVALID_METADATA"),
        ("source_license", "123", "INVALID_METADATA"),
        ("approval_status", "superseded", "INVALID_METADATA"),
        ("approval_status", "true", "INVALID_METADATA"),
    ],
)
def test_invalid_metadata_fails_with_a_named_error(valid_fields, field, value, code):
    _assert_error(code, _source(valid_fields | {field: value}))


@pytest.mark.parametrize(
    "source_path",
    [
        "../data/policies/en/FIXTURE-999-EN-V2.md",
        "data/policies/fr/../en/FIXTURE-999-EN-V2.md",
        "/data/policies/en/FIXTURE-999-EN-V2.md",
        "C:/data/policies/en/FIXTURE-999-EN-V2.md",
        r"data\policies\en\FIXTURE-999-EN-V2.md",
        "data/policies/en/FIXTURE-999-EN-V1.md",
        "data/policies/fr/FIXTURE-999-EN-V2.md",
        "data/policies/en/FIXTURE-999-EN-V2.MD",
        "elsewhere/en/FIXTURE-999-EN-V2.md",
    ],
)
def test_invalid_source_path_is_rejected(valid_fields, source_path):
    _assert_error("INVALID_SOURCE_PATH", _source(valid_fields), source_path=source_path)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("allowed_roles", "[confidential_token]", "INVALID_ALLOWED_ROLES"),
        ("allowed_roles", '"confidential_token', "MALFORMED_FRONT_MATTER"),
        ("effective_date", "confidential_token", "INVALID_EFFECTIVE_DATE"),
    ],
)
def test_validation_errors_do_not_expose_restricted_input(valid_fields, field, value, code):
    source = _source(valid_fields | {field: value, "classification": "RESTRICTED"})
    error = _assert_error(code, source)

    assert error.args == (code,)
    assert "confidential_token" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
