"""Validate the restricted YAML front matter used by AskPolicy Markdown sources."""

from datetime import date
from io import StringIO
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator


Language = Literal["fr", "en", "ar"]
Classification = Literal["PUBLIC", "INTERNAL", "RESTRICTED"]
ReadRole = Literal["employee", "hr", "finance", "it_security"]
ApprovalStatus = Literal["approved", "not_approved"]

_POLICY_ID = re.compile(r"[A-Z]+-[0-9]{3}")
_DOCUMENT_ID = re.compile(r"([A-Z]+-[0-9]{3})-(FR|EN|AR)-V([1-9][0-9]*)")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_INTEGER = re.compile(r"0|[1-9][0-9]*")
_NUMBER = re.compile(
    r"[+-]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9][0-9_]*)(?:e[+-]?[0-9_]+)?"
    r"|[+-]?(?:0[xob][0-9a-f_]+|\.(?:inf|nan))",
    re.IGNORECASE,
)
_FIELD = re.compile(r"([a-zA-Z_][a-zA-Z_0-9-]*):(?:[ \t]+(.*))?")
_BLOCK_ITEM = re.compile(r"( *)-[ \t]+(.*)")
_FLOW_LIST = re.compile(r"\[([^\[\]]*)\](?:[ \t]+#.*)?")
_SINGLE_QUOTED = re.compile(r"'((?:[^']|'')*)'(?:[ \t]+#.*)?")
_DOUBLE_QUOTED = re.compile(r'("(?:[^"\\]|\\.)*")(?:[ \t]+#.*)?')
_COMMENT = re.compile(r"[ \t]+#")
_UNSUPPORTED_CHAR = re.compile(r"[\x00-\x08\x0a-\x1f\x7f-\x9f\u2028\u2029\ud800-\udfff\ufffe\uffff]")


class MetadataValidationError(ValueError):
    """A controlled failure identified by a stable, content-free code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class PolicyMetadata(BaseModel):
    """Validated metadata for one document, independent of corpus membership."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)

    policy_id: str
    version: Annotated[int, Field(gt=0)]
    language: Language
    document_id: str
    title: str
    department: str
    classification: Classification
    allowed_roles: Annotated[tuple[ReadRole, ...], Field(min_length=1)]
    effective_date: date
    expires_at: date | None
    supersedes: str | None
    approval_status: ApprovalStatus
    source_type: str
    source_license: str

    @field_validator("policy_id")
    @classmethod
    def _validate_policy_id(cls, value: str) -> str:
        if not _POLICY_ID.fullmatch(value):
            raise ValueError("Invalid policy identity")
        return value

    @field_validator("document_id")
    @classmethod
    def _validate_document_id(cls, value: str, info: ValidationInfo) -> str:
        match = _DOCUMENT_ID.fullmatch(value)
        if match is None:
            raise ValueError("Invalid document identity")
        family, language, version = match.groups()
        if (
            info.data.get("policy_id", family) != family
            or info.data.get("language", language.lower()) != language.lower()
            or str(info.data.get("version", version)) != version
        ):
            raise ValueError("Inconsistent document identity")
        return value

    @field_validator("title", "department", "source_type", "source_license")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Expected nonempty text")
        return value

    @field_validator("allowed_roles")
    @classmethod
    def _unique_roles(cls, roles: tuple[ReadRole, ...]) -> tuple[ReadRole, ...]:
        if len(roles) != len(set(roles)):
            raise ValueError("Duplicate read role")
        return roles

    @field_validator("expires_at")
    @classmethod
    def _validate_expiration(cls, value: date | None, info: ValidationInfo) -> date | None:
        effective = info.data.get("effective_date")
        if value is not None and effective is not None and value <= effective:
            raise ValueError("Expiration must be after the effective date")
        return value

    @field_validator("supersedes")
    @classmethod
    def _validate_supersession(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        match = _DOCUMENT_ID.fullmatch(value)
        if match is None:
            raise ValueError("Invalid predecessor identity")
        family, language, version = match.groups()
        if (
            family != info.data.get("policy_id")
            or language.lower() != info.data.get("language")
            or int(version) >= info.data.get("version", 0)
        ):
            raise ValueError("Expected an earlier version in the same family and language")
        return value


def _malformed() -> MetadataValidationError:
    return MetadataValidationError("MALFORMED_FRONT_MATTER")


def _quoted_scalar(value: str) -> str:
    if value.startswith("'"):
        match = _SINGLE_QUOTED.fullmatch(value)
        if match is None:
            raise _malformed()
        return match[1].replace("''", "'")

    match = _DOUBLE_QUOTED.fullmatch(value)
    if match is not None:
        try:
            decoded = json.loads(match[1])
        except json.JSONDecodeError:
            pass
        else:
            if not _UNSUPPORTED_CHAR.search(decoded):
                return decoded
    raise _malformed()


def _scalar(value: str) -> str | int | bool | None:
    if value.startswith(("'", '"')):
        return _quoted_scalar(value)

    value = _COMMENT.split(value, maxsplit=1)[0].rstrip(" \t")
    if (
        not value
        or value[0] in "!&*{}[],|>@\x60#%\"'"
        or value in {"-", "?", ":", "---", "..."}
        or value.startswith(("- ", "-\t", "? ", "?\t", ": ", ":\t"))
        or re.search(r":(?:[ \t]|$)", value)
    ):
        raise _malformed()

    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE", "false", "False", "FALSE"}:
        return value.lower() == "true"
    if _INTEGER.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            pass
        raise _malformed()
    if _NUMBER.fullmatch(value):
        raise _malformed()
    return value


def _flow_roles(value: str) -> tuple[object, ...]:
    match = _FLOW_LIST.fullmatch(value)
    if match is None:
        raise _malformed()
    items = match[1].strip(" \t")
    if items.endswith(","):
        items = items[:-1].rstrip(" \t")
        if not items:
            raise _malformed()
    if not items:
        return ()
    # Role names have no commas, brackets, or comment markers, even when quoted.
    tokens = items.split(",")
    if any("#" in token for token in tokens):
        raise _malformed()
    return tuple(_scalar(token.strip(" \t")) for token in tokens)


def _front_matter(markdown: str) -> dict[str, object]:
    # Read only the opening lines. The Markdown body may itself contain delimiters.
    lines = iter(StringIO(markdown, newline="\n"))
    opening = next(lines, "")
    if opening not in {"---\n", "---\r\n"}:
        raise _malformed()

    fields: dict[str, object] = {}
    block_roles: list[object] | None = None
    block_indent: int | None = None
    for raw in lines:
        line = raw.removesuffix("\n").removesuffix("\r") if raw.endswith("\n") else raw
        if _UNSUPPORTED_CHAR.search(line):
            raise _malformed()
        if line == "---":
            if block_roles is not None:
                fields["allowed_roles"] = tuple(block_roles) if block_roles else None
            return fields
        if not line.strip(" \t") or line.lstrip(" \t").startswith("#"):
            continue

        item = _BLOCK_ITEM.fullmatch(line)
        if item is not None:
            if block_roles is None:
                raise _malformed()
            indent = len(item[1])
            if block_indent is not None and indent != block_indent:
                raise _malformed()
            block_indent = indent
            block_roles.append(_scalar(item[2].rstrip(" \t")))
            continue

        if block_roles is not None:
            fields["allowed_roles"] = tuple(block_roles) if block_roles else None
            block_roles = None

        field = _FIELD.fullmatch(line)
        if field is None:
            raise _malformed()
        key, raw_value = field.groups()
        if key in fields:
            raise _malformed()
        if key not in PolicyMetadata.model_fields:
            raise MetadataValidationError("INVALID_METADATA")
        value = (raw_value or "").rstrip(" \t")
        if key == "allowed_roles" and (not value or value.startswith("#")):
            block_roles = []
        elif key == "allowed_roles" and value.startswith("["):
            fields[key] = _flow_roles(value)
        else:
            fields[key] = _scalar(value)

    raise _malformed()


_FIELD_ERRORS = {
    "policy_id": "INVALID_DOCUMENT_ID",
    "document_id": "INVALID_DOCUMENT_ID",
    "version": "INVALID_DOCUMENT_ID",
    "language": "INVALID_DOCUMENT_ID",
    "classification": "UNKNOWN_CLASSIFICATION",
    "allowed_roles": "INVALID_ALLOWED_ROLES",
    "effective_date": "INVALID_EFFECTIVE_DATE",
    "expires_at": "INVALID_EXPIRATION_DATE",
    "supersedes": "INVALID_SUPERSESSION",
}


def _parse_date(value: object, code: str) -> date:
    if isinstance(value, str) and _DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise MetadataValidationError(code)


def parse_policy_metadata(
    markdown: str, *, source_path: str | None = None
) -> PolicyMetadata:
    """Parse one Markdown source and return frozen, typed metadata.

    The supported subset uses a flat mapping, single-line scalars (including
    single quotes and JSON-style double quotes), and flow or block role lists.
    Unsupported YAML syntax fails with MALFORMED_FRONT_MATTER. The optional
    source path is the canonical POSIX path under data/policies.
    """
    fields = _front_matter(markdown)
    if "allowed_roles" not in fields:
        raise MetadataValidationError("MISSING_ALLOWED_ROLES")
    if "effective_date" not in fields:
        raise MetadataValidationError("MISSING_EFFECTIVE_DATE")

    fields["effective_date"] = _parse_date(fields["effective_date"], "INVALID_EFFECTIVE_DATE")
    if fields.get("expires_at") is not None:
        fields["expires_at"] = _parse_date(fields["expires_at"], "INVALID_EXPIRATION_DATE")

    try:
        metadata = PolicyMetadata.model_validate(fields)
    except ValidationError as error:
        # Keep only the model-defined location and error type. Pydantic's
        # original error may carry restricted source metadata.
        issue = error.errors(include_input=False, include_context=False, include_url=False)[0]
        field = issue["loc"][0]
        code = "INVALID_METADATA" if issue["type"] == "missing" else _FIELD_ERRORS.get(
            field, "INVALID_METADATA"
        )
    else:
        if source_path is not None and source_path != (
            f"data/policies/{metadata.language}/{metadata.document_id}.md"
        ):
            raise MetadataValidationError("INVALID_SOURCE_PATH")
        return metadata

    raise MetadataValidationError(code)
