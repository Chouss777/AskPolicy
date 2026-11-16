"""Verify the prototype's HS256 tokens and strict identity claims."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import re
from typing import Literal, get_args

import jwt
from jwt.utils import base64url_decode
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


ReadRole = Literal["employee", "hr", "finance", "it_security"]
Capability = Literal["ingestion_admin"]
READ_ROLES: frozenset[ReadRole] = frozenset(get_args(ReadRole))
_COMPACT_JWT = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_PLACEHOLDERS = {"placeholder", "changeme", "change-me", "replace-me", "your-secret", "todo"}
_SIGNATURES = jwt.PyJWS()


class AuthenticationError(ValueError):
    """A content-free failure identified by a stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _valid_config_text(value: object) -> bool:
    if type(value) is not str or not value or value != value.strip() or not value.isprintable():
        return False
    label = value.casefold().replace("_", "-")
    return not any(label == marker or label.startswith(marker + "-") for marker in _PLACEHOLDERS)


@dataclass(frozen=True, slots=True)
class JWTConfig:
    """Explicit server configuration. HS256 requires a 32-byte or longer key."""

    signing_secret: SecretStr = field(repr=False)
    issuer: str
    audience: str
    algorithm: Literal["HS256"]

    def __post_init__(self) -> None:
        if (
            type(self.signing_secret) is not SecretStr
            or not _valid_config_text(self.issuer)
            or not _valid_config_text(self.audience)
            or type(self.algorithm) is not str
            or self.algorithm != "HS256"
        ):
            raise AuthenticationError("INVALID_JWT_CONFIG")
        secret = self.signing_secret.get_secret_value()
        if not _valid_config_text(secret) or len(secret.encode("utf-8")) < 32 or len(set(secret)) < 2:
            raise AuthenticationError("INVALID_JWT_CONFIG")
        try:
            jwt.get_algorithm_by_name("HS256").prepare_key(secret)
        except jwt.PyJWTError:
            raise AuthenticationError("INVALID_JWT_CONFIG") from None


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedIdentity:
    subject: str
    grants: tuple[ReadRole, ...]
    capabilities: tuple[Capability, ...]


class _TokenClaims(BaseModel):
    model_config = ConfigDict(strict=True, hide_input_in_errors=True)

    sub: str
    iss: str
    aud: str
    iat: int
    exp: int
    nbf: int | None = None
    roles: list[ReadRole]
    capabilities: list[Capability] = Field(default_factory=list)

    @field_validator("sub")
    @classmethod
    def _validate_subject(cls, subject: str) -> str:
        if not subject or subject != subject.strip() or not subject.isprintable():
            raise ValueError("Invalid subject")
        return subject

    @field_validator("nbf", mode="before")
    @classmethod
    def _validate_not_before(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("Expected an integer NumericDate")
        return value

    @field_validator("roles", "capabilities")
    @classmethod
    def _validate_grants(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Duplicate grant")
        return values


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("Duplicate JWT field")
        fields[name] = value
    return fields


def _reject_json_constant(_value: str) -> None:
    raise ValueError("Invalid JSON constant")


def _parse_json(raw: bytes) -> object:
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant,
    )


@dataclass(frozen=True, slots=True)
class JWTVerifier:
    """Verify compact signed JWTs using only the injected HS256 allow-list.

    Signed claims require a subject, issuer, single string audience, integer
    iat/exp, and a JSON roles array. Optional nbf is an integer. Capabilities
    are a separate optional JSON array. Both grant arrays forbid duplicates.
    Empty read grants are valid and cannot activate a document-read role.
    """

    config: JWTConfig

    def __post_init__(self) -> None:
        if type(self.config) is not JWTConfig:
            raise AuthenticationError("INVALID_JWT_CONFIG")

    def verify(self, token: str | None) -> VerifiedIdentity:
        if token is None or token == "":
            raise AuthenticationError("MISSING_TOKEN")
        if type(token) is not str or _COMPACT_JWT.fullmatch(token) is None:
            raise AuthenticationError("INVALID_TOKEN")

        try:
            # PyJWT verifies the signature. Separately reject ambiguous JSON
            # headers and use strict types for the signed payload.
            raw = _SIGNATURES.decode(
                token, key=self.config.signing_secret.get_secret_value(),
                algorithms=(self.config.algorithm,),
                options={"verify_signature": True, "enforce_minimum_key_length": True},
            )
            _parse_json(base64url_decode(token.split(".", 1)[0]))
        except (jwt.PyJWTError, ValueError, RecursionError):
            raise AuthenticationError("INVALID_TOKEN") from None
        try:
            claims = _TokenClaims.model_validate(_parse_json(raw))
        except (ValueError, RecursionError):
            raise AuthenticationError("INVALID_CLAIMS") from None

        if claims.iss != self.config.issuer:
            raise AuthenticationError("INVALID_ISSUER")
        if claims.aud != self.config.audience:
            raise AuthenticationError("INVALID_AUDIENCE")
        if claims.exp <= claims.iat or (claims.nbf is not None and claims.nbf >= claims.exp):
            raise AuthenticationError("INVALID_CLAIMS")
        now = datetime.now(UTC).timestamp()
        if now >= claims.exp:
            raise AuthenticationError("TOKEN_EXPIRED")
        if now < claims.iat or (claims.nbf is not None and now < claims.nbf):
            raise AuthenticationError("TOKEN_NOT_YET_VALID")
        return VerifiedIdentity(claims.sub, tuple(claims.roles), tuple(claims.capabilities))
