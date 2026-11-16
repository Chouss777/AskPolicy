"""Application settings schema and explicit JWT runtime configuration."""

from collections.abc import Mapping
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, SecretStr

if TYPE_CHECKING:
    from .auth.jwt import JWTConfig


# DEFERRED (Phase 3): environment loading is not implemented in
# Phase 1. Pydantic v2 moved BaseSettings into the separate
# pydantic-settings package, which is not in the locked stack
# (docs/decisions.md §2). When Phase 3 requires env loading from
# .env, evaluate whether pydantic-settings justifies a stack
# amendment under docs/decisions.md §18. Do not add it without
# following that procedure.
class Settings(BaseModel):
    JWT_SECRET: SecretStr | None = None
    JWT_ISSUER: str | None = None
    JWT_AUDIENCE: str | None = None
    OLLAMA_HOST: str | None = None
    DB_PATH: Path | None = None
    API_PORT: int | None = None
    UI_PORT: int | None = None


def load_jwt_config(environ: Mapping[str, str] | None = None) -> "JWTConfig":
    """Load HS256 configuration from a mapping or the process environment.

    JWT_SECRET, JWT_ISSUER, and JWT_AUDIENCE are required. Invalid values
    raise AuthenticationError with the content-free INVALID_JWT_CONFIG code.
    This loader does not read dotenv files or configure unrelated settings.
    """
    # Keep the legacy Settings schema usable independently of the auth package.
    from .auth.jwt import AuthenticationError, JWTConfig

    source = os.environ if environ is None else environ
    if not isinstance(source, Mapping):
        raise AuthenticationError("INVALID_JWT_CONFIG")
    secret = source.get("JWT_SECRET")
    issuer = source.get("JWT_ISSUER")
    audience = source.get("JWT_AUDIENCE")
    if type(secret) is not str or type(issuer) is not str or type(audience) is not str:
        raise AuthenticationError("INVALID_JWT_CONFIG")
    return JWTConfig(signing_secret=SecretStr(secret), issuer=issuer, audience=audience, algorithm="HS256")
