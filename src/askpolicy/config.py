"""Phase 1 configuration schema."""

from pathlib import Path

from pydantic import BaseModel, SecretStr


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
