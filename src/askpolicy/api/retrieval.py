"""Strict HTTP retrieval schemas and the shared authentication boundary."""

import json
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..auth.context import AuthenticatedContext
from ..auth.jwt import ReadRole
from .retrieval_runtime import QueryLimitError


Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Rank = Annotated[int, Field(ge=1, le=30)]


class SearchQuery(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    query: str = Field(min_length=1, max_length=4096)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("EMPTY_QUERY")
        return value


class SearchResult(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    child_id: str = Field(pattern=r"^child:[0-9a-f]{64}$")
    parent_id: str = Field(pattern=r"^parent:[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^[A-Z]+-[0-9]{3}-(FR|EN|AR)-V[1-9][0-9]*$")
    rrf_score: float = Field(gt=0, le=2/61, allow_inf_nan=False)
    bm25_rank: Rank | None
    faiss_rank: Rank | None

    @model_validator(mode="after")
    def rank_score(self):
        ranks = (self.bm25_rank, self.faiss_rank)
        if all(rank is None for rank in ranks) or self.rrf_score != sum(1/(60+r) for r in ranks if r is not None):
            raise ValueError("INVALID_RANK_SCORE")
        return self


class SearchResponse(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    release_id: Digest
    snapshot_id: Digest
    active_role: ReadRole
    results: tuple[SearchResult, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def unique_ordered(self):
        if len({h.child_id for h in self.results}) != len(self.results):
            raise ValueError("DUPLICATE_CHILD")
        if self.results != tuple(sorted(self.results, key=lambda h:(-h.rrf_score,h.child_id))):
            raise ValueError("INVALID_ORDER")
        return self


class RetrievalNoStore:
    """Pure ASGI response-header wrapper, including auth and validation errors."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def no_store(message):
            if message["type"] == "http.response.start" and scope.get("path","").startswith("/retrieval/"):
                headers = [(key,value) for key,value in message.get("headers",[]) if key.lower()!=b"cache-control"]
                message = {**message,"headers":[*headers,(b"cache-control",b"no-store")]}
            await send(message)
        await self.app(scope, receive, no_store)


def register_retrieval(app, require_context, runtime) -> None:
    async def validated_query(request: Request, context: AuthenticatedContext = Depends(require_context)) -> SearchQuery:
        try:
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > 32768:
                    raise ValueError("REQUEST_TOO_LARGE")
                raw.extend(chunk)
            return SearchQuery.model_validate(json.loads(raw))
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid query") from None

    @app.post("/retrieval/search", response_model=SearchResponse)
    def search(
        context: AuthenticatedContext = Depends(require_context),
        payload: SearchQuery = Depends(validated_query),
    ) -> SearchResponse:
        try:
            return runtime.search(context, payload.query)
        except QueryLimitError:
            raise HTTPException(status_code=422, detail="Invalid query") from None
        except Exception:
            raise HTTPException(status_code=503, detail="Retrieval unavailable") from None

    app.add_middleware(RetrievalNoStore)
