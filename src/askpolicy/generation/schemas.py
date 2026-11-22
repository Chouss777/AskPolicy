"""Validate LLM output structure only, without trusting its content.

Validation does not establish citation existence, authorization, membership in
supplied evidence, snapshot identity, quotation accuracy, or semantic support.
Model-provided status never overrides server authorization or controlled-refusal
rules. Those checks belong to subsequent security and citation-validation steps.
Do not log model payloads or raw validation-error details.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Citation(BaseModel):
    """A structurally valid citation whose quotation remains unchanged."""

    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)

    chunk_id: str
    quote: str

    @field_validator("chunk_id", "quote")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Citation fields must contain non-whitespace text")
        return value


class GenerationOutput(BaseModel):
    """An answer with citations, or an explanation-free refusal signal."""

    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)

    status: Literal["answer", "refusal"]
    answer: str | None
    citations: list[Citation]

    @model_validator(mode="after")
    def _consistent_output(self) -> "GenerationOutput":
        if self.status == "answer":
            if self.answer is None or not self.answer.strip():
                raise ValueError("Answer must contain non-whitespace text")
            if not self.citations:
                raise ValueError("Answer requires at least one citation")
        elif self.answer is not None or self.citations:
            raise ValueError("Refusal requires a null answer and empty citations")
        return self
