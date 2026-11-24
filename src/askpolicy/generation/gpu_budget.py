"""Review-only 8 GiB local Ollama resource profile for AskPolicy.

Not wired into the production client. The locked generation model stays
qwen3:4b-instruct until the §18 stack-change requirements are met. No model
pull, loading, network, retrieval, inference or authorization occurs here.
"""

from dataclasses import dataclass
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class GpuBudgetApprovalError(ValueError):
    """Content-free refusal to use an unapproved or unexpected model."""

    def __init__(self) -> None:
        super().__init__("GPU_PROFILE_NOT_APPROVED")


@dataclass(frozen=True, slots=True)
class EightGiBProfile:
    """Opt-in configuration proposal; never considered production authorization."""

    production_model: str = "qwen3:4b-instruct"
    proposed_generator: str = "qwen3.5:4b"
    # Locally observed by the owner on 2026-11-22. This is not an approved pin.
    observed_generator_digest: str = (
        "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"
    )
    active_embedding_backend: str = "e5_cpu"
    active_embedding_model: str = "intfloat/multilingual-e5-small"
    proposed_ollama_embedder: str = "qwen3-embedding:0.6b"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_device: str = "cpu"
    reranker_enabled: bool = False
    generation_num_ctx: int = 4096
    generation_num_predict: int = 512
    generation_keep_alive: str = "5m"
    embedding_keep_alive: str = "0"
    thinking_enabled: bool = False
    max_loaded_models: int = 1
    num_parallel: int = 1
    max_queue: int = 8

    def daemon_environment(self) -> dict[str, str]:
        """Environment intended for Ollama server start (not app .env loader)."""
        return {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_CONTEXT_LENGTH": str(self.generation_num_ctx),
            "OLLAMA_KEEP_ALIVE": self.generation_keep_alive,
            "OLLAMA_MAX_LOADED_MODELS": str(self.max_loaded_models),
            "OLLAMA_NUM_PARALLEL": str(self.num_parallel),
            "OLLAMA_MAX_QUEUE": str(self.max_queue),
            "OLLAMA_NO_CLOUD": "1",
        }

    def candidate_chat_options(self) -> dict[str, object]:
        """Request options only. NOT an authorized model selection or request."""
        return {
            "model": self.proposed_generator,
            "stream": False,
            "think": self.thinking_enabled,
            "keep_alive": self.generation_keep_alive,
            "options": {
                "num_ctx": self.generation_num_ctx,
                "num_predict": self.generation_num_predict,
                "temperature": 0,
            },
        }

    def candidate_embedding_options(self) -> dict[str, object]:
        """Experimental-only. Never mix output with the pinned E5 FAISS index."""
        return {
            "model": self.proposed_ollama_embedder,
            "truncate": False,
            "keep_alive": self.embedding_keep_alive,
        }

    def check_proposed_generator_digest(
        self, *, installed_digest: str, owner_approved_digest: str | None
    ) -> None:
        """Fail closed until owner explicitly records the observed exact pin."""
        if (
            type(installed_digest) is not str
            or type(owner_approved_digest) is not str
            or _SHA256.fullmatch(installed_digest) is None
            or _SHA256.fullmatch(owner_approved_digest) is None
            or installed_digest != owner_approved_digest
            or installed_digest != self.observed_generator_digest
        ):
            raise GpuBudgetApprovalError()


EIGHT_GIB_PROPOSAL = EightGiBProfile()
