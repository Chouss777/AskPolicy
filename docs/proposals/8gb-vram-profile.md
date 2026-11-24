# Proposed 8 GiB VRAM profile (NOT PRODUCTION APPROVED)

Current pinned generator remains `qwen3:4b-instruct` until the §18 governance requirements and explicit owner approval are met. This proposal does not edit AGENTS.md, decisions.md, current retrieval models, or runtime configuration.

## Recommended components

| Role | Selection | Execution |
|---|---|---|
| Generator candidate | `qwen3.5:4b` Q4_K_M, observed full digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` | GPU, one request; observed 100% GPU at 4096 context |
| Current embedding | `intfloat/multilingual-e5-small` | CPU; preserve current indexed vectors and 384D E5 contract |
| Experimental Ollama embedding | `qwen3-embedding:0.6b` | Not enabled. Use sequentially with `keep_alive=0` ONLY after separate benchmark, index rebuild, pin and approval |
| Current reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | CPU, disabled in production per Phase 5 decision |

The Qwen3.5 9B published download size (roughly 6.6–7.6 GB across variants) makes loaded-VRAM/KV-cache headroom too tight for a reliable 8 GiB laptop. `granite4.2:8b` is another candidate but lacks AskPolicy comparative quality evidence. The owner's Qwen3.5 4B audit found 3.14 GB reported VRAM, 100% GPU, four successful bounded smoke probes and 60.70 tokens/s in a separate short test. None constitutes a full benchmark.

## Settings for the Ollama daemon

```dotenv
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_CONTEXT_LENGTH=4096
OLLAMA_KEEP_ALIVE=5m
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_QUEUE=8
OLLAMA_NO_CLOUD=1
```

Apply these to the **Ollama server process** (quit/restart Ollama Desktop on Windows); do not assume FastAPI `.env` changes Ollama. The companion PowerShell script previews changes by default and modifies user environment only when passed `-Apply`.

## Future generation API request after authorization

```json
{"model":"qwen3.5:4b","stream":false,"think":false,"keep_alive":"5m","options":{"num_ctx":4096,"num_predict":512,"temperature":0}}
```

Use actual `GenerationOutput.model_json_schema()` as /api/chat `format`. This is only a candidate profile: model selection is not authorized. Bound request duration and response size, verify the digest, and preserve Step 118–120 Pydantic/citation/refusal guards. Account for prompts, complete retrieved evidence and JSON completion within 4096; fail closed on truncation. Increase to 8192 only after verifying VRAM residency under load.

Experimental `/api/embed` options: `{"model":"qwen3-embedding:0.6b","truncate":false,"keep_alive":"0"}`. A single-model residency limit can evict the generator when using an Ollama embedder; CPU E5 avoids this loading penalty. Never combine embeddings produced by a replacement with existing E5 FAISS indexes.

## Required gates

- Inspect `ollama ps` for 100% GPU at real 4096-context workloads and record `nvidia-smi` free/used VRAM across repeated requests; sizes above are not OOM guarantees.
- Compare FR/EN/AR correctness, grounded citations, refusal accuracy, p50/p95 latency on synthetic development cases; never use frozen cases for tuning.
- Pin full installed digest from `/api/tags`. The candidate digest checker is an identity check, NOT proof of §18 approval.
- A model or embedding replacement needs reproducible failure, attempted remedy within current stack, test diff, new acceptance criteria and explicit owner approval under `docs/decisions.md` §18.

References: https://docs.ollama.com/faq , https://docs.ollama.com/api/chat , https://docs.ollama.com/api/embed , https://ollama.com/library/qwen3.5 , https://ollama.com/library/qwen3-embedding