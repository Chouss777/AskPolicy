# AGENTS.md — AskPolicy

## 1. Prime Directive — Verified Engineering

This directive has the highest priority among all project-level instructions in this file.

Never implement a feature without:
1. Its associated tests.
2. Explicit, measurable acceptance criteria that CI can verify.

Never claim a feature works based solely on a successful demonstration.

**A demo is an anecdote. A test is evidence.**

If a feature cannot be reduced to a verifiable acceptance criterion, it is not yet sufficiently specified. Stop and ask the human owner before implementing it.

Rules:
- Define acceptance criteria before writing implementation code.
- Implement features and their associated tests in the same task.
- Never claim tests passed unless they were actually executed.
- Never weaken assertions, remove failing tests, or modify expected outputs merely to obtain a passing suite.
- Never change frozen evaluation labels to improve reported metrics.
- Never invent benchmarks, security outcomes, latency measurements, or execution output.
- Never report incomplete functionality as production-ready.
- Never silently change architectural decisions.
- If an acceptance test fails, diagnose the failure before continuing.
- Temporary `xfail` markers require documented reasons and must be removed when implementation is complete.
- No skipped or xfailed security-critical tests are permitted at release.
- Every behavioral or security invariant must name an enforcing test in this file or `docs/decisions.md`.

The authoritative feature acceptance matrix is in `docs/decisions.md` §11.

## 2. Project Identity

**Project:** AskPolicy — Secure Multilingual Enterprise RAG.

**Purpose:** Answer internal policy questions using role-based access control, verified citations, controlled refusals, and reproducible evaluation.

**Organization:** Atlas Services, a fictional Moroccan company with 200 employees. All policies, identities, and scenarios are synthetic. Never use real company or personal data.

**Languages:** FR and EN for all eight policy families; AR for Remote Work, Leave, Expenses, and Data Handling.

Core objectives:
1. Retrieve relevant authorized policy evidence.
2. Prevent unauthorized document retrieval and disclosure.
3. Generate answers grounded in authorized evidence.
4. Validate citations before returning any answer.
5. Refuse when authorized evidence is insufficient or invalid.
6. Evaluate retrieval, generation, security, and latency reproducibly.

Do not expand scope without explicit human approval. Enforcement: `tests/integration/test_corpus_manifest.py`, `tests/eval/test_dataset_split.py`.

## 3. Locked Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Authentication | PyJWT + pwdlib[argon2] |
| Vector retrieval | FAISS-cpu, `IndexFlatIP` |
| Vector normalization | L2-normalized embeddings |
| Lexical retrieval | BM25S |
| Embeddings | `intfloat/multilingual-e5-small` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (conditional) |
| Local LLM | Ollama + `qwen3:4b-instruct` |
| Metadata | SQLite |
| UI | Streamlit (HTTP client of FastAPI only) |
| Containerization | Docker Compose + named volumes |
| Testing | pytest |
| Dependency management | uv with `uv.lock` |
| Automation and CI | Makefile + GitHub Actions |

Pin the exact embedding and reranker revisions; pin and record the installed Ollama model digest. Record the versions and digest in evaluation manifests; reject unapproved mismatches.

The reranker is **conditional**: enable in the deployed pipeline only after development-set retrieval quality and measured latency justify it.

Do not introduce LangChain, ChromaDB, Qdrant, Celery, Redis, Kubernetes, vLLM, paid cloud LLM APIs, or agent-orchestration frameworks. Other new runtime dependencies require explicit approval. Follow the stack-change procedure in `docs/decisions.md` §18.

Enforcement: `tests/unit/test_stack_contract.py`, `tests/eval/test_reranker_decision.py`.

## 4. Module Boundaries

- `src/askpolicy/api/`: HTTP routes, request validation, response serialization.
- `src/askpolicy/auth/`: JWT validation, identity and active-role resolution.
- `src/askpolicy/ingestion/`: Markdown parsing, chunking, metadata validation, ingestion.
- `src/askpolicy/retrieval/`: BM25S, FAISS, RRF, conditional reranking.
- `src/askpolicy/generation/`: Ollama integration, structured generation.
- `src/askpolicy/security/`: ACL checks, citation validation, refusal safeguards.
- `src/askpolicy/storage/`: SQLite and corpus/index snapshots.
- `src/askpolicy/observability/`: request IDs, structured logs, timings.
- `src/askpolicy/config.py`: validated configuration.
- `ui/`: Streamlit display only; calls FastAPI over HTTP.
- `eval/`: datasets, configurations, metrics, runners, reports.
- `tests/`: unit, integration, security, evaluation-contract checks.
- `docs/`: architecture, decisions, threat model, evaluation methodology.

Never collapse retrieval, generation, authentication, or security into one module. Never implement document authorization only in UI or route handlers. Never allow direct Streamlit-to-SQLite/FAISS/BM25S access or LLM-based permission decisions. Do not create extra services outside the locked deployment architecture.

Enforcement: `tests/unit/test_architecture_boundaries.py`, `tests/integration/test_ui_api_boundary.py`, `tests/security/test_role_scoping.py`.

## 5. Corpus Contract

**Full scope:** exactly 26 versioned Markdown policies: 20 active (8 FR, 8 EN, 4 AR) and 6 superseded.

Families: Remote Work, Leave, Expenses, Travel, Access Control, Data Handling, Incident Reporting, Procurement.

AR families: Remote Work, Leave, Expenses, Data Handling.

Superseded versions:
- Remote Work FR V1 and EN V1.
- Expenses FR V1.
- Access Control FR V1.
- Leave FR V1.
- Data Handling FR V1.

Use Markdown with YAML front matter. Required document metadata:
`policy_id`, `document_id`, `version`, `language`, `title`, `department`, `classification`, `allowed_roles`, `effective_date`, `expires_at`, `supersedes`, `approval_status`, `source_type`, `source_license`.

Classification values: `PUBLIC`, `INTERNAL`, `RESTRICTED`. Permissions are explicit even for generally readable documents; no default permission is inferred from a classification. Missing/unknown security metadata is rejected.

`data/corpus_manifest.yaml` must list each document with metadata, source path, version relationships, source license, and SHA-256 hash. Preserve document/version, section hierarchy, parent ID, child ID, original text offsets, authorization metadata, and snapshot ID in chunk records.

Parse headings, lists, procedures, and tables without losing table headers or row relationships. A parent expansion must not cross document, version, or ACL boundaries. Keep stable chunk IDs for deterministic rebuilds.

Enforcement: `tests/unit/test_metadata_validation.py`, `tests/integration/test_corpus_manifest.py`, `tests/integration/test_chunk_boundaries.py`, `tests/integration/test_version_selection.py`.

## 6. SQLite, Versioning, and Snapshots

Version-controlled Markdown files are canonical source documents. **SQLite is the validated runtime source of truth**; FAISS and BM25S are derived, rebuildable indexes.

Searchable at snapshot validity time iff a policy is **approved AND effective AND unexpired AND not superseded**. Current-policy search excludes superseded versions. Historical as-of-date search is out of scope.

Every index must declare its exact corpus snapshot ID. Both role-scoped BM25S and FAISS indexes must match the published SQLite snapshot, including embedding revision and chunk identities. Any mismatch or missing active index causes retrieval to fail closed with **HTTP 503**, not silent fallback.

Build and verify new indexes before atomic snapshot publication; a failed build must leave the previous valid published snapshot intact. Do not partially modify an active index. If a time-based transition makes the active snapshot invalid, republish a valid snapshot or refuse affected retrieval until corrected. Unpublished Markdown modifications must not silently change published answers.

Hash deterministically from canonical content/configuration, excluding timestamps, temporary paths, and nondeterministic ordering.

Enforcement: `tests/integration/test_ingestion_deterministic.py`, `tests/integration/test_snapshot_consistency.py`, `tests/integration/test_snapshot_publication.py`, `tests/integration/test_index_rebuild.py`, `tests/integration/test_version_selection.py`.

## 7. FAISS and E5 Invariants

Use FAISS-cpu `IndexFlatIP` with L2-normalized query and passage embeddings. With unit vectors, the inner-product score equals cosine similarity. **It is not an L2 distance:** do not apply `sqrt()` and do not label raw similarity as calibrated confidence percentage.

Use the identical pinned revision of `intfloat/multilingual-e5-small` at indexing and query time.

Mandatory encoder prefixes:
- Query input: `query: `
- Passage input: `passage: `

Every indexed vector must have L2 norm in `[0.999, 1.001]`. Validate model input length according to the tokenizer, preserve original text for exact citations, and record the embedding revision in index metadata.

Enforcement: `tests/unit/test_faiss_normalization.py`, `tests/unit/test_e5_prefix.py`, `tests/integration/test_embedding_revision.py`.

## 8. Retrieval Invariants

Support BM25S, FAISS, RRF, and an optionally enabled Cross-Encoder reranker. Search only the active, authorized role-scoped index; **never** global-search-then-filter. Recheck ACL before reranking, before generation, and before source return. Restrict parent expansion to authorized evidence. Deduplicate by chunk ID; break ranking ties deterministically.

Initial budgets: BM25 top 30; dense top 30; RRF top 20 using constant 60; optional rerank top 20; select up to five authorized evidence results for generation subject to context budget. Changes to these initial values require recorded development-set evidence and approval.

Use identical corpus snapshots, query sets, relevance labels, and ACL conditions for retrieval ablations: BM25, dense, hybrid RRF, and hybrid-plus-rerank. Keep all four available for evaluation even when the deployed pipeline omits reranking.

Enforcement: `tests/security/test_role_scoping.py`, `tests/security/test_parent_expansion.py`, `tests/unit/test_rrf.py`, `tests/eval/test_ablation.py`, `tests/eval/test_reranker_decision.py`.

## 9. Authentication and RBAC

Four document-read roles: `employee`, `hr`, `finance`, `it_security`. Separate administrative capability: `ingestion_admin`, which **does not** grant document read access through retrieval/source APIs.

Exactly **one active read role** per authenticated request/session. A role is allowed only if the backend verifies it is among the identity's grants. Never trust role declarations from a request body, natural-language query, or Streamlit state independently of verified JWT grants.

Switching among granted roles may occur without reauthentication while the session is valid, but requires server-side verification and invalidation of all role-dependent cached answers, chunks, and citations.

Verify JWT signature, expiry, issuer, and audience. Missing/expired/forged token → HTTP 401. Ungranted active role → HTTP 403. Missing/unknown classifications or grants → deny. Every source endpoint requires fresh authorization. Administrative ingestion privilege confers no implicit document-reading privilege.

Enforcement: `tests/security/test_auth.py`, `tests/security/test_role_grant.py`, `tests/security/test_role_switch.py`, `tests/security/test_default_deny.py`, `tests/security/test_admin_isolation.py`, `tests/security/test_source_access.py`, `tests/integration/test_ui_cache_isolation.py`.

## 10. Generation, Citations, and Refusals

Use Ollama `qwen3:4b-instruct` with its approved pinned digest. Generate strict structured output validated by Pydantic v2 before rendering.

A citation passes mechanical validation only when:
1. Its `chunk_id` exists in the **authorized retrieved evidence actually supplied** for this request.
2. The chunk belongs to the current published snapshot and remains authorized.
3. Its verbatim quotation is an **exact substring** of the stored original chunk text.

Reject invented IDs, unauthorized/out-of-set references, mismatched quotes, and invalid structured output. **Fail closed:** any material citation failure results in controlled refusal, never an unvalidated partial answer. Semantic entailment is **not** guaranteed by an exact quote; measure substantive claim support separately on manually reviewed cases.

Refuse if authorized evidence is insufficient, policies conflict without a documented resolution, version validity is uncertain, or required validation fails. Unavailable and unauthorized evidence must use indistinguishable external refusal wording where document-existence disclosure is a risk; do not reveal restricted titles or IDs.

Enforcement: `tests/unit/test_citation_validator.py`, `tests/integration/test_fail_closed.py`, `tests/unit/test_generation_schema.py`, `tests/security/test_refusal_indistinguishability.py`, `tests/eval/test_e2e.py`.

## 11. Threats, Logging, and Data Handling

Treat user prompts and retrieved documents as **untrusted data**, not sources of authorization or system instructions. Prompt injection may not change the active role, authorize tools, or reveal restricted evidence.

Never log raw JWTs, signing secrets, passwords, complete prompts, restricted chunk text, or confidential answer payloads. Log only bounded operational metadata: request ID, role, snapshot, status, validation result, result count, and latency. Use synthetic identities/data only.

Security regression checks must pass before accepting any PR that changes auth, retrieval, ACLs, citations, or source access.

Enforcement: `tests/security/test_prompt_injection.py`, `tests/security/test_logging_hygiene.py`, `tests/security/test_role_scoping.py`, `tests/security/test_source_access.py`.

## 12. Streamlit Contract

Streamlit is presentation-only, using FastAPI via HTTP. Never directly query SQLite, FAISS, or BM25S from UI. Do not cache user-specific answer or source payloads globally or across active-role changes. Clear role-dependent session state after switching. Render only server-validated response content; fetch cited sources through authenticated API endpoints.

Enforcement: `tests/integration/test_ui_api_boundary.py`, `tests/integration/test_ui_cache_isolation.py`, `tests/security/test_source_access.py`.

## 13. Docker, Ports, and Persistence

Locked host ports: API `8000`, UI `8501`, Ollama `11434`.

Named Docker volumes are mandatory for SQLite, FAISS indexes, BM25S indexes, and Ollama model files. Ordinary `docker compose down` preserves named volumes; `docker compose down -v` removes project-managed named volumes, so do not use `-v` for ordinary shutdown.

Use one API replica: do not `docker compose --scale` the API. SQLite's single-writer architecture is sufficient for this MVP; do not introduce PostgreSQL for hypothetical concurrency.

Distinguish Ollama daemon readiness (`/api/tags`), model availability (`ollama show qwen3:4b-instruct`), and inference readiness (bounded smoke inference). A healthy daemon does not prove the model is loaded or able to answer.

Enforcement: `tests/integration/test_compose_contract.py`, `tests/integration/test_ollama_readiness.py`.

## 14. Evaluation Contract

**Full scope: 80 query-role executions.**
- 48 answerable.
- 16 unanswerable.
- 16 role-sensitive executions (= eight authorized/unauthorized pairs).
- 20 development cases, 60 frozen held-out cases.

**Reduced scope:** If the project owner explicitly approves the optional fallback under `docs/decisions.md` §17, use **60 cases**: 36 answerable + 12 unanswerable + 12 role-sensitive executions (= six pairs), divided into **15 development + 45 frozen held-out** cases. Keep all security tests, citation-validation requirements, retrieval ablations, and security acceptance thresholds unchanged.

Keep related question-role pairs in the same split. Select configurations using **development cases only**, then freeze and report on held-out cases without re-tuning to test outcomes. Manually verify labels and source passages.

Required metrics: Recall@5, MRR@10, nDCG@10, semantic citation support, refusal accuracy, unauthorized retrieval rate, p50/p95 latency; additionally log mechanical citation failures, language/category breakdowns, hardware and per-stage timings. Distinguish unverified targets from measured results; report negative findings. Full four-way ablations and semantic review use the same fixed corpus configuration.

Enforcement: `tests/eval/test_dataset_split.py`, `tests/eval/test_ablation.py`, `tests/eval/test_reranker_decision.py`, `tests/eval/test_e2e.py`.

## 15. Required Make Targets

The Makefile must provide:
- `make setup`
- `make ingest`
- `make test`
- `make eval-retrieval`
- `make eval-security`
- `make eval-generation`
- `make run`

**`make setup`** — Idempotently ensure locked Python 3.11 dependencies are installed using uv and the lockfile; copy `.env.example` to `.env` only when absent; validate Docker Compose configuration; verify Ollama daemon availability; pull `qwen3:4b-instruct` only if absent; and verify the installed model against the approved pinned digest. Never overwrite an existing `.env` or silently replace a pinned model. On missing prerequisites, fail with an actionable error rather than pretending setup succeeded.

**`make ingest`** — Validate the canonical Markdown corpus and produce a consistent, publishable SQLite/index snapshot.

**`make test`** — Run unit and applicable integration/security-critical deterministic tests. Fail on any required failing test.

**`make eval-retrieval`** — Evaluate BM25, dense, hybrid RRF, and hybrid-plus-rerank on the configured frozen benchmark; write `eval/report.json`.

**`make eval-security`** — Execute 12–16 parameterized adversarial scenario families; report unauthorized evidence/citation violations.

**`make eval-generation`** — Run local Ollama end-to-end evaluation with schemas, citations, refusals, and latency metrics.

**`make run`** — Start the locally configured application using the approved runtime workflow.

Full LLM evaluation may run locally at release rather than on each hosted CI job. CI must still check deterministic tests and result-artifact contracts.

Enforcement: `tests/integration/test_make_targets.py`, `tests/integration/test_compose_contract.py`, `tests/integration/test_ollama_readiness.py`, `tests/eval/test_ablation.py`, `tests/eval/test_e2e.py`.

## 16. Codex Workflow

Before coding: read `AGENTS.md`, `docs/decisions.md`, and `docs/codex-checklist.md`; identify one requirement, its acceptance criteria, and the exact permitted implementation/test files.

One atomic task per invocation. Modify only the specified file/module and associated tests. No new dependencies, unrelated refactors, locked-stack changes, cross-boundary edits, altered gold labels, or weakened assertions. If blocked, report the reproducible failure and stop for human approval before changing architecture/scope.

After each task report:
1. Files changed.
2. Feature implemented.
3. Tests added/updated.
4. Commands actually run and observed outcomes.
5. Tests not run and why.
6. Remaining blockers/limitations.

Never claim success without test execution evidence. Review work using `docs/codex-checklist.md`. Enforcement: `tests/unit/test_stack_contract.py`, `tests/unit/test_architecture_boundaries.py`, applicable acceptance tests.

## 17. Repository and Secrets

Required root artifacts: `AGENTS.md`, `docs/decisions.md`, `docs/codex-checklist.md`, `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`, `.env.example`, and `compose.yaml`.

Commit `.env.example` but **never `.env`**. No real company data, JWT signing secrets, passwords, live tokens, runtime SQLite databases, downloaded model weights, or generated confidential indexes in Git.

Enforcement: `tests/integration/test_repository_contract.py`, `tests/security/test_logging_hygiene.py`.

## 18. Mandatory Phase Order

Do not reorder:
1. **Contract:** instructions, decisions, checklist, packaging, lockfile, structure, test skeleton.
2. **Corpus:** two or three representative policies, ACLs, first manually verified development questions.
3. **Ingestion:** Markdown → validation → SQLite → reproducible snapshot hash.
4. **Security:** JWT, RBAC, role-scoped index-selection contract, first adversarial tests.
5. **Retrieval:** BM25 → normalized FAISS → RRF → conditional reranking and ablations.
6. **Generation:** Ollama → Pydantic → validated citations → controlled refusals.
7. **Full evaluation:** query cases, ablations, security attacks, metrics, error analysis.
8. **Delivery:** Docker Compose, Streamlit, CI, README, report, demo video.

Current-phase acceptance gates must pass before advancing. Future-phase tests may be marked incomplete but not reported as passing; no skipped or xfailed release-critical tests. Treat calendar estimates as planning guidance. Apply the measured retrieval-readiness milestone and owner-approved scope-change procedure in `docs/decisions.md` §17.

Enforcement: phase-specific tests in `docs/decisions.md` §11 and human review using `docs/codex-checklist.md`.

## 19. Out of Scope

No fine-tuning, autonomous agents, Kubernetes, cloud hosting, historical as-of-date retrieval, OCR, production enterprise IAM, unapproved infrastructure, or automatic execution of policy-related actions. Changes require explicit approval under `docs/decisions.md` §18.

Enforcement: `tests/unit/test_stack_contract.py`, `tests/unit/test_architecture_boundaries.py`, human scope review in `docs/codex-checklist.md`.

## 20. Final Instruction

Prioritize: **security correctness → evidence integrity → testability → reproducibility → retrieval quality → generation quality → performance → UI polish**.

When uncertain, preserve the contract and ask. A feature is complete only after its measurable acceptance tests pass. A successful demo is not a substitute for test evidence.
