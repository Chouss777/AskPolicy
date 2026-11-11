# AskPolicy — Architecture Decisions & Acceptance Contract

**Document:** `docs/decisions.md`  
**Project:** AskPolicy — Secure Multilingual Enterprise RAG  
**Status:** LOCKED  
**Owner:** Solo project developer  
**Timebox:** Four weeks  
**Goal:** A reproducible, security-aware Applied AI / LLM Engineering portfolio project for PFE applications.

---

## 1. Product Definition

AskPolicy is a local-first assistant for internal policy questions. It retrieves authorized evidence, enforces RBAC before retrieval, generates structured responses with verified citations, refuses unsupported requests, and evaluates reliability and security reproducibly.

**Atlas Services** is a fictional Moroccan company with 200 employees. All policy documents, users, access grants, and incidents are **synthetic**. No real company data is permitted.

Supported languages: French and English for every policy family; Arabic for Remote Work, Leave, Expenses, and Data Handling.

The MVP must demonstrate:
1. Structured, versioned policy ingestion.
2. Hybrid BM25/dense retrieval and optionally justified reranking.
3. Deterministic server-side authorization before search.
4. Local LLM generation restricted to authorized evidence.
5. Mechanical citation checking and separate semantic-support evaluation.
6. Safe refusals and no unauthorized document disclosure in executed security tests.
7. Reproducible quality, security, and latency evaluation.
8. A locally reproducible API and demonstration UI.

AskPolicy is a **security-aware prototype**, not an enterprise security certification. Never claim universal security, legal compliance, or guaranteed semantic correctness.

---

## 2. Frozen Technology Decisions

| Component | Locked choice | Reason |
|---|---|---|
| Language | Python 3.11 | Stable, reproducible compatible environment. |
| API | FastAPI + Uvicorn | Typed HTTP interface without needless infrastructure. |
| Schemas | Pydantic v2 | Strict metadata, API, and generated-output validation. |
| Authentication | PyJWT | Signed JWT verification without production IAM integration. |
| Password hashing | pwdlib[argon2] | Modern local demo identity hashing. |
| Vector index | FAISS-cpu `IndexFlatIP` | Exact nearest-neighbor ranking suits a small corpus. |
| Similarity | L2-normalized vectors and inner product | Cosine similarity after normalization. |
| Lexical retrieval | BM25S | Fast, local, serverless BM25 index. |
| Embeddings | `intfloat/multilingual-e5-small` | Lightweight multilingual representations. |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual reranking, conditional on measured gain. |
| Generation | Ollama + `qwen3:4b-instruct` | Local, structured, no paid inference API. |
| Runtime metadata | SQLite | Validated corpus and ACL source of truth. |
| UI | Streamlit | Fast presentation layer over HTTP API. |
| Containerization | Docker Compose | Local orchestration with few services. |
| Persistence | Named Docker volumes | Persistence of DB, indexes, models. |
| Testing | pytest | Deterministic tests and parameterized adversarial cases. |
| Dependency management | uv + `uv.lock` | Repeatable installation. |
| CI | GitHub Actions | Automated verification without hosted LLM inference. |

### FAISS semantics

Use `IndexFlatIP` with normalized query and passage embeddings. This index returns **inner-product similarity**, not L2 or squared-L2 distance. For unit vectors, inner product equals cosine similarity. Never apply `sqrt()` and never turn raw similarity into a calibrated “confidence %”.

**Enforcing test:** `tests/unit/test_faiss_normalization.py`.

### Why FAISS instead of pgvector

The corpus is small; exact search is inexpensive. SQLite and local role-scoped indexes are easier to audit and run on one laptop. Postgres/pgvector is deliberately excluded from the MVP. Indexes remain rebuildable from SQLite.

### E5 embedding contract

Use the exact pinned `intfloat/multilingual-e5-small` revision at ingestion and query time. Prefix inputs with `query: ` and `passage: ` respectively. Store model revision in index metadata. **Tests:** `tests/unit/test_e5_prefix.py`, `tests/integration/test_embedding_revision.py`.

### Reranker inclusion

The reranker candidate is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Benchmark on **development cases**, comparing Recall@5, nDCG@10, and p50/p95 latency with/without reranking on declared hardware. Record measured differences and the human-approved quality/latency decision before enabling it in the deployed path. No arbitrary gain threshold or unmeasured latency assumption is accepted. Always retain the fourth ablation configuration for reporting even if deployment omits reranking.

**Enforcing test:** `tests/eval/test_reranker_decision.py`.

### Exact model versions

Capture pinned embedding and reranker model revisions and the installed Ollama model digest in the reproducibility manifest. Fail validation on unapproved model changes. Do not fabricate a digest; the first approval records the actual installed digest before a run can be declared reproducible.

---

## 3. Corpus Specification

**Full scope: 26 Markdown documents**: 20 active + 6 superseded.

### Active documents

| Policy family | FR | EN | AR |
|---|---|---|---|
| Remote Work | V2 | V2 | V2 |
| Leave | V2 | V2 | V2 |
| Expenses | V2 | V2 | V2 |
| Travel | V1 | V1 | — |
| Access Control | V2 | V2 | — |
| Data Handling | V2 | V2 | V2 |
| Incident Reporting | V1 | V1 | — |
| Procurement | V1 | V1 | — |
| **Active totals** | **8** | **8** | **4** |

Only Remote Work, Leave, Expenses, and Data Handling have Arabic active versions.

### Superseded documents

| Family | Superseded versions | Count |
|---|---|---:|
| Remote Work | FR V1, EN V1 | 2 |
| Expenses | FR V1 | 1 |
| Access Control | FR V1 | 1 |
| Leave | FR V1 | 1 |
| Data Handling | FR V1 | 1 |
| **Total** | | **6** |

Exactly five families have at least one superseded document. **20 + 6 = 26**.

### Canonical document contract

Use Markdown with validated YAML front matter. Every policy document must contain the keys below, as required by `AGENTS.md` §5.

| Field | Definition |
|---|---|
| `policy_id` | Language-independent identity of policy family. |
| `document_id` | Unique language/version document identity. |
| `version` | Version number. |
| `language` | `fr`, `en`, or `ar`. |
| `title` | Policy title. |
| `department` | Policy owner department. |
| `classification` | One of `PUBLIC`, `INTERNAL`, `RESTRICTED`. |
| `allowed_roles` | Explicit allowed read roles. Must be non-null. |
| `effective_date` | Effective date. Must be non-null. |
| `expires_at` | Expiration date or `null` when no expiration applies. |
| `supersedes` | Previous `document_id` or `null` when there is no superseded document. |
| `approval_status` | Approved or not-approved state. |
| `source_type` | Synthetic policy provenance. |
| `source_license` | Redistribution license/provenance terms. |

Omitting either `expires_at` or `supersedes` is invalid. When neither applies, both keys have explicit YAML `null` values.

```yaml
expires_at: null
supersedes: null
```

Missing or malformed security fields cause ingestion failure. Missing `allowed_roles` never means “all roles”. Unknown classifications are rejected.

### Corpus manifest

The checked-in `data/corpus_manifest.yaml` enumerates every document, source path, identity, language, version, approval, effective/expiration dates, supersession relationship, classification, allowed roles, provenance/license, and content SHA-256. It must agree with the actual files; changed files require explicit revalidation and snapshot publication.

**Tests:** `tests/unit/test_metadata_validation.py`, `tests/integration/test_corpus_manifest.py`.

### Chunking and tables

Preserve headings, hierarchy, procedures, exceptions, tables with column headings, parent-child section relationships, and original source offsets. Children inherit authorization/version metadata. Parent expansion may not cross document/version/ACL boundaries. Chunk IDs are deterministic for equivalent inputs. Do not add OCR.

**Test:** `tests/integration/test_chunk_boundaries.py`.

---

## 4. Versioning and Validity

A document is searchable at the snapshot's validity time **if and only if** it is:

`approved AND effective AND unexpired AND not superseded`.

Current-policy search always excludes superseded versions. Historical as-of-date retrieval is **out of scope**. Version identifiers must remain visible in evidence metadata. If a scheduled expiration/effective-date transition invalidates an active snapshot, publish a new valid snapshot or fail closed for affected retrieval; never silently serve stale permissions/policies.

**Tests:** `tests/integration/test_version_selection.py`, `tests/integration/test_snapshot_consistency.py`.

---

## 5. Storage and Snapshot Architecture

**Canonical flow:**

`Version-controlled Markdown → metadata validation → section/chunk extraction → validated SQLite snapshot → authorized role-scoped BM25S/FAISS builds → atomic publication`

Markdown is the source document of record. SQLite is the validated runtime source of truth. FAISS and BM25S indexes are **derived and rebuildable**.

A published snapshot includes a stable identifier, canonical content/configuration hashes, source manifest hash, published validity, model/index configuration and chunk identities. Hashing excludes build timestamps, temporary paths, and nondeterministic file ordering.

All indexes used together must declare the same snapshot ID as SQLite; compare embedding revision and relevant metadata as well. If any required active index is missing/mismatched, fail retrieval with **HTTP 503**. Never mix snapshots in a single request. New builds must succeed completely before their pointer is atomically published. A failed build leaves the previous valid snapshot untouched. Unpublished Markdown edits must not silently change the published query corpus.

**Tests:** `tests/integration/test_ingestion_deterministic.py`, `tests/integration/test_snapshot_consistency.py`, `tests/integration/test_snapshot_publication.py`, `tests/integration/test_index_rebuild.py`.

---

## 6. Authentication and Authorization

Read roles: `employee`, `hr`, `finance`, `it_security`. Separate capability: `ingestion_admin`, which can publish validated snapshots but has **no automatic read access** through retrieval or source endpoints.

One active read role per request/session. The backend checks that the role is in the authenticated identity's verified grants. A user with multiple grants may switch context without reauthentication while the authentication session remains valid, but the new context must be reauthorized server-side and **all role-dependent answer/evidence caches invalidated**.

Verify JWT signature, expiry, issuer, audience and grants. Missing, expired or forged JWT → HTTP 401. Requested ungranted role → HTTP 403. Never trust natural-language role claims or frontend state independently of verified grants. Unknown/missing classification/grants are denied by default.

Authorization checkpoints:
1. Before choosing a role-scoped index.
2. Before reranking.
3. Before evidence reaches generation.
4. Before returning a source passage.
5. Before presenting citations.

Never search a global unrestricted index and post-filter results. Parent expansion and direct source access require independent permission checks.

**Tests:** `tests/security/test_auth.py`, `tests/security/test_role_grant.py`, `tests/security/test_role_switch.py`, `tests/security/test_role_scoping.py`, `tests/security/test_default_deny.py`, `tests/security/test_source_access.py`, `tests/security/test_parent_expansion.py`, `tests/security/test_admin_isolation.py`, `tests/integration/test_ui_cache_isolation.py`.

---

## 7. Retrieval Architecture

**BM25S:** authorized lexical retrieval, initial top 30.  
**FAISS:** normalized E5 + `IndexFlatIP`, authorized semantic retrieval, initial top 30.  
**Fusion:** reciprocal rank fusion (`k=60`), deduplicated and deterministic top 20.  
**Reranker:** optional multilingual Cross-Encoder on up to 20 candidates.  
**Generation evidence:** up to five top-ranked, authorized passages after additional ACL validation and context constraints.

Initial budgets are development configuration. Deviations must be measured on the development set, documented, and approved; do not tune against the frozen held-out set.

Required four-way ablation: BM25 only, dense only, hybrid RRF, hybrid RRF + reranker. Every configuration uses identical snapshot, labels, role conditions, and query set. Reranking remains available for experimental reporting even if not deployed.

**Tests:** `tests/unit/test_rrf.py`, `tests/security/test_role_scoping.py`, `tests/security/test_parent_expansion.py`, `tests/eval/test_ablation.py`, `tests/eval/test_reranker_decision.py`.

---

## 8. Generation and Citation Contract

Local generation uses Ollama with `qwen3:4b-instruct`; record and verify its approved pinned digest. Validate structured output via strict Pydantic v2 schemas before display.

A returned citation is mechanically valid only when the chunk ID exists in the **authorized retrieved evidence supplied to the model for that request**, belongs to the active snapshot and remains authorized, and the quoted string is an **exact substring of original stored chunk text**. Invalid IDs, out-of-set evidence, invented quotes, and wrong versions are rejected.

**Fail closed:** invalid structure or material citation → controlled refusal and **no partial answer**. Citation membership and exact quotations do not prove semantic support: measure claim-level support separately against manually reviewed gold facts.

Refuse for insufficient authorized evidence, unresolved conflicts, version uncertainty, or failed validation. Unauthorized/unavailable cases must have indistinguishable external refusal wording where source existence would leak. Never expose restricted titles/IDs through refusals.

**Tests:** `tests/unit/test_citation_validator.py`, `tests/integration/test_fail_closed.py`, `tests/unit/test_generation_schema.py`, `tests/security/test_refusal_indistinguishability.py`, `tests/eval/test_e2e.py`.

---

## 9. Security Threat Model

| Threat | Enforced safeguard |
|---|---|
| Missing/forged/expired JWT | Verify signed claims, reject 401. |
| Role forged in request/prompt | Server-side grant verification, reject 403. |
| Unauthorized passages in search | Select authorized role-scoped index before searching. |
| Direct restricted source lookup | Independent source ACL check. |
| Parent-child ACL leakage | Check expansion boundaries. |
| Indirect prompt injection in policies | Treat source text as untrusted data, keep permissions outside LLM. |
| Fabricated/unsupported citations | Membership, quote integrity; semantic review separately. |
| Stale/superseded versions | Validated snapshot and time-bound rules. |
| Sensitive data in logs | Structured operational metadata only. |
| Role-dependent cache leakage | Clear user/role-dependent data on role switch. |
| Partially updated indexes | Atomic snapshot publication. |

Execute **12–16 distinct adversarial scenario families**, parameterized across roles, classifications, access methods, and version conditions. Cover missing/expired/forged tokens, ungranted roles, prompt impersonation, HR retrieval, source IDs, unknown classification, parent expansion, fabricated citations, quote mismatch, out-of-set citations, indirect prompt injection, expired/superseded content, cache role switching, and ingestion-admin read isolation.

Security acceptance on executed tests: **zero unauthorized chunks, zero unauthorized citations, and zero mechanically invalid displayed citations**. This is a bounded test result, not a proof of absence of all vulnerabilities.

**Tests:** `tests/security/test_auth.py`, `tests/security/test_role_grant.py`, `tests/security/test_role_scoping.py`, `tests/security/test_default_deny.py`, `tests/security/test_source_access.py`, `tests/security/test_parent_expansion.py`, `tests/security/test_role_switch.py`, `tests/security/test_admin_isolation.py`, `tests/security/test_prompt_injection.py`, `tests/security/test_refusal_indistinguishability.py`, `tests/security/test_logging_hygiene.py`, `tests/integration/test_fail_closed.py`.

---

## 10. Evaluation Specification

### Full scope — 80 query-role cases

| Category | Cases |
|---|---:|
| Answerable | 48 |
| Unanswerable | 16 |
| Role-sensitive executions | 16 (8 authorized/unauthorized pairs) |
| **Total** | **80** |

**Split:** 20 development + 60 frozen held-out cases. Keep paired questions within one split.

### Reduced scope — if the Week 1 contingency activates

If `docs/decisions.md` §17 activates the reduction, evaluation becomes **60 query-role cases**:
- 36 answerable.
- 12 unanswerable.
- 12 role-sensitive executions (= six authorized/unauthorized pairs).
- **15 development + 45 frozen held-out** cases.

Paired question families must remain within the same split. All security tests, citation validation, retrieval ablations, and security acceptance criteria remain mandatory. The active-scope case counts must be reflected consistently in dataset validation and end-to-end reports.

### Gold record requirements

Each evaluation case contains: query ID, question-family ID, language, active role, query, expected answer/refusal status, authorized gold source sections where applicable, relevance grades, required factual content where applicable, case type, and annotation review status. Manually verify source relevance and factual labels. LLM-drafted candidates cannot self-certify correctness.

Freeze the held-out split before final benchmarking. Tune/query-expand/choose the reranker only on development cases. Never change gold labels to favor a configuration or repeatedly optimize based on held-out results.

### Required metrics

| Metric | Operational definition |
|---|---|
| Recall@5 | Macro-average fraction of authorized gold-relevant sections represented in top five, across eligible queries. |
| MRR@10 | Mean reciprocal rank of first relevant source within ten results. |
| nDCG@10 | Graded relevance ranking quality normalized to ideal ranking. |
| Citation support | Proportion of factual claims truly supported by cited authorized evidence, verified separately from mechanical integrity. |
| Refusal accuracy | Correct answer/refusal decisions divided by evaluated cases. |
| Unauthorized retrieval rate | Unauthorized returned passages divided by returned passages; also report violating query count. |
| p50/p95 latency | Median and 95th percentile measured request latency under declared conditions. |

Additionally report invalid citation counts, unauthorized citations, answer correctness against required facts, performance by FR/EN/AR and question type, hardware, cold/warm state, and stage timings. Comparisons must use identical queries/snapshots and paired differences. Bootstrap uncertainty may be reported with deterministic resampling and grouping by question family; record seed and number of resamples.

### Evaluation outputs

`make eval-retrieval` generates `eval/report.json` identifying dataset hash, snapshot, retrieval configuration, model revisions, metrics, case counts, and measured timings. `make eval-security` and `make eval-generation` produce machine-readable security and generation results with actual measurements. Full model inference may execute locally outside routine hosted CI; CI validates applicable deterministic logic and report artifact contracts.

**Tests:** `tests/eval/test_dataset_split.py`, `tests/eval/test_ablation.py`, `tests/eval/test_reranker_decision.py`, `tests/eval/test_e2e.py`.

---

## 11. Acceptance Criteria per Feature

All test paths are relative to repository root. Every feature is complete only after its enforcing test **passes normally**, not merely via a demo, skip, or expected failure.

| Feature | Verifiable acceptance criteria | Enforcing test |
|---|---|---|
| Ingestion determinism | Same canonical input/config produces same snapshot hash across two clean builds. | `tests/integration/test_ingestion_deterministic.py` |
| Metadata validation | Malformed YAML, missing `allowed_roles`, missing `effective_date` cause named errors and abort ingestion. | `tests/unit/test_metadata_validation.py` |
| Snapshot integrity | SQLite/index snapshot mismatch → HTTP 503; no stale-data fallback. | `tests/integration/test_snapshot_consistency.py` |
| Authentication | Missing, expired, forged JWT → 401. | `tests/security/test_auth.py` |
| Role grants | JWT grants only `employee`, active-role request `hr` → 403. | `tests/security/test_role_grant.py` |
| Role-scoped retrieval | Employee sees zero unauthorized HR chunks among top 20. | `tests/security/test_role_scoping.py` |
| Default-deny ACL | Unknown classification rejected and not retrievable. | `tests/security/test_default_deny.py` |
| Refusal indistinguishability | Unauthorized and unavailable cases use identical visible refusal text, no restricted identity disclosure. | `tests/security/test_refusal_indistinguishability.py` |
| E5 prefixes | Captured query inputs start `query: ` and documents start `passage: `. | `tests/unit/test_e5_prefix.py` |
| FAISS normalization | All indexed vector norms ∈ [0.999, 1.001]; correct IP semantics; no `sqrt()` or confidence percentage claim. | `tests/unit/test_faiss_normalization.py` |
| Citation valid | Allowed supplied chunk and exact quotation pass mechanical validation. | `tests/unit/test_citation_validator.py` |
| Citation invalid | Invented ID, quote mismatch, and out-of-set citation rejected. | `tests/unit/test_citation_validator.py` |
| Fail-closed generation | Invalid citation produces refusal, never partial answer. | `tests/integration/test_fail_closed.py` |
| Reranker decision | Dev Recall@5 delta and measured p95 recorded; activation matches approved decision. | `tests/eval/test_reranker_decision.py` |
| Retrieval ablation | Four configs run on frozen benchmark with machine-readable `eval/report.json`. | `tests/eval/test_ablation.py` |
| End-to-end | Full: 60 held-out cases; reduced: 45; security leakage 0, invalid displayed citations 0, p95 recorded. | `tests/eval/test_e2e.py` |
| Streamlit cache isolation | No user-specific answer/source content persists across role switch. | `tests/integration/test_ui_cache_isolation.py` |
| Corpus manifest | Required docs, version links, ACLs, hashes and licenses agree with files. | `tests/integration/test_corpus_manifest.py` |
| Parent-child structure | Expansion respects document, version, source-offset and ACL boundaries. | `tests/integration/test_chunk_boundaries.py` |
| Policy validity | Only approved, effective, unexpired, non-superseded documents searchable. | `tests/integration/test_version_selection.py` |
| Atomic publication | Failed build retains prior good snapshot; no mixed readers. | `tests/integration/test_snapshot_publication.py` |
| Rebuildability | SQLite snapshot reconstructs authorized indexes with same identities. | `tests/integration/test_index_rebuild.py` |
| Embedding consistency | Index/query revisions match persisted model metadata. | `tests/integration/test_embedding_revision.py` |
| RRF correctness | Known rankings yield expected deterministic deduplicated output. | `tests/unit/test_rrf.py` |
| Role switching | Only granted roles usable, all role-dependent caches cleared. | `tests/security/test_role_switch.py` |
| Ingestion admin isolation | Admin-only capability cannot read restricted content by default. | `tests/security/test_admin_isolation.py` |
| Source access | Unauthorized direct source requests rejected. | `tests/security/test_source_access.py` |
| Parent authorization | Expansion never exposes a restricted parent via allowed child. | `tests/security/test_parent_expansion.py` |
| Prompt injection | Covered malicious prompts/documents cannot override deterministic ACLs. | `tests/security/test_prompt_injection.py` |
| Logging hygiene | No raw JWT, secrets, complete prompts, or restricted passages in logs. | `tests/security/test_logging_hygiene.py` |
| Generation schema | Invalid or unexpected LLM JSON rejected before display. | `tests/unit/test_generation_schema.py` |
| Dataset split | Correct active-scope totals; role pairs never split across dev/test. | `tests/eval/test_dataset_split.py` |
| UI/API boundary | Streamlit calls HTTP API, never retrieval DB or indexes. | `tests/integration/test_ui_api_boundary.py` |
| Docker topology | Required services, ports and named volumes configured. | `tests/integration/test_compose_contract.py` |
| Ollama readiness | Distinguish daemon, installed model, and bounded inference readiness. | `tests/integration/test_ollama_readiness.py` |
| Makefile targets | `setup`, `ingest`, `test`, `eval-retrieval`, `eval-security`, `eval-generation`, `run` exist and perform documented actions. | `tests/integration/test_make_targets.py` |
| Repository contract | Required files present; secrets and runtime artifacts excluded. | `tests/integration/test_repository_contract.py` |
| Locked stack | Declared direct dependencies comply with approved stack and approval policy. | `tests/unit/test_stack_contract.py` |
| Module boundaries | Prohibited cross-layer imports and direct UI/storage coupling rejected. | `tests/unit/test_architecture_boundaries.py` |

Do not weaken security checks to obtain a green build. No skipped or xfailed mandatory security test at release. Semantic support relies on a manually audited benchmark as well as automated result-schema checks.

---

## 12. Security Test Scope

Implement **12–16 distinct adversarial scenario families** (parameterize relevant roles, documents, and access methods):
1. Missing JWT.
2. Expired JWT.
3. Forged signature.
4. Ungranted role activation.
5. Prompt-based identity/role impersonation.
6. Employee asks for HR-restricted passages.
7. Direct restricted source access by ID.
8. Unknown classification.
9. Parent expansion crosses ACL.
10. Fabricated citation ID.
11. Incorrect quotation.
12. Citation outside supplied retrieval evidence.
13. Malicious instructions embedded in a policy.
14. Superseded/expired policy retrieval.
15. Cache content persists across role switch.
16. Ingestion admin attempts content access.

Acceptance: no unauthorized chunks returned, no unauthorized citations exposed, no mechanically invalid citations displayed in the executed suite. Keep documented limitations in `docs/threat-model.md`. See tests enumerated in §9 and §11.

---

## 13. Observability and Reproducibility

Operational logs are subject to `AGENTS.md` §11. Limit them to bounded, non-sensitive values for `request_id`, `active_role`, `corpus_snapshot_id`, `response_status`, `validation_status`, and pipeline latency measurements.

Never log raw JWTs, passwords, signing secrets, complete prompts, restricted chunk text, or confidential answer payloads, including sensitive generated answers.

For each evaluated request, record the bounded request metadata, `query_language`, `retrieval_config`, result count, and per-stage timings in evaluation artifacts under `AGENTS.md` §14. Evaluation artifacts may also contain `query_id`, dataset and configuration hashes, aggregate metrics, per-case metric values, model revisions, and hardware information. Use only synthetic policies, identities, and queries in these artifacts.

Evaluation manifests must record Git commit, Python version, lockfile state, corpus/dataset hashes, embedding/reranker revisions, Ollama digest, generation parameters, relevant random seeds, hardware/VRAM/RAM, evaluation time and commands. Label deterministic tests, retrieval metrics, semantic reviews, attack results, and measured latency distinctly.

**Tests:** `tests/security/test_logging_hygiene.py`, `tests/eval/test_ablation.py`, `tests/eval/test_e2e.py`.

---

## 14. Docker and Deployment Decisions

Three primary services: FastAPI API, Streamlit UI, Ollama inference.

| Service | Host port |
|---|---:|
| FastAPI | 8000 |
| Streamlit | 8501 |
| Ollama | 11434 |

Named volumes persist SQLite data, FAISS indexes, BM25S indexes, and Ollama model files. `docker compose down` preserves volumes; `docker compose down -v` removes project-managed named volumes. Never use `-v` for ordinary shutdown.

Use one API replica. Do not add `docker compose --scale` or replace SQLite without approved change. Check Ollama daemon (`/api/tags`), installed model (`ollama show qwen3:4b-instruct`), and bounded inference separately.

**Tests:** `tests/integration/test_compose_contract.py`, `tests/integration/test_ollama_readiness.py`.

---

## 15. Mandatory Development Phase Order

Do not reorder these eight phases:

**Phase 1 — Contract:** `AGENTS.md`, `docs/decisions.md`, `docs/codex-checklist.md`, `pyproject.toml`, `uv.lock`, package structure, test skeleton. Exit: version/stack files valid, test discovery works, expected future test states documented.

**Phase 2 — Corpus:** two or three representative synthetic Markdown policies (one generally accessible and one HR restricted), full YAML metadata, first manually checked dev cases. Exit: policy metadata and source links validate.

**Phase 3 — Ingestion:** Markdown → metadata checks → chunking → SQLite → deterministic snapshot hash. Exit: malformed metadata rejected and clean-rebuild snapshot hash stable.

**Phase 4 — Security:** signed JWT, verified grants, active read role, deny-by-default and role-index selection interfaces, first attack checks. Exit: auth/grant/default-deny tests pass; retrieval-dependent integration security tests become mandatory once retrieval exists.

**Phase 5 — Retrieval:** BM25 → normalized FAISS → RRF → conditional reranker and ablations. Exit: role isolation passes, 4-way ablation recorded, inclusion decision justified on dev metrics.

**Phase 6 — Generation:** Ollama, Pydantic output, citations, refusal. Exit: valid citations pass, invalid citations refuse, structure and ACL tests pass.

**Phase 7 — Full Evaluation:** active-scope frozen cases, ablations, adversarial suite, multilingual breakdown, error analysis and latency. Exit: measured reproducible report and mandatory security gates pass.

**Phase 8 — Delivery:** Docker, Streamlit, CI, README, architecture, threat model, evaluation report, three-minute demo. Exit: clean installation, tested API/UI, no secrets, actual results.

---

## 16. Four-Week Delivery Gates

| Week | Work | Evidence to pass |
|---|---|---|
| 1 | Contract, sample corpus, ingestion, offline development BM25 smoke test | Validated corpus, stable snapshot hash, baseline metric. |
| 2 | JWT/RBAC, authorized index selection, BM25/FAISS integration | Auth/grant/default-deny tests green; role-scoped retrieval underway. |
| 3 | RRF/reranker study, Ollama, citation/refusal flow | Four retrieval configurations, passing ACL and citation tests. |
| 4 | Full evaluation, deployment, CI, report, demo | Held-out and security results, clean setup, publishable materials. |

The Week 1 BM25 baseline is an isolated, offline evaluation smoke test that uses development cases and only validated, publicly accessible corpus evidence. Run it without exposing a retrieval API or granting document access.

The eight phases proceed in the order specified in §15. Production retrieval integration in Phase 5 begins only after Phase 4 authentication and RBAC are fully implemented and the Phase 4 exit criteria pass.

The measured Week 1 gate in §17 is mandatory and takes precedence over the full-scope target when triggered.

---

## 17. Week 1 Contingency — Measured Gate

At the **end of Week 1**, inspect three objective conditions:
1. **At least 20 documents have been ingested and validated.**
2. **The snapshot hash is reproducible across clean runs.**
3. **A development-set BM25 baseline exists with Recall@5 strictly greater than 0.**

If **any** condition fails, immediately activate the reduced scope. There is no subjective override. Document the active scope in the evaluation/configuration manifests.

### Full scope if all gates pass

Eight families; four Arabic families; 26 documents; 80 eval cases; 20 dev + 60 held-out.

### Reduced scope if any gate fails

Drop **Procurement** and **Incident Reporting**. Keep Remote Work, Leave, Expenses, Travel, Access Control, Data Handling (six families).

Arabic only on **Remote Work** and **Expenses**.

Documents: **12 active FR/EN + 2 active AR + 6 superseded = 20 total**.

Evaluation: **36 answerable + 12 unanswerable + 12 role-sensitive executions (6 pairs) = 60**, split into **15 development + 45 frozen held-out**. Keep paired question families in the same split.

### Never cut the following

JWT verification; deterministic RBAC; role-scoped retrieval; deny-by-default ACL; citation integrity; controlled refusal; complete security adversarial scenarios; BM25; FAISS; RRF; conditional reranker evaluation; four-way retrieval ablation; end-to-end evaluation; reproduction evidence; Docker delivery.

The contingency reduces corpus/language/label breadth, not security or verification depth. Adapt dataset counts in acceptance checks to the active scope.

---

## 18. When the Stack May Change

A stack change is permitted **ONLY** when all three conditions hold:

1. **A specific, reproducible failure is documented** (test case, error log, or benchmark).
2. **The failure cannot be resolved within the current stack after at least one documented attempt.**
3. **The replacement is proposed with (a) a diff of the affected tests, (b) new acceptance criteria, and (c) a written rationale.**

**Otherwise no.**

“Codex found it awkward” is not a concrete problem.

“CI fails on test X with error Y, attempts in commits A and B” is a concrete problem.

Meeting all three conditions is necessary but **not sufficient**: the human owner must explicitly approve the change. Until approved, leave the locked stack intact. After approval, update this document, `AGENTS.md`, affected tests, the lockfile, and run relevant regressions.

The same governance applies to changes in locked architecture, scope, and phase order.

---

## 19. Codex Change Contract

Every task must:
1. Read `AGENTS.md`, this document, and `docs/codex-checklist.md`.
2. Specify one focused requirement, permitted implementation/test paths, and criteria.
3. Implement only requested behavior with its corresponding tests.
4. Execute the appropriate tests rather than inventing results.
5. Report changed files, added/modified tests, actual commands and outputs, unexecuted tests, blockers and limitations.

Codex must not introduce unapproved dependencies, replace technologies, refactor unrelated modules, weaken assertions, modify frozen labels, or claim feature completion based on demos. Human review against `docs/codex-checklist.md` is required before accepting a task.

---

## 20. Testing and CI Policy

### Environment bootstrap — `make setup`

`make setup` is the **idempotent** environment bootstrap and must be safe to rerun. It must:

1. Install the locked **Python 3.11** dependencies through **uv**, respecting `uv.lock`.
2. Copy `.env.example` to `.env` **only if `.env` is absent**; never overwrite existing secrets or configuration.
3. Validate the Docker Compose configuration and report errors clearly.
4. Check that the Ollama daemon is reachable before any model operations.
5. Pull `qwen3:4b-instruct` **only when not already available**.
6. Verify the installed model against the approved **pinned Ollama digest**; fail on unexpected mismatch rather than silently changing it.

If a prerequisite is unavailable (Python, uv, Docker Compose, Ollama, or the approved digest), fail with an actionable diagnostic. Repeat runs must not duplicate work, overwrite `.env`, or silently alter approved model revisions. Model pin approval and recorded-digest verification are governed by §2.

**Verification:** `tests/integration/test_make_targets.py`, `tests/integration/test_compose_contract.py`, `tests/integration/test_ollama_readiness.py`, and dependency checks in `tests/unit/test_stack_contract.py`.

### Other Make targets

- `make ingest`: validate, ingest, and publish a consistent corpus snapshot.
- `make test`: run unit and deterministic applicable integration/security-critical tests; nonzero exit if any mandatory test fails.
- `make eval-retrieval`: run all four retrieval configurations and write `eval/report.json`.
- `make eval-security`: run parameterized adversarial authorization/citation tests with machine-readable outcomes.
- `make eval-generation`: run local Ollama generation evaluation with citation, refusal, and latency outputs.
- `make run`: start the approved local runtime.

### CI layers

- **Fast on push/PR:** unit tests, mocked deterministic integration, JWT/ACL/citation checks, static contracts.
- **Integration/release candidate:** SQLite, FAISS/BM25S snapshot consistency, real authorized retrieval, security suite.
- **Full local release evaluation:** frozen benchmark, Ollama inference, manual semantic citation audit, p50/p95 hardware-specific timings.

Mandatory current-phase tests must pass normally. Temporary `xfail` requires a documented reason; remove it before declaring feature complete. No skipped or xfailed release-critical security tests. CI should check evaluation output schemas even when heavy live-model inference runs locally rather than on every hosted push.

---

## 21. Deliverables

**Code:** typed FastAPI backend, auth/RBAC, ingestion, local BM25S/FAISS/RRF, conditional reranker, Ollama generation, Pydantic schema validation, citations/refusal, Streamlit UI.

**Data/config:** synthetic Markdown corpus, validated `data/corpus_manifest.yaml`, development and frozen held-out query sets, retrieval config variants, pinned dependency/model manifests.

**Tests:** unit, integration, security, evaluation-contract suites, all with named acceptance criteria.

**Documentation:** `AGENTS.md`, `docs/decisions.md`, `docs/codex-checklist.md`, `docs/architecture.md`, `docs/threat-model.md`, `docs/evaluation-methodology.md`, `docs/evaluation-report.md`, `README.md`.

**Runtime:** Docker Compose, named volumes, health/readiness, `.env.example`, Makefile, CI workflow.

**Portfolio:** public GitHub repo, actual reproducible results, architecture diagram, security evidence, honest limitations, and three-minute demonstration video.

---

## 22. Explicitly Out of Scope

No fine-tuning; autonomous agents or agent frameworks; Kubernetes; cloud hosting; historical as-of-date retrieval; OCR; production enterprise IAM; paid hosted LLM inference; unapproved data infrastructure; automatic execution of policy-related business actions. Adding these requires documented human-approved changes under §18.

---

## 23. Release Acceptance

AskPolicy is ready for portfolio release only when:

1. The locked architecture is implemented or approved deviations are documented.
2. The corpus and evaluations match the **active full or reduced scope**.
3. Metadata, manifest, and snapshot integrity tests pass.
4. Role grants and document ACLs are enforced by the backend.
5. All mandatory security regressions pass normally.
6. Executed tests expose zero unauthorized chunks and citations.
7. Executed tests expose zero mechanically invalid displayed citations.
8. Four-way retrieval ablations are reproducible.
9. Generation evaluation uses the frozen held-out set.
10. Hardware-specific latency and failures are honestly documented.
11. Docker starts from a clean documented environment.
12. Exact model revisions/digests and dataset/config hashes are recorded.
13. README provides reproducible setup, evaluation, and test instructions.
14. No secrets, true company policies, or personal data enter the repo.
15. Demo uses the actual application and measured results.
16. Report clearly distinguishes targets, assumptions, and measurements.
17. No release-critical tests remain skipped or xfailed.

These conditions verify the declared project specification; they are **not** proof of production-level security or universal answer correctness.

---

## 24. Final Engineering Principle

AskPolicy's objective is not to maximize frameworks or features. It is to demonstrate that a developer can **engineer, secure, measure, and reproduce** a local enterprise-oriented LLM application.

**Security correctness → evidence integrity → testability → reproducibility → retrieval/generation quality → performance → presentation.**

No untested feature is complete. No stack change without documented failure and approval. No invented benchmark. No demo substituted for test evidence.
