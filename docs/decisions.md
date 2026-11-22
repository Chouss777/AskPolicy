# AskPolicy — Architecture Decisions & Acceptance Contract

**Document:** `docs/decisions.md`  
**Project:** AskPolicy — Secure Multilingual Enterprise RAG  
**Status:** LOCKED  
**Owner:** Solo project developer  
**Planning estimate:** Four weeks\
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
| E5 CPU runtime | `torch==2.15.0` from the explicit `https://download.pytorch.org/whl/cpu` index and `transformers==5.23.0` | Owner-approved local inference runtime under §18. |
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

Use revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` of `intfloat/multilingual-e5-small` at ingestion and query time. Prefix inputs with `query: ` and `passage: ` respectively. Count the prefix and special tokens toward the 512-token limit, and reject overlength inputs explicitly. Use attention-mask mean pooling and L2-normalized 384-dimensional float32 embeddings on CPU. Fail closed on missing or mismatched artifacts, nonfinite values, or zero norms. Store the model revision in index metadata. **Tests:** `tests/unit/test_e5_prefix.py`, `tests/unit/test_faiss_normalization.py`, `tests/integration/test_embedding_revision.py`.

Provision the exact model and tokenizer outside Git and verify their checksums. Both encoder paths load verified local files with `local_files_only=True` and `trust_remote_code=False`. Model weights must use `use_safetensors=True`. Dependency and model verification includes the stack contract test, an explicitly executed CPU smoke test against the pinned files, and the applicable role-scoping regressions.

### Reranker inclusion

The reranker candidate is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Benchmark on **development cases**, comparing Recall@5, nDCG@10, and p50/p95 latency with/without reranking on declared hardware. Record measured differences and the human-approved quality/latency decision before enabling it in the deployed path. No arbitrary gain threshold or unmeasured latency assumption is accepted. Always retain the fourth ablation configuration for reporting even if deployment omits reranking.

**Enforcing test:** `tests/eval/test_reranker_decision.py`.

### Step 116 owner decision — v1 retrieval baseline

On 2026-11-22, the project owner explicitly approved retaining **Hybrid BM25S + E5/FAISS RRF** as the AskPolicy v1 retrieval baseline, with **production Cross-Encoder reranking disabled**. Preserve BM25, FAISS, Hybrid RRF, and Hybrid RRF + Cross-Encoder for evaluation. No retrieval architecture, algorithm, model revision, ranking budget, or configuration change is approved by this decision.

FAISS-only selection and any embedding fine-tuning are deferred until **after the v1 portfolio release**. This is a deferral, not advance approval: any later architecture or scope change still requires separate approval under §18; fine-tuning remains out of scope under §22 and `AGENTS.md` §19 until explicitly approved.

The owner acknowledges that **FAISS outperformed Hybrid on all three macro quality metrics in the small approved development sample**. Retaining Hybrid is a deliberate decision for architecture continuity, preservation of tested RBAC boundaries, reproducibility, and completion of v1. It is not a claim that Hybrid won the benchmark or that FAISS is inherently less secure. No statistical superiority or broader multilingual performance conclusion follows from this sample.

#### Measured Step 112–114 evidence

Step 112 executed the four modes on the same published snapshot, approved labels, queries and role conditions: 20 development executions, 14 positive-gold cases, six no-positive cases excluded from quality means, 17 question families and 50 gold section references. Positive language groups were FR 6, EN 4 and AR 4. Step 114 independently verified the stored per-case/macro metrics and all 20 RRF reconstructions. This decision records those historical measurements; Step 116 does not rerun model inference or a retrieval benchmark.

| Mode | Recall@5 | MRR@10 | nDCG@10 | Warm p50 seconds | Warm p95 seconds |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5476190476190476 | 0.9523809523809523 | 0.6104075558218088 | 0.0034319000005780254 | 0.004305120000935859 |
| FAISS | 0.5892857142857142 | 1.0 | 0.7219831671154645 | 0.021969399998852168 | 0.026169804999699408 |
| Hybrid | 0.5357142857142856 | 0.9642857142857143 | 0.6287460933335888 | 0.04606510000121489 | 0.049169684998560115 |
| Hybrid + CE | 0.6924603174603174 | 0.869047619047619 | 0.719221936220623 | 0.48695730000144977 | 0.5658439600001657 |

Cross-Encoder versus Hybrid increased Recall@5 by 0.15674603174603186 and nDCG@10 by 0.09047584288703425, but reduced MRR@10 by 0.09523809523809534. Warm total time increased from 0.9168448999953398 to 9.797386600002937 seconds (10.685980365984188 times). Cases `dev-candidate-009` and `dev-candidate-010` both changed from first-relevant rank 1 to 3. In 009, the original rank-1 gold child actually moved to rank 4; another gold child became rank 3. The owner **does not approve Cross-Encoder activation**, given the measured CPU latency and ranking regressions despite the Recall/nDCG gains. No arbitrary quality or latency threshold is introduced.

These were exploratory offline CPU float32 timings on Windows AMD64, Intel Family 6 Model 186, 16 logical CPUs and four Torch threads: 20 observations per mode, fixed execution order, OS file cache not flushed. They are not production API or end-to-end generation latency. Recorded peak working set was 2180.84 MiB. Stage percentiles are not additive. Zero unauthorized retrieval hits were recorded across all 20 executions and all three role-sensitive pairs in each mode; this is bounded evidence, not universal security certification.

The ignored local evidence is [Step 112 report](../runtime/reports/step112_dev_20_four_way_diagnostic.json), [Step 114 analysis](../runtime/reports/step114_architecture_analysis.json), and [Step 114 owner packet](../runtime/reports/step114_owner_decision_packet.md). Their SHA-256 values are recorded below. These artifacts need not exist in CI. The Step 112 historical execution commit is `a9ca231409446cfa74fb380fdbbbad82948d5d39`; Step 114's analysis baseline is `be9b5e31b52e03875a9e0592f594cfaa19393155`. Step 116 verified their source binding against the current committed inputs at `d198c60fe8b78b7dde9cb5eb31287fc8ef2fdf28`. The historical Step 114 `AWAITING_OWNER_APPROVAL` status is preserved; this recorded owner decision resolves the v1 inclusion choice without rewriting historical artifacts.

The following decision record is enforced by `tests/eval/test_reranker_decision.py`; it records approval and existing settings, and is not a new runtime configuration source.

```json
{
  "decision_id": "step116_v1_retrieval",
  "decision_date": "2026-11-22",
  "owner_approved": true,
  "approval_source": "explicit_project_owner_step116_instruction",
  "deployment_baseline": "hybrid",
  "production_reranker_enabled": false,
  "evaluation_modes": ["bm25", "faiss", "hybrid", "hybrid_rerank"],
  "retrieval_configuration_change_approved": false,
  "budgets": {"bm25_top_k": 30, "faiss_top_k": 30, "rrf_k": 60, "hybrid_top_k": 20, "rerank_top_k": 20, "generation_evidence_max": 5},
  "faiss_only_selection": "DEFERRED_UNTIL_AFTER_V1_PORTFOLIO_RELEASE",
  "embedding_fine_tuning": "DEFERRED_UNTIL_AFTER_V1_PORTFOLIO_RELEASE",
  "post_release_changes_require_separate_approval": true,
  "step112_report_sha256": "ad38924f4cd46f28c95d891060a135756a6add125377d97c0fdb867a94b8eec0",
  "step114_analysis_sha256": "f0aca0f80e2dc874a618abf737595383e095797b7aec559dd941168245c70d90",
  "step114_packet_sha256": "44d62451a57fb75a3f2096cad521f9d9ff81a17519c91cfdc41712b6f5186fe3",
  "dataset_sha256": "d3280f8570ca81f2f1b2c300e9e2c1385a1d5113f07d7f42d1f286331551dc5b",
  "snapshot_id": "ddd98004d41bc5fcd9021078c036e3b07cbe94066f528a9bbbc44eef45453794",
  "development_diagnostic_complete": true,
  "contractual_frozen_ablation_complete": false,
  "phase5_status": "OPEN"
}
```

#### Step 116 Phase 5 exit audit

**Historical audit, retained for provenance.** The Step 116 decision record's `phase5_status: OPEN`, the dispositions below, and the sequencing tension describe the contract at commit `6e4c9878f75801f051d90db4d5385abe39c421e1`. The Step 117 owner-approved sequencing clarification in §§11, 15 and 17 supersedes the interpretation that frozen ablation blocks Phase 5. The Hybrid/CE architecture decision remains unchanged. This history does not declare the current Phase 5 closed.

The Phase 5 exit wording audited in Step 116 was:

> Exit: role isolation passes, 4-way ablation recorded, inclusion decision justified on dev metrics.

The mandatory phase order in `AGENTS.md` §18 and §15 below is unchanged. Read the §15 exit wording together with the feature acceptance matrix in §11 and retrieval-readiness milestone in §17; passing development diagnostics does not waive a frozen-benchmark requirement. This table identifies inspected evidence; actual Step 116 test executions and results must be reported separately.

| Required gate | Repository evidence and enforcing tests | Audit disposition |
|---|---|---|
| At least 20 documents ingested and validated (§17) | 26 documents, including 20 active and six superseded; `tests/integration/test_manifest_ingestion.py`, `tests/integration/test_corpus_manifest.py`, `tests/integration/test_ingestion_deterministic.py`. | Implemented evidence; requires passing applicable tests. |
| Reproducible clean-build snapshot (§17) | `ddd98004d41bc5fcd9021078c036e3b07cbe94066f528a9bbbc44eef45453794`; independent ingestions into fresh databases in separate processes, directories and hash seeds in `tests/integration/test_ingestion_deterministic.py::test_independent_clean_builds_in_separate_processes`; historical results in `docs/phase3-closeout.md`. | Implemented evidence; requires passing the clean-build test. |
| Development BM25 Recall@5 strictly greater than zero (§17) | Checked-in `eval/benchmarks/bm25_public_smoke.json`: isolated public-evidence Recall@5 = 1.0, one eligible seed case; `tests/eval/test_bm25_public_smoke.py`. Step 112's broader development BM25 result is 0.5476190476190476 over 14 positive cases. Keep those denominators distinct. | Measured milestone evidence; requires passing deterministic smoke regression. |
| Role isolation and retrieval-dependent security (§15; §11) | `tests/security/test_role_scoping.py`, `tests/security/test_role_faiss_scoping.py`, `tests/security/test_hybrid_role_scoping.py`, `tests/security/test_published_index_scoping.py`, `tests/security/test_retrieval_api_auth.py`, `tests/security/test_retrieval_api_scoping.py`, `tests/security/test_offline_reranking_scoping.py`, and `tests/security/test_dev_eval_role_scoping.py`; snapshot/publication tests enforce fail-closed bindings. | Implemented bounded coverage; requires passing security regressions. No claim about future generation/source/UI features. |
| Four-way development evidence (§15) | Historical Step 112 report and Step 114 review; deterministic checks in `tests/eval/test_dev_20_four_way_contract.py`, `tests/eval/test_dev_four_way_diagnostic.py`, and `tests/eval/test_dev_retrieval_architecture_review.py`. | Exploratory development diagnostics complete; all four modes remain available. |
| Human-approved inclusion decision justified on development metrics (§15; §11) | The Step 116 owner decision above, enforced by `tests/eval/test_reranker_decision.py`, retains Hybrid and disables production reranking. | Owner approval recorded; runtime behavior must match the decision. |
| Contractual retrieval ablation (§11) | Required: four configurations on the frozen benchmark, machine-readable `eval/report.json`, enforced by `tests/eval/test_ablation.py`. At the Step 116 baseline both files are absent and `make eval-retrieval` prints a Phase 7 placeholder and exits 1. Existing diagnostics explicitly have `full_ablation_complete: false`. | **NOT MET. Phase 5 remains OPEN.** A passing deterministic suite cannot substitute for this missing acceptance evidence. |

The definite remaining Phase 5 blocker is the contractual frozen-benchmark four-way ablation, its report and enforcing test, and an executable evaluation target. §15 schedules full frozen evaluation in Phase 7 while §11 places this ablation in the retrieval acceptance matrix; this sequencing tension is reported, not resolved by silently reclassifying the gate. Do not advance to Phase 6 until the actual Phase 5 gates pass, or the owner separately approves an explicit contract/phase-order amendment under §18. Step 116 approves neither a waiver nor such an amendment, and accesses no held-out cases. Future generation, citations, refusal, source access and UI gates remain pending; they are not reported as passed by this audit.


### Exact model versions

Capture pinned embedding and reranker model revisions and the installed Ollama model digest in the reproducibility manifest. Fail validation on unapproved model changes. Do not fabricate a digest; the first approval records the actual installed digest before a run can be declared reproducible.

Step 103 verified `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` revision `1427fd652930e4ba29e8149678df786c240d8825` against the actual provisioned files and completed a real CPU smoke.

| Verified file | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors` | 470592698 | `5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4` |
| `tokenizer.json` | 17082660 | `62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626` |
| `config.json` | 891 | `cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77` |
| `tokenizer_config.json` | 435 | `e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e` |
| `special_tokens_map.json` | 239 | `378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d` |

Provision only this allowlist outside Git. The offline loader uses the existing `torch==2.15.0+cpu` installation and `transformers==5.23.0` with CPU float32, `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`, and `use_fast=True`. Step 103 permits bounded smoke scoring with overlength rejection above 512 joint query/passage tokens including special tokens. Production overflow handling, segment aggregation and reranker activation remain subject to separate owner decisions. The development quality and latency inclusion gate above remains in force.

Enforcing checks are `tests/unit/test_cross_encoder_checkpoint.py`, `tests/unit/test_cross_encoder_scoring.py`, and the explicitly executed `tests/local/real_cross_encoder_smoke.py`.

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
Filename convention: `<document_id>.md` — enforced by `tests/integration/test_corpus_seed.py`.

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

### Reduced scope — if the owner approves the fallback

If the project owner approves the reduced scope under §17, evaluation uses **60 query-role cases**.

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

### Phase 5 — Development retrieval gate

Under the limited Step 117 owner approval in §17, Phase 5 requires the full approved **20 development query-role cases** in a **four-way retrieval comparison** across all four configurations: **BM25S, E5/FAISS, Hybrid RRF, and Hybrid + CE**. Record valid, reproducible development evidence using identical snapshot, labels, roles and query conditions across configurations, with the existing metric definitions, model revisions, latency measurements and provenance requirements intact.

Phase 5 also requires passing retrieval-readiness under §17 (at least 20 validated documents, a reproducible clean-build snapshot hash, and development BM25 Recall@5 strictly greater than 0), role-isolation/security evidence with zero unauthorized retrieval in applicable executed checks, and a documented, owner-approved reranker inclusion decision justified by development quality and measured latency. All applicable current-phase tests must pass normally. Enforcement: `tests/eval/test_dev_20_four_way_contract.py`, `tests/eval/test_reranker_decision.py`, `tests/eval/test_phase_gate_sequencing_contract.py`, and the applicable ingestion, snapshot and security tests named below.

### Phase 7 — Frozen retrieval gate

The existing **Retrieval ablation** criterion in the table below remains mandatory as the Phase 7 **four-way retrieval ablation**: execute all **60 frozen held-out cases** under the full scope through all four configurations, **BM25S, E5/FAISS, Hybrid RRF, and Hybrid + CE**, with identical snapshot, labels, roles and query conditions across configurations. `make eval-retrieval` must execute successfully, `tests/eval/test_ablation.py` must be implemented and pass normally, and the run must generate machine-readable `eval/report.json`.

Phase 7 retains end-to-end generation, citation integrity and semantic citation support, refusal accuracy, adversarial security testing, multilingual analysis, error analysis, measured p50/p95 latency on declared hardware, provenance and reproducibility evaluation under §§10, 12, 13 and 23. All existing security thresholds and release requirements remain mandatory. Full-scope evaluation remains **80 executions: 20 development + 60 frozen held-out**, with **48 answerable, 16 unanswerable and 16 role-sensitive executions**. No reduced-scope fallback is activated.

A completed development diagnostic is not a completed frozen evaluation. Do not relabel development-only evidence, set `full_ablation_complete` to true on that basis, waive a mandatory test, or report pending Phase 7 requirements as passed. The four evaluation configurations remain available even though the Step 116 v1 decision disables production Cross-Encoder reranking. Enforcement of the gate distinction: `tests/eval/test_phase_gate_sequencing_contract.py`.

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
| Development retrieval comparison (Phase 5) | All 20 approved development cases, four configurations, reproducible evidence, retrieval-readiness and security gates, and owner-approved inclusion decision pass. | `tests/eval/test_dev_20_four_way_contract.py`, `tests/eval/test_reranker_decision.py`, `tests/eval/test_phase_gate_sequencing_contract.py` |
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

**Phase 5 — Retrieval:** BM25 → normalized FAISS → RRF → conditional reranker and ablations. Exit: the full approved 20-case development four-way retrieval comparison, reproducible retrieval-readiness under §17, role-isolation/security gates, and documented owner-approved reranker inclusion decision pass under §11. Closure requires human review; development evidence does not complete Phase 7's frozen evaluation.

**Phase 6 — Generation:** Ollama, Pydantic output, citations, refusal. Exit: valid citations pass, invalid citations refuse, structure and ACL tests pass.

**Phase 7 — Full Evaluation:** active-scope frozen cases, ablations, adversarial suite, multilingual breakdown, error analysis and latency. Exit: all 60 frozen held-out cases complete the mandatory four-way retrieval ablation under §11, `make eval-retrieval` succeeds, `tests/eval/test_ablation.py` passes, and `eval/report.json` is generated; end-to-end generation, citation, refusal, adversarial security, multilingual analysis, measured latency, provenance and reproducibility requirements pass. All mandatory security and portfolio-release gates remain intact.

**Phase 8 — Delivery:** Docker, Streamlit, CI, README, architecture, threat model, evaluation report, three-minute demo. Exit: clean installation, tested API/UI, no secrets, actual results.

---

## 16. Four-Week Delivery Estimate

Four weeks is a planning estimate. A phase can advance once its measurable acceptance criteria pass, in the order specified in §15. Calendar dates neither delay progression nor trigger changes to scope.

| Estimated week | Work | Acceptance evidence |
|---|---|---|
| 1 | Contract, sample corpus, ingestion, offline development BM25 smoke test | Validated corpus, stable snapshot hash, baseline metric. |
| 2 | JWT/RBAC, authorized index selection, BM25/FAISS integration | Auth/grant/default-deny tests green; role-scoped retrieval underway. |
| 3 | RRF/reranker study, Ollama, citation/refusal flow | Four retrieval configurations, passing ACL and citation tests. |
| 4 | Full evaluation, deployment, CI, report, demo | Held-out and security results, clean setup, publishable materials. |

The development BM25 baseline is an isolated, offline evaluation smoke test that uses development cases and only validated, publicly accessible corpus evidence. Run it without exposing a retrieval API or granting document access.

The eight phases proceed in the order specified in §15. Production retrieval integration in Phase 5 begins only after Phase 4 authentication and RBAC are fully implemented and the Phase 4 exit criteria pass.

The measured retrieval-readiness milestone in §17 is mandatory before Phase 5 closes.

---

## 17. Retrieval Readiness and Optional Scope Fallback

**Owner decision (Step 59).** The project owner approved milestone-based progression. This supersedes the former end-of-Week-1 rule that automatically reduced scope when any of the three measurements failed. The measurement thresholds remain mandatory.

### Step 117 owner approval — evaluation-gate sequencing only

The project owner explicitly approved this narrow amendment after Step 116 was committed at `6e4c9878f75801f051d90db4d5385abe39c421e1`: Phase 5 may close after its approved 20-case development four-way retrieval comparison, reproducible retrieval-readiness and security gates, and documented owner-approved reranker inclusion decision pass. Phase 7 must still perform the full frozen 60-case four-way retrieval ablation through successful `make eval-retrieval`, passing `tests/eval/test_ablation.py`, and machine-readable `eval/report.json`, with all existing end-to-end, security, citation, refusal, latency, provenance and reproducibility requirements mandatory.

This approval is limited to phase sequencing. It resolves the historical Step 116 interpretation in §2 without waiving the frozen ablation criterion or changing the eight-phase order. It is separate from Step 59 scope governance, Step 90/103 model-runtime approvals and Step 116's architecture decision. Hybrid RRF remains selected for v1, production Cross-Encoder remains disabled, and BM25S, E5/FAISS, Hybrid RRF and Hybrid + CE remain available for evaluation. FAISS-only reconsideration and embedding fine-tuning remain deferred until after the v1 portfolio release and require separate approval.

Full scope remains **80 query-role executions: 20 development + 60 frozen held-out; 48 answerable + 16 unanswerable + 16 role-sensitive executions**. No reduced-scope fallback is activated. No evaluation scope reduction, label change, architecture/configuration change, or skipped/xfail mandatory test is authorized. All three retrieval-readiness conditions below remain unchanged. The governance in §18, testing policy in §20 and release acceptance in §23 remain unchanged.

A phase advances only after its applicable acceptance gates pass. Step 117 authorizes neither automatic Phase 5 closure nor a phase transition or Phase 6 implementation. After an evidence-backed audit, report `PHASE 5 READY FOR HUMAN CLOSURE REVIEW` only if every applicable Phase 5 gate passes; otherwise report `PHASE 5 OPEN` and the precise missing gate. Phase 7 remains pending until its own gates execute and pass; development-only diagnostics must never be reported as completed frozen evaluation. Enforcement: `tests/eval/test_phase_gate_sequencing_contract.py`.

### Retrieval-readiness milestone

Verify all three conditions before closing the retrieval-readiness milestone in Phase 5.

1. **At least 20 documents have been ingested and validated.**
2. **The snapshot hash is reproducible across clean runs.**
3. **A development-set BM25 baseline exists with Recall@5 strictly greater than 0.**

Collect the ingestion count and clean-build snapshot hashes in Phase 3 once ingestion and snapshot construction can run. Measure the isolated development BM25 baseline once its prerequisites are available, using the offline setup in §16. Record the actual results. Phase 2 retains its source-corpus exit criteria under §15, and Phase 4 authentication and RBAC remain prerequisites for production retrieval integration.

### Authorized full scope

Full scope remains authorized. It covers eight families in French and English and four families in Arabic, with 26 source documents (20 active and six superseded) and 80 query-role executions (20 development and 60 frozen held-out).

### Optional reduced-scope fallback

A failed measurement leaves the readiness milestone open. Full scope stays active until the project owner explicitly approves a scope change. In addition to the governance in §18, a reduction requires objective evidence of a blocker, a proposed scope-change record with an impact analysis, explicit project-owner approval, and preservation of all security, evaluation, and reproducibility safeguards. Record the approved decision and active scope in the evaluation/configuration manifests.

Drop **Procurement** and **Incident Reporting**. Keep Remote Work, Leave, Expenses, Travel, Access Control, Data Handling (six families).

Arabic only on **Remote Work** and **Expenses**.

Documents: **12 active FR/EN + 2 active AR + 6 superseded = 20 total**.

Evaluation: **36 answerable + 12 unanswerable + 12 role-sensitive executions (6 pairs) = 60**, split into **15 development + 45 frozen held-out**. Keep paired question families in the same split.

### Never cut the following

JWT verification; deterministic RBAC; role-scoped retrieval; deny-by-default ACL; citation integrity; controlled refusal; complete security adversarial scenarios; BM25; FAISS; RRF; conditional reranker evaluation; four-way retrieval ablation; end-to-end evaluation; reproduction evidence; Docker delivery.

The fallback reduces corpus, language, and label breadth while preserving security and verification depth. Adapt dataset counts in acceptance checks to the active scope.

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

### Step 90 owner approval — E5 CPU runtime

The owner approved only `torch==2.15.0` from the explicit PyTorch CPU index `https://download.pytorch.org/whl/cpu`, `transformers==5.23.0`, and `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Only PyTorch uses the CPU index. Keep both dependency pins and the checkpoint revision exact. Further substitutions require a separate §18 approval. The locked architecture, scope, and phase order continue to apply.

At baseline `cb1691a`, `uv run --locked --no-sync python -B -c "from transformers import AutoModel, AutoTokenizer"` exited 1 with `ModuleNotFoundError: No module named 'transformers'`. The corresponding `import torch` probe exited 1 with `ModuleNotFoundError: No module named 'torch'`. FAISS and NumPy imports worked, but the locked environment had no E5 runtime or cached model. A fresh locked install must select compatible CPU wheels for Windows AMD64 and Linux x86-64 on CPython 3.11. Keep dependency hashes in `uv.lock` and record the approved model revision and verified artifact hashes for offline reuse.

### Step 103 owner approval — selected reranker checkpoint

The owner approved the exact reranker revision and five verified artifacts recorded in §2, external provisioning, the offline CPU loader and bounded real smoke scoring. Existing dependencies continue to apply. Step 103 rejects overlength pairs without truncation, splitting or aggregation. Production reranker activation and production overflow handling require separate approval.

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
