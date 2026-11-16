# Phase 3 Closeout — Ingestion

**Phase 3 is complete.** Step 77 verified both mandatory exit conditions against the implementation at `b4fb46f`. The draft snapshot identity remains **`UNPUBLISHED`**. Phase 4 security is the next phase in the locked order.

## Locked exit conditions and evidence

[Decisions §15](decisions.md#15-mandatory-development-phase-order) defines the gate verbatim.

> **Phase 3 — Ingestion:** Markdown → metadata checks → chunking → SQLite → deterministic snapshot hash. Exit: malformed metadata rejected and clean-rebuild snapshot hash stable.

[Decisions §11](decisions.md#11-acceptance-criteria-per-feature) requires named errors for malformed front matter and missing `allowed_roles` or `effective_date`, and matching snapshot hashes across two clean builds with the same canonical input and configuration.

| Requirement | Production code and boundary | Executed evidence |
|---|---|---|
| Reject malformed security metadata | The ingestion [metadata validator](../src/askpolicy/ingestion/metadata.py) parses the supported front-matter subset and validates typed fields, explicit roles, and ISO calendar dates. The [coordinator](../src/askpolicy/ingestion/coordinator.py) aborts on invalid metadata before writing. | [Metadata validation](../tests/unit/test_metadata_validation.py) passed 122 tests. Coordinator tests reproduce named errors for malformed front matter, missing roles, and missing effective dates even after recomputing the fixture's source hash. |
| Extract Markdown sections and chunks | The ingestion [extractor](../src/askpolicy/ingestion/chunking.py) retains section hierarchy, table headers and rows, original text spans, and document, version, language, and ACL metadata on derived records. | [Chunk boundaries](../tests/integration/test_chunk_boundaries.py) passed 65 tests covering real corpus spans, table structure, stable IDs, and rejection of cross-document, cross-version, or cross-ACL parent-child links. |
| Commit a validated SQLite draft | The [manifest coordinator](../src/askpolicy/ingestion/coordinator.py) checks corpus completeness, paths, source hashes, metadata, and supersession before passing extracted records to [SQLite storage](../src/askpolicy/storage/sqlite.py) for one write transaction. | [Manifest ingestion](../tests/integration/test_manifest_ingestion.py) passed 55 tests, and [SQLite persistence](../tests/integration/test_sqlite_persistence.py) passed 37. The suites reopened a committed database, rejected invalid or partial imports and broken source hashes, and used SQLite triggers to induce failures after real inserts and verify that all batch rows rolled back. |
| Reproduce the draft snapshot hash | The [snapshot identity module](../src/askpolicy/storage/snapshot.py) reads one committed SQLite view, checks every document against the trusted manifest, and replays extraction to verify stored sections and chunks before hashing. | [Ingestion determinism](../tests/integration/test_ingestion_deterministic.py) passed 44 tests. Two independent full ingestions yielded identical identities. Tests also reject direct SQL changes to ACLs, version links, source hashes, spans, and chunk text, including source edits with a recomputed SQLite hash. |

The real coordinator, extractor, and SQLite engine execute on the acceptance path. The failure injections exercise production rollback and rejection logic. All six targeted suites passed normally under the [Codex checklist](codex-checklist.md).

## Corpus and supersession

The committed draft contained **26 documents, 161 parent sections, and 387 child chunks**. The [approved source manifest](../data/corpus_manifest.yaml) covers all 26 documents and preserves their metadata, paths, and canonical source hashes. The coordinator verified every source and resolved all six links.

| Successor | Predecessor |
|---|---|
| `ACCESS-001-FR-V2` | `ACCESS-001-FR-V1` |
| `DATA-001-FR-V2` | `DATA-001-FR-V1` |
| `EXPENSES-001-FR-V2` | `EXPENSES-001-FR-V1` |
| `LEAVE-001-FR-V2` | `LEAVE-001-FR-V1` |
| `REMOTE-001-EN-V2` | `REMOTE-001-EN-V1` |
| `REMOTE-001-FR-V2` | `REMOTE-001-FR-V1` |

## Clean-rebuild identity

Step 77 reran the [separate-process build test](../tests/integration/test_ingestion_deterministic.py). Two independent Windows Python 3.11 processes ran the real coordinator with the same full corpus and configuration into fresh SQLite databases in different directories, with different filenames and Python hash seeds. Both builds returned the same counts and four digests.

| Field | SHA-256 from both builds |
|---|---|
| `candidate_snapshot_id` | `ddd98004d41bc5fcd9021078c036e3b07cbe94066f528a9bbbc44eef45453794` |
| `content_hash` | `01c8b608b3db10d6d4bdaed050aa2c192c2befcfdba007f106771e9b63d99c94` |
| `config_hash` | `8895e443ca6227fe66976bf6ce671bf5e426f680726e9eb5363f48264e9a7715` |
| `source_manifest_hash` | `3306035cb2742ae2323b9d7c1a73f5ea128a08ddf85f17bef7fefcaa20ad9960` |

Each digest uses SHA-256 over UTF-8 JSON framed as a domain label and payload. The content component covers ordered logical SQLite rows and counts, including ACLs, stored source hashes, and version links. The manifest component covers all 16 fields of each approved record in document-ID order, preserving explicit nulls and role order. The configuration component covers snapshot format 1, `utf8-lf-v1`, `structural-blocks-v1`, `canonical-json-v1`, and final identity encoding. The hashing inputs omit database paths, timestamps, and randomness. Separate tests verify invariance to source CRLF, manifest reordering, and physical SQLite row order.

The default `json-object-v1` payload names `version`, `content_hash`, `config_hash`, and `source_manifest_hash`. The alternate `json-array-v1` payload uses those four values in that order. Each uses its own `askpolicy/draft-snapshot/identity/<encoding>` domain. The tests independently reconstruct both payload digests and verify that only the configuration digest and candidate ID change when switching encodings.

Manifest integrity verifies the agreement of stored corpus evidence with approved sources. Runtime authorization must separately verify the identity, active role, and grants before any evidence can be accessed. The fingerprint is `UNPUBLISHED` and represents a validated draft.

## Policy eligibility

The [eligibility selector](../src/askpolicy/storage/eligibility.py) implements approved **and** effective **and** unexpired **and** non-superseded selection. Effective dates are inclusive and expirations exclusive. Supersession structurally excludes the predecessor even when the successor is ineligible. The result carries the validity date and earliest known future effective or expiration date for reevaluation.

[Version selection](../tests/integration/test_version_selection.py) passed 33 tests. At the fixed date **September 1, 2026**, the real corpus yielded **20 eligible current versions** (8 French, 8 English, 4 Arabic) and **6 excluded superseded versions**. These counts apply to the tested date and corpus. Selection is role-independent. For example, an eligible Leave policy with `RESTRICTED` classification still has `allowed_roles: [hr]`.

## Phase 4 entry and remaining gates

1. Begin with signed JWT verification in the authentication boundary under [AGENTS.md §9](../AGENTS.md#9-authentication-and-rbac) and [Decisions §6](decisions.md#6-authentication-and-authorization). Use the locked PyJWT stack and verify signature, expiry, issuer, and audience. The first enforcing test is `tests/security/test_auth.py`, including HTTP 401 for missing, expired, and forged tokens.
2. Verify exactly one active read role against backend-verified grants. Enforce HTTP 403 for an ungranted role, default-deny for missing or unknown classifications and grants, and separate ingestion-admin capability from read access. Define the role-scoped index-selection interface and exercise the first adversarial cases. Phase 4 exits when the authentication, grant, and default-deny tests pass. Retrieval-dependent security checks become mandatory once retrieval exists. [Decisions §11](decisions.md#11-acceptance-criteria-per-feature), [Decisions §15](decisions.md#15-mandatory-development-phase-order)
3. Phase 5 production retrieval requires the Phase 4 exit gate. Bind eligibility to the exact validated snapshot and enforce published validity transitions. Build role-scoped BM25S and FAISS indexes, verify SQLite/index identities, chunk membership, and pinned revisions and digests where applicable, and publish atomically only after successful builds. Missing or mismatched active indexes must produce HTTP 503. [Decisions §5](decisions.md#5-storage-and-snapshot-architecture), [Decisions §16](decisions.md#16-four-week-delivery-estimate)
4. The [retrieval-readiness milestone](decisions.md#17-retrieval-readiness-and-optional-scope-fallback) remains open. This audit verifies the minimum ingested document count and clean-build hash. The offline development BM25 baseline using validated publicly accessible evidence still needs a measured Recall@5 strictly greater than zero. Full project scope remains authorized.

## Deferred work and audit limits

- [Configuration](../src/askpolicy/config.py) defines a settings schema. Runtime `.env` loading remains deferred to runtime bootstrap because §15's Phase 3 gate does not require it. The existing inventory deferral remains visible. Adding `pydantic-settings` would require the locked stack-change procedure. [Decisions §18](decisions.md#18-when-the-stack-may-change)
- The [Makefile](../Makefile) runs pytest through `make test`. `make ingest` still displays a Phase 3 placeholder. Its documented production recipe requires a publishable SQLite/index snapshot, while the Phase 3 exit gate requires a validated SQLite draft and reproducible hash. Production ingest, setup, evaluation, run targets, and deployment remain pending. [AGENTS.md §15](../AGENTS.md#15-required-make-targets)
- The Windows directory-escape test tries a real junction and falls back to simulated reparse-point metadata if Windows denies creation. Test output does not identify the branch taken. The file-symlink test simulates filesystem metadata. Passing these tests establishes rejection before redirected traversal or reads under the tested conditions. Clean-build verification ran on Windows.

## Observed test results

| Run | Pass | Fail | Skip | Xfail | Exit |
|---|---:|---:|---:|---:|---:|
| Six Phase 3 suites via `uv run --locked --no-sync python -m pytest` with the paths linked above and `-q` | 356 | 0 | 0 | 0 | 0 |
| `make test` | 680 | 0 | 0 | 0 | 0 |
| `uv run --locked --no-sync python -m pytest tests/integration/test_ingestion_deterministic.py -q -s -k independent_clean_builds` | 1 | 0 | 0 | 0 | 0 |

The digest capture deselected 43 cases that had already passed in the full runs. Runtime authentication, publication, index consistency, and model evaluation were outside this audit and remain pending in the [test inventory](test-inventory.md).
