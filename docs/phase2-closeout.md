# Phase 2 Closeout — Corpus

**Status:** Complete. Step 63 records acceptance of the source corpus, checked-in manifest, and first manually reviewed development cases at baseline `3ce2cfe`.

## Exit criteria and evidence

[Decisions §15](decisions.md#15-mandatory-development-phase-order) defines the Phase 2 requirement verbatim:

> **Phase 2 — Corpus:** two or three representative synthetic Markdown policies (one generally accessible and one HR restricted), full YAML metadata, first manually checked dev cases. Exit: policy metadata and source links validate.

| Criterion | Repository evidence | Enforcing tests |
|---|---|---|
| Representative synthetic Markdown policies | [Travel FR](../data/policies/fr/TRAVEL-001-FR-V1.md) is `PUBLIC` with explicit grants to all four read roles. [Leave FR](../data/policies/fr/LEAVE-001-FR-V2.md) is `RESTRICTED` to `hr`. | [Seed corpus](../tests/integration/test_corpus_seed.py) checks exact front matter, filenames, sections, tables, and selected rules. |
| Full YAML metadata and valid policy metadata | All 26 sources have the 14 required front-matter fields. The [source manifest](../data/corpus_manifest.yaml) agrees with source metadata, paths, ACLs, provenance, and licenses. | [Corpus manifest](../tests/integration/test_corpus_manifest.py) checks exact coverage and metadata, recomputes source hashes, and rejects malformed records. |
| First manually checked development cases | The [development candidates](../eval/datasets/dev_candidates.jsonl) and [owner’s annotation review](annotation-review.md) record five approved seed cases, including the later full-corpus decision for Candidate 005. | [Development candidate tests](../tests/eval/test_dev_candidates.py) check the recorded approval, labels, role-sensitive pair, and empty evidence on refusals. |
| Valid source links | The [corpus rule matrix](corpus-rule-matrix.md) cites existing policy sections and records all six resolved supersession links. The approved answer labels cite sections accessible to their assigned roles. | [Batch 1](../tests/integration/test_corpus_batch1.py), [Batch 2](../tests/integration/test_corpus_batch2.py), and [Batch 3](../tests/integration/test_corpus_batch3.py) check rule references and version relationships. [Development candidate tests](../tests/eval/test_dev_candidates.py) check cited headings and read roles. |

## Corpus and manifest

Full scope remains authorized under [Decisions §17](decisions.md#17-retrieval-readiness-and-optional-scope-fallback). The table lists every slot in the [manifest](../data/corpus_manifest.yaml), with exact document identities and ACLs in [the rule matrix §E](corpus-rule-matrix.md#e-document-identities-and-acls).

| Policy family | Active FR | Active EN | Active AR | Superseded |
|---|---|---|---|---|
| Remote Work | `REMOTE-001-FR-V2` | `REMOTE-001-EN-V2` | `REMOTE-001-AR-V2` | `REMOTE-001-FR-V1`, `REMOTE-001-EN-V1` |
| Leave | `LEAVE-001-FR-V2` | `LEAVE-001-EN-V2` | `LEAVE-001-AR-V2` | `LEAVE-001-FR-V1` |
| Expenses | `EXPENSES-001-FR-V2` | `EXPENSES-001-EN-V2` | `EXPENSES-001-AR-V2` | `EXPENSES-001-FR-V1` |
| Travel | `TRAVEL-001-FR-V1` | `TRAVEL-001-EN-V1` | — | — |
| Access Control | `ACCESS-001-FR-V2` | `ACCESS-001-EN-V2` | — | `ACCESS-001-FR-V1` |
| Data Handling | `DATA-001-FR-V2` | `DATA-001-EN-V2` | `DATA-001-AR-V2` | `DATA-001-FR-V1` |
| Incident Reporting | `INCIDENT-001-FR-V1` | `INCIDENT-001-EN-V1` | — | — |
| Procurement | `PROCUREMENT-001-FR-V1` | `PROCUREMENT-001-EN-V1` | — | — |
| **Totals** | **8** | **8** | **4** | **6 (5 FR, 1 EN)** |

All six successor-to-predecessor links resolve:

| Successor | Predecessor |
|---|---|
| `ACCESS-001-FR-V2` | `ACCESS-001-FR-V1` |
| `DATA-001-FR-V2` | `DATA-001-FR-V1` |
| `EXPENSES-001-FR-V2` | `EXPENSES-001-FR-V1` |
| `LEAVE-001-FR-V2` | `LEAVE-001-FR-V1` |
| `REMOTE-001-EN-V2` | `REMOTE-001-EN-V1` |
| `REMOTE-001-FR-V2` | `REMOTE-001-FR-V1` |

Independent PowerShell/.NET verification recomputed all 26 SHA-256 values from the complete Markdown sources, including front matter, using strict UTF-8 and CRLF-to-LF normalization. All 26 matched the checked-in manifest. The [manifest acceptance test](../tests/integration/test_corpus_manifest.py) also checks source hashes and exact agreement with policy metadata and version links.

## Approved development cases

| Candidate | Role | Approved expectation | Recorded source |
|---|---|---|---|
| `dev-candidate-001` | `employee` | Answer with the standard lodging cap of 850 MAD per night | `TRAVEL-001-FR-V1` §4 |
| `dev-candidate-002` | `hr` | Answer with at least five working days’ notice for planned leave | `LEAVE-001-FR-V2` §2 |
| `dev-candidate-003` | `hr` | Answer that HR alone can access the register, balances, and supporting documents | `LEAVE-001-FR-V2` §4 |
| `dev-candidate-004` | `employee` | Refuse the same register question | No authorized supporting source |
| `dev-candidate-005` | `employee` | Refuse to invent or confirm a monthly sports-subscription contribution | No authorized supporting source |

The [Candidate 005 decision](annotation-review.md#candidate-005-final-approval) records the owner’s November 12, 2026 approval after review of the full 26-document corpus. None of the 15 employee-accessible current policies specifies the requested amount. Whether Atlas Services offers the benefit remains undetermined. The record has `revalidation_required: false`. Revalidate if eligible evidence changes.

## Regression evidence

Step 63 ran `make test`, which collected 324 tests. All 324 passed, with zero failures, skips, or expected failures. The command exited 0.

## Limitations

- The tests establish structural integrity, source consistency, and coverage of selected multilingual clauses. Independent multilingual certification remains pending. See [corpus rule matrix §G](corpus-rule-matrix.md#g-rules-needing-human-semantic-review).
- Travel’s standard cap remains 850 MAD per night. The procedure for emergency lodging above that cap is unspecified and excluded from evaluation until the owner approves an interpretation. See the [open Travel interpretation](annotation-review.md#open-travel-interpretation).
- The five approved cases satisfy the initial development-case requirement. The full evaluation dataset and held-out freeze remain later evaluation work.

## Phase 3 handoff

[Decisions §15](decisions.md#15-mandatory-development-phase-order) places ingestion next in the locked eight-phase order. The [test inventory](test-inventory.md) keeps Phase 3 metadata validation, chunk boundaries, runtime version selection, and ingestion determinism pending. Phase 3 must validate Markdown, preserve chunk structure and offsets, persist validated data in SQLite, reject malformed metadata, and demonstrate a stable snapshot hash across clean builds.

The measured [retrieval-readiness milestone](decisions.md#17-retrieval-readiness-and-optional-scope-fallback) remains open. The ingested and validated document count, clean-build snapshot reproducibility, and offline development BM25 Recall@5 remain unmeasured. Runtime authorization remains Phase 4 work and must pass before production retrieval integration in Phase 5. Step 63 closes Phase 2. The locked stack, scope, and architecture govern the handoff.
