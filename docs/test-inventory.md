# Acceptance Test Inventory

The criterion names and enforcing test paths come from `docs/decisions.md` §11. Phases follow §15. A criterion is pending until its enforcing tests are implemented. Phase 5 records the retrieval ablation and reranker decision. Phase 7 runs the full frozen evaluation and adversarial suite.

The three Phase 1 entries have meaningful tests. Repository contract coverage is limited to the current directory and file checks. Module boundaries coverage is limited to package layout. The broader repository exclusion checks and import and UI isolation checks remain unverified.

| Acceptance criterion | Enforcing test path | Intended implementation phase |
|---|---|---|
| Ingestion determinism | `tests/integration/test_ingestion_deterministic.py` | 3 — pending |
| Metadata validation | `tests/unit/test_metadata_validation.py` | 3 — pending |
| Snapshot integrity | `tests/integration/test_snapshot_consistency.py` | 5 — pending |
| Authentication | `tests/security/test_auth.py` | 4 — pending |
| Role grants | `tests/security/test_role_grant.py` | 4 — pending |
| Role-scoped retrieval | `tests/security/test_role_scoping.py` | 5 — pending |
| Default-deny ACL | `tests/security/test_default_deny.py` | 4 — pending |
| Refusal indistinguishability | `tests/security/test_refusal_indistinguishability.py` | 6 — pending |
| E5 prefixes | `tests/unit/test_e5_prefix.py` | 5 — pending |
| FAISS normalization | `tests/unit/test_faiss_normalization.py` | 5 — pending |
| Citation valid | `tests/unit/test_citation_validator.py` | 6 — pending |
| Citation invalid | `tests/unit/test_citation_validator.py` | 6 — pending |
| Fail-closed generation | `tests/integration/test_fail_closed.py` | 6 — pending |
| Reranker decision | `tests/eval/test_reranker_decision.py` | 5 — pending |
| Retrieval ablation | `tests/eval/test_ablation.py` | 5 — pending |
| End-to-end | `tests/eval/test_e2e.py` | 7 — pending |
| Streamlit cache isolation | `tests/integration/test_ui_cache_isolation.py` | 8 — pending |
| Corpus manifest | `tests/integration/test_corpus_manifest.py` | 2 — implemented |
| Parent-child structure | `tests/integration/test_chunk_boundaries.py` | 3 — pending |
| Policy validity | `tests/integration/test_version_selection.py` | 3 — pending |
| Atomic publication | `tests/integration/test_snapshot_publication.py` | 5 — pending |
| Rebuildability | `tests/integration/test_index_rebuild.py` | 5 — pending |
| Embedding consistency | `tests/integration/test_embedding_revision.py` | 5 — pending |
| RRF correctness | `tests/unit/test_rrf.py` | 5 — pending |
| Role switching | `tests/security/test_role_switch.py` | 4 — pending |
| Ingestion admin isolation | `tests/security/test_admin_isolation.py` | 4 — pending |
| Source access | `tests/security/test_source_access.py` | 4 — pending |
| Parent authorization | `tests/security/test_parent_expansion.py` | 5 — pending |
| Prompt injection | `tests/security/test_prompt_injection.py` | 6 — pending |
| Logging hygiene | `tests/security/test_logging_hygiene.py` | 4 — pending |
| Generation schema | `tests/unit/test_generation_schema.py` | 6 — pending |
| Dataset split | `tests/eval/test_dataset_split.py` | 7 — pending |
| UI/API boundary | `tests/integration/test_ui_api_boundary.py` | 8 — pending |
| Docker topology | `tests/integration/test_compose_contract.py` | 8 — pending |
| Ollama readiness | `tests/integration/test_ollama_readiness.py` | 8 — pending |
| Makefile targets | `tests/integration/test_make_targets.py` | 8 — pending |
| Repository contract | `tests/integration/test_repository_contract.py` | 1 — implemented |
| Locked stack | `tests/unit/test_stack_contract.py` | 1 — implemented |
| Module boundaries | `tests/unit/test_architecture_boundaries.py` | 1 — implemented |

## Open deferrals

- config.py environment loading — deferred to Phase 3 — may require pydantic-settings — subject to docs/decisions.md §18
