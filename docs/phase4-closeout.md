# Phase 4 Closeout — Security

**Phase 4 is complete.** Step 83 inspected the production authentication and authorization boundaries at `9a46393` and reran the named acceptance suites. All three mandatory exit checks passed. The Phase 3 draft identity remains `UNPUBLISHED`. Production retrieval and snapshot publication remain pending.

## Locked exit gate and evidence

[Decisions §15](decisions.md#15-mandatory-development-phase-order) states the gate verbatim.

> **Phase 4 — Security:** signed JWT, verified grants, active read role, deny-by-default and role-index selection interfaces, first attack checks. Exit: auth/grant/default-deny tests pass; retrieval-dependent integration security tests become mandatory once retrieval exists.

| Gate or required interface | Production evidence | Observed test evidence |
|---|---|---|
| Authentication | [JWT verifier](../src/askpolicy/auth/jwt.py) verifies signatures with an explicit server-side HS256 allowlist. It requires a nonempty subject, exact issuer and audience, integer UTC `iat` and `exp`, and a signed `roles` array. It checks optional `nbf` and keeps signed administrative capabilities separate. | [Authentication tests](../tests/security/test_auth.py), 79 passed. Real signed tokens exercise the verifier and ASGI dependency. Missing, expired, forged, malformed, unsigned, and unsupported-algorithm tokens return HTTP 401. Wrong issuer or audience, missing claims, malformed time claims, duplicate grant values or JWT fields, and unknown roles or capabilities also fail closed. |
| Verified grants and one active read role | [Request authentication](../src/askpolicy/auth/context.py) verifies the bearer token and checks exactly one `X-Active-Role` against its signed grants. It creates a fresh immutable context for each authenticated request. | [Role-grant tests](../tests/security/test_role_grant.py), 19 passed. An employee token requesting HR receives HTTP 403. Missing, malformed, or duplicate role selections fail. Query and body role claims have no authority. Repeated requests can use different granted roles without inheriting another request's active role or combining grants. |
| Default-deny ACL | [Document ACL](../src/askpolicy/security/acl.py) uses only a verified context's active role. `PUBLIC`, `INTERNAL`, and `RESTRICTED` each require a known classification and explicit, unique, recognized role grants. Missing, unknown, or malformed values and unverified contexts deny access. | [Default-deny tests](../tests/security/test_default_deny.py), 27 passed. The ACL denies an HR-only policy to an employee-active context even when the identity also holds the HR grant. These tests exercise the ACL decision directly. Search and source-return enforcement remain pending. |
| Administrative isolation | The [verifier](../src/askpolicy/auth/jwt.py) separates `ingestion_admin` from read grants. The [request boundary](../src/askpolicy/auth/context.py) permits only explicitly granted read roles. | [Admin isolation tests](../tests/security/test_admin_isolation.py), 7 passed. An admin-only token cannot activate any read role. An admin token with an employee grant activates employee and receives only that role's privileges. |
| Role-index selection interface | [Role-index selection](../src/askpolicy/security/index_selection.py) derives an immutable role key solely from a verified active-role context. It accepts no caller-supplied namespace override and performs no index I/O. | [Role-index tests](../tests/security/test_role_index_selection.py), 8 passed. They reject a forged context and demonstrate deterministic selection with no global fallback. The role key is preparation for building and selecting authorized indexes. |

The audit inspected the source boundaries under [AGENTS.md §4](../AGENTS.md#4-module-boundaries) and the role contract under [AGENTS.md §9](../AGENTS.md#9-authentication-and-rbac). [The architecture test](../tests/unit/test_architecture_boundaries.py) verifies package layout. Retrieval and UI isolation require their dedicated integration suites.

## Executable API and configuration

The [FastAPI factory](../src/askpolicy/api/app.py) validates configuration and constructs the real verifier and authentication dependency before creating the application. [Runtime configuration](../src/askpolicy/config.py) reads `JWT_SECRET`, `JWT_ISSUER`, and `JWT_AUDIENCE` from the process environment or a trusted explicit mapping. The existing config validation rejects missing, weak, or placeholder settings. No default key, automatic `.env` parsing, or new dependency was introduced. Explicit `JWTConfig` injection is a Python factory argument outside the HTTP interface.

The only registered route is `GET /auth/verify`, which returns `authenticated` and the verified `active_role` with `Cache-Control: no-store`. Documentation and OpenAPI routes are disabled. The application has no route for documents, draft data, or retrieval. [API bootstrap tests](../tests/security/test_api_bootstrap.py), 46 passed, verify actual HTTP 200, 401, and 403 responses, fail-closed startup, credential hygiene on the tested paths, dependency registration, and the absence of document or index I/O. Uvicorn's factory was loaded without opening a listener. [Runtime configuration tests](../tests/unit/test_runtime_security_config.py), 21 passed, verify the loader and existing settings compatibility.

JWT verification requires a token signed by a configured trusted issuer. Username/password login, Argon2 credential verification, token issuance, refresh or revocation, managed synthetic identities, and persistent sessions are application capabilities still to be implemented. [Decisions §15](decisions.md#15-mandatory-development-phase-order) does not include them in the Phase 4 exit gate.

## Adversarial coverage and deferred integration

[Decisions §9](decisions.md#9-security-threat-model) and [Decisions §12](decisions.md#12-security-test-scope) define the broader attack suite. The audit confirmed missing, expired, and forged token rejection and ungranted role activation denial at the HTTP boundary. Query, body, and prompt role impersonation were exercised at authentication. Unknown classification, employee attempts to use HR rights, and ingestion-admin read isolation were exercised at the ACL or active-role boundary.

| Integration requirement | Pending evidence at this baseline |
|---|---|
| Retrieval role isolation and expired or superseded policy exclusion | `tests/security/test_role_scoping.py` is absent. Actual role-scoped indexes, runtime snapshot validity, and employee queries for HR-only passages still need integration coverage. |
| Direct source access and parent expansion | `tests/security/test_source_access.py` and `tests/security/test_parent_expansion.py` are absent. Before enabling those paths, enforce fresh source authorization and parent expansion within document, version, snapshot, and ACL boundaries. |
| Role switching, session state, and Streamlit | `tests/security/test_role_switch.py` and `tests/integration/test_ui_cache_isolation.py` are absent. Request-local isolation is verified. Future role-dependent answers, chunks, citations, and UI state need invalidation and their own enforcement tests. |
| Operational logging | `tests/security/test_logging_hygiene.py` is absent. The authentication and bootstrap tests check their captured logs and responses. Pipeline-wide logging controls and verification remain pending. |
| Generation and full adversarial runs | Indirect policy prompt injection, fabricated or out-of-set citation IDs, quotation mismatch, controlled refusal, and the full cross-phase adversarial suite require later pipeline integrations. |

The [test inventory](test-inventory.md) retains the locked paths and planning phases for the deferred suites. This closeout covers the Phase 4 gate, with production deployment and full attack coverage outside the evidence collected.

## Phase 5 handoff and security gates

The Phase 4 authentication and RBAC prerequisite in [Decisions §16](decisions.md#16-four-week-delivery-estimate) is satisfied. For production retrieval, select indexes by the verified active role before any search. Global search followed by filtering is prohibited. Verify the published SQLite snapshot ID, chunk identities, and relevant model revisions against both role-scoped indexes. A missing or mismatched active index must fail closed with HTTP 503. Validate new builds before atomic publication, preserve the prior valid snapshot when a build fails, and enforce policy validity transitions. [Decisions §5](decisions.md#5-storage-and-snapshot-architecture), [Decisions §6](decisions.md#6-authentication-and-authorization)

Make role isolation and snapshot consistency tests mandatory as retrieval is integrated. Require the corresponding authorization tests before enabling source return or parent expansion. Recheck ACLs before reranking, generation, and source return. Phase 5 exits after role isolation passes, a four-way ablation is recorded, and the reranker decision is justified with development metrics. [Decisions §11](decisions.md#11-acceptance-criteria-per-feature), [Decisions §15](decisions.md#15-mandatory-development-phase-order)

The first Phase 5 task should be the isolated offline BM25S development smoke test with validated, publicly accessible evidence. Record the corpus identity, development cases, and measured Recall@5. Run it without a retrieval API or changes to document access. Phase 3 already verified the validated ingestion count and clean-rebuild hash. The remaining readiness measurement must show development-set Recall@5 strictly greater than zero before Phase 5 closes. [Phase 3 closeout](phase3-closeout.md), [Decisions §17](decisions.md#17-retrieval-readiness-and-optional-scope-fallback)

## Observed verification

| Executed command | Pass | Fail | Skip | Xfail | Exit |
|---|---:|---:|---:|---:|---:|
| `uv run --locked --no-sync python -m pytest` with the seven Phase 4 test paths linked above and `-v` | 207 | 0 | 0 | 0 | 0 |
| `uv run --locked --no-sync python -m pytest tests/security/ -q` | 186 | 0 | 0 | 0 | 0 |
| `make test` | 887 | 0 | 0 | 0 | 0 |
