# AskPolicy Codex Completion Checklist

## 1. Pre-task verification

- [ ] Read [AGENTS.md](../AGENTS.md), [docs/decisions.md](decisions.md), and this checklist.
- [ ] Identify the single requested feature or change and its scope.
- [ ] Identify the measurable acceptance criteria in `docs/decisions.md` §11 before coding.
- [ ] Identify the test files that enforce the applicable criteria.
- [ ] Confirm exactly which files may be edited before making changes.

## 2. Implementation verification

- [ ] Use the locked technology stack and respect module boundaries.
- [ ] Add or update enforcing tests in the same task as feature implementation.
- [ ] Keep edits within the approved scope and files.
- [ ] Preserve assertions and frozen evaluation labels.

## 3. Security verification

- [ ] Verify authentication and role boundaries where applicable.
- [ ] Verify that ACLs deny access by default where applicable.
- [ ] Verify citation integrity and controlled refusal on validation failure where applicable.
- [ ] Run applicable security regressions for changes affecting authentication, retrieval, generation, ACLs, citations, or source access.

## 4. Test verification

- [ ] List the tests actually executed and record `PASS`, `FAIL`, `SKIP`, and `XFAIL` separately.
- [ ] List unexecuted tests and explain why they were omitted. Report passes only for tests that ran and passed.
- [ ] Diagnose acceptance test failures before continuing. Document the reason for any temporary `XFAIL`.
- [ ] Declare a feature complete only after all mandatory tests pass normally.

## 5. Completion report

- [ ] List files changed and functionality implemented.
- [ ] List tests added or modified.
- [ ] List commands actually executed and report the observed test results.
- [ ] Report remaining issues, blockers, limitations, and any deviations from the approved contract.

## 6. Human approval

- [ ] Obtain explicit human approval under `docs/decisions.md` §18 before changing the stack, architecture, or scope.
- [ ] Stop after the requested task and wait for human review before starting another task.
