# Development Annotation Review

## Review scope

The project owner approved the five French development candidates after reviewing the Step 42 evidence from the two synthetic seed policies, `TRAVEL-001-FR-V1` and `LEAVE-001-FR-V2`. The approval applies to the current two-policy seed corpus. No held-out dataset was reviewed or approved.

## Relevance grade interpretation

Grade 3 = directly and fully relevant evidence that explicitly supports the required answer.

This interpretation applies to the existing grade-3 annotations in this seed review. The full relevance grading rubric remains to be established.

## Decisions

| Candidate | Human review decision |
|---|---|
| `dev-candidate-001` | Approved |
| `dev-candidate-002` | Approved |
| `dev-candidate-003` | Approved |
| `dev-candidate-004` | Approved |
| `dev-candidate-005` | Approved conditionally |

`dev-candidate-005` is unanswerable in the current seed corpus. Its `revalidation_required` flag records an outstanding human review obligation for the expanded corpus.

## Revalidation gate

Revalidate `dev-candidate-005` after corpus expansion and before evaluation dataset freezing.

1. Inspect the expanded active corpus.
2. Apply the `employee` role's authorized-source visibility rules.
3. Determine whether an authorized policy answers the sports-subscription question.
4. Keep, revise, or replace the candidate based on the actual evidence.
5. Record the human review decision.

Do not automatically clear `revalidation_required`.

## Limitations

- Structural tests check record shape and source references. Retrieval quality requires separate evaluation.
- Runtime RBAC has not been implemented.
- Full semantic YAML validation is unavailable.
- The complete development evaluation dataset is still pending.
