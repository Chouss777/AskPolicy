---
policy_id: ACCESS-001
document_id: ACCESS-001-EN-V2
version: 2
language: en
title: Access Control Policy
department: IT Security
classification: RESTRICTED
allowed_roles: [it_security]
effective_date: "2026-09-01"
expires_at: null
supersedes: null
approval_status: approved
source_type: synthetic_policy
source_license: project_owned
---

# Access Control Policy

## 1. Purpose and scope

This internal procedure of Atlas Services, a fictional Moroccan company, is for IT Security. It governs granting, modifying, and revoking work access to company resources.

## 2. Request validation

The request identifies the resource, the business need, the requested scope, and the end date for temporary access. The line manager confirms the need. IT Security verifies the requester's work identity and checks whether the permissions fit the assignment.

**Conditional approval.** If the need and scope are justified, access may be granted after line manager validation and IT Security approval.

If the request is incomplete or the requested scope exceeds the need, IT Security asks for a correction before granting access.

## 3. Least-privilege grants

| Request type | Permitted scope | Control |
|---|---|---|
| New access | Functions and resources necessary for the assignment | Verify the validated need before granting access |
| Change | Permissions necessary for the new tasks | Remove permissions that are no longer needed |
| Temporary access | Only the resources and duration strictly necessary | Record the end date and remove access at expiry |

IT Security records the permissions granted, the approvals, and the duration. It limits each grant to the necessary scope.

## 4. Review and revocation

IT Security reviews permissions when a role changes and at scheduled reviews. The line manager confirms any remaining needs. Permissions that are no longer needed are removed. When someone leaves, IT Security revokes access as soon as the departure takes effect, upon notification from the responsible department. IT Security records the decision and the date it was carried out.

## 5. Exceptions and follow-up

**Exception.** If a temporary need is urgent, IT Security may prioritize the request after line manager validation. IT Security approval is still required before any access is granted.

IT Security documents the justification, its approval, the scope, and the end date. It verifies removal at expiry and records the follow-up. If the need is disputed, the grant remains suspended during review.
