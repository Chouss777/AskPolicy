"""Trusted binding of the explicit Step 125 owner decision to the existing client.

Keyword arguments only check consistency with that decision; they cannot select
another model or grant approval. Never populate approval from request data,
environment variables or an observed installed digest. No I/O occurs here.
The client still requires explicit approval and verifies installed identity.
"""

from .ollama_client import MODEL, GeneratorApproval, OllamaClientError


_APPROVED_DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
_APPROVAL_REFERENCE = "docs/decisions.md#step-125-generator-digest-approval"


def get_generator_approval(
    *,
    model: str = MODEL,
    digest: str = _APPROVED_DIGEST,
    approval_reference: str = _APPROVAL_REFERENCE,
) -> GeneratorApproval:
    """Return the owner-approved record, rejecting any inconsistent override.

    This trusted configuration records approval, not daemon or inference
    readiness. Its public error contains no supplied values.
    """
    if (
        type(model) is not str or model != MODEL
        or type(digest) is not str or digest != _APPROVED_DIGEST
        or type(approval_reference) is not str or approval_reference != _APPROVAL_REFERENCE
    ):
        raise OllamaClientError("OLLAMA_MODEL_NOT_APPROVED")
    return GeneratorApproval(MODEL, _APPROVED_DIGEST, _APPROVAL_REFERENCE)
