"""Guard untrusted generation output using trusted request evidence only.

Server code owns eligibility, verified request authority, current published IDs,
and the exact hydrated evidence subset sent to the model. Neither this enum nor
AuthorizedPassage is a provenance receipt; never construct trusted inputs from
client or model metadata. This module does no retrieval, hydration, or inference.
Mechanical citation validation does not establish semantic support or truth.
"""

from enum import Enum
import json

from fastapi import HTTPException

from ..auth.context import ActiveRoleError, AuthenticatedContext, is_verified_context
from ..auth.jwt import AuthenticationError
from ..generation.schemas import GenerationOutput
from ..retrieval.evidence_hydration import AuthorizedPassage
from .citation_validator import validate_citations


class EvidenceEligibility(Enum):
    """Closed server decisions; strings, booleans and client reasons are invalid."""

    ELIGIBLE = "eligible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNAUTHORIZED = "unauthorized"
    UNAVAILABLE = "unavailable"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    UNCERTAIN_VALIDITY = "uncertain_validity"


def _controlled_refusal() -> GenerationOutput:
    return GenerationOutput(status="refusal", answer=None, citations=[])


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError


def _checked_answer(raw_response, context, evidence, release_id, snapshot_id):
    # Accept JSON text only. Reject mappings and Pydantic instances rather than
    # treating earlier validation, model_construct, or model_copy as authority.
    if type(raw_response) is not str:
        raise ValueError
    # Reject ambiguous keys/nonstandard numbers, then let Pydantic's JSON
    # parser also reject malformed Unicode such as unpaired surrogate escapes.
    json.loads(
        raw_response, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
    )
    output = GenerationOutput.model_validate_json(raw_response)
    if output.status != "answer":
        raise ValueError
    validate_citations(
        output.citations, evidence, context=context,
        release_id=release_id, snapshot_id=snapshot_id,
    )
    return output


def guard_generation_response(
    raw_response: str, *, context: AuthenticatedContext,
    evidence: tuple[AuthorizedPassage, ...], release_id: str, snapshot_id: str,
    eligibility: EvidenceEligibility,
) -> GenerationOutput:
    """Preserve a fully checked answer or construct the same safe refusal.

    Authenticate before calling this function. Unverified handles raise the
    existing ActiveRoleError; authentication/role errors and HTTP 401/403 from
    dependencies propagate unchanged. Callers must preserve their auth handling.
    This is not an HTTP endpoint or a replacement authentication system.

    Eligibility is server-only, checked without coercion. All denied decisions,
    malformed generation data, model refusals and validation failures return a
    fresh refusal with no reason or source metadata. No partial answer escapes.
    Do not log raw inputs, error details, or traceback locals at the call site.
    """
    if not is_verified_context(context):
        raise ActiveRoleError("UNVERIFIED_CONTEXT")
    if type(eligibility) is not EvidenceEligibility or eligibility is not EvidenceEligibility.ELIGIBLE:
        return _controlled_refusal()
    try:
        return _checked_answer(raw_response, context, evidence, release_id, snapshot_id)
    except (AuthenticationError, ActiveRoleError):
        raise
    except HTTPException as error:
        if error.status_code in (401, 403):
            raise
    except Exception:
        pass
    # Never copy a model's refusal or expose a caught exception as public data.
    return _controlled_refusal()
