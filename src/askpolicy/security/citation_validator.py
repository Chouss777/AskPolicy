"""Fail-closed mechanical checks against the exact model-supplied evidence.

Trusted server code must supply the request's verified authentication context,
published release/snapshot IDs, and the exact subset of AuthorizedPassage values
actually sent to the model. Obtain those passages through hydrate_published_hits;
their dataclass type and matching bindings are not proof of hydration provenance.
Never populate this boundary from client/model metadata or raw rankings.

This module performs no search, hydration, publication lookup, or semantic review.
It cannot independently establish which passages reached a model, recheck document
ACL metadata absent from AuthorizedPassage, or detect a publication change outside
the trusted request context. The existing hydration boundary remains authoritative.
"""

from ..auth.context import AuthenticatedContext, is_verified_context
from ..generation.schemas import Citation
from ..retrieval.evidence_hydration import AuthorizedPassage
from ..storage.index_store import valid_id


class CitationValidationError(ValueError):
    """One stable failure code, without source details or nested exceptions."""

    code = "CITATION_VALIDATION_FAILED"

    def __init__(self) -> None:
        super().__init__(self.code)


def _validate(citations, evidence, context, release_id, snapshot_id) -> None:
    if (
        not is_verified_context(context)
        or not valid_id(release_id) or not valid_id(snapshot_id)
        or type(citations) is not list or not citations
        or type(evidence) is not tuple or not evidence
    ):
        raise ValueError

    role = context.active_role
    passages = {}
    # Check every supplied passage, including uncited entries. A valid citation
    # must not hide duplicate identities or an invalid mixed-context evidence set.
    for passage in evidence:
        if type(passage) is not AuthorizedPassage:
            raise ValueError
        if (
            type(passage.release_id) is not str or passage.release_id != release_id
            or type(passage.snapshot_id) is not str or passage.snapshot_id != snapshot_id
            or type(passage.active_role) is not str or passage.active_role != role
            or any(type(value) is not str or not value.strip() for value in (
                passage.child_id, passage.parent_id, passage.document_id,
            ))
            or type(passage.text) is not str or not passage.text
            or type(passage.start) is not int or type(passage.end) is not int
            or not 0 <= passage.start < passage.end
            or passage.end - passage.start != len(passage.text)
            or passage.child_id in passages
        ):
            raise ValueError
        passages[passage.child_id] = passage

    for citation in citations:
        if type(citation) is not Citation or citation.model_extra is not None:
            raise ValueError
        # Pydantic instances can be mutated or built with model_construct/copy.
        # Revalidate the fields without serializing or logging invalid payloads.
        checked = Citation.model_validate(vars(citation))
        passage = passages.get(checked.chunk_id)
        if passage is None or checked.quote not in passage.text:
            raise ValueError


def validate_citations(
    citations: list[Citation], evidence: tuple[AuthorizedPassage, ...], *,
    context: AuthenticatedContext, release_id: str, snapshot_id: str,
) -> None:
    """Accept a nonempty answer-citation batch only if every check passes.

    Success returns None; failure raises CitationValidationError, never a partial
    accepted list. Refusal handling belongs to the caller and does not invoke this
    answer-only check. Exact, case-sensitive Unicode substring matching performs
    no trimming or normalization and does not establish semantic entailment.
    On failure the caller must discard the entire answer and use server-controlled
    refusal rules. Model-provided status never grants authorization.
    """
    try:
        _validate(citations, evidence, context, release_id, snapshot_id)
        return None
    except Exception:
        pass
    # Raise outside the handler: sensitive nested exception frames must not be
    # attached as __context__. Callers must not log payloads or traceback locals.
    raise CitationValidationError()
