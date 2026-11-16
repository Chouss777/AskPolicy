"""Derive the trusted role scope for later index selection."""

from dataclasses import dataclass

from ..auth.context import AuthenticatedContext, is_verified_context
from ..auth.jwt import ReadRole


class IndexSelectionError(ValueError):
    """A content-free rejection of an unverified selection context."""

    code = "UNVERIFIED_CONTEXT"

    def __init__(self) -> None:
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class RoleIndexScope:
    role: ReadRole


def select_role_index(context: AuthenticatedContext) -> RoleIndexScope:
    """Return the active role scope without opening an index."""
    if not is_verified_context(context):
        raise IndexSelectionError()
    return RoleIndexScope(role=context.active_role)
