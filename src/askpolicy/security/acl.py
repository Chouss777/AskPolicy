"""Evaluate explicit document grants for the verified active read role."""

from ..auth.context import is_verified_context
from ..auth.jwt import READ_ROLES


_CLASSIFICATIONS = {"PUBLIC", "INTERNAL", "RESTRICTED"}


def can_read_document(
    context: object, *, classification: object = None, allowed_roles: object = None,
) -> bool:
    """Deny invalid metadata and allow only an explicit active-role grant.

    Accept only a list or tuple of unique, known read roles. The caller is
    responsible for supplying the validated metadata of the intended document.
    """
    if not is_verified_context(context) or type(classification) is not str or classification not in _CLASSIFICATIONS:
        return False
    if type(allowed_roles) not in (list, tuple):
        return False
    roles = tuple(allowed_roles)
    if not roles or any(type(role) is not str or role not in READ_ROLES for role in roles):
        return False
    return len(set(roles)) == len(roles) and context.active_role in roles
