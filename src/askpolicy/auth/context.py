"""Resolve one signed read-role grant at the authenticated request boundary."""

from collections.abc import Callable, Coroutine
import re
from typing import Any, TypeGuard, cast
from weakref import WeakKeyDictionary

from fastapi import HTTPException, Request

from .jwt import (
    READ_ROLES,
    AuthenticationError,
    Capability,
    JWTVerifier,
    ReadRole,
    VerifiedIdentity,
)


_BEARER = re.compile(r"Bearer +([^\s]+)", re.ASCII | re.IGNORECASE)


class ActiveRoleError(ValueError):
    """A content-free error for role resolution or context provenance."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class AuthenticatedContext:
    """An immutable request handle populated by the authentication boundary.

    Verified data lives in an identity-keyed weak registry for the life of the
    request handle. A lookalike, copy, or manually constructed handle has no
    registered authority. Application code and configured verifiers are trusted.
    """

    __slots__ = ("__weakref__",)

    @property
    def subject(self) -> str:
        return _context_state(self)[0].subject

    @property
    def grants(self) -> tuple[ReadRole, ...]:
        return _context_state(self)[0].grants

    @property
    def active_role(self) -> ReadRole:
        return _context_state(self)[1]

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        return _context_state(self)[0].capabilities


_verified_contexts: WeakKeyDictionary[AuthenticatedContext, tuple[VerifiedIdentity, ReadRole]] = WeakKeyDictionary()


def is_verified_context(context: object) -> TypeGuard[AuthenticatedContext]:
    return type(context) is AuthenticatedContext and context in _verified_contexts


def _context_state(context: AuthenticatedContext) -> tuple[VerifiedIdentity, ReadRole]:
    if not is_verified_context(context):
        raise ActiveRoleError("UNVERIFIED_CONTEXT")
    return _verified_contexts[context]


def authenticate_request(
    verifier: JWTVerifier, token: str | None, *, active_role: str | None,
) -> AuthenticatedContext:
    """Verify a token and register a context for one explicitly granted role."""
    if type(verifier) is not JWTVerifier:
        raise AuthenticationError("INVALID_JWT_CONFIG")
    identity = verifier.verify(token)
    if type(active_role) is not str or active_role not in READ_ROLES:
        raise ActiveRoleError("INVALID_ACTIVE_ROLE")
    if active_role not in identity.grants:
        raise ActiveRoleError("ROLE_NOT_GRANTED")
    context = AuthenticatedContext()
    _verified_contexts[context] = (identity, cast(ReadRole, active_role))
    return context


def make_auth_dependency(
    verifier: JWTVerifier,
) -> Callable[[Request], Coroutine[Any, Any, AuthenticatedContext]]:
    """Build a FastAPI dependency for one Bearer and one X-Active-Role header.

    Authentication failures return 401, including malformed or duplicate
    Authorization headers. Authenticated requests with missing, malformed,
    duplicate, unknown or ungranted active-role declarations return 403.
    """
    if type(verifier) is not JWTVerifier:
        raise AuthenticationError("INVALID_JWT_CONFIG")

    async def require_context(request: Request) -> AuthenticatedContext:
        authorization = request.headers.getlist("Authorization")
        bearer = _BEARER.fullmatch(authorization[0]) if len(authorization) == 1 else None
        roles = request.headers.getlist("X-Active-Role")
        try:
            return authenticate_request(
                verifier, bearer[1] if bearer else None, active_role=roles[0] if len(roles) == 1 else None,
            )
        except AuthenticationError:
            raise HTTPException(
                status_code=401, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"},
            ) from None
        except ActiveRoleError:
            raise HTTPException(status_code=403, detail="Access denied") from None

    return require_context
