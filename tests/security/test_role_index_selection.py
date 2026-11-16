from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
import socket
import sqlite3
from types import SimpleNamespace

import pytest

from .test_auth import secured


@pytest.mark.parametrize("role", ["employee", "hr", "finance", "it_security"])
def test_index_scope_is_the_verified_active_role(secured, role):
    token = secured.sign(roles=["employee", "hr", "finance", "it_security"])
    context = secured.authenticate(token, role)
    scope = secured.index.select_role_index(context)
    assert isinstance(scope, secured.index.RoleIndexScope)
    assert asdict(scope) == {"role": role}
    assert secured.index.select_role_index(context) == scope
    with pytest.raises(FrozenInstanceError):
        scope.role = "global"


@pytest.mark.parametrize("grant, target", [("employee", "hr"), ("hr", "it_security")])
def test_namespace_cannot_be_overridden_without_a_verified_grant(secured, grant, target):
    token = secured.sign(roles=[grant])
    assert secured.request(token, target).status == 403
    context = secured.authenticate(token, grant)
    with pytest.raises(TypeError):
        secured.index.select_role_index(context, role=target)
    assert secured.index.select_role_index(context).role == grant

    both = secured.sign(roles=[grant, target])
    assert secured.index.select_role_index(secured.authenticate(both, grant)).role == grant
    assert secured.index.select_role_index(secured.authenticate(both, target)).role == target


def test_unverified_context_never_falls_back_to_a_global_scope(secured):
    valid = secured.authenticate(secured.sign(), "employee")
    assert secured.index.select_role_index(valid).role == "employee"
    for invalid in (None, "global", SimpleNamespace(active_role="hr"), secured.context.AuthenticatedContext()):
        with pytest.raises(secured.index.IndexSelectionError) as raised:
            secured.index.select_role_index(invalid)
        assert str(raised.value) == raised.value.code == "UNVERIFIED_CONTEXT"


def test_scope_selection_has_no_index_io_or_fabricated_snapshot_data(secured, monkeypatch):
    context = secured.authenticate(secured.sign(), "employee")

    def unexpected_io(*args, **kwargs):
        pytest.fail("Scope selection must be independent of filesystem, databases, and the network")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", unexpected_io)
        patch.setattr(Path, "open", unexpected_io)
        patch.setattr(sqlite3, "connect", unexpected_io)
        patch.setattr(socket, "create_connection", unexpected_io)
        scope = secured.index.select_role_index(context)
    assert asdict(scope) == {"role": "employee"}
