from copy import copy
from types import SimpleNamespace

import pytest

from .test_auth import secured


@pytest.mark.parametrize("classification", ["PUBLIC", "INTERNAL", "RESTRICTED"])
def test_each_classification_requires_an_explicit_active_role(secured, classification):
    employee = secured.authenticate(secured.sign(), "employee")
    assert secured.acl.can_read_document(employee, classification=classification, allowed_roles=["employee"])
    assert not secured.acl.can_read_document(employee, classification=classification, allowed_roles=["hr"])
    assert not secured.acl.can_read_document(employee, classification=classification)
    assert not secured.acl.can_read_document(None, classification=classification, allowed_roles=["employee"])


def test_hr_policy_uses_only_the_active_role(secured):
    token = secured.sign(roles=["employee", "hr"])
    employee = secured.authenticate(token, "employee")
    hr = secured.authenticate(token, "hr")
    policy = {"classification": "RESTRICTED", "allowed_roles": ("hr",)}
    assert employee.grants == hr.grants == ("employee", "hr")
    assert not secured.acl.can_read_document(employee, **policy)
    assert secured.acl.can_read_document(hr, **policy)


@pytest.mark.parametrize("role", ["employee", "hr", "it_security"])
def test_it_only_policy_permits_only_it_security(secured, role):
    token = secured.sign(roles=["employee", "hr", "it_security"])
    context = secured.authenticate(token, role)
    allowed = secured.acl.can_read_document(context, classification="RESTRICTED", allowed_roles=("it_security",))
    assert allowed is (role == "it_security")


@pytest.mark.parametrize("classification", [None, "", "SECRET", "public", 1, ["PUBLIC"], {"scope": "PUBLIC"}])
def test_invalid_classification_denies_even_with_a_matching_grant(secured, classification):
    context = secured.authenticate(secured.sign(), "employee")
    assert not secured.acl.can_read_document(context, classification=classification, allowed_roles=["employee"])
    assert not secured.acl.can_read_document(context, allowed_roles=["employee"])


@pytest.mark.parametrize("grants", [
    None, [], (), "employee", {"employee"}, {"employee": True}, ["employee", "unknown"],
    ["employee", "employee"], ["employee", "ingestion_admin"], ["employee", ["hr"]],
    ["employee", True], [None, "employee"],
])
def test_invalid_acl_denies_even_if_it_mentions_the_active_role(secured, grants):
    context = secured.authenticate(secured.sign(), "employee")
    assert not secured.acl.can_read_document(context, classification="PUBLIC", allowed_roles=grants)


def test_forged_or_copied_context_cannot_authorize_any_document(secured):
    token = secured.sign(roles=["employee", "hr"])
    employee = secured.authenticate(token, "employee")
    metadata = {"classification": "PUBLIC", "allowed_roles": ("employee", "hr")}
    assert secured.acl.can_read_document(employee, **metadata)

    forged = (
        SimpleNamespace(subject=employee.subject, grants=employee.grants, active_role="hr", capabilities=()),
        secured.verifier.verify(token),
        secured.context.AuthenticatedContext(),
        copy(employee),
    )
    for context in forged:
        assert not secured.acl.can_read_document(context, **metadata)
