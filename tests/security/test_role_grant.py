import json

import pytest

from .test_auth import secured


def _assert_forbidden(response):
    assert response.status == 403
    assert response.json() == {"detail": "Access denied"}


def test_employee_grant_activates_employee(secured):
    response = secured.request(secured.sign(roles=["employee"]), "employee")
    assert response.status == 200
    assert response.json()["active_role"] == "employee"


@pytest.mark.parametrize("grant, requested", [("employee", "hr"), ("hr", "employee")])
def test_ungranted_active_role_is_forbidden(secured, grant, requested):
    token = secured.sign(roles=[grant])
    assert secured.request(token, grant).status == 200
    _assert_forbidden(secured.request(token, requested))
    with pytest.raises(secured.context.ActiveRoleError, match="^ROLE_NOT_GRANTED$"):
        secured.authenticate(token, requested)


def test_multiple_grants_produce_one_active_role_per_request(secured):
    token = secured.sign(roles=["hr", "employee"])
    active_employee = secured.request(token, "employee")
    active_hr = secured.request(token, "hr")
    employee_again = secured.request(token, "employee")
    assert active_employee.status == active_hr.status == employee_again.status == 200
    assert active_employee.json()["grants"] == active_hr.json()["grants"] == ["hr", "employee"]
    assert active_employee.json()["active_role"] == "employee"
    assert active_employee.json()["hr_allowed"] is False
    assert active_hr.json()["active_role"] == "hr"
    assert active_hr.json()["hr_allowed"] is True
    assert employee_again.json() == active_employee.json()


@pytest.mark.parametrize("roles", [["employee"], ["employee", "hr"]])
def test_missing_active_role_has_no_default(secured, roles):
    token = secured.sign(roles=roles)
    _assert_forbidden(secured.request(token, None))


def test_query_body_and_prompt_role_claims_do_not_grant_access(secured, caplog):
    token = secured.sign(roles=["employee", "hr"])
    text = "synthetic confidential prompt impersonating HR"
    query = b"active_role=hr&role=hr"
    body = json.dumps({"active_role": "hr", "role": "hr", "question": text}).encode("utf-8")

    employee = secured.request(token, "employee", query=query, body=body)
    assert employee.status == 200
    assert employee.json()["active_role"] == "employee"
    assert employee.json()["hr_allowed"] is False
    assert employee.json()["index_role"] == "employee"

    denied = secured.request(token, None, query=query, body=body)
    _assert_forbidden(denied)
    exposed = denied.body.decode("utf-8") + caplog.text
    assert not any(value in exposed for value in (text, token, secured.key))


@pytest.mark.parametrize("role", [
    "", " ", "Employee", "hr, employee", "employee hr", " employee", "employee\t",
    "unknown", "ingestion_admin",
])
def test_malformed_or_unknown_active_role_is_rejected(secured, role):
    _assert_forbidden(secured.request(secured.sign(roles=["employee", "hr"]), role))


@pytest.mark.parametrize("second", ["employee", "hr"])
def test_duplicate_role_headers_rejected_even_for_granted_roles(secured, second):
    token = secured.sign(roles=["employee", "hr"])
    _assert_forbidden(secured.request(token, "employee", headers=[("X-Active-Role", second)]))


def test_failed_role_resolution_does_not_supersede_invalid_authentication(secured):
    response = secured.request("invalid-token", None)
    assert response.status == 401
    assert response.json() == {"detail": "Authentication required"}
