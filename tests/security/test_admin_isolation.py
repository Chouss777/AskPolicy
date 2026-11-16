import pytest

from .test_auth import secured


@pytest.mark.parametrize("active_role", [None, "employee", "hr", "finance", "it_security", "ingestion_admin"])
def test_admin_only_token_cannot_activate_a_read_role(secured, active_role):
    token = secured.sign(roles=[], capabilities=["ingestion_admin"])
    identity = secured.verifier.verify(token)
    assert identity.grants == ()
    assert identity.capabilities == ("ingestion_admin",)
    response = secured.request(token, active_role)
    assert response.status == 403
    assert response.json() == {"detail": "Access denied"}
    assert not secured.acl.can_read_document(identity, classification="PUBLIC", allowed_roles=["employee"])


def test_admin_with_employee_grant_has_only_employee_privileges(secured):
    token = secured.sign(roles=["employee"], capabilities=["ingestion_admin"])
    response = secured.request(token, "employee")
    assert response.status == 200
    assert response.json()["active_role"] == "employee"
    assert response.json()["grants"] == ["employee"]
    assert response.json()["capabilities"] == ["ingestion_admin"]
    assert response.json()["hr_allowed"] is False
    assert response.json()["index_role"] == "employee"

    employee = secured.authenticate(token, "employee")
    assert secured.acl.can_read_document(employee, classification="PUBLIC", allowed_roles=["employee"])
    assert not secured.acl.can_read_document(employee, classification="RESTRICTED", allowed_roles=["it_security"])
    assert secured.request(token, "hr").status == 403
    assert secured.request(token, "ingestion_admin").status == 403
