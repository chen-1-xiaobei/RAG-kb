from auth import Principal


def test_public_is_always_allowed():
    principal = Principal("u-1", frozenset())
    assert principal.allowed_groups == {"public"}


def test_server_principal_controls_groups():
    principal = Principal("u-2", frozenset({"财务"}))
    assert principal.allowed_groups == {"public", "财务"}
    assert "人事" not in principal.allowed_groups

