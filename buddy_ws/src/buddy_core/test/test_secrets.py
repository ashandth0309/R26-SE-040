import pytest

from buddy_core.secrets import (
    BuddySecretError,
    get_secret,
)


def test_optional_secret_missing(monkeypatch):
    monkeypatch.delenv(
        "BUDDY_TASK06_TEST_SECRET",
        raising=False,
    )

    assert get_secret(
        "BUDDY_TASK06_TEST_SECRET",
        required=False,
    ) is None


def test_required_secret_missing(monkeypatch):
    monkeypatch.delenv(
        "BUDDY_TASK06_TEST_SECRET",
        raising=False,
    )

    with pytest.raises(BuddySecretError):
        get_secret(
            "BUDDY_TASK06_TEST_SECRET",
            required=True,
        )


def test_required_secret_present(monkeypatch):
    monkeypatch.setenv(
        "BUDDY_TASK06_TEST_SECRET",
        "temporary-test-value",
    )

    value = get_secret(
        "BUDDY_TASK06_TEST_SECRET",
        required=True,
    )

    assert value == "temporary-test-value"
