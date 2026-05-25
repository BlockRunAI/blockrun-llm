"""Unit tests for RealFaceClient input validation."""

import os
import pytest

from blockrun_llm import RealFaceClient


@pytest.fixture
def client():
    # Deterministic dummy key — never actually signs against a live endpoint
    # in unit tests; we only exercise local validation paths.
    os.environ.setdefault("BLOCKRUN_WALLET_KEY", "0x" + "11" * 32)
    return RealFaceClient()


# --- init() validation ------------------------------------------------------


def test_init_rejects_empty_name(client):
    with pytest.raises(ValueError, match="name is required"):
        client.init(name="")


def test_init_rejects_whitespace_name(client):
    with pytest.raises(ValueError, match="name is required"):
        client.init(name="   ")


def test_init_rejects_long_name(client):
    with pytest.raises(ValueError, match="64 chars or fewer"):
        client.init(name="a" * 65)


def test_init_rejects_bad_group_id(client):
    with pytest.raises(ValueError, match="legacy_rf_"):
        client.init(name="ok", group_id="rf_123")


# --- status() / wait_for_active() validation --------------------------------


def test_status_rejects_bad_group_id(client):
    with pytest.raises(ValueError, match="legacy_rf_"):
        client.status(group_id="not-a-group")


def test_status_rejects_empty_group_id(client):
    with pytest.raises(ValueError, match="legacy_rf_"):
        client.status(group_id="")


def test_wait_for_active_rejects_bad_group_id(client):
    with pytest.raises(ValueError, match="legacy_rf_"):
        client.wait_for_active(group_id="nope")


def test_wait_for_active_rejects_nonpositive_interval(client):
    with pytest.raises(ValueError, match="poll_interval_seconds must be positive"):
        client.wait_for_active(group_id="legacy_rf_1", poll_interval_seconds=0)


# --- enroll() validation ----------------------------------------------------


def test_enroll_rejects_empty_name(client):
    with pytest.raises(ValueError, match="name is required"):
        client.enroll(name="", image_url="https://example.com/x.jpg", group_id="legacy_rf_1")


def test_enroll_rejects_long_name(client):
    with pytest.raises(ValueError, match="64 chars or fewer"):
        client.enroll(name="a" * 65, image_url="https://example.com/x.jpg", group_id="legacy_rf_1")


def test_enroll_rejects_non_http_url(client):
    with pytest.raises(ValueError, match="image_url must be an http"):
        client.enroll(name="ok", image_url="ftp://example.com/x.jpg", group_id="legacy_rf_1")


def test_enroll_rejects_empty_url(client):
    with pytest.raises(ValueError, match="image_url must be an http"):
        client.enroll(name="ok", image_url="", group_id="legacy_rf_1")


def test_enroll_rejects_bad_group_id(client):
    with pytest.raises(ValueError, match="legacy_rf_"):
        client.enroll(name="ok", image_url="https://example.com/x.jpg", group_id="bad")


# --- utilities --------------------------------------------------------------


def test_get_wallet_address(client):
    addr = client.get_wallet_address()
    assert addr.startswith("0x")
    assert len(addr) == 42
