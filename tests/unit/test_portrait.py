"""Unit tests for PortraitClient input validation."""

import os

import pytest

from blockrun_llm import PortraitClient


@pytest.fixture
def client():
    # Deterministic dummy key — never actually signs against a live endpoint
    # in unit tests; we only exercise local validation paths.
    os.environ.setdefault("BLOCKRUN_WALLET_KEY", "0x" + "11" * 32)
    return PortraitClient()


def test_enroll_rejects_empty_name(client):
    with pytest.raises(ValueError, match="name is required"):
        client.enroll(name="", image_url="https://example.com/x.jpg")


def test_enroll_rejects_whitespace_name(client):
    with pytest.raises(ValueError, match="name is required"):
        client.enroll(name="   ", image_url="https://example.com/x.jpg")


def test_enroll_rejects_long_name(client):
    long_name = "a" * 65
    with pytest.raises(ValueError, match="64 chars or fewer"):
        client.enroll(name=long_name, image_url="https://example.com/x.jpg")


def test_enroll_rejects_non_http_url(client):
    with pytest.raises(ValueError, match="image_url must be an http"):
        client.enroll(name="ok", image_url="ftp://example.com/x.jpg")


def test_enroll_rejects_empty_url(client):
    with pytest.raises(ValueError, match="image_url must be an http"):
        client.enroll(name="ok", image_url="")


def test_get_wallet_address(client):
    addr = client.get_wallet_address()
    assert addr.startswith("0x")
    assert len(addr) == 42
