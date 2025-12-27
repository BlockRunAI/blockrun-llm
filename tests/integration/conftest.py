"""Pytest configuration for integration tests."""

import os
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: Integration tests requiring funded wallet and API access"
    )


@pytest.fixture(scope="session")
def wallet_private_key():
    """Get wallet private key from environment variable.

    Returns None if not set, which will cause integration tests to be skipped.
    """
    return os.environ.get("BLOCKRUN_WALLET_KEY")


@pytest.fixture(scope="session")
def production_api_url():
    """Get production API URL."""
    return "https://api.blockrun.ai"
