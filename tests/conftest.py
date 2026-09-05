"""Test-wide fixtures.

``BLOCKRUN_API_KEY`` is cleared for every test. Without this, every test that
asserts "no credential configured" — and every test that expects a wallet
client — fails on the machine of anyone who actually has a key exported, which
is every developer working on the API-key rail. Tests that want the variable
set it explicitly with ``monkeypatch.setenv``, which overrides this.
"""

import pytest

from blockrun_llm.apikey import ENV_API_KEY, ENV_API_KEY_URL


@pytest.fixture(autouse=True)
def _clear_api_key_env(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv(ENV_API_KEY_URL, raising=False)
