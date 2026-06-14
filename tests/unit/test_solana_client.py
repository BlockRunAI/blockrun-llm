"""Unit tests for SolanaLLMClient."""

import pytest
import os
from blockrun_llm.solana_client import SolanaLLMClient

TEST_BS58_KEY = (
    "433C7KFcM4y1ZEVdZYSH7wheSNAM384UcbgXEyD5FV7Q2HsQ1BwjEDx4GbBZUqPkZTVhFPyLyuZnzK8wCeAkU7wG"
)


class TestSolanaLLMClientInit:
    def test_init_with_key(self):
        client = SolanaLLMClient(private_key=TEST_BS58_KEY)
        assert client is not None

    def test_init_from_env(self):
        os.environ["SOLANA_WALLET_KEY"] = TEST_BS58_KEY
        client = SolanaLLMClient()
        assert client is not None
        del os.environ["SOLANA_WALLET_KEY"]

    def test_raises_without_key(self, monkeypatch):
        # No env var AND no wallet session on disk → must still raise. Patch the
        # session loader so the test is deterministic regardless of whether the
        # machine running it happens to have ~/.blockrun/.solana-session.
        monkeypatch.delenv("SOLANA_WALLET_KEY", raising=False)
        monkeypatch.setattr("blockrun_llm.solana_wallet.load_solana_wallet", lambda: None)
        with pytest.raises(ValueError, match="[Pp]rivate key required"):
            SolanaLLMClient()

    def test_init_from_session_file(self, monkeypatch):
        # No env var, but a wallet session exists on disk → auto-load it (parity
        # with the Base LLMClient, which already falls back to load_wallet()).
        monkeypatch.delenv("SOLANA_WALLET_KEY", raising=False)
        monkeypatch.setattr("blockrun_llm.solana_wallet.load_solana_wallet", lambda: TEST_BS58_KEY)
        client = SolanaLLMClient()
        assert client is not None

    def test_default_api_url(self):
        client = SolanaLLMClient(private_key=TEST_BS58_KEY)
        assert client.is_solana()

    def test_custom_api_url(self):
        client = SolanaLLMClient(
            private_key=TEST_BS58_KEY, api_url="https://custom.example.com/api"
        )
        assert not client.is_solana()

    def test_get_wallet_address(self):
        client = SolanaLLMClient(private_key=TEST_BS58_KEY)
        addr = client.get_wallet_address()
        assert isinstance(addr, str)
        assert len(addr) >= 32

    def test_get_spending_initial(self):
        client = SolanaLLMClient(private_key=TEST_BS58_KEY)
        spending = client.get_spending()
        assert spending["total_usd"] == 0.0
        assert spending["calls"] == 0
