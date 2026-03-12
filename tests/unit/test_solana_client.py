"""Unit tests for SolanaLLMClient."""

import pytest
import os
from blockrun_llm.solana_client import SolanaLLMClient

TEST_BS58_KEY = (
    "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviQeRjpzKCY8trDwpvBMTKTpNFbCJsBZthJ4tCs6o62rr"
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

    def test_raises_without_key(self):
        saved = os.environ.pop("SOLANA_WALLET_KEY", None)
        with pytest.raises(ValueError, match="[Pp]rivate key required"):
            SolanaLLMClient()
        if saved:
            os.environ["SOLANA_WALLET_KEY"] = saved

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
