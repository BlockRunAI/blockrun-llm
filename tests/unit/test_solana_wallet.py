"""Unit tests for Solana wallet utilities."""

import pytest
from blockrun_llm.solana_wallet import (
    create_solana_wallet,
    solana_key_to_bytes,
    get_solana_public_key,
)

# A valid test bs58 secret key (64 bytes, valid keypair from deterministic seed)
TEST_BS58_KEY = (
    "433C7KFcM4y1ZEVdZYSH7wheSNAM384UcbgXEyD5FV7Q2HsQ1BwjEDx4GbBZUqPkZTVhFPyLyuZnzK8wCeAkU7wG"
)


class TestCreateSolanaWallet:
    def test_returns_address_and_key(self):
        wallet = create_solana_wallet()
        assert "address" in wallet
        assert "private_key" in wallet
        assert len(wallet["address"]) >= 32  # base58 pubkey
        assert len(wallet["private_key"]) >= 86  # bs58 64-byte key

    def test_unique_wallets(self):
        w1 = create_solana_wallet()
        w2 = create_solana_wallet()
        assert w1["address"] != w2["address"]
        assert w1["private_key"] != w2["private_key"]


class TestSolanaKeyToBytes:
    def test_valid_key(self):
        b = solana_key_to_bytes(TEST_BS58_KEY)
        assert isinstance(b, bytes)
        assert len(b) == 64

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Invalid Solana private key"):
            solana_key_to_bytes("not-a-valid-key!!!")


class TestGetSolanaPublicKey:
    def test_returns_base58_address(self):
        addr = get_solana_public_key(TEST_BS58_KEY)
        assert isinstance(addr, str)
        assert len(addr) >= 32
        # Should be valid base58 (only alphanumeric, no 0/O/I/l)
        import re

        assert re.match(r"^[1-9A-HJ-NP-Za-km-z]+$", addr)
