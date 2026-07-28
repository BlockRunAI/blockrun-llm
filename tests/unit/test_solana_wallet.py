"""Unit tests for Solana wallet utilities."""

import pytest

from blockrun_llm.solana_wallet import (
    create_solana_wallet,
    get_solana_public_key,
    solana_key_to_bytes,
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

    def test_accepts_solana_cli_json_array(self):
        import json

        canonical = solana_key_to_bytes(TEST_BS58_KEY)
        as_json = json.dumps(list(canonical))
        assert solana_key_to_bytes(as_json) == canonical

    def test_accepts_64_byte_hex_with_or_without_0x(self):
        canonical = solana_key_to_bytes(TEST_BS58_KEY)
        hex_key = canonical.hex()
        assert solana_key_to_bytes(hex_key) == canonical
        assert solana_key_to_bytes("0x" + hex_key) == canonical

    def test_evm_key_gets_explicit_hint(self):
        evm_key = "0x" + "ab" * 32  # 32-byte hex — Base/EVM format
        with pytest.raises(ValueError, match="EVM"):
            solana_key_to_bytes(evm_key)

    def test_unrecognized_input_lists_accepted_formats(self):
        with pytest.raises(ValueError, match="base58"):
            solana_key_to_bytes("not/a/key!!")


class TestGetOrCreateSolanaWalletErrorSources:
    def test_bad_env_key_names_the_env_var(self, monkeypatch, tmp_path):
        from blockrun_llm import solana_wallet

        monkeypatch.setattr(solana_wallet, "SOLANA_WALLET_FILE", tmp_path / ".solana-session")
        monkeypatch.setenv("SOLANA_WALLET_KEY", "not/a/key!!")
        with pytest.raises(ValueError, match="SOLANA_WALLET_KEY"):
            solana_wallet.get_or_create_solana_wallet()

    def test_bad_session_file_names_the_path(self, monkeypatch, tmp_path):
        from blockrun_llm import solana_wallet

        session = tmp_path / ".solana-session"
        session.write_text("not/a/key!!")
        monkeypatch.setattr(solana_wallet, "SOLANA_WALLET_FILE", session)
        monkeypatch.delenv("SOLANA_WALLET_KEY", raising=False)
        with pytest.raises(ValueError, match="solana-session"):
            solana_wallet.get_or_create_solana_wallet()

    def test_adopts_session_file_in_json_array_format(self, monkeypatch, tmp_path):
        import json

        from blockrun_llm import solana_wallet

        wallet = create_solana_wallet()
        canonical = solana_key_to_bytes(wallet["private_key"])
        session = tmp_path / ".solana-session"
        session.write_text(json.dumps(list(canonical)))
        monkeypatch.setattr(solana_wallet, "SOLANA_WALLET_FILE", session)
        monkeypatch.delenv("SOLANA_WALLET_KEY", raising=False)
        result = solana_wallet.get_or_create_solana_wallet()
        assert result["address"] == wallet["address"]
        assert result["is_new"] is False


class TestGetSolanaPublicKey:
    def test_returns_base58_address(self):
        addr = get_solana_public_key(TEST_BS58_KEY)
        assert isinstance(addr, str)
        assert len(addr) >= 32
        # Should be valid base58 (only alphanumeric, no 0/O/I/l)
        import re

        assert re.match(r"^[1-9A-HJ-NP-Za-km-z]+$", addr)
