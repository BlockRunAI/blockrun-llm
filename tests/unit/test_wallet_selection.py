"""Regression tests for canonical wallet selection."""

import os
from pathlib import Path

import pytest
from eth_account import Account

from blockrun_llm import solana_wallet, wallet

PROVIDER_KEY = "0x" + "2" * 64
CANONICAL_KEY = "0x" + "1" * 64
LEGACY_KEY = "0x" + "3" * 64


def _fake_home(monkeypatch, home: Path) -> None:
    """Point Path.home() at a temp dir so scan_wallets() reads real fixtures."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


def _write_provider_wallet(home: Path, filename: str = "wallet.json") -> Path:
    """Create a provider wallet whose "address" field is a lie."""
    provider_dir = home / ".agentcash"
    provider_dir.mkdir(exist_ok=True)
    target = provider_dir / filename
    target.write_text('{"privateKey":"' + PROVIDER_KEY + '","address":"0xNotTheRealAddress"}')
    return target


def _blockrun_dir(home: Path) -> Path:
    blockrun_dir = home / ".blockrun"
    blockrun_dir.mkdir(exist_ok=True)
    return blockrun_dir


def test_load_wallet_prefers_blockrun_session_over_provider_wallet(monkeypatch, tmp_path):
    """A newer wallet.json from another app must not replace BlockRun's wallet."""
    blockrun_dir = tmp_path / ".blockrun"
    blockrun_dir.mkdir()
    canonical_file = blockrun_dir / ".session"
    canonical_key = "0x" + "1" * 64
    canonical_file.write_text(canonical_key)

    provider_dir = tmp_path / ".agentcash"
    provider_dir.mkdir()
    (provider_dir / "wallet.json").write_text(
        '{"privateKey":"0x' + "2" * 64 + '","address":"0xprovider"}'
    )

    monkeypatch.setattr(wallet, "WALLET_DIR", blockrun_dir)
    monkeypatch.setattr(wallet, "WALLET_FILE", canonical_file)
    monkeypatch.setattr(
        wallet,
        "scan_wallets",
        lambda: (_ for _ in ()).throw(AssertionError("automatic scan must not run")),
    )

    assert wallet.load_wallet() == canonical_key


def test_load_solana_wallet_prefers_blockrun_session_over_provider_wallet(monkeypatch, tmp_path):
    """A provider Solana wallet must not replace BlockRun's active wallet."""
    blockrun_dir = tmp_path / ".blockrun"
    blockrun_dir.mkdir()
    canonical_file = blockrun_dir / ".solana-session"
    canonical_key = "canonical-solana-key"
    canonical_file.write_text(canonical_key)

    monkeypatch.setattr(solana_wallet, "WALLET_DIR", blockrun_dir)
    monkeypatch.setattr(solana_wallet, "SOLANA_WALLET_FILE", canonical_file)
    monkeypatch.setattr(
        solana_wallet,
        "scan_solana_wallets",
        lambda: (_ for _ in ()).throw(AssertionError("automatic scan must not run")),
    )

    assert solana_wallet.load_solana_wallet() == canonical_key


def test_get_or_create_wallet_does_not_adopt_provider_wallet(monkeypatch, tmp_path):
    """get_or_create_wallet() is what callers hit — it must not scan either.

    load_wallet() and get_or_create_wallet() carry separate resolution logic,
    so covering only load_wallet() would let the bug return through the door
    users actually walk through.
    """
    monkeypatch.delenv("BLOCKRUN_WALLET_KEY", raising=False)
    monkeypatch.delenv("BASE_CHAIN_WALLET_KEY", raising=False)
    _fake_home(monkeypatch, tmp_path)
    blockrun_dir = _blockrun_dir(tmp_path)
    _write_provider_wallet(tmp_path)

    monkeypatch.setattr(wallet, "WALLET_DIR", blockrun_dir)
    monkeypatch.setattr(wallet, "WALLET_FILE", blockrun_dir / ".session")

    # The provider wallet is genuinely on disk and discoverable...
    assert len(wallet.scan_wallets()) == 1

    address, key, is_new = wallet.get_or_create_wallet()

    # ...but a brand new wallet is minted instead of adopting it.
    assert is_new is True
    assert key != PROVIDER_KEY
    assert address != Account.from_key(PROVIDER_KEY).address


def test_get_or_create_wallet_honours_legacy_wallet_key(monkeypatch, tmp_path):
    """A funded ~/.blockrun/wallet.key must not be replaced by a new wallet.

    The TypeScript SDK resolves this file via loadWallet(); Python must match,
    otherwise a legacy user silently loses access to their funds.
    """
    monkeypatch.delenv("BLOCKRUN_WALLET_KEY", raising=False)
    monkeypatch.delenv("BASE_CHAIN_WALLET_KEY", raising=False)
    _fake_home(monkeypatch, tmp_path)
    blockrun_dir = _blockrun_dir(tmp_path)
    (blockrun_dir / "wallet.key").write_text(LEGACY_KEY)

    monkeypatch.setattr(wallet, "WALLET_DIR", blockrun_dir)
    monkeypatch.setattr(wallet, "WALLET_FILE", blockrun_dir / ".session")

    address, key, is_new = wallet.get_or_create_wallet()

    assert is_new is False
    assert key == LEGACY_KEY
    assert address == Account.from_key(LEGACY_KEY).address


def test_provider_wallet_cannot_hijack_even_when_written_last(monkeypatch, tmp_path):
    """The core invariant: another app must never take over the active wallet.

    Discovery sorts by modification time, so the hijack is simply "write a
    wallet.json last". Here the provider wallet is the newest file on disk and
    claims a plausible address; BlockRun's own session must still win.
    """
    monkeypatch.delenv("BLOCKRUN_WALLET_KEY", raising=False)
    monkeypatch.delenv("BASE_CHAIN_WALLET_KEY", raising=False)
    _fake_home(monkeypatch, tmp_path)
    blockrun_dir = _blockrun_dir(tmp_path)
    (blockrun_dir / ".session").write_text(CANONICAL_KEY)

    monkeypatch.setattr(wallet, "WALLET_DIR", blockrun_dir)
    monkeypatch.setattr(wallet, "WALLET_FILE", blockrun_dir / ".session")

    # Provider writes after us, from two different directories.
    provider = _write_provider_wallet(tmp_path)
    second_dir = tmp_path / ".someotherprovider"
    second_dir.mkdir()
    (second_dir / "wallet.json").write_text(
        '{"privateKey":"' + LEGACY_KEY + '","address":"0xAlsoNotOurs"}'
    )
    os.utime(provider, (2**31, 2**31))

    assert len(wallet.scan_wallets()) == 2

    address, key, is_new = wallet.get_or_create_wallet()

    assert key == CANONICAL_KEY
    assert address == Account.from_key(CANONICAL_KEY).address
    assert is_new is False


def test_migration_notice_derives_address_from_key_not_file_claim(monkeypatch, tmp_path):
    """The notice must name the address the discovered key actually controls."""
    _fake_home(monkeypatch, tmp_path)
    _blockrun_dir(tmp_path)
    _write_provider_wallet(tmp_path)

    new_address = Account.from_key(CANONICAL_KEY).address
    notice = wallet.format_wallet_migration_notice(new_address)

    assert notice is not None
    # The real address derived from the key, never the file's bogus claim.
    assert Account.from_key(PROVIDER_KEY).address in notice
    assert "0xNotTheRealAddress" not in notice
    assert new_address in notice
    # Never leak the discovered private key.
    assert PROVIDER_KEY not in notice


def test_migration_notice_is_silent_when_nothing_discovered(monkeypatch, tmp_path):
    """No provider wallets means no scary notice."""
    _fake_home(monkeypatch, tmp_path)
    _blockrun_dir(tmp_path)

    assert wallet.format_wallet_migration_notice("0xabc") is None


def test_solana_migration_notice_lists_discovered_wallets(monkeypatch, tmp_path):
    """Solana counterpart surfaces discovered wallets the same way."""
    # CI runs this file on Python 3.9, where the solana extra is not installed.
    pytest.importorskip("solders")
    _fake_home(monkeypatch, tmp_path)
    _blockrun_dir(tmp_path)
    provider_dir = tmp_path / ".agentcash"
    provider_dir.mkdir(exist_ok=True)

    discovered = solana_wallet.create_solana_wallet()
    (provider_dir / "solana-wallet.json").write_text(
        '{"privateKey":"' + discovered["private_key"] + '","address":"NotTheRealAddress"}'
    )

    notice = solana_wallet.format_solana_wallet_migration_notice("NewWalletAddress")

    assert notice is not None
    assert discovered["address"] in notice
    assert "NotTheRealAddress" not in notice
    assert discovered["private_key"] not in notice


def test_solana_migration_notice_is_silent_when_nothing_discovered(monkeypatch, tmp_path):
    """No provider Solana wallets means no notice."""
    pytest.importorskip("solders")
    _fake_home(monkeypatch, tmp_path)
    _blockrun_dir(tmp_path)

    assert solana_wallet.format_solana_wallet_migration_notice("NewWalletAddress") is None
