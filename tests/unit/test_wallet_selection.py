"""Regression tests for canonical wallet selection."""

from blockrun_llm import solana_wallet, wallet


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
