"""
BlockRun Solana Wallet Management.

Stores keys as bs58-encoded strings at ~/.blockrun/.solana-session.
Requires: solders>=0.21.0, base58>=2.1.0
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

WALLET_DIR = Path.home() / ".blockrun"
SOLANA_WALLET_FILE = WALLET_DIR / ".solana-session"


def _require_solders() -> None:
    try:
        import solders  # noqa: F401
    except ImportError:
        raise ImportError(
            "Solana support requires 'solders' and 'base58' packages. "
            "Install with: pip install blockrun-llm[solana]"
        )


def create_solana_wallet() -> Dict[str, str]:
    """
    Create a new Solana wallet.

    Returns:
        Dict with 'address' (base58 pubkey) and 'private_key' (bs58 secret key)
    """
    _require_solders()
    from solders.keypair import Keypair  # type: ignore
    import base58  # type: ignore

    kp = Keypair()
    secret = bytes(kp)  # 64 bytes
    return {
        "address": str(kp.pubkey()),
        "private_key": base58.b58encode(secret).decode(),
    }


def solana_key_to_bytes(private_key: str) -> bytes:
    """
    Convert a bs58 private key string to bytes (64 bytes).

    Args:
        private_key: bs58-encoded 64-byte Solana secret key

    Returns:
        64-byte secret key as bytes

    Raises:
        ValueError: If key is invalid
    """
    try:
        import base58  # type: ignore
        decoded = base58.b58decode(private_key)
        if len(decoded) != 64:
            raise ValueError(f"Expected 64 bytes, got {len(decoded)}")
        return decoded
    except Exception as e:
        raise ValueError(f"Invalid Solana private key: {e}") from e


def get_solana_public_key(private_key: str) -> str:
    """
    Get the Solana public key (address) from a bs58 private key.

    Args:
        private_key: bs58-encoded 64-byte Solana secret key

    Returns:
        Base58 public key string
    """
    _require_solders()
    from solders.keypair import Keypair  # type: ignore

    secret = solana_key_to_bytes(private_key)
    kp = Keypair.from_seed(secret[:32])
    return str(kp.pubkey())


def save_solana_wallet(private_key: str) -> Path:
    WALLET_DIR.mkdir(exist_ok=True)
    SOLANA_WALLET_FILE.write_text(private_key)
    SOLANA_WALLET_FILE.chmod(0o600)
    return SOLANA_WALLET_FILE


def load_solana_wallet() -> Optional[str]:
    if SOLANA_WALLET_FILE.exists():
        key = SOLANA_WALLET_FILE.read_text().strip()
        if key:
            return key
    return None


def get_or_create_solana_wallet() -> Dict[str, object]:
    """
    Get existing Solana wallet or create new one.

    Priority: SOLANA_WALLET_KEY env var → ~/.blockrun/.solana-session → create new

    Returns:
        Dict with 'address', 'private_key', 'is_new'
    """
    env_key = os.environ.get("SOLANA_WALLET_KEY")
    if env_key:
        return {"private_key": env_key, "address": get_solana_public_key(env_key), "is_new": False}

    file_key = load_solana_wallet()
    if file_key:
        return {"private_key": file_key, "address": get_solana_public_key(file_key), "is_new": False}

    wallet = create_solana_wallet()
    save_solana_wallet(wallet["private_key"])
    return {**wallet, "is_new": True}
