"""
BlockRun Solana Wallet Management.

Stores keys as bs58-encoded strings at ~/.blockrun/.solana-session.
Requires: solders>=0.21.0, base58>=2.1.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .solana_client import SolanaLLMClient

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
        return {
            "private_key": file_key,
            "address": get_solana_public_key(file_key),
            "is_new": False,
        }

    wallet = create_solana_wallet()
    save_solana_wallet(wallet["private_key"])
    return {**wallet, "is_new": True}


def setup_agent_solana_wallet(silent: bool = False) -> "SolanaLLMClient":
    """
    Set up Solana wallet for agent use and return a SolanaLLMClient.

    This is the entry point for Claude Code skills and other agent runtimes.
    It auto-creates a Solana wallet if needed and prints address if new.

    Args:
        silent: If True, don't print welcome message (default: False)

    Returns:
        Configured SolanaLLMClient ready for use

    Example:
        from blockrun_llm import setup_agent_solana_wallet

        client = setup_agent_solana_wallet()
        response = client.chat("openai/gpt-5.2", "Hello!")
    """
    import sys

    result = get_or_create_solana_wallet()

    if result["is_new"] and not silent:
        print(f"New Solana wallet created: {result['address']}", file=sys.stderr)

    from .solana_client import SolanaLLMClient

    return SolanaLLMClient(private_key=result["private_key"])


def get_solana_usdc_balance(address: str, rpc_url: Optional[str] = None) -> float:
    """
    Get USDC-SPL balance for a Solana address.

    Args:
        address: Solana wallet address (base58)
        rpc_url: Solana RPC endpoint (default: mainnet-beta)

    Returns:
        USDC balance as float (6 decimals)
    """
    import httpx

    rpc = rpc_url or "https://api.mainnet-beta.solana.com"
    # USDC mint on Solana mainnet
    usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    try:
        resp = httpx.post(
            rpc,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    address,
                    {"mint": usdc_mint},
                    {"encoding": "jsonParsed"},
                ],
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        accounts = data.get("result", {}).get("value", [])
        if not accounts:
            return 0.0

        # Sum all USDC token accounts (usually just one)
        total = 0.0
        for acct in accounts:
            info = acct.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            token_amount = info.get("tokenAmount", {})
            total += float(token_amount.get("uiAmount", 0))
        return total

    except Exception:
        return 0.0


# QR code file paths for Solana
SOLANA_QR_FILE = WALLET_DIR / "solana_qr.png"
SOLANA_QR_ASCII_FILE = WALLET_DIR / "solana_qr.txt"


def generate_solana_qr_ascii(address: str) -> str:
    """
    Generate ASCII QR code for Solana wallet funding.
    Uses solana: URI scheme. Caches to ~/.blockrun/solana_qr.txt.

    Args:
        address: Solana wallet address (base58)

    Returns:
        ASCII art QR code string
    """
    solana_uri = f"solana:{address}"
    cache_key = f"v1:{solana_uri}"

    # Try cache
    if SOLANA_QR_ASCII_FILE.exists():
        try:
            cached = SOLANA_QR_ASCII_FILE.read_text()
            lines = cached.split("\n", 1)
            if len(lines) == 2 and lines[0] == cache_key:
                return lines[1]
        except Exception:
            pass

    # Generate new QR
    try:
        import qrcode
        from io import StringIO

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=1,
        )
        qr.add_data(solana_uri)
        qr.make(fit=True)

        f = StringIO()
        qr.print_ascii(out=f, invert=True)
        qr_ascii = f.getvalue()

        # Cache
        try:
            WALLET_DIR.mkdir(exist_ok=True)
            SOLANA_QR_ASCII_FILE.write_text(f"{cache_key}\n{qr_ascii}")
        except Exception:
            pass

        return qr_ascii

    except ImportError:
        return f"[QR code requires 'qrcode' package: pip install qrcode[pil]]\nAddress: {address}"


def save_solana_wallet_qr(address: str, path: Optional[str] = None) -> str:
    """
    Save Solana QR code as PNG image.

    Args:
        address: Solana wallet address (base58)
        path: Optional custom path (default: ~/.blockrun/solana_qr.png)

    Returns:
        Path to saved QR image, or empty string on failure
    """
    try:
        import qrcode

        solana_uri = f"solana:{address}"

        qr = qrcode.QRCode(
            version=4,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(solana_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        save_path = Path(path) if path else SOLANA_QR_FILE
        save_path.parent.mkdir(exist_ok=True)
        img.save(str(save_path))

        return str(save_path)

    except ImportError:
        return ""


def open_solana_wallet_qr(address: str) -> str:
    """
    Generate Solana QR code and open it in the default image viewer.

    Args:
        address: Solana wallet address (base58)

    Returns:
        Path to saved QR image
    """
    import subprocess
    import platform

    qr_path = save_solana_wallet_qr(address)
    if qr_path:
        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", qr_path], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["start", qr_path], shell=True, check=True)
            else:
                subprocess.run(["xdg-open", qr_path], check=True)
        except Exception:
            pass
    return qr_path
