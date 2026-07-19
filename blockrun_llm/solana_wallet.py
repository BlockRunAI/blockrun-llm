"""
BlockRun Solana Wallet Management.

Stores keys as bs58-encoded strings at ~/.blockrun/.solana-session.
Requires: solders>=0.21.0, base58>=2.1.0
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

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

    kp = Keypair()
    return {
        "address": str(kp.pubkey()),
        "private_key": str(kp),  # bs58-encoded 64-byte keypair
    }


def solana_key_to_bytes(private_key: str) -> bytes:
    """
    Convert a bs58 private key string to bytes (64 bytes).

    Accepts both 64-byte full keypairs and 32-byte seeds (from agentcash
    and other providers). 32-byte seeds are automatically expanded.

    Args:
        private_key: bs58-encoded Solana secret key (32 or 64 bytes)

    Returns:
        64-byte secret key as bytes

    Raises:
        ValueError: If key is invalid
    """
    try:
        from solders.keypair import Keypair  # type: ignore

        try:
            kp = Keypair.from_base58_string(private_key)
            decoded = bytes(kp)
            if len(decoded) == 64:
                return decoded
        except Exception:
            pass

        # Fallback: try as 32-byte seed
        import base58 as b58

        decoded = b58.b58decode(private_key)
        if len(decoded) == 32:
            kp = Keypair.from_seed(decoded)
            return bytes(kp)
        elif len(decoded) == 64:
            kp = Keypair.from_seed(decoded[:32])
            return bytes(kp)

        raise ValueError(f"Expected 32 or 64 bytes, got {len(decoded)}")
    except Exception as e:
        # Wrap every failure — including the ``ValueError`` modern ``base58``
        # raises on invalid characters — in the documented message. A bare
        # ``except ValueError: raise`` here used to leak base58's raw
        # "Invalid character" error past the wrapper, breaking callers (and
        # the test) that match on "Invalid Solana private key".
        raise ValueError(f"Invalid Solana private key: {e}") from e


def get_solana_public_key(private_key: str) -> str:
    """
    Get the Solana public key (address) from a bs58 private key.

    Accepts both 64-byte full keypairs and 32-byte seeds.

    Args:
        private_key: bs58-encoded Solana secret key (32 or 64 bytes)

    Returns:
        Base58 public key string
    """
    _require_solders()
    from solders.keypair import Keypair  # type: ignore

    try:
        secret = solana_key_to_bytes(private_key)
        kp = Keypair.from_seed(secret[:32])
        return str(kp.pubkey())
    except ValueError:
        # 32-byte seed
        import base58 as b58

        decoded = b58.b58decode(private_key)
        if len(decoded) == 32:
            kp = Keypair.from_seed(decoded)
            return str(kp.pubkey())
        raise


def save_solana_wallet(private_key: str) -> Path:
    WALLET_DIR.mkdir(exist_ok=True)
    SOLANA_WALLET_FILE.write_text(private_key)
    SOLANA_WALLET_FILE.chmod(0o600)
    return SOLANA_WALLET_FILE


def _expand_solana_seed(private_key: str) -> str:
    """If private_key is a 32-byte seed, expand to 64-byte keypair bs58 string."""
    import base58 as b58
    from solders.keypair import Keypair  # type: ignore

    decoded = b58.b58decode(private_key)
    if len(decoded) == 32:
        kp = Keypair.from_seed(decoded)
        return b58.b58encode(bytes(kp)).decode()
    return private_key


def scan_solana_wallets() -> List[Dict[str, str]]:
    """
    Discover ~/.<dir>/solana-wallet.json files from other providers.

    Each file should contain JSON with "privateKey" and "address" fields.
    Results are sorted by modification time (most recent first). Discovery is
    opt-in and must never replace the canonical BlockRun wallet automatically.
    32-byte seeds are automatically converted to 64-byte keypairs.

    Returns:
        List of dicts with 'private_key' and 'address', most recent first
    """
    home = Path.home()
    results: List[tuple] = []  # (mtime, private_key, address)

    try:
        for entry in home.iterdir():
            if not entry.name.startswith(".") or not entry.is_dir():
                continue
            wallet_file = entry / "solana-wallet.json"
            if not wallet_file.is_file():
                continue
            try:
                data = json.loads(wallet_file.read_text())
                pk = data.get("privateKey", "")
                addr = data.get("address", "")
                if pk and addr:
                    # Expand 32-byte seeds to full keypairs
                    try:
                        pk = _expand_solana_seed(pk)
                    except Exception:
                        pass
                    mtime = wallet_file.stat().st_mtime
                    results.append((mtime, pk, addr))
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass

    # Sort by modification time, most recent first
    results.sort(key=lambda x: x[0], reverse=True)
    return [{"private_key": pk, "address": addr} for _, pk, addr in results]


def load_solana_wallet() -> Optional[str]:
    """
    Load Solana wallet private key.

    Priority:
    1. ~/.blockrun/.solana-session
    """
    # The canonical BlockRun wallet always wins over a discovered provider key.
    if SOLANA_WALLET_FILE.exists():
        try:
            key = SOLANA_WALLET_FILE.read_text().strip()
        except OSError:
            return None  # unreadable (bad perms/ownership) → treat as "no wallet"
        if key:
            return key
    return None


def get_or_create_solana_wallet() -> Dict[str, object]:
    """
    Get existing Solana wallet or create new one.

    Priority:
    1. SOLANA_WALLET_KEY env var
    2. ~/.blockrun/.solana-session
    3. Create new

    Returns:
        Dict with 'address', 'private_key', 'is_new'
    """
    # 1. Environment variable
    env_key = os.environ.get("SOLANA_WALLET_KEY")
    if env_key:
        return {"private_key": env_key, "address": get_solana_public_key(env_key), "is_new": False}

    # 2. Canonical BlockRun session file. scan_solana_wallets() is exposed
    # only for an explicit migration flow.
    if SOLANA_WALLET_FILE.exists():
        file_key = SOLANA_WALLET_FILE.read_text().strip()
        if file_key:
            return {
                "private_key": file_key,
                "address": get_solana_public_key(file_key),
                "is_new": False,
            }

    # 3. Create new
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
