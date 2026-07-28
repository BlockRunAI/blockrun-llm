"""
BlockRun Solana Wallet Management.

Stores keys as bs58-encoded strings at ~/.blockrun/.solana-session.
Requires: solders>=0.21.0, base58>=2.1.0
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

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


def create_solana_wallet() -> dict[str, str]:
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
    Convert a Solana private key string to bytes (64 bytes).

    Accepts a bs58-encoded 64-byte keypair (standard Solana format), a
    bs58-encoded 32-byte seed from other providers (automatically expanded),
    the Solana CLI JSON byte-array format (``~/.config/solana/id.json``), or
    a 64-byte hex string with or without ``0x``. A 32-byte hex key is
    rejected with an explicit hint that it is the EVM (Base) wallet format.

    Args:
        private_key: Solana secret key in any accepted encoding

    Returns:
        64-byte secret key as bytes

    Raises:
        ValueError: If key is invalid
    """
    key = private_key.strip()

    # Solana CLI JSON array format: [12,34,...] with 64 (or 32) byte values
    if key.startswith("["):
        try:
            parsed = json.loads(key)
        except json.JSONDecodeError as e:
            raise ValueError(
                "Invalid Solana private key: looks like a JSON byte array " "but is not valid JSON"
            ) from e
        if not isinstance(parsed, list) or not all(
            isinstance(n, int) and 0 <= n <= 255 for n in parsed
        ):
            raise ValueError(
                "Invalid Solana private key: JSON array must contain only " "byte values (0-255)"
            )
        if len(parsed) not in (32, 64):
            raise ValueError(
                f"Invalid Solana key length: expected 32 or 64 bytes, got {len(parsed)}"
            )
        return _expand_key_bytes(bytes(parsed))

    # Hex forms. bs58 keys are 86-88 chars, so 64/128 hex chars are unambiguous.
    hex_body = key[2:] if key[:2] in ("0x", "0X") else key
    if len(hex_body) in (64, 128) and all(c in "0123456789abcdefABCDEF" for c in hex_body):
        if len(hex_body) == 64:
            raise ValueError(
                "Invalid Solana private key: this is a 32-byte hex key — the "
                "EVM (Base) wallet format, not a Solana key. Solana keys are "
                "64 bytes, usually base58-encoded (86-88 characters)."
            )
        return _expand_key_bytes(bytes.fromhex(hex_body))

    try:
        from solders.keypair import Keypair  # type: ignore

        try:
            kp = Keypair.from_base58_string(key)
            decoded = bytes(kp)
            if len(decoded) == 64:
                return decoded
        except Exception:
            pass

        # Fallback: try as 32-byte seed
        import base58 as b58

        decoded = b58.b58decode(key)
        if len(decoded) in (32, 64):
            return _expand_key_bytes(decoded)

        raise ValueError(f"Expected 32 or 64 bytes, got {len(decoded)}")
    except Exception as e:
        # Wrap every failure — including the ``ValueError`` modern ``base58``
        # raises on invalid characters — in the documented message. A bare
        # ``except ValueError: raise`` here used to leak base58's raw
        # "Invalid character" error past the wrapper, breaking callers (and
        # the test) that match on "Invalid Solana private key".
        raise ValueError(
            f"Invalid Solana private key: {e}. Expected a base58-encoded "
            "64-byte key (standard Solana format), a 64-byte hex string, or "
            "a Solana CLI JSON byte array."
        ) from e


def _expand_key_bytes(decoded: bytes) -> bytes:
    """Expand a 32-byte seed (or normalize a 64-byte keypair) to 64 bytes."""
    _require_solders()
    from solders.keypair import Keypair  # type: ignore

    if len(decoded) == 32:
        return bytes(Keypair.from_seed(decoded))
    return bytes(Keypair.from_seed(decoded[:32]))


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

    # solana_key_to_bytes handles every accepted encoding, including 32-byte
    # seeds, so a failure here is final — no fallback decode.
    secret = solana_key_to_bytes(private_key)
    kp = Keypair.from_seed(secret[:32])
    return str(kp.pubkey())


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


def scan_solana_wallets() -> list[dict[str, str]]:
    """
    Discover ~/.<dir>/solana-wallet.json files from other providers.

    Each file should contain JSON with "privateKey" and "address" fields.
    Results are sorted by modification time (most recent first). Discovery is
    opt-in and must never replace the canonical BlockRun wallet automatically.
    32-byte seeds are automatically converted to 64-byte keypairs.

    Returns:
        List of dicts with 'private_key', 'address' and 'source', most recent
        first. 'address' is the file's own claim — use
        list_discovered_solana_wallets() for an address derived from the key.
    """
    home = Path.home()
    results: list[tuple] = []  # (mtime, private_key, address, source)

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
                    results.append((mtime, pk, addr, str(wallet_file)))
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass

    # Sort by modification time, most recent first
    results.sort(key=lambda x: x[0], reverse=True)
    return [{"private_key": pk, "address": addr, "source": src} for _, pk, addr, src in results]


def list_discovered_solana_wallets() -> list[dict[str, str]]:
    """
    List Solana wallets from other applications, safe to show to a user.

    Solana counterpart of ``wallet.list_discovered_wallets``: no secret key is
    returned and the address is derived from the key rather than trusted from
    the file. Nothing here is active — adopt one with import_solana_wallet().

    Returns:
        List of dicts with 'address' and 'source', most recent first
    """
    listed = []
    for entry in scan_solana_wallets():
        try:
            address = get_solana_public_key(entry["private_key"])
        except Exception:
            continue
        listed.append({"address": address, "source": entry.get("source", "")})
    return listed


def import_solana_wallet(address: str) -> str:
    """
    Adopt a discovered Solana wallet, making it the active BlockRun wallet.

    Solana counterpart of ``wallet.import_wallet``. Matching is done against the
    address derived from each discovered key, and the current
    ~/.blockrun/.solana-session is backed up before being overwritten.

    Args:
        address: Address to adopt, as shown by list_discovered_solana_wallets()

    Returns:
        The adopted address

    Raises:
        ValueError: If no discovered wallet derives to that address
    """
    wanted = address.strip()

    for entry in scan_solana_wallets():
        try:
            derived = get_solana_public_key(entry["private_key"])
        except Exception:
            continue

        # Base58 is case-sensitive — compare exactly, unlike EVM hex.
        if derived != wanted:
            continue

        if SOLANA_WALLET_FILE.exists():
            current = SOLANA_WALLET_FILE.read_text().strip()
            if current and current != entry["private_key"]:
                backup = SOLANA_WALLET_FILE.with_name(f".solana-session.backup-{int(time.time())}")
                backup.write_text(current)
                backup.chmod(0o600)

        save_solana_wallet(entry["private_key"])
        return derived

    available = [w["address"] for w in list_discovered_solana_wallets()]
    raise ValueError(
        f"No discovered wallet controls {address}. "
        f"Available: {', '.join(available) if available else 'none'}"
    )


def load_solana_wallet() -> str | None:
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


def _public_key_from(private_key: str, source: str) -> str:
    """Derive the public key, attributing failures to where the key was loaded from."""
    try:
        return get_solana_public_key(private_key)
    except ValueError as e:
        raise ValueError(f"{e} (key loaded from {source})") from e


def get_or_create_solana_wallet() -> dict[str, object]:
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
        return {
            "private_key": env_key,
            "address": _public_key_from(env_key, "the SOLANA_WALLET_KEY environment variable"),
            "is_new": False,
        }

    # 2. Canonical BlockRun session file. scan_solana_wallets() is exposed
    # only for an explicit migration flow.
    if SOLANA_WALLET_FILE.exists():
        file_key = SOLANA_WALLET_FILE.read_text().strip()
        if file_key:
            return {
                "private_key": file_key,
                "address": _public_key_from(file_key, str(SOLANA_WALLET_FILE)),
                "is_new": False,
            }

    # 3. Create new
    wallet = create_solana_wallet()
    save_solana_wallet(wallet["private_key"])
    return {**wallet, "is_new": True}


def format_solana_wallet_migration_notice(new_address: str) -> str | None:
    """
    Warn when a new Solana wallet was created while provider wallets exist.

    Solana counterpart of ``wallet.format_wallet_migration_notice``. Addresses
    are derived from the discovered secret key rather than trusted from the
    file's "address" field.

    Args:
        new_address: Address of the wallet that was just created

    Returns:
        Formatted notice, or None if nothing was discovered
    """
    try:
        discovered = scan_solana_wallets()
    except Exception:
        return None

    addresses = []
    for entry in discovered:
        try:
            addresses.append(get_solana_public_key(entry["private_key"]))
        except Exception:
            continue

    if not addresses:
        return None

    found = "\n".join(f"  {addr}" for addr in addresses)
    return f"""
NOTICE: BlockRun created a new Solana wallet, but also found existing
wallet(s) belonging to other applications on this system:

{found}

BlockRun now uses only its own wallet:

  {new_address}

Discovered wallets are never adopted automatically — one may belong to a
different application, or have been planted to make you fund an address you
do not control.

If an address above is yours and holds your USDC, adopt it deliberately:

  from blockrun_llm import import_solana_wallet
  import_solana_wallet("<address-from-the-list-above>")

Your current wallet is backed up first. You can also set
SOLANA_WALLET_KEY=<private-key> for a single run without changing anything.
"""


def setup_agent_solana_wallet(silent: bool = False) -> SolanaLLMClient:
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

    if result["is_new"]:
        # Printed even when silent: `silent` suppresses the welcome message,
        # and losing sight of a funded wallet is not something to stay quiet
        # about.
        notice = format_solana_wallet_migration_notice(str(result["address"]))
        if notice:
            print(notice, file=sys.stderr)

        if not silent:
            print(f"New Solana wallet created: {result['address']}", file=sys.stderr)

    from .solana_client import SolanaLLMClient

    return SolanaLLMClient(private_key=result["private_key"])


def get_solana_usdc_balance(address: str, rpc_url: str | None = None) -> float:
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
        from io import StringIO

        import qrcode

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


def save_solana_wallet_qr(address: str, path: str | None = None) -> str:
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
    import platform
    import subprocess

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
