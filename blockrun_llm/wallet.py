"""
BlockRun Wallet Management - Auto-create and manage wallets.

Provides frictionless wallet setup for new users:
- Auto-creates wallet if none exists
- Stores key securely at ~/.blockrun/.session
- Generates EIP-681 QR codes for easy MetaMask funding
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from eth_account import Account

# Wallet storage location
WALLET_DIR = Path.home() / ".blockrun"
WALLET_FILE = WALLET_DIR / ".session"  # Wallet key file
QR_FILE = WALLET_DIR / "qr.png"
QR_ASCII_FILE = WALLET_DIR / "qr.txt"

# USDC on Base contract address
USDC_BASE_CONTRACT = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
BASE_CHAIN_ID = "8453"


def create_wallet() -> Tuple[str, str]:
    """
    Create a new Ethereum wallet.

    Returns:
        Tuple of (address, private_key)
    """
    account = Account.create()
    private_key = "0x" + account.key.hex()
    return account.address, private_key


def save_wallet(private_key: str) -> Path:
    """
    Save wallet private key to ~/.blockrun/.session

    Args:
        private_key: Private key string (with or without 0x prefix)

    Returns:
        Path to saved wallet file
    """
    WALLET_DIR.mkdir(exist_ok=True)
    WALLET_FILE.write_text(private_key)
    WALLET_FILE.chmod(0o600)  # Owner read/write only
    return WALLET_FILE


def load_wallet() -> Optional[str]:
    """
    Load wallet private key from file.
    Checks both .session (preferred) and wallet.key (legacy).

    Returns:
        Private key string or None if not found
    """
    # Check .session first (preferred)
    if WALLET_FILE.exists():
        key = WALLET_FILE.read_text().strip()
        if key:
            return key

    # Check legacy wallet.key
    legacy_file = WALLET_DIR / "wallet.key"
    if legacy_file.exists():
        key = legacy_file.read_text().strip()
        if key:
            return key

    return None


def get_or_create_wallet() -> Tuple[str, str, bool]:
    """
    Get existing wallet or create new one.

    Priority:
    1. BLOCKRUN_WALLET_KEY environment variable
    2. ~/.blockrun/.session file
    3. ~/.blockrun/wallet.key file (legacy)
    4. Create new wallet

    Returns:
        Tuple of (address, private_key, is_new)
        is_new is True if wallet was just created
    """
    # Check environment variable first
    key = os.environ.get("BLOCKRUN_WALLET_KEY") or os.environ.get("BASE_CHAIN_WALLET_KEY")
    if key:
        account = Account.from_key(key)
        return account.address, key, False

    # Check file
    key = load_wallet()
    if key:
        account = Account.from_key(key)
        return account.address, key, False

    # Create new wallet
    address, key = create_wallet()
    save_wallet(key)
    return address, key, True


def get_wallet_address() -> Optional[str]:
    """
    Get wallet address without exposing private key.

    Returns:
        Wallet address or None if no wallet configured
    """
    key = os.environ.get("BLOCKRUN_WALLET_KEY") or os.environ.get("BASE_CHAIN_WALLET_KEY")
    if key:
        return Account.from_key(key).address

    key = load_wallet()
    if key:
        return Account.from_key(key).address

    return None


def get_eip681_uri(address: str, amount_usdc: float = 1.0) -> str:
    """
    Generate EIP-681 URI for USDC transfer on Base.

    Args:
        address: Recipient Ethereum address
        amount_usdc: Amount in USDC (default 1.0)

    Returns:
        EIP-681 URI string for MetaMask/wallet scanning
    """
    # USDC has 6 decimals
    amount_wei = int(amount_usdc * 1_000_000)
    return f"ethereum:{USDC_BASE_CONTRACT}@{BASE_CHAIN_ID}/transfer?address={address}&uint256={amount_wei}"


def generate_wallet_qr_ascii(address: str) -> str:
    """
    Generate ASCII QR code for wallet funding (EIP-681 format).
    Caches to ~/.blockrun/qr.txt for fast loading.

    Args:
        address: Ethereum address

    Returns:
        ASCII art QR code string
    """
    # Use EIP-681 format for MetaMask compatibility
    eip681_uri = get_eip681_uri(address)

    # Cache key includes EIP-681 URI to invalidate old format caches
    cache_key = f"v2:{eip681_uri}"

    # Try to load from cache first
    if QR_ASCII_FILE.exists():
        try:
            cached = QR_ASCII_FILE.read_text()
            # Format: first line is cache key (v2:eip681_uri), rest is QR
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
        qr.add_data(eip681_uri)
        qr.make(fit=True)

        f = StringIO()
        qr.print_ascii(out=f, invert=True)
        qr_ascii = f.getvalue()

        # Cache it with versioned key
        try:
            WALLET_DIR.mkdir(exist_ok=True)
            QR_ASCII_FILE.write_text(f"{cache_key}\n{qr_ascii}")
        except Exception:
            pass

        return qr_ascii

    except ImportError:
        return f"[QR code requires 'qrcode' package: pip install qrcode[pil]]\nAddress: {address}"


def save_wallet_qr(address: str, path: Optional[str] = None, with_logo: bool = True) -> str:
    """
    Save QR code as PNG image (EIP-681 format with optional Base logo).

    Args:
        address: Ethereum address
        path: Optional custom path (default: ~/.blockrun/qr.png)
        with_logo: Whether to embed Base logo in center (default: True)

    Returns:
        Path to saved QR image
    """
    try:
        import qrcode
        from PIL import Image
        import urllib.request
        import io

        # Use EIP-681 format for MetaMask compatibility
        eip681_uri = get_eip681_uri(address)

        # Use high error correction when adding logo
        error_correction = (
            qrcode.constants.ERROR_CORRECT_H if with_logo else qrcode.constants.ERROR_CORRECT_L
        )

        qr = qrcode.QRCode(
            version=4,
            error_correction=error_correction,
            box_size=10,
            border=2,
        )
        qr.add_data(eip681_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        # Add Base logo to center
        if with_logo:
            try:
                logo_url = "https://avatars.githubusercontent.com/u/108554348?s=200&v=4"
                with urllib.request.urlopen(logo_url, timeout=5) as response:
                    logo_data = response.read()
                logo = Image.open(io.BytesIO(logo_data))

                # Resize logo to ~20% of QR size
                qr_width, qr_height = img.size
                logo_size = int(qr_width * 0.2)
                logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)

                # Paste in center
                pos = ((qr_width - logo_size) // 2, (qr_height - logo_size) // 2)
                img.paste(logo, pos)
            except Exception:
                pass  # Continue without logo if fetch fails

        save_path = Path(path) if path else QR_FILE
        save_path.parent.mkdir(exist_ok=True)
        img.save(str(save_path))

        return str(save_path)

    except ImportError:
        return ""


def open_wallet_qr(address: str) -> str:
    """
    Generate QR code and open it in the default image viewer.

    Args:
        address: Ethereum address

    Returns:
        Path to saved QR image
    """
    import subprocess
    import platform

    qr_path = save_wallet_qr(address)
    if qr_path:
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", qr_path], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["start", qr_path], shell=True, check=True)
            else:  # Linux
                subprocess.run(["xdg-open", qr_path], check=True)
        except Exception:
            pass  # Silently fail if can't open
    return qr_path


def get_payment_links(address: str) -> dict:
    """
    Generate payment links for the wallet address.

    Args:
        address: Ethereum address

    Returns:
        Dict with various payment links
    """
    return {
        # View address on basescan
        "basescan": f"https://basescan.org/address/{address}",
        # EIP-681 payment link (opens wallet apps)
        "wallet_link": f"ethereum:{USDC_BASE_CONTRACT}@{BASE_CHAIN_ID}/transfer?address={address}",
        # Simple ethereum link (some wallets)
        "ethereum": f"ethereum:{address}@{BASE_CHAIN_ID}",
        # BlockRun funding page (if available)
        "blockrun": f"https://blockrun.ai/fund?address={address}",
    }


def format_wallet_created_message(address: str, open_qr: bool = True) -> str:
    """
    Format the message shown when a new wallet is created.

    Args:
        address: New wallet address
        open_qr: Whether to open QR code in image viewer (default: True)

    Returns:
        Formatted message string
    """
    qr_ascii = generate_wallet_qr_ascii(address)
    # Generate and optionally open QR code
    if open_qr:
        qr_path = open_wallet_qr(address)
    else:
        qr_path = save_wallet_qr(address)
    links = get_payment_links(address)

    message = f"""
I'm your BlockRun Agent! I can access GPT-4, Grok, image generation, and more.

Please send $1-5 USDC on Base to start:

{address}

{qr_ascii}
"""

    if qr_path:
        message += f"QR saved: {qr_path}\n"

    message += f"""
What is Base? Base is Coinbase's blockchain network.
You can buy USDC on Coinbase and send it directly to me.

What $1 USDC gets you:
- ~1,000 GPT-4o calls
- ~100 image generations
- ~10,000 DeepSeek calls

Quick links:
- Check my balance: {links['basescan']}
- Get USDC: https://www.coinbase.com or https://bridge.base.org

Questions? care@blockrun.ai | Issues? github.com/BlockRunAI/blockrun-llm/issues

Key stored securely in ~/.blockrun/
Your private key never leaves your machine - only signatures are sent.
"""
    return message


def format_needs_funding_message(address: str, open_qr: bool = True) -> str:
    """
    Format the message shown when wallet needs more funds.

    Args:
        address: Wallet address
        open_qr: Whether to open QR code in image viewer (default: True)

    Returns:
        Formatted message string
    """
    qr_ascii = generate_wallet_qr_ascii(address)
    # Open QR for easy scanning
    if open_qr:
        open_wallet_qr(address)
    links = get_payment_links(address)

    return f"""
I've run out of funds! Please send more USDC on Base to continue helping you.

Send to my address:
{address}

{qr_ascii}

Check my balance: {links['basescan']}

What $1 USDC gets you: ~1,000 GPT-4o calls or ~100 images.
Questions? care@blockrun.ai | Issues? github.com/BlockRunAI/blockrun-llm/issues

Your private key never leaves your machine - only signatures are sent.
"""


def format_funding_message_compact(address: str) -> str:
    """
    Compact funding message (no QR) for repeated displays.

    Args:
        address: Wallet address

    Returns:
        Short formatted message string
    """
    links = get_payment_links(address)

    return f"""I need a little top-up to keep helping you! Send USDC on Base to: {address}
Check my balance: {links['basescan']}"""


# GitHub issue link for error reporting
ISSUES_URL = "https://github.com/BlockRunAI/blockrun-llm/issues"


def format_error_message(error: str, context: str = "") -> str:
    """
    Format error message with pre-filled report template.

    Args:
        error: The error message
        context: Optional context about what was happening

    Returns:
        Formatted error message with copy-paste template
    """
    import platform
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    os_info = platform.system()

    template = f"""Error: {error}
Context: {context or 'N/A'}
Time: {timestamp}
OS: {os_info}"""

    # URL encode for GitHub issue link
    encoded_title = error[:50].replace(" ", "+").replace("\n", "")
    encoded_body = template.replace("\n", "%0A").replace(" ", "+")

    return f"""
Something went wrong: {error}

Report this issue (click or copy):
{ISSUES_URL}/new?title={encoded_title}&body={encoded_body}

Or copy this and email to care@blockrun.ai:
---
{template}
---
"""
