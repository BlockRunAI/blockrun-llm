# SolanaLLMClient Python SDK Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `SolanaLLMClient` class to `blockrun-llm` Python SDK so Solana developers can pay for AI calls with Solana USDC via x402.

**Architecture:** New `SolanaLLMClient` class in `blockrun_llm/solana_client.py` (mirrors `LLMClient` but uses Solana keypair). New `create_solana_payment_payload` in `blockrun_llm/x402.py`. Solana keypair management in `blockrun_llm/solana_wallet.py`. Uses `solders` for keypair/transaction and `httpx` (already a dep) for Solana RPC. All existing Base/EVM code is untouched.

**Tech Stack:** Python 3.9+, `solders>=0.21.0` (new optional dep), `httpx` (already a dep), `base58>=2.1.0` (new optional dep)

---

### Task 1: Add Solana deps to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add optional Solana deps**

In `[project.optional-dependencies]`, add:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black==24.10.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]
solana = [
    "solders>=0.21.0",
    "base58>=2.1.0",
]
```

**Step 2: Install Solana extras in dev env**

```bash
cd /Users/vickyfu/Documents/blockrun-web/blockrun-llm
source /Users/vickyfu/myenv_py313/bin/activate
pip install solders>=0.21.0 base58>=2.1.0
```
Expected: both packages install

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add optional Solana dependencies to pyproject.toml"
```

---

### Task 2: Add Solana wallet utilities

**Files:**
- Create: `blockrun_llm/solana_wallet.py`
- Test: `tests/unit/test_solana_wallet.py`

**Context:** Solana keys are bs58-encoded 64-byte secret keys. Address is base58 public key. Key stored at `~/.blockrun/.solana-session`.

**Step 1: Write failing tests**

Create `tests/unit/test_solana_wallet.py`:

```python
"""Unit tests for Solana wallet utilities."""
import pytest
from blockrun_llm.solana_wallet import (
    create_solana_wallet,
    solana_key_to_bytes,
    get_solana_public_key,
)

# A valid test bs58 secret key (64 bytes encoded)
TEST_BS58_KEY = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviQeRjpzKCY8trDwpvBMTKTpNFbCJsBZthJ4tCs6o62rr"


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
        assert re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', addr)
```

**Step 2: Run to verify fails**

```bash
source /Users/vickyfu/myenv_py313/bin/activate
cd /Users/vickyfu/Documents/blockrun-web/blockrun-llm
pytest tests/unit/test_solana_wallet.py -v
```
Expected: FAIL "ImportError: cannot import name"

**Step 3: Implement `blockrun_llm/solana_wallet.py`**

```python
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
    kp = Keypair.from_bytes(secret)
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
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_solana_wallet.py -v
```
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add blockrun_llm/solana_wallet.py tests/unit/test_solana_wallet.py
git commit -m "feat: add Solana wallet utilities"
```

---

### Task 3: Add create_solana_payment_payload to x402.py

**Files:**
- Modify: `blockrun_llm/x402.py`
- Test: `tests/unit/test_x402.py`

**Context:** The 402 response from `sol.blockrun.ai` has:
- `network: "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"`
- `amount: "1000"` (micro USDC, 6 decimals)
- `payTo: "AQqnMFBwGZEoti85aTVRy8XYpKrho7GaMDx9ZB3CEeKA"`
- `asset: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"` (Solana USDC)
- `extra.feePayer: "2wKupLR9q6wXYppw8Gr2NvWxKBUqm4PPJKkQfoxHDBg4"` (CDP fee payer)

The transaction is an SPL Token TransferChecked with compute budget instructions, signed by the user's keypair. The feePayer (CDP) will co-sign on the server side.

The ATA (Associated Token Account) is derived as PDA: `[owner_bytes, TOKEN_PROGRAM_ID_bytes, mint_bytes]` under ASSOCIATED_TOKEN_PROGRAM_ID.

**Step 1: Write failing tests**

Add to `tests/unit/test_x402.py`:

```python
# Add at the end of test_x402.py:

class TestCreateSolanaPaymentPayload:
    """Tests for Solana payment payload creation."""

    TEST_BS58_KEY = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviQeRjpzKCY8trDwpvBMTKTpNFbCJsBZthJ4tCs6o62rr"
    TEST_FEE_PAYER = "2wKupLR9q6wXYppw8Gr2NvWxKBUqm4PPJKkQfoxHDBg4"
    TEST_RECIPIENT = "AQqnMFBwGZEoti85aTVRy8XYpKrho7GaMDx9ZB3CEeKA"

    def test_payload_structure(self):
        """Should create valid Solana payment payload."""
        from blockrun_llm.x402 import create_solana_payment_payload
        import json, base64

        payload = create_solana_payment_payload(
            private_key=self.TEST_BS58_KEY,
            recipient=self.TEST_RECIPIENT,
            amount="1000",
            fee_payer=self.TEST_FEE_PAYER,
        )

        assert isinstance(payload, str)
        decoded = json.loads(base64.b64decode(payload))
        assert decoded["x402Version"] == 2
        assert "transaction" in decoded["payload"]
        assert decoded["accepted"]["network"].startswith("solana:")
        assert decoded["accepted"]["asset"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    def test_payload_transaction_is_base64(self):
        """Transaction field should be base64-encoded."""
        from blockrun_llm.x402 import create_solana_payment_payload
        import json, base64

        payload = create_solana_payment_payload(
            private_key=self.TEST_BS58_KEY,
            recipient=self.TEST_RECIPIENT,
            amount="1000",
            fee_payer=self.TEST_FEE_PAYER,
        )
        decoded = json.loads(base64.b64decode(payload))
        # Should be valid base64
        tx_bytes = base64.b64decode(decoded["payload"]["transaction"])
        assert len(tx_bytes) > 0
```

**Step 2: Run to verify fails**

```bash
pytest tests/unit/test_x402.py::TestCreateSolanaPaymentPayload -v
```
Expected: FAIL "cannot import name 'create_solana_payment_payload'"

**Step 3: Add `create_solana_payment_payload` to `blockrun_llm/x402.py`**

Append to the end of `blockrun_llm/x402.py`:

```python
# ============================================================
# Solana x402 Payment
# ============================================================

SOLANA_NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
USDC_SOLANA = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# SPL program IDs
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bRS"

# Compute budget defaults (match @x402/svm)
DEFAULT_COMPUTE_UNIT_PRICE_MICROLAMPORTS = 1
DEFAULT_COMPUTE_UNIT_LIMIT = 8000


def _get_ata(owner: str, mint: str) -> str:
    """Derive Associated Token Account address."""
    from solders.pubkey import Pubkey  # type: ignore

    owner_pk = Pubkey.from_string(owner)
    mint_pk = Pubkey.from_string(mint)
    token_program = Pubkey.from_string(TOKEN_PROGRAM_ID)
    assoc_program = Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM_ID)

    seeds = [bytes(owner_pk), bytes(token_program), bytes(mint_pk)]
    ata, _ = Pubkey.find_program_address(seeds, assoc_program)
    return str(ata)


def _get_latest_blockhash(rpc_url: str) -> str:
    """Fetch latest blockhash from Solana RPC."""
    import httpx
    resp = httpx.post(
        rpc_url,
        json={"jsonrpc": "2.0", "id": 1, "method": "getLatestBlockhash",
              "params": [{"commitment": "finalized"}]},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["result"]["value"]["blockhash"]


def create_solana_payment_payload(
    private_key: str,
    recipient: str,
    amount: str,
    fee_payer: str,
    resource_url: str = "https://sol.blockrun.ai/api/v1/chat/completions",
    resource_description: str = "BlockRun Solana AI API call",
    max_timeout_seconds: int = 300,
    extra: Optional[Dict[str, Any]] = None,
    extensions: Optional[Dict[str, Any]] = None,
    rpc_url: str = "https://api.mainnet-beta.solana.com",
) -> str:
    """
    Create a signed Solana x402 v2 payment payload.

    Builds an SPL TransferChecked transaction signed by the user's Solana keypair.
    The CDP facilitator (feePayer) co-signs on the server side.

    Args:
        private_key: bs58-encoded 64-byte Solana secret key
        recipient: Payment recipient Solana address (base58)
        amount: Amount in micro USDC (6 decimals, e.g. "1000" = $0.001)
        fee_payer: CDP facilitator address that pays SOL transaction fees (base58)
        resource_url: URL of the resource being accessed
        resource_description: Description for the payment
        max_timeout_seconds: Max timeout for the payment
        extra: Extra info included in payment (e.g. feePayer)
        extensions: x402 extensions dict
        rpc_url: Solana RPC endpoint

    Returns:
        Base64-encoded signed payment payload
    """
    try:
        from solders.keypair import Keypair  # type: ignore
        from solders.pubkey import Pubkey  # type: ignore
        from solders.hash import Hash  # type: ignore
        from solders.instruction import Instruction, AccountMeta  # type: ignore
        from solders.message import MessageV0  # type: ignore
        from solders.transaction import VersionedTransaction  # type: ignore
        import base58  # type: ignore
    except ImportError:
        raise ImportError(
            "Solana payment requires 'solders' and 'base58'. "
            "Install with: pip install blockrun-llm[solana]"
        )

    # Load keypair
    secret = base58.b58decode(private_key)
    keypair = Keypair.from_bytes(secret)
    owner_pubkey = keypair.pubkey()

    # Derive ATAs
    source_ata = _get_ata(str(owner_pubkey), USDC_SOLANA)
    dest_ata = _get_ata(recipient, USDC_SOLANA)

    # Get latest blockhash
    blockhash = _get_latest_blockhash(rpc_url)

    # Build compute budget instructions
    # ComputeBudgetProgram.setComputeUnitLimit
    compute_budget_id = Pubkey.from_string("ComputeBudget111111111111111111111111111111")

    # setComputeUnitLimit instruction: discriminator=2, units=u32 LE
    import struct
    limit_data = bytes([2]) + struct.pack("<I", DEFAULT_COMPUTE_UNIT_LIMIT)
    limit_ix = Instruction(compute_budget_id, limit_data, [])

    # setComputeUnitPrice instruction: discriminator=3, microLamports=u64 LE
    price_data = bytes([3]) + struct.pack("<Q", DEFAULT_COMPUTE_UNIT_PRICE_MICROLAMPORTS)
    price_ix = Instruction(compute_budget_id, price_data, [])

    # Build TransferChecked instruction
    token_program_id = Pubkey.from_string(TOKEN_PROGRAM_ID)
    mint_pubkey = Pubkey.from_string(USDC_SOLANA)
    source_ata_pk = Pubkey.from_string(source_ata)
    dest_ata_pk = Pubkey.from_string(dest_ata)
    recipient_pk = Pubkey.from_string(recipient)

    # TransferChecked: discriminator=12, amount=u64 LE, decimals=u8
    transfer_data = bytes([12]) + struct.pack("<Q", int(amount)) + bytes([6])
    transfer_accounts = [
        AccountMeta(source_ata_pk, is_signer=False, is_writable=True),
        AccountMeta(mint_pubkey, is_signer=False, is_writable=False),
        AccountMeta(dest_ata_pk, is_signer=False, is_writable=True),
        AccountMeta(owner_pubkey, is_signer=True, is_writable=False),
    ]
    transfer_ix = Instruction(token_program_id, transfer_data, transfer_accounts)

    # Build v0 message — order: limit, price, transfer (matches @x402/svm)
    fee_payer_pk = Pubkey.from_string(fee_payer)
    message = MessageV0.try_compile(
        payer=fee_payer_pk,
        instructions=[limit_ix, price_ix, transfer_ix],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.from_string(blockhash),
    )

    # Create and sign transaction (partial — only user signs)
    tx = VersionedTransaction(message, [keypair])

    # Serialize
    tx_b64 = base64.b64encode(bytes(tx)).decode()

    # Build payload
    payment_data = {
        "x402Version": 2,
        "resource": {
            "url": resource_url,
            "description": resource_description,
            "mimeType": "application/json",
        },
        "accepted": {
            "scheme": "exact",
            "network": SOLANA_NETWORK,
            "amount": amount,
            "asset": USDC_SOLANA,
            "payTo": recipient,
            "maxTimeoutSeconds": max_timeout_seconds,
            "extra": extra or {"feePayer": fee_payer},
        },
        "payload": {
            "transaction": tx_b64,
        },
        "extensions": extensions or {},
    }

    return base64.b64encode(json.dumps(payment_data).encode()).decode()


def is_solana_network(network: str) -> bool:
    """Check if a network string represents Solana."""
    return network.startswith("solana:")


def extract_solana_payment_details(payment_required: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Solana payment details from a 402 response.
    Finds the Solana network option in accepts[].
    """
    accepts = payment_required.get("accepts", [])
    option = next((o for o in accepts if is_solana_network(o.get("network", ""))), None)
    if not option:
        raise ValueError("No Solana payment option found in 402 response")

    amount = option.get("amount") or option.get("maxAmountRequired")
    if not amount:
        raise ValueError("No amount in Solana payment requirements")

    return {
        "amount": amount,
        "recipient": option.get("payTo"),
        "network": option.get("network"),
        "asset": option.get("asset"),
        "max_timeout_seconds": option.get("maxTimeoutSeconds", 300),
        "extra": option.get("extra", {}),
        "resource": payment_required.get("resource"),
    }
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_x402.py::TestCreateSolanaPaymentPayload -v
```
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add blockrun_llm/x402.py tests/unit/test_x402.py
git commit -m "feat: add create_solana_payment_payload to x402"
```

---

### Task 4: Add SolanaLLMClient class

**Files:**
- Create: `blockrun_llm/solana_client.py`
- Test: `tests/unit/test_solana_client.py`

**Step 1: Write failing tests**

Create `tests/unit/test_solana_client.py`:

```python
"""Unit tests for SolanaLLMClient."""
import pytest
import os
from blockrun_llm.solana_client import SolanaLLMClient

TEST_BS58_KEY = "5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviQeRjpzKCY8trDwpvBMTKTpNFbCJsBZthJ4tCs6o62rr"


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
        with pytest.raises(ValueError, match="private key required"):
            SolanaLLMClient()
        if saved:
            os.environ["SOLANA_WALLET_KEY"] = saved

    def test_default_api_url(self):
        client = SolanaLLMClient(private_key=TEST_BS58_KEY)
        assert client.is_solana()

    def test_custom_api_url(self):
        client = SolanaLLMClient(
            private_key=TEST_BS58_KEY,
            api_url="https://custom.example.com/api"
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
```

**Step 2: Run to verify fails**

```bash
pytest tests/unit/test_solana_client.py -v
```
Expected: FAIL "No module named 'blockrun_llm.solana_client'"

**Step 3: Implement `blockrun_llm/solana_client.py`**

```python
"""
BlockRun Solana LLM Client.

Usage:
    from blockrun_llm import SolanaLLMClient

    # SOLANA_WALLET_KEY env var (bs58-encoded Solana secret key)
    client = SolanaLLMClient()

    # Or pass key directly
    client = SolanaLLMClient(private_key="your-bs58-key")

    # Same API as LLMClient
    response = client.chat("openai/gpt-4o", "gm Solana")
    print(response)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from .types import ChatResponse, APIError, PaymentError
from .x402 import (
    create_solana_payment_payload,
    extract_solana_payment_details,
    parse_payment_required,
    SOLANA_NETWORK,
)
from .solana_wallet import get_solana_public_key
from .validation import validate_api_url, sanitize_error_response, validate_resource_url

SOLANA_API_URL = "https://sol.blockrun.ai/api"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0


def _get_user_agent() -> str:
    from . import __version__
    return f"blockrun-python/{__version__}"


class SolanaLLMClient:
    """
    BlockRun LLM Client for Solana — pays via Solana USDC x402.

    Connects to sol.blockrun.ai by default.
    """

    SOLANA_API_URL = SOLANA_API_URL

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: str = SOLANA_API_URL,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        key = private_key or os.environ.get("SOLANA_WALLET_KEY")
        if not key:
            raise ValueError(
                "Private key required. Pass private_key or set SOLANA_WALLET_KEY env var."
            )
        self._private_key = key
        validate_api_url(api_url)
        self._api_url = api_url.rstrip("/")
        self._rpc_url = rpc_url
        self._timeout = timeout
        self._session_total_usd = 0.0
        self._session_calls = 0
        self._address: Optional[str] = None

    def get_wallet_address(self) -> str:
        if not self._address:
            self._address = get_solana_public_key(self._private_key)
        return self._address

    def is_solana(self) -> bool:
        return "sol.blockrun.ai" in self._api_url

    def get_spending(self) -> Dict[str, Any]:
        return {"total_usd": self._session_total_usd, "calls": self._session_calls}

    def chat(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        search: bool = False,
    ) -> str:
        """Simple 1-line chat."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat_completion(
            model, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            search=search,
        )
        return result.choices[0].message.content or ""

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """Full chat completion (OpenAI-compatible)."""
        body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        return self._request_with_payment("/v1/chat/completions", body)

    def list_models(self) -> List[Dict[str, Any]]:
        with httpx.Client(timeout=self._timeout) as http:
            resp = http.get(f"{self._api_url}/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    def _request_with_payment(
        self, endpoint: str, body: Dict[str, Any]
    ) -> ChatResponse:
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        with httpx.Client(timeout=self._timeout) as http:
            response = http.post(url, json=body, headers=headers)

        if response.status_code == 402:
            return self._handle_payment_and_retry(url, body, response)

        if not response.is_success:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return ChatResponse(**response.json())

    def _handle_payment_and_retry(
        self, url: str, body: Dict[str, Any], response: httpx.Response
    ) -> ChatResponse:
        # Get payment header
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                resp_body = response.json()
                if resp_body.get("accepts") or resp_body.get("x402Version"):
                    import base64, json
                    payment_header = base64.b64encode(
                        json.dumps(resp_body).encode()
                    ).decode()
            except Exception:
                pass

        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = parse_payment_required(payment_header)
        details = extract_solana_payment_details(payment_required)

        if not details["network"].startswith("solana:"):
            raise PaymentError(
                f"Expected Solana network, got: {details['network']}. "
                "Use LLMClient for Base payments."
            )

        fee_payer = (details.get("extra") or {}).get("feePayer")
        if not fee_payer:
            raise PaymentError("Missing feePayer in 402 extra field")

        resource_info = details.get("resource") or {}
        resource_url = validate_resource_url(
            resource_info.get("url") or f"{self._api_url}/v1/chat/completions",
            self._api_url,
        )

        payment_payload = create_solana_payment_payload(
            private_key=self._private_key,
            recipient=details["recipient"],
            amount=details["amount"],
            fee_payer=fee_payer,
            resource_url=resource_url,
            resource_description=resource_info.get("description") or "BlockRun Solana AI API call",
            max_timeout_seconds=details["max_timeout_seconds"],
            extra=details.get("extra"),
            rpc_url=self._rpc_url,
        )

        headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        with httpx.Client(timeout=self._timeout) as http:
            retry_response = http.post(url, json=body, headers=headers)

        if retry_response.status_code == 402:
            raise PaymentError("Payment rejected. Check your Solana USDC balance.")

        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        cost_usd = float(details["amount"]) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd

        return ChatResponse(**retry_response.json())
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_solana_client.py -v
```
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add blockrun_llm/solana_client.py tests/unit/test_solana_client.py
git commit -m "feat: add SolanaLLMClient for Solana USDC payments"
```

---

### Task 5: Update exports and version

**Files:**
- Modify: `blockrun_llm/__init__.py`
- Modify: `pyproject.toml` (version bump)

**Step 1: Read current `__init__.py` exports**

```bash
head -60 blockrun_llm/__init__.py
```

**Step 2: Add SolanaLLMClient to exports**

In `blockrun_llm/__init__.py`, add alongside `LLMClient`:

```python
from .solana_client import SolanaLLMClient
```

And in `__all__` (if present):
```python
"SolanaLLMClient",
```

**Step 3: Bump version in `pyproject.toml`**

Change `version = "0.4.1"` → `version = "0.5.0"` (minor bump, new feature).

Also update `blockrun_llm/__init__.py` version string if present.

**Step 4: Run full test suite**

```bash
pytest tests/unit/ -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add blockrun_llm/__init__.py pyproject.toml
git commit -m "feat: export SolanaLLMClient and bump to 0.5.0"
```

---

### Task 6: Update README and publish

**Files:**
- Modify: `README.md`

**Step 1: Add Solana section to README**

Find the "Supported Chains" table and update it:

```markdown
| Chain | Network | Payment | Status |
|-------|---------|---------|--------|
| **Base** | Base Mainnet (Chain ID: 8453) | USDC | ✅ Primary |
| **Base Testnet** | Base Sepolia (Chain ID: 84532) | Testnet USDC | ✅ Development |
| **Solana** | Solana Mainnet | USDC (SPL) | ✅ New |
```

Add a new section after Quick Start:

```markdown
## Solana Support

Pay for AI calls with Solana USDC via [sol.blockrun.ai](https://sol.blockrun.ai):

\`\`\`python
from blockrun_llm import SolanaLLMClient

# SOLANA_WALLET_KEY env var (bs58-encoded Solana secret key)
client = SolanaLLMClient()

# Or pass key directly
client = SolanaLLMClient(private_key="your-bs58-solana-key")

# Same API as LLMClient
response = client.chat("openai/gpt-4o", "gm Solana")
print(response)

# Live Search with Grok (Solana payment)
tweet = client.chat("xai/grok-3-mini", "What is trending on X?", search=True)
\`\`\`

**Setup:**
\`\`\`bash
pip install blockrun-llm[solana]
export SOLANA_WALLET_KEY="your-bs58-solana-key"
\`\`\`

**Endpoint:** `https://sol.blockrun.ai/api`
**Payment:** Solana USDC (SPL Token, mainnet)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Solana section to README"
```

**Step 3: Build and publish**

```bash
source /Users/vickyfu/myenv_py313/bin/activate
pip install build
python -m build
pip install twine
twine upload dist/blockrun_llm-0.5.0*
```

**Step 4: Push to GitHub**

```bash
git push
```
