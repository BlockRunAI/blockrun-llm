"""
x402 Payment Protocol v2 Implementation for BlockRun.

This module handles creating signed payment payloads for the x402 v2 protocol.
The private key is used ONLY for local signing and NEVER leaves the client.
"""

import json
import time
import base64
import secrets
from typing import Dict, Any, Optional
from eth_account import Account
from eth_account.messages import encode_typed_data


# Chain and token constants for mainnet
BASE_CHAIN_ID = 8453
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# Chain and token constants for testnet (Base Sepolia)
BASE_SEPOLIA_CHAIN_ID = 84532
USDC_BASE_SEPOLIA = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"


def get_chain_config(network: str) -> tuple[int, str]:
    """
    Get chain ID and USDC contract address for a given network.

    Args:
        network: Network identifier in EIP-155 format (e.g., "eip155:8453" or "eip155:84532")

    Returns:
        Tuple of (chain_id, usdc_address)
    """
    if network == "eip155:84532" or network == "base-sepolia":
        return BASE_SEPOLIA_CHAIN_ID, USDC_BASE_SEPOLIA
    # Default to mainnet
    return BASE_CHAIN_ID, USDC_BASE


def get_usdc_domain_name(network: str) -> str:
    """
    Get the EIP-712 domain name for USDC on a given network.

    Mainnet USDC uses "USD Coin", testnet USDC uses "USDC".

    Args:
        network: Network identifier in EIP-155 format

    Returns:
        The EIP-712 domain name for signing
    """
    if network == "eip155:84532" or network == "base-sepolia":
        return "USDC"
    return "USD Coin"


def create_nonce() -> str:
    """Generate a random bytes32 nonce."""
    return "0x" + secrets.token_hex(32)


def create_payment_payload(
    account: Account,
    recipient: str,
    amount: str,  # In micro USDC (6 decimals)
    network: str = "eip155:8453",
    resource_url: str = "https://blockrun.ai/api/v1/chat/completions",
    resource_description: str = "BlockRun AI API call",
    max_timeout_seconds: int = 300,
    extra: Optional[Dict[str, str]] = None,
    extensions: Optional[Dict[str, Any]] = None,
    asset: Optional[str] = None,
) -> str:
    """
    Create a signed x402 v2 payment payload.

    This uses EIP-712 typed data signing to create a payment authorization
    that the CDP facilitator can verify and settle.

    Args:
        account: eth-account Account instance
        recipient: Payment recipient address (checksummed)
        amount: Amount in micro USDC (6 decimals, e.g., "1000" = $0.001)
        network: Network identifier (e.g., "eip155:8453" for Base mainnet, "eip155:84532" for Base Sepolia)
        resource_url: URL of the resource being accessed
        resource_description: Description of the resource
        max_timeout_seconds: Max timeout for the payment (default: 300)
        extra: Extra info for USDC domain (name, version)
        asset: USDC contract address (optional, derived from network if not provided)

    Returns:
        Base64-encoded signed payment payload
    """
    # Current timestamp
    now = int(time.time())
    valid_after = now - 600  # 10 minutes before (allows for clock skew)
    valid_before = now + max_timeout_seconds

    # Generate random nonce
    nonce = create_nonce()

    # Get chain config based on network
    chain_id, default_usdc = get_chain_config(network)

    # Use provided asset address or default for the network
    usdc_address = asset or default_usdc

    # EIP-712 domain for USDC (mainnet or testnet based on network)
    default_domain_name = get_usdc_domain_name(network)
    domain = {
        "name": extra.get("name", default_domain_name) if extra else default_domain_name,
        "version": extra.get("version", "2") if extra else "2",
        "chainId": chain_id,
        "verifyingContract": usdc_address,
    }

    # EIP-712 types for TransferWithAuthorization
    types = {
        "TransferWithAuthorization": [
            {"name": "from", "type": "address"},
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "validAfter", "type": "uint256"},
            {"name": "validBefore", "type": "uint256"},
            {"name": "nonce", "type": "bytes32"},
        ],
    }

    # Message to sign
    message = {
        "from": account.address,
        "to": recipient,
        "value": int(amount),
        "validAfter": valid_after,
        "validBefore": valid_before,
        "nonce": bytes.fromhex(nonce[2:]),  # Remove 0x prefix
    }

    # Sign using EIP-712
    signable = encode_typed_data(domain, types, message)
    signed = account.sign_message(signable)

    # Create x402 v2 payment payload
    payment_data = {
        "x402Version": 2,
        "resource": {
            "url": resource_url,
            "description": resource_description,
            "mimeType": "application/json",
        },
        "accepted": {
            "scheme": "exact",
            "network": network,
            "amount": amount,
            "asset": usdc_address,
            "payTo": recipient,
            "maxTimeoutSeconds": max_timeout_seconds,
            "extra": extra or {"name": default_domain_name, "version": "2"},
        },
        "payload": {
            "signature": (
                "0x" + signed.signature.hex()
                if not signed.signature.hex().startswith("0x")
                else signed.signature.hex()
            ),
            "authorization": {
                "from": account.address,
                "to": recipient,
                "value": amount,
                "validAfter": str(valid_after),
                "validBefore": str(valid_before),
                "nonce": nonce,
            },
        },
        "extensions": extensions or {},
    }

    # Encode as base64
    return base64.b64encode(json.dumps(payment_data).encode()).decode()


def parse_payment_required(header_value: str) -> Dict[str, Any]:
    """
    Parse the X-Payment-Required header from a 402 response.

    Args:
        header_value: Base64-encoded payment requirements

    Returns:
        Decoded payment requirements dict
    """
    try:
        decoded = base64.b64decode(header_value)
        return json.loads(decoded)
    except Exception:
        # Don't expose internal error details
        raise ValueError("Failed to parse payment required header: invalid format")


def extract_payment_details(payment_required: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract payment details from parsed payment required response.

    Supports both v1 and v2 formats.

    Args:
        payment_required: Parsed payment required dict

    Returns:
        Dict with amount, recipient, network, asset, and extra info
    """
    accepts = payment_required.get("accepts", [])
    if not accepts:
        raise ValueError("No payment options in payment required response")

    # Take the first option
    option = accepts[0]

    # Support both v1 (maxAmountRequired) and v2 (amount) formats
    amount = option.get("amount") or option.get("maxAmountRequired")
    if not amount:
        raise ValueError("No amount found in payment requirements")

    return {
        "amount": amount,
        "recipient": option.get("payTo"),
        "network": option.get("network"),
        "asset": option.get("asset"),
        "scheme": option.get("scheme"),
        "maxTimeoutSeconds": option.get("maxTimeoutSeconds", 300),
        "extra": option.get("extra"),
        "resource": payment_required.get("resource"),
    }


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
        from solders.signature import Signature  # type: ignore
        import base58  # type: ignore
    except ImportError:
        raise ImportError(
            "Solana payment requires 'solders' and 'base58'. "
            "Install with: pip install blockrun-llm[solana]"
        )

    # Load keypair from first 32 bytes (seed)
    secret = base58.b58decode(private_key)
    keypair = Keypair.from_seed(secret[:32])
    owner_pubkey = keypair.pubkey()

    # Derive ATAs
    source_ata = _get_ata(str(owner_pubkey), USDC_SOLANA)
    dest_ata = _get_ata(recipient, USDC_SOLANA)

    # Get latest blockhash
    blockhash = _get_latest_blockhash(rpc_url)

    # Build compute budget instructions
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

    # Partial sign: user signs, fee_payer (CDP) co-signs on server side
    # fee_payer is always first signer (index 0), owner is second (index 1)
    msg_bytes = bytes(message)
    user_sig = keypair.sign_message(msg_bytes)
    null_sig = Signature.default()  # placeholder for fee_payer
    tx = VersionedTransaction.populate(message, [null_sig, user_sig])

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
