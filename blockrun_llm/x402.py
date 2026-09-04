"""
x402 Payment Protocol v2 Implementation for BlockRun.

This module handles creating signed payment payloads for the x402 v2 protocol.
The private key is used ONLY for local signing and NEVER leaves the client.
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from typing import Any

from eth_account.messages import encode_typed_data
from eth_account.signers.local import LocalAccount

# Chain and token constants for mainnet
BASE_CHAIN_ID = 8453
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# Chain and token constants for testnet (Base Sepolia)
BASE_SEPOLIA_CHAIN_ID = 84532
USDC_BASE_SEPOLIA = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"


# BlockRun's x402 builder code — the ERC-8021 Schema 2 service code (`s`) that
# tags every payment this SDK signs as BlockRun-originated for on-chain
# attribution. See https://docs.cdp.coinbase.com/x402/core-concepts/builder-codes
BLOCKRUN_SERVICE_CODE = "blockrun"


def with_builder_code_service_code(
    extensions: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge BlockRun's service code (``s``) into the payload's ``builder-code``
    extension, preserving any app code (``a``) the server echoed back in its 402.

    The CDP facilitator reads ``builder-code.info.s`` and encodes it into the
    settlement calldata suffix — no CBOR/encoding happens client-side.
    """
    merged: dict[str, Any] = dict(extensions or {})
    existing = dict(merged.get("builder-code") or {})
    info = dict(existing.get("info") or {})
    info["s"] = [BLOCKRUN_SERVICE_CODE]
    existing["info"] = info
    merged["builder-code"] = existing
    return merged


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
    account: LocalAccount,
    recipient: str,
    amount: str,  # In micro USDC (6 decimals)
    network: str = "eip155:8453",
    resource_url: str = "https://blockrun.ai/api/v1/chat/completions",
    resource_description: str = "BlockRun AI API call",
    max_timeout_seconds: int = 300,
    extra: dict[str, str] | None = None,
    extensions: dict[str, Any] | None = None,
    asset: str | None = None,
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
        "extensions": with_builder_code_service_code(extensions),
    }

    # Encode as base64
    return base64.b64encode(json.dumps(payment_data).encode()).decode()


def parse_payment_required(header_value: str) -> dict[str, Any]:
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


def extract_payment_details(payment_required: dict[str, Any]) -> dict[str, Any]:
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
# Solana x402 Payment — delegated to official x402 SDK
# ============================================================
# The Solana payment implementation has been replaced by the
# official x402 Python SDK (pip install x402[svm]).
# See solana_client.py for usage.


def is_solana_network(network: str) -> bool:
    """Check if a network string represents Solana."""
    return network.startswith("solana:")
