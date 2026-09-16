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

from eth_account import Account
from eth_account.messages import encode_typed_data

# Chain and token constants for mainnet
BASE_CHAIN_ID = 8453
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# Chain and token constants for testnet (Base Sepolia)
BASE_SEPOLIA_CHAIN_ID = 84532
USDC_BASE_SEPOLIA = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"

# Circle's Arc (arc.blockrun.ai). USDC is the chain's native token, exposed as
# the ERC-20 at 0x3600…0000; its EIP-712 domain name is "USDC", not Base's
# "USD Coin".
ARC_CHAIN_ID = 5042
USDC_ARC = "0x3600000000000000000000000000000000000000"

# The EVM networks a BlockRun gateway settles on, keyed by the CAIP-2 `network`
# a 402 carries, each with the SDK's OWN chain id, USDC address and EIP-712
# domain. The 402 SELECTS a network from this table; it never supplies the
# domain — until the Arc release the table fell back to Base for any network
# it did not know and took `asset` and `extra` from the 402 as given, which on
# arc.blockrun.ai signed chainId 8453 against Arc's contract: an invalid
# signature, a 401 from the facilitator, after the SDK had reported a payment.
EVM_NETWORKS: dict[str, dict] = {
    "eip155:8453": {
        "name": "Base",
        "chain_id": BASE_CHAIN_ID,
        "usdc": USDC_BASE,
        "domain": {
            "name": "USD Coin",
            "version": "2",
            "chainId": BASE_CHAIN_ID,
            "verifyingContract": USDC_BASE,
        },
    },
    "eip155:5042": {
        "name": "Arc",
        "chain_id": ARC_CHAIN_ID,
        "usdc": USDC_ARC,
        "domain": {
            "name": "USDC",
            "version": "2",
            "chainId": ARC_CHAIN_ID,
            "verifyingContract": USDC_ARC,
        },
    },
    "eip155:84532": {
        "name": "Base Sepolia",
        "chain_id": BASE_SEPOLIA_CHAIN_ID,
        "usdc": USDC_BASE_SEPOLIA,
        "domain": {
            "name": "USDC",
            "version": "2",
            "chainId": BASE_SEPOLIA_CHAIN_ID,
            "verifyingContract": USDC_BASE_SEPOLIA,
        },
    },
}
# The pre-CAIP alias this SDK accepted for the testnet.
_NETWORK_ALIASES = {"base-sepolia": "eip155:84532"}


def evm_network(network: str) -> dict:
    """The table entry for a 402's `network`, or a ValueError naming what IS supported."""
    net = EVM_NETWORKS.get(_NETWORK_ALIASES.get(network, network))
    if net is None:
        raise ValueError(
            f'Unsupported x402 network "{network}": this SDK signs USDC payments on '
            + ", ".join(EVM_NETWORKS)
        )
    return net


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
    """Chain ID and USDC contract for a network — see EVM_NETWORKS. Raises for an unknown one."""
    net = evm_network(network)
    return net["chain_id"], net["usdc"]


def get_usdc_domain_name(network: str) -> str:
    """The EIP-712 domain name for USDC on a network — "USD Coin" on Base, "USDC" on Arc and Base Sepolia."""
    return evm_network(network)["domain"]["name"]


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
        extra: The 402's `extra`. Accepted for compatibility; the domain comes from EVM_NETWORKS.
        asset: The 402's `asset`. Checked against the network's USDC; a mismatch raises ValueError.

    Returns:
        Base64-encoded signed payment payload
    """
    # Current timestamp
    now = int(time.time())
    valid_after = now - 600  # 10 minutes before (allows for clock skew)
    valid_before = now + max_timeout_seconds

    # Generate random nonce
    nonce = create_nonce()

    # The domain is the SDK's own value for the 402's network — never the 402's
    # `extra` (see EVM_NETWORKS). A 402 naming a network the table lacks, or an
    # asset that is not that network's USDC, is refused rather than signed.
    net = evm_network(network)
    usdc_address = net["usdc"]
    if asset and asset.lower() != usdc_address.lower():
        raise ValueError(
            f"x402 asset mismatch: the 402 asks for {asset} on {network}, "
            f"but this SDK only pays USDC there ({usdc_address})"
        )
    domain = dict(net["domain"])

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
            "extra": {"name": domain["name"], "version": domain["version"]},
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
