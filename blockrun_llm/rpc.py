"""
BlockRun RPC Client - Multi-chain JSON-RPC (Tatum gateway) via x402 micropayments.

One endpoint, 40+ chains: Ethereum, Base, Solana, Polygon, BSC, Arbitrum,
Optimism, Avalanche, Bitcoin, Sui, and more. Standard JSON-RPC 2.0
passthrough — no API key, pay-per-call in USDC.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator

Usage:
    from blockrun_llm import RpcClient

    client = RpcClient()  # Uses BLOCKRUN_WALLET_KEY from env

    # EVM chains speak eth_* JSON-RPC
    block = client.call("ethereum", "eth_blockNumber")
    print(block.result)  # e.g. "0x1499f7c"

    balance = client.call(
        "base", "eth_getBalance",
        ["0x4200000000000000000000000000000000000006", "latest"],
    )

    # Non-EVM chains speak their native JSON-RPC
    slot = client.call("solana", "getSlot")

    # Batch: one payment, per-element pricing ($0.002 x N)
    responses = client.batch("polygon", [
        {"method": "eth_blockNumber"},
        {"method": "eth_gasPrice"},
    ])

Pricing:
    Flat $0.002 per JSON-RPC call; a batch charges per element.

Networks:
    40 curated chains (see SUPPORTED_NETWORKS) plus common aliases
    (eth, arb, op, matic, bnb, avax, sol, btc, xrp, dot, ...). Unknown but
    well-formed slugs fall through to a generic `{slug}-mainnet` gateway
    attempt, so new Tatum chains work without an SDK update.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from eth_account import Account

from .apikey import (
    api_key_base_url,
    auth_headers,
    missing_credential_error,
    payment_mode,
    raise_for_api_key_402,
    resolve_api_key,
)
from .tx_log import paid_request_error_prefix
from .types import APIError, PaymentError, RpcResponse, retry_after_of
from .validation import (
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
)
from .x402 import create_payment_payload, extract_payment_details, parse_payment_required

load_dotenv()

# Curated chains accepted by /v1/rpc/{network}. Mirrors backend
# TATUM_RPC_CHAINS (src/lib/tatum.ts, verified live 2026-06-07).
# EVM chains use eth_* JSON-RPC; non-EVM (Solana / UTXO / NEAR / Sui /
# XRP Ledger / Polkadot) speak their own JSON-RPC dialect.
SUPPORTED_NETWORKS = [
    # EVM
    "ethereum",
    "base",
    "arbitrum",
    "arbitrum-nova",
    "optimism",
    "polygon",
    "bsc",
    "avalanche",
    "fantom",
    "cronos",
    "celo",
    "gnosis",
    "zksync",
    "berachain",
    "unichain",
    "monad",
    "chiliz",
    "moonbeam",
    "aurora",
    "flare",
    "oasis",
    "kaia",
    "sonic",
    "xdc",
    "abstract",
    "hyperevm",
    "plume",
    "ronin",
    "rootstock",
    # Non-EVM (JSON-RPC-compatible)
    "solana",
    "bitcoin",
    "litecoin",
    "dogecoin",
    "bitcoin-cash",
    "near",
    "sui",
    "ripple",
    "polkadot",
    "kusama",
    "zcash",
]

# Common short names the gateway also accepts (resolved server-side).
NETWORK_ALIASES = {
    "eth": "ethereum",
    "arb": "arbitrum",
    "arbitrum-one": "arbitrum",
    "arb-one": "arbitrum",
    "arb-nova": "arbitrum-nova",
    "op": "optimism",
    "matic": "polygon",
    "pol": "polygon",
    "bnb": "bsc",
    "binance": "bsc",
    "binance-smart-chain": "bsc",
    "avax": "avalanche",
    "ftm": "fantom",
    "bera": "berachain",
    "klaytn": "kaia",
    "chz": "chiliz",
    "hyperliquid": "hyperevm",
    "rsk": "rootstock",
    "sol": "solana",
    "btc": "bitcoin",
    "ltc": "litecoin",
    "doge": "dogecoin",
    "bch": "bitcoin-cash",
    "xrp": "ripple",
    "xrpl": "ripple",
    "dot": "polkadot",
    "zec": "zcash",
}

# Flat price per JSON-RPC call (batch = N x this). Informational only —
# the actual quote always comes from the 402 challenge.
RPC_PRICE_USD = 0.002


class RpcClient:
    """
    BlockRun Multi-chain RPC Client.

    Standard JSON-RPC 2.0 access to 40+ chains through BlockRun's Tatum
    gateway with automatic x402 micropayments on Base chain.

    Flat $0.002 per call; a JSON-RPC batch charges per element.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_TIMEOUT = 60.0  # upstream gateway timeout is 20s

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the BlockRun RPC client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 60)
        """
        from .wallet import load_wallet

        # Account rail first, and before any wallet variable is read: an API
        # key is a complete credential on its own, so demanding a private key
        # alongside it would make every API-key user invent a wallet they
        # never use. See apikey.py for the precedence rule.
        api_key = resolve_api_key(private_key)
        key = (
            None
            if api_key
            else (
                private_key
                or os.environ.get("BLOCKRUN_WALLET_KEY")
                or os.environ.get("BASE_CHAIN_WALLET_KEY")
                or load_wallet()
            )
        )
        if not api_key and not key:
            raise missing_credential_error()

        if key:
            validate_private_key(key)
        self.api_key = api_key
        # No wallet on the account rail: nothing is signed locally.
        self.account = Account.from_key(key) if key else None

        # BLOCKRUN_API_URL names an x402 gateway; an API-key client must not
        # follow it and hand the key to a host set up for another rail.
        api_url_raw = (
            api_key_base_url(api_url)
            if api_key
            else (api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL)
        )
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout
        self._client = httpx.Client(headers=auth_headers(api_key), timeout=timeout)

    def call(
        self,
        network: str,
        method: str,
        params: list[Any] | None = None,
        *,
        id: str | int = 1,
    ) -> RpcResponse:
        """
        Make a single JSON-RPC 2.0 call. Flat $0.002.

        Args:
            network: Chain name (e.g. "ethereum", "base", "solana") or a
                     common alias ("eth", "sol", "matic", ...). See
                     SUPPORTED_NETWORKS / NETWORK_ALIASES.
            method: Chain RPC method, e.g. "eth_blockNumber", "eth_call",
                    "eth_getBalance" (EVM) or "getSlot", "getAccountInfo"
                    (Solana).
            params: Method-specific params array (optional).
            id: JSON-RPC request id (default: 1).

        Returns:
            RpcResponse with `result` (or JSON-RPC `error`), plus
            `network`, `cache_hit` and `tx_hash` metadata.

        Raises:
            PaymentError: If wallet has insufficient balance
            APIError: If the API returns an error

        Example:
            block = client.call("ethereum", "eth_blockNumber")
            print(int(block.result, 16))
        """
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            body["params"] = params

        data, headers = self._request_with_payment(network, body)
        return self._to_response(data, headers)

    def batch(
        self,
        network: str,
        requests: list[dict[str, Any]],
    ) -> list[RpcResponse]:
        """
        Make a JSON-RPC 2.0 batch call. Priced per element ($0.002 x N).

        Args:
            network: Chain name or alias (see call()).
            requests: List of dicts each with a "method" key and optional
                      "params" / "id". "jsonrpc" and missing ids are
                      filled in automatically.

        Returns:
            List of RpcResponse, in upstream order.

        Example:
            out = client.batch("base", [
                {"method": "eth_blockNumber"},
                {"method": "eth_gasPrice"},
            ])
        """
        if not requests:
            raise ValueError("batch requires at least one request")
        body = []
        for i, req in enumerate(requests):
            if "method" not in req:
                raise ValueError(f"batch request {i} is missing 'method'")
            entry = {"jsonrpc": "2.0", "id": i + 1, **req}
            body.append(entry)

        data, headers = self._request_with_payment(network, body)
        if not isinstance(data, list):
            # Upstream collapsed the batch (shouldn't happen) — wrap it.
            data = [data]
        return [self._to_response(item, headers) for item in data]

    @staticmethod
    def _to_response(data: Any, headers: httpx.Headers) -> RpcResponse:
        if not isinstance(data, dict):
            data = {"result": data}
        return RpcResponse(
            **data,
            network=headers.get("x-network"),
            cache_hit=headers.get("x-cache", "").upper() == "HIT",
            tx_hash=headers.get("x-payment-receipt"),
        )

    def _request_with_payment(
        self, network: str, body: dict[str, Any] | list[dict[str, Any]]
    ) -> tuple:
        """POST the JSON-RPC body with automatic x402 payment handling."""
        endpoint = f"/v1/rpc/{network}"
        url = f"{self.api_url}{endpoint}"

        response = self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 402:
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(response, self.api_key)
            return self._handle_payment_and_retry(url, endpoint, body, response)

        if response.status_code != 200:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
                retry_after=retry_after_of(response),
            )

        return response.json(), response.headers

    def _handle_payment_and_retry(
        self,
        url: str,
        endpoint: str,
        body: dict[str, Any] | list[dict[str, Any]],
        response: httpx.Response,
    ) -> tuple:
        """Handle 402 response: parse requirements, sign payment, retry."""
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                resp_body = response.json()
                if "x402" in resp_body:
                    payment_header = resp_body
            except Exception:
                pass

        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        if isinstance(payment_header, str):
            payment_required = parse_payment_required(payment_header)
        else:
            payment_required = payment_header

        details = extract_payment_details(payment_required)
        resource = details.get("resource") or {}
        extensions = payment_required.get("extensions", {})

        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=resource.get("url", f"{self.api_url}{endpoint}"),
            resource_description=resource.get("description", "BlockRun Multi-chain RPC"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
        )

        retry_response = self._client.post(
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "PAYMENT-SIGNATURE": payment_payload,
            },
        )

        if retry_response.status_code == 402:
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(retry_response, self.api_key)
            raise PaymentError("Payment was rejected. Check your wallet balance.")

        if retry_response.status_code != 200:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"{paid_request_error_prefix(retry_response.headers)}: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
                retry_after=retry_after_of(retry_response),
            )

        return retry_response.json(), retry_response.headers

    @property
    def payment_mode(self) -> str:
        """Which rail this client pays on: ``"apikey"`` or ``"wallet"``.

        Worth checking once at startup when both a key and a wallet are
        configured in the environment: it is the difference between
        spending credit and spending USDC."""
        return payment_mode(self)

    def get_wallet_address(self) -> str:
        # No address on the account rail: payment comes from prepaid
        # credit, so there is nothing to return but the empty string.
        if self.api_key:
            return ""
        """Get the wallet address being used for payments."""
        return self.account.address

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
