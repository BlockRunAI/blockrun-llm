"""
BlockRun Music Client - Generate music tracks via x402 micropayments.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator

Usage:
    from blockrun_llm import MusicClient

    client = MusicClient()  # Uses BLOCKRUN_WALLET_KEY from env

    # Generate an instrumental track
    result = client.generate("upbeat synthwave with neon pads")
    print(result.data[0].url)  # CDN URL — download within 24h

    # With lyrics
    result = client.generate(
        "upbeat pop song",
        instrumental=False,
        lyrics="Hello world, this is my song...",
    )

Pricing: $0.1575/track
Note: Generated URLs expire in ~24h — download immediately if needed.
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
from .types import APIError, MusicResponse, PaymentError
from .validation import (
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
)
from .x402 import create_payment_payload, extract_payment_details, parse_payment_required

load_dotenv()


class MusicClient:
    """
    BlockRun Music Generation Client.

    Generate full-length ~3 minute music tracks using MiniMax Music 2.5+
    with automatic x402 micropayments on Base chain.

    Pricing: $0.1575/track
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MODEL = "minimax/music-2.5+"
    DEFAULT_TIMEOUT = 210.0  # music gen takes 1-3 min

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 210.0,
    ):
        """
        Initialize the BlockRun Music client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 210 for music generation)
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

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        instrumental: bool = True,
        lyrics: str | None = None,
    ) -> MusicResponse:
        """
        Generate a music track from a text prompt.

        Takes 1-3 minutes. Returns a CDN URL valid for ~24h.

        Args:
            prompt: Music style, mood, or description.
                    E.g. "upbeat synthwave with neon pads", "chill lo-fi beats",
                    "epic orchestral film score"
            model: Model ID (default: "minimax/music-2.5+")
                   Options: "minimax/music-2.5+", "minimax/music-2.5"
            instrumental: Generate without vocals (default: True)
            lyrics: Custom lyrics — cannot be used with instrumental=True

        Returns:
            MusicResponse with track URL, duration, and optional lyrics

        Raises:
            ValueError: If both instrumental=True and lyrics are provided
            PaymentError: If wallet has insufficient balance
            APIError: If the API returns an error

        Example:
            result = client.generate("chill lo-fi beats with piano")
            print(result.data[0].url)  # Download this — expires in 24h

        Example with lyrics:
            result = client.generate(
                "upbeat pop", instrumental=False,
                lyrics="Hello world, this is my song..."
            )
        """
        if instrumental and lyrics and lyrics.strip():
            raise ValueError("Cannot specify lyrics when instrumental is True")

        body: dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "prompt": prompt,
            "instrumental": instrumental,
        }
        if lyrics and lyrics.strip():
            body["lyrics"] = lyrics.strip()

        return self._request_with_payment("/v1/audio/generations", body)

    def _request_with_payment(self, endpoint: str, body: dict[str, Any]) -> MusicResponse:
        """Make a request with automatic x402 payment handling."""
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
            return self._handle_payment_and_retry(url, body, response)

        if response.status_code != 200:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return MusicResponse(**response.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        body: dict[str, Any],
        response: httpx.Response,
    ) -> MusicResponse:
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
            resource_url=resource.get("url", f"{self.api_url}/v1/audio/generations"),
            resource_description=resource.get("description", "BlockRun Music Generation"),
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
            )

        data = retry_response.json()
        # Attach tx hash from response header
        tx_hash = retry_response.headers.get("x-payment-receipt") or retry_response.headers.get(
            "X-Payment-Receipt"
        )
        if tx_hash:
            data["txHash"] = tx_hash

        return MusicResponse(**data)

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
