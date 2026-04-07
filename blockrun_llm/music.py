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

import os
from typing import Optional, Dict, Any
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import MusicResponse, APIError, PaymentError
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
)

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
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
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

        key = (
            private_key
            or os.environ.get("BLOCKRUN_WALLET_KEY")
            or os.environ.get("BASE_CHAIN_WALLET_KEY")
            or load_wallet()
        )
        if not key:
            raise ValueError(
                "Private key required. Either:\n"
                "  1. Pass private_key parameter\n"
                "  2. Set BLOCKRUN_WALLET_KEY environment variable\n"
                "  3. Place key in ~/.blockrun/.session\n"
                "NOTE: Your key never leaves your machine - only signatures are sent."
            )

        validate_private_key(key)
        self.account = Account.from_key(key)

        api_url_raw = api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        instrumental: bool = True,
        lyrics: Optional[str] = None,
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

        body: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "prompt": prompt,
            "instrumental": instrumental,
        }
        if lyrics and lyrics.strip():
            body["lyrics"] = lyrics.strip()

        return self._request_with_payment("/v1/audio/generations", body)

    def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> MusicResponse:
        """Make a request with automatic x402 payment handling."""
        url = f"{self.api_url}{endpoint}"

        response = self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 402:
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
        body: Dict[str, Any],
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
            raise PaymentError("Payment was rejected. Check your wallet balance.")

        if retry_response.status_code != 200:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        data = retry_response.json()
        # Attach tx hash from response header
        tx_hash = retry_response.headers.get("x-payment-receipt") or retry_response.headers.get("X-Payment-Receipt")
        if tx_hash:
            data["txHash"] = tx_hash

        return MusicResponse(**data)

    def get_wallet_address(self) -> str:
        """Get the wallet address being used for payments."""
        return self.account.address

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
