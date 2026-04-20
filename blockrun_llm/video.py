"""
BlockRun Video Client - Generate short AI videos via x402 micropayments.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator

Usage:
    from blockrun_llm import VideoClient

    client = VideoClient()  # Uses BLOCKRUN_WALLET_KEY from env

    # Text-to-video
    result = client.generate("a red apple slowly spinning on a wooden table")
    print(result.data[0].url)  # permanent blockrun-hosted MP4 URL
    print(result.data[0].duration_seconds)  # 8

    # Image-to-video
    result = client.generate(
        "the subject turns its head and smiles",
        image_url="https://example.com/portrait.jpg",
    )

Pricing: $0.05/second (xAI Grok Imagine Video). 8-second default -> $0.42 billed.
Generation takes ~30-120s end-to-end; the client blocks until the video is ready
because the BlockRun gateway handles polling + GCS backup internally.
"""

import os
from typing import Optional, Dict, Any
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import VideoResponse, APIError, PaymentError
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
)

load_dotenv()


class VideoClient:
    """
    BlockRun Video Generation Client.

    Generates 8-second MP4 clips using xAI's Grok Imagine Video
    with automatic x402 micropayments on Base chain.

    Pricing: $0.05/second (default 8s -> $0.42/clip with margin).
    Generated URLs are permanent (mirrored to BlockRun storage).
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MODEL = "xai/grok-imagine-video"
    DEFAULT_TIMEOUT = 300.0  # video gen + polling can take up to 3 min

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the BlockRun Video client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 300 for video generation)
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
        image_url: Optional[str] = None,
        duration_seconds: Optional[int] = None,
    ) -> VideoResponse:
        """
        Generate a video clip from a text prompt (or text + image).

        Blocks until the video is ready (30-120s typical). Returns a permanent URL
        pointing to BlockRun's mirrored copy of the clip.

        Args:
            prompt: Text description of the video.
            model: Model ID (default: "xai/grok-imagine-video")
            image_url: Optional seed image URL for image-to-video.
            duration_seconds: Duration to bill for (defaults to model's default — 8s for grok-imagine-video).

        Returns:
            VideoResponse with the clip URL, duration, and upstream request_id.

        Raises:
            PaymentError: If wallet has insufficient balance.
            APIError: If the API returns an error (content policy, rate limit, etc.).

        Example:
            result = client.generate("a hummingbird hovering near a red flower")
            print(result.data[0].url)  # permanent MP4 URL
        """
        body: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "prompt": prompt,
        }
        if image_url:
            body["image_url"] = image_url
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds

        return self._request_with_payment("/v1/videos/generations", body)

    def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> VideoResponse:
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

        return VideoResponse(**response.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> VideoResponse:
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
            resource_url=resource.get("url", f"{self.api_url}/v1/videos/generations"),
            resource_description=resource.get("description", "BlockRun Video Generation"),
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
        tx_hash = retry_response.headers.get("x-payment-receipt") or retry_response.headers.get(
            "X-Payment-Receipt"
        )
        if tx_hash:
            data["txHash"] = tx_hash

        return VideoResponse(**data)

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
