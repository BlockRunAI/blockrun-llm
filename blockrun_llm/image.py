"""
BlockRun Image Client - Generate images via x402 micropayments.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator
4. Your actual private key is NEVER transmitted to any server

This is the same security model as signing any blockchain transaction.

Usage:
    from blockrun_llm import ImageClient

    # Initialize with private key from env (BLOCKRUN_WALLET_KEY)
    client = ImageClient()

    # Generate an image
    result = client.generate("A cute cat wearing a space helmet")
    print(result.data[0].url)

    # With specific model
    result = client.generate("prompt", model="google/nano-banana-pro")
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from eth_account import Account

from .tx_log import paid_request_error_prefix
from .types import APIError, ImageResponse, PaymentError
from .validation import (
    build_payment_rejected_error,
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
    validate_resource_url,
)
from .x402 import create_payment_payload, extract_payment_details, parse_payment_required

# Load environment variables
load_dotenv()


class ImageClient:
    """
    BlockRun Image Generation Client.

    Generate images using Nano Banana (Google Gemini), DALL-E 3,
    GPT Image 1, or CogView-4 (Zhipu AI)
    with automatic x402 micropayments on Base chain.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MODEL = "google/nano-banana"
    DEFAULT_SIZE = "1024x1024"

    # Image generation slow-path polling. Models like ``openai/gpt-image-2``
    # and ``openai/dall-e-3`` routinely exceed the gateway's 30s inline
    # window and come back as ``202 + poll_url`` instead of the finished
    # image. The client replays the same PAYMENT-SIGNATURE on every poll;
    # settlement only happens on the first completed poll, so giving up
    # before then costs the caller nothing.
    IMAGE_POLL_INTERVAL_SECONDS = 5.0
    IMAGE_POLL_BUDGET_SECONDS = 300.0

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 200.0,  # gpt-image-2 at >=1536px can take ~180s server-side; 200s gives buffer
    ):
        """
        Initialize the BlockRun Image client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 200 for images — gpt-image-2 at large sizes can run ~180s server-side)

        Raises:
            ValueError: If no private key is provided or found in env
        """
        # Get private key from param, environment, or ~/.blockrun/.session file
        from .wallet import load_wallet

        key = (
            private_key
            or os.environ.get("BLOCKRUN_WALLET_KEY")
            or os.environ.get("BASE_CHAIN_WALLET_KEY")
            or load_wallet()  # Loads from ~/.blockrun/.session
        )
        if not key:
            raise ValueError(
                "Private key required. Either:\n"
                "  1. Pass private_key parameter\n"
                "  2. Set BLOCKRUN_WALLET_KEY environment variable\n"
                "  3. Place key in ~/.blockrun/.session\n"
                "NOTE: Your key never leaves your machine - only signatures are sent."
            )

        # Validate private key format
        validate_private_key(key)

        # Initialize wallet account (key stays local, never transmitted)
        self.account = Account.from_key(key)

        # Validate and set API URL
        api_url_raw = api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout

        # HTTP client
        self._client = httpx.Client(timeout=timeout)

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        n: int = 1,
        **kwargs: Any,
    ) -> ImageResponse:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            model: Model ID (default: "google/nano-banana")
                   Options: "google/nano-banana", "google/nano-banana-pro",
                            "openai/dall-e-3", "openai/gpt-image-1",
                            "openai/gpt-image-2", "zai/cogview-4",
                            "xai/grok-imagine-image", "xai/grok-imagine-image-pro",
                            "black-forest/flux-1.1-pro"
            size: Image size (default: "1024x1024")
            n: Number of images to generate (default: 1)

        Returns:
            ImageResponse with generated image URLs

        Example:
            result = client.generate("A sunset over mountains")
            print(result.data[0].url)  # Image URL or data URL

        Raises:
            TypeError: If unexpected keyword arguments are passed
        """
        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            hint = (
                " `quality` is Solana-only (SolanaLLMClient.image) — the Base gateway "
                "has no such field and would silently ignore it."
                if "quality" in kwargs
                else ""
            )
            raise TypeError(
                f"generate() got unexpected keyword argument(s): {unsupported}. "
                f"Valid parameters are: prompt, model, size, n.{hint}"
            )

        # Build request body
        body: dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "prompt": prompt,
            "size": size or self.DEFAULT_SIZE,
            "n": n,
        }

        # Make request (with automatic payment handling)
        return self._request_with_payment("/v1/images/generations", body)

    def edit(
        self,
        prompt: str,
        image: str | list[str],
        *,
        model: str | None = None,
        mask: str | None = None,
        size: str | None = None,
        n: int = 1,
        **kwargs: Any,
    ) -> ImageResponse:
        """
        Edit an image using img2img, or fuse multiple source images.

        Args:
            prompt: Text description of the desired edit
            image: A single base64 "data:image/...;base64,..." data URI, or a
                   list of 1-4 such data URIs to fuse multiple sources (e.g. a
                   reference photo + a brand logo). Plain URLs are not accepted —
                   the source must be a data URI.
            model: Model ID (default: "openai/gpt-image-2")
                   Edit-supported: "openai/gpt-image-1", "openai/gpt-image-2",
                                   "google/nano-banana", "google/nano-banana-pro"
                   Multi-image caps: openai/* up to 4, google/* up to 3.
            mask: Optional base64-encoded mask image (OpenAI gpt-image-* only;
                  cannot be combined with multiple source images).
            size: Image size (default: "1024x1024")
            n: Number of images to generate (default: 1)

        Returns:
            ImageResponse with edited image URLs

        Example:
            # Single-image edit
            result = client.edit(
                "Make the sky purple",
                image="data:image/png;base64,..."
            )

            # Multi-image fusion (Nano Banana)
            result = client.edit(
                "Place the logo on the t-shirt",
                image=["data:image/png;base64,...", "data:image/png;base64,..."],
                model="google/nano-banana",
            )
            print(result.data[0].url)

        Raises:
            TypeError: If unexpected keyword arguments are passed
        """
        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            hint = (
                " `quality` is Solana-only (SolanaLLMClient.image_edit) — the Base "
                "gateway has no such field and would silently ignore it."
                if "quality" in kwargs
                else ""
            )
            raise TypeError(
                f"edit() got unexpected keyword argument(s): {unsupported}. "
                f"Valid parameters are: prompt, image, model, mask, size, n.{hint}"
            )

        body: dict[str, Any] = {
            "model": model or "openai/gpt-image-2",
            "prompt": prompt,
            "image": image,
            "size": size or self.DEFAULT_SIZE,
            "n": n,
        }
        if mask is not None:
            body["mask"] = mask

        return self._request_with_payment("/v1/images/image2image", body)

    def _request_with_payment(self, endpoint: str, body: dict[str, Any]) -> ImageResponse:
        """
        Make a request with automatic x402 payment handling.

        1. Send initial request
        2. If 402, parse payment requirements
        3. Sign payment locally
        4. Retry with X-Payment header
        """
        url = f"{self.api_url}{endpoint}"

        # First attempt (will likely return 402)
        response = self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        # Handle 402 Payment Required
        if response.status_code == 402:
            return self._handle_payment_and_retry(url, body, response)

        # Handle other errors
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

        # Parse successful response
        return ImageResponse(**response.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        body: dict[str, Any],
        response: httpx.Response,
    ) -> ImageResponse:
        """Handle 402 response: parse requirements, sign payment, retry."""
        # Get payment required header (x402 library uses lowercase)
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            # Try to get from response body
            try:
                resp_body = response.json()
                if "x402" in resp_body:
                    payment_header = resp_body
            except Exception:
                pass

        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        # Parse payment requirements
        if isinstance(payment_header, str):
            payment_required = parse_payment_required(payment_header)
        else:
            payment_required = payment_header

        # Extract payment details
        details = extract_payment_details(payment_required)

        # Create signed payment payload (v2 format)
        resource = details.get("resource") or {}
        # Pass through extensions from server (for Bazaar discovery)
        extensions = payment_required.get("extensions", {})
        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=validate_resource_url(
                resource.get("url", f"{self.api_url}/v1/images/generations"), self.api_url
            ),
            resource_description=resource.get("description", "BlockRun Image Generation"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
        )

        # Retry with payment (x402 library expects PAYMENT-SIGNATURE header)
        retry_response = self._client.post(
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "PAYMENT-SIGNATURE": payment_payload,
            },
        )

        # Check for errors
        if retry_response.status_code == 402:
            raise build_payment_rejected_error(retry_response)

        if retry_response.status_code == 200:
            return ImageResponse(**retry_response.json())

        if retry_response.status_code == 202:
            # Slow-path async flow — gateway returned a job stub with a
            # poll_url. Replay the same signature on each poll until the
            # upstream finishes; settlement happens on the completed poll.
            return self._poll_until_completed(retry_response, payment_payload)

        try:
            error_body = retry_response.json()
        except Exception:
            error_body = {"error": "Request failed"}
        raise APIError(
            f"{paid_request_error_prefix(retry_response.headers)}: {retry_response.status_code}",
            retry_response.status_code,
            sanitize_error_response(error_body),
        )

    def _absolute_url(self, url: str) -> str:
        """Resolve a relative ``poll_url`` against the configured API host.

        Server-returned poll URLs look like ``/api/v1/images/generations/<id>``;
        our ``self.api_url`` already ends with ``/api`` so we strip it once
        to avoid double-prefixing.
        """
        if url.startswith(("http://", "https://")):
            return url
        base = self.api_url.removesuffix("/api")
        return f"{base}{url}"

    def _poll_until_completed(
        self,
        submit_resp: httpx.Response,
        payment_payload: str,
    ) -> ImageResponse:
        """Poll the gateway's ``poll_url`` with the same PAYMENT-SIGNATURE
        until the upstream returns the finished image.

        Settlement happens on the first ``status=completed`` poll, so
        timeout = no spend. Returns the parsed :class:`ImageResponse`.
        """
        import time as _time

        try:
            submit_data = submit_resp.json()
        except Exception:
            submit_data = {}

        poll_url_rel = submit_data.get("poll_url")
        job_id = submit_data.get("id")
        if not poll_url_rel:
            raise APIError(
                "Slow-path 202 missing poll_url",
                202,
                {"response": submit_data},
            )

        poll_url = self._absolute_url(poll_url_rel)
        poll_headers = {"PAYMENT-SIGNATURE": payment_payload}
        deadline = _time.monotonic() + self.IMAGE_POLL_BUDGET_SECONDS
        last_status = submit_data.get("status", "queued")

        while _time.monotonic() < deadline:
            _time.sleep(self.IMAGE_POLL_INTERVAL_SECONDS)

            poll_resp = self._client.get(poll_url, headers=poll_headers)
            try:
                poll_data = poll_resp.json()
            except Exception:
                poll_data = {}
            last_status = poll_data.get("status", last_status)

            if poll_resp.status_code == 402:
                # Settlement failed on this poll — surface the gateway reason.
                raise build_payment_rejected_error(poll_resp)

            if last_status == "failed":
                raise APIError(
                    f"Image generation failed upstream: {poll_data.get('error', 'unknown')}",
                    poll_resp.status_code,
                    sanitize_error_response(poll_data if isinstance(poll_data, dict) else {}),
                )

            if poll_resp.status_code == 200 and last_status == "completed":
                return ImageResponse(**poll_data)

            if poll_resp.status_code in (202, 504):
                # 202 = still queued/in_progress; 504 = transient upstream
                # hiccup. Both retriable inside the budget.
                continue

            if poll_resp.status_code != 200:
                try:
                    error_body = poll_resp.json()
                except Exception:
                    error_body = {"error": "Request failed"}
                raise APIError(
                    f"Image poll failed: HTTP {poll_resp.status_code}",
                    poll_resp.status_code,
                    sanitize_error_response(error_body),
                )

        raise APIError(
            (
                f"Image generation did not complete within "
                f"{self.IMAGE_POLL_BUDGET_SECONDS:.0f}s "
                f"(last status: {last_status}). Settlement only happens on "
                "completion, so no payment was taken."
            ),
            504,
            {"id": job_id, "last_status": last_status},
        )

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
