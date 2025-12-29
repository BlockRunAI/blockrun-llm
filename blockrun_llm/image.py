"""
BlockRun Image Client - Generate images via x402 micropayments.

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

import os
from typing import Optional, Dict, Any
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import ImageResponse, APIError, PaymentError
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
    validate_resource_url,
)


# Load environment variables
load_dotenv()


class ImageClient:
    """
    BlockRun Image Generation Client.

    Generate images using Nano Banana (Google Gemini) or DALL-E 3
    with automatic x402 micropayments on Base chain.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MODEL = "google/nano-banana"
    DEFAULT_SIZE = "1024x1024"

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 120.0,  # Images take longer to generate
    ):
        """
        Initialize the BlockRun Image client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 120 for images)

        Raises:
            ValueError: If no private key is provided or found in env
        """
        # Get private key from param or environment
        key = private_key or os.environ.get("BLOCKRUN_WALLET_KEY")
        if not key:
            raise ValueError(
                "Private key required. Either pass private_key parameter or set "
                "BLOCKRUN_WALLET_KEY environment variable."
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
        model: Optional[str] = None,
        size: Optional[str] = None,
        n: int = 1,
    ) -> ImageResponse:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            model: Model ID (default: "google/nano-banana")
                   Options: "google/nano-banana", "google/nano-banana-pro",
                            "openai/dall-e-3", "openai/gpt-image-1"
            size: Image size (default: "1024x1024")
            n: Number of images to generate (default: 1)

        Returns:
            ImageResponse with generated image URLs

        Example:
            result = client.generate("A sunset over mountains")
            print(result.data[0].url)  # Image URL or data URL
        """
        # Build request body
        body: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "prompt": prompt,
            "size": size or self.DEFAULT_SIZE,
            "n": n,
        }

        # Make request (with automatic payment handling)
        return self._request_with_payment("/v1/images/generations", body)

    def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> ImageResponse:
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
        body: Dict[str, Any],
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
                resource.get("url", f"{self.api_url}/v1/images/generations"),
                self.api_url
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

        return ImageResponse(**retry_response.json())

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
