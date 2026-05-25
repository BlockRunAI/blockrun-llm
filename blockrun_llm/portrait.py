"""
BlockRun Portrait Client — enroll Virtual Portraits via x402 micropayments.

A Virtual Portrait is an AI-generated character image registered as a
face/character reference asset. After enrollment ($0.50 USDC, one-time,
no KYC), you get back a `ta_xxxxxxxx` asset id that can be passed as
`real_face_asset_id` to `VideoClient.generate()` on Seedance 2.0 or 2.0-fast
to keep the same character across multiple videos.

For a *real* person's likeness, use `RealFaceClient` instead — it enrolls
a real face for $0.01 via a brief on-phone liveness check (no KYC) and
yields a `ta_` id usable the same way. Virtual Portraits are for
AI-generated personas, mascots, avatars, and virtual spokespeople.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. The key is used locally to
sign an EIP-3009 USDC transfer authorization; only the signature is
transmitted in the PAYMENT-SIGNATURE header.

Usage:
    from blockrun_llm import PortraitClient

    client = PortraitClient()  # Uses BLOCKRUN_WALLET_KEY from env

    portrait = client.enroll(
        name="My Spokesperson",
        image_url="https://example.com/character.jpg",
    )
    print(portrait.asset_id)              # ta_abcdef1234567890
    print(portrait.settlement.tx_hash)    # 0x9f3a…

    # List wallet's enrolled portraits (free, no payment)
    listing = client.list_portraits()
    for p in listing.portraits:
        print(p.assetId, p.name)
"""

import os
from typing import Optional, Dict, Any
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import (
    PortraitEnrollment,
    PortraitList,
    APIError,
    PaymentError,
)
from .x402 import (
    create_payment_payload,
    parse_payment_required,
    extract_payment_details,
)
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
    validate_resource_url,
)

load_dotenv()


# Hard limits enforced upstream; mirror locally to fail fast.
_MAX_NAME_LEN = 64


class PortraitClient:
    """
    BlockRun Virtual Portrait Client.

    Wraps `POST /v1/portrait/enroll` ($0.50 USDC, one-time) and the free
    `GET /v1/wallet/<address>/portraits` listing endpoint.

    The enrollment endpoint settles AFTER the portrait is successfully
    registered upstream, so failed enrollments (content filter, network
    error) return 502 with no charge.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    ENROLL_ENDPOINT = "/v1/portrait/enroll"

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY).
            api_url: API endpoint URL (default https://blockrun.ai/api).
            timeout: Per-HTTP-call timeout in seconds.
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

    # ------------------------------------------------------------------
    # Enrollment ($0.50 USDC)
    # ------------------------------------------------------------------

    def enroll(self, name: str, image_url: str) -> PortraitEnrollment:
        """
        Enroll a Virtual Portrait. Costs $0.50 USDC on Base, one-time.

        Args:
            name: Display name (1-64 chars).
            image_url: Public `https://` URL pointing to a JPG/PNG/WEBP
                image (max 10 MB). Server-side fetched at enrollment time.

        Returns:
            PortraitEnrollment with the `ta_xxxxxxxx` asset id, settlement
            tx hash, and usage hints.

        Raises:
            ValueError: If name or image_url fails local validation.
            PaymentError: If wallet balance is insufficient or the payment
                is rejected.
            APIError: For 4xx/5xx upstream errors (502 = enrollment failed,
                no payment was taken — safe to retry with a different image).
        """
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > _MAX_NAME_LEN:
            raise ValueError(f"name must be {_MAX_NAME_LEN} chars or fewer (got {len(name)})")
        if not image_url or not image_url.lower().startswith(("https://", "http://")):
            raise ValueError("image_url must be an http(s) URL")

        body: Dict[str, Any] = {
            "name": name,
            "image_url": image_url,
        }
        return self._post_with_payment(self.ENROLL_ENDPOINT, body)

    # ------------------------------------------------------------------
    # Listing (free, rate-limited)
    # ------------------------------------------------------------------

    def list_portraits(self, wallet_address: Optional[str] = None) -> PortraitList:
        """
        List portraits enrolled by a wallet. Free, but rate-limited to
        ~20 requests / hour / IP (shared with the wallet-reconciliation
        bucket).

        Args:
            wallet_address: Wallet to query. Defaults to the client's own
                address.

        Returns:
            PortraitList with the wallet address and each portrait's
            asset id, name, image url, and enrollment tx hash.
        """
        addr = wallet_address or self.account.address
        url = f"{self.api_url}/v1/wallet/{addr}/portraits"
        resp = self._client.get(url)
        if resp.status_code == 429:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Rate limit exceeded"}
            raise APIError(
                "Rate limit exceeded on portrait listing",
                resp.status_code,
                sanitize_error_response(error_body),
            )
        if resp.status_code != 200:
            self._raise_api_error(resp, "Portrait listing failed")
        return PortraitList(**resp.json())

    # ------------------------------------------------------------------
    # Internal: x402 paid POST
    # ------------------------------------------------------------------

    def _post_with_payment(self, endpoint: str, body: Dict[str, Any]) -> PortraitEnrollment:
        url = f"{self.api_url}{endpoint}"

        resp = self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code == 402:
            return self._handle_payment_and_retry(url, body, resp)

        if resp.status_code != 200:
            self._raise_api_error(resp, "Enrollment failed")

        return PortraitEnrollment(**resp.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> PortraitEnrollment:
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                resp_body = response.json()
                if isinstance(resp_body, dict) and (
                    "x402Version" in resp_body or "accepts" in resp_body
                ):
                    payment_header = resp_body
            except Exception:
                pass
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = (
            parse_payment_required(payment_header)
            if isinstance(payment_header, str)
            else payment_header
        )
        details = extract_payment_details(payment_required)
        resource = details.get("resource") or {}
        extensions = payment_required.get("extensions", {})

        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=validate_resource_url(resource.get("url", url), self.api_url),
            resource_description=resource.get(
                "description", "BlockRun Virtual Portrait Enrollment"
            ),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
        )

        retry = self._client.post(
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "PAYMENT-SIGNATURE": payment_payload,
            },
        )

        if retry.status_code == 402:
            raise PaymentError("Payment was rejected. Check your wallet balance.")

        if retry.status_code == 502:
            # Enrollment failed upstream — no payment was taken per the spec.
            self._raise_api_error(
                retry,
                "Portrait enrollment failed upstream (no payment taken — safe to retry)",
            )

        if retry.status_code != 200:
            self._raise_api_error(retry, "Enrollment failed after payment")

        return PortraitEnrollment(**retry.json())

    def _raise_api_error(self, resp: httpx.Response, prefix: str) -> None:
        try:
            error_body = resp.json()
        except Exception:
            error_body = {"error": "Request failed"}
        raise APIError(
            f"{prefix}: HTTP {resp.status_code}",
            resp.status_code,
            sanitize_error_response(error_body),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_wallet_address(self) -> str:
        """Return the wallet address used for payments."""
        return self.account.address

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
