"""
BlockRun Video Client - Generate short AI videos via x402 micropayments.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator

Async flow (client-polled):
    POST /v1/videos/generations         -> 402 -> sign -> 202 { id, poll_url }
    GET  /v1/videos/generations/{id}    -> loop until status=completed

The client signs ONCE and replays the same PAYMENT-SIGNATURE on every poll.
Settlement happens only on the first completed poll, so upstream failure or
the caller giving up = zero charge.

Usage:
    from blockrun_llm import VideoClient

    client = VideoClient()  # Uses BLOCKRUN_WALLET_KEY from env

    result = client.generate("a red apple slowly spinning on a wooden table")
    print(result.data[0].url)            # permanent blockrun-hosted MP4 URL
    print(result.data[0].duration_seconds)
"""

import os
import time
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

    Supports xAI Grok Imagine Video and ByteDance Seedance (1.5 Pro /
    2.0 Fast / 2.0 Pro) with automatic x402 micropayments on Base.

    Pricing:
      xai/grok-imagine-video       $0.05/sec, 8s default
      bytedance/seedance-1.5-pro   $0.03/sec, 5s default (up to 10s)
      bytedance/seedance-2.0-fast  $0.15/sec, 5s default (up to 10s)
      bytedance/seedance-2.0       $0.30/sec, 5s default (up to 10s)

    Returned URLs are permanent (mirrored to BlockRun storage).
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MODEL = "xai/grok-imagine-video"
    DEFAULT_TIMEOUT = 360.0  # overall budget: submit (~20s) + poll loop (5min)
    POLL_INTERVAL_SECONDS = 5.0
    # Upstream job TTL is 24-48h; we use a per-generate budget instead.
    DEFAULT_GENERATE_BUDGET_SECONDS = 300.0
    # Advertised signed-auth window. Server-side default is 300s; we bump to
    # 600s so the signature stays valid across the async polling window.
    MAX_TIMEOUT_SECONDS = 600

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 360.0,
    ):
        """
        Initialize the BlockRun Video client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Per-HTTP-call timeout in seconds (submit+each poll).
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
        budget_seconds: Optional[float] = None,
    ) -> VideoResponse:
        """
        Generate a video clip from a text prompt (or text + image).

        Submits an async job, then polls until the video is ready. Typical
        total wall-time is 60-180s. If upstream takes longer than the budget
        (default 5min), we raise without charging.

        Args:
            prompt: Text description of the video.
            model: Model ID (default: xai/grok-imagine-video).
            image_url: Optional seed image URL for image-to-video.
            duration_seconds: Billed duration (defaults to model's default).
            budget_seconds: Overall polling budget (default 300s).

        Returns:
            VideoResponse with the clip URL, duration, upstream request_id,
            and the settlement tx hash.

        Raises:
            PaymentError: If wallet balance is insufficient.
            APIError: If upstream fails, the job times out, or any transport
                error occurs.
        """
        body: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "prompt": prompt,
        }
        if image_url:
            body["image_url"] = image_url
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds

        budget = budget_seconds if budget_seconds is not None else self.DEFAULT_GENERATE_BUDGET_SECONDS

        return self._submit_and_poll(body, budget)

    # ------------------------------------------------------------------
    # Internal: async submit + poll
    # ------------------------------------------------------------------

    def _submit_and_poll(self, body: Dict[str, Any], budget_seconds: float) -> VideoResponse:
        submit_url = f"{self.api_url}/v1/videos/generations"

        # Step 1: unauth POST -> 402 with payment requirements
        resp402 = self._client.post(
            submit_url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if resp402.status_code != 402:
            self._raise_api_error(resp402, "Expected 402 on first POST")

        payment_required = self._extract_payment_required(resp402)
        details = extract_payment_details(payment_required)
        resource = details.get("resource") or {}
        extensions = payment_required.get("extensions", {})

        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=resource.get("url", submit_url),
            resource_description=resource.get("description", "BlockRun Video Generation"),
            # Ensure the signed authorization covers the entire polling window.
            max_timeout_seconds=max(details.get("maxTimeoutSeconds", 0) or 0, self.MAX_TIMEOUT_SECONDS),
            extra=details.get("extra"),
            extensions=extensions,
        )

        # Step 2: submit job with payment -> 202 { id, poll_url }
        submit_resp = self._client.post(
            submit_url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "PAYMENT-SIGNATURE": payment_payload,
            },
        )

        if submit_resp.status_code == 402:
            raise PaymentError("Payment was rejected. Check your wallet balance.")

        if submit_resp.status_code not in (200, 202):
            self._raise_api_error(submit_resp, "Submit failed")

        submit_data = submit_resp.json()
        job_id = submit_data.get("id")
        poll_url_rel = submit_data.get("poll_url")
        if not job_id or not poll_url_rel:
            raise APIError(
                "Submit response missing id/poll_url",
                submit_resp.status_code,
                {"response": submit_data},
            )

        poll_url = self._absolute(poll_url_rel)

        # Step 3: poll with the same PAYMENT-SIGNATURE until completed
        deadline = time.monotonic() + budget_seconds
        last_status = submit_data.get("status", "queued")

        while time.monotonic() < deadline:
            time.sleep(self.POLL_INTERVAL_SECONDS)

            poll_resp = self._client.get(
                poll_url,
                headers={"PAYMENT-SIGNATURE": payment_payload},
            )

            try:
                poll_data = poll_resp.json()
            except Exception:
                poll_data = {}

            last_status = poll_data.get("status", last_status)

            if poll_resp.status_code == 202 and last_status in ("queued", "in_progress"):
                continue

            if last_status == "failed":
                raise APIError(
                    f"Upstream generation failed: {poll_data.get('error', 'unknown')}",
                    poll_resp.status_code,
                    sanitize_error_response(poll_data),
                )

            if poll_resp.status_code == 200 and last_status == "completed":
                tx_hash = poll_resp.headers.get("x-payment-receipt") or poll_resp.headers.get(
                    "X-Payment-Receipt"
                )
                if tx_hash:
                    poll_data["txHash"] = tx_hash
                return VideoResponse(**poll_data)

            if poll_resp.status_code not in (200, 202, 504):
                self._raise_api_error(poll_resp, "Poll failed")
            # status 504 on a poll = transient upstream hiccup; retry

        raise APIError(
            f"Video generation did not complete within {budget_seconds:.0f}s "
            f"(last status: {last_status}). No payment was taken.",
            504,
            {"id": job_id, "last_status": last_status},
        )

    def _absolute(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        # self.api_url already ends without '/'; poll_url starts with '/api/...'
        base = self.api_url[: -len("/api")] if self.api_url.endswith("/api") else self.api_url
        return f"{base}{url}"

    def _extract_payment_required(self, resp: httpx.Response) -> Dict[str, Any]:
        header = resp.headers.get("payment-required")
        if header:
            return parse_payment_required(header)
        # Fallback: body contains the x402 PaymentRequired document
        try:
            body = resp.json()
        except Exception:
            body = None
        if isinstance(body, dict) and ("x402Version" in body or "accepts" in body):
            return body
        raise PaymentError("402 response but no payment requirements found")

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
