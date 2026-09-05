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

The client signs once and replays the same PAYMENT-SIGNATURE on every poll,
re-signing automatically if the 600s authorization window lapses mid-poll.
Settlement happens only on the first completed poll, so upstream failure or
a client timeout alone cannot establish the final billing status.

Usage:
    from blockrun_llm import VideoClient

    client = VideoClient()  # Uses BLOCKRUN_WALLET_KEY from env

    result = client.generate("a red apple slowly spinning on a wooden table")
    print(result.data[0].url)            # permanent blockrun-hosted MP4 URL
    print(result.data[0].duration_seconds)
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from eth_account import Account

from .api_key import EvmAccountMode, resolve_api_auth
from .types import APIError, PaymentError, VideoResponse
from .validation import (
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
    validate_video_input_type,
)
from .x402 import create_payment_payload, extract_payment_details, parse_payment_required

load_dotenv()


class VideoClient(EvmAccountMode):
    """
    BlockRun Video Generation Client.

    Supports xAI Grok Imagine Video and ByteDance Seedance (1.5 Pro /
    2.0 Fast / 2.0 Pro) with automatic x402 micropayments on Base.

    Pricing (approx. 5s 720p clip):
      xai/grok-imagine-video       $0.050/sec  (8s default → ~$0.40)
      bytedance/seedance-1.5-pro   $4.32/M tok flat   (~$0.46 / 5s)
      bytedance/seedance-2.0-fast  $11.20/M text or $6.60/M image  (~$1.19 / $0.70 / 5s)
      bytedance/seedance-2.0       $14.00/M text or $8.60/M image  (~$1.49 / $0.91 / 5s)

    Seedance 2.0 fast/pro additionally accept `real_face_asset_id` —
    a `ta_xxxxxx` face/character asset for consistency across multiple
    videos. The asset can be either:
      - a Virtual Portrait (AI-generated character) enrolled via
        `PortraitClient` / `POST /v1/portrait/enroll` ($0.01 USDC), or
      - a RealFace (a real person's likeness) enrolled via
        `RealFaceClient` / `POST /v1/realface/enroll` ($0.01 USDC, no
        KYC — just a brief on-phone liveness check).
    Both flows return the same `ta_` id. seedance-1.5-pro does NOT
    support these assets. Mutually exclusive with `image_url`.
    Resolution and generate_audio can be overridden per call. Returned
    URLs are permanent (mirrored to BlockRun storage).
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MODEL = "xai/grok-imagine-video"
    DEFAULT_TIMEOUT = 360.0  # per-HTTP-call timeout (submit / each poll)
    POLL_INTERVAL_SECONDS = 5.0
    # 15 min: generation itself is 1-3 min, but the upstream pipeline can lag
    # the status read-path several minutes behind actual completion (observed:
    # video done in 100s, status flipped ~7.5min later). Jobs stay claimable
    # ~48h, so a patient default beats a premature give-up.
    DEFAULT_GENERATE_BUDGET_SECONDS = 900.0
    # Advertised signed-auth window. Server-side default is 300s; we bump to
    # 600s so the signature stays valid across the async polling window.
    # Budgets longer than this window are handled by re-signing mid-poll.
    MAX_TIMEOUT_SECONDS = 600
    # Max mid-poll re-signs after a 402 (signature expiry). A fresh signature
    # that 402s again means a genuine payment problem, not expiry.
    MAX_POLL_RESIGNS = 2

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 360.0,
        api_key: str | None = None,
    ):
        """
        Initialize the BlockRun Video client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Per-HTTP-call timeout in seconds (submit+each poll).
        """
        from .wallet import load_wallet

        self._api_auth = resolve_api_auth(api_key, private_key, api_url)
        if not self._api_auth:
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

        api_url_raw = (
            self._api_auth.api_url
            if self._api_auth
            else (api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL)
        )
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout
        self._client = httpx.Client(auth=self._api_auth, follow_redirects=False, timeout=timeout)

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        image_url: str | None = None,
        last_frame_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        real_face_asset_id: str | None = None,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        generate_audio: bool | None = None,
        seed: int | None = None,
        watermark: bool | None = None,
        return_last_frame: bool | None = None,
        input_type: str | None = None,
        budget_seconds: float | None = None,
    ) -> VideoResponse:
        """
        Generate a video clip from a text prompt (or text + image / face asset).

        Submits an async job, then polls until the video is ready. Typical
        total wall-time is 60-180s, but upstream status can lag several
        minutes behind actual completion. If upstream takes longer than the
        budget (default 15min), we raise without charging — the job stays
        claimable ~48h via the poll_url in the error details.

        Args:
            prompt: Text description of the video.
            model: Model ID (default: xai/grok-imagine-video).
            image_url: Optional seed image URL for image-to-video.
            last_frame_url: First-and-last-frame interpolation — a second
                image that seeds the FINAL frame so the model tweens from
                `image_url` -> `last_frame_url`. Requires `image_url` and a
                Seedance model (bytedance/seedance-1.5-pro, seedance-2.0,
                or seedance-2.0-fast). Priced identically to image-to-video.
            reference_image_urls: Omni / multi-reference — up to 9 reference
                image URLs for character/style consistency (Seedance 2.0
                only). Cite them as "image 1", "image 2" in the prompt.
                Mutually exclusive with `image_url`, `last_frame_url`, and
                `real_face_asset_id`.
            real_face_asset_id: A `ta_xxxxxx` face/character asset for
                identity consistency — either a Virtual Portrait (AI
                character, via `PortraitClient`, $0.01) or a RealFace
                (real person, via `RealFaceClient`, $0.01, no KYC).
                Seedance 2.0 fast/pro only. Mutually exclusive with
                `image_url`.
            duration_seconds: Billed duration (defaults to model's default).
            aspect_ratio: `adaptive` / `16:9` / `9:16` / `1:1` / `4:3` /
                `3:4` / `21:9` / `9:21` (Seedance only; Grok ignores).
            resolution: Output resolution — `360p` / `480p` / `720p` /
                `1080p` / `4K`. Seedance defaults to `720p`; Grok ignores.
            generate_audio: Synced audio in the output. Seedance defaults
                to `True` for text-to-video and `False` for image- or
                face-conditioned generation. Grok ignores this field.
            seed: Deterministic generation seed (Seedance only).
            watermark: Add the provider watermark (Seedance only).
            return_last_frame: Also return the final frame as an image
                (Seedance only).
            input_type: Optional assertion of the seed mode you intend —
                `text` / `image` / `first_last_frame` / `reference`. Purely a
                guard: the gateway infers the mode from the seed fields above
                and rejects with 400 (before charging) if your declared value
                disagrees. Use it when a caller builds the seed fields
                dynamically and a silently-wrong mode would be expensive — a
                dropped `image_url` yields a text-to-video clip you still pay
                for, whereas declaring `input_type="image"` turns that into an
                error. Leave unset to accept whatever the inputs imply.
            budget_seconds: Overall polling budget (default 900s).

        Returns:
            VideoResponse with the clip URL, duration, upstream request_id,
            and the settlement tx hash.

        Raises:
            ValueError: If mutually-exclusive image inputs are combined
                (see above), `last_frame_url` is passed without `image_url`,
                `real_face_asset_id` is malformed, or `input_type` is not one
                of the four accepted values.
            PaymentError: If wallet balance is insufficient.
            APIError: If upstream fails, the job times out, or any transport
                error occurs.
        """
        if image_url and real_face_asset_id:
            raise ValueError(
                "image_url and real_face_asset_id are mutually exclusive; pass at most one."
            )
        if last_frame_url and not image_url:
            raise ValueError(
                "last_frame_url requires image_url: image_url seeds the FIRST frame and "
                "last_frame_url the FINAL frame — send both."
            )
        if last_frame_url and real_face_asset_id:
            raise ValueError(
                "last_frame_url and real_face_asset_id are mutually exclusive; "
                "first-and-last-frame uses image_url + last_frame_url."
            )
        if reference_image_urls:
            if image_url or last_frame_url or real_face_asset_id:
                raise ValueError(
                    "reference_image_urls is mutually exclusive with image_url, "
                    "last_frame_url, and real_face_asset_id."
                )
            if len(reference_image_urls) > 9:
                raise ValueError("reference_image_urls accepts at most 9 images.")
        if real_face_asset_id is not None and not real_face_asset_id.startswith("ta_"):
            raise ValueError(
                "real_face_asset_id must start with 'ta_' "
                "(a Virtual Portrait or RealFace asset id, e.g. 'ta_abc123xyz' — "
                "enroll via PortraitClient / POST /v1/portrait/enroll or "
                "RealFaceClient / POST /v1/realface/enroll)"
            )
        validate_video_input_type(input_type)

        body: dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "prompt": prompt,
        }
        if image_url:
            body["image_url"] = image_url
        if last_frame_url:
            body["last_frame_url"] = last_frame_url
        if reference_image_urls:
            body["reference_image_urls"] = reference_image_urls
        if real_face_asset_id:
            body["real_face_asset_id"] = real_face_asset_id
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        if aspect_ratio is not None:
            body["aspect_ratio"] = aspect_ratio
        if resolution is not None:
            body["resolution"] = resolution
        if generate_audio is not None:
            body["generate_audio"] = generate_audio
        if seed is not None:
            body["seed"] = seed
        if watermark is not None:
            body["watermark"] = watermark
        if return_last_frame is not None:
            body["return_last_frame"] = return_last_frame
        if input_type is not None:
            body["input_type"] = input_type

        budget = (
            budget_seconds if budget_seconds is not None else self.DEFAULT_GENERATE_BUDGET_SECONDS
        )

        return self._submit_and_poll(body, budget)

    def generate_from_content(
        self,
        content: list[dict[str, Any]],
        *,
        model: str | None = None,
        budget_seconds: float | None = None,
        **options: Any,
    ) -> VideoResponse:
        """
        Generate a video from a standard Seedance ``content[]`` body.

        This targets the gateway's ``POST /v1/videos`` endpoint, which accepts
        the mainstream multimodal ``content`` array (text + a single reference
        image) used by other Seedance APIs, so callers already holding a
        ``content[]``-shaped request can submit it unchanged. The gateway
        validates unsupported inputs *before* charging and then delegates to
        the same x402 submit+poll pipeline as :meth:`generate`.

        Most SDK users should prefer :meth:`generate` (structured kwargs like
        ``image_url`` / ``last_frame_url``) — this method exists for migrating
        existing ``content[]`` payloads with no reshaping.

        Args:
            content: The Seedance ``content`` array, e.g.
                ``[{"type": "text", "text": "a red apple spinning"}]`` or a
                text item plus ``{"type": "image_url", "image_url": {...}}``.
            model: Model ID (default: the gateway's standard Seedance model).
            budget_seconds: Overall polling budget (default 900s).
            **options: Extra top-level body fields forwarded verbatim
                (``resolution``, ``duration_seconds``, ``aspect_ratio``,
                ``generate_audio``, ``seed``, ``watermark`` …).

        Returns:
            VideoResponse with the clip URL, duration, upstream request_id,
            and the settlement tx hash.
        """
        if not content:
            raise ValueError("content must be a non-empty list of Seedance content items.")

        body: dict[str, Any] = {"content": content, **options}
        if model is not None:
            body["model"] = model

        budget = (
            budget_seconds if budget_seconds is not None else self.DEFAULT_GENERATE_BUDGET_SECONDS
        )
        return self._submit_and_poll(body, budget, submit_path="/v1/videos")

    # ------------------------------------------------------------------
    # Internal: async submit + poll
    # ------------------------------------------------------------------

    def _submit_and_poll(
        self,
        body: dict[str, Any],
        budget_seconds: float,
        submit_path: str = "/v1/videos/generations",
    ) -> VideoResponse:
        submit_url = f"{self.api_url}{submit_path}"

        # Step 1: unauth POST -> 402 with payment requirements
        resp402 = self._client.post(
            submit_url,
            json=body,
            headers={"Content-Type": "application/json"},
        )
        if self._api_auth:
            return VideoResponse(
                **self._api_auth.poll(
                    self._client, resp402, budget_seconds, self.POLL_INTERVAL_SECONDS
                )
            )

        if resp402.status_code != 402:
            self._raise_api_error(resp402, "Expected 402 on first POST")

        payment_payload = self._sign_from_challenge(resp402, submit_url)

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

        # Step 3: poll with the same PAYMENT-SIGNATURE until completed. The
        # signed authorization is valid for MAX_TIMEOUT_SECONDS (600s); when a
        # poll 402s after that window, we fetch a fresh challenge from the
        # same poll_url and re-sign with the same wallet — the gateway
        # enforces wallet binding, not signature equality.
        deadline = time.monotonic() + budget_seconds
        last_status = submit_data.get("status", "queued")
        resigns_left = self.MAX_POLL_RESIGNS

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

            # Terminal success is keyed on status, NOT the HTTP code — the
            # gateway settles the moment a poll reports completed, so coupling
            # success to a literal 200 would spin to the deadline (and report
            # "not charged") on a completed-but-non-200 poll the caller was
            # already charged for. Mirrors the Go/TS SDKs.
            if last_status == "completed":
                tx_hash = poll_resp.headers.get("x-payment-receipt") or poll_resp.headers.get(
                    "X-Payment-Receipt"
                )
                if tx_hash:
                    poll_data["txHash"] = tx_hash
                return VideoResponse(**poll_data)

            if poll_resp.status_code == 402:
                # Mid-poll 402 = the signed authorization expired (600s
                # window) on a budget longer than that. Re-challenge +
                # re-sign and keep going. A fresh signature that 402s again
                # is a genuine payment problem.
                if resigns_left > 0:
                    resigns_left -= 1
                    challenge = self._client.get(poll_url)
                    if challenge.status_code == 402:
                        payment_payload = self._sign_from_challenge(challenge, poll_url)
                        continue
                raise PaymentError(
                    "Payment verification failed mid-poll (not a signature-expiry). "
                    "Check the wallet balance and that you poll from the wallet "
                    "that submitted the job."
                )

            if poll_resp.status_code not in (200, 202, 504):
                self._raise_api_error(poll_resp, "Poll failed")
            # status 504 on a poll = transient upstream hiccup; retry

        raise APIError(
            f"Video generation did not complete within {budget_seconds:.0f}s "
            f"(last status: {last_status}). A polling timeout does not confirm billing status. "
            "Resume the existing poll_url using the same account or wallet; check "
            "account Activity or wallet receipts before submitting another job.",
            504,
            {"id": job_id, "last_status": last_status, "poll_url": poll_url},
        )

    def _sign_from_challenge(self, resp402: httpx.Response, fallback_url: str) -> str:
        """Parse an x402 challenge response and sign a payment payload for it.

        Used for the initial submit AND for mid-poll re-signing after the
        600s authorization window lapses on long polls.
        """
        payment_required = self._extract_payment_required(resp402)
        details = extract_payment_details(payment_required)
        resource = details.get("resource") or {}
        extensions = payment_required.get("extensions", {})
        return create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=resource.get("url", fallback_url),
            resource_description=resource.get("description", "BlockRun Video Generation"),
            # Cover as much of the polling window as the auth allows.
            max_timeout_seconds=max(
                details.get("maxTimeoutSeconds", 0) or 0, self.MAX_TIMEOUT_SECONDS
            ),
            extra=details.get("extra"),
            extensions=extensions,
        )

    def _absolute(self, url: str) -> str:
        if url.startswith(("http://", "https://")):
            return url
        # self.api_url already ends without '/'; poll_url starts with '/api/...'
        base = self.api_url.removesuffix("/api")
        return f"{base}{url}"

    def _extract_payment_required(self, resp: httpx.Response) -> dict[str, Any]:
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
        self._require_wallet_mode()
        return self.account.address

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
