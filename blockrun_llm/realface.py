"""
BlockRun RealFace Client — enroll a real person's face via x402 micropayments.

A RealFace registers a *real person's* likeness as a face/character reference
asset. Unlike a Virtual Portrait (AI-generated character, see PortraitClient),
RealFace proves the enroller is the same person in the photo via a brief
on-phone liveness check (nod + blink, ~1 minute). **No KYC** — no government
ID, no account login, just the liveness step. After enrollment ($0.01 USDC,
one-time) you get back a `ta_xxxxxxxx` asset id that can be passed as
`real_face_asset_id` to `VideoClient.generate()` on Seedance 2.0 / 2.0-fast
to keep the same person across multiple videos.

The flow is three steps:

    1. init()           — FREE. Returns a group_id + an h5_link the real
                          person scans on their phone.
    2. (phone liveness) — The rights-holder opens h5_link, allows camera,
                          nods + blinks. ~60 seconds. Nothing goes to BlockRun.
    3. enroll()         — $0.01 USDC. Uploads the face photo, matches it
                          against the live capture, returns the ta_xxx asset.

Use status() (or the wait_for_active() helper) between steps 2 and 3 to detect
when the person has finished the phone check.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. The key is used locally to
sign an EIP-3009 USDC transfer authorization; only the signature is
transmitted in the PAYMENT-SIGNATURE header.

Usage:
    from blockrun_llm import RealFaceClient

    client = RealFaceClient()  # Uses BLOCKRUN_WALLET_KEY from env

    # 1. Start enrollment (free). Render init.h5_link as a QR for the person.
    init = client.init(name="Jane — Q3 spokesperson")
    print(init.h5_link)  # show as QR; they scan + do the liveness check

    # 2. Wait until they finish the phone liveness check (polls status).
    client.wait_for_active(init.group_id)

    # 3. Finalize ($0.01 USDC) with the person's face photo.
    rf = client.enroll(
        name="Jane — Q3 spokesperson",
        image_url="https://example.com/jane.jpg",
        group_id=init.group_id,
    )
    print(rf.asset_id)              # ta_abcdef1234567890
    print(rf.settlement.tx_hash)    # 0x9f3a…

    # List the wallet's enrolled RealFaces (free, no payment)
    listing = client.list_realfaces()
    for r in listing.realfaces:
        print(r.assetId, r.name)
"""

from __future__ import annotations

import os
import re
import time
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
    wallet_only,
)
from .tx_log import paid_request_error_prefix
from .types import (
    APIError,
    PaymentError,
    RealFaceEnrollment,
    RealFaceInit,
    RealFaceList,
    RealFaceStatus,
    retry_after_of,
)
from .validation import (
    raise_api_error,
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
    validate_resource_url,
)
from .x402 import (
    create_payment_payload,
    extract_payment_details,
    parse_payment_required,
)

load_dotenv()


# Hard limits enforced upstream; mirror locally to fail fast.
_MAX_NAME_LEN = 64
# Upstream group ids look like "legacy_rf_8137"; validate to fail fast.
_GROUP_ID_RE = re.compile(r"^legacy_rf_\d+$")


class RealFaceClient:
    """
    BlockRun RealFace Client.

    Wraps the three-step real-person enrollment flow:
      - `POST /v1/realface/init`    (free, rate-limited)
      - `GET  /v1/realface/status`  (free, rate-limited)
      - `POST /v1/realface/enroll`  ($0.01 USDC, one-time)
    plus the free `GET /v1/wallet/<address>/realfaces` listing endpoint.

    The enroll endpoint settles AFTER the asset is successfully matched and
    registered upstream, so failed enrollments (group not active, face
    mismatch, network error) return an error with no charge.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    INIT_ENDPOINT = "/v1/realface/init"
    STATUS_ENDPOINT = "/v1/realface/status"
    ENROLL_ENDPOINT = "/v1/realface/enroll"

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 60.0,
    ):
        """
        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY).
            api_url: API endpoint URL (default https://blockrun.ai/api).
            timeout: Per-HTTP-call timeout in seconds.
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

    # ------------------------------------------------------------------
    # Step 1: init (free, rate-limited)
    # ------------------------------------------------------------------

    def init(self, name: str, group_id: str | None = None) -> RealFaceInit:
        """
        Start (or refresh) a RealFace enrollment. Free, but rate-limited to
        ~10 calls / hour / IP (each call creates an upstream session).

        Args:
            name: Display name for the asset group (1-64 chars).
            group_id: If set, refresh the h5_link for this existing group
                instead of creating a new one. Use when the original 120s
                H5 session expired before the person finished scanning.

        Returns:
            RealFaceInit with `group_id`, the `h5_link` to give the real
            person (render as a QR), and `expires_in_seconds`.

        Raises:
            ValueError: If name or group_id fails local validation.
            APIError: For 4xx/5xx upstream errors (429 = rate limited).
        """
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > _MAX_NAME_LEN:
            raise ValueError(f"name must be {_MAX_NAME_LEN} chars or fewer (got {len(name)})")
        if group_id is not None and not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")

        body: dict[str, Any] = {"name": name}
        if group_id:
            body["groupId"] = group_id

        url = f"{self.api_url}{self.INIT_ENDPOINT}"
        resp = self._client.post(url, json=body, headers={"Content-Type": "application/json"})
        raise_for_api_key_402(resp, self.api_key)
        if resp.status_code != 200:
            self._raise_api_error(resp, "RealFace init failed")
        return RealFaceInit(**resp.json())

    # ------------------------------------------------------------------
    # Step 2 helper: status / wait_for_active (free, rate-limited)
    # ------------------------------------------------------------------

    def status(self, group_id: str) -> RealFaceStatus:
        """
        Poll the state of a RealFace asset group. Free, but rate-limited.

        Args:
            group_id: The `legacy_rf_…` id returned by init().

        Returns:
            RealFaceStatus; `ready_to_finalize` is True once the real person
            has completed the phone liveness check (status == "active").

        Raises:
            ValueError: If group_id fails local validation.
            APIError: For 4xx/5xx upstream errors.
        """
        if not group_id or not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")

        url = f"{self.api_url}{self.STATUS_ENDPOINT}"
        resp = self._client.get(url, params={"groupId": group_id})
        raise_for_api_key_402(resp, self.api_key)
        if resp.status_code != 200:
            self._raise_api_error(resp, "RealFace status check failed")
        return RealFaceStatus(**resp.json())

    def wait_for_active(
        self,
        group_id: str,
        timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 4.0,
    ) -> RealFaceStatus:
        """
        Block until the group is active (the real person finished the phone
        liveness check), then return its status. Convenience wrapper around
        repeated status() polling.

        Args:
            group_id: The `legacy_rf_…` id returned by init().
            timeout_seconds: Give up after this long (default 180s; the H5
                session itself expires ~120s after each init/refresh).
            poll_interval_seconds: Seconds between status checks (default 4s,
                matching the studio UI; keep >=3s to respect rate limits).

        Returns:
            RealFaceStatus with `ready_to_finalize == True`.

        Raises:
            ValueError: If group_id fails local validation.
            TimeoutError: If the group is not active within timeout_seconds.
            APIError: For 4xx/5xx upstream errors during polling.
        """
        if not group_id or not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")

        deadline = time.monotonic() + timeout_seconds
        while True:
            state = self.status(group_id)
            if state.ready_to_finalize:
                return state
            if time.monotonic() + poll_interval_seconds >= deadline:
                raise TimeoutError(
                    f"RealFace group {group_id} not active after {timeout_seconds:.0f}s "
                    f"(last status: {state.status!r}). The person may not have finished the "
                    f"phone liveness check; call init(group_id=…) to refresh an expired h5_link."
                )
            time.sleep(poll_interval_seconds)

    # ------------------------------------------------------------------
    # Step 3: enroll ($0.01 USDC)
    # ------------------------------------------------------------------

    def enroll(self, name: str, image_url: str, group_id: str) -> RealFaceEnrollment:
        """
        Finalize a RealFace enrollment. Costs $0.01 USDC on Base, one-time.

        Requires the real person to have already completed the phone liveness
        check (group status == "active"; use wait_for_active() to block on it).

        Args:
            name: Display name (1-64 chars).
            image_url: Public `https://` URL pointing to a JPG/PNG/WEBP photo
                of the same person (max 10 MB). Server-side fetched and
                matched against the live H5 capture.
            group_id: The `legacy_rf_…` id returned by init().

        Returns:
            RealFaceEnrollment with the `ta_xxxxxxxx` asset id, settlement tx
            hash, and usage hints.

        Raises:
            ValueError: If any argument fails local validation.
            PaymentError: If wallet balance is insufficient or the payment
                is rejected.
            APIError: For upstream errors. No payment is taken on these:
                425 = group not active yet (do the phone check first),
                422 = face did not match the live capture (try a clearer
                photo), 502 = upstream upload/status failure.
        """
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > _MAX_NAME_LEN:
            raise ValueError(f"name must be {_MAX_NAME_LEN} chars or fewer (got {len(name)})")
        if not image_url or not image_url.lower().startswith(("https://", "http://")):
            raise ValueError("image_url must be an http(s) URL")
        if not group_id or not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")

        body: dict[str, Any] = {
            "name": name,
            "image_url": image_url,
            "group_id": group_id,
        }
        return self._post_with_payment(self.ENROLL_ENDPOINT, body)

    # ------------------------------------------------------------------
    # Listing (free, rate-limited)
    # ------------------------------------------------------------------

    def list_realfaces(self, wallet_address: str | None = None) -> RealFaceList:
        """
        List RealFaces enrolled by a wallet. Free, but rate-limited to
        ~20 requests / hour / IP (shared with the wallet-reconciliation
        bucket).

        Args:
            wallet_address: Wallet to query. Defaults to the client's own
                address.

        Returns:
            RealFaceList with the wallet address and each RealFace's asset id,
            name, image url, and enrollment tx hash.
        """
        # Keyed by wallet, so the account rail has no default to fall back on:
        # say which argument is missing instead of an AttributeError on None.
        if not wallet_address and self.account is None:
            raise wallet_only("list_realfaces")
        addr = wallet_address or self.account.address
        url = f"{self.api_url}/v1/wallet/{addr}/realfaces"
        resp = self._client.get(url)
        if resp.status_code == 429:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Rate limit exceeded"}
            raise APIError(
                "Rate limit exceeded on RealFace listing",
                resp.status_code,
                sanitize_error_response(error_body),
                retry_after=retry_after_of(resp),
            )
        raise_for_api_key_402(resp, self.api_key)
        if resp.status_code != 200:
            self._raise_api_error(resp, "RealFace listing failed")
        return RealFaceList(**resp.json())

    # ------------------------------------------------------------------
    # Internal: x402 paid POST
    # ------------------------------------------------------------------

    def _post_with_payment(self, endpoint: str, body: dict[str, Any]) -> RealFaceEnrollment:
        url = f"{self.api_url}{endpoint}"

        resp = self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code == 402:
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(resp, self.api_key)
            return self._handle_payment_and_retry(url, body, resp)

        if resp.status_code != 200:
            self._raise_api_error(resp, "RealFace enrollment failed")

        return RealFaceEnrollment(**resp.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        body: dict[str, Any],
        response: httpx.Response,
    ) -> RealFaceEnrollment:
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
            resource_description=resource.get("description", "BlockRun RealFace Enrollment"),
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
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(retry, self.api_key)
            raise PaymentError("Payment was rejected. Check your wallet balance.")

        if retry.status_code == 425:
            # Group is not active — the person hasn't finished the phone
            # liveness check. No payment was taken.
            self._raise_api_error(
                retry,
                "RealFace group not active yet — the person must finish the phone "
                "liveness check first (no payment taken)",
            )

        if retry.status_code == 422:
            # Face did not match the live H5 capture. No payment was taken.
            self._raise_api_error(
                retry,
                "RealFace match failed — the photo did not match the live capture "
                "(try a clearer front-facing photo of the same person; no payment taken)",
            )

        if retry.status_code == 502:
            # Upstream upload / status failure — no payment was taken.
            self._raise_api_error(
                retry,
                "RealFace enrollment failed upstream (no payment taken — safe to retry)",
            )

        if retry.status_code != 200:
            self._raise_api_error(
                retry, f"RealFace enrollment: {paid_request_error_prefix(retry.headers)}"
            )

        return RealFaceEnrollment(**retry.json())

    def _raise_api_error(self, resp: httpx.Response, prefix: str) -> None:
        raise_api_error(resp, prefix)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
        """Return the wallet address used for payments."""
        return self.account.address

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
