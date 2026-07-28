"""
BlockRun Phone Client - Twilio-backed phone lookup + number provisioning via x402.

Endpoints (all under /v1/phone/...):
    POST /lookup            $0.01    Carrier + line type lookup
    POST /lookup/fraud      $0.05    Carrier + SIM-swap / call-forwarding fraud signals
    POST /numbers/buy       $5.00    Provision a US/CA number (30-day lease, bound to wallet)
    POST /numbers/renew     $5.00    Extend an existing number by 30 days
    POST /numbers/list      $0.001   List the wallet's active numbers
    POST /numbers/release   free     Release a provisioned number (still goes through x402
                                     so the backend can identify the wallet)

After buying a number you can use it as the `from_` caller-ID in VoiceClient.call().

Usage:
    from blockrun_llm import PhoneClient

    client = PhoneClient()

    # Lookup a number
    info = client.lookup("+14155552671")
    print(info)

    # Buy a number (US, optional area code)
    bought = client.buy_number(country="US", area_code="415")
    print(bought["phone_number"], bought["expires_at"])

    # List your active numbers
    print(client.list_numbers())

    # Renew / release
    client.renew_number(bought["phone_number"])
    client.release_number(bought["phone_number"])

SECURITY NOTE: your private key never leaves your machine. Only EIP-712
signatures are sent in the PAYMENT-SIGNATURE header.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from eth_account import Account
from typing_extensions import Self

from .tx_log import paid_request_error_prefix
from .types import APIError, PaymentError
from .validation import (
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
)
from .x402 import (
    create_payment_payload,
    extract_payment_details,
    parse_payment_required,
)

load_dotenv()


# Mirrors src/lib/twilio.ts PHONE_PRICES on the backend (settled USDC amount).
PHONE_PRICES: dict[str, float] = {
    "lookup": 0.01,
    "lookup/fraud": 0.05,
    "numbers/buy": 5.00,
    "numbers/renew": 5.00,
    "numbers/list": 0.001,
    "numbers/release": 0.0,
}


class PhoneClient:
    """
    BlockRun Phone Client.

    Wraps the `/v1/phone/*` x402 endpoints. Use this for phone-number lookup
    (carrier + fraud) and for provisioning the caller-ID numbers required by
    VoiceClient.call().
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
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
                "  3. Place key in ~/.blockrun/.session"
            )

        validate_private_key(key)
        self.account = Account.from_key(key)

        api_url_raw = api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------ Lookup

    def lookup(self, phone_number: str) -> dict[str, Any]:
        """
        Carrier + line-type lookup. ~$0.01.

        Args:
            phone_number: E.164 number (e.g. "+14155552671").

        Returns:
            Twilio Lookup payload with carrier, line_type_intelligence, etc.
        """
        self._require_e164(phone_number)
        return self._request("lookup", {"phoneNumber": phone_number.strip()})

    def lookup_fraud(self, phone_number: str) -> dict[str, Any]:
        """
        Lookup + fraud signals (SIM swap, call forwarding). ~$0.05.

        Args:
            phone_number: E.164 number.

        Returns:
            Lookup payload including SIM-swap + call-forwarding intelligence.
        """
        self._require_e164(phone_number)
        return self._request("lookup/fraud", {"phoneNumber": phone_number.strip()})

    # ------------------------------------------------------------- Provisioning

    def buy_number(
        self,
        country: str = "US",
        area_code: str | None = None,
    ) -> dict[str, Any]:
        """
        Provision a dedicated phone number for 30 days. $5.00.

        Args:
            country: ISO country code, "US" or "CA" (default "US").
            area_code: Optional 3-digit area-code hint. Availability not guaranteed —
                the backend falls back to any number in the country if the area
                code can't be matched.

        Returns:
            Dict with:
                - phone_number (str): the E.164 number you now own
                - expires_at  (str): ISO-8601 expiry (30 days out)
                - chain       (str): "base" | "solana"
                - message     (str): human-readable note
                - txHash      (str, optional): on-chain payment receipt

        Note: payment is settled only after Twilio confirms the purchase, so
        failed purchases do NOT charge your wallet.
        """
        if country not in ("US", "CA"):
            raise ValueError("country must be 'US' or 'CA'")
        body: dict[str, Any] = {"country": country}
        if area_code is not None:
            if not (isinstance(area_code, str) and area_code.isdigit() and len(area_code) == 3):
                raise ValueError("area_code must be a 3-digit string, e.g. '415'")
            body["areaCode"] = area_code
        return self._request("numbers/buy", body)

    def renew_number(self, phone_number: str) -> dict[str, Any]:
        """
        Extend an existing provisioned number by 30 days. $5.00.

        Args:
            phone_number: E.164 number your wallet owns.

        Returns:
            Dict with phone_number, new expires_at, and txHash.

        Raises:
            APIError(403): wallet doesn't own this number or it has expired.
        """
        self._require_e164(phone_number)
        return self._request("numbers/renew", {"phoneNumber": phone_number.strip()})

    def list_numbers(self) -> dict[str, Any]:
        """
        List the wallet's active phone numbers. ~$0.001.

        Returns:
            Dict with:
                - numbers: list of {phone_number, chain, expires_at, active}
                - count: int
                - txHash: str
        """
        return self._request("numbers/list", {})

    def release_number(self, phone_number: str) -> dict[str, Any]:
        """
        Release a provisioned number back to the Twilio pool. Free, but the
        request still flows through x402 so the backend can verify ownership.

        Args:
            phone_number: E.164 number your wallet owns.

        Returns:
            Dict with {released: True, phone_number}.
        """
        self._require_e164(phone_number)
        return self._request("numbers/release", {"phoneNumber": phone_number.strip()})

    # ---------------------------------------------------------------- Internals

    @staticmethod
    def _require_e164(value: str) -> None:
        if not value or not isinstance(value, str):
            raise ValueError("phone_number is required (E.164 format, e.g. '+14155552671')")
        v = value.strip()
        if not v.startswith("+") or not v[1:].isdigit() or not (8 <= len(v) <= 16):
            raise ValueError(f"phone_number must be E.164 (e.g. '+14155552671'), got {value!r}")

    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url}/v1/phone/{path}"
        response = self._client.post(url, json=body, headers={"Content-Type": "application/json"})
        if response.status_code == 402:
            return self._handle_payment_and_retry(url, body, response)
        return self._unwrap(response)

    def _handle_payment_and_retry(
        self,
        url: str,
        body: dict[str, Any],
        response: httpx.Response,
    ) -> dict[str, Any]:
        payment_header: Any = response.headers.get("payment-required")
        if not payment_header:
            try:
                resp_body = response.json()
                if "x402" in resp_body or "accepts" in resp_body:
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
            resource_url=resource.get("url", url),
            resource_description=resource.get("description", "BlockRun Phone"),
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
        data = self._unwrap(retry, after_payment=True)
        tx_hash = retry.headers.get("x-payment-receipt") or retry.headers.get("X-Payment-Receipt")
        if tx_hash and isinstance(data, dict):
            data.setdefault("txHash", tx_hash)
        return data

    @staticmethod
    def _unwrap(response: httpx.Response, *, after_payment: bool = False) -> dict[str, Any]:
        if response.status_code == 200:
            return response.json()
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": "Request failed"}
        prefix = paid_request_error_prefix(response.headers) if after_payment else "API error"
        raise APIError(
            f"{prefix}: {response.status_code}",
            response.status_code,
            sanitize_error_response(error_body),
        )

    # ------------------------------------------------------------------ Helpers

    def get_wallet_address(self) -> str:
        """Return the EVM wallet address used for payments."""
        return self.account.address

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
