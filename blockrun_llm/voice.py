"""
BlockRun Voice Call Client - AI-powered outbound phone calls via x402 micropayments.

The AI agent calls a phone number (E.164) and conducts a conversation based on your
'task' instructions. Speech-to-text, LLM reasoning, and text-to-speech are all handled
upstream by Bland.ai; BlockRun handles billing through x402.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator

Usage:
    from blockrun_llm import VoiceClient

    client = VoiceClient()  # Uses BLOCKRUN_WALLET_KEY from env

    # Initiate a call (paid, $0.54)
    result = client.call(
        to="+14155552671",
        task="You are a friendly assistant calling to confirm a 3pm dentist appointment.",
        max_duration=5,
    )
    print(result["call_id"])

    # Poll for status, transcript, and recording (free)
    status = client.get_status(result["call_id"])
    print(status)

Pricing: $0.54 per outbound call (regardless of duration up to max_duration).
"""

import os
from typing import Optional, Dict, Any, List
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import APIError, PaymentError
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
)
from .tx_log import paid_request_error_prefix

load_dotenv()


# Built-in Bland.ai voice presets — any string accepted by Bland is also valid.
VOICE_PRESETS: List[str] = ["nat", "josh", "maya", "june", "paige", "derek", "florian"]

# Bland.ai conversation models
CALL_MODELS: List[str] = ["base", "enhanced", "turbo"]

# Settled price per call (USD)
CALL_PRICE_USD: float = 0.54


class VoiceClient:
    """
    BlockRun Voice Call Client.

    Initiates AI-powered outbound phone calls. The AI agent dials the recipient and
    conducts a real-time conversation following your 'task' description.

    Pricing: $0.54 per call. Status polling is free.

    Caller-ID requirements: every call needs a `from` number your wallet owns.
    Provision one with PhoneClient.buy_number() before placing calls; if your
    wallet owns exactly one active number, the backend auto-picks it.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_TIMEOUT = 60.0  # call initiation returns quickly; long-poll status separately

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the BlockRun Voice client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds
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

    def call(
        self,
        to: str,
        task: str,
        *,
        from_: Optional[str] = None,
        voice: Optional[str] = None,
        max_duration: int = 5,
        language: str = "en-US",
        first_sentence: Optional[str] = None,
        wait_for_greeting: Optional[bool] = None,
        interruption_threshold: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initiate an AI-powered outbound phone call.

        Args:
            to: Destination phone number in E.164 format (e.g. "+14155552671").
                US and Canada supported.
            task: Natural-language instructions for the AI agent
                (10-4000 chars). Describe what the call should accomplish.
            from_: Your provisioned BlockRun phone number (E.164). Shown as caller ID.
                Must be owned by your wallet — buy one via PhoneClient.buy_number()
                or POST /v1/phone/numbers/buy ($5 / 30-day lease).
                Use the trailing-underscore form because 'from' is a Python keyword.

                If omitted:
                - wallet owns exactly 1 active number → that number is used automatically
                - wallet owns 0  → APIError(403) "no_active_number" (buy one first)
                - wallet owns 2+ → APIError(400) "ambiguous_from" (pass `from_` explicitly;
                  the error body lists your_active_numbers so the agent can pick)
            voice: One of VOICE_PRESETS (nat, josh, maya, june, paige, derek, florian)
                or any custom Bland.ai voice ID.
            max_duration: Maximum call length in minutes (1-30, default 5).
            language: BCP-47 language code for STT/TTS (default "en-US").
            first_sentence: Optional opening line the agent says before listening.
            wait_for_greeting: If True, wait for the recipient to speak first.
            interruption_threshold: Sensitivity for detecting recipient interruptions
                (50-500ms). Lower = quicker to yield the floor.
            model: Conversation model — "base", "enhanced", or "turbo".

        Returns:
            Dict with keys:
                - call_id (str): Bland.ai call identifier
                - status (str): Initial status (usually "queued")
                - poll_url (str): URL to poll for transcript/recording
                - message (str): Human-readable note
                - txHash (str, optional): On-chain payment receipt

        Raises:
            ValueError: If arguments are out of range
            PaymentError: If wallet has insufficient balance
            APIError: If the API or upstream provider returns an error

        Example:
            result = client.call(
                to="+14155552671",
                task="Call the user and confirm they want to reschedule to Tuesday 2pm.",
                voice="maya",
                max_duration=3,
            )
            print(result["call_id"])
        """
        if not to or not to.strip():
            raise ValueError("'to' phone number is required (E.164 format)")
        if not task or len(task.strip()) < 10:
            raise ValueError("'task' must be at least 10 characters")
        if len(task) > 4000:
            raise ValueError("'task' must be at most 4000 characters")
        if max_duration < 1 or max_duration > 30:
            raise ValueError("max_duration must be between 1 and 30 minutes")
        if model is not None and model not in CALL_MODELS:
            raise ValueError(f"model must be one of {CALL_MODELS}")
        if interruption_threshold is not None and not (50 <= interruption_threshold <= 500):
            raise ValueError("interruption_threshold must be between 50 and 500")

        body: Dict[str, Any] = {
            "to": to.strip(),
            "task": task.strip(),
            "max_duration": max_duration,
            "language": language,
        }
        if from_:
            body["from"] = from_.strip()
        if voice:
            body["voice"] = voice
        if first_sentence:
            body["first_sentence"] = first_sentence.strip()
        if wait_for_greeting is not None:
            body["wait_for_greeting"] = wait_for_greeting
        if interruption_threshold is not None:
            body["interruption_threshold"] = interruption_threshold
        if model:
            body["model"] = model

        return self._request_with_payment("/v1/voice/call", body)

    def get_status(self, call_id: str) -> Dict[str, Any]:
        """
        Poll the status of an in-progress or completed call. Free — no payment.

        Args:
            call_id: The 'call_id' returned by call().

        Returns:
            Dict with the full Bland.ai call record, including:
                - status: "queued" | "in-progress" | "completed" | "failed" | ...
                - transcripts: List of turns once available
                - recording_url: Audio URL once the call ends
                - duration, started_at, ended_at, etc.

        Raises:
            APIError: If the call is not found (404) or upstream errors.
        """
        if not call_id or not call_id.strip():
            raise ValueError("call_id is required")

        url = f"{self.api_url}/v1/voice/call/{call_id.strip()}"
        response = self._client.get(url, headers={"Accept": "application/json"})

        if response.status_code == 404:
            raise APIError(f"Call not found: {call_id}", 404, {"call_id": call_id})

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

        return response.json()

    def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST with automatic x402 payment handling."""
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

        return response.json()

    def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """Handle 402: parse requirements, sign payment, retry."""
        payment_header = response.headers.get("payment-required")
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
            resource_url=resource.get("url", f"{self.api_url}/v1/voice/call"),
            resource_description=resource.get("description", "BlockRun Voice Call"),
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
                f"{paid_request_error_prefix(retry_response.headers)}: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        data = retry_response.json()
        tx_hash = retry_response.headers.get("x-payment-receipt") or retry_response.headers.get(
            "X-Payment-Receipt"
        )
        if tx_hash:
            data["txHash"] = tx_hash
        return data

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
