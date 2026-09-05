"""
BlockRun Speech Client - Text-to-speech and sound effects (ElevenLabs) via x402 micropayments.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator

Usage:
    from blockrun_llm import SpeechClient

    client = SpeechClient()  # Uses BLOCKRUN_WALLET_KEY from env

    # Text-to-speech (paid, price scales with character count)
    result = client.generate("Hello from BlockRun!", voice="sarah")
    print(result.data[0].url)  # audio URL

    # Sound effects (paid, flat $0.05/generation)
    result = client.sound_effect("rain on a tin roof, distant thunder")
    print(result.data[0].url)

    # List available voices (free, rate-limited)
    voices = client.list_voices()

Models & pricing:
    elevenlabs/flash-v2.5        $0.05/1k chars  ~75ms latency, 32 languages (default)
    elevenlabs/turbo-v2.5        $0.05/1k chars  ~250ms latency, 32 languages
    elevenlabs/multilingual-v2   $0.10/1k chars  long-form narration, 29 languages
    elevenlabs/v3                $0.10/1k chars  max expressiveness, 70+ languages
    elevenlabs/sound-effects     $0.05/generation (up to 22s)

Price = (characters / 1000) x model rate, minimum $0.001/request.
"""

from __future__ import annotations

import os
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
)
from .tx_log import paid_request_error_prefix
from .types import APIError, PaymentError, SpeechResponse
from .validation import (
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
)
from .x402 import create_payment_payload, extract_payment_details, parse_payment_required

load_dotenv()

# Friendly voice aliases accepted by /v1/audio/speech (raw ElevenLabs
# voice_ids pass through unchanged). Mirrors backend VOICE_ALIASES.
VOICE_ALIASES = [
    "sarah",  # Mature, reassuring, confident (default)
    "george",  # Warm, captivating storyteller
    "laura",  # Enthusiast, quirky
    "charlie",  # Deep, confident, energetic
    "river",  # Relaxed, neutral, informative
    "roger",  # Laid-back, casual, resonant
    "callum",  # Husky trickster
    "harry",  # Fierce warrior
]


class SpeechClient:
    """
    BlockRun Speech Client (BlockRun Voice).

    Text-to-speech and sound-effect generation using ElevenLabs models
    with automatic x402 micropayments on Base chain.

    TTS pricing scales with input characters; sound effects are flat
    $0.05/generation.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MODEL = "elevenlabs/flash-v2.5"
    DEFAULT_SOUNDFX_MODEL = "elevenlabs/sound-effects"
    DEFAULT_VOICE = "sarah"
    DEFAULT_TIMEOUT = 120.0  # synthesis is synchronous (<1s for Flash)

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the BlockRun Speech client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 120)
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

    def generate(
        self,
        input: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        response_format: str | None = None,
        speed: float | None = None,
    ) -> SpeechResponse:
        """
        Synthesize speech from text (OpenAI-compatible TTS).

        Price scales with character count: (chars / 1000) x model rate,
        minimum $0.001/request. Synthesis is synchronous.

        Args:
            input: Text to synthesize. Per-model character caps apply
                   (flash/turbo 40k, multilingual-v2 10k, v3 5k).
            model: Speech model ID (default: "elevenlabs/flash-v2.5")
                   Options: "elevenlabs/flash-v2.5", "elevenlabs/turbo-v2.5",
                   "elevenlabs/multilingual-v2", "elevenlabs/v3"
            voice: Voice alias (sarah, george, laura, charlie, river, roger,
                   callum, harry) or a raw ElevenLabs voice_id
                   (default: "sarah")
            response_format: "mp3" (default), "opus", "pcm", or "wav"
            speed: Playback speed 0.7-1.2 (optional)

        Returns:
            SpeechResponse with audio URL, format, and character count

        Raises:
            PaymentError: If wallet has insufficient balance
            APIError: If the API returns an error

        Example:
            result = client.generate("Welcome to BlockRun.", voice="george")
            print(result.data[0].url)
        """
        body: dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "input": input,
        }
        if voice:
            body["voice"] = voice
        if response_format:
            body["response_format"] = response_format
        if speed is not None:
            body["speed"] = speed

        return self._request_with_payment("/v1/audio/speech", body)

    # OpenAI-style alias
    speak = generate

    def sound_effect(
        self,
        text: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        prompt_influence: float | None = None,
        response_format: str | None = None,
    ) -> SpeechResponse:
        """
        Generate a cinematic sound effect from a text prompt.

        Flat $0.05/generation, up to 22 seconds of audio.

        Args:
            text: Sound effect description (max 1000 chars).
                  E.g. "rain on a tin roof", "sci-fi door whoosh"
            model: Model ID (default: "elevenlabs/sound-effects")
            duration_seconds: Target duration 0.5-22s (optional; auto if unset)
            prompt_influence: 0-1, higher follows the prompt more literally
            response_format: "mp3" (default), "opus", "pcm", or "wav"

        Returns:
            SpeechResponse with audio URL and format

        Example:
            result = client.sound_effect("crackling campfire at night")
            print(result.data[0].url)
        """
        body: dict[str, Any] = {
            "model": model or self.DEFAULT_SOUNDFX_MODEL,
            "text": text,
        }
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        if prompt_influence is not None:
            body["prompt_influence"] = prompt_influence
        if response_format:
            body["response_format"] = response_format

        return self._request_with_payment("/v1/audio/sound-effects", body)

    def list_voices(self) -> list[dict[str, Any]]:
        """
        List available voices for TTS (free, rate-limited 60 req/min/IP).

        Returns:
            List of voice dicts. Pass a voice's `alias` (if present) or
            `voice_id` as the `voice` argument to generate().
        """
        response = self._client.get(f"{self.api_url}/v1/audio/voices")

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

        return response.json().get("data", [])

    def _request_with_payment(self, endpoint: str, body: dict[str, Any]) -> SpeechResponse:
        """Make a request with automatic x402 payment handling."""
        url = f"{self.api_url}{endpoint}"

        response = self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 402:
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(response, self.api_key)
            return self._handle_payment_and_retry(url, endpoint, body, response)

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

        return SpeechResponse(**response.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        endpoint: str,
        body: dict[str, Any],
        response: httpx.Response,
    ) -> SpeechResponse:
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
            resource_url=resource.get("url", f"{self.api_url}{endpoint}"),
            resource_description=resource.get("description", "BlockRun Voice"),
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
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(retry_response, self.api_key)
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
        # Attach tx hash from response header
        tx_hash = retry_response.headers.get("x-payment-receipt") or retry_response.headers.get(
            "X-Payment-Receipt"
        )
        if tx_hash:
            data["txHash"] = tx_hash

        return SpeechResponse(**data)

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
        """Get the wallet address being used for payments."""
        return self.account.address

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
