"""
AnthropicClient — Use the official Anthropic SDK with BlockRun's API.

Wraps anthropic.Anthropic with automatic x402 micropayments on Base chain.
Your private key is used ONLY for local EIP-712 signing and NEVER leaves your machine.

Usage:
    from blockrun_llm import AnthropicClient

    client = AnthropicClient()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.content[0].text)
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from eth_account import Account

from .apikey import (
    PAYMENT_MODE_API_KEY,
    PAYMENT_MODE_WALLET,
    api_key_base_url,
    resolve_api_key,
)
from .validation import validate_api_url, validate_private_key
from .wallet import load_wallet
from .x402 import create_payment_payload, extract_payment_details, parse_payment_required

load_dotenv()

# Default chat HTTP timeout (seconds). Was 120; reasoning models (opus-4.8) think
# 200–300s+, which the old default cut off mid-generation. Override via the
# BLOCKRUN_CHAT_TIMEOUT env var. Mirrors client.py / solana_client.py.
DEFAULT_CHAT_TIMEOUT = float(os.environ.get("BLOCKRUN_CHAT_TIMEOUT", "600"))


class _BlockRunX402Transport(httpx.BaseTransport):
    """Custom httpx transport that intercepts 402 responses and signs x402 payments."""

    def __init__(
        self, account: Account, api_url: str, base_transport: httpx.BaseTransport | None = None
    ):
        self._account = account
        self._api_url = api_url
        self._base = base_transport or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._base.handle_request(request)

        if response.status_code != 402:
            return response

        # Read the 402 body so we can parse payment requirements
        response.read()

        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                resp_body = response.json()
                if "x402" in resp_body:
                    payment_header = resp_body
            except Exception:
                pass

        if not payment_header:
            return response

        if isinstance(payment_header, str):
            payment_required = parse_payment_required(payment_header)
        else:
            payment_required = payment_header

        details = extract_payment_details(payment_required)

        resource = details.get("resource") or {}
        extensions = payment_required.get("extensions", {})
        payment_payload = create_payment_payload(
            account=self._account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=resource.get("url", f"{self._api_url}/v1/messages"),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        request.headers["PAYMENT-SIGNATURE"] = payment_payload
        return self._base.handle_request(request)

    def close(self) -> None:
        self._base.close()


class AnthropicClient:
    """BlockRun-powered Anthropic client with automatic x402 payments.

    Drop-in replacement for anthropic.Anthropic that routes through BlockRun's
    multi-model API gateway with automatic USDC micropayments on Base chain.

    Your private key is used ONLY for local EIP-712 signing and NEVER transmitted.

    Usage:
        from blockrun_llm import AnthropicClient

        client = AnthropicClient()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.content[0].text)

        # Works with any BlockRun model in Anthropic format
        response = client.messages.create(
            model="openai/gpt-5.5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello from GPT!"}]
        )
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = DEFAULT_CHAT_TIMEOUT,
        **kwargs,
    ):
        """
        Initialize the BlockRun Anthropic client.

        Args:
            private_key: BlockRun API key or Base wallet private key. With no argument,
                         BLOCKRUN_API_KEY takes precedence over wallet configuration.
                         Wallet keys are used for local signing only.
            api_url: BlockRun API endpoint (default: https://blockrun.ai/api).
            timeout: Request timeout in seconds (default: 600, override via
                     BLOCKRUN_CHAT_TIMEOUT env). Reasoning models need 200–300s+.
            **kwargs: Additional keyword arguments passed to anthropic.Anthropic.

        Raises:
            ImportError: If the `anthropic` package is not installed.
            ValueError: If no wallet is configured.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient.\n"
                "Install it with: pip install blockrun-llm[anthropic]"
            )

        # Resolve the payment method before reading or parsing a wallet.
        self.api_key = resolve_api_key(private_key)
        self.payment_mode = PAYMENT_MODE_API_KEY if self.api_key else PAYMENT_MODE_WALLET

        # An ambiguous failed POST may already be billed, so retrying is an
        # explicit caller choice rather than a default inherited from Anthropic.
        # This has to hold on BOTH rails, and it matters more on the wallet one:
        # the transport below signs a fresh payment for every 402 it sees, so a
        # 5xx retried after the gateway already settled signs and settles again.
        # At Anthropic's default of 2 that is three on-chain USDC transfers for
        # one messages.create().
        kwargs.setdefault("max_retries", 0)

        if self.api_key:
            self._api_url = api_key_base_url(api_url)
            validate_api_url(self._api_url)
            self._client = anthropic.Anthropic(
                base_url=self._api_url,
                api_key=self.api_key,
                http_client=httpx.Client(timeout=timeout, follow_redirects=False),
                **kwargs,
            )
            return

        key = (
            private_key
            or os.environ.get("BLOCKRUN_WALLET_KEY")
            or os.environ.get("BASE_CHAIN_WALLET_KEY")
            or load_wallet()
        )
        if not key:
            raise ValueError(
                "No wallet configured. Either:\n"
                "  1. Set BLOCKRUN_WALLET_KEY environment variable\n"
                "  2. Pass private_key to AnthropicClient()\n"
                "  3. For agent use: call setup_agent_wallet() first"
            )

        if not key.startswith("0x"):
            key = "0x" + key

        validate_private_key(key)
        account = Account.from_key(key)

        api_url_resolved = api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL
        validate_api_url(api_url_resolved)
        self._api_url = api_url_resolved.rstrip("/")

        transport = _BlockRunX402Transport(
            account=account,
            api_url=self._api_url,
        )

        http_client = httpx.Client(transport=transport, timeout=timeout)

        self._client = anthropic.Anthropic(
            base_url=self._api_url,
            api_key="blockrun",
            http_client=http_client,
            **kwargs,
        )

    @property
    def messages(self):
        """Access the Messages API (client.messages.create(...))."""
        return self._client.messages

    def __getattr__(self, name):
        return getattr(self._client, name)
