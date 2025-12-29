"""
BlockRun LLM Client - Main SDK entry point.

Usage:
    from blockrun_llm import LLMClient

    # Initialize with private key from env (BLOCKRUN_WALLET_KEY)
    client = LLMClient()

    # Or pass private key directly
    client = LLMClient(private_key="0x...")

    # Simple 1-line chat
    response = client.chat("gpt-4o", "What is 2+2?")
    print(response)

    # Full chat with messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    result = client.chat_completion("gpt-4o", messages)
    print(result.choices[0].message.content)
"""

import os
from typing import List, Dict, Any, Optional, Union
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import ChatMessage, ChatResponse, APIError, PaymentError
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    validate_model,
    validate_max_tokens,
    validate_temperature,
    validate_top_p,
    sanitize_error_response,
    validate_resource_url,
)


# Load environment variables
load_dotenv()


class LLMClient:
    """
    BlockRun LLM Gateway Client.

    Provides access to multiple LLM providers (OpenAI, Anthropic, Google, etc.)
    with automatic x402 micropayments on Base chain.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the BlockRun LLM client.

        Args:
            private_key: EVM wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 60)

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

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Simple 1-line chat interface.

        Args:
            model: Model ID (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-2.5-pro")
            prompt: User message
            system: Optional system prompt
            max_tokens: Max tokens to generate (default: 1024)
            temperature: Sampling temperature

        Returns:
            Assistant's response text

        Example:
            response = client.chat("gpt-4o", "What is the capital of France?")
            print(response)  # "The capital of France is Paris."
        """
        messages: List[Dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        result = self.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return result.choices[0].message.content

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ChatResponse:
        """
        Full chat completion interface (OpenAI-compatible).

        Args:
            model: Model ID
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            ChatResponse object with choices and usage

        Example:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ]
            result = client.chat_completion("gpt-4o", messages)
        """
        # Validate inputs
        validate_model(model)
        validate_max_tokens(max_tokens)
        validate_temperature(temperature)
        validate_top_p(top_p)

        # Build request body
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p

        # Make request (with automatic payment handling)
        return self._request_with_payment("/v1/chat/completions", body)

    def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> ChatResponse:
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
        return ChatResponse(**response.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> ChatResponse:
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
                resource.get("url", f"{self.api_url}/v1/chat/completions"),
                self.api_url
            ),
            resource_description=resource.get("description", "BlockRun AI API call"),
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

        return ChatResponse(**retry_response.json())

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models with pricing.

        Returns:
            List of model information dicts
        """
        response = self._client.get(f"{self.api_url}/v1/models")

        if response.status_code != 200:
            raise APIError(
                f"Failed to list models: {response.status_code}",
                response.status_code,
            )

        return response.json().get("data", [])

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


# Async client for async/await usage
class AsyncLLMClient:
    """
    Async version of BlockRun LLM Client.

    Usage:
        async with AsyncLLMClient() as client:
            response = await client.chat("gpt-4o", "Hello!")
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        key = private_key or os.environ.get("BLOCKRUN_WALLET_KEY")
        if not key:
            raise ValueError(
                "Private key required. Set BLOCKRUN_WALLET_KEY env or pass private_key."
            )

        # Validate private key format
        validate_private_key(key)

        self.account = Account.from_key(key)

        # Validate and set API URL
        api_url_raw = api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Async 1-line chat interface."""
        messages: List[Dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        result = await self.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return result.choices[0].message.content

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ChatResponse:
        """Async full chat completion interface."""
        # Validate inputs
        validate_model(model)
        validate_max_tokens(max_tokens)
        validate_temperature(temperature)
        validate_top_p(top_p)

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p

        return await self._request_with_payment("/v1/chat/completions", body)

    async def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> ChatResponse:
        """Make async request with automatic payment handling."""
        url = f"{self.api_url}{endpoint}"

        response = await self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 402:
            return await self._handle_payment_and_retry(url, body, response)

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

        return ChatResponse(**response.json())

    async def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> ChatResponse:
        """Handle 402 response asynchronously."""
        # Get payment required header (x402 library uses lowercase)
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
                resource.get("url", f"{self.api_url}/v1/chat/completions"),
                self.api_url
            ),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
        )

        # Retry with payment (x402 library expects PAYMENT-SIGNATURE header)
        retry_response = await self._client.post(
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
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        return ChatResponse(**retry_response.json())

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models asynchronously."""
        response = await self._client.get(f"{self.api_url}/v1/models")

        if response.status_code != 200:
            raise APIError(
                f"Failed to list models: {response.status_code}",
                response.status_code,
            )

        return response.json().get("data", [])

    def get_wallet_address(self) -> str:
        """Get the wallet address."""
        return self.account.address

    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
