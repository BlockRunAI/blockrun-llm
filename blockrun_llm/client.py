"""
BlockRun LLM Client - Main SDK entry point.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator
4. Your actual private key is NEVER transmitted to any server

This is the same security model as:
- Signing a MetaMask transaction
- Any on-chain swap or trade
- Standard EIP-3009 TransferWithAuthorization

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
from typing import List, Dict, Any, Optional
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import (
    ChatResponse,
    APIError,
    PaymentError,
)
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


# =============================================================================
# Standalone Functions (no wallet required)
# =============================================================================


def list_models(api_url: str = "https://blockrun.ai/api") -> List[Dict[str, Any]]:
    """
    List available LLM models with pricing (no wallet required).

    This is a standalone function that queries the public API endpoint.
    No wallet or authentication needed.

    Args:
        api_url: API endpoint (default: https://blockrun.ai/api)

    Returns:
        List of model dicts with id, name, provider, pricing, context window, etc.

    Example:
        from blockrun_llm import list_models
        models = list_models()
        for m in models:
            print(f"{m['id']}: ${m.get('inputPrice', 'N/A')}/M input")
    """
    with httpx.Client(timeout=30) as client:
        # Use /pricing endpoint which includes full model details
        response = client.get(f"{api_url.rstrip('/')}/pricing")
        if response.status_code != 200:
            raise APIError(
                f"Failed to list models: {response.status_code}",
                response.status_code,
                {},
            )
        data = response.json()
        return data.get("models", [])


def list_image_models(api_url: str = "https://blockrun.ai/api") -> List[Dict[str, Any]]:
    """
    List available image generation models without requiring wallet.

    This is a standalone function that queries the public API endpoint.
    No wallet or authentication needed.

    Args:
        api_url: API endpoint (default: https://blockrun.ai/api)

    Returns:
        List of image model dicts with id, pricing, etc.
        Returns empty list if endpoint not available.

    Example:
        from blockrun_llm import list_image_models
        models = list_image_models()
        for m in models:
            print(f"{m['id']}: ${m.get('pricePerImage', 'N/A')}/image")
    """
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{api_url.rstrip('/')}/v1/images/models")
        if response.status_code == 404:
            # Endpoint not available yet - return empty list
            return []
        if response.status_code != 200:
            raise APIError(
                f"Failed to list image models: {response.status_code}",
                response.status_code,
                {},
            )
        return response.json().get("data", [])


# =============================================================================
# LLM Client Class (requires wallet)
# =============================================================================


class LLMClient:
    """
    BlockRun LLM Gateway Client.

    Provides access to multiple LLM providers (OpenAI, Anthropic, Google, etc.)
    with automatic x402 micropayments on Base chain.

    Security: Your private key is used ONLY for local EIP-712 signing.
    The key NEVER leaves your machine - only signatures are transmitted.
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
            private_key: Base chain wallet private key (or set BLOCKRUN_WALLET_KEY env var)
                         NOTE: Key is used for LOCAL signing only - never transmitted
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 60)

        Raises:
            ValueError: If no wallet is configured. For agent use, call setup_agent_wallet() first.

        Security:
            Your private key NEVER leaves your machine. It is only used to sign
            EIP-712 typed data locally. Only the signature is sent to the server.
        """
        # Get private key from param, environment, or ~/.blockrun/.session file
        # SECURITY: Key is stored in memory only, used for LOCAL signing
        from .wallet import load_wallet

        key = (
            private_key
            or os.environ.get("BLOCKRUN_WALLET_KEY")
            or os.environ.get("BASE_CHAIN_WALLET_KEY")
            or load_wallet()  # Loads from ~/.blockrun/.session
        )
        if not key:
            raise ValueError(
                "No wallet configured. Either:\n"
                "  1. Set BLOCKRUN_WALLET_KEY environment variable\n"
                "  2. Pass private_key to LLMClient()\n"
                "  3. For agent use: call setup_agent_wallet() first"
            )

        # Normalize private key format (add 0x prefix if missing)
        if key and not key.startswith("0x"):
            key = "0x" + key

        # Validate private key format
        validate_private_key(key)

        # Initialize wallet account
        # SECURITY: Key stays local, only used to sign EIP-712 messages
        # The key is NEVER transmitted - only signatures are sent
        self.account = Account.from_key(key)

        # Validate and set API URL
        api_url_raw = api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout

        # HTTP client
        self._client = httpx.Client(timeout=timeout)

        # Session spending tracking
        self._session_total_usd: float = 0.0
        self._session_calls: int = 0

    def get_spending(self) -> Dict[str, Any]:
        """
        Get current session spending.

        Returns:
            Dict with total_usd and calls count

        Example:
            spending = client.get_spending()
            print(f"Spent ${spending['total_usd']:.4f} across {spending['calls']} calls")
        """
        return {
            "total_usd": self._session_total_usd,
            "calls": self._session_calls,
        }

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Simple 1-line chat interface.

        Args:
            model: Model ID (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4", "xai/grok-3")
            prompt: User message
            system: Optional system prompt
            max_tokens: Max tokens to generate (default: 1024)
            temperature: Sampling temperature
            search: Enable xAI Live Search (shortcut for search_parameters={"mode": "on"})
            search_parameters: Full xAI Live Search configuration (for Grok models)
                See: https://docs.x.ai/docs/guides/live-search

        Returns:
            Assistant's response text

        Example:
            response = client.chat("openai/gpt-4o", "What is the capital of France?")

            # Check spending after calls
            spending = client.get_spending()
            print(f"Spent ${spending['total_usd']:.4f}")

            # With xAI Live Search (for real-time X/Twitter data)
            response = client.chat(
                "xai/grok-3",
                "What are the latest posts from @blockrunai?",
                search=True  # Enable live search
            )
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
            search=search,
            search_parameters=search_parameters,
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
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        Full chat completion interface (OpenAI-compatible).

        Args:
            model: Model ID
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            search: Enable xAI Live Search (shortcut for search_parameters={"mode": "on"})
            search_parameters: Full xAI Live Search configuration (for Grok models)

        Returns:
            ChatResponse object with choices, usage, and citations (if search enabled)

        Raises:
            PaymentError: If budget is set and would be exceeded

        Example:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ]
            result = client.chat_completion("gpt-4o", messages)

            # With xAI Live Search
            result = client.chat_completion(
                "xai/grok-3",
                [{"role": "user", "content": "Latest news about AI?"}],
                search=True
            )
            print(result.citations)  # URLs of sources used
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

        # Handle xAI Live Search parameters
        if search_parameters is not None:
            body["search_parameters"] = search_parameters
        elif search is True:
            # Simple shortcut: search=True enables live search with defaults
            body["search_parameters"] = {"mode": "on"}

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
        """
        Handle 402 response: parse requirements, sign payment locally, retry.

        SECURITY: Payment signing happens entirely on your machine.
        Only the signature is sent - your private key never leaves.
        """
        # Get payment required header (x402 library uses lowercase)
        payment_header = response.headers.get("payment-required")
        price_info = {}
        if not payment_header:
            # Try to get from response body
            try:
                resp_body = response.json()
                if "x402" in resp_body:
                    payment_header = resp_body
                # Extract price info for spending report
                price_info = resp_body.get("price", {})
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

        # Get the cost being paid
        cost_usd = (
            float(price_info.get("amount", 0))
            if price_info
            else float(details.get("amount", 0)) / 1e6
        )

        # Create signed payment payload (v2 format)
        # SECURITY: Signing happens locally - only the signature is sent to server
        resource = details.get("resource") or {}
        # Pass through extensions from server (for Bazaar discovery)
        extensions = payment_required.get("extensions", {})
        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=validate_resource_url(
                resource.get("url", f"{self.api_url}/v1/chat/completions"), self.api_url
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

        # Parse response
        chat_response = ChatResponse(**retry_response.json())

        # Update session spending
        self._session_calls += 1
        self._session_total_usd += cost_usd

        return chat_response

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available LLM models with pricing.

        Returns:
            List of model information dicts
        """
        response = self._client.get(f"{self.api_url}/v1/models")

        if response.status_code != 200:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"Failed to list models: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return response.json().get("data", [])

    def list_image_models(self) -> List[Dict[str, Any]]:
        """
        List available image generation models with pricing.

        Returns:
            List of image model information dicts
        """
        response = self._client.get(f"{self.api_url}/v1/images/models")

        if response.status_code != 200:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"Failed to list image models: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return response.json().get("data", [])

    def list_all_models(self) -> List[Dict[str, Any]]:
        """
        List all available models (both LLM and image) with pricing.

        Returns:
            List of all model information dicts with 'type' field ('llm' or 'image')

        Example:
            models = client.list_all_models()
            for model in models:
                if model['type'] == 'llm':
                    print(f"LLM: {model['id']} - ${model['inputPrice']}/M input")
                else:
                    print(f"Image: {model['id']} - ${model['pricePerImage']}/image")
        """
        # Get LLM models
        llm_models = self.list_models()
        for model in llm_models:
            model["type"] = "llm"

        # Get image models
        image_models = self.list_image_models()
        for model in image_models:
            model["type"] = "image"

        return llm_models + image_models

    def get_wallet_address(self) -> str:
        """Get the wallet address being used for payments."""
        return self.account.address

    def get_balance(self) -> float:
        """
        Get USDC balance on Base network.

        Returns:
            float: USDC balance (6 decimal places normalized)

        Example:
            balance = client.get_balance()
            print(f"Balance: ${balance:.2f} USDC")
        """
        # USDC contract on Base
        usdc_contract = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

        # balanceOf(address) function selector
        selector = "0x70a08231"
        # Pad wallet address to 32 bytes
        padded_address = self.account.address[2:].lower().zfill(64)
        data = selector + padded_address

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": usdc_contract, "data": data}, "latest"],
            "id": 1,
        }

        # Use public Base RPC
        response = httpx.post("https://mainnet.base.org", json=payload, timeout=10)
        result = response.json().get("result", "0x0")

        # Convert from hex and normalize (USDC has 6 decimals)
        balance_raw = int(result, 16)
        return balance_raw / 1_000_000

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
        """
        Initialize the async BlockRun LLM client.

        Raises:
            ValueError: If no wallet is configured
        """
        from .wallet import load_wallet

        key = (
            private_key
            or os.environ.get("BLOCKRUN_WALLET_KEY")
            or os.environ.get("BASE_CHAIN_WALLET_KEY")
            or load_wallet()  # Loads from ~/.blockrun/.session
        )
        if not key:
            raise ValueError(
                "No wallet configured. Either:\n"
                "  1. Set BLOCKRUN_WALLET_KEY environment variable\n"
                "  2. Pass private_key to AsyncLLMClient()\n"
                "  3. For agent use: call setup_agent_wallet() first"
            )

        # Normalize private key format (add 0x prefix if missing)
        if key and not key.startswith("0x"):
            key = "0x" + key

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
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async 1-line chat interface with optional xAI Live Search."""
        messages: List[Dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        result = await self.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            search=search,
            search_parameters=search_parameters,
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
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """Async full chat completion interface with optional xAI Live Search."""
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

        # Handle xAI Live Search parameters
        if search_parameters is not None:
            body["search_parameters"] = search_parameters
        elif search is True:
            # Simple shortcut: search=True enables live search with defaults
            body["search_parameters"] = {"mode": "on"}

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
        # SECURITY: Signing happens locally - only the signature is sent to server
        resource = details.get("resource") or {}
        # Pass through extensions from server (for Bazaar discovery)
        extensions = payment_required.get("extensions", {})
        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:8453"),
            resource_url=validate_resource_url(
                resource.get("url", f"{self.api_url}/v1/chat/completions"), self.api_url
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
        """List available LLM models asynchronously."""
        response = await self._client.get(f"{self.api_url}/v1/models")

        if response.status_code != 200:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"Failed to list models: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return response.json().get("data", [])

    async def list_image_models(self) -> List[Dict[str, Any]]:
        """List available image generation models asynchronously."""
        response = await self._client.get(f"{self.api_url}/v1/images/models")

        if response.status_code != 200:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"Failed to list image models: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return response.json().get("data", [])

    async def list_all_models(self) -> List[Dict[str, Any]]:
        """
        List all available models (both LLM and image) asynchronously.

        Returns:
            List of all model information dicts with 'type' field ('llm' or 'image')
        """
        # Get LLM models
        llm_models = await self.list_models()
        for model in llm_models:
            model["type"] = "llm"

        # Get image models
        image_models = await self.list_image_models()
        for model in image_models:
            model["type"] = "image"

        return llm_models + image_models

    def get_wallet_address(self) -> str:
        """Get the wallet address."""
        return self.account.address

    async def get_balance(self) -> float:
        """
        Get USDC balance on Base network.

        Returns:
            float: USDC balance (6 decimal places normalized)

        Example:
            balance = await client.get_balance()
            print(f"Balance: ${balance:.2f} USDC")
        """
        # USDC contract on Base
        usdc_contract = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

        # balanceOf(address) function selector
        selector = "0x70a08231"
        # Pad wallet address to 32 bytes
        padded_address = self.account.address[2:].lower().zfill(64)
        data = selector + padded_address

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": usdc_contract, "data": data}, "latest"],
            "id": 1,
        }

        # Use public Base RPC
        async with httpx.AsyncClient(timeout=10) as http_client:
            response = await http_client.post("https://mainnet.base.org", json=payload)
        result = response.json().get("result", "0x0")

        # Convert from hex and normalize (USDC has 6 decimals)
        balance_raw = int(result, 16)
        return balance_raw / 1_000_000

    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
