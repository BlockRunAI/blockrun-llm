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
    response = client.chat("gpt-5.2", "What is 2+2?")
    print(response)

    # Full chat with messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    result = client.chat_completion("gpt-5.2", messages)
    print(result.choices[0].message.content)
"""

import os
from typing import List, Dict, Any, Optional, Union
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import (
    ChatResponse,
    ImageResponse,
    APIError,
    PaymentError,
    RoutingDecision,
    SmartChatResponse,
    RoutingProfile,
    SearchResult,
    XUserLookupResponse,
    XFollowersResponse,
    XFollowingsResponse,
    XUserInfoResponse,
    XVerifiedFollowersResponse,
    XTweetsResponse,
    XMentionsResponse,
    XTweetLookupResponse,
    XTweetRepliesResponse,
    XTweetThreadResponse,
    XSearchResponse,
    XTrendingResponse,
    XArticlesRisingResponse,
    XAuthorAnalyticsResponse,
    XCompareAuthorsResponse,
)
from .router import route as route_request
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


# User-Agent for client identification in server logs
# Version read lazily to avoid circular import with __init__.py
def _get_user_agent() -> str:
    from . import __version__

    return f"blockrun-python/{__version__}"


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

    Networks:
        - Mainnet: https://blockrun.ai/api (Base, Chain ID 8453)
        - Testnet: https://testnet.blockrun.ai/api (Base Sepolia, Chain ID 84532)

    Testnet Usage:
        For development and testing without real USDC:

        client = LLMClient(api_url="https://testnet.blockrun.ai/api")

        # Or use the testnet convenience method
        from blockrun_llm import testnet_client
        client = testnet_client()

        Note: Testnet has limited models (openai/gpt-oss-20b, openai/gpt-oss-120b)
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    TESTNET_API_URL = "https://testnet.blockrun.ai/api"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 120.0,
        search_timeout: float = 300.0,
    ):
        """
        Initialize the BlockRun LLM client.

        Args:
            private_key: Base chain wallet private key (or set BLOCKRUN_WALLET_KEY env var)
                         NOTE: Key is used for LOCAL signing only - never transmitted
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 120). Used for regular chat requests.
            search_timeout: Timeout for xAI Live Search requests (default: 300 = 5 minutes).
                           Live Search can be slow as it searches X, web, and news sources.
                           Auto-detected when search_parameters or search=True is passed.

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
        self.search_timeout = search_timeout

        # HTTP client (default timeout, will be overridden for search requests)
        self._client = httpx.Client(timeout=timeout)

        # Session spending tracking
        self._session_total_usd: float = 0.0
        self._session_calls: int = 0
        self._last_call_cost: float = 0.0

        # Model pricing cache for smart routing
        self._model_pricing_cache: Optional[Dict[str, Dict[str, float]]] = None

    def _get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """
        Get model pricing for smart routing.

        Returns:
            Dict mapping model_id -> {"input_price": x, "output_price": y,
            "flat_price": z}. ``flat_price`` is 0 for per-token billing and
            non-zero (USD per call) for flat-billed models.

        The /v1/models response uses the nested ``pricing.input``/``pricing.output``
        shape today; older snapshots used top-level ``inputPrice``/``outputPrice``.
        Both are accepted so the SDK keeps working through backend transitions.
        """
        if self._model_pricing_cache is not None:
            return self._model_pricing_cache

        models = self.list_models()
        pricing: Dict[str, Dict[str, float]] = {}
        for model in models:
            model_id = model.get("id", "")
            block = model.get("pricing") or {}
            input_price = block.get("input", model.get("inputPrice", model.get("input_price", 0)))
            output_price = block.get(
                "output", model.get("outputPrice", model.get("output_price", 0))
            )
            flat_price = block.get("flat", model.get("flatPrice", 0))
            pricing[model_id] = {
                "input_price": float(input_price or 0),
                "output_price": float(output_price or 0),
                "flat_price": float(flat_price or 0),
            }
        self._model_pricing_cache = pricing
        return pricing

    def smart_chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        routing_profile: RoutingProfile = "auto",
    ) -> SmartChatResponse:
        """
        Smart chat with automatic model routing.

        Routes requests to the cheapest capable model using ClawRouter's
        14-dimension rule-based scoring algorithm (<1ms, 100% local).

        Args:
            prompt: User message
            system: Optional system prompt
            max_tokens: Max tokens to generate (default: 1024)
            temperature: Sampling temperature
            routing_profile: "free" | "eco" | "auto" | "premium"
                - free: nvidia/gpt-oss-120b only (FREE)
                - eco: Cheapest models per tier (DeepSeek, xAI)
                - auto: Best balance of cost/quality (default)
                - premium: Top-tier models (OpenAI, Anthropic)

        Returns:
            SmartChatResponse with response, model, and routing decision

        Example:
            result = client.smart_chat("What is 2+2?")
            print(result.response)  # '4'
            print(result.model)     # 'google/gemini-2.5-flash'
            print(f"Saved {result.routing.savings * 100:.0f}%")

            # With routing profile
            result = client.smart_chat(
                "Prove the Riemann hypothesis",
                routing_profile="premium"  # Use top-tier models for complex tasks
            )
        """
        # Get model pricing for routing decision
        model_pricing = self._get_model_pricing()
        max_output_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        # Route the request
        decision = route_request(
            prompt=prompt,
            system_prompt=system,
            max_output_tokens=max_output_tokens,
            model_pricing=model_pricing,
            routing_profile=routing_profile,
        )

        # Make the chat request with selected model
        response = self.chat(
            model=decision["model"],
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return SmartChatResponse(
            response=response,
            model=decision["model"],
            routing=RoutingDecision(**decision),
        )

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
            model: Model ID (e.g., "openai/gpt-5.2", "anthropic/claude-sonnet-4.6", "openai/gpt-5.2")
            prompt: User message
            system: Optional system prompt
            max_tokens: Max tokens to generate (default: 1024)
            temperature: Sampling temperature
            search: Enable xAI Live Search (shortcut for search_parameters={"mode": "on"})
            search_parameters: Full xAI Live Search configuration (for search-enabled models)
                See: https://docs.x.ai/docs/guides/live-search

        Returns:
            Assistant's response text

        Example:
            response = client.chat("openai/gpt-5.2", "What is the capital of France?")

            # Check spending after calls
            spending = client.get_spending()
            print(f"Spent ${spending['total_usd']:.4f}")

            # With xAI Live Search (for real-time X/Twitter data)
            response = client.chat(
                "openai/gpt-5.2",
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
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
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
            search_parameters: Full xAI Live Search configuration (for search-enabled models)
            tools: List of tool definitions for function calling
            tool_choice: Tool selection strategy ("none", "auto", "required", or specific tool)

        Returns:
            ChatResponse object with choices, usage, and citations (if search enabled)

        Raises:
            PaymentError: If budget is set and would be exceeded

        Example:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ]
            result = client.chat_completion("gpt-5.2", messages)

            # With xAI Live Search
            result = client.chat_completion(
                "openai/gpt-5.2",
                [{"role": "user", "content": "Latest news about AI?"}],
                search=True
            )
            print(result.citations)  # URLs of sources used

            # With tool calling
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }]
            result = client.chat_completion("gpt-5.2", messages, tools=tools)
            if result.choices[0].message.tool_calls:
                for tc in result.choices[0].message.tool_calls:
                    print(f"Call: {tc.function.name}({tc.function.arguments})")
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

        # Handle tool calling
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

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
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        # First attempt (will likely return 402)
        response = self._client.post(url, json=body, headers=req_headers)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.post(url, json=body, headers=req_headers)

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
            network=details.get("network", "eip155:84532" if self.is_testnet() else "eip155:8453"),
            resource_url=validate_resource_url(
                resource.get("url", f"{self.api_url}/v1/chat/completions"), self.api_url
            ),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        # Retry with payment (x402 library expects PAYMENT-SIGNATURE header)
        # Use longer timeout for Live Search requests
        is_search_request = "search_parameters" in body or body.get("search") is True
        request_timeout = self.search_timeout if is_search_request else self.timeout

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = self._client.post(
            url, json=body, headers=payment_headers, timeout=request_timeout
        )
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.post(
                url, json=body, headers=payment_headers, timeout=request_timeout
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
        response_data = retry_response.json()
        chat_response = ChatResponse(**response_data)

        # Update session spending
        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd

        # Save full response locally (cost log + response archive)
        from .cache import save_to_cache

        save_to_cache("/v1/chat/completions", body, response_data, cost_usd=cost_usd)

        return chat_response

    def _request_with_payment_raw(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request with automatic x402 payment handling, returning raw JSON.

        Same flow as _request_with_payment() but returns Dict instead of ChatResponse.
        Used for endpoints that don't return the chat completion shape.
        Checks local cache first to avoid paying twice for the same data.
        """
        from .cache import get_cached, save_to_cache

        # Check cache first — don't pay twice for same data
        cached = get_cached(endpoint, body)
        if cached is not None:
            return cached

        url = f"{self.api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        response = self._client.post(url, json=body, headers=req_headers)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.post(url, json=body, headers=req_headers)

        if response.status_code == 402:
            result = self._handle_payment_and_retry_raw(url, body, response)
            # Save paid response to cache
            save_to_cache(endpoint, body, result, cost_usd=self._last_call_cost)
            return result

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

    def _handle_payment_and_retry_raw(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """Handle 402 response for raw endpoints: parse requirements, sign payment, retry."""
        payment_header = response.headers.get("payment-required")
        price_info = {}
        if not payment_header:
            try:
                resp_body = response.json()
                if "x402" in resp_body:
                    payment_header = resp_body
                price_info = resp_body.get("price", {})
            except Exception:
                pass

        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        if isinstance(payment_header, str):
            payment_required = parse_payment_required(payment_header)
        else:
            payment_required = payment_header

        details = extract_payment_details(payment_required)

        cost_usd = (
            float(price_info.get("amount", 0))
            if price_info
            else float(details.get("amount", 0)) / 1e6
        )

        resource = details.get("resource") or {}
        extensions = payment_required.get("extensions", {})
        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:84532" if self.is_testnet() else "eip155:8453"),
            resource_url=validate_resource_url(resource.get("url", url), self.api_url),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = self._client.post(
            url, json=body, headers=payment_headers, timeout=self.timeout
        )
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.post(
                url, json=body, headers=payment_headers, timeout=self.timeout
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

        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd

        return retry_response.json()

    def _get_with_payment_raw(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        GET with automatic x402 payment handling, returning raw JSON.

        Same flow as _request_with_payment_raw() but uses GET with query params
        instead of POST with JSON body. Used for Predexon prediction market endpoints.
        """
        from .cache import get_cached, save_to_cache

        cache_key_body = params or {}
        cached = get_cached(endpoint, cache_key_body)
        if cached is not None:
            return cached

        url = f"{self.api_url}{endpoint}"
        req_headers = {"User-Agent": _get_user_agent()}

        response = self._client.get(url, params=params, headers=req_headers)

        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.get(url, params=params, headers=req_headers)

        if response.status_code == 402:
            result = self._handle_get_payment_and_retry(url, params, response)
            save_to_cache(endpoint, cache_key_body, result, cost_usd=self._last_call_cost)
            return result

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

    def _handle_get_payment_and_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]],
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """Handle 402 response for GET endpoints: parse requirements, sign payment, retry with GET."""
        payment_header = response.headers.get("payment-required")
        price_info = {}
        if not payment_header:
            try:
                resp_body = response.json()
                if "x402" in resp_body:
                    payment_header = resp_body
                price_info = resp_body.get("price", {})
            except Exception:
                pass

        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        if isinstance(payment_header, str):
            payment_required = parse_payment_required(payment_header)
        else:
            payment_required = payment_header

        details = extract_payment_details(payment_required)

        cost_usd = (
            float(price_info.get("amount", 0))
            if price_info
            else float(details.get("amount", 0)) / 1e6
        )

        resource = details.get("resource") or {}
        extensions = payment_required.get("extensions", {})
        payment_payload = create_payment_payload(
            account=self.account,
            recipient=details["recipient"],
            amount=details["amount"],
            network=details.get("network", "eip155:84532" if self.is_testnet() else "eip155:8453"),
            resource_url=validate_resource_url(resource.get("url", url), self.api_url),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        payment_headers = {
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        retry_response = self._client.get(
            url, params=params, headers=payment_headers, timeout=self.timeout
        )
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.get(
                url, params=params, headers=payment_headers, timeout=self.timeout
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

        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd

        return retry_response.json()

    def image_edit(
        self,
        prompt: str,
        image: str,
        *,
        model: str = "openai/gpt-image-1",
        mask: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
    ) -> ImageResponse:
        """
        Edit an image using img2img.

        Args:
            prompt: Text description of the desired edit
            image: Base64-encoded image or URL of the source image
            model: Model ID (default: "openai/gpt-image-1")
            mask: Optional base64-encoded mask image
            size: Output image size (default: "1024x1024")
            n: Number of images to generate (default: 1)

        Returns:
            ImageResponse with edited image URLs
        """
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image": image,
            "size": size,
            "n": n,
        }
        if mask is not None:
            body["mask"] = mask

        data = self._request_with_payment_raw("/v1/images/image2image", body)
        return ImageResponse(**data)

    def search(
        self,
        query: str,
        *,
        sources: Optional[List[str]] = None,
        max_results: int = 10,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> SearchResult:
        """
        Standalone search (web, X/Twitter, news).

        Args:
            query: Search query
            sources: Source types to search (e.g. ["web", "x", "news"])
            max_results: Maximum number of results (default: 10)
            from_date: Start date filter (YYYY-MM-DD)
            to_date: End date filter (YYYY-MM-DD)

        Returns:
            SearchResult with summary and citations
        """
        body: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }
        if sources is not None:
            body["sources"] = sources
        if from_date is not None:
            body["from_date"] = from_date
        if to_date is not None:
            body["to_date"] = to_date

        data = self._request_with_payment_raw("/v1/search", body)
        return SearchResult(**data)

    # ── Exa Web Search (Powered by Exa) ─────────────────────────────────────

    def exa(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Generic Exa endpoint proxy via x402 USDC on Base.

        Args:
            path: Exa endpoint — one of: "search", "find-similar", "contents", "answer"
            body: Request body (see https://docs.exa.ai)

        Example::

            result = client.exa("search", {"query": "latest AI research", "numResults": 5})
        """
        return self._request_with_payment_raw(f"/v1/exa/{path}", body)

    def exa_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Neural and keyword web search via Exa ($0.01/request, Base USDC).

        Args:
            query: Search query string
            **kwargs: Additional Exa parameters (numResults, category, useAutoprompt, etc.)

        Example::

            results = client.exa_search("latest AI papers", numResults=5)
        """
        return self._request_with_payment_raw("/v1/exa/search", {"query": query, **kwargs})

    def exa_find_similar(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Find pages semantically similar to a given URL via Exa
        ($0.01/request, Base USDC).

        Args:
            url: URL to find similar pages for
            **kwargs: Additional Exa parameters (numResults, etc.)

        Example::

            similar = client.exa_find_similar("https://openai.com/research/gpt-4", numResults=5)
        """
        return self._request_with_payment_raw("/v1/exa/find-similar", {"url": url, **kwargs})

    def exa_contents(self, urls: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Extract full text content from URLs via Exa ($0.002/URL, Base USDC).

        Args:
            urls: List of URLs to extract content from
            **kwargs: Additional Exa parameters (text, highlights, summary, etc.)

        Example::

            data = client.exa_contents(["https://arxiv.org/abs/2303.08774"])
        """
        return self._request_with_payment_raw("/v1/exa/contents", {"urls": urls, **kwargs})

    def exa_answer(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """AI-generated answer grounded in live web search via Exa
        ($0.01/request, Base USDC).

        Args:
            query: Question to answer
            **kwargs: Additional Exa parameters

        Example::

            answer = client.exa_answer("What is the current state of AI safety research?")
        """
        return self._request_with_payment_raw("/v1/exa/answer", {"query": query, **kwargs})

    def x_user_lookup(self, usernames: Union[List[str], str]) -> XUserLookupResponse:
        """
        Look up X/Twitter user profiles by username.

        Powered by AttentionVC. $0.002 per user (min $0.02, max $0.20).

        Args:
            usernames: Single username or list of usernames (without @)

        Returns:
            XUserLookupResponse with user profiles
        """
        if isinstance(usernames, str):
            usernames = [usernames]

        body: Dict[str, Any] = {"usernames": usernames}
        data = self._request_with_payment_raw("/v1/x/users/lookup", body)
        return XUserLookupResponse(**data)

    def x_followers(self, username: str, *, cursor: Optional[str] = None) -> XFollowersResponse:
        """
        Get followers of an X/Twitter user.

        Powered by AttentionVC. $0.05 per page (~200 accounts).

        Args:
            username: X/Twitter username (without @)
            cursor: Pagination cursor from previous response

        Returns:
            XFollowersResponse with follower list
        """
        body: Dict[str, Any] = {"username": username}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/users/followers", body)
        return XFollowersResponse(**data)

    def x_followings(self, username: str, *, cursor: Optional[str] = None) -> XFollowingsResponse:
        """
        Get accounts an X/Twitter user is following.

        Powered by AttentionVC. $0.05 per page (~200 accounts).

        Args:
            username: X/Twitter username (without @)
            cursor: Pagination cursor from previous response

        Returns:
            XFollowingsResponse with following list
        """
        body: Dict[str, Any] = {"username": username}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/users/followings", body)
        return XFollowingsResponse(**data)

    def x_user_info(self, username: str) -> XUserInfoResponse:
        """
        Get detailed profile info for a single X/Twitter user.

        Powered by AttentionVC. $0.002 per request.

        Args:
            username: X/Twitter username (without @)

        Returns:
            XUserInfoResponse with detailed profile data
        """
        body: Dict[str, Any] = {"username": username}
        data = self._request_with_payment_raw("/v1/x/users/info", body)
        return XUserInfoResponse(**data)

    def x_verified_followers(
        self, user_id: str, *, cursor: Optional[str] = None
    ) -> XVerifiedFollowersResponse:
        """
        Get verified (blue-check) followers of an X/Twitter user.

        Powered by AttentionVC. $0.048 per page.

        Args:
            user_id: X/Twitter user ID (not username)
            cursor: Pagination cursor from previous response

        Returns:
            XVerifiedFollowersResponse with verified follower list
        """
        body: Dict[str, Any] = {"userId": user_id}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/users/verified-followers", body)
        return XVerifiedFollowersResponse(**data)

    def x_user_tweets(
        self,
        username: str,
        *,
        include_replies: bool = False,
        cursor: Optional[str] = None,
    ) -> XTweetsResponse:
        """
        Get tweets posted by an X/Twitter user.

        Powered by AttentionVC. $0.032 per page.

        Args:
            username: X/Twitter username (without @)
            include_replies: Include reply tweets (default: False)
            cursor: Pagination cursor from previous response

        Returns:
            XTweetsResponse with tweet list
        """
        body: Dict[str, Any] = {"username": username, "includeReplies": include_replies}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/users/tweets", body)
        return XTweetsResponse(**data)

    def x_user_mentions(
        self,
        username: str,
        *,
        since_time: Optional[str] = None,
        until_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> XMentionsResponse:
        """
        Get tweets that mention an X/Twitter user.

        Powered by AttentionVC. $0.032 per page.

        Args:
            username: X/Twitter username (without @)
            since_time: Start time filter (ISO8601 or Unix timestamp)
            until_time: End time filter (ISO8601 or Unix timestamp)
            cursor: Pagination cursor from previous response

        Returns:
            XMentionsResponse with mention tweets
        """
        body: Dict[str, Any] = {"username": username}
        if since_time is not None:
            body["sinceTime"] = since_time
        if until_time is not None:
            body["untilTime"] = until_time
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/users/mentions", body)
        return XMentionsResponse(**data)

    def x_tweet_lookup(self, tweet_ids: Union[List[str], str]) -> XTweetLookupResponse:
        """
        Fetch full tweet data for up to 200 tweet IDs.

        Powered by AttentionVC. $0.16 per batch.

        Args:
            tweet_ids: Single tweet ID or list of tweet IDs (max 200)

        Returns:
            XTweetLookupResponse with tweet data
        """
        if isinstance(tweet_ids, str):
            tweet_ids = [tweet_ids]

        body: Dict[str, Any] = {"tweet_ids": tweet_ids}
        data = self._request_with_payment_raw("/v1/x/tweets/lookup", body)
        return XTweetLookupResponse(**data)

    def x_tweet_replies(
        self,
        tweet_id: str,
        *,
        query_type: str = "Latest",
        cursor: Optional[str] = None,
    ) -> XTweetRepliesResponse:
        """
        Get replies to a specific tweet.

        Powered by AttentionVC. $0.032 per page.

        Args:
            tweet_id: The tweet ID to get replies for
            query_type: Sort order - 'Latest' or 'Default'
            cursor: Pagination cursor from previous response

        Returns:
            XTweetRepliesResponse with reply tweets
        """
        body: Dict[str, Any] = {"tweetId": tweet_id, "queryType": query_type}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/tweets/replies", body)
        return XTweetRepliesResponse(**data)

    def x_tweet_thread(
        self, tweet_id: str, *, cursor: Optional[str] = None
    ) -> XTweetThreadResponse:
        """
        Get the full thread context for a tweet.

        Powered by AttentionVC. $0.032 per page.

        Args:
            tweet_id: The tweet ID to get thread for
            cursor: Pagination cursor from previous response

        Returns:
            XTweetThreadResponse with thread tweets
        """
        body: Dict[str, Any] = {"tweetId": tweet_id}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/tweets/thread", body)
        return XTweetThreadResponse(**data)

    def x_search(
        self,
        query: str,
        *,
        query_type: str = "Latest",
        cursor: Optional[str] = None,
    ) -> XSearchResponse:
        """
        Search X/Twitter with advanced query operators.

        Powered by AttentionVC. $0.032 per page.

        Args:
            query: Search query (supports Twitter search operators)
            query_type: Sort order - 'Latest', 'Top', or 'Default'
            cursor: Pagination cursor from previous response

        Returns:
            XSearchResponse with matching tweets
        """
        body: Dict[str, Any] = {"query": query, "queryType": query_type}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/search", body)
        return XSearchResponse(**data)

    def x_trending(self) -> XTrendingResponse:
        """
        Get current trending topics on X/Twitter.

        Powered by AttentionVC. $0.002 per request.

        Returns:
            XTrendingResponse with trending topics
        """
        data = self._request_with_payment_raw("/v1/x/trending", {})
        return XTrendingResponse(**data)

    def x_articles_rising(self) -> XArticlesRisingResponse:
        """
        Get rising/viral articles from X/Twitter.

        Powered by AttentionVC intelligence layer. $0.05 per request.

        Returns:
            XArticlesRisingResponse with rising articles
        """
        data = self._request_with_payment_raw("/v1/x/articles/rising", {})
        return XArticlesRisingResponse(**data)

    def x_author_analytics(self, handle: str) -> XAuthorAnalyticsResponse:
        """
        Get author analytics and intelligence metrics for an X/Twitter user.

        Powered by AttentionVC intelligence layer. $0.02 per request.

        Args:
            handle: X/Twitter handle (without @)

        Returns:
            XAuthorAnalyticsResponse with analytics data
        """
        body: Dict[str, Any] = {"handle": handle}
        data = self._request_with_payment_raw("/v1/x/authors", body)
        return XAuthorAnalyticsResponse(**data)

    def x_compare_authors(self, handle1: str, handle2: str) -> XCompareAuthorsResponse:
        """
        Compare two X/Twitter authors side-by-side with intelligence metrics.

        Powered by AttentionVC intelligence layer. $0.05 per request.

        Args:
            handle1: First X/Twitter handle (without @)
            handle2: Second X/Twitter handle (without @)

        Returns:
            XCompareAuthorsResponse with comparison data
        """
        body: Dict[str, Any] = {"handle1": handle1, "handle2": handle2}
        data = self._request_with_payment_raw("/v1/x/compare", body)
        return XCompareAuthorsResponse(**data)

    # ── Prediction Markets (Powered by Predexon) ────────────────────────────

    def pm(self, path: str, **params: Any) -> Dict[str, Any]:
        """
        Query Predexon prediction market data (GET endpoints).

        Access real-time data across Polymarket, Kalshi, Limitless, Opinion,
        Predict.Fun, dFlow, sports, and Binance Futures. Powered by Predexon v2.
        Tier 1 = $0.001/call, Tier 2 = $0.005/call.

        Args:
            path: Endpoint path, e.g. "polymarket/events", "kalshi/markets/12345"
            **params: Query parameters passed to the endpoint

        Returns:
            Raw response dict from Predexon API

        Example:
            events = client.pm("polymarket/events")
            market = client.pm("kalshi/markets/KXBTC-25MAR14")
            results = client.pm("polymarket/search", q="bitcoin")
            # v2 canonical cross-venue
            markets = client.pm("markets", venue="polymarket", status="active")
            # v2 sports
            games = client.pm("sports/markets", league="NBA")
            # v2 wallet identity
            ident = client.pm("polymarket/wallet/identity/0xabc...")
        """
        return self._get_with_payment_raw(f"/v1/pm/{path}", params or None)

    def pm_query(self, path: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structured query for Predexon prediction market data (POST endpoints).

        For endpoints that require a JSON body, e.g. bulk wallet identity lookup.
        Tier 1 = $0.001/call, Tier 2 = $0.005/call.

        Args:
            path: Endpoint path, e.g. "polymarket/wallet/identities"
            query: JSON body for the structured query

        Returns:
            Raw response dict from Predexon API

        Example:
            # v2 bulk wallet identity (up to 200 addresses)
            batch = client.pm_query("polymarket/wallet/identities", {
                "addresses": ["0xabc...", "0xdef..."],
            })
        """
        return self._request_with_payment_raw(f"/v1/pm/{path}", query)

    # ── PM convenience helpers (Predexon v2) ────────────────────────────────
    # Thin wrappers over pm() / pm_query() for the most common v2 endpoints.
    # All accept arbitrary keyword filters that are forwarded as query params.

    def pm_markets(self, **params: Any) -> Dict[str, Any]:
        """List canonical cross-venue markets (Predexon v2).

        Filter with venue=, status=, category=, league=, event_id=,
        pagination_key=. Tier 1 ($0.001/call).
        """
        return self.pm("markets", **params)

    def pm_listings(self, **params: Any) -> Dict[str, Any]:
        """List venue-native executable listings flattened across canonical
        markets (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("markets/listings", **params)

    def pm_outcome(self, predexon_id: str) -> Dict[str, Any]:
        """Resolve a canonical Predexon outcome ID to its market context and
        venue listings (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm(f"outcomes/{predexon_id}")

    def pm_polymarket_markets_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket markets with cursor-based keyset pagination
        (use pagination_key=). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/markets/keyset", **params)

    def pm_polymarket_events_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket events with cursor-based keyset pagination
        (use pagination_key=). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/events/keyset", **params)

    def pm_sports_categories(self) -> Dict[str, Any]:
        """List available sports categories. Tier 1 ($0.001/call)."""
        return self.pm("sports/categories")

    def pm_sports_markets(self, **params: Any) -> Dict[str, Any]:
        """List sports markets grouped by game. Filter with league=,
        sport_type=, status=, venue=. Tier 1 ($0.001/call)."""
        return self.pm("sports/markets", **params)

    def pm_wallet_identity(self, wallet: str) -> Dict[str, Any]:
        """Fetch identity + profile metadata for one wallet (ENS, Twitter,
        portfolio, etc.). Tier 2 ($0.005/call)."""
        return self.pm(f"polymarket/wallet/identity/{wallet}")

    def pm_wallet_identities(self, addresses: List[str]) -> Dict[str, Any]:
        """Bulk identity lookup for up to 200 wallet addresses (POST).
        Tier 2 ($0.005/call)."""
        return self.pm_query("polymarket/wallet/identities", {"addresses": addresses})

    def pm_wallet_cluster(self, address: str) -> Dict[str, Any]:
        """Discover wallets connected to a seed address via on-chain transfers
        and identity proofs. Tier 2 ($0.005/call)."""
        return self.pm(f"polymarket/wallet/{address}/cluster")

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

    def is_testnet(self) -> bool:
        """Check if client is configured for testnet."""
        return "testnet.blockrun.ai" in self.api_url

    def get_balance(self) -> float:
        """
        Get USDC balance on Base network.

        Automatically detects mainnet vs testnet based on API URL:
        - Mainnet: Base (Chain ID 8453)
        - Testnet: Base Sepolia (Chain ID 84532)

        Returns:
            float: USDC balance (6 decimal places normalized)

        Example:
            balance = client.get_balance()
            print(f"Balance: ${balance:.2f} USDC")
        """
        # USDC contracts
        # Mainnet: Base
        # Testnet: Base Sepolia
        if self.is_testnet():
            usdc_contract = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
            rpcs = [
                "https://sepolia.base.org",
                "https://base-sepolia-rpc.publicnode.com",
            ]
        else:
            usdc_contract = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
            rpcs = [
                "https://base.publicnode.com",
                "https://mainnet.base.org",
                "https://base.meowrpc.com",
            ]

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

        last_error = None
        for rpc in rpcs:
            try:
                response = httpx.post(rpc, json=payload, timeout=10)
                result = response.json().get("result", "0x0")
                # Convert from hex and normalize (USDC has 6 decimals)
                balance_raw = int(result, 16)
                return balance_raw / 1_000_000
            except Exception as e:
                last_error = e
                continue

        # If all RPCs failed, raise the last error
        raise last_error or Exception("All RPCs failed")

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
            response = await client.chat("gpt-5.2", "Hello!")

        # For testnet:
        async with AsyncLLMClient(api_url="https://testnet.blockrun.ai/api") as client:
            response = await client.chat("openai/gpt-oss-20b", "Hello!")
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    TESTNET_API_URL = "https://testnet.blockrun.ai/api"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 120.0,
        search_timeout: float = 300.0,
    ):
        """
        Initialize the async BlockRun LLM client.

        Args:
            private_key: Base chain wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 120). Used for regular chat requests.
            search_timeout: Timeout for xAI Live Search requests (default: 300 = 5 minutes).
                           Auto-detected when search_parameters or search=True is passed.

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
        self.search_timeout = search_timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._last_call_cost: float = 0.0

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
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> ChatResponse:
        """Async full chat completion interface with optional xAI Live Search and tool calling."""
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

        # Handle tool calling
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        return await self._request_with_payment("/v1/chat/completions", body)

    async def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> ChatResponse:
        """Make async request with automatic payment handling."""
        url = f"{self.api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        response = await self._client.post(url, json=body, headers=req_headers)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            response = await self._client.post(url, json=body, headers=req_headers)

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
            network=details.get("network", "eip155:84532" if self.is_testnet() else "eip155:8453"),
            resource_url=validate_resource_url(
                resource.get("url", f"{self.api_url}/v1/chat/completions"), self.api_url
            ),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        # Retry with payment (x402 library expects PAYMENT-SIGNATURE header)
        # Use longer timeout for Live Search requests
        is_search_request = "search_parameters" in body or body.get("search") is True
        request_timeout = self.search_timeout if is_search_request else self.timeout

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = await self._client.post(
            url, json=body, headers=payment_headers, timeout=request_timeout
        )
        if retry_response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            retry_response = await self._client.post(
                url, json=body, headers=payment_headers, timeout=request_timeout
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

        # Extract cost and save locally
        price_info = {}
        try:
            resp_body = response.json()
            price_info = resp_body.get("price", {})
        except Exception:
            pass
        cost_usd = (
            float(price_info.get("amount", 0))
            if price_info
            else float(details.get("amount", 0)) / 1e6
        )
        self._last_call_cost = cost_usd

        response_data = retry_response.json()
        from .cache import save_to_cache

        save_to_cache("/v1/chat/completions", body, response_data, cost_usd=cost_usd)

        return ChatResponse(**response_data)

    async def _request_with_payment_raw(
        self, endpoint: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make async request with automatic payment handling, returning raw JSON."""
        from .cache import get_cached, save_to_cache

        # Check cache first
        cached = get_cached(endpoint, body)
        if cached is not None:
            return cached

        url = f"{self.api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        response = await self._client.post(url, json=body, headers=req_headers)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            response = await self._client.post(url, json=body, headers=req_headers)

        if response.status_code == 402:
            result = await self._handle_payment_and_retry_raw(url, body, response)
            save_to_cache(endpoint, body, result, cost_usd=self._last_call_cost)
            return result

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

    async def _handle_payment_and_retry_raw(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """Handle 402 response asynchronously for raw endpoints."""
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
            network=details.get("network", "eip155:84532" if self.is_testnet() else "eip155:8453"),
            resource_url=validate_resource_url(resource.get("url", url), self.api_url),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = await self._client.post(
            url, json=body, headers=payment_headers, timeout=self.timeout
        )
        if retry_response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            retry_response = await self._client.post(
                url, json=body, headers=payment_headers, timeout=self.timeout
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

        cost_usd = float(details.get("amount", 0)) / 1e6
        self._last_call_cost = cost_usd

        return retry_response.json()

    async def _get_with_payment_raw(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async GET with x402 payment handling, returning raw JSON."""
        from .cache import get_cached, save_to_cache

        cache_key_body = params or {}
        cached = get_cached(endpoint, cache_key_body)
        if cached is not None:
            return cached

        url = f"{self.api_url}{endpoint}"
        req_headers = {"User-Agent": _get_user_agent()}

        response = await self._client.get(url, params=params, headers=req_headers)

        if response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            response = await self._client.get(url, params=params, headers=req_headers)

        if response.status_code == 402:
            result = await self._handle_get_payment_and_retry(url, params, response)
            save_to_cache(endpoint, cache_key_body, result, cost_usd=self._last_call_cost)
            return result

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

    async def _handle_get_payment_and_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]],
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """Handle 402 response asynchronously for GET endpoints."""
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
            network=details.get("network", "eip155:84532" if self.is_testnet() else "eip155:8453"),
            resource_url=validate_resource_url(resource.get("url", url), self.api_url),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        payment_headers = {
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        retry_response = await self._client.get(
            url, params=params, headers=payment_headers, timeout=self.timeout
        )
        if retry_response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            retry_response = await self._client.get(
                url, params=params, headers=payment_headers, timeout=self.timeout
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

        cost_usd = float(details.get("amount", 0)) / 1e6
        self._last_call_cost = cost_usd

        return retry_response.json()

    async def image_edit(
        self,
        prompt: str,
        image: str,
        *,
        model: str = "openai/gpt-image-1",
        mask: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
    ) -> ImageResponse:
        """Async image editing (img2img)."""
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image": image,
            "size": size,
            "n": n,
        }
        if mask is not None:
            body["mask"] = mask

        data = await self._request_with_payment_raw("/v1/images/image2image", body)
        return ImageResponse(**data)

    async def search(
        self,
        query: str,
        *,
        sources: Optional[List[str]] = None,
        max_results: int = 10,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> SearchResult:
        """Async standalone search."""
        body: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }
        if sources is not None:
            body["sources"] = sources
        if from_date is not None:
            body["from_date"] = from_date
        if to_date is not None:
            body["to_date"] = to_date

        data = await self._request_with_payment_raw("/v1/search", body)
        return SearchResult(**data)

    async def x_user_lookup(self, usernames: Union[List[str], str]) -> XUserLookupResponse:
        """Async X/Twitter user lookup. Powered by AttentionVC."""
        if isinstance(usernames, str):
            usernames = [usernames]

        body: Dict[str, Any] = {"usernames": usernames}
        data = await self._request_with_payment_raw("/v1/x/users/lookup", body)
        return XUserLookupResponse(**data)

    async def x_followers(
        self, username: str, *, cursor: Optional[str] = None
    ) -> XFollowersResponse:
        """Async get X/Twitter followers. Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username}
        if cursor is not None:
            body["cursor"] = cursor

        data = await self._request_with_payment_raw("/v1/x/users/followers", body)
        return XFollowersResponse(**data)

    async def x_followings(
        self, username: str, *, cursor: Optional[str] = None
    ) -> XFollowingsResponse:
        """Async get X/Twitter followings. Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username}
        if cursor is not None:
            body["cursor"] = cursor

        data = await self._request_with_payment_raw("/v1/x/users/followings", body)
        return XFollowingsResponse(**data)

    async def x_user_info(self, username: str) -> XUserInfoResponse:
        """Async get single X/Twitter user info. Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username}
        data = await self._request_with_payment_raw("/v1/x/users/info", body)
        return XUserInfoResponse(**data)

    async def x_verified_followers(
        self, user_id: str, *, cursor: Optional[str] = None
    ) -> XVerifiedFollowersResponse:
        """Async get verified followers. Powered by AttentionVC."""
        body: Dict[str, Any] = {"userId": user_id}
        if cursor is not None:
            body["cursor"] = cursor
        data = await self._request_with_payment_raw("/v1/x/users/verified-followers", body)
        return XVerifiedFollowersResponse(**data)

    async def x_user_tweets(
        self, username: str, *, include_replies: bool = False, cursor: Optional[str] = None
    ) -> XTweetsResponse:
        """Async get user tweets. Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username, "includeReplies": include_replies}
        if cursor is not None:
            body["cursor"] = cursor
        data = await self._request_with_payment_raw("/v1/x/users/tweets", body)
        return XTweetsResponse(**data)

    async def x_user_mentions(
        self,
        username: str,
        *,
        since_time: Optional[str] = None,
        until_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> XMentionsResponse:
        """Async get user mentions. Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username}
        if since_time is not None:
            body["sinceTime"] = since_time
        if until_time is not None:
            body["untilTime"] = until_time
        if cursor is not None:
            body["cursor"] = cursor
        data = await self._request_with_payment_raw("/v1/x/users/mentions", body)
        return XMentionsResponse(**data)

    async def x_tweet_lookup(self, tweet_ids: Union[List[str], str]) -> XTweetLookupResponse:
        """Async batch tweet lookup. Powered by AttentionVC."""
        if isinstance(tweet_ids, str):
            tweet_ids = [tweet_ids]
        body: Dict[str, Any] = {"tweet_ids": tweet_ids}
        data = await self._request_with_payment_raw("/v1/x/tweets/lookup", body)
        return XTweetLookupResponse(**data)

    async def x_tweet_replies(
        self, tweet_id: str, *, query_type: str = "Latest", cursor: Optional[str] = None
    ) -> XTweetRepliesResponse:
        """Async get tweet replies. Powered by AttentionVC."""
        body: Dict[str, Any] = {"tweetId": tweet_id, "queryType": query_type}
        if cursor is not None:
            body["cursor"] = cursor
        data = await self._request_with_payment_raw("/v1/x/tweets/replies", body)
        return XTweetRepliesResponse(**data)

    async def x_tweet_thread(
        self, tweet_id: str, *, cursor: Optional[str] = None
    ) -> XTweetThreadResponse:
        """Async get tweet thread. Powered by AttentionVC."""
        body: Dict[str, Any] = {"tweetId": tweet_id}
        if cursor is not None:
            body["cursor"] = cursor
        data = await self._request_with_payment_raw("/v1/x/tweets/thread", body)
        return XTweetThreadResponse(**data)

    async def x_search(
        self, query: str, *, query_type: str = "Latest", cursor: Optional[str] = None
    ) -> XSearchResponse:
        """Async X/Twitter search. Powered by AttentionVC."""
        body: Dict[str, Any] = {"query": query, "queryType": query_type}
        if cursor is not None:
            body["cursor"] = cursor
        data = await self._request_with_payment_raw("/v1/x/search", body)
        return XSearchResponse(**data)

    async def x_trending(self) -> XTrendingResponse:
        """Async get trending topics. Powered by AttentionVC."""
        data = await self._request_with_payment_raw("/v1/x/trending", {})
        return XTrendingResponse(**data)

    async def x_articles_rising(self) -> XArticlesRisingResponse:
        """Async get rising articles. Powered by AttentionVC."""
        data = await self._request_with_payment_raw("/v1/x/articles/rising", {})
        return XArticlesRisingResponse(**data)

    async def x_author_analytics(self, handle: str) -> XAuthorAnalyticsResponse:
        """Async get author analytics. Powered by AttentionVC."""
        body: Dict[str, Any] = {"handle": handle}
        data = await self._request_with_payment_raw("/v1/x/authors", body)
        return XAuthorAnalyticsResponse(**data)

    async def x_compare_authors(self, handle1: str, handle2: str) -> XCompareAuthorsResponse:
        """Async compare two authors. Powered by AttentionVC."""
        body: Dict[str, Any] = {"handle1": handle1, "handle2": handle2}
        data = await self._request_with_payment_raw("/v1/x/compare", body)
        return XCompareAuthorsResponse(**data)

    # ── Prediction Markets (Powered by Predexon) ────────────────────────────

    async def pm(self, path: str, **params: Any) -> Dict[str, Any]:
        """Async query Predexon prediction market data (GET). Powered by Predexon."""
        return await self._get_with_payment_raw(f"/v1/pm/{path}", params or None)

    async def pm_query(self, path: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Async structured query for Predexon data (POST). Powered by Predexon."""
        return await self._request_with_payment_raw(f"/v1/pm/{path}", query)

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

    def is_testnet(self) -> bool:
        """Check if client is configured for testnet."""
        return "testnet.blockrun.ai" in self.api_url

    async def get_balance(self) -> float:
        """
        Get USDC balance on Base network.

        Automatically detects mainnet vs testnet based on API URL:
        - Mainnet: Base (Chain ID 8453)
        - Testnet: Base Sepolia (Chain ID 84532)

        Returns:
            float: USDC balance (6 decimal places normalized)

        Example:
            balance = await client.get_balance()
            print(f"Balance: ${balance:.2f} USDC")
        """
        # USDC contracts
        # Mainnet: Base
        # Testnet: Base Sepolia
        if self.is_testnet():
            usdc_contract = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
            rpcs = [
                "https://sepolia.base.org",
                "https://base-sepolia-rpc.publicnode.com",
            ]
        else:
            usdc_contract = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
            rpcs = [
                "https://base.publicnode.com",
                "https://mainnet.base.org",
                "https://base.meowrpc.com",
            ]

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

        last_error = None
        async with httpx.AsyncClient(timeout=10) as http_client:
            for rpc in rpcs:
                try:
                    response = await http_client.post(rpc, json=payload)
                    result = response.json().get("result", "0x0")
                    # Convert from hex and normalize (USDC has 6 decimals)
                    balance_raw = int(result, 16)
                    return balance_raw / 1_000_000
                except Exception as e:
                    last_error = e
                    continue

        # If all RPCs failed, raise the last error
        raise last_error or Exception("All RPCs failed")

    async def close(self):
        """Close the async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# =============================================================================
# Testnet Convenience Functions
# =============================================================================


def testnet_client(private_key: Optional[str] = None, **kwargs) -> LLMClient:
    """
    Create a testnet LLM client for development and testing.

    This is a convenience function that creates an LLMClient configured
    for the BlockRun testnet (Base Sepolia).

    Args:
        private_key: Base Sepolia wallet private key (or set BLOCKRUN_WALLET_KEY env var)
        **kwargs: Additional arguments passed to LLMClient

    Returns:
        LLMClient configured for testnet

    Example:
        from blockrun_llm import testnet_client

        client = testnet_client()  # Uses BLOCKRUN_WALLET_KEY
        response = client.chat("openai/gpt-oss-20b", "Hello!")

    Testnet Setup:
        1. Get testnet ETH from https://www.alchemy.com/faucets/base-sepolia
        2. Get testnet USDC from https://faucet.circle.com/
        3. Use your wallet with testnet funds

    Available Testnet Models:
        - openai/gpt-oss-20b
        - openai/gpt-oss-120b
    """
    return LLMClient(
        private_key=private_key,
        api_url=LLMClient.TESTNET_API_URL,
        **kwargs,
    )


async def async_testnet_client(private_key: Optional[str] = None, **kwargs) -> AsyncLLMClient:
    """
    Create an async testnet LLM client for development and testing.

    This is a convenience function that creates an AsyncLLMClient configured
    for the BlockRun testnet (Base Sepolia).

    Args:
        private_key: Base Sepolia wallet private key (or set BLOCKRUN_WALLET_KEY env var)
        **kwargs: Additional arguments passed to AsyncLLMClient

    Returns:
        AsyncLLMClient configured for testnet

    Example:
        from blockrun_llm import async_testnet_client

        async with async_testnet_client() as client:
            response = await client.chat("openai/gpt-oss-20b", "Hello!")
    """
    return AsyncLLMClient(
        private_key=private_key,
        api_url=AsyncLLMClient.TESTNET_API_URL,
        **kwargs,
    )
