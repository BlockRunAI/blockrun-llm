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
import sys
import json as _json
from typing import AsyncIterator, Iterator, List, Dict, Any, Optional, Tuple, Union
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import (
    ChatResponse,
    ChatCompletionChunk,
    ImageResponse,
    APIError,
    PaymentError,
    RoutingDecision,
    SmartChatResponse,
    RoutingProfile,
    SearchResult,
    stream_choice_content,
    stream_choice_finish_reason,
    chunk_usage_dict,
)
from .router import route as route_request
from .tx_log import TransactionLogger, decode_settlement_header, _resolve_log_dir
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_eth_address,
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
    List available image generation models without requiring a wallet.

    Filters the unified ``/v1/models`` catalog by ``categories: ["image"]``.
    The dedicated ``/v1/images/models`` endpoint was deprecated server-side;
    image models now live alongside chat models under one catalog.
    """
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{api_url.rstrip('/')}/v1/models")
        if response.status_code != 200:
            raise APIError(
                f"Failed to list models: {response.status_code}",
                response.status_code,
                {},
            )
        models = response.json().get("data", [])
    return [m for m in models if "image" in (m.get("categories") or [])]


# =============================================================================
# Shared helpers
# =============================================================================


def _should_fallback(exc: Exception) -> bool:
    """Whether ``exc`` is the kind of transient failure that warrants trying
    the next model in a fallback chain.

    True for: timeouts, network/connection errors, and APIError with 5xx
    status codes typically associated with upstream availability problems.

    False for: 4xx client errors, PaymentError (wallet/balance issues), and
    everything else — those are not "swap upstream and retry" situations.
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.NetworkError):
        return True
    if isinstance(exc, APIError) and exc.status_code in (502, 503, 504, 522, 524):
        return True
    return False


def _detect_network(api_url: str) -> str:
    """Map an API URL to the canonical network label used in billing
    records. Returns ``base-mainnet`` / ``base-sepolia`` / ``solana-mainnet``
    / ``unknown``.
    """
    if not api_url:
        return "unknown"
    if "sol.blockrun" in api_url:
        return "solana-mainnet"
    if "testnet" in api_url:
        return "base-sepolia"
    if "blockrun.ai" in api_url:
        return "base-mainnet"
    return "unknown"


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
        transaction_log: Union[bool, str, "os.PathLike[str]", None] = None,
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
            transaction_log: Opt-in per-call log written to a project folder.
                           ``True`` → ``./log/``; pass a string/Path for a custom dir;
                           ``None`` (default) honors the ``BLOCKRUN_TX_LOG`` env var
                           (set to ``1`` or a path). Each paid call appends one row to
                           ``transactions.jsonl`` (model, input, output, cost_usd,
                           tx_hash, on-chain amount, payer, payee, network) and
                           writes a pretty-printed JSON file next to it.

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

        self._client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

        # Session spending tracking
        self._session_total_usd: float = 0.0
        self._session_calls: int = 0
        self._last_call_cost: float = 0.0

        # Model pricing cache for smart routing
        self._model_pricing_cache: Optional[Dict[str, Dict[str, float]]] = None

        # Opt-in transaction log + last on-chain settlement payload. The
        # settlement is populated from X-PAYMENT-RESPONSE on every paid retry
        # and cleared right before save_to_cache fires so it can't bleed
        # across calls when logging is disabled.
        log_dir = _resolve_log_dir(transaction_log)
        self._tx_logger: Optional[TransactionLogger] = (
            TransactionLogger(log_dir) if log_dir is not None else None
        )
        self._last_settlement: Optional[Dict[str, Any]] = None

    def _capture_settlement(self, response: httpx.Response) -> Optional[Dict[str, Any]]:
        """Decode the x402 settlement header on a successful paid response.

        Returns the decoded settlement dict (also stashed on
        ``self._last_settlement``) so callers can pass it straight into
        ``save_to_cache``. ``None`` when the facilitator didn't include a
        settlement header — older facilitators / cached free responses.
        """
        header = response.headers.get("x-payment-response") or response.headers.get(
            "X-PAYMENT-RESPONSE"
        )
        settlement = decode_settlement_header(header)
        self._last_settlement = settlement
        return settlement

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

        # Make the chat request with selected model. Pass the tier's remaining
        # models as fallbacks so a hung upstream (e.g. NVIDIA NIM) doesn't
        # hard-fail when smart_chat could just walk to the next visible model.
        response = self.chat(
            model=decision["model"],
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            fallback_models=decision.get("fallbacks") or None,
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
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        **extra: Any,
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
            response_format=response_format,
            stop=stop,
            fallback_models=fallback_models,
            **extra,
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
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        **extra: Any,
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
            response_format: OpenAI response format, e.g. {"type": "json_object"} for JSON mode.
                Works across all providers — the gateway natively forwards it to OpenAI/Azure
                and injects a raw-JSON system instruction (stripping any code fence) for
                Anthropic/Bedrock models.
            stop: Up to 4 stop sequences (str or list of str). The gateway forwards these
                natively to OpenAI and maps them to stop_sequences for Anthropic/Bedrock.

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

        # OpenAI-compatible response shaping (honored by the gateway across providers)
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop

        # Passthrough: forward any other caller-supplied params verbatim. Named
        # params above take precedence; `extra` only fills keys not already set.
        for k, v in extra.items():
            if v is not None:
                body.setdefault(k, v)

        # Walk [model, *fallback_models] on retriable errors (timeouts, 5xx,
        # network errors). Default behavior — single attempt — is preserved
        # when fallback_models is None or empty.
        attempts = [model, *(fallback_models or [])]
        last_exc: Optional[Exception] = None
        for i, attempt_model in enumerate(attempts):
            body["model"] = attempt_model
            try:
                return self._request_with_payment("/v1/chat/completions", body)
            except Exception as exc:
                if not _should_fallback(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] {attempt_model} -> {next_model} "
                        f"({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        # Exhausted all attempts — re-raise the last retriable error.
        assert last_exc is not None  # at least one attempt always runs
        raise last_exc

    # ------------------------------------------------------------------
    # Streaming (SSE) chat completions
    # ------------------------------------------------------------------

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        **extra: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """
        Stream a chat completion via Server-Sent Events.

        Yields one :class:`ChatCompletionChunk` per SSE ``data:`` line until
        the upstream emits ``data: [DONE]``. The first chunk's ``delta`` is
        typically ``{"role": "assistant"}``; subsequent chunks carry
        ``content`` deltas; the final chunk carries ``finish_reason``.

        Payment flow is the same as :meth:`chat_completion`: the first
        request returns 402, the SDK signs an EIP-712 payment locally, then
        re-issues the request with ``stream=true`` and the
        ``PAYMENT-SIGNATURE`` header. Free models (e.g.
        ``nvidia/deepseek-v4-flash``) skip the 402 and stream directly.

        Fallback semantics
        ------------------
        ``fallback_models=[...]`` walks the list when the primary upstream
        produces a retriable error (timeouts, network errors, 5xx). Unlike
        the non-streaming :meth:`chat_completion` path, fallback is only
        possible **before the first chunk is yielded** — once any byte has
        reached the caller, switching models would concatenate two distinct
        responses. After-first-chunk failures propagate to the caller.

        Example::

            for chunk in client.chat_completion_stream(
                "nvidia/deepseek-v4-flash",
                [{"role": "user", "content": "Hello"}],
                fallback_models=["nvidia/llama-4-maverick"],
            ):
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)

        Note: ``search`` / ``search_parameters`` are not supported in stream
        mode by the BlockRun backend — the server will reject with 400.
        Codex / GPT-5.4 Pro also do not support streaming.
        """
        validate_model(model)
        validate_max_tokens(max_tokens)
        validate_temperature(temperature)
        validate_top_p(top_p)

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if search_parameters is not None:
            body["search_parameters"] = search_parameters
        elif search is True:
            body["search_parameters"] = {"mode": "on"}
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop

        # Passthrough: forward any other caller-supplied params verbatim.
        for k, v in extra.items():
            if v is not None:
                body.setdefault(k, v)

        attempts = [model, *(fallback_models or [])]
        last_exc: Optional[Exception] = None

        for i, attempt_model in enumerate(attempts):
            body["model"] = attempt_model
            inner = self._stream_with_payment("/v1/chat/completions", body)
            chunks_yielded = 0
            try:
                for chunk in inner:
                    chunks_yielded += 1
                    yield chunk
                return  # finished cleanly
            except Exception as exc:
                if chunks_yielded > 0:
                    # Already streamed partial output; can't swap models now.
                    raise
                if not _should_fallback(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] stream {attempt_model} -> {next_model} "
                        f"({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        # Exhausted all attempts — re-raise the last retriable error.
        assert last_exc is not None  # at least one attempt always runs
        raise last_exc

    # Streaming retry policy. Both the probe (unauthenticated) and the
    # paid-retry (with PAYMENT-SIGNATURE) honor this — total tries per
    # phase is ``1 + len(_STREAM_5XX_BACKOFFS)`` (== 4 here). Exponential
    # backoff so we don't hammer a struggling upstream.
    _STREAM_5XX_STATUSES = (500, 502, 503, 504)
    _STREAM_5XX_BACKOFFS = (1.0, 2.0, 4.0)

    def _stream_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
    ) -> Iterator[ChatCompletionChunk]:
        """
        Run the 402 → sign → retry dance, then yield SSE chunks.

        Free models return 200 + SSE on the first request; paid models
        return JSON 402 first, after which we sign locally and re-stream.
        Transient 5xx responses (NVIDIA NIM hiccups, etc.) are retried
        in-band with exponential backoff before raising.
        """
        url = f"{self.api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        is_search = "search_parameters" in body or body.get("search") is True
        timeout = self.search_timeout if is_search else self.timeout

        # ----- Phase 1: probe (no payment header) -----
        payment_headers: Optional[Dict[str, str]] = None
        cost_usd = 0.0

        backoffs = self._STREAM_5XX_BACKOFFS
        for attempt in range(len(backoffs) + 1):
            with self._client.stream(
                "POST", url, json=body, headers=req_headers, timeout=timeout
            ) as resp1:
                if resp1.status_code == 200:
                    # Free model (or already-authed session) — stream directly.
                    yield from self._iter_sse_chunks(resp1)
                    return
                resp1.read()
                if resp1.status_code == 402:
                    payment_headers, cost_usd = self._sign_payment_from_response(body, resp1)
                    break  # advance to phase 2
                if resp1.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import time

                    time.sleep(backoffs[attempt])
                    continue
                # Out of retries on 5xx, or non-retriable 4xx.
                self._raise_stream_error(resp1, after_payment=False)
        else:
            # Loop exhausted without 402 or 200 — shouldn't reach here because
            # the final iteration above raises, but defensive.
            raise APIError("stream probe exhausted retries", 0, None)

        # ----- Phase 2: stream with PAYMENT-SIGNATURE -----
        assert payment_headers is not None  # break implies signing succeeded
        for attempt in range(len(backoffs) + 1):
            with self._client.stream(
                "POST", url, json=body, headers=payment_headers, timeout=timeout
            ) as resp2:
                if resp2.status_code == 200:
                    if cost_usd > 0:
                        self._session_calls += 1
                        self._session_total_usd += cost_usd
                        self._last_call_cost = cost_usd
                        self._capture_settlement(resp2)
                    yield from self._iter_and_archive(resp2, body, cost_usd, streaming=True)
                    return
                resp2.read()
                if resp2.status_code == 402:
                    raise PaymentError("Payment was rejected. Check your wallet balance.")
                if resp2.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import time

                    time.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp2, after_payment=True)

    def _iter_and_archive(
        self,
        response: httpx.Response,
        body: Dict[str, Any],
        cost_usd: float,
        *,
        streaming: bool = True,
    ) -> Iterator[ChatCompletionChunk]:
        """Yield each SSE chunk, accumulate content for the local archive,
        then once ``data: [DONE]`` arrives ``save_to_cache`` the assembled
        ``chat.completion`` response so paid streaming calls show up in
        ``~/.blockrun/cost_log.jsonl`` and ``~/.blockrun/data/`` the same
        way non-stream paid calls do."""
        assembled_id: Optional[str] = None
        assembled_model: Optional[str] = None
        assembled_created: int = 0
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None

        for chunk in self._iter_sse_chunks(response):
            if chunk.choices:
                choice = chunk.choices[0]
                content = stream_choice_content(choice)
                if content:
                    content_parts.append(content)
                fr = stream_choice_finish_reason(choice)
                if fr:
                    finish_reason = fr
            if assembled_id is None and chunk.id:
                assembled_id = chunk.id
                assembled_model = chunk.model
                assembled_created = chunk.created
            _usage = chunk_usage_dict(chunk)
            if _usage is not None:
                usage_dict = _usage
            yield chunk

        # Stream complete (saw [DONE]). Free models have cost_usd == 0; only
        # archive paid calls to mirror the non-stream save_to_cache path.
        if cost_usd > 0:
            from .cache import save_to_cache

            response_data: Dict[str, Any] = {
                "id": assembled_id or "stream",
                "object": "chat.completion",
                "created": assembled_created or int(__import__("time").time()),
                "model": assembled_model or body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "".join(content_parts),
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "stream": streaming,
            }
            if usage_dict:
                response_data["usage"] = usage_dict
            try:
                save_to_cache(
                    "/v1/chat/completions",
                    body,
                    response_data,
                    cost_usd=cost_usd,
                    **self._billing_meta(),
                )
            except Exception:
                # Logging never breaks the call.
                pass
            self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)

    @staticmethod
    def _iter_sse_chunks(response: httpx.Response) -> Iterator[ChatCompletionChunk]:
        """Parse a ``text/event-stream`` response into chunk objects.

        OpenAI format: each event is ``data: {json}\\n\\n``; the terminator is
        ``data: [DONE]\\n\\n``. Non-``data:`` lines (comments, heartbeats)
        are ignored, and malformed chunks are skipped rather than abort the
        stream — partial output is still useful.
        """
        for raw_line in response.iter_lines():
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[6:].strip()
            if payload == "[DONE]":
                return
            try:
                chunk_dict = _json.loads(payload)
            except Exception:
                continue
            try:
                yield ChatCompletionChunk(**chunk_dict)
            except Exception:
                # Schema drift — surface the raw dict shape via a permissive
                # model construction to avoid silently dropping output.
                yield ChatCompletionChunk.model_construct(**chunk_dict)

    def _sign_payment_from_response(
        self,
        body: Dict[str, Any],
        response: httpx.Response,
    ) -> Tuple[Dict[str, str], float]:
        """
        Extract a 402's payment requirements, sign locally, and return
        ``(headers_with_PAYMENT_SIGNATURE, cost_usd)``.

        Mirrors the inline signing logic in :meth:`_handle_payment_and_retry`
        but returns the signed headers instead of doing the retry POST —
        which lets the streaming path open an SSE connection for the retry.
        """
        payment_header = response.headers.get("payment-required")
        price_info: Dict[str, Any] = {}
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
            resource_url=validate_resource_url(
                resource.get("url", f"{self.api_url}/v1/chat/completions"), self.api_url
            ),
            resource_description=resource.get("description", "BlockRun AI API call"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
            asset=details.get("asset"),
        )

        return (
            {
                "Content-Type": "application/json",
                "User-Agent": _get_user_agent(),
                "PAYMENT-SIGNATURE": payment_payload,
            },
            cost_usd,
        )

    @staticmethod
    def _raise_stream_error(response: httpx.Response, *, after_payment: bool) -> None:
        """Common error path for unexpected HTTP statuses during streaming."""
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": "Stream request failed"}
        prefix = "API error after payment" if after_payment else "API error"
        raise APIError(
            f"{prefix}: {response.status_code}",
            response.status_code,
            sanitize_error_response(error_body),
        )

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
        self._capture_settlement(retry_response)

        # Save full response locally (cost log + response archive)
        from .cache import save_to_cache

        save_to_cache(
            "/v1/chat/completions",
            body,
            response_data,
            cost_usd=cost_usd,
            **self._billing_meta(),
        )
        self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)

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
            save_to_cache(
                endpoint,
                body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            self._log_transaction(endpoint, body, result, self._last_call_cost)
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
        self._capture_settlement(retry_response)

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
            save_to_cache(
                endpoint,
                cache_key_body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            self._log_transaction(endpoint, cache_key_body, result, self._last_call_cost)
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
        self._capture_settlement(retry_response)

        return retry_response.json()

    def image_edit(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        model: str = "openai/gpt-image-2",
        mask: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
    ) -> ImageResponse:
        """
        Edit an image using img2img, or fuse multiple source images.

        Args:
            prompt: Text description of the desired edit
            image: A single base64 "data:image/...;base64,..." data URI, or a
                   list of 1-4 such data URIs to fuse multiple sources. Plain
                   URLs are not accepted — the source must be a data URI.
            model: Model ID (default: "openai/gpt-image-2")
                   Edit-supported: "openai/gpt-image-1", "openai/gpt-image-2",
                                   "google/nano-banana", "google/nano-banana-pro".
                   Multi-image caps: openai/* up to 4, google/* up to 3.
            mask: Optional base64-encoded mask image (OpenAI gpt-image-* only;
                  cannot be combined with multiple source images).
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

    def pm_polymarket_markets(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket markets (Predexon v2). Tier 1 ($0.001/call).

        For high-volume traversal use ``pm_polymarket_markets_keyset()``.
        """
        return self.pm("polymarket/markets", **params)

    def pm_polymarket_events(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket events (Predexon v2). Tier 1 ($0.001/call).

        For high-volume traversal use ``pm_polymarket_events_keyset()``.
        """
        return self.pm("polymarket/events", **params)

    def pm_polymarket_markets_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket markets with cursor-based keyset pagination
        (use pagination_key=). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/markets/keyset", **params)

    def pm_polymarket_events_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket events with cursor-based keyset pagination
        (use pagination_key=). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/events/keyset", **params)

    def pm_polymarket_positions(self, **params: Any) -> Dict[str, Any]:
        """Polymarket open positions (per-wallet, market-level PnL).
        Tier 1 ($0.001/call)."""
        return self.pm("polymarket/positions", **params)

    def pm_polymarket_trades(self, **params: Any) -> Dict[str, Any]:
        """Recent Polymarket trades (token, side, shares, price, tx_hash).
        Tier 1 ($0.001/call)."""
        return self.pm("polymarket/trades", **params)

    def pm_polymarket_leaderboard(self, **params: Any) -> Dict[str, Any]:
        """Polymarket trader leaderboard (rank by window, sort_by).
        Tier 1 ($0.001/call)."""
        return self.pm("polymarket/leaderboard", **params)

    def pm_kalshi_markets(self, **params: Any) -> Dict[str, Any]:
        """List Kalshi markets (CFTC-regulated event contracts).
        Tier 1 ($0.001/call)."""
        return self.pm("kalshi/markets", **params)

    def pm_limitless_markets(self, **params: Any) -> Dict[str, Any]:
        """List Limitless markets (binary AMM-style outcomes).
        Tier 1 ($0.001/call)."""
        return self.pm("limitless/markets", **params)

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

    # ── DefiLlama (DeFi protocols / TVL / yields / prices) ──────────────────

    def defi(self, path: str, **params: Any) -> Dict[str, Any]:
        """
        Query DefiLlama DeFi data (GET passthrough). Powered by DefiLlama.

        $0.005/call for protocols / protocol/{slug} / chains / yields;
        $0.001/call for prices/{coins}.

        Args:
            path: Endpoint path — "protocols", "protocol/{slug}", "chains",
                "yields", or "prices/{coins}" (coins comma-separated, e.g.
                "coingecko:bitcoin,base:0x...").
            **params: Query parameters passed through to DefiLlama.

        Example::

            protocols = client.defi("protocols")
            aave = client.defi("protocol/aave")
        """
        return self._get_with_payment_raw(f"/v1/defillama/{path}", params or None)

    def defi_protocols(self) -> Dict[str, Any]:
        """All DeFi protocols with TVL ($0.005/call)."""
        return self.defi("protocols")

    def defi_protocol(self, slug: str) -> Dict[str, Any]:
        """Single protocol details + historical TVL ($0.005/call)."""
        return self.defi(f"protocol/{slug}")

    def defi_chains(self) -> Dict[str, Any]:
        """Current TVL of every chain ($0.005/call)."""
        return self.defi("chains")

    def defi_yields(self, **params: Any) -> Dict[str, Any]:
        """Yield pools with APY/TVL ($0.005/call)."""
        return self.defi("yields", **params)

    def defi_prices(self, coins: Union[List[str], str]) -> Dict[str, Any]:
        """Token price lookup ($0.001/call).

        Args:
            coins: Coin ids like "coingecko:bitcoin" or "{chain}:{address}" —
                a list or a pre-joined comma-separated string.
        """
        joined = ",".join(coins) if isinstance(coins, list) else coins
        return self.defi(f"prices/{joined}")

    # ── 0x DEX (swap quotes + gasless) — free passthrough ───────────────────

    def dex(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Query the 0x Swap / Gasless APIs (free — no x402 payment; BlockRun
        takes an on-chain affiliate fee on executed swaps instead).

        Args:
            path: Endpoint path — "price", "quote", "gasless/price",
                "gasless/quote", "gasless/submit" (POST), "gasless/status/{hash}",
                "gasless/approval-tokens", "gasless/chains", "swap/chains".
            method: "GET" (default) or "POST" (gasless/submit only).
            body: JSON body for POST endpoints.
            **params: Query parameters (chainId, sellToken, buyToken,
                sellAmount, taker, ...).

        Example::

            quote = client.dex("quote", chainId=8453,
                               sellToken="0x...", buyToken="0x...",
                               sellAmount="1000000", taker="0x...")
        """
        endpoint = f"/v1/zerox/{path}"
        if method.upper() == "POST":
            return self._request_with_payment_raw(endpoint, body or {})
        return self._get_with_payment_raw(endpoint, params or None)

    def dex_price(self, **params: Any) -> Dict[str, Any]:
        """Indicative Permit2 swap price — no commitment (free)."""
        return self.dex("price", **params)

    def dex_quote(self, **params: Any) -> Dict[str, Any]:
        """Firm Permit2 swap quote with permit2.eip712 + tx data (free)."""
        return self.dex("quote", **params)

    def dex_gasless_price(self, **params: Any) -> Dict[str, Any]:
        """Gasless indicative price quote (free)."""
        return self.dex("gasless/price", **params)

    def dex_gasless_quote(self, **params: Any) -> Dict[str, Any]:
        """Gasless firm quote — returns trade.eip712 to sign (free)."""
        return self.dex("gasless/quote", **params)

    def dex_gasless_submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a signed gasless trade; the 0x relayer pays gas (free)."""
        return self.dex("gasless/submit", method="POST", body=body)

    def dex_gasless_status(self, trade_hash: str) -> Dict[str, Any]:
        """Poll a gasless trade's status by tradeHash (free)."""
        return self.dex(f"gasless/status/{trade_hash}")

    def dex_chains(self) -> Dict[str, Any]:
        """Chains where the Swap API is supported (free)."""
        return self.dex("swap/chains")

    def dex_gasless_chains(self) -> Dict[str, Any]:
        """Chains where the Gasless API is supported (free)."""
        return self.dex("gasless/chains")

    # ── Modal Sandbox (pay-per-call cloud compute) ───────────────────────────

    def modal(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call the Modal sandbox compute API (POST passthrough).

        Args:
            path: "sandbox/create" ($0.01 CPU / $0.05 GPU), "sandbox/exec"
                ($0.001), "sandbox/status" ($0.001), "sandbox/terminate" ($0.001).
            body: JSON body for the endpoint.
        """
        return self._request_with_payment_raw(f"/v1/modal/{path}", body or {})

    def modal_sandbox_create(self, **body: Any) -> Dict[str, Any]:
        """Create a sandboxed compute environment ($0.01 CPU / $0.05 GPU).

        Common fields: image ("python:3.11"), gpu (optional GPU type),
        timeout. Returns a sandbox_id for exec/status/terminate.
        """
        return self.modal("sandbox/create", body)

    def modal_sandbox_exec(
        self, sandbox_id: str, command: List[str], **body: Any
    ) -> Dict[str, Any]:
        """Execute a command in a sandbox; returns stdout/stderr ($0.001)."""
        return self.modal("sandbox/exec", {"sandbox_id": sandbox_id, "command": command, **body})

    def modal_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Check a sandbox's status ($0.001)."""
        return self.modal("sandbox/status", {"sandbox_id": sandbox_id})

    def modal_sandbox_terminate(self, sandbox_id: str) -> Dict[str, Any]:
        """Terminate a sandbox ($0.001)."""
        return self.modal("sandbox/terminate", {"sandbox_id": sandbox_id})

    # ── Coinbase Onramp ──────────────────────────────────────────────────────

    def onramp(self, address: str) -> Dict[str, Any]:
        """Mint a one-time Coinbase Onramp link to fund a wallet with fiat (FREE).

        Opens the door to buying Base USDC with a card or bank (60+ fiat
        currencies) via pay.coinbase.com. FREE — the x402 signature only
        authenticates the wallet, so the funding ``address`` MUST equal the
        signing wallet (use ``client.get_wallet_address()``). Base / USDC only.

        The returned URL is single-use and expires in ~5 minutes, so mint it at
        click time and never cache it.

        Args:
            address: Destination wallet (0x-prefixed Base address). Must match
                the signing wallet, since the link funds that exact address.

        Returns:
            Dict with a ``url`` pointing at ``https://pay.coinbase.com/``.

        Example::

            link = client.onramp(client.get_wallet_address())
            print(link["url"])  # open in a browser to buy USDC on Base
        """
        validate_eth_address(address)
        data = self._request_with_payment_raw(
            "/v1/onramp/token",
            {"address": address, "network": "base", "asset": "USDC"},
        )
        url = data.get("url") if isinstance(data, dict) else None
        if not isinstance(url, str) or not url.startswith("https://pay.coinbase.com/"):
            raise APIError("gateway returned no onramp url", 0, None)
        return data

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
            List of image model information dicts (id, name, pricing, etc.)

        Notes:
            The dedicated ``/v1/images/models`` endpoint was deprecated
            server-side; the catalog now lives in ``/v1/models`` with
            ``categories: ["image", ...]``. This method filters the unified
            catalog so existing callers keep working.
        """
        return [m for m in self.list_models() if "image" in (m.get("categories") or [])]

    def list_all_models(self) -> List[Dict[str, Any]]:
        """
        List all available models (chat, image, music, etc.) with pricing.

        Returns:
            List of all model information dicts with a ``type`` field set to
            the first category (``llm`` for chat, ``image`` / ``music`` /
            ``audio`` etc. for media). Backwards-compat: chat models always
            report ``type: "llm"``.
        """
        all_models = self.list_models()
        for m in all_models:
            cats = m.get("categories") or []
            if "chat" in cats:
                m["type"] = "llm"
            elif "image" in cats:
                m["type"] = "image"
            elif "music" in cats or "audio" in cats:
                m["type"] = "music"
            else:
                m["type"] = cats[0] if cats else "llm"
        return all_models

    def get_wallet_address(self) -> str:
        """Get the wallet address being used for payments."""
        return self.account.address

    def is_testnet(self) -> bool:
        """Check if client is configured for testnet."""
        return "testnet.blockrun.ai" in self.api_url

    def _billing_meta(self) -> Dict[str, Optional[str]]:
        """Return billing metadata (wallet / network / client_kind) for the
        cost log. Used by ``save_to_cache`` call sites."""
        return {
            "wallet": self.account.address,
            "network": _detect_network(self.api_url),
            "client_kind": type(self).__name__,
        }

    def _log_transaction(
        self,
        endpoint: str,
        body: Dict[str, Any],
        response: Any,
        cost_usd: float,
    ) -> None:
        """Append one row to the project-local transaction log, if enabled.

        Pulls the on-chain settlement out of ``self._last_settlement``
        (captured from ``X-PAYMENT-RESPONSE`` on the paid retry) and
        consumes it — so a subsequent free / cached call right after a
        paid one cannot reuse stale tx fields. No-op when the logger is
        disabled; never raises (best-effort logging by design)."""
        logger = self._tx_logger
        if logger is None:
            return
        settlement = self._last_settlement
        self._last_settlement = None
        try:
            logger.log(
                endpoint=endpoint,
                request=body,
                response=response,
                cost_usd=cost_usd,
                model=(body.get("model") if isinstance(body, dict) else None),
                wallet=self.account.address,
                network=_detect_network(self.api_url),
                client_kind=type(self).__name__,
                settlement=settlement,
            )
        except Exception:
            pass

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
        transaction_log: Union[bool, str, "os.PathLike[str]", None] = None,
    ):
        """
        Initialize the async BlockRun LLM client.

        Args:
            private_key: Base chain wallet private key (or set BLOCKRUN_WALLET_KEY env var)
            api_url: API endpoint URL (default: https://blockrun.ai/api)
            timeout: Request timeout in seconds (default: 120). Used for regular chat requests.
            search_timeout: Timeout for xAI Live Search requests (default: 300 = 5 minutes).
                           Auto-detected when search_parameters or search=True is passed.
            transaction_log: Same opt-in per-call log as ``LLMClient``. ``True`` →
                           ``./log/``; pass a string/Path for a custom dir; ``None``
                           honors the ``BLOCKRUN_TX_LOG`` env var. See ``LLMClient``
                           for the full record schema.

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
        # Default httpx pool (max_connections=100) is exhausted by ~50 concurrent
        # paid requests because each request uses two HTTP connections: Phase 1
        # (402 probe) + Phase 2 (authenticated SSE stream).  Raise the limit so
        # high-concurrency deployments don't hit pool exhaustion before hitting
        # any upstream rate limit.
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        self._last_call_cost: float = 0.0

        log_dir = _resolve_log_dir(transaction_log)
        self._tx_logger: Optional[TransactionLogger] = (
            TransactionLogger(log_dir) if log_dir is not None else None
        )
        self._last_settlement: Optional[Dict[str, Any]] = None

    def _capture_settlement(self, response: httpx.Response) -> Optional[Dict[str, Any]]:
        """Async-client twin of :meth:`LLMClient._capture_settlement`."""
        header = response.headers.get("x-payment-response") or response.headers.get(
            "X-PAYMENT-RESPONSE"
        )
        settlement = decode_settlement_header(header)
        self._last_settlement = settlement
        return settlement

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
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        **extra: Any,
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
            response_format=response_format,
            stop=stop,
            fallback_models=fallback_models,
            **extra,
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
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        **extra: Any,
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

        # OpenAI-compatible response shaping (honored by the gateway across providers)
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop

        # Passthrough: forward any other caller-supplied params verbatim.
        for k, v in extra.items():
            if v is not None:
                body.setdefault(k, v)

        # Walk [model, *fallback_models] on retriable errors. See sync
        # chat_completion() above for the rationale.
        attempts = [model, *(fallback_models or [])]
        last_exc: Optional[Exception] = None
        for i, attempt_model in enumerate(attempts):
            body["model"] = attempt_model
            try:
                return await self._request_with_payment("/v1/chat/completions", body)
            except Exception as exc:
                if not _should_fallback(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] {attempt_model} -> {next_model} "
                        f"({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Streaming (SSE) chat completions — async mirror of LLMClient
    # ------------------------------------------------------------------

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        search: Optional[bool] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        **extra: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Async streaming chat completion. See :meth:`LLMClient.chat_completion_stream`
        for protocol details and the ``fallback_models`` semantics —
        identical here, only the iteration protocol differs (``async for``).
        """
        validate_model(model)
        validate_max_tokens(max_tokens)
        validate_temperature(temperature)
        validate_top_p(top_p)

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if search_parameters is not None:
            body["search_parameters"] = search_parameters
        elif search is True:
            body["search_parameters"] = {"mode": "on"}
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop

        # Passthrough: forward any other caller-supplied params verbatim.
        for k, v in extra.items():
            if v is not None:
                body.setdefault(k, v)

        attempts = [model, *(fallback_models or [])]
        last_exc: Optional[Exception] = None

        for i, attempt_model in enumerate(attempts):
            body["model"] = attempt_model
            inner = self._stream_with_payment("/v1/chat/completions", body)
            chunks_yielded = 0
            try:
                async for chunk in inner:
                    chunks_yielded += 1
                    yield chunk
                return
            except Exception as exc:
                if chunks_yielded > 0:
                    raise
                if not _should_fallback(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] stream {attempt_model} -> {next_model} "
                        f"({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        assert last_exc is not None
        raise last_exc

    async def _stream_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Async version of LLMClient._stream_with_payment.

        Honors :data:`LLMClient._STREAM_5XX_STATUSES` and
        :data:`LLMClient._STREAM_5XX_BACKOFFS` for retries (in-band exponential
        backoff on transient upstream errors before raising).
        """
        url = f"{self.api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        is_search = "search_parameters" in body or body.get("search") is True
        timeout = self.search_timeout if is_search else self.timeout

        backoffs = LLMClient._STREAM_5XX_BACKOFFS
        statuses_5xx = LLMClient._STREAM_5XX_STATUSES

        # ----- Phase 1: probe (no payment header) -----
        payment_headers: Optional[Dict[str, str]] = None
        cost_usd = 0.0

        for attempt in range(len(backoffs) + 1):
            async with self._client.stream(
                "POST", url, json=body, headers=req_headers, timeout=timeout
            ) as resp1:
                if resp1.status_code == 200:
                    async for chunk in self._aiter_sse_chunks(resp1):
                        yield chunk
                    return
                await resp1.aread()
                if resp1.status_code == 402:
                    payment_headers, cost_usd = self._sign_payment_from_response(body, resp1)
                    break
                if resp1.status_code in statuses_5xx and attempt < len(backoffs):
                    import asyncio

                    await asyncio.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp1, after_payment=False)
        else:
            raise APIError("stream probe exhausted retries", 0, None)

        # ----- Phase 2: stream with PAYMENT-SIGNATURE -----
        assert payment_headers is not None
        for attempt in range(len(backoffs) + 1):
            async with self._client.stream(
                "POST", url, json=body, headers=payment_headers, timeout=timeout
            ) as resp2:
                if resp2.status_code == 200:
                    # AsyncLLMClient only tracks ``_last_call_cost`` (no session
                    # totals in the async path — matches the existing async
                    # chat_completion convention).
                    if cost_usd > 0:
                        self._last_call_cost = cost_usd
                        self._capture_settlement(resp2)
                    async for chunk in self._aiter_and_archive(
                        resp2, body, cost_usd, streaming=True
                    ):
                        yield chunk
                    return
                await resp2.aread()
                if resp2.status_code == 402:
                    raise PaymentError("Payment was rejected. Check your wallet balance.")
                if resp2.status_code in statuses_5xx and attempt < len(backoffs):
                    import asyncio

                    await asyncio.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp2, after_payment=True)

    async def _aiter_and_archive(
        self,
        response: httpx.Response,
        body: Dict[str, Any],
        cost_usd: float,
        *,
        streaming: bool = True,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Async mirror of :meth:`LLMClient._iter_and_archive`. Writes the
        assembled ``chat.completion`` response to ``~/.blockrun/data/`` and
        the cost row to ``~/.blockrun/cost_log.jsonl`` once the stream
        finishes — only for paid calls (cost_usd > 0)."""
        assembled_id: Optional[str] = None
        assembled_model: Optional[str] = None
        assembled_created: int = 0
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None

        async for chunk in self._aiter_sse_chunks(response):
            if chunk.choices:
                choice = chunk.choices[0]
                content = stream_choice_content(choice)
                if content:
                    content_parts.append(content)
                fr = stream_choice_finish_reason(choice)
                if fr:
                    finish_reason = fr
            if assembled_id is None and chunk.id:
                assembled_id = chunk.id
                assembled_model = chunk.model
                assembled_created = chunk.created
            _usage = chunk_usage_dict(chunk)
            if _usage is not None:
                usage_dict = _usage
            yield chunk

        if cost_usd > 0:
            from .cache import save_to_cache

            response_data: Dict[str, Any] = {
                "id": assembled_id or "stream",
                "object": "chat.completion",
                "created": assembled_created or int(__import__("time").time()),
                "model": assembled_model or body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "".join(content_parts),
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "stream": streaming,
            }
            if usage_dict:
                response_data["usage"] = usage_dict
            try:
                save_to_cache(
                    "/v1/chat/completions",
                    body,
                    response_data,
                    cost_usd=cost_usd,
                    **self._billing_meta(),
                )
            except Exception:
                pass
            self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)

    @staticmethod
    async def _aiter_sse_chunks(response: httpx.Response) -> AsyncIterator[ChatCompletionChunk]:
        """Async variant of :meth:`LLMClient._iter_sse_chunks`."""
        async for raw_line in response.aiter_lines():
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[6:].strip()
            if payload == "[DONE]":
                return
            try:
                chunk_dict = _json.loads(payload)
            except Exception:
                continue
            try:
                yield ChatCompletionChunk(**chunk_dict)
            except Exception:
                yield ChatCompletionChunk.model_construct(**chunk_dict)

    # Reuse the sync helpers — Python class-attribute lookup binds them
    # correctly to whatever self is passed when the bound method is called.
    _sign_payment_from_response = LLMClient._sign_payment_from_response
    _raise_stream_error = LLMClient._raise_stream_error

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
        self._capture_settlement(retry_response)

        response_data = retry_response.json()
        from .cache import save_to_cache

        save_to_cache(
            "/v1/chat/completions",
            body,
            response_data,
            cost_usd=cost_usd,
            **self._billing_meta(),
        )
        self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)

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
            save_to_cache(
                endpoint,
                body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            self._log_transaction(endpoint, body, result, self._last_call_cost)
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
        self._capture_settlement(retry_response)

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
            save_to_cache(
                endpoint,
                cache_key_body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            self._log_transaction(endpoint, cache_key_body, result, self._last_call_cost)
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
        self._capture_settlement(retry_response)

        return retry_response.json()

    async def image_edit(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        model: str = "openai/gpt-image-2",
        mask: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
    ) -> ImageResponse:
        """Async image editing (img2img). ``image`` may be a single data URI or
        a list of 1-4 data URIs for multi-image fusion (openai/* up to 4,
        google/* up to 3)."""
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

    # ── Prediction Markets (Powered by Predexon) ────────────────────────────

    async def pm(self, path: str, **params: Any) -> Dict[str, Any]:
        """Async query Predexon prediction market data (GET). Powered by Predexon."""
        return await self._get_with_payment_raw(f"/v1/pm/{path}", params or None)

    async def pm_query(self, path: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Async structured query for Predexon data (POST). Powered by Predexon."""
        return await self._request_with_payment_raw(f"/v1/pm/{path}", query)

    async def pm_markets(self, **params: Any) -> Dict[str, Any]:
        """List canonical cross-venue markets (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("markets", **params)

    async def pm_listings(self, **params: Any) -> Dict[str, Any]:
        """List venue-native executable listings (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("markets/listings", **params)

    async def pm_outcome(self, predexon_id: str) -> Dict[str, Any]:
        """Resolve a canonical Predexon outcome ID (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm(f"outcomes/{predexon_id}")

    async def pm_polymarket_markets(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket markets (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/markets", **params)

    async def pm_polymarket_events(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket events (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/events", **params)

    async def pm_polymarket_markets_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket markets with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/markets/keyset", **params)

    async def pm_polymarket_events_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket events with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/events/keyset", **params)

    async def pm_polymarket_positions(self, **params: Any) -> Dict[str, Any]:
        """Polymarket open positions (per-wallet, market-level PnL).
        Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/positions", **params)

    async def pm_polymarket_trades(self, **params: Any) -> Dict[str, Any]:
        """Recent Polymarket trades. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/trades", **params)

    async def pm_polymarket_leaderboard(self, **params: Any) -> Dict[str, Any]:
        """Polymarket trader leaderboard. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/leaderboard", **params)

    async def pm_kalshi_markets(self, **params: Any) -> Dict[str, Any]:
        """List Kalshi markets. Tier 1 ($0.001/call)."""
        return await self.pm("kalshi/markets", **params)

    async def pm_limitless_markets(self, **params: Any) -> Dict[str, Any]:
        """List Limitless markets. Tier 1 ($0.001/call)."""
        return await self.pm("limitless/markets", **params)

    async def pm_sports_categories(self) -> Dict[str, Any]:
        """List available sports categories. Tier 1 ($0.001/call)."""
        return await self.pm("sports/categories")

    async def pm_sports_markets(self, **params: Any) -> Dict[str, Any]:
        """List sports markets grouped by game. Tier 1 ($0.001/call)."""
        return await self.pm("sports/markets", **params)

    async def pm_wallet_identity(self, wallet: str) -> Dict[str, Any]:
        """Identity + profile for one wallet. Tier 2 ($0.005/call)."""
        return await self.pm(f"polymarket/wallet/identity/{wallet}")

    async def pm_wallet_identities(self, addresses: List[str]) -> Dict[str, Any]:
        """Bulk identity for up to 200 wallet addresses. Tier 2 ($0.005/call)."""
        return await self.pm_query("polymarket/wallet/identities", {"addresses": addresses})

    async def pm_wallet_cluster(self, address: str) -> Dict[str, Any]:
        """Wallet-cluster discovery (on-chain transfers + identity proofs).
        Tier 2 ($0.005/call)."""
        return await self.pm(f"polymarket/wallet/{address}/cluster")

    # ── DefiLlama (DeFi protocols / TVL / yields / prices) ──────────────────

    async def defi(self, path: str, **params: Any) -> Dict[str, Any]:
        """Async query DefiLlama DeFi data (GET). $0.005/call ($0.001 for prices)."""
        return await self._get_with_payment_raw(f"/v1/defillama/{path}", params or None)

    async def defi_protocols(self) -> Dict[str, Any]:
        """Async: all DeFi protocols with TVL ($0.005/call)."""
        return await self.defi("protocols")

    async def defi_protocol(self, slug: str) -> Dict[str, Any]:
        """Async: single protocol details + historical TVL ($0.005/call)."""
        return await self.defi(f"protocol/{slug}")

    async def defi_chains(self) -> Dict[str, Any]:
        """Async: current TVL of every chain ($0.005/call)."""
        return await self.defi("chains")

    async def defi_yields(self, **params: Any) -> Dict[str, Any]:
        """Async: yield pools with APY/TVL ($0.005/call)."""
        return await self.defi("yields", **params)

    async def defi_prices(self, coins: Union[List[str], str]) -> Dict[str, Any]:
        """Async: token price lookup ($0.001/call)."""
        joined = ",".join(coins) if isinstance(coins, list) else coins
        return await self.defi(f"prices/{joined}")

    # ── 0x DEX (swap quotes + gasless) — free passthrough ───────────────────

    async def dex(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """Async query the 0x Swap / Gasless APIs (free passthrough)."""
        endpoint = f"/v1/zerox/{path}"
        if method.upper() == "POST":
            return await self._request_with_payment_raw(endpoint, body or {})
        return await self._get_with_payment_raw(endpoint, params or None)

    async def dex_price(self, **params: Any) -> Dict[str, Any]:
        """Async: indicative Permit2 swap price (free)."""
        return await self.dex("price", **params)

    async def dex_quote(self, **params: Any) -> Dict[str, Any]:
        """Async: firm Permit2 swap quote (free)."""
        return await self.dex("quote", **params)

    async def dex_gasless_price(self, **params: Any) -> Dict[str, Any]:
        """Async: gasless indicative price quote (free)."""
        return await self.dex("gasless/price", **params)

    async def dex_gasless_quote(self, **params: Any) -> Dict[str, Any]:
        """Async: gasless firm quote — returns trade.eip712 to sign (free)."""
        return await self.dex("gasless/quote", **params)

    async def dex_gasless_submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Async: submit a signed gasless trade (free)."""
        return await self.dex("gasless/submit", method="POST", body=body)

    async def dex_gasless_status(self, trade_hash: str) -> Dict[str, Any]:
        """Async: poll a gasless trade's status (free)."""
        return await self.dex(f"gasless/status/{trade_hash}")

    async def dex_chains(self) -> Dict[str, Any]:
        """Async: chains where the Swap API is supported (free)."""
        return await self.dex("swap/chains")

    async def dex_gasless_chains(self) -> Dict[str, Any]:
        """Async: chains where the Gasless API is supported (free)."""
        return await self.dex("gasless/chains")

    # ── Modal Sandbox (pay-per-call cloud compute) ───────────────────────────

    async def modal(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async call the Modal sandbox compute API (POST passthrough)."""
        return await self._request_with_payment_raw(f"/v1/modal/{path}", body or {})

    async def modal_sandbox_create(self, **body: Any) -> Dict[str, Any]:
        """Async: create a sandbox ($0.01 CPU / $0.05 GPU)."""
        return await self.modal("sandbox/create", body)

    async def modal_sandbox_exec(
        self, sandbox_id: str, command: List[str], **body: Any
    ) -> Dict[str, Any]:
        """Async: execute a command in a sandbox ($0.001)."""
        return await self.modal(
            "sandbox/exec", {"sandbox_id": sandbox_id, "command": command, **body}
        )

    async def modal_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Async: check a sandbox's status ($0.001)."""
        return await self.modal("sandbox/status", {"sandbox_id": sandbox_id})

    async def modal_sandbox_terminate(self, sandbox_id: str) -> Dict[str, Any]:
        """Async: terminate a sandbox ($0.001)."""
        return await self.modal("sandbox/terminate", {"sandbox_id": sandbox_id})

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
        """List available image generation models asynchronously.

        ``/v1/images/models`` was deprecated server-side; this filters the
        unified ``/v1/models`` catalog by ``categories: ["image"]`` so existing
        callers keep working.
        """
        models = await self.list_models()
        return [m for m in models if "image" in (m.get("categories") or [])]

    async def list_all_models(self) -> List[Dict[str, Any]]:
        """
        List all available models (chat, image, music, etc.) asynchronously.

        Returns:
            List of all model information dicts with ``type`` set per category.
        """
        all_models = await self.list_models()
        for m in all_models:
            cats = m.get("categories") or []
            if "chat" in cats:
                m["type"] = "llm"
            elif "image" in cats:
                m["type"] = "image"
            elif "music" in cats or "audio" in cats:
                m["type"] = "music"
            else:
                m["type"] = cats[0] if cats else "llm"
        return all_models

    def get_wallet_address(self) -> str:
        """Get the wallet address."""
        return self.account.address

    def is_testnet(self) -> bool:
        """Check if client is configured for testnet."""
        return "testnet.blockrun.ai" in self.api_url

    def _billing_meta(self) -> Dict[str, Optional[str]]:
        """Billing metadata for cost-log entries."""
        return {
            "wallet": self.account.address,
            "network": _detect_network(self.api_url),
            "client_kind": type(self).__name__,
        }

    def _log_transaction(
        self,
        endpoint: str,
        body: Dict[str, Any],
        response: Any,
        cost_usd: float,
    ) -> None:
        """Async-client twin of :meth:`LLMClient._log_transaction`."""
        logger = self._tx_logger
        if logger is None:
            return
        settlement = self._last_settlement
        self._last_settlement = None
        try:
            logger.log(
                endpoint=endpoint,
                request=body,
                response=response,
                cost_usd=cost_usd,
                model=(body.get("model") if isinstance(body, dict) else None),
                wallet=self.account.address,
                network=_detect_network(self.api_url),
                client_kind=type(self).__name__,
                settlement=settlement,
            )
        except Exception:
            pass

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
