"""
BlockRun Solana LLM Client.

Usage:
    from blockrun_llm import SolanaLLMClient

    # SOLANA_WALLET_KEY env var (bs58-encoded Solana secret key)
    client = SolanaLLMClient()

    # Or pass key directly
    client = SolanaLLMClient(private_key="your-bs58-key")

    # Same API as LLMClient
    response = client.chat("openai/gpt-5.2", "gm Solana")
    print(response)
"""

from __future__ import annotations

import json as _json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import httpx

from .types import (
    ChatCompletionChunk,
    ChatResponse,
    ImageResponse,
    APIError,
    PaymentError,
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
from .solana_wallet import get_solana_public_key
from .validation import validate_api_url, sanitize_error_response

try:
    from x402 import x402ClientSync
    from x402.mechanisms.svm import KeypairSigner
    from x402.mechanisms.svm.exact.register import register_exact_svm_client
    from x402.http.utils import decode_payment_required_header, encode_payment_signature_header

    _HAS_X402 = True
except ImportError:
    _HAS_X402 = False

SOLANA_API_URL = "https://sol.blockrun.ai/api"


def _create_signer(private_key: str) -> KeypairSigner:
    """Create a KeypairSigner, handling both full keypair and seed-only formats."""
    try:
        return KeypairSigner.from_base58(private_key)
    except (ValueError, Exception):
        # Fallback: might be a 32-byte seed (agentcash, etc.)
        import base58 as b58
        from solders.keypair import Keypair

        decoded = b58.b58decode(private_key)
        if len(decoded) == 32:
            kp = Keypair.from_seed(decoded)
            full_key = b58.b58encode(bytes(kp)).decode()
            return KeypairSigner.from_base58(full_key)
        raise


DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0


def _get_user_agent() -> str:
    from . import __version__

    return f"blockrun-python/{__version__}"


def _should_fallback_solana(exc: Exception) -> bool:
    """Whether an exception during Solana streaming is retriable enough to
    warrant trying the next ``fallback_models`` entry. Matches the Base
    :func:`blockrun_llm.client._should_fallback` semantics:

    - Timeouts and network errors → fall back
    - APIError with 5xx-ish status → fall back
    - 4xx and PaymentError → propagate
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.NetworkError):
        return True
    if isinstance(exc, APIError) and exc.status_code in (502, 503, 504, 522, 524):
        return True
    return False


class SolanaLLMClient:
    """
    BlockRun LLM Client for Solana — pays via Solana USDC x402.

    Connects to sol.blockrun.ai by default.
    """

    SOLANA_API_URL = SOLANA_API_URL

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: str = SOLANA_API_URL,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not _HAS_X402:
            raise ImportError(
                "Solana payment requires the x402 SDK. "
                "Install with: pip install blockrun-llm[solana]"
            )
        key = private_key or os.environ.get("SOLANA_WALLET_KEY")
        if not key:
            raise ValueError(
                "Private key required. Pass private_key or set SOLANA_WALLET_KEY env var."
            )
        self._private_key = key
        validate_api_url(api_url)
        self._api_url = api_url.rstrip("/")
        self._rpc_url = rpc_url
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._session_total_usd = 0.0
        self._session_calls = 0
        self._last_call_cost: float = 0.0
        self._address: Optional[str] = None

        # Initialize x402 SDK client for Solana payment signing
        self._x402_client = x402ClientSync()
        signer = _create_signer(self._private_key)
        register_exact_svm_client(self._x402_client, signer, rpc_url=rpc_url)

    def get_wallet_address(self) -> str:
        if not self._address:
            self._address = get_solana_public_key(self._private_key)
        return self._address

    def is_solana(self) -> bool:
        return "sol.blockrun.ai" in self._api_url

    def get_balance(self) -> float:
        """Get USDC balance on Solana (matches LLMClient.get_balance() API)."""
        from .solana_wallet import get_solana_usdc_balance

        return get_solana_usdc_balance(self.get_wallet_address(), rpc_url=self._rpc_url)

    def get_spending(self) -> Dict[str, Any]:
        return {"total_usd": self._session_total_usd, "calls": self._session_calls}

    def _billing_meta(self) -> Dict[str, Optional[str]]:
        """Billing metadata for cost-log entries."""
        return {
            "wallet": self.get_wallet_address(),
            "network": "solana-mainnet" if self.is_solana() else "solana-other",
            "client_kind": type(self).__name__,
        }

    def chat(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        search: bool = False,
    ) -> str:
        """Simple 1-line chat."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat_completion(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            search=search,
        )
        return result.choices[0].message.content or ""

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> ChatResponse:
        """Full chat completion (OpenAI-compatible).

        Supports OpenAI-style function calling via ``tools`` /
        ``tool_choice`` — the BlockRun gateway forwards them to the
        upstream model unchanged (Base and Solana use the same backend
        schema; the only chain difference is the payment leg).
        """
        body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        return self._request_with_payment("/v1/chat/completions", body)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def list_models(self) -> List[Dict[str, Any]]:
        resp = self._client.get(f"{self._api_url}/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    @staticmethod
    def _extract_payment_header(response: httpx.Response) -> Optional[str]:
        """Extract x402 payment header from a 402 response (header or body)."""
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                import base64
                import json

                resp_body = response.json()
                if resp_body.get("accepts") or resp_body.get("x402Version"):
                    payment_header = base64.b64encode(json.dumps(resp_body).encode()).decode()
            except Exception:
                pass
        return payment_header

    # ------------------------------------------------------------------
    # Streaming (SSE) chat completions
    # ------------------------------------------------------------------

    # Retry policy mirrors LLMClient. ``1 + len(_STREAM_5XX_BACKOFFS)`` tries
    # per phase (probe / paid-retry), exponential backoff in seconds.
    _STREAM_5XX_STATUSES = (500, 502, 503, 504)
    _STREAM_5XX_BACKOFFS = (1.0, 2.0, 4.0)

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        fallback_models: Optional[List[str]] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """
        Stream a chat completion via Server-Sent Events, paid in Solana USDC
        via x402. Mirrors :meth:`LLMClient.chat_completion_stream` semantics:

        - Yields one :class:`ChatCompletionChunk` per ``data:`` line until
          the upstream emits ``data: [DONE]``.
        - Free models stream on the first request; paid models do the
          402 → sign locally with the SVM signer → retry with
          ``PAYMENT-SIGNATURE`` dance before the first chunk.
        - 5xx upstream errors are retried in-band with exponential
          backoff (1s / 2s / 4s).
        - ``fallback_models`` walks the chain on retriable errors, but
          only **before** the first chunk has been yielded (mid-stream
          fallback would concatenate two distinct responses).
        - ``tools`` / ``tool_choice`` work the same as on Base — the
          gateway forwards them to the upstream model regardless of
          chain.

        Note: ``search_parameters`` is rejected by the BlockRun gateway in
        stream mode (HTTP 400). Codex / GPT-5.4-Pro also can't stream.
        """
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

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
                    raise  # mid-stream — can't fall back
                if not _should_fallback_solana(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] solana stream {attempt_model} -> "
                        f"{next_model} ({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        assert last_exc is not None
        raise last_exc

    def _stream_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
    ) -> Iterator[ChatCompletionChunk]:
        """402 → sign (SVM) → retry → SSE iter. Same shape as the Base
        :meth:`LLMClient._stream_with_payment`; differs only in the
        signing path (we go through the x402 SDK's SVM client)."""
        url = f"{self._api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        backoffs = self._STREAM_5XX_BACKOFFS

        # ----- Phase 1: probe (no payment header) -----
        payment_headers: Optional[Dict[str, str]] = None
        cost_usd = 0.0

        for attempt in range(len(backoffs) + 1):
            with self._client.stream(
                "POST", url, json=body, headers=req_headers, timeout=self._timeout
            ) as resp1:
                if resp1.status_code == 200:
                    # Free model — stream directly.
                    yield from self._iter_sse_chunks(resp1)
                    return
                resp1.read()
                if resp1.status_code == 402:
                    payment_headers, cost_usd = self._sign_payment_from_response(resp1)
                    break
                if resp1.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import time

                    time.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp1, after_payment=False)
        else:
            raise APIError("solana stream probe exhausted retries", 0, None)

        # ----- Phase 2: stream with PAYMENT-SIGNATURE -----
        assert payment_headers is not None
        for attempt in range(len(backoffs) + 1):
            with self._client.stream(
                "POST", url, json=body, headers=payment_headers, timeout=self._timeout
            ) as resp2:
                if resp2.status_code == 200:
                    if cost_usd > 0:
                        self._session_calls += 1
                        self._session_total_usd += cost_usd
                        self._last_call_cost = cost_usd
                    yield from self._iter_and_archive(resp2, body, cost_usd)
                    return
                resp2.read()
                if resp2.status_code == 402:
                    raise PaymentError(
                        "Payment rejected. Check your Solana USDC balance."
                    )
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
    ) -> Iterator[ChatCompletionChunk]:
        """Yield SSE chunks; on stream completion, archive the assembled
        response to ``~/.blockrun/data/`` and append a row to
        ``~/.blockrun/cost_log.jsonl``. Paid streaming calls now show up
        in the same audit trail as non-stream paid calls.

        ``cost_usd == 0`` skips the archive (free models / unauth probe)."""
        assembled_id: Optional[str] = None
        assembled_model: Optional[str] = None
        assembled_created: int = 0
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None

        for chunk in self._iter_sse_chunks(response):
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.content:
                    content_parts.append(choice.delta.content)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            if assembled_id is None and chunk.id:
                assembled_id = chunk.id
                assembled_model = chunk.model
                assembled_created = chunk.created
            if chunk.usage is not None:
                usage_dict = chunk.usage.model_dump(exclude_none=True)
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
                "stream": True,
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

    @staticmethod
    def _iter_sse_chunks(response: httpx.Response) -> Iterator[ChatCompletionChunk]:
        """OpenAI-format SSE parser. ``data: <json>\\n\\n`` lines, terminated
        by ``data: [DONE]``. Malformed chunks are skipped, not raised."""
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
                yield ChatCompletionChunk.model_construct(**chunk_dict)

    def _sign_payment_from_response(
        self,
        response: httpx.Response,
    ) -> Tuple[Dict[str, str], float]:
        """Extract a 402 response's payment requirements, sign locally with
        the SVM x402 client, return ``(headers_with_PAYMENT_SIGNATURE,
        cost_usd)``. Mirrors the inline logic in
        :meth:`_handle_payment_and_retry` but returns headers instead of
        making the retry POST itself — lets the streaming path open an
        SSE connection for the retry."""
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._x402_client.create_payment_payload(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        cost_usd = float(payment_payload.accepted.amount) / 1e6

        return (
            {
                "Content-Type": "application/json",
                "User-Agent": _get_user_agent(),
                "PAYMENT-SIGNATURE": encoded_payment,
            },
            cost_usd,
        )

    @staticmethod
    def _raise_stream_error(response: httpx.Response, *, after_payment: bool) -> None:
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
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        response = self._client.post(url, json=body, headers=headers)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.post(url, json=body, headers=headers)

        if response.status_code == 402:
            return self._handle_payment_and_retry(url, body, response)

        if not response.is_success:
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

    def _handle_payment_and_retry(
        self, url: str, body: Dict[str, Any], response: httpx.Response
    ) -> ChatResponse:
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        # Use x402 SDK to decode 402 response and create signed payment
        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._x402_client.create_payment_payload(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = self._client.post(url, json=body, headers=payment_headers)
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.post(url, json=body, headers=payment_headers)

        if retry_response.status_code == 402:
            raise PaymentError("Payment rejected. Check your Solana USDC balance.")

        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        cost_usd = float(payment_payload.accepted.amount) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd

        # Save full response locally
        response_data = retry_response.json()
        from .cache import save_to_cache

        save_to_cache(
            "/v1/chat/completions",
            body,
            response_data,
            cost_usd=cost_usd,
            **self._billing_meta(),
        )

        return ChatResponse(**response_data)

    def _request_with_payment_raw(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request with Solana x402 payment, returning raw JSON."""
        from .cache import get_cached, save_to_cache

        # Check cache first — don't pay twice for same data
        cached = get_cached(endpoint, body)
        if cached is not None:
            return cached

        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        response = self._client.post(url, json=body, headers=headers)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.post(url, json=body, headers=headers)

        if response.status_code == 402:
            result = self._handle_payment_and_retry_raw(url, body, response)
            save_to_cache(
                endpoint,
                body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            return result

        if not response.is_success:
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
        self, url: str, body: Dict[str, Any], response: httpx.Response
    ) -> Dict[str, Any]:
        """Handle 402 for raw endpoints with Solana payment."""
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        # Use x402 SDK to decode 402 response and create signed payment
        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._x402_client.create_payment_payload(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = self._client.post(url, json=body, headers=payment_headers)
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.post(url, json=body, headers=payment_headers)

        if retry_response.status_code == 402:
            raise PaymentError("Payment rejected. Check your Solana USDC balance.")

        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        cost_usd = float(payment_payload.accepted.amount) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd

        return retry_response.json()

    def _get_with_payment_raw(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """GET with Solana x402 payment, returning raw JSON."""
        from .cache import get_cached, save_to_cache

        cache_key_body = params or {}
        cached = get_cached(endpoint, cache_key_body)
        if cached is not None:
            return cached

        url = f"{self._api_url}{endpoint}"
        headers = {"User-Agent": _get_user_agent()}

        response = self._client.get(url, params=params, headers=headers)

        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.get(url, params=params, headers=headers)

        if response.status_code == 402:
            result = self._handle_get_payment_and_retry(url, params, response)
            save_to_cache(
                endpoint,
                cache_key_body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            return result

        if not response.is_success:
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
        self, url: str, params: Optional[Dict[str, Any]], response: httpx.Response
    ) -> Dict[str, Any]:
        """Handle 402 for GET endpoints with Solana payment."""
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._x402_client.create_payment_payload(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        payment_headers = {
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        retry_response = self._client.get(url, params=params, headers=payment_headers)
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.get(url, params=params, headers=payment_headers)

        if retry_response.status_code == 402:
            raise PaymentError("Payment rejected. Check your Solana USDC balance.")

        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        cost_usd = float(payment_payload.accepted.amount) / 1e6
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
        """Edit an image using img2img (Solana payment)."""
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
        """Standalone search (Solana payment)."""
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

    def x_user_lookup(self, usernames: Union[List[str], str]) -> XUserLookupResponse:
        """Look up X/Twitter user profiles (Solana payment). Powered by AttentionVC."""
        if isinstance(usernames, str):
            usernames = [usernames]

        body: Dict[str, Any] = {"usernames": usernames}
        data = self._request_with_payment_raw("/v1/x/users/lookup", body)
        return XUserLookupResponse(**data)

    def x_followers(self, username: str, *, cursor: Optional[str] = None) -> XFollowersResponse:
        """Get X/Twitter followers (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/users/followers", body)
        return XFollowersResponse(**data)

    def x_followings(self, username: str, *, cursor: Optional[str] = None) -> XFollowingsResponse:
        """Get X/Twitter followings (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username}
        if cursor is not None:
            body["cursor"] = cursor

        data = self._request_with_payment_raw("/v1/x/users/followings", body)
        return XFollowingsResponse(**data)

    def x_user_info(self, username: str) -> XUserInfoResponse:
        """Get single X/Twitter user info (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"username": username}
        data = self._request_with_payment_raw("/v1/x/users/info", body)
        return XUserInfoResponse(**data)

    def x_verified_followers(
        self, user_id: str, *, cursor: Optional[str] = None
    ) -> XVerifiedFollowersResponse:
        """Get verified followers (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"userId": user_id}
        if cursor is not None:
            body["cursor"] = cursor
        data = self._request_with_payment_raw("/v1/x/users/verified-followers", body)
        return XVerifiedFollowersResponse(**data)

    def x_user_tweets(
        self, username: str, *, include_replies: bool = False, cursor: Optional[str] = None
    ) -> XTweetsResponse:
        """Get user tweets (Solana payment). Powered by AttentionVC."""
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
        """Get user mentions (Solana payment). Powered by AttentionVC."""
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
        """Batch tweet lookup (Solana payment). Powered by AttentionVC."""
        if isinstance(tweet_ids, str):
            tweet_ids = [tweet_ids]
        body: Dict[str, Any] = {"tweet_ids": tweet_ids}
        data = self._request_with_payment_raw("/v1/x/tweets/lookup", body)
        return XTweetLookupResponse(**data)

    def x_tweet_replies(
        self, tweet_id: str, *, query_type: str = "Latest", cursor: Optional[str] = None
    ) -> XTweetRepliesResponse:
        """Get tweet replies (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"tweetId": tweet_id, "queryType": query_type}
        if cursor is not None:
            body["cursor"] = cursor
        data = self._request_with_payment_raw("/v1/x/tweets/replies", body)
        return XTweetRepliesResponse(**data)

    def x_tweet_thread(
        self, tweet_id: str, *, cursor: Optional[str] = None
    ) -> XTweetThreadResponse:
        """Get tweet thread (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"tweetId": tweet_id}
        if cursor is not None:
            body["cursor"] = cursor
        data = self._request_with_payment_raw("/v1/x/tweets/thread", body)
        return XTweetThreadResponse(**data)

    def x_search(
        self, query: str, *, query_type: str = "Latest", cursor: Optional[str] = None
    ) -> XSearchResponse:
        """X/Twitter search (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"query": query, "queryType": query_type}
        if cursor is not None:
            body["cursor"] = cursor
        data = self._request_with_payment_raw("/v1/x/search", body)
        return XSearchResponse(**data)

    def x_trending(self) -> XTrendingResponse:
        """Get trending topics (Solana payment). Powered by AttentionVC."""
        data = self._request_with_payment_raw("/v1/x/trending", {})
        return XTrendingResponse(**data)

    def x_articles_rising(self) -> XArticlesRisingResponse:
        """Get rising articles (Solana payment). Powered by AttentionVC."""
        data = self._request_with_payment_raw("/v1/x/articles/rising", {})
        return XArticlesRisingResponse(**data)

    def x_author_analytics(self, handle: str) -> XAuthorAnalyticsResponse:
        """Get author analytics (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"handle": handle}
        data = self._request_with_payment_raw("/v1/x/authors", body)
        return XAuthorAnalyticsResponse(**data)

    def x_compare_authors(self, handle1: str, handle2: str) -> XCompareAuthorsResponse:
        """Compare two authors (Solana payment). Powered by AttentionVC."""
        body: Dict[str, Any] = {"handle1": handle1, "handle2": handle2}
        data = self._request_with_payment_raw("/v1/x/compare", body)
        return XCompareAuthorsResponse(**data)

    # ── Prediction Markets (Powered by Predexon) ────────────────────────────

    def pm(self, path: str, **params: Any) -> Dict[str, Any]:
        """Query Predexon prediction market data (GET, Solana payment). Powered by Predexon."""
        return self._get_with_payment_raw(f"/v1/pm/{path}", params or None)

    def pm_query(self, path: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Structured query for Predexon data (POST, Solana payment). Powered by Predexon."""
        return self._request_with_payment_raw(f"/v1/pm/{path}", query)

    def pm_markets(self, **params: Any) -> Dict[str, Any]:
        """List canonical cross-venue markets (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("markets", **params)

    def pm_listings(self, **params: Any) -> Dict[str, Any]:
        """List venue-native executable listings (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("markets/listings", **params)

    def pm_outcome(self, predexon_id: str) -> Dict[str, Any]:
        """Resolve a canonical Predexon outcome ID (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm(f"outcomes/{predexon_id}")

    def pm_polymarket_markets(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket markets (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/markets", **params)

    def pm_polymarket_events(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket events (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/events", **params)

    def pm_polymarket_markets_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket markets with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/markets/keyset", **params)

    def pm_polymarket_events_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket events with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/events/keyset", **params)

    def pm_polymarket_positions(self, **params: Any) -> Dict[str, Any]:
        """Polymarket open positions (per-wallet, market-level PnL).
        Tier 1 ($0.001/call)."""
        return self.pm("polymarket/positions", **params)

    def pm_polymarket_trades(self, **params: Any) -> Dict[str, Any]:
        """Recent Polymarket trades. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/trades", **params)

    def pm_polymarket_leaderboard(self, **params: Any) -> Dict[str, Any]:
        """Polymarket trader leaderboard. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/leaderboard", **params)

    def pm_kalshi_markets(self, **params: Any) -> Dict[str, Any]:
        """List Kalshi markets. Tier 1 ($0.001/call)."""
        return self.pm("kalshi/markets", **params)

    def pm_limitless_markets(self, **params: Any) -> Dict[str, Any]:
        """List Limitless markets. Tier 1 ($0.001/call)."""
        return self.pm("limitless/markets", **params)

    def pm_sports_categories(self) -> Dict[str, Any]:
        """List available sports categories. Tier 1 ($0.001/call)."""
        return self.pm("sports/categories")

    def pm_sports_markets(self, **params: Any) -> Dict[str, Any]:
        """List sports markets grouped by game. Tier 1 ($0.001/call)."""
        return self.pm("sports/markets", **params)

    def pm_wallet_identity(self, wallet: str) -> Dict[str, Any]:
        """Identity + profile for one wallet. Tier 2 ($0.005/call)."""
        return self.pm(f"polymarket/wallet/identity/{wallet}")

    def pm_wallet_identities(self, addresses: List[str]) -> Dict[str, Any]:
        """Bulk identity for up to 200 wallet addresses. Tier 2 ($0.005/call)."""
        return self.pm_query("polymarket/wallet/identities", {"addresses": addresses})

    def pm_wallet_cluster(self, address: str) -> Dict[str, Any]:
        """Wallet-cluster discovery (on-chain transfers + identity proofs).
        Tier 2 ($0.005/call)."""
        return self.pm(f"polymarket/wallet/{address}/cluster")

    # ── Exa Web Search (Powered by Exa) ─────────────────────────────────────

    def exa(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Generic Exa endpoint proxy (POST, Solana payment). Powered by Exa.

        Args:
            path: Exa endpoint — one of: "search", "find-similar", "contents", "answer"
            body: Request body (see Exa API docs)

        Example::

            result = client.exa("search", {"query": "latest AI research", "numResults": 5})
        """
        return self._request_with_payment_raw(f"/v1/exa/{path}", body)

    def exa_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Neural and keyword web search via Exa (Solana payment, $0.01/request).

        Args:
            query: Search query string
            **kwargs: Additional Exa parameters (numResults, category, useAutoprompt, etc.)

        Example::

            results = client.exa_search("latest AI papers", numResults=5)
        """
        return self._request_with_payment_raw("/v1/exa/search", {"query": query, **kwargs})

    def exa_find_similar(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Find pages semantically similar to a given URL via Exa (Solana payment, $0.01/request).

        Args:
            url: URL to find similar pages for
            **kwargs: Additional Exa parameters (numResults, etc.)

        Example::

            results = client.exa_find_similar("https://openai.com/research/gpt-4", numResults=5)
        """
        return self._request_with_payment_raw("/v1/exa/find-similar", {"url": url, **kwargs})

    def exa_contents(self, urls: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Extract full text content from URLs via Exa (Solana payment, $0.002/URL).

        Args:
            urls: List of URLs to extract content from
            **kwargs: Additional Exa parameters (text, highlights, summary, etc.)

        Example::

            data = client.exa_contents(["https://arxiv.org/abs/2303.08774"])
        """
        return self._request_with_payment_raw("/v1/exa/contents", {"urls": urls, **kwargs})

    def exa_answer(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """AI-generated answer grounded in live web search via Exa (Solana payment, $0.01/request).

        Args:
            query: Question to answer
            **kwargs: Additional Exa parameters

        Example::

            answer = client.exa_answer("What is the current state of AI safety research?")
        """
        return self._request_with_payment_raw("/v1/exa/answer", {"query": query, **kwargs})


# ===========================================================================
# AsyncSolanaLLMClient — async mirror of SolanaLLMClient (chat only, v0.22.0)
# ===========================================================================
#
# Scope for the first release: chat completions, sync **and** streaming. Image,
# music, video, exa, predexon are sync-only on Solana for now — same as the
# Solana sync class shipped initially. They can be added in follow-up releases.


class AsyncSolanaLLMClient:
    """
    Async BlockRun Solana LLM Client — pays via Solana USDC x402.

    Mirrors :class:`SolanaLLMClient` but exposes ``await``-able methods so
    Python ``asyncio`` callers (FastAPI handlers, LiteLLM Proxy, etc.) don't
    have to thread-pool around blocking I/O.

    Usage::

        client = AsyncSolanaLLMClient()                  # SOLANA_WALLET_KEY env
        resp = await client.chat_completion(
            "openai/gpt-5.5",
            [{"role": "user", "content": "gm Solana"}],
        )
        await client.close()
    """

    SOLANA_API_URL = SOLANA_API_URL
    _STREAM_5XX_STATUSES = SolanaLLMClient._STREAM_5XX_STATUSES
    _STREAM_5XX_BACKOFFS = SolanaLLMClient._STREAM_5XX_BACKOFFS

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: str = SOLANA_API_URL,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not _HAS_X402:
            raise ImportError(
                "Solana payment requires the x402 SDK. "
                "Install with: pip install blockrun-llm[solana]"
            )
        key = private_key or os.environ.get("SOLANA_WALLET_KEY")
        if not key:
            raise ValueError(
                "Private key required. Pass private_key or set SOLANA_WALLET_KEY env var."
            )
        self._private_key = key
        validate_api_url(api_url)
        self._api_url = api_url.rstrip("/")
        self._rpc_url = rpc_url
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._session_total_usd = 0.0
        self._session_calls = 0
        self._last_call_cost: float = 0.0
        self._address: Optional[str] = None

        # Async x402 client + same SVM signer the sync class uses.
        from x402 import x402Client  # local import to keep optional dep clean

        self._x402_client = x402Client()
        signer = _create_signer(self._private_key)
        register_exact_svm_client(self._x402_client, signer, rpc_url=rpc_url)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncSolanaLLMClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Identity / state
    # ------------------------------------------------------------------

    def get_wallet_address(self) -> str:
        if not self._address:
            self._address = get_solana_public_key(self._private_key)
        return self._address

    def is_solana(self) -> bool:
        return "sol.blockrun.ai" in self._api_url

    def get_spending(self) -> Dict[str, Any]:
        return {"total_usd": self._session_total_usd, "calls": self._session_calls}

    def _billing_meta(self) -> Dict[str, Optional[str]]:
        return {
            "wallet": self.get_wallet_address(),
            "network": "solana-mainnet" if self.is_solana() else "solana-other",
            "client_kind": type(self).__name__,
        }

    # ------------------------------------------------------------------
    # Non-streaming chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        search: bool = False,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = await self.chat_completion(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            search=search,
        )
        return result.choices[0].message.content or ""

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> ChatResponse:
        body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        return await self._request_with_payment("/v1/chat/completions", body)

    async def list_models(self) -> List[Dict[str, Any]]:
        resp = await self._client.get(f"{self._api_url}/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        fallback_models: Optional[List[str]] = None,
    ) -> "AsyncSolanaIterator":
        """Async streaming. Same protocol semantics as the sync
        :meth:`SolanaLLMClient.chat_completion_stream`; only the iteration
        protocol differs (``async for``)."""
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

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
                if not _should_fallback_solana(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] async solana stream {attempt_model} -> "
                        f"{next_model} ({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        assert last_exc is not None
        raise last_exc

    async def _stream_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
    ):
        """Async version of :meth:`SolanaLLMClient._stream_with_payment`."""
        url = f"{self._api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        backoffs = self._STREAM_5XX_BACKOFFS

        # ----- Phase 1: probe (no payment header) -----
        payment_headers: Optional[Dict[str, str]] = None
        cost_usd = 0.0

        for attempt in range(len(backoffs) + 1):
            async with self._client.stream(
                "POST", url, json=body, headers=req_headers, timeout=self._timeout
            ) as resp1:
                if resp1.status_code == 200:
                    async for chunk in self._aiter_sse_chunks(resp1):
                        yield chunk
                    return
                await resp1.aread()
                if resp1.status_code == 402:
                    payment_headers, cost_usd = await self._sign_payment_from_response(resp1)
                    break
                if resp1.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import asyncio

                    await asyncio.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp1, after_payment=False)
        else:
            raise APIError("solana stream probe exhausted retries", 0, None)

        # ----- Phase 2: stream with PAYMENT-SIGNATURE -----
        assert payment_headers is not None
        for attempt in range(len(backoffs) + 1):
            async with self._client.stream(
                "POST", url, json=body, headers=payment_headers, timeout=self._timeout
            ) as resp2:
                if resp2.status_code == 200:
                    if cost_usd > 0:
                        self._session_calls += 1
                        self._session_total_usd += cost_usd
                        self._last_call_cost = cost_usd
                    async for chunk in self._aiter_and_archive(resp2, body, cost_usd):
                        yield chunk
                    return
                await resp2.aread()
                if resp2.status_code == 402:
                    raise PaymentError(
                        "Payment rejected. Check your Solana USDC balance."
                    )
                if resp2.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import asyncio

                    await asyncio.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp2, after_payment=True)

    @staticmethod
    async def _aiter_sse_chunks(response: httpx.Response):
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

    async def _aiter_and_archive(
        self,
        response: httpx.Response,
        body: Dict[str, Any],
        cost_usd: float,
    ):
        """Async version of :meth:`SolanaLLMClient._iter_and_archive`."""
        assembled_id: Optional[str] = None
        assembled_model: Optional[str] = None
        assembled_created: int = 0
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None

        async for chunk in self._aiter_sse_chunks(response):
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.content:
                    content_parts.append(choice.delta.content)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            if assembled_id is None and chunk.id:
                assembled_id = chunk.id
                assembled_model = chunk.model
                assembled_created = chunk.created
            if chunk.usage is not None:
                usage_dict = chunk.usage.model_dump(exclude_none=True)
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
                "stream": True,
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

    # ------------------------------------------------------------------
    # Payment + transport helpers
    # ------------------------------------------------------------------

    async def _sign_payment_from_response(
        self,
        response: httpx.Response,
    ) -> Tuple[Dict[str, str], float]:
        payment_header = SolanaLLMClient._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")
        payment_required = decode_payment_required_header(payment_header)
        payment_payload = await self._x402_client.create_payment_payload(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)
        cost_usd = float(payment_payload.accepted.amount) / 1e6
        return (
            {
                "Content-Type": "application/json",
                "User-Agent": _get_user_agent(),
                "PAYMENT-SIGNATURE": encoded_payment,
            },
            cost_usd,
        )

    # Reuse the sync class's pure helper — it doesn't touch async state.
    _raise_stream_error = SolanaLLMClient._raise_stream_error

    async def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> ChatResponse:
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        response = await self._client.post(url, json=body, headers=headers)
        if response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            response = await self._client.post(url, json=body, headers=headers)

        if response.status_code == 402:
            return await self._handle_payment_and_retry(url, body, response)

        if not response.is_success:
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
        self, url: str, body: Dict[str, Any], response: httpx.Response
    ) -> ChatResponse:
        payment_headers, cost_usd = await self._sign_payment_from_response(response)

        retry_response = await self._client.post(url, json=body, headers=payment_headers)
        if retry_response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            retry_response = await self._client.post(url, json=body, headers=payment_headers)

        if retry_response.status_code == 402:
            raise PaymentError("Payment rejected. Check your Solana USDC balance.")
        if not retry_response.is_success:
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

        response_data = retry_response.json()
        from .cache import save_to_cache

        save_to_cache(
            "/v1/chat/completions",
            body,
            response_data,
            cost_usd=cost_usd,
            **self._billing_meta(),
        )
        return ChatResponse(**response_data)


# A typing placeholder so the chat_completion_stream return type docs above
# don't reference a name pyright can't resolve.
AsyncSolanaIterator = Any
