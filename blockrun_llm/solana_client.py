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

import os
from typing import Any, Dict, List, Optional, Union

import httpx

from .types import (
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
    ) -> ChatResponse:
        """Full chat completion (OpenAI-compatible)."""
        body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
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

        save_to_cache("/v1/chat/completions", body, response_data, cost_usd=cost_usd)

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
            save_to_cache(endpoint, body, result, cost_usd=self._last_call_cost)
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
            save_to_cache(endpoint, cache_key_body, result, cost_usd=self._last_call_cost)
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
