"""
BlockRun X (Twitter) Client - AttentionVC-partnered X/Twitter API via x402.

Backend endpoints under /api/v1/x/*:

    Users
      POST /v1/x/users/lookup            { usernames }
      POST /v1/x/users/info              { username }
      POST /v1/x/users/followers         { username, cursor? }
      POST /v1/x/users/following         { username, cursor? }   (alias: followings)
      POST /v1/x/users/followings        { username, cursor? }
      POST /v1/x/users/verified-followers{ userId, cursor? }
      POST /v1/x/users/tweets            { username?, userId?, cursor?, includeReplies? }
      POST /v1/x/users/mentions          { username, sinceTime?, untilTime?, cursor? }
    Tweets
      POST /v1/x/tweets/lookup           { tweet_ids }
      POST /v1/x/tweets/replies          { tweetId, cursor?, queryType? }
      POST /v1/x/tweets/thread           { tweetId, cursor? }
    Search / Discovery
      POST /v1/x/search                  { query, queryType?, cursor? }
      POST /v1/x/trending                {}
      POST /v1/x/articles/rising         {}

Every call is gated by x402 with a per-call price. The client handles the
402 → sign → retry dance automatically; your private key never leaves the
machine.

Usage:
    from blockrun_llm import XClient

    x = XClient()
    info = x.user_info("elonmusk")
    followers = x.followers("paulg")
    results = x.search("x402 micropayments", query_type="Latest")
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Dict, Any, List, Union, Literal
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import (
    APIError,
    PaymentError,
    XUserLookupResponse,
    XUserInfoResponse,
    XFollowersResponse,
    XFollowingsResponse,
    XVerifiedFollowersResponse,
    XTweetsResponse,
    XMentionsResponse,
    XTweetLookupResponse,
    XTweetRepliesResponse,
    XTweetThreadResponse,
    XSearchResponse,
    XTrendingResponse,
    XArticlesRisingResponse,
)
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
)


load_dotenv()


class XClient:
    """
    BlockRun X/Twitter Client.

    .. deprecated::
        BlockRun's ``/v1/x/*`` (AttentionVC-partnered) integration was
        removed from the backend on 2026-04-30 (commit 80dcf52). All
        ``XClient`` calls will return HTTP 404 until a replacement upstream
        is wired up. The class is kept in the SDK so existing imports do
        not break; instantiation emits a ``DeprecationWarning``.

    Every method issues a POST, hits the x402 gate, signs the payment, and
    returns the parsed response. Errors raise :class:`APIError` or
    :class:`PaymentError`.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        warnings.warn(
            "BlockRun's /v1/x/* (AttentionVC) integration was removed "
            "2026-04-30. All XClient calls will return HTTP 404 until a "
            "replacement X data upstream is reintroduced.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    # ───────── User endpoints ─────────

    def user_lookup(self, usernames: Union[str, List[str]]) -> XUserLookupResponse:
        """Batch user lookup. Accepts a list or comma-separated string."""
        data = self._post("/v1/x/users/lookup", {"usernames": usernames})
        return XUserLookupResponse(**data)

    def user_info(self, username: str) -> XUserInfoResponse:
        """Single user profile by username."""
        data = self._post("/v1/x/users/info", {"username": username})
        return XUserInfoResponse(**data)

    def followers(self, username: str, *, cursor: Optional[str] = None) -> XFollowersResponse:
        body: Dict[str, Any] = {"username": username}
        if cursor:
            body["cursor"] = cursor
        data = self._post("/v1/x/users/followers", body)
        return XFollowersResponse(**data)

    def following(self, username: str, *, cursor: Optional[str] = None) -> XFollowingsResponse:
        """Alias for :meth:`followings` — matches the backend path
        `/v1/x/users/following` (singular)."""
        body: Dict[str, Any] = {"username": username}
        if cursor:
            body["cursor"] = cursor
        data = self._post("/v1/x/users/following", body)
        return XFollowingsResponse(**data)

    def followings(self, username: str, *, cursor: Optional[str] = None) -> XFollowingsResponse:
        """`/v1/x/users/followings` (plural) variant."""
        body: Dict[str, Any] = {"username": username}
        if cursor:
            body["cursor"] = cursor
        data = self._post("/v1/x/users/followings", body)
        return XFollowingsResponse(**data)

    def verified_followers(
        self, user_id: str, *, cursor: Optional[str] = None
    ) -> XVerifiedFollowersResponse:
        body: Dict[str, Any] = {"userId": user_id}
        if cursor:
            body["cursor"] = cursor
        data = self._post("/v1/x/users/verified-followers", body)
        return XVerifiedFollowersResponse(**data)

    def user_tweets(
        self,
        *,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        cursor: Optional[str] = None,
        include_replies: Optional[bool] = None,
    ) -> XTweetsResponse:
        """Fetch a user's tweets. Either username or user_id is required."""
        if not username and not user_id:
            raise ValueError("Either username or user_id is required")
        body: Dict[str, Any] = {}
        if username:
            body["username"] = username
        if user_id:
            body["userId"] = user_id
        if cursor:
            body["cursor"] = cursor
        if include_replies is not None:
            body["includeReplies"] = include_replies
        data = self._post("/v1/x/users/tweets", body)
        return XTweetsResponse(**data)

    def mentions(
        self,
        username: str,
        *,
        since_time: Optional[str] = None,
        until_time: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> XMentionsResponse:
        body: Dict[str, Any] = {"username": username}
        if since_time:
            body["sinceTime"] = since_time
        if until_time:
            body["untilTime"] = until_time
        if cursor:
            body["cursor"] = cursor
        data = self._post("/v1/x/users/mentions", body)
        return XMentionsResponse(**data)

    # ───────── Tweet endpoints ─────────

    def tweet_lookup(self, tweet_ids: Union[str, List[str]]) -> XTweetLookupResponse:
        """Batch tweet lookup. Accepts a list or comma-separated string."""
        data = self._post("/v1/x/tweets/lookup", {"tweet_ids": tweet_ids})
        return XTweetLookupResponse(**data)

    def tweet_replies(
        self,
        tweet_id: str,
        *,
        cursor: Optional[str] = None,
        query_type: Optional[Literal["Latest", "Default"]] = None,
    ) -> XTweetRepliesResponse:
        body: Dict[str, Any] = {"tweetId": tweet_id}
        if cursor:
            body["cursor"] = cursor
        if query_type:
            body["queryType"] = query_type
        data = self._post("/v1/x/tweets/replies", body)
        return XTweetRepliesResponse(**data)

    def tweet_thread(self, tweet_id: str, *, cursor: Optional[str] = None) -> XTweetThreadResponse:
        body: Dict[str, Any] = {"tweetId": tweet_id}
        if cursor:
            body["cursor"] = cursor
        data = self._post("/v1/x/tweets/thread", body)
        return XTweetThreadResponse(**data)

    # ───────── Search & discovery ─────────

    def search(
        self,
        query: str,
        *,
        query_type: Optional[Literal["Latest", "Top", "Default"]] = None,
        cursor: Optional[str] = None,
    ) -> XSearchResponse:
        body: Dict[str, Any] = {"query": query}
        if query_type:
            body["queryType"] = query_type
        if cursor:
            body["cursor"] = cursor
        data = self._post("/v1/x/search", body)
        return XSearchResponse(**data)

    def trending(self) -> XTrendingResponse:
        data = self._post("/v1/x/trending", {})
        return XTrendingResponse(**data)

    def articles_rising(self) -> XArticlesRisingResponse:
        data = self._post("/v1/x/articles/rising", {})
        return XArticlesRisingResponse(**data)

    # ───────── Internals ─────────

    def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_url}{endpoint}"
        response = self._client.post(url, json=body, headers={"Content-Type": "application/json"})
        if response.status_code == 402:
            return self._pay_and_retry(url, body, response)
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

    def _pay_and_retry(
        self, url: str, body: Dict[str, Any], response: httpx.Response
    ) -> Dict[str, Any]:
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
            resource_description=resource.get("description", "BlockRun X API"),
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
        if retry.status_code != 200:
            try:
                error_body = retry.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry.status_code}",
                retry.status_code,
                sanitize_error_response(error_body),
            )
        return retry.json()

    def get_wallet_address(self) -> str:
        return self.account.address

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "XClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
