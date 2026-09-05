"""
BlockRun Surf Client - asksurf.ai crypto-data gateway via x402 micropayments.

Surf is a single backend partner exposing ~83 crypto-intelligence endpoints
(exchange data, on-chain SQL, prediction markets, wallet/social analytics, …).

Pricing is tiered:
    Tier 1  $0.001  market data, lists, single-token reads
    Tier 2  $0.005  AI-derived intelligence (rankings, trends, search)
    Tier 3  $0.020  heavy LLM reports + on-chain SQL/structured queries

Usage:
    from blockrun_llm import SurfClient

    client = SurfClient()

    # Discovery
    print(SurfClient.endpoints())                       # full catalog (list of dicts)
    print(client.price("market/ranking"))               # 0.001
    print(client.endpoint_info("onchain/sql"))          # {'method': 'POST', 'tier': 3, ...}

    # GET endpoints — pass query params
    data = client.get("market/ranking", {"limit": 20})
    price = client.get("exchange/price", {"pair": "BTC/USDT"})

    # POST endpoints — JSON body
    result = client.post("onchain/sql", {"query": "SELECT count() FROM ethereum.blocks"})

    # Generic helper (auto-routes GET/POST from the catalog)
    out = client.call("token/holders", params={"address": "0x...", "chain": "ethereum"})

SECURITY NOTE: your private key never leaves your machine. Only EIP-712
signatures are sent in the PAYMENT-SIGNATURE header.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from eth_account import Account
from typing_extensions import Self

from .apikey import (
    api_key_base_url,
    auth_headers,
    missing_credential_error,
    payment_mode,
    raise_for_api_key_402,
    resolve_api_key,
)
from .tx_log import paid_request_error_prefix
from .types import APIError, PaymentError
from .validation import (
    sanitize_error_response,
    validate_api_url,
    validate_private_key,
)
from .x402 import (
    create_payment_payload,
    extract_payment_details,
    parse_payment_required,
)

load_dotenv()


# Mirrors src/lib/surf.ts SURF_TIER_*_PRICE on the backend.
SURF_TIER_PRICES: dict[int, float] = {
    1: 0.001,
    2: 0.005,
    3: 0.020,
}


# Mirrors src/lib/surf.ts SURF_ENDPOINTS. Each tuple is (path, method, tier, required_params).
# Keep this list in sync when backend endpoints change — used for discovery, parameter
# validation, and auto GET/POST routing in SurfClient.call().
_SURF_CATALOG: list[tuple[str, str, int, tuple[str, ...]]] = [
    # exchange
    ("exchange/markets", "GET", 1, ()),
    ("exchange/price", "GET", 1, ("pair",)),
    ("exchange/perp", "GET", 1, ("pair",)),
    ("exchange/depth", "GET", 2, ("pair",)),
    ("exchange/klines", "GET", 2, ("pair",)),
    ("exchange/funding-history", "GET", 2, ("pair",)),
    ("exchange/long-short-ratio", "GET", 2, ("pair",)),
    # fund
    ("fund/detail", "GET", 1, ()),
    ("fund/portfolio", "GET", 1, ()),
    ("fund/ranking", "GET", 1, ("metric",)),
    # market
    ("market/ranking", "GET", 1, ()),
    ("market/fear-greed", "GET", 1, ()),
    ("market/futures", "GET", 1, ()),
    ("market/price", "GET", 1, ("symbol",)),
    ("market/etf", "GET", 1, ("symbol",)),
    ("market/options", "GET", 1, ("symbol",)),
    ("market/liquidation/exchange-list", "GET", 2, ()),
    ("market/liquidation/order", "GET", 2, ()),
    ("market/liquidation/chart", "GET", 2, ("symbol",)),
    ("market/onchain-indicator", "GET", 2, ("symbol", "metric")),
    ("market/price-indicator", "GET", 2, ("indicator", "symbol")),
    # news
    ("news/feed", "GET", 1, ()),
    ("news/detail", "GET", 1, ("id",)),
    # onchain
    ("onchain/bridge/ranking", "GET", 1, ()),
    ("onchain/yield/ranking", "GET", 1, ()),
    ("onchain/gas-price", "GET", 1, ("chain",)),
    ("onchain/tx", "GET", 1, ("hash", "chain")),
    ("onchain/schema", "GET", 3, ()),
    ("onchain/query", "POST", 3, ()),
    ("onchain/sql", "POST", 3, ()),
    # prediction-market
    ("prediction-market/category-metrics", "GET", 1, ()),
    ("prediction-market/polymarket/ranking", "GET", 1, ()),
    ("prediction-market/polymarket/trades", "GET", 1, ()),
    ("prediction-market/polymarket/markets", "GET", 1, ("market_slug",)),
    ("prediction-market/polymarket/events", "GET", 1, ("event_slug",)),
    ("prediction-market/polymarket/prices", "GET", 1, ("condition_id",)),
    ("prediction-market/polymarket/volumes", "GET", 1, ("condition_id",)),
    ("prediction-market/polymarket/open-interest", "GET", 1, ("condition_id",)),
    ("prediction-market/polymarket/positions", "GET", 2, ("address",)),
    ("prediction-market/polymarket/activity", "GET", 2, ("address",)),
    ("prediction-market/kalshi/ranking", "GET", 1, ()),
    ("prediction-market/kalshi/markets", "GET", 1, ("market_ticker",)),
    ("prediction-market/kalshi/events", "GET", 1, ("event_ticker",)),
    ("prediction-market/kalshi/prices", "GET", 1, ("ticker",)),
    ("prediction-market/kalshi/trades", "GET", 1, ("ticker",)),
    ("prediction-market/kalshi/volumes", "GET", 1, ("ticker",)),
    ("prediction-market/kalshi/open-interest", "GET", 1, ("ticker",)),
    # project
    ("project/detail", "GET", 1, ()),
    ("project/defi/metrics", "GET", 1, ("metric",)),
    ("project/defi/ranking", "GET", 1, ("metric",)),
    # search
    ("search/airdrop", "GET", 2, ()),
    ("search/events", "GET", 2, ()),
    ("search/kalshi", "GET", 2, ()),
    ("search/polymarket", "GET", 2, ()),
    ("search/web", "GET", 2, ("q",)),
    ("search/project", "GET", 2, ("q",)),
    ("search/news", "GET", 2, ("q",)),
    ("search/wallet", "GET", 2, ("q",)),
    ("search/fund", "GET", 2, ("q",)),
    ("search/social/people", "GET", 2, ("q",)),
    ("search/social/posts", "GET", 2, ("q",)),
    # social
    ("social/detail", "GET", 2, ()),
    ("social/ranking", "GET", 2, ()),
    ("social/smart-followers/history", "GET", 2, ()),
    ("social/mindshare", "GET", 2, ("q", "interval")),
    ("social/tweets", "GET", 1, ("ids",)),
    ("social/tweet/replies", "GET", 1, ("tweet_id",)),
    ("social/user", "GET", 1, ("handle",)),
    ("social/user/followers", "GET", 1, ("handle",)),
    ("social/user/following", "GET", 1, ("handle",)),
    ("social/user/posts", "GET", 1, ("handle",)),
    ("social/user/replies", "GET", 1, ("handle",)),
    # token
    ("token/tokenomics", "GET", 1, ()),
    ("token/dex-trades", "GET", 2, ("address",)),
    ("token/holders", "GET", 2, ("address", "chain")),
    ("token/transfers", "GET", 2, ("address", "chain")),
    # wallet
    ("wallet/detail", "GET", 2, ("address",)),
    ("wallet/history", "GET", 2, ("address",)),
    ("wallet/net-worth", "GET", 2, ("address",)),
    ("wallet/transfers", "GET", 2, ("address",)),
    ("wallet/protocols", "GET", 2, ("address",)),
    ("wallet/labels/batch", "GET", 2, ("addresses",)),
    # web
    ("web/fetch", "GET", 2, ("url",)),
]

_CATALOG_BY_PATH: dict[str, tuple[str, int, tuple[str, ...]]] = {
    path: (method, tier, required) for path, method, tier, required in _SURF_CATALOG
}


class SurfClient:
    """
    BlockRun Surf Client.

    Wraps the `/v1/surf/*` partner proxy. Use SurfClient.endpoints() for discovery
    and `get()` / `post()` / `call()` to fetch data. Payment is automatic on every
    request via x402 (tier 1 / 2 / 3 → $0.001 / $0.005 / $0.020).
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        private_key: str | None = None,
        api_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
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

    # ------------------------------------------------------------ Discovery API

    @staticmethod
    def endpoints() -> list[dict[str, Any]]:
        """Return the full Surf endpoint catalog with method, tier, and price."""
        return [
            {
                "path": path,
                "method": method,
                "tier": tier,
                "price_usd": SURF_TIER_PRICES[tier],
                "required_params": list(required),
            }
            for path, method, tier, required in _SURF_CATALOG
        ]

    @staticmethod
    def endpoint_info(path: str) -> dict[str, Any] | None:
        """Return catalog info for one path, or None if unknown."""
        entry = _CATALOG_BY_PATH.get(path)
        if not entry:
            return None
        method, tier, required = entry
        return {
            "path": path,
            "method": method,
            "tier": tier,
            "price_usd": SURF_TIER_PRICES[tier],
            "required_params": list(required),
        }

    @staticmethod
    def price(path: str) -> float:
        """Return the settled USDC price for a Surf endpoint."""
        info = SurfClient.endpoint_info(path)
        if info is None:
            raise ValueError(f"Unknown Surf endpoint: {path!r}")
        return info["price_usd"]

    # ---------------------------------------------------------------- Core HTTP

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET an `/v1/surf/{path}` endpoint with optional query params."""
        self._validate_path(path, "GET", params or {})
        return self._request("GET", path, params=params, json_body=None)

    def post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST an `/v1/surf/{path}` endpoint with an optional JSON body."""
        self._validate_path(path, "POST", body or {})
        return self._request("POST", path, params=None, json_body=body)

    def call(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generic helper that auto-routes to GET or POST based on the catalog.

        For GET endpoints the supplied `params` become query string entries; for
        POST endpoints both `params` and `body` get merged into the JSON body
        (body wins on conflict).
        """
        info = self.endpoint_info(path)
        if info is None:
            raise ValueError(
                f"Unknown Surf endpoint: {path!r}. "
                f"Try SurfClient.endpoints() to list available paths."
            )
        if info["method"] == "GET":
            merged_params = {**(params or {}), **(body or {})}
            return self.get(path, merged_params or None)
        merged_body = {**(params or {}), **(body or {})}
        return self.post(path, merged_body or None)

    # ---------------------------------------------------------------- Internals

    def _validate_path(self, path: str, method: str, supplied: dict[str, Any]) -> None:
        info = self.endpoint_info(path)
        if info is None:
            return  # allow forward-compat with newer backend endpoints
        if info["method"] != method:
            raise ValueError(
                f"Surf endpoint {path!r} requires method {info['method']}, got {method}"
            )
        missing = [p for p in info["required_params"] if p not in supplied]
        if missing:
            raise ValueError(f"Surf endpoint {path!r} is missing required params: {missing}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        url = f"{self.api_url}/v1/surf/{path}"
        headers = {"Content-Type": "application/json"} if json_body is not None else {}
        response = self._client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=headers,
        )
        if response.status_code == 402:
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(response, self.api_key)
            return self._handle_payment_and_retry(method, url, params, json_body, response)
        return self._unwrap(response)

    def _handle_payment_and_retry(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        response: httpx.Response,
    ) -> dict[str, Any]:
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
            resource_description=resource.get("description", "BlockRun Surf"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
        )

        retry_headers: dict[str, str] = {"PAYMENT-SIGNATURE": payment_payload}
        if json_body is not None:
            retry_headers["Content-Type"] = "application/json"

        retry = self._client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=retry_headers,
        )
        if retry.status_code == 402:
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(retry, self.api_key)
            raise PaymentError("Payment was rejected. Check your wallet balance.")
        data = self._unwrap(retry, after_payment=True)
        tx_hash = retry.headers.get("x-payment-receipt") or retry.headers.get("X-Payment-Receipt")
        if tx_hash and isinstance(data, dict):
            data.setdefault("txHash", tx_hash)
        return data

    @staticmethod
    def _unwrap(response: httpx.Response, *, after_payment: bool = False) -> dict[str, Any]:
        if response.status_code == 200:
            return response.json()
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": "Request failed"}
        prefix = paid_request_error_prefix(response.headers) if after_payment else "API error"
        raise APIError(
            f"{prefix}: {response.status_code}",
            response.status_code,
            sanitize_error_response(error_body),
        )

    # ------------------------------------------------------------------ Helpers

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
        """Return the EVM wallet address used for payments."""
        return self.account.address

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
