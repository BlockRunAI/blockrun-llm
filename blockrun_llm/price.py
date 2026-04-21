"""
BlockRun Price Client - Pyth-backed market data via x402.

Backend endpoints (payment gating mirrors CategoryConfig.paid in
blockrun/src/lib/pyth-handler.ts — crypto/fx/commodity are free across
price+history+list; only usstock and stocks/{market} charge):

    GET /v1/crypto/price/{symbol}              (free)
    GET /v1/crypto/history/{symbol}?...        (free)
    GET /v1/crypto/list?q=&limit=              (free)
    GET /v1/fx/price/{symbol}                  (free)
    GET /v1/fx/history/{symbol}?...            (free)
    GET /v1/fx/list                            (free)
    GET /v1/commodity/price/{symbol}           (free)
    GET /v1/commodity/history/{symbol}?...     (free)
    GET /v1/commodity/list                     (free)
    GET /v1/usstock/price/{symbol}             (paid — legacy alias for stocks/us)
    GET /v1/usstock/history/{symbol}?...       (paid)
    GET /v1/usstock/list                       (free)
    GET /v1/stocks/{market}/price/{symbol}     (paid — market ∈ {us,hk,jp,kr,gb,de,fr,nl,ie,lu,cn,ca})
    GET /v1/stocks/{market}/history/{symbol}   (paid)
    GET /v1/stocks/{market}/list               (free)

Usage:
    from blockrun_llm import PriceClient

    p = PriceClient()
    btc = p.price("crypto", "BTC-USD")
    aapl = p.price("stocks", "AAPL", market="us")
    bars = p.history("stocks", "AAPL", resolution="D", from_ts=1700000000, to_ts=1710000000, market="us")
    symbols = p.list_symbols("crypto", q="sol")
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, Literal
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import APIError, PaymentError, PricePoint, PriceHistoryResponse, SymbolListResponse
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
)


load_dotenv()

Category = Literal["crypto", "fx", "commodity", "usstock", "stocks"]
Resolution = Literal["1", "5", "15", "60", "240", "D", "W", "M"]
Session = Literal["pre", "post", "on"]
Market = Literal["us", "hk", "jp", "kr", "gb", "de", "fr", "nl", "ie", "lu", "cn", "ca"]


class PriceClient:
    """
    BlockRun Pyth-backed market data client.

    Free endpoints (crypto/fx/commodity price) work without a wallet but a
    wallet is still required at construction time so paid endpoints (stocks,
    history) work seamlessly. If you only need free data, set
    ``require_wallet=False``.
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        require_wallet: bool = True,
    ):
        from .wallet import load_wallet

        key = (
            private_key
            or os.environ.get("BLOCKRUN_WALLET_KEY")
            or os.environ.get("BASE_CHAIN_WALLET_KEY")
            or load_wallet()
        )
        if not key and require_wallet:
            raise ValueError(
                "Private key required for paid endpoints. Either:\n"
                "  1. Pass private_key parameter\n"
                "  2. Set BLOCKRUN_WALLET_KEY environment variable\n"
                "  3. Place key in ~/.blockrun/.session\n"
                "  4. Pass require_wallet=False if only using free endpoints."
            )

        self.account = None
        if key:
            validate_private_key(key)
            self.account = Account.from_key(key)

        api_url_raw = api_url or os.environ.get("BLOCKRUN_API_URL") or self.DEFAULT_API_URL
        validate_api_url(api_url_raw)
        self.api_url = api_url_raw.rstrip("/")

        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ───────── Price ─────────

    def price(
        self,
        category: Category,
        symbol: str,
        *,
        market: Optional[Market] = None,
        session: Optional[Session] = None,
    ) -> PricePoint:
        """
        Fetch a realtime price quote.

        For ``stocks`` category the ``market`` param is required.
        """
        endpoint = self._category_path(category, market, "price", symbol)
        params: Dict[str, Any] = {}
        if session is not None:
            params["session"] = session
        data = self._get_with_payment(endpoint, params=params)
        return PricePoint(
            symbol=data.get("symbol", symbol.upper()),
            price=data["price"],
            publish_time=data.get("publishTime"),
            confidence=data.get("confidence"),
            feed_id=data.get("feedId"),
            **{
                k: v
                for k, v in data.items()
                if k not in {"symbol", "price", "publishTime", "confidence", "feedId"}
            },
        )

    def history(
        self,
        category: Category,
        symbol: str,
        *,
        resolution: Resolution = "D",
        from_ts: int,
        to_ts: int,
        market: Optional[Market] = None,
        session: Optional[Session] = None,
    ) -> PriceHistoryResponse:
        """
        Fetch OHLC bars between two Unix timestamps (seconds).
        """
        endpoint = self._category_path(category, market, "history", symbol)
        params: Dict[str, Any] = {
            "resolution": resolution,
            "from": from_ts,
            "to": to_ts,
        }
        if session is not None:
            params["session"] = session
        data = self._get_with_payment(endpoint, params=params)
        return PriceHistoryResponse(
            symbol=data.get("symbol", symbol.upper()),
            resolution=data.get("resolution", resolution),
            bars=data.get("bars", []),
            **{k: v for k, v in data.items() if k not in {"symbol", "resolution", "bars"}},
        )

    def list_symbols(
        self,
        category: Category,
        *,
        q: Optional[str] = None,
        limit: int = 100,
        market: Optional[Market] = None,
    ) -> SymbolListResponse:
        """
        List available symbols in a category (free discovery endpoint).
        """
        endpoint = self._category_path(category, market, "list", None)
        params: Dict[str, Any] = {"limit": limit}
        if q:
            params["q"] = q
        data = self._get_with_payment(endpoint, params=params)
        # Backend returns either a bare array or an object with "symbols".
        if isinstance(data, list):
            return SymbolListResponse(symbols=data, count=len(data))
        return SymbolListResponse(
            symbols=data.get("symbols", data.get("feeds", [])),
            count=data.get("count"),
            **{k: v for k, v in data.items() if k not in {"symbols", "feeds", "count"}},
        )

    # ───────── Internals ─────────

    def _category_path(
        self,
        category: Category,
        market: Optional[str],
        kind: str,
        symbol: Optional[str],
    ) -> str:
        if category == "stocks":
            if not market:
                raise ValueError("market is required for category='stocks' (e.g. market='us')")
            base = f"/v1/stocks/{market}"
        elif category in ("crypto", "fx", "commodity", "usstock"):
            base = f"/v1/{category}"
        else:
            raise ValueError(f"Unknown category: {category}")
        if symbol is None:
            return f"{base}/{kind}"
        return f"{base}/{kind}/{symbol.upper()}"

    def _get_with_payment(self, endpoint: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.api_url}{endpoint}"
        response = self._client.get(url, params=params)
        if response.status_code == 402:
            if self.account is None:
                raise PaymentError(
                    f"{endpoint} returned 402 Payment Required but no wallet is configured."
                )
            return self._pay_and_retry(url, params, response)
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
        self,
        url: str,
        params: Optional[Dict[str, Any]],
        response: httpx.Response,
    ) -> Any:
        payment_header: Any = response.headers.get("payment-required")
        if not payment_header:
            try:
                resp_body = response.json()
                payment_header = resp_body.get("x402") or resp_body
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
            resource_description=resource.get("description", "BlockRun Price Data"),
            max_timeout_seconds=details.get("maxTimeoutSeconds", 300),
            extra=details.get("extra"),
            extensions=extensions,
        )

        retry = self._client.get(
            url,
            params=params,
            headers={"PAYMENT-SIGNATURE": payment_payload},
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

    def get_wallet_address(self) -> Optional[str]:
        return self.account.address if self.account else None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PriceClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
