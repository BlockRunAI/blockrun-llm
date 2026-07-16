"""
BlockRun Search Client - Standalone Grok Live Search via x402 micropayments.

Backend endpoint: POST /api/v1/search
Pricing: $0.025/source + margin (default 10 sources ≈ $0.26)

Usage:
    from blockrun_llm import SearchClient

    client = SearchClient()
    result = client.search("Latest news on x402 adoption", sources=["x", "web"])
    print(result.summary)
    for citation in (result.citations or []):
        print(citation)

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Only EIP-712 signatures are sent
in the PAYMENT-SIGNATURE header.
"""

import os
from typing import Optional, Dict, Any, List, Literal
import httpx
from eth_account import Account
from dotenv import load_dotenv

from .types import SearchResult, APIError, PaymentError
from .x402 import create_payment_payload, parse_payment_required, extract_payment_details
from .validation import (
    validate_private_key,
    validate_api_url,
    sanitize_error_response,
)
from .tx_log import paid_request_error_prefix


load_dotenv()

SearchSourceLiteral = Literal["x", "web", "news"]


class SearchClient:
    """
    BlockRun Search Client.

    Calls the standalone `/v1/search` endpoint which routes through Grok Live
    Search and returns a synthesized summary plus citations. Each source used
    costs $0.025 (plus margin).
    """

    DEFAULT_API_URL = "https://blockrun.ai/api"
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_MAX_RESULTS = 10

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
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

    def search(
        self,
        query: str,
        *,
        sources: Optional[List[SearchSourceLiteral]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> SearchResult:
        """
        Run a live search query.

        Args:
            query: Search query (1-1000 chars).
            sources: Subset of ["x", "web", "news"] (default: ["x", "web"]).
            max_results: 1-50 (default 10). Price scales with this.
            from_date, to_date: YYYY-MM-DD filters (optional).

        Returns:
            SearchResult with summary, citations, and sources_used.
        """
        if not query or len(query) > 1000:
            raise ValueError("query must be 1-1000 characters")
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")

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

        data = self._request_with_payment("/v1/search", body)
        return SearchResult(**data)

    def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_url}{endpoint}"
        response = self._client.post(url, json=body, headers={"Content-Type": "application/json"})
        if response.status_code == 402:
            return self._handle_payment_and_retry(url, body, response)
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

    def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
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
            resource_description=resource.get("description", "BlockRun Search"),
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
                f"{paid_request_error_prefix(retry.headers)}: {retry.status_code}",
                retry.status_code,
                sanitize_error_response(error_body),
            )
        return retry.json()

    def get_wallet_address(self) -> str:
        return self.account.address

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SearchClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
