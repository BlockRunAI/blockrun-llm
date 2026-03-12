"""
BlockRun Solana LLM Client.

Usage:
    from blockrun_llm import SolanaLLMClient

    # SOLANA_WALLET_KEY env var (bs58-encoded Solana secret key)
    client = SolanaLLMClient()

    # Or pass key directly
    client = SolanaLLMClient(private_key="your-bs58-key")

    # Same API as LLMClient
    response = client.chat("openai/gpt-4o", "gm Solana")
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
    XUser,
    XFollowersResponse,
    XFollowingsResponse,
    XFollower,
)
from .x402 import (
    create_solana_payment_payload,
    extract_solana_payment_details,
    parse_payment_required,
)
from .solana_wallet import get_solana_public_key
from .validation import validate_api_url, sanitize_error_response, validate_resource_url

SOLANA_API_URL = "https://sol.blockrun.ai/api"
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
        self._session_total_usd = 0.0
        self._session_calls = 0
        self._address: Optional[str] = None

    def get_wallet_address(self) -> str:
        if not self._address:
            self._address = get_solana_public_key(self._private_key)
        return self._address

    def is_solana(self) -> bool:
        return "sol.blockrun.ai" in self._api_url

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

    def list_models(self) -> List[Dict[str, Any]]:
        with httpx.Client(timeout=self._timeout) as http:
            resp = http.get(f"{self._api_url}/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    def _request_with_payment(self, endpoint: str, body: Dict[str, Any]) -> ChatResponse:
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        with httpx.Client(timeout=self._timeout) as http:
            response = http.post(url, json=body, headers=headers)

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
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                import base64, json

                resp_body = response.json()
                if resp_body.get("accepts") or resp_body.get("x402Version"):
                    payment_header = base64.b64encode(json.dumps(resp_body).encode()).decode()
            except Exception:
                pass

        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = parse_payment_required(payment_header)
        details = extract_solana_payment_details(payment_required)

        if not details["network"].startswith("solana:"):
            raise PaymentError(
                f"Expected Solana network, got: {details['network']}. "
                "Use LLMClient for Base payments."
            )

        fee_payer = (details.get("extra") or {}).get("feePayer")
        if not fee_payer:
            raise PaymentError("Missing feePayer in 402 extra field")

        resource_info = details.get("resource") or {}
        resource_url = validate_resource_url(
            resource_info.get("url") or f"{self._api_url}/v1/chat/completions",
            self._api_url,
        )

        payment_payload = create_solana_payment_payload(
            private_key=self._private_key,
            recipient=details["recipient"],
            amount=details["amount"],
            fee_payer=fee_payer,
            resource_url=resource_url,
            resource_description=resource_info.get("description") or "BlockRun Solana AI API call",
            max_timeout_seconds=details["max_timeout_seconds"],
            extra=details.get("extra"),
            rpc_url=self._rpc_url,
        )

        headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        with httpx.Client(timeout=self._timeout) as http:
            retry_response = http.post(url, json=body, headers=headers)

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

        cost_usd = float(details["amount"]) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd

        return ChatResponse(**retry_response.json())

    def _request_with_payment_raw(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request with Solana x402 payment, returning raw JSON."""
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}

        with httpx.Client(timeout=self._timeout) as http:
            response = http.post(url, json=body, headers=headers)

        if response.status_code == 402:
            return self._handle_payment_and_retry_raw(url, body, response)

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
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                import base64, json

                resp_body = response.json()
                if resp_body.get("accepts") or resp_body.get("x402Version"):
                    payment_header = base64.b64encode(json.dumps(resp_body).encode()).decode()
            except Exception:
                pass

        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = parse_payment_required(payment_header)
        details = extract_solana_payment_details(payment_required)

        if not details["network"].startswith("solana:"):
            raise PaymentError(
                f"Expected Solana network, got: {details['network']}. "
                "Use LLMClient for Base payments."
            )

        fee_payer = (details.get("extra") or {}).get("feePayer")
        if not fee_payer:
            raise PaymentError("Missing feePayer in 402 extra field")

        resource_info = details.get("resource") or {}
        resource_url = validate_resource_url(
            resource_info.get("url") or url,
            self._api_url,
        )

        payment_payload = create_solana_payment_payload(
            private_key=self._private_key,
            recipient=details["recipient"],
            amount=details["amount"],
            fee_payer=fee_payer,
            resource_url=resource_url,
            resource_description=resource_info.get("description") or "BlockRun Solana AI API call",
            max_timeout_seconds=details["max_timeout_seconds"],
            extra=details.get("extra"),
            rpc_url=self._rpc_url,
        )

        headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": payment_payload,
        }

        with httpx.Client(timeout=self._timeout) as http:
            retry_response = http.post(url, json=body, headers=headers)

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

        cost_usd = float(details["amount"]) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd

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
