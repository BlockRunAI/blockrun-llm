"""Account-key authentication shared by synchronous and asynchronous clients."""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

import httpx
from eth_account.signers.local import LocalAccount

from .types import APIError
from .validation import validate_api_url

API_KEY_URL = "https://api.blockrun.ai"
PORTAL_URL = "https://user.blockrun.ai"


def resolve_api_auth(
    api_key: str | None, private_key: str | None, api_url: str | None
) -> ApiKeyAuth | None:
    if api_key is not None and private_key is not None:
        raise ValueError("Pass either api_key or private_key, not both.")
    key = (
        api_key
        if api_key is not None
        else (os.getenv("BLOCKRUN_API_KEY") if private_key is None else None)
    )
    if key is None:
        return None
    if not re.fullmatch(r"brk_[A-Za-z0-9_-]+", key.strip()):
        raise ValueError(f"Invalid BlockRun API key. Create one at {PORTAL_URL}/dashboard/keys.")
    return ApiKeyAuth(key.strip(), api_url or os.getenv("BLOCKRUN_API_BASE_URL") or API_KEY_URL)


class ApiKeyAuth(httpx.Auth):
    """Never signs/replays an account's quota error as a wallet payment."""

    def __init__(self, key: str, api_url: str) -> None:
        validate_api_url(api_url)
        url = httpx.URL(api_url)
        if url.userinfo or url.query or url.fragment:
            raise ValueError("API URL cannot contain credentials, query, or fragment")
        self.api_url = api_url.rstrip("/").removesuffix("/v1")
        self.__key = key
        self.raise_errors = True

    def resolve_url(self, path: str) -> str:
        url = httpx.URL(self.api_url + "/").join(path)
        base = httpx.URL(self.api_url)
        if (url.scheme, url.host, url.port) != (base.scheme, base.host, base.port) or url.userinfo:
            raise ValueError("Refusing to send a BlockRun API key to another origin")
        if base.path == "/" and url.path.startswith("/api/v1/"):
            url = url.copy_with(path=url.path[4:])
        return str(url)

    def _prepare(self, request: httpx.Request) -> None:
        request.url = httpx.URL(self.resolve_url(str(request.url)))
        for name in list(request.headers):
            if "payment" in name.lower() or name.lower() == "x-api-key":
                del request.headers[name]
        request.headers["authorization"] = f"Bearer {self.__key}"

    def _error(self, response: httpx.Response) -> APIError:
        try:
            body = response.json()
        except ValueError:
            body = {}
        detail = body.get("error", body) if isinstance(body, dict) else {}
        safe = {
            name: detail[name].replace(self.__key, "[REDACTED]")
            for name in ("message", "code", "type", "param")
            if isinstance(detail, dict) and isinstance(detail.get(name), str)
        }
        hint = f" Top up at {PORTAL_URL}/dashboard/credits." if response.status_code == 402 else ""
        error = APIError(
            f"BlockRun account API error: {response.status_code}.{hint}", response.status_code, safe
        )
        error.retry_after = response.headers.get("retry-after")
        return error

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        self._prepare(request)
        response = yield request
        if self.raise_errors and not response.is_success:
            response.read()
            error = self._error(response)
            response.close()
            raise error

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        self._prepare(request)
        response = yield request
        if self.raise_errors and not response.is_success:
            await response.aread()
            error = self._error(response)
            await response.aclose()
            raise error

    @staticmethod
    def _terminal(data: dict[str, Any]) -> bool:
        if data.get("status") in ("failed", "cancelled", "canceled"):
            raise APIError("Account API job failed or was cancelled", 502)
        return data.get("status") == "completed"

    def poll(
        self, client: httpx.Client, response: httpx.Response, budget: float, interval: float = 2.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + budget
        data = response.json()
        if self._terminal(data):
            return cast(dict[str, Any], data)
        path = data.get("poll_url")
        if not path:
            if response.status_code == 202 or data.get("status") in (
                "queued",
                "in_progress",
                "processing",
            ):
                raise APIError("Async response missing poll_url", response.status_code)
            return cast(dict[str, Any], data)
        url = self.resolve_url(path)
        while time.monotonic() < deadline:
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            data = client.get(url, timeout=remaining).json()
            if self._terminal(data):
                return cast(dict[str, Any], data)
        raise APIError(
            "Account job polling timed out; check the job before resubmitting",
            504,
            {"poll_url": url},
        )

    async def apoll(
        self,
        client: httpx.AsyncClient,
        response: httpx.Response,
        budget: float,
        interval: float = 2.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + budget
        data = response.json()
        if self._terminal(data):
            return cast(dict[str, Any], data)
        path = data.get("poll_url")
        if not path:
            if response.status_code == 202 or data.get("status") in (
                "queued",
                "in_progress",
                "processing",
            ):
                raise APIError("Async response missing poll_url", response.status_code)
            return cast(dict[str, Any], data)
        url = self.resolve_url(path)
        while time.monotonic() < deadline:
            await asyncio.sleep(min(interval, max(0, deadline - time.monotonic())))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            data = (await client.get(url, timeout=remaining)).json()
            if self._terminal(data):
                return cast(dict[str, Any], data)
        raise APIError(
            "Account job polling timed out; check the job before resubmitting",
            504,
            {"poll_url": url},
        )


class AccountMode:
    _api_auth: ApiKeyAuth | None = None

    @property
    def auth_mode(self) -> str:
        return "api-key" if self._api_auth else "wallet"

    def _require_wallet_mode(self) -> None:
        if self._api_auth:
            raise ValueError(
                f"This operation requires a wallet. Account usage: {PORTAL_URL}/dashboard"
            )


class EvmAccountMode(AccountMode):
    _wallet_account: LocalAccount | None = None

    @property
    def account(self) -> LocalAccount:
        self._require_wallet_mode()
        if self._wallet_account is None:
            raise ValueError("No wallet configured")
        return self._wallet_account

    @account.setter
    def account(self, value: LocalAccount) -> None:
        self._wallet_account = value
