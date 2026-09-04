"""Generic account API access for Responses and all gateway service endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx
from typing_extensions import Self

from .api_key import resolve_api_auth


class APIClient:
    """Account-billed GET/POST/stream/poll client; no wallet is created."""

    def __init__(
        self, api_key: str | None = None, api_url: str | None = None, timeout: float = 600
    ):
        auth = resolve_api_auth(api_key, None, api_url)
        if auth is None:
            raise ValueError("Set BLOCKRUN_API_KEY or pass api_key")
        self._auth = auth
        self._client = httpx.Client(auth=auth, timeout=timeout, follow_redirects=False)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._client.get(self._auth.resolve_url(path), params=params).json()

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._client.post(self._auth.resolve_url(path), json=body or {}).json()

    def poll(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        budget_seconds: float = 900,
        interval_seconds: float = 2,
    ) -> dict[str, Any]:
        response = self._client.post(self._auth.resolve_url(path), json=body or {})
        return self._auth.poll(self._client, response, budget_seconds, interval_seconds)

    @contextmanager
    def stream(self, path: str, body: dict[str, Any]) -> Iterator[httpx.Response]:
        with self._client.stream(
            "POST", self._auth.resolve_url(path), json={**body, "stream": True}
        ) as response:
            yield response

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncAPIClient:
    """Async mirror of APIClient, including streaming and task polling."""

    def __init__(
        self, api_key: str | None = None, api_url: str | None = None, timeout: float = 600
    ):
        auth = resolve_api_auth(api_key, None, api_url)
        if auth is None:
            raise ValueError("Set BLOCKRUN_API_KEY or pass api_key")
        self._auth = auth
        self._client = httpx.AsyncClient(auth=auth, timeout=timeout, follow_redirects=False)

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return (await self._client.get(self._auth.resolve_url(path), params=params)).json()

    async def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return (await self._client.post(self._auth.resolve_url(path), json=body or {})).json()

    async def poll(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        budget_seconds: float = 900,
        interval_seconds: float = 2,
    ) -> dict[str, Any]:
        response = await self._client.post(self._auth.resolve_url(path), json=body or {})
        return await self._auth.apoll(self._client, response, budget_seconds, interval_seconds)

    @asynccontextmanager
    async def stream(self, path: str, body: dict[str, Any]) -> AsyncIterator[httpx.Response]:
        async with self._client.stream(
            "POST", self._auth.resolve_url(path), json={**body, "stream": True}
        ) as response:
            yield response

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
