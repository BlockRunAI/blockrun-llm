"""Tests for per-use-case + per-call HTTP timeout routing on the Solana client.

Covers the second half of issue #7. v0.34.0 defines ``DEFAULT_CHAT_TIMEOUT``
(120s), ``DEFAULT_IMAGE_TIMEOUT`` (200s) and ``DEFAULT_SEARCH_TIMEOUT`` (300s),
but the constants are only useful if each method actually *applies* the right
one to its httpx request — and if a caller's per-call ``timeout=`` overrides it.

Pre-fix every method flowed through the single ``httpx.Client(timeout=...)``
default, so ``image()`` and ``search()`` silently used the chat budget. These
tests assert the real per-request timeout that reaches the transport via
``request.extensions["timeout"]`` so a future regression that drops the routing
fails loudly.

Mocked at the httpx transport level (no network); the x402 signer and the
decode/encode helpers are stubbed exactly like ``test_streaming_solana``.
"""

from __future__ import annotations

import unittest.mock as mock
from typing import List

import httpx
import pytest

pytest.importorskip("x402")
pytest.importorskip("solders")

from blockrun_llm import SolanaLLMClient  # noqa: E402
from blockrun_llm import solana_client as sol  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _read_timeout(request: httpx.Request) -> float:
    """The resolved read timeout that reached the transport for this request."""
    return request.extensions["timeout"]["read"]


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a cache miss + no-op writes so every call hits the transport."""
    monkeypatch.setattr("blockrun_llm.cache.get_cached", lambda *a, **k: None)
    monkeypatch.setattr("blockrun_llm.cache.save_to_cache", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _stub_x402_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blockrun_llm.solana_client.decode_payment_required_header",
        lambda header: {"stub": True},
    )
    monkeypatch.setattr(
        "blockrun_llm.solana_client.encode_payment_signature_header",
        lambda payload: "stub-signature",
    )


def _make_client(transport: httpx.MockTransport, **kwargs: float) -> SolanaLLMClient:
    with (
        mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
        mock.patch("blockrun_llm.solana_client._create_signer"),
    ):
        client = SolanaLLMClient(
            private_key="bogus_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
            **kwargs,
        )

    class _FakePayload:
        class accepted:
            amount = "1000000"
            pay_to = "GsbwXfJraMomNxBcpR3DBNxnKwZbyq7YCoDdSLDwzxdV"

    client._x402_client = mock.MagicMock()
    client._x402_client.create_payment_payload.return_value = _FakePayload()
    client._client = httpx.Client(transport=transport)
    # Pre-seed the wallet address so billing metadata doesn't try to base58
    # decode the patched-out bogus key (system program address — valid b58).
    client._address = "11111111111111111111111111111111"
    return client


def _payment_required(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        402,
        headers={"content-type": "application/json", "payment-required": "stub-header"},
        json={"error": "Payment Required"},
    )


_CHAT_OK = {
    "id": "chatcmpl-1",
    "created": 1700000000,
    "model": "openai/gpt-5.2",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
}
_IMAGE_OK = {"created": 1700000000, "data": [{"url": "https://blockrun.ai/i.png"}]}
_SEARCH_OK = {"query": "q", "summary": "s"}


def _json_flow(calls: List[httpx.Request], ok_body: dict) -> httpx.MockTransport:
    """402 on the unsigned probe, then the success body once signed."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if "PAYMENT-SIGNATURE" not in request.headers:
            return _payment_required(request)
        return httpx.Response(200, json=ok_body, headers={"content-type": "application/json"})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Per-use-case defaults
# ---------------------------------------------------------------------------


def test_chat_uses_chat_timeout_default() -> None:
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, _CHAT_OK))
    client.chat_completion("openai/gpt-5.2", [{"role": "user", "content": "hi"}])
    assert _read_timeout(calls[-1]) == sol.DEFAULT_CHAT_TIMEOUT


def test_image_uses_image_timeout_default() -> None:
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, _IMAGE_OK))
    client.image("a cat", model="openai/gpt-image-2")
    # Probe + signed submit both carry the image budget, not the chat one.
    assert _read_timeout(calls[0]) == sol.DEFAULT_IMAGE_TIMEOUT
    assert _read_timeout(calls[-1]) == sol.DEFAULT_IMAGE_TIMEOUT
    assert sol.DEFAULT_IMAGE_TIMEOUT != sol.DEFAULT_CHAT_TIMEOUT  # regression: not chat


def test_search_uses_search_timeout_default() -> None:
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, _SEARCH_OK))
    client.search("deep query")
    assert _read_timeout(calls[-1]) == sol.DEFAULT_SEARCH_TIMEOUT


def test_exa_uses_search_timeout_default() -> None:
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, {"results": []}))
    client.exa_search("latest AI papers")
    assert _read_timeout(calls[-1]) == sol.DEFAULT_SEARCH_TIMEOUT


# ---------------------------------------------------------------------------
# Per-call override wins over every default
# ---------------------------------------------------------------------------


def test_chat_per_call_override() -> None:
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, _CHAT_OK))
    client.chat_completion("openai/gpt-5.2", [{"role": "user", "content": "hi"}], timeout=7.0)
    assert _read_timeout(calls[0]) == 7.0
    assert _read_timeout(calls[-1]) == 7.0


def test_image_per_call_override() -> None:
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, _IMAGE_OK))
    client.image("a cat", model="openai/gpt-image-2", timeout=9.0)
    assert _read_timeout(calls[-1]) == 9.0


def test_search_per_call_override() -> None:
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, _SEARCH_OK))
    client.search("q", timeout=11.0)
    assert _read_timeout(calls[-1]) == 11.0


# ---------------------------------------------------------------------------
# Constructor-level overrides
# ---------------------------------------------------------------------------


def test_constructor_image_and_search_timeout_respected() -> None:
    img_calls: List[httpx.Request] = []
    img_client = _make_client(_json_flow(img_calls, _IMAGE_OK), image_timeout=42.0)
    img_client.image("a cat", model="openai/gpt-image-2")
    assert _read_timeout(img_calls[-1]) == 42.0

    s_calls: List[httpx.Request] = []
    s_client = _make_client(_json_flow(s_calls, _SEARCH_OK), search_timeout=99.0)
    s_client.search("q")
    assert _read_timeout(s_calls[-1]) == 99.0


def test_legacy_flat_timeout_still_governs_chat() -> None:
    """Backwards-compat: old ``SolanaLLMClient(timeout=...)`` callers keep
    controlling the chat budget through the single keyword."""
    calls: List[httpx.Request] = []
    client = _make_client(_json_flow(calls, _CHAT_OK), timeout=33.0)
    client.chat_completion("openai/gpt-5.2", [{"role": "user", "content": "hi"}])
    assert _read_timeout(calls[-1]) == 33.0


# ---------------------------------------------------------------------------
# Async mirror — the threading is symmetric, so cover chat both ways.
# ---------------------------------------------------------------------------


def _make_async_client(transport: httpx.MockTransport, **kwargs: float):
    from blockrun_llm.solana_client import AsyncSolanaLLMClient

    with (
        mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
        mock.patch("blockrun_llm.solana_client._create_signer"),
        mock.patch("x402.x402Client"),
    ):
        client = AsyncSolanaLLMClient(
            private_key="bogus_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
            **kwargs,
        )

    class _FakePayload:
        class accepted:
            amount = "1000000"
            pay_to = "GsbwXfJraMomNxBcpR3DBNxnKwZbyq7YCoDdSLDwzxdV"

    client._x402_client = mock.MagicMock()
    client._x402_client.create_payment_payload = mock.AsyncMock(return_value=_FakePayload())
    client._client = httpx.AsyncClient(transport=transport)
    client._address = "11111111111111111111111111111111"
    return client


async def test_async_chat_uses_chat_timeout_default() -> None:
    calls: List[httpx.Request] = []
    client = _make_async_client(_json_flow(calls, _CHAT_OK))
    await client.chat_completion("openai/gpt-5.2", [{"role": "user", "content": "hi"}])
    assert _read_timeout(calls[-1]) == sol.DEFAULT_CHAT_TIMEOUT
    await client.close()


async def test_async_chat_per_call_override() -> None:
    calls: List[httpx.Request] = []
    client = _make_async_client(_json_flow(calls, _CHAT_OK))
    await client.chat_completion(
        "openai/gpt-5.2", [{"role": "user", "content": "hi"}], timeout=13.0
    )
    assert _read_timeout(calls[0]) == 13.0
    assert _read_timeout(calls[-1]) == 13.0
    await client.close()
