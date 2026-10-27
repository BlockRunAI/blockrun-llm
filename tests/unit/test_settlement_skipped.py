"""A paid response the gateway served WITHOUT settling books $0.

The gateways answer some paid requests without settling them and say so on
the 200:

* sol ``/v1/pm/*`` empty result: ``X-Payment-Waived: empty-result``,
  ``X-Payment-Settled: false``, ``X-Settlement-Skipped: empty-result`` and no
  PAYMENT-RESPONSE.
* sol ``/v1/rpc/*`` unexecuted JSON-RPC: ``X-Payment-Settled: false``,
  ``X-Settlement-Skipped: unexecuted``.
* chat free fallback: ``X-Settlement-Skipped: free-fallback``.
* base pm route: ``X-Payment-Waived: empty-result``.

No money moved for those calls, so every booking site (session totals,
``_last_call_cost``, cost log via ``save_to_cache``, tx log, spend-limit
accounting) must record 0.0, on Base and Solana, sync and async.

``X-Payment-Settled: false`` ALONE is not a skip marker: it means settlement
had not landed when the response was written, and the call is still owed. It
must keep booking the full quote.
"""

from __future__ import annotations

import base64
import json
import warnings
from typing import Any
from unittest import mock

import httpx
import pytest

from blockrun_llm import AsyncLLMClient, LLMClient
from blockrun_llm.tx_log import booked_cost_usd, settlement_skipped
from blockrun_llm.types import SpendLimitError

from ..helpers import TEST_PRIVATE_KEY, build_chat_response, build_payment_required_response

MESSAGES = [{"role": "user", "content": "hi"}]


def _b64(value: Any) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode()


QUOTE_USD = 1.0  # build_payment_required_response / the Solana fake payload: 1 USDC

# What sol.blockrun.ai sends on an empty pm search (the full marker set).
SOL_PM_EMPTY = {
    "X-Payment-Waived": "empty-result",
    "X-Payment-Settled": "false",
    "X-Settlement-Skipped": "empty-result",
}

# (response headers, expected booked cost)
CASES = [
    pytest.param(SOL_PM_EMPTY, 0.0, id="skipped"),
    pytest.param({"X-Payment-Settled": "false"}, QUOTE_USD, id="settled-false-alone-still-owed"),
    pytest.param({}, QUOTE_USD, id="normal-paid"),
]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


class TestSettlementSkippedHelper:
    @pytest.mark.parametrize(
        "headers",
        [
            SOL_PM_EMPTY,
            {"X-Payment-Settled": "false", "X-Settlement-Skipped": "unexecuted"},  # sol rpc
            {"X-Settlement-Skipped": "free-fallback"},  # chat free fallback
            {"X-Payment-Waived": "empty-result"},  # base pm
            {"x-settlement-skipped": "empty-result"},  # case-insensitive, plain dict
            {"X-PAYMENT-WAIVED": ""},  # waived: presence is enough
        ],
    )
    def test_skip_markers(self, headers: dict[str, str]) -> None:
        assert settlement_skipped(headers) is True
        assert settlement_skipped(httpx.Headers(headers)) is True
        assert booked_cost_usd(headers, 0.25) == 0.0

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"X-Payment-Settled": "false"},  # late settlement: still owed
            {"X-Payment-Settled": "true"},
            {"X-Settlement-Skipped": ""},  # empty value is not a marker
            {"X-Settlement-Skipped": "   "},
            {"PAYMENT-RESPONSE": "e30="},
        ],
    )
    def test_not_skipped(self, headers: dict[str, str]) -> None:
        assert settlement_skipped(headers) is False
        assert settlement_skipped(httpx.Headers(headers)) is False
        assert booked_cost_usd(headers, 0.25) == 0.25

    def test_never_raises(self) -> None:
        assert settlement_skipped(None) is False
        assert settlement_skipped(object()) is False

    @pytest.mark.parametrize(
        "headers",
        [
            # A value that says it was NOT skipped.
            {"X-Settlement-Skipped": "false"},
            {"X-Settlement-Skipped": "0", "X-Payment-Waived": "none"},
            # Settlement evidence beats the marker.
            {
                "X-Settlement-Skipped": "free-fallback",
                "PAYMENT-RESPONSE": _b64({"success": True, "transaction": "0xpaid"}),
            },
            {
                "X-Payment-Waived": "empty-result",
                "PAYMENT-RESPONSE": _b64({"success": True, "signature": "5sig"}),
            },
            {
                "X-Payment-Waived": "empty-result",
                "PAYMENT-RESPONSE": _b64(
                    {"success": True, "transaction": "", "extra": {"chargedAmount": "4000"}}
                ),
            },
            {
                "X-Settlement-Skipped": "unexecuted",
                "PAYMENT-RESPONSE": _b64({"success": True, "transaction": "", "amount": "4000"}),
            },
            {"X-Settlement-Skipped": "unexecuted", "X-Payment-Receipt": "0xpaid"},
        ],
    )
    def test_evidence_or_negative_value_books(
        self, headers: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        assert settlement_skipped(headers) is False
        assert settlement_skipped(httpx.Headers(headers)) is False
        assert booked_cost_usd(headers, 0.25) == 0.25

    def test_conflict_is_written_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        headers = {
            "X-Settlement-Skipped": "free-fallback",
            "PAYMENT-RESPONSE": _b64({"success": True, "transaction": "0xpaid"}),
        }
        assert settlement_skipped(headers) is False
        err = capsys.readouterr().err
        assert "skipped settlement ('free-fallback')" in err
        assert "booked as charged" in err

    @pytest.mark.parametrize(
        "headers",
        [
            # blockrun.ai's free fallback: a receipt with an empty transaction and no amount.
            {
                "X-Settlement-Skipped": "free-fallback",
                "PAYMENT-RESPONSE": _b64({"success": True, "transaction": ""}),
            },
            # A failed receipt is no evidence.
            {
                "X-Settlement-Skipped": "unexecuted",
                "PAYMENT-RESPONSE": _b64({"success": False, "transaction": "0xfailed"}),
            },
        ],
    )
    def test_receipt_without_evidence_stays_skipped(
        self, headers: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        assert settlement_skipped(headers) is True
        assert booked_cost_usd(headers, 0.25) == 0.0
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


@pytest.fixture
def cost_log(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the cost every save_to_cache call books; never touch the disk."""
    booked: list[float] = []

    def _save(*_a: Any, **kw: Any) -> None:
        booked.append(kw.get("cost_usd"))

    monkeypatch.setattr("blockrun_llm.cache.get_cached", lambda *a, **k: None)
    monkeypatch.setattr("blockrun_llm.cache.save_to_cache", _save)
    return booked


def _record_tx_log(client: Any) -> list[float]:
    logged: list[float] = []

    def _log(_endpoint: Any, _body: Any, _resp: Any, cost_usd: float, **_kw: Any) -> None:
        logged.append(cost_usd)

    client._log_transaction = _log
    return logged


def _evm_handler(ok_headers: dict[str, str], ok_body: Any, *, sse: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={"payment-required": build_payment_required_response()},
            )
        if sse:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", **ok_headers},
                content=ok_body,
            )
        return httpx.Response(200, json=ok_body, headers=ok_headers)

    return handler


_SSE = (
    b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},'
    b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
)
_PM_BODY = {"markets": [], "pagination": {"has_more": False}}


def _assert_quiet(caught: list[warnings.WarningMessage], capsys: pytest.CaptureFixture) -> None:
    assert not caught, [str(w.message) for w in caught]
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Base (EVM) — LLMClient
# ---------------------------------------------------------------------------


def _evm_client(handler: Any, **kwargs: Any) -> LLMClient:
    client = LLMClient(private_key=TEST_PRIVATE_KEY, **kwargs)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.parametrize("headers,expected", CASES)
class TestEvmSync:
    def test_chat(self, headers, expected, cost_log, capsys) -> None:
        client = _evm_client(_evm_handler(headers, build_chat_response()))
        tx = _record_tx_log(client)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resp = client.chat_completion("openai/gpt-5.5", MESSAGES)
        assert resp.cost_usd == expected
        assert client.get_spending() == {"total_usd": expected, "calls": 1, "ceiling_usd": 0.0}
        assert client._last_call_cost == expected
        assert cost_log == [expected] and tx == [expected]
        if expected == 0.0:
            _assert_quiet(caught, capsys)

    def test_chat_stream(self, headers, expected, cost_log) -> None:
        client = _evm_client(_evm_handler(headers, _SSE, sse=True))
        tx = _record_tx_log(client)
        chunks = list(client.chat_completion_stream("openai/gpt-5.5", MESSAGES))
        assert chunks and all(c.cost_usd == expected for c in chunks)
        assert client.get_spending()["total_usd"] == expected
        assert client._last_call_cost == expected
        # A $0 stream is not archived (same rule as a free model).
        expect_rows = [expected] if expected else []
        assert cost_log == expect_rows and tx == expect_rows

    def test_pm_get(self, headers, expected, cost_log, capsys) -> None:
        client = _evm_client(_evm_handler(headers, _PM_BODY))
        tx = _record_tx_log(client)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert client.pm("polymarket/search", q="zzz") == _PM_BODY
        assert client.get_spending()["total_usd"] == expected
        assert client._last_call_cost == expected
        assert cost_log == [expected] and tx == [expected]
        if expected == 0.0:
            _assert_quiet(caught, capsys)

    def test_pm_query_post(self, headers, expected, cost_log) -> None:
        client = _evm_client(_evm_handler(headers, _PM_BODY))
        tx = _record_tx_log(client)
        assert client.pm_query("polymarket/wallet/identities", {"wallets": []}) == _PM_BODY
        assert client.get_spending()["total_usd"] == expected
        assert client._last_call_cost == expected
        assert cost_log == [expected] and tx == [expected]


def test_evm_skipped_calls_do_not_consume_the_session_limit(cost_log) -> None:
    """Spend-limit accounting reads the session total: two skipped $1 calls
    leave the whole $1.5 budget, so a real $1 call still goes through."""
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={"payment-required": build_payment_required_response()},
            )
        state["n"] += 1
        return httpx.Response(200, json=_PM_BODY, headers=SOL_PM_EMPTY if state["n"] <= 2 else {})

    client = _evm_client(handler, max_session_cost=1.5)
    client.pm("a")
    client.pm("b")
    client.pm("c")  # would raise SpendLimitError if the skipped calls had booked
    assert client.get_spending()["total_usd"] == QUOTE_USD
    with pytest.raises(SpendLimitError):
        client.pm("d")


@pytest.mark.parametrize("headers,expected", CASES)
class TestEvmAsync:
    """AsyncLLMClient tracks only ``_last_call_cost`` (no session totals)."""

    def _client(self, handler: Any) -> AsyncLLMClient:
        client = AsyncLLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    async def test_chat(self, headers, expected, cost_log) -> None:
        client = self._client(_evm_handler(headers, build_chat_response()))
        tx = _record_tx_log(client)
        resp = await client.chat_completion("openai/gpt-5.5", MESSAGES)
        assert resp.cost_usd == expected
        assert client._last_call_cost == expected
        assert cost_log == [expected] and tx == [expected]
        await client.close()

    async def test_chat_stream(self, headers, expected, cost_log) -> None:
        client = self._client(_evm_handler(headers, _SSE, sse=True))
        tx = _record_tx_log(client)
        chunks = [c async for c in client.chat_completion_stream("openai/gpt-5.5", MESSAGES)]
        assert chunks and all(c.cost_usd == expected for c in chunks)
        assert client._last_call_cost == expected
        expect_rows = [expected] if expected else []
        assert cost_log == expect_rows and tx == expect_rows
        await client.close()

    async def test_pm_get(self, headers, expected, cost_log) -> None:
        client = self._client(_evm_handler(headers, _PM_BODY))
        tx = _record_tx_log(client)
        assert await client.pm("polymarket/search", q="zzz") == _PM_BODY
        assert client._last_call_cost == expected
        assert cost_log == [expected] and tx == [expected]
        await client.close()

    async def test_pm_query_post(self, headers, expected, cost_log) -> None:
        client = self._client(_evm_handler(headers, _PM_BODY))
        tx = _record_tx_log(client)
        assert await client.pm_query("polymarket/wallet/identities", {"wallets": []}) == _PM_BODY
        assert client._last_call_cost == expected
        assert cost_log == [expected] and tx == [expected]
        await client.close()


# ---------------------------------------------------------------------------
# Solana — SolanaLLMClient / AsyncSolanaLLMClient
# ---------------------------------------------------------------------------


class _FakePayload:
    class accepted:
        amount = "1000000"
        pay_to = "GsbwXfJraMomNxBcpR3DBNxnKwZbyq7YCoDdSLDwzxdV"


@pytest.fixture
def sol(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("x402")
    pytest.importorskip("solders")
    monkeypatch.setattr(
        "blockrun_llm.solana_client.decode_payment_required_header", lambda header: {"stub": True}
    )
    monkeypatch.setattr(
        "blockrun_llm.solana_client.encode_payment_signature_header", lambda payload: "stub-sig"
    )
    from blockrun_llm import solana_client

    def make(handler: Any, *, is_async: bool = False, **kwargs: Any):
        cls = solana_client.AsyncSolanaLLMClient if is_async else solana_client.SolanaLLMClient
        with (
            mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
            mock.patch("blockrun_llm.solana_client._create_signer"),
        ):
            client = cls(
                private_key="bogus_signer_is_patched",
                api_url="https://sol.blockrun.ai/api",
                rpc_url="http://test",
                **kwargs,
            )
        client._x402_client = mock.MagicMock()
        if is_async:
            client._x402_client.create_payment_payload = mock.AsyncMock(return_value=_FakePayload())
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        else:
            client._x402_client.create_payment_payload.return_value = _FakePayload()
            client._client = httpx.Client(transport=httpx.MockTransport(handler))
        client._address = "11111111111111111111111111111111"
        return client

    return make


def _sol_handler(ok_headers: dict[str, str], ok_body: Any, *, sse: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                headers={"content-type": "application/json", "payment-required": "stub"},
                json={"error": "Payment Required"},
            )
        if sse:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream", **ok_headers}, content=ok_body
            )
        return httpx.Response(200, json=ok_body, headers=ok_headers)

    return handler


_IMAGE_BODY = {"model": "openai/gpt-image-2", "prompt": "a cat"}
_IMAGE_OK = {"created": 1, "model": "openai/gpt-image-2", "data": [{"url": "https://cdn/x.png"}]}
_VIDEO_BODY = {"model": "xai/grok-imagine-video", "prompt": "a cat"}
_VIDEO_DONE = {
    "status": "completed",
    "created": 1,
    "model": "xai/grok-imagine-video",
    "data": [{"url": "https://cdn/v.mp4"}],
}
_POLL_KW: dict[str, Any] = {"poll_budget_seconds": 5.0, "poll_interval_seconds": 0.001}


def _media_handler(
    *, submit_status: int, submit_headers: dict[str, str], poll_headers: dict[str, str]
):
    """probe 402 → signed POST (200 inline or 202 + poll_url) → GET poll completed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if "PAYMENT-SIGNATURE" not in request.headers:
                return httpx.Response(
                    402,
                    headers={"content-type": "application/json", "payment-required": "stub"},
                    json={"error": "Payment Required"},
                )
            if submit_status == 200:
                return httpx.Response(200, json=_IMAGE_OK, headers=submit_headers)
            return httpx.Response(
                202,
                json={"id": "JOB", "poll_url": "/api/v1/x/JOB", "status": "queued"},
                headers=submit_headers,
            )
        return httpx.Response(200, json=_VIDEO_DONE, headers=poll_headers)

    return handler


def _assert_sol_booked(client: Any, expected: float, *, sync: bool = True) -> None:
    assert client._last_call_cost == expected
    assert client._session_total_usd == expected
    assert client._session_calls == 1
    if sync:
        assert client.get_spending()["total_usd"] == expected


@pytest.mark.parametrize("headers,expected", CASES)
class TestSolanaSync:
    def test_chat(self, sol, headers, expected, cost_log, capsys) -> None:
        client = sol(_sol_handler(headers, build_chat_response()))
        tx = _record_tx_log(client)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.chat_completion("openai/gpt-5.5", MESSAGES)
        _assert_sol_booked(client, expected)
        assert cost_log == [expected] and tx == [expected]
        if expected == 0.0:
            _assert_quiet(caught, capsys)

    def test_chat_stream(self, sol, headers, expected, cost_log) -> None:
        client = sol(_sol_handler(headers, _SSE, sse=True))
        tx = _record_tx_log(client)
        assert list(client.chat_completion_stream("openai/gpt-5.5", MESSAGES))
        _assert_sol_booked(client, expected)
        expect_rows = [expected] if expected else []
        assert cost_log == expect_rows and tx == expect_rows

    def test_pm_get(self, sol, headers, expected, cost_log, capsys) -> None:
        client = sol(_sol_handler(headers, _PM_BODY))
        tx = _record_tx_log(client)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert client.pm("markets/search", q="zzz") == _PM_BODY
        _assert_sol_booked(client, expected)
        assert cost_log == [expected] and tx == [expected]
        if expected == 0.0:
            _assert_quiet(caught, capsys)

    def test_pm_query_post(self, sol, headers, expected, cost_log) -> None:
        client = sol(_sol_handler(headers, _PM_BODY))
        tx = _record_tx_log(client)
        assert client.pm_query("polymarket/wallet/identities", {"wallets": []}) == _PM_BODY
        _assert_sol_booked(client, expected)
        assert cost_log == [expected] and tx == [expected]

    def test_image_inline(self, sol, headers, expected, cost_log) -> None:
        client = sol(_media_handler(submit_status=200, submit_headers=headers, poll_headers={}))
        tx = _record_tx_log(client)
        client._request_image_with_payment(
            "/v1/images/generations", dict(_IMAGE_BODY), settled_at_submit=True, **_POLL_KW
        )
        _assert_sol_booked(client, expected)
        assert cost_log == [expected] and tx == [expected]

    def test_image_settled_at_submit(self, sol, headers, expected, cost_log) -> None:
        # The 202 submit is the settling response; the poll carries no marker.
        client = sol(_media_handler(submit_status=202, submit_headers=headers, poll_headers={}))
        tx = _record_tx_log(client)
        client._request_image_with_payment(
            "/v1/images/generations", dict(_IMAGE_BODY), settled_at_submit=True, **_POLL_KW
        )
        _assert_sol_booked(client, expected)
        assert cost_log == [expected] and tx == [expected]

    def test_video_completion_poll(self, sol, headers, expected, cost_log) -> None:
        # Video settles on the completed poll, so that response's markers count.
        client = sol(_media_handler(submit_status=202, submit_headers={}, poll_headers=headers))
        tx = _record_tx_log(client)
        client._request_image_with_payment(
            "/v1/videos/generations", dict(_VIDEO_BODY), max_resigns=2, **_POLL_KW
        )
        _assert_sol_booked(client, expected)
        assert cost_log == [expected] and tx == [expected]


def test_solana_skipped_calls_do_not_consume_the_session_limit(sol, cost_log) -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                headers={"content-type": "application/json", "payment-required": "stub"},
                json={"error": "Payment Required"},
            )
        state["n"] += 1
        return httpx.Response(200, json=_PM_BODY, headers=SOL_PM_EMPTY if state["n"] <= 2 else {})

    client = sol(handler, max_session_cost=1.5)
    client.pm_query("a", {})
    client.pm_query("b", {})
    client.pm_query("c", {})  # would raise SpendLimitError if the skipped calls had booked
    assert client.get_spending()["total_usd"] == QUOTE_USD
    with pytest.raises(SpendLimitError):
        client.pm_query("d", {})


@pytest.mark.parametrize("headers,expected", CASES)
class TestSolanaAsync:
    async def test_chat(self, sol, headers, expected, cost_log) -> None:
        client = sol(_sol_handler(headers, build_chat_response()), is_async=True)
        tx = _record_tx_log(client)
        await client.chat_completion("openai/gpt-5.5", MESSAGES)
        _assert_sol_booked(client, expected, sync=False)
        assert cost_log == [expected] and tx == [expected]
        await client._client.aclose()

    async def test_chat_stream(self, sol, headers, expected, cost_log) -> None:
        client = sol(_sol_handler(headers, _SSE, sse=True), is_async=True)
        tx = _record_tx_log(client)
        assert [c async for c in client.chat_completion_stream("openai/gpt-5.5", MESSAGES)]
        _assert_sol_booked(client, expected, sync=False)
        expect_rows = [expected] if expected else []
        assert cost_log == expect_rows and tx == expect_rows
        await client._client.aclose()

    async def test_pm_get(self, sol, headers, expected, cost_log) -> None:
        client = sol(_sol_handler(headers, _PM_BODY), is_async=True)
        tx = _record_tx_log(client)
        assert await client.pm("markets/search", q="zzz") == _PM_BODY
        _assert_sol_booked(client, expected, sync=False)
        assert cost_log == [expected] and tx == [expected]
        await client._client.aclose()

    async def test_pm_query_post(self, sol, headers, expected, cost_log) -> None:
        client = sol(_sol_handler(headers, _PM_BODY), is_async=True)
        tx = _record_tx_log(client)
        assert await client.pm_query("polymarket/wallet/identities", {"wallets": []}) == _PM_BODY
        _assert_sol_booked(client, expected, sync=False)
        assert cost_log == [expected] and tx == [expected]
        await client._client.aclose()

    async def test_image_inline(self, sol, headers, expected, cost_log) -> None:
        client = sol(
            _media_handler(submit_status=200, submit_headers=headers, poll_headers={}),
            is_async=True,
        )
        tx = _record_tx_log(client)
        await client._request_image_with_payment(
            "/v1/images/generations", dict(_IMAGE_BODY), settled_at_submit=True, **_POLL_KW
        )
        _assert_sol_booked(client, expected, sync=False)
        assert cost_log == [expected] and tx == [expected]
        await client._client.aclose()

    async def test_image_settled_at_submit(self, sol, headers, expected, cost_log) -> None:
        client = sol(
            _media_handler(submit_status=202, submit_headers=headers, poll_headers={}),
            is_async=True,
        )
        tx = _record_tx_log(client)
        await client._request_image_with_payment(
            "/v1/images/generations", dict(_IMAGE_BODY), settled_at_submit=True, **_POLL_KW
        )
        _assert_sol_booked(client, expected, sync=False)
        assert cost_log == [expected] and tx == [expected]
        await client._client.aclose()

    async def test_video_completion_poll(self, sol, headers, expected, cost_log) -> None:
        client = sol(
            _media_handler(submit_status=202, submit_headers={}, poll_headers=headers),
            is_async=True,
        )
        tx = _record_tx_log(client)
        await client._request_image_with_payment(
            "/v1/videos/generations", dict(_VIDEO_BODY), max_resigns=2, **_POLL_KW
        )
        _assert_sol_booked(client, expected, sync=False)
        assert cost_log == [expected] and tx == [expected]
        await client._client.aclose()
