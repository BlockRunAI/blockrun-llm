"""Safety contract for Solana paid-leg re-sign retries.

The line between safe and unsafe is the payment PHASE, not the cause: a
pre-broadcast rejection never settled and is free to re-sign, a settlement
failure may already have paid. These tests pin both directions, and they build
the errors the way production does — through
:func:`blockrun_llm.validation.build_payment_rejected_error` from the literal
402 bodies blockrun-sol emits — so a gateway wording change fails here rather
than silently disabling the retry.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from blockrun_llm.solana_client import (
    AsyncSolanaLLMClient,
    SolanaLLMClient,
    _is_safe_resign_error,
    _normalize_reason,
)
from blockrun_llm.types import PaymentError
from blockrun_llm.validation import build_payment_rejected_error


class _FakeResponse:
    """Minimal stand-in for the httpx.Response that build_* consumes."""

    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        return self._body


def gateway_error(body: dict[str, object]) -> PaymentError:
    """Build a PaymentError exactly as the paid legs do, from a raw 402 body."""
    return build_payment_rejected_error(_FakeResponse(body))


def payment_error(body: dict[str, object]) -> PaymentError:
    return PaymentError("payment rejected", status_code=402, response=body)


# --- Literal gateway bodies --------------------------------------------------
#
# Family A — /v1/chat/completions: carries `code` and a `message`.
# Family B — the ~16 other paid routes: `error` + `reason` only, NO code,
#            NO message. These are the routes the raw POST/GET wrappers serve.

CHAT_VERIFY_EXPIRED = {
    "error": "Payment verification failed",
    "message": "Message @bc1max on Telegram for help.",
    "code": "PAYMENT_INVALID",
    "reason": "expired_signature",
}
CHAT_VERIFY_UNAVAILABLE = {
    "error": "Payment verification failed",
    "message": "Message @bc1max on Telegram for help.",
    "code": "PAYMENT_INVALID",
    "reason": "verification_unavailable",
}
CHAT_VERIFY_CATCHALL = {
    "error": "Payment verification failed",
    "message": "Message @bc1max on Telegram for help.",
    "code": "PAYMENT_INVALID",
    "reason": "verification_failed",
}
CHAT_REPLAY = {
    "error": "Payment authorization already used",
    "message": "This payment signature was already redeemed. Sign a new payment for each request.",
    "code": "PAYMENT_REPLAY",
}
CHAT_UNDERPAID = {
    "error": "Payment below quoted price",
    "message": (
        "The signed payment is less than the quoted price for this request. "
        "Re-fetch the 402 quote and sign the amount it specifies."
    ),
    "code": "PAYMENT_UNDERPAID",
}
CHAT_SETTLE = {
    "error": "Payment settlement failed",
    "message": "Message @bc1max on Telegram for help.",
    "code": "SETTLEMENT_FAILED",
    "reason": "expired_signature",
}

RAW_VERIFY_EXPIRED = {"error": "Payment verification failed", "reason": "expired_signature"}
RAW_VERIFY_UNAVAILABLE = {
    "error": "Payment verification failed",
    "reason": "verification_unavailable",
}
RAW_VERIFY_CATCHALL = {"error": "Payment verification failed", "reason": "verification_failed"}
RAW_VERIFY_NO_FUNDS = {"error": "Payment verification failed", "reason": "insufficient_funds"}
RAW_SETTLE_EXPIRED = {"error": "Payment settlement failed", "reason": "expired_signature"}
RAW_SETTLE_CATCHALL = {"error": "Payment settlement failed", "reason": "settlement_failed"}

# The Anthropic-compatible route folds the phase and the reason into one string.
MESSAGES_VERIFY_EXPIRED = {"error": "Payment verification failed: expired_signature"}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(CHAT_VERIFY_EXPIRED, id="chat-expired-signature"),
        pytest.param(CHAT_VERIFY_UNAVAILABLE, id="chat-verifier-outage"),
        pytest.param(CHAT_VERIFY_CATCHALL, id="chat-verify-catchall"),
        pytest.param(CHAT_REPLAY, id="chat-replay-nonce"),
        pytest.param(CHAT_UNDERPAID, id="chat-underpaid"),
        pytest.param(RAW_VERIFY_EXPIRED, id="raw-expired-signature"),
        pytest.param(RAW_VERIFY_UNAVAILABLE, id="raw-verifier-outage"),
        pytest.param(RAW_VERIFY_CATCHALL, id="raw-verify-catchall"),
        pytest.param(MESSAGES_VERIFY_EXPIRED, id="messages-folded-reason"),
    ],
)
def test_pre_broadcast_rejections_are_safe_to_resign(body: dict[str, object]) -> None:
    """Nothing was broadcast, so a fresh signature costs the payer nothing."""
    assert _is_safe_resign_error(gateway_error(body)) is True


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(CHAT_SETTLE, id="chat-settlement-failed"),
        pytest.param(RAW_SETTLE_EXPIRED, id="raw-settle-expired-signature"),
        pytest.param(RAW_SETTLE_CATCHALL, id="raw-settle-catchall"),
        pytest.param(RAW_VERIFY_NO_FUNDS, id="raw-insufficient-funds"),
        pytest.param({"message": "transaction_simulation_failed"}, id="bare-simulation-failure"),
        pytest.param(
            {"error": "Payment verification failed", "invalidMessage": "InvalidAccountData"},
            id="no-usdc-token-account",
        ),
        pytest.param({"error": "Some unrelated gateway error"}, id="unknown-title"),
    ],
)
def test_settlement_and_terminal_failures_are_never_resigned(body: dict[str, object]) -> None:
    assert _is_safe_resign_error(gateway_error(body)) is False


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {"error": "Payment verification failed", "reason": "settlement_failed"},
            id="reason-contradicts-title",
        ),
        pytest.param(
            {"error": "Payment verification failed", "code": "SETTLEMENT_FAILED"},
            id="code-contradicts-title",
        ),
        pytest.param({"reason": "settlement_failed"}, id="reason-only"),
    ],
)
def test_any_settlement_marker_wins_over_a_verify_title(body: dict[str, object]) -> None:
    """The phase gate is checked first and on all three fields, so a body whose
    title says verify but whose code/reason says settlement stays terminal. A
    settle rejection must never be re-signed on the strength of one field."""
    assert _is_safe_resign_error(gateway_error(body)) is False


def test_phase_titles_match_by_prefix_not_substring() -> None:
    """`_normalize_reason` strips separators, so a substring test would straddle
    word boundaries. A verify failure whose prose merely mentions settlement
    must still be recognized as verify phase."""
    body = {
        "error": "Payment verification failed",
        "message": (
            "Payment verification failed: upstream reported that a prior "
            "payment settlement failed and was retried"
        ),
        "code": "PAYMENT_INVALID",
    }
    normalized_message = _normalize_reason(str(body["message"]))
    # The settlement title really is present mid-string — a substring test here
    # would misread a verify-phase rejection as a broadcast and refuse to retry.
    assert "paymentsettlementfailed" in normalized_message
    assert not normalized_message.startswith("paymentsettlementfailed")
    assert _is_safe_resign_error(payment_error(body)) is True


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(PaymentError("402 response but no payment requirements found"), id="no-body"),
        pytest.param(PaymentError("x", status_code=402, response=None), id="none-response"),
        pytest.param(PaymentError("x", status_code=402, response="not-a-dict"), id="str-response"),
    ],
)
def test_missing_or_malformed_response_is_terminal(exc: PaymentError) -> None:
    """Silence is never treated as permission to re-sign."""
    assert _is_safe_resign_error(exc) is False


# --- Retry wiring: all eight call sites --------------------------------------


def _sync_client() -> SolanaLLMClient:
    client = object.__new__(SolanaLLMClient)
    client._PAYMENT_RETRY_BACKOFFS = (0.0, 0.0, 0.0, 0.0)  # type: ignore[misc]
    return client


def _async_client() -> AsyncSolanaLLMClient:
    client = object.__new__(AsyncSolanaLLMClient)
    client._PAYMENT_RETRY_BACKOFFS = (0.0, 0.0, 0.0, 0.0)  # type: ignore[misc]
    return client


SYNC_SITES = [
    ("_request_once", "_request_with_payment", ("/v1/chat/completions", {}), {}, "ok"),
    (
        "_request_with_payment_raw_once",
        "_request_with_payment_raw",
        ("/v1/search", {}),
        {},
        {"ok": True},
    ),
    ("_get_with_payment_raw_once", "_get_with_payment_raw", ("/v1/pm/markets",), {}, {"ok": True}),
]

ASYNC_SITES = [
    ("_request_once", "_request_with_payment", ("/v1/chat/completions", {}), {}, "ok"),
    (
        "_request_with_payment_raw_once",
        "_request_with_payment_raw",
        ("/v1/search", {}),
        {},
        {"ok": True},
    ),
    ("_get_with_payment_raw_once", "_get_with_payment_raw", ("/v1/rpc/solana",), {}, {"ok": True}),
]


@pytest.mark.parametrize("once_name,wrapper_name,args,kwargs,result", SYNC_SITES)
def test_sync_sites_retry_a_pre_broadcast_rejection(
    monkeypatch: pytest.MonkeyPatch,
    once_name: str,
    wrapper_name: str,
    args: tuple,
    kwargs: dict,
    result: object,
) -> None:
    client = _sync_client()
    once = Mock(side_effect=[gateway_error(RAW_VERIFY_EXPIRED), result])
    monkeypatch.setattr(client, once_name, once)
    assert getattr(client, wrapper_name)(*args, **kwargs) == result
    assert once.call_count == 2


@pytest.mark.parametrize("once_name,wrapper_name,args,kwargs,result", SYNC_SITES)
def test_sync_sites_never_replay_a_settlement_failure(
    monkeypatch: pytest.MonkeyPatch,
    once_name: str,
    wrapper_name: str,
    args: tuple,
    kwargs: dict,
    result: object,
) -> None:
    client = _sync_client()
    once = Mock(side_effect=gateway_error(RAW_SETTLE_EXPIRED))
    monkeypatch.setattr(client, once_name, once)
    with pytest.raises(PaymentError):
        getattr(client, wrapper_name)(*args, **kwargs)
    assert once.call_count == 1


@pytest.mark.parametrize("once_name,wrapper_name,args,kwargs,result", SYNC_SITES)
def test_sync_sites_bound_the_retry_and_raise_payment_error(
    monkeypatch: pytest.MonkeyPatch,
    once_name: str,
    wrapper_name: str,
    args: tuple,
    kwargs: dict,
    result: object,
) -> None:
    """The loop must stop at _MAX_PAYMENT_RETRIES + 1 attempts and surface the
    gateway's PaymentError, not the loop-exhausted guard."""
    client = _sync_client()
    once = Mock(side_effect=[gateway_error(RAW_VERIFY_EXPIRED) for _ in range(20)])
    monkeypatch.setattr(client, once_name, once)
    with pytest.raises(PaymentError, match="Payment rejected by gateway"):
        getattr(client, wrapper_name)(*args, **kwargs)
    assert once.call_count == SolanaLLMClient._MAX_PAYMENT_RETRIES + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("once_name,wrapper_name,args,kwargs,result", ASYNC_SITES)
async def test_async_sites_retry_a_pre_broadcast_rejection(
    monkeypatch: pytest.MonkeyPatch,
    once_name: str,
    wrapper_name: str,
    args: tuple,
    kwargs: dict,
    result: object,
) -> None:
    client = _async_client()
    once = AsyncMock(side_effect=[gateway_error(RAW_VERIFY_EXPIRED), result])
    monkeypatch.setattr(client, once_name, once)
    assert await getattr(client, wrapper_name)(*args, **kwargs) == result
    assert once.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("once_name,wrapper_name,args,kwargs,result", ASYNC_SITES)
async def test_async_sites_never_replay_a_settlement_failure(
    monkeypatch: pytest.MonkeyPatch,
    once_name: str,
    wrapper_name: str,
    args: tuple,
    kwargs: dict,
    result: object,
) -> None:
    client = _async_client()
    once = AsyncMock(side_effect=gateway_error(RAW_SETTLE_EXPIRED))
    monkeypatch.setattr(client, once_name, once)
    with pytest.raises(PaymentError):
        await getattr(client, wrapper_name)(*args, **kwargs)
    assert once.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("once_name,wrapper_name,args,kwargs,result", ASYNC_SITES)
async def test_async_sites_bound_the_retry(
    monkeypatch: pytest.MonkeyPatch,
    once_name: str,
    wrapper_name: str,
    args: tuple,
    kwargs: dict,
    result: object,
) -> None:
    client = _async_client()
    once = AsyncMock(side_effect=[gateway_error(RAW_VERIFY_EXPIRED) for _ in range(20)])
    monkeypatch.setattr(client, once_name, once)
    with pytest.raises(PaymentError, match="Payment rejected by gateway"):
        await getattr(client, wrapper_name)(*args, **kwargs)
    assert once.await_count == AsyncSolanaLLMClient._MAX_PAYMENT_RETRIES + 1


# --- Streaming: output is never replayed -------------------------------------


def test_sync_stream_does_not_resign_once_a_chunk_was_yielded() -> None:
    """The paid leg already delivered output; re-signing would bill twice for
    one answer even though the rejection itself is pre-broadcast."""
    client = _sync_client()
    calls = {"n": 0}

    def once(endpoint: str, body: dict, timeout: float | None = None):
        calls["n"] += 1
        yield {"chunk": calls["n"]}
        raise gateway_error(RAW_VERIFY_EXPIRED)

    client._stream_once = once  # type: ignore[assignment]
    stream = client._stream_with_payment("/v1/chat/completions", {})
    assert next(stream) == {"chunk": 1}
    with pytest.raises(PaymentError):
        next(stream)
    assert calls["n"] == 1


def test_sync_stream_resigns_when_nothing_was_yielded() -> None:
    client = _sync_client()
    calls = {"n": 0}

    def once(endpoint: str, body: dict, timeout: float | None = None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gateway_error(RAW_VERIFY_EXPIRED)
        yield {"chunk": "ok"}

    client._stream_once = once  # type: ignore[assignment]
    assert list(client._stream_with_payment("/v1/chat/completions", {})) == [{"chunk": "ok"}]
    assert calls["n"] == 2


def test_sync_stream_never_replays_a_settlement_failure() -> None:
    client = _sync_client()
    calls = {"n": 0}

    def once(endpoint: str, body: dict, timeout: float | None = None):
        calls["n"] += 1
        raise gateway_error(RAW_SETTLE_EXPIRED)
        yield  # pragma: no cover - makes `once` a generator

    client._stream_once = once  # type: ignore[assignment]
    with pytest.raises(PaymentError):
        list(client._stream_with_payment("/v1/chat/completions", {}))
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_async_stream_does_not_resign_once_a_chunk_was_yielded() -> None:
    client = _async_client()
    calls = {"n": 0}

    async def once(endpoint: str, body: dict, timeout: float | None = None):
        calls["n"] += 1
        yield {"chunk": calls["n"]}
        raise gateway_error(RAW_VERIFY_EXPIRED)

    client._stream_once = once  # type: ignore[assignment]
    stream = client._stream_with_payment("/v1/chat/completions", {})
    assert await stream.__anext__() == {"chunk": 1}
    with pytest.raises(PaymentError):
        await stream.__anext__()
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_async_stream_resigns_when_nothing_was_yielded() -> None:
    client = _async_client()
    calls = {"n": 0}

    async def once(endpoint: str, body: dict, timeout: float | None = None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gateway_error(RAW_VERIFY_EXPIRED)
        yield {"chunk": "ok"}

    client._stream_once = once  # type: ignore[assignment]
    seen = [c async for c in client._stream_with_payment("/v1/chat/completions", {})]
    assert seen == [{"chunk": "ok"}]
    assert calls["n"] == 2


# --- Backoff table -----------------------------------------------------------


def test_real_backoff_table_is_used_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises the shipped tuple and the index clamp, which the zeroed
    per-test override otherwise hides."""
    import time as _time

    client = object.__new__(SolanaLLMClient)
    slept: list[float] = []
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))
    once = Mock(side_effect=[gateway_error(RAW_VERIFY_EXPIRED) for _ in range(20)])
    monkeypatch.setattr(client, "_request_once", once)
    with pytest.raises(PaymentError):
        client._request_with_payment("/v1/chat/completions", {})
    assert slept == list(SolanaLLMClient._PAYMENT_RETRY_BACKOFFS)
