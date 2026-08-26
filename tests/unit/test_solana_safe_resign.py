"""Safety contract for Solana paid-leg re-sign retries."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from blockrun_llm.solana_client import (
    AsyncSolanaLLMClient,
    SolanaLLMClient,
    _is_safe_stale_resign_error,
)
from blockrun_llm.types import PaymentError


def payment_error(body: dict[str, object]) -> PaymentError:
    return PaymentError("payment rejected", status_code=402, response=body)


@pytest.mark.parametrize(
    "body",
    [
        {
            "code": "PAYMENT_BLOCKHASH_STALE",
            "reason": "blockhash_expired",
            "message": "Payment verification failed",
        },
        {
            "code": "PAYMENT_INVALID",
            "reason": "expired_signature",
            "message": "Payment verification failed",
        },
        {
            "code": "PAYMENT_INVALID",
            "message": "Payment verification failed",
            "invalidMessage": "BlockhashNotFound",
        },
        {"message": "Payment verification failed: expired_signature"},
    ],
)
def test_explicit_verification_stale_is_safe_to_resign(body: dict[str, object]) -> None:
    assert _is_safe_stale_resign_error(payment_error(body)) is True


@pytest.mark.parametrize(
    "body",
    [
        {"message": "transaction_simulation_failed"},
        {"code": "PAYMENT_BLOCKHASH_STALE", "reason": "blockhash_expired"},
        {"invalidMessage": "BlockhashNotFound"},
        {"code": "PAYMENT_INVALID", "reason": "insufficient_funds"},
        {
            "code": "PAYMENT_BLOCKHASH_STALE",
            "message": "Payment settlement failed: BlockhashNotFound",
            "invalidMessage": "BlockhashNotFound",
        },
        {
            "code": "SETTLEMENT_FAILED",
            "reason": "expired_signature",
            "message": "Payment settlement failed",
        },
    ],
)
def test_ambiguous_terminal_or_settlement_failure_is_never_resigned(
    body: dict[str, object],
) -> None:
    assert _is_safe_stale_resign_error(payment_error(body)) is False


def test_sync_request_retries_with_a_fresh_negotiation_only_for_safe_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(SolanaLLMClient)
    client._PAYMENT_RETRY_BACKOFFS = (0.0, 0.0)
    stale = payment_error(
        {
            "code": "PAYMENT_INVALID",
            "reason": "expired_signature",
            "message": "Payment verification failed",
        }
    )
    request_once = Mock(side_effect=[stale, "ok"])
    monkeypatch.setattr(client, "_request_once", request_once)

    assert client._request_with_payment("/v1/chat/completions", {}) == "ok"
    assert request_once.call_count == 2


def test_sync_request_does_not_replay_a_settlement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(SolanaLLMClient)
    client._PAYMENT_RETRY_BACKOFFS = (0.0, 0.0)
    settlement = payment_error(
        {
            "code": "SETTLEMENT_FAILED",
            "reason": "expired_signature",
            "message": "Payment settlement failed",
        }
    )
    request_once = Mock(side_effect=settlement)
    monkeypatch.setattr(client, "_request_once", request_once)

    with pytest.raises(PaymentError):
        client._request_with_payment("/v1/chat/completions", {})
    assert request_once.call_count == 1


def test_sync_raw_post_uses_the_same_safe_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(SolanaLLMClient)
    client._PAYMENT_RETRY_BACKOFFS = (0.0, 0.0)
    stale = payment_error(
        {
            "code": "PAYMENT_INVALID",
            "reason": "expired_signature",
            "message": "Payment verification failed",
        }
    )
    request_once = Mock(side_effect=[stale, {"ok": True}])
    monkeypatch.setattr(client, "_request_with_payment_raw_once", request_once)

    assert client._request_with_payment_raw("/v1/search", {}) == {"ok": True}
    assert request_once.call_count == 2


@pytest.mark.asyncio
async def test_async_request_uses_the_same_safe_phase_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(AsyncSolanaLLMClient)
    client._PAYMENT_RETRY_BACKOFFS = (0.0, 0.0)
    stale = payment_error(
        {
            "code": "PAYMENT_INVALID",
            "reason": "expired_signature",
            "message": "Payment verification failed",
        }
    )
    request_once = AsyncMock(side_effect=[stale, "ok"])
    monkeypatch.setattr(client, "_request_once", request_once)

    assert await client._request_with_payment("/v1/chat/completions", {}) == "ok"
    assert request_once.await_count == 2


@pytest.mark.asyncio
async def test_async_raw_get_does_not_replay_a_settlement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(AsyncSolanaLLMClient)
    client._PAYMENT_RETRY_BACKOFFS = (0.0, 0.0)
    settlement = payment_error(
        {
            "code": "SETTLEMENT_FAILED",
            "reason": "expired_signature",
            "message": "Payment settlement failed",
        }
    )
    request_once = AsyncMock(side_effect=settlement)
    monkeypatch.setattr(client, "_get_with_payment_raw_once", request_once)

    with pytest.raises(PaymentError):
        await client._get_with_payment_raw("/v1/rpc/solana")
    assert request_once.await_count == 1
