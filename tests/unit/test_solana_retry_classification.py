"""Tests for the permanent-vs-transient classification on Solana paths.

Covers issue #6: ``transaction_simulation_failed`` (and a couple of close
cousins) must be classified as PERMANENT so the SDK does not waste
5+ minutes of wall-clock re-running 30-180s upstream generations only
to hit the same wall. Even when the exception type is "transient"
(httpx.Timeout / NetworkError), if the underlying reason text matches a
permanent classification, no fallback.
"""

from __future__ import annotations

import httpx
import pytest

from blockrun_llm.solana_client import (
    _is_permanent_payment_error,
    _should_fallback_solana,
)
from blockrun_llm.types import APIError, PaymentError


class TestIsPermanentPaymentError:
    @pytest.mark.parametrize(
        "reason",
        [
            "transaction_simulation_failed",
            "invalid_exact_svm_payload_transaction_simulation_failed",  # CDP long form
            "blockhash not found",
            "block height exceeded",
            "insufficient funds",
            "insufficient balance for tx fee",
            "invalid signature on payload",
            "invalid_payload: amount mismatch",
            "payment_expired after 300s",
            "authorization is used (replay)",
            "TRANSACTION_SIMULATION_FAILED",  # case-insensitive
        ],
    )
    def test_known_permanent_reasons(self, reason: str) -> None:
        assert _is_permanent_payment_error(reason) is True

    @pytest.mark.parametrize(
        "reason",
        [
            "",  # empty
            "503 Service Unavailable",
            "Connection reset by peer",
            "Read timeout after 60s",
            "facilitator internal error",
            "rate limit exceeded",
        ],
    )
    def test_transient_reasons_pass_through(self, reason: str) -> None:
        assert _is_permanent_payment_error(reason) is False


class TestShouldFallbackSolana:
    """``fallback_models`` decision matches Base semantics + permanent guard."""

    def test_payment_error_never_falls_back(self) -> None:
        # PaymentError always propagates — fallback would re-run on a new model
        # but the payment is wallet-side, not provider-side.
        exc = PaymentError(
            "Payment rejected by gateway: transaction_simulation_failed",
            status_code=402,
            response={"details": "transaction_simulation_failed"},
        )
        assert _should_fallback_solana(exc) is False

    def test_timeout_falls_back_for_transient_reason(self) -> None:
        exc = httpx.ReadTimeout("upstream took too long")
        assert _should_fallback_solana(exc) is True

    def test_timeout_does_NOT_fall_back_when_reason_is_permanent(self) -> None:
        """Defensive guard from issue #6: even a transient exception type
        must not trigger fallback if the wrapped reason is a permanent
        payment classification."""
        exc = httpx.ReadTimeout("transaction_simulation_failed during settle")
        assert _should_fallback_solana(exc) is False

    def test_network_error_falls_back(self) -> None:
        exc = httpx.NetworkError("connection reset")
        assert _should_fallback_solana(exc) is True

    def test_network_error_with_permanent_reason_does_not(self) -> None:
        exc = httpx.NetworkError("blockhash not found in cache")
        assert _should_fallback_solana(exc) is False

    @pytest.mark.parametrize("status", [502, 503, 504, 522, 524])
    def test_5xx_api_error_falls_back(self, status: int) -> None:
        exc = APIError("upstream sick", status_code=status, response=None)
        assert _should_fallback_solana(exc) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_api_error_does_not_fall_back(self, status: int) -> None:
        exc = APIError("client error", status_code=status, response=None)
        assert _should_fallback_solana(exc) is False


class TestPerMethodTimeoutConstants:
    """v0.34.0 introduces per-use-case defaults; pin the values."""

    def test_constants_have_workload_appropriate_values(self) -> None:
        from blockrun_llm import solana_client as mod

        # Chat: long enough for streaming opus + 8k tokens
        assert mod.DEFAULT_CHAT_TIMEOUT >= 120.0
        # Image: covers gpt-image-2 at 1536px (~180s server-side)
        assert mod.DEFAULT_IMAGE_TIMEOUT >= 180.0
        # Search: Grok Live Search with deep web/X tool-use
        assert mod.DEFAULT_SEARCH_TIMEOUT >= 180.0
        # Fast lookups: pyth / x_user_info return in ~1-2s
        assert mod.DEFAULT_FAST_TIMEOUT <= 60.0
        # Backwards compatibility: flat DEFAULT_TIMEOUT must be ≥ chat
        assert mod.DEFAULT_TIMEOUT >= mod.DEFAULT_CHAT_TIMEOUT

    def test_default_timeout_no_longer_60s(self) -> None:
        """The historical 60s default truncated long chats and slow images.
        v0.34.0 raises it to chat-grade so legacy callers stop dying at 60s."""
        from blockrun_llm import solana_client as mod

        assert mod.DEFAULT_TIMEOUT != 60.0
        assert mod.DEFAULT_TIMEOUT > 60.0
