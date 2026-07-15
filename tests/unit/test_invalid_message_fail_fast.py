"""The client half of the verify retry-storm fix (gateway side: blockrun-sol
``x402-solana.ts`` classifyInvalidMessage).

A payer whose USDC token account was never created fails simulation with
``InvalidAccountData``. The gateway's coarse ``invalidReason`` collapses that to
``transaction_simulation_failed``, which ``_UNRECOVERABLE_PAYMENT_PATTERNS``
deliberately omits (it IS recoverable under concurrent load) — so the SDK burned
all 5 payment attempts on a wallet that could never pay. The gateway now returns
the facilitator's ``invalidMessage`` alongside the enum; these tests pin the SDK
reading it and failing fast.
"""

from __future__ import annotations

from typing import Any

from blockrun_llm.solana_client import (
    _is_unrecoverable_payment_error,
    _is_permanent_payment_error,
)
from blockrun_llm.validation import build_payment_rejected_error


class _FakeResponse:
    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        return self._body


class TestInvalidMessageReachesTheClassifier:
    """build_payment_rejected_error must fold invalidMessage into the message —
    the classifiers only ever see ``str(exc)``."""

    def test_invalid_message_is_surfaced_in_the_error_string(self) -> None:
        exc = build_payment_rejected_error(
            _FakeResponse(
                {
                    "error": "Payment verification failed",
                    "code": "PAYMENT_INVALID",
                    "debug": "transaction_simulation_failed",
                    "invalidMessage": "InvalidAccountData",
                }
            )
        )
        assert "InvalidAccountData" in str(exc)
        assert exc.response is not None
        assert exc.response["invalidMessage"] == "InvalidAccountData"

    def test_absent_invalid_message_leaves_the_message_unchanged(self) -> None:
        exc = build_payment_rejected_error(
            _FakeResponse({"error": "Payment settlement failed", "details": "insufficient_funds"})
        )
        assert "insufficient_funds" in str(exc)

    def test_oversized_invalid_message_is_dropped(self) -> None:
        exc = build_payment_rejected_error(
            _FakeResponse({"error": "Payment verification failed", "invalidMessage": "x" * 500})
        )
        assert exc.response is not None
        assert "invalidMessage" not in exc.response


class TestUnrecoverableClassification:
    def test_invalid_account_data_is_unrecoverable(self) -> None:
        """An unfunded wallet: no fresh nonce/blockhash can make this pass."""
        assert (
            _is_unrecoverable_payment_error(
                "Payment rejected by gateway: transaction_simulation_failed (InvalidAccountData)"
            )
            is True
        )

    def test_spelling_variants_all_classify(self) -> None:
        for msg in (
            "invalid account data",
            "invalid_account_data",
            "InvalidAccountData",
            "AccountNotFound",
            "Error processing Instruction 0: invalid account data",
        ):
            assert _is_unrecoverable_payment_error(
                f"Payment rejected by gateway: transaction_simulation_failed ({msg})"
            ), msg

    def test_bare_simulation_failure_stays_recoverable(self) -> None:
        """Without an invalidMessage we know nothing more than before — keep the
        whole-request retry that exists to ride out concurrent-load failures."""
        assert (
            _is_unrecoverable_payment_error(
                "Payment rejected by gateway: transaction_simulation_failed"
            )
            is False
        )

    def test_blockhash_stays_recoverable_on_the_client(self) -> None:
        """Deliberate asymmetry with the gateway: it stops retrying the SAME dead
        header, but re-signing with a FRESH blockhash is exactly what fixes this,
        and re-signing is what the SDK's whole-request retry does."""
        for msg in ("BlockhashNotFound", "BlockHeightExceeded"):
            assert (
                _is_unrecoverable_payment_error(
                    f"Payment rejected by gateway: transaction_simulation_failed ({msg})"
                )
                is False
            ), msg

    def test_transient_errors_still_retry(self) -> None:
        assert _is_unrecoverable_payment_error("503 Service Unavailable") is False
        assert _is_unrecoverable_payment_error("") is False


class TestPermanentClassifierUnaffected:
    """_is_permanent_payment_error governs the *fallback-model* decision and
    already treats simulation/blockhash as permanent. The new patterns must not
    perturb it."""

    def test_still_permanent(self) -> None:
        assert _is_permanent_payment_error("transaction_simulation_failed") is True
        assert (
            _is_permanent_payment_error(
                "Payment rejected by gateway: transaction_simulation_failed (InvalidAccountData)"
            )
            is True
        )

    def test_still_transient(self) -> None:
        assert _is_permanent_payment_error("503 Service Unavailable") is False
