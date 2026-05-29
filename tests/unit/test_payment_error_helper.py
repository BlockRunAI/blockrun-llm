"""Tests for the 402-retry payment-rejected helper.

These cover the regression where a Solana settlement failure
(``transaction_simulation_failed``, ``insufficient_funds``, ...) was
swallowed by a generic ``"Payment rejected. Check your Solana USDC
balance."`` message, leaving customers no way to diagnose.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from blockrun_llm.types import PaymentError
from blockrun_llm.validation import build_payment_rejected_error


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response`` that ``.json()``."""

    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class TestPaymentErrorEnrichment:
    def test_payment_error_carries_status_and_response(self) -> None:
        """The new kwargs are public API — callers and proxies use them."""
        exc = PaymentError(
            "Payment rejected by gateway: transaction_simulation_failed",
            status_code=402,
            response={"message": "Payment settlement failed", "details": "transaction_simulation_failed"},
        )
        assert exc.status_code == 402
        assert exc.response is not None
        assert exc.response["details"] == "transaction_simulation_failed"
        assert "transaction_simulation_failed" in str(exc)

    def test_payment_error_backwards_compatible_no_kwargs(self) -> None:
        """Pre-0.32.0 callers raise ``PaymentError("...")`` — still works."""
        exc = PaymentError("Payment rejected")
        assert exc.status_code is None
        assert exc.response is None
        assert str(exc) == "Payment rejected"


class TestBuildPaymentRejectedError:
    def test_preserves_gateway_details(self) -> None:
        """The whole reason this helper exists: ``details`` must survive
        from the gateway's body to ``exc.response`` and into ``str(exc)``."""
        gateway_body: Dict[str, Any] = {
            "error": "Payment settlement failed",
            "details": "transaction_simulation_failed",
        }
        exc = build_payment_rejected_error(_FakeResponse(gateway_body))

        assert isinstance(exc, PaymentError)
        assert exc.status_code == 402
        assert exc.response is not None
        assert exc.response["details"] == "transaction_simulation_failed"
        # The message should mention the real reason, not a generic line.
        assert "transaction_simulation_failed" in str(exc)
        assert "Check your" not in str(exc)  # generic fallback should be gone

    def test_truncates_overly_long_details(self) -> None:
        """Defensive: if a future server bug stuffs free-form text into
        ``details`` we don't want to leak unbounded payloads."""
        huge = "x" * 1024
        exc = build_payment_rejected_error(
            _FakeResponse({"error": "Payment settlement failed", "details": huge})
        )
        # details > 256 chars is rejected — falls back to sanitized message
        assert exc.response is not None
        assert "details" not in exc.response
        assert "Payment settlement failed" in str(exc)

    def test_handles_non_string_details(self) -> None:
        """If the gateway sends a non-string ``details`` (list, dict, None),
        we drop it rather than crashing — message uses the fallback."""
        exc = build_payment_rejected_error(
            _FakeResponse({"error": "Payment settlement failed", "details": ["a", "b"]})
        )
        assert exc.response is not None
        assert "details" not in exc.response
        assert "Payment settlement failed" in str(exc)

    def test_handles_unparseable_body(self) -> None:
        """Gateway returned HTML / empty / malformed JSON — we still
        raise a usable PaymentError (status_code=402, generic message)."""
        exc = build_payment_rejected_error(_FakeResponse(ValueError("not json")))
        assert isinstance(exc, PaymentError)
        assert exc.status_code == 402
        # No raise — falls through to generic "Payment rejected by gateway"
        assert str(exc).startswith("Payment rejected")

    def test_handles_non_dict_body(self) -> None:
        """Gateway returned a JSON array or string — treat as empty dict."""
        exc = build_payment_rejected_error(_FakeResponse(["nope"]))
        assert isinstance(exc, PaymentError)
        assert exc.status_code == 402
        assert exc.response == {"code": None, "message": "API request failed"}
