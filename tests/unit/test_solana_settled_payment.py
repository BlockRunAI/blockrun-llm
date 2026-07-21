"""Solana half of the settled-payment guard.

Signing is settlement on either chain. The Base client learned not to let the
fallback chain buy a retry after a payment went out; this pins the same rule for
Solana, where the transfer is SPL USDC.

Guarded with importorskip: the 3.9 CI job installs the SDK without the solana
extra, and an unguarded Solana test file turns that job red (see #19/#20).
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("x402")
pytest.importorskip("solders")

from blockrun_llm.client import _mark_settled  # noqa: E402
from blockrun_llm.solana_client import _should_fallback_solana  # noqa: E402
from blockrun_llm.types import APIError, PaymentError  # noqa: E402


class TestSolanaSettledTag:
    """Mirrors TestSettledTagClassification in test_settled_payment.py."""

    def test_untagged_timeout_still_falls_back(self):
        assert _should_fallback_solana(httpx.ReadTimeout("boom")) is True

    def test_untagged_503_still_falls_back(self):
        assert _should_fallback_solana(APIError("upstream", 503, None)) is True

    def test_settled_timeout_does_not_fall_back(self):
        assert _should_fallback_solana(_mark_settled(httpx.ReadTimeout("boom"))) is False

    def test_settled_network_error_does_not_fall_back(self):
        assert _should_fallback_solana(_mark_settled(httpx.ConnectError("boom"))) is False

    def test_settled_5xx_does_not_fall_back(self):
        """The dominant post-settlement failure, and the one the Base fix
        originally missed."""
        assert _should_fallback_solana(_mark_settled(APIError("upstream", 503, None))) is False

    def test_payment_error_still_refused(self):
        """Pre-existing behavior must survive the new first check."""
        assert _should_fallback_solana(PaymentError("insufficient balance")) is False

    def test_permanent_payment_reason_still_refused(self):
        """The issue #6 guard: a transient type carrying a permanent reason."""
        assert _should_fallback_solana(httpx.ReadTimeout("transaction_simulation_failed")) is False

    def test_both_chains_agree_on_the_tag(self):
        """The tag has to mean the same thing in both fallback chains, or one
        of them keeps paying twice."""
        from blockrun_llm.client import _should_fallback

        for exc in (
            httpx.ReadTimeout("boom"),
            httpx.ConnectError("boom"),
            APIError("upstream", 503, None),
        ):
            assert _should_fallback(exc) is True
            assert _should_fallback_solana(exc) is True
            assert _should_fallback(_mark_settled(exc)) is False
            assert _should_fallback_solana(_mark_settled(exc)) is False
