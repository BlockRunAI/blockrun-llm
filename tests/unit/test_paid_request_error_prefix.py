"""The paid-request error message must not claim money moved when it didn't.

"API error after payment" read as *your funds are gone* on every failure. That
is usually false — gateways settle on success, so the settle call sits after the
upstream work and a failed paid request normally moves nothing.

This is a regression test for a real misdiagnosis: an image-edit 500 was
reported as lost USDC by two readers before anyone checked the gateway's settle
ordering. The wording alone caused it.
"""

import base64
import json

import httpx
import pytest

from blockrun_llm.tx_log import paid_request_error_prefix


def _settlement_header(tx_hash="0xabc123", **extra):
    payload = {"transaction": tx_hash, "network": "base", "success": True, **extra}
    return base64.b64encode(json.dumps(payload).encode()).decode()


class TestUnsettled:
    """No settlement header = the ordinary shape of a failure that cost nothing."""

    def test_no_header_does_not_claim_payment_was_taken(self):
        msg = paid_request_error_prefix(httpx.Headers({}))
        assert "no settlement recorded" in msg
        assert "SETTLED" not in msg
        # The specific phrase that caused the false alarm must be gone.
        assert "after payment" not in msg

    @pytest.mark.parametrize(
        "bad", ["", "!!!not-base64!!!", "e30=", base64.b64encode(b"[]").decode()]
    )
    def test_unparseable_header_degrades_to_unsettled(self, bad):
        """An error path must never raise while reporting an error."""
        msg = paid_request_error_prefix(httpx.Headers({"X-PAYMENT-RESPONSE": bad}))
        assert "no settlement recorded" in msg

    def test_header_without_tx_hash_is_not_a_settlement(self):
        payload = base64.b64encode(json.dumps({"network": "base"}).encode()).decode()
        msg = paid_request_error_prefix(httpx.Headers({"X-PAYMENT-RESPONSE": payload}))
        assert "no settlement recorded" in msg


class TestSettled:
    """A settlement header means funds really did move — say so, and name the tx."""

    def test_reports_settlement_and_tx_hash(self):
        msg = paid_request_error_prefix(
            httpx.Headers({"X-PAYMENT-RESPONSE": _settlement_header("0xdeadbeef")})
        )
        assert "SETTLED" in msg
        assert "0xdeadbeef" in msg, "the tx hash is what makes this actionable"

    def test_solana_signature_field_also_counts(self):
        payload = base64.b64encode(json.dumps({"signature": "5xSolSig"}).encode()).decode()
        msg = paid_request_error_prefix(httpx.Headers({"X-PAYMENT-RESPONSE": payload}))
        assert "SETTLED" in msg and "5xSolSig" in msg


def test_the_two_cases_are_distinguishable():
    """The whole point: they used to be the same string."""
    unsettled = paid_request_error_prefix(httpx.Headers({}))
    settled = paid_request_error_prefix(httpx.Headers({"X-PAYMENT-RESPONSE": _settlement_header()}))
    assert unsettled != settled
