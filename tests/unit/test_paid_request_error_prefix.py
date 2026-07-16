"""The paid-request error message must not claim money moved — or that it didn't.

"API error after payment" read as *your funds are gone* on every failure, which
is usually false. This is a regression test for a real misdiagnosis: an
image-edit 500 was reported as lost USDC by two readers before anyone checked
the gateway's settle ordering. The wording alone caused it.

The second half of the file guards the opposite error, which is worse. Both
gateways send the settlement under the x402 v2 name ``PAYMENT-RESPONSE`` and
neither ever sends ``X-PAYMENT-RESPONSE`` — so reading only the legacy name
decodes nothing in production and reports every settled failure as unsettled.
And on Solana's paid chat path, settle runs in parallel with the upstream call
and the error re-raises before it lands, so the requests that DID charge (the
ones the gateway logs as ``CHARGED BUT REQUEST FAILED — refund manually``) are
exactly the ones arriving with no header. Absence cannot be sold as "you weren't
charged".
"""

import base64
import json

import httpx
import pytest

from blockrun_llm.tx_log import paid_request_error_prefix, read_settlement_header

# What our gateways actually send (x402 v2). The legacy name is still accepted
# for other facilitators, so both are exercised everywhere it matters.
SPEC_NAME = "PAYMENT-RESPONSE"
LEGACY_NAME = "X-PAYMENT-RESPONSE"
BOTH_NAMES = [SPEC_NAME, LEGACY_NAME]


def _settlement_header(tx_hash="0xabc123", **extra):
    payload = {"transaction": tx_hash, "network": "base", "success": True, **extra}
    return base64.b64encode(json.dumps(payload).encode()).decode()


class TestHeaderName:
    """The bug that made the whole mechanism dead code in production."""

    def test_spec_name_is_read(self):
        """Regression: the SDK read only the legacy name, which no gateway sends.

        blockrun and blockrun-sol emit `PAYMENT-RESPONSE` (36 and 25 call sites
        respectively) and `X-PAYMENT-RESPONSE` zero times. The sidecar hit this
        in blockrun-litellm 0.6.0, live-verified against a real paid call.
        """
        msg = paid_request_error_prefix(httpx.Headers({SPEC_NAME: _settlement_header("0xfeed")}))
        assert "SETTLED" in msg and "0xfeed" in msg

    def test_legacy_name_still_accepted(self):
        msg = paid_request_error_prefix(httpx.Headers({LEGACY_NAME: _settlement_header("0xbeef")}))
        assert "SETTLED" in msg and "0xbeef" in msg

    def test_spec_name_wins_when_both_present(self):
        headers = httpx.Headers(
            {SPEC_NAME: _settlement_header("0xspec"), LEGACY_NAME: _settlement_header("0xlegacy")}
        )
        assert "0xspec" in paid_request_error_prefix(headers)

    def test_reader_never_raises_on_a_junk_mapping(self):
        class Hostile:
            def get(self, _name):
                raise RuntimeError("headers exploded")

        assert read_settlement_header(Hostile()) is None


class TestUnsettled:
    """No settlement header means UNKNOWN, and must never be sold as "free"."""

    def test_no_header_does_not_claim_payment_was_taken(self):
        msg = paid_request_error_prefix(httpx.Headers({}))
        assert "SETTLED" not in msg
        # The specific phrase that caused the false alarm must be gone.
        assert "after payment" not in msg

    def test_no_header_does_not_claim_payment_was_NOT_taken(self):
        """The inverse error, and the more expensive one.

        Solana settles in parallel and re-raises before settle lands, so a
        charged-but-failed request carries no header. Promising "payment likely
        not taken" there is a false all-clear on exactly the request that needs
        a manual refund.
        """
        msg = paid_request_error_prefix(httpx.Headers({}))
        assert "likely not taken" not in msg
        assert "check your wallet history" in msg, "must point somewhere authoritative"

    @pytest.mark.parametrize("name", BOTH_NAMES)
    @pytest.mark.parametrize(
        "bad", ["", "!!!not-base64!!!", "e30=", base64.b64encode(b"[]").decode()]
    )
    def test_unparseable_header_degrades_to_unsettled(self, name, bad):
        """An error path must never raise while reporting an error."""
        msg = paid_request_error_prefix(httpx.Headers({name: bad}))
        assert "no settlement reported" in msg

    @pytest.mark.parametrize("name", BOTH_NAMES)
    def test_header_without_tx_hash_is_not_a_settlement(self, name):
        payload = base64.b64encode(json.dumps({"network": "base"}).encode()).decode()
        msg = paid_request_error_prefix(httpx.Headers({name: payload}))
        assert "no settlement reported" in msg

    def test_success_true_without_tx_hash_is_still_unsettled(self):
        """`success` is not a settle signal: the gateways hard-code it to true
        even when settle didn't land, so older clients don't surface an error.
        A tx hash is the only field that means money moved — which is what the
        gateways gate their own revenue accounting on."""
        payload = base64.b64encode(json.dumps({"success": True, "network": "base"}).encode())
        msg = paid_request_error_prefix(httpx.Headers({SPEC_NAME: payload.decode()}))
        assert "SETTLED" not in msg


class TestSettled:
    """A settlement header means funds really did move — say so, and name the tx."""

    def test_reports_settlement_and_tx_hash(self):
        msg = paid_request_error_prefix(
            httpx.Headers({SPEC_NAME: _settlement_header("0xdeadbeef")})
        )
        assert "SETTLED" in msg
        assert "0xdeadbeef" in msg, "the tx hash is what makes this actionable"

    @pytest.mark.parametrize("name", BOTH_NAMES)
    def test_solana_signature_field_also_counts(self, name):
        payload = base64.b64encode(json.dumps({"signature": "5xSolSig"}).encode()).decode()
        msg = paid_request_error_prefix(httpx.Headers({name: payload}))
        assert "SETTLED" in msg and "5xSolSig" in msg


def test_the_two_cases_are_distinguishable():
    """The whole point: they used to be the same string."""
    unsettled = paid_request_error_prefix(httpx.Headers({}))
    settled = paid_request_error_prefix(httpx.Headers({SPEC_NAME: _settlement_header()}))
    assert unsettled != settled
