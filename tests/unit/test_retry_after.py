"""``Retry-After`` has to survive the SDK boundary, on every client.

The gateway sets the header deliberately — it is what turns a refused request
into a caller that waits instead of one that spins against the limit. It
matters most on the account rail, where limits are per key and a 429 is the
normal way a busy customer is asked to slow down.

The table below is the point of this file: one 429 fixture replayed through
every public service method, asserting the header arrives on the raised error.
A raise site that forgets it fails here rather than in a customer's retry loop.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from blockrun_llm import (
    ImageClient,
    LLMClient,
    MusicClient,
    PhoneClient,
    PortraitClient,
    PriceClient,
    RealFaceClient,
    RpcClient,
    SearchClient,
    SpeechClient,
    SurfClient,
    VideoClient,
    VoiceClient,
)
from blockrun_llm.types import APIError, retry_after_of

KEY = "brk_live_retry_after_fixture"
PHONE = "+12025550123"
WALLET = "0x" + "01" * 20
RETRY_AFTER = "17"

# (class, method, args, kwargs) — every public entry point that can surface a 429.
CASES = [
    (LLMClient, "chat", ("openai/gpt-5.2", "hi"), {}),
    (LLMClient, "list_models", (), {}),
    (ImageClient, "generate", ("test",), {}),
    (VideoClient, "generate", ("test",), {"duration_seconds": 5}),
    (MusicClient, "generate", ("test music",), {}),
    (SpeechClient, "generate", ("hello",), {}),
    (SpeechClient, "sound_effect", ("rain",), {}),
    (SpeechClient, "list_voices", (), {}),
    (VoiceClient, "call", (PHONE, "Read a test message"), {}),
    (VoiceClient, "get_status", ("fixture-call",), {}),
    (PhoneClient, "lookup", (PHONE,), {}),
    (PhoneClient, "lookup_fraud", (PHONE,), {}),
    (PhoneClient, "buy_number", (), {"area_code": "202"}),
    (PhoneClient, "renew_number", (PHONE,), {}),
    (PhoneClient, "list_numbers", (), {}),
    (PhoneClient, "release_number", (PHONE,), {}),
    (PortraitClient, "enroll", ("fixture", "https://example.com/test.png"), {}),
    (PortraitClient, "list_portraits", (WALLET,), {}),
    (RealFaceClient, "init", ("fixture",), {}),
    (RealFaceClient, "status", ("legacy_rf_123",), {}),
    (RealFaceClient, "enroll", ("fixture", "https://example.com/t.png", "legacy_rf_123"), {}),
    (RealFaceClient, "list_realfaces", (WALLET,), {}),
    (SearchClient, "search", ("test",), {}),
    (SurfClient, "get", ("market/ranking",), {}),
    (SurfClient, "post", ("onchain/sql", {"query": "SELECT 1"}), {}),
    (PriceClient, "price", ("crypto", "BTC-USD"), {}),
    (PriceClient, "price", ("stocks", "AAPL"), {"market": "us"}),
    (PriceClient, "history", ("crypto", "BTC-USD"), {"from_ts": 1, "to_ts": 2}),
    (RpcClient, "call", ("solana", "getSlot"), {}),
    (RpcClient, "batch", ("base", [{"method": "eth_blockNumber"}]), {}),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0].__name__}.{c[1]}" for c in CASES])
def test_a_429_carries_retry_after_to_the_caller(case, monkeypatch):
    cls, method, args, kwargs = case
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    monkeypatch.setenv("BLOCKRUN_WALLET_KEY", "must-not-read-wallet")

    def handler(request):
        return httpx.Response(
            429,
            json={"error": {"code": "rate_limited", "message": "slow down"}},
            headers={"retry-after": RETRY_AFTER},
        )

    with patch("blockrun_llm.wallet.load_wallet", side_effect=AssertionError("wallet read")):
        client = cls()
    client._client.close()
    client._client = httpx.Client(
        headers=client._client.headers, transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(APIError) as failure:
            getattr(client, method)(*args, **kwargs)
        assert failure.value.status_code == 429
        assert failure.value.retry_after == RETRY_AFTER
        assert failure.value.retry_after_seconds == 17.0
    finally:
        client.close()


class TestRetryAfterParsing:
    def test_the_raw_header_is_kept_verbatim(self):
        err = APIError("rate limited", 429, None, retry_after="17")
        assert err.retry_after == "17"
        assert err.retry_after_seconds == 17.0

    def test_absent_header_is_none_not_zero(self):
        """Zero would read as "retry immediately", which is the opposite of
        what an unknown wait means."""
        err = APIError("boom", 500)
        assert err.retry_after is None
        assert err.retry_after_seconds is None

    @pytest.mark.parametrize(
        "raw",
        ["Wed, 21 Oct 2026 07:28:00 GMT", "soon", "", "   ", "-5"],
        ids=["http-date", "garbage", "empty", "whitespace", "negative"],
    )
    def test_unparseable_delays_do_not_become_a_number(self, raw):
        """The HTTP-date form is legal and this SDK does not translate it. A
        caller sleeping on a fabricated number is worse than one that knows it
        has to decide for itself."""
        err = APIError("rate limited", 429, None, retry_after=raw)
        assert err.retry_after_seconds is None

    def test_a_date_header_is_still_handed_back_raw(self):
        raw = "Wed, 21 Oct 2026 07:28:00 GMT"
        err = APIError("rate limited", 429, None, retry_after=raw)
        assert err.retry_after == raw

    def test_fractional_seconds_survive(self):
        assert APIError("x", 429, None, retry_after="0.5").retry_after_seconds == 0.5


class TestRetryAfterOf:
    def test_reads_the_header_case_insensitively(self):
        resp = httpx.Response(429, headers={"Retry-After": "30"})
        assert retry_after_of(resp) == "30"

    def test_missing_header_is_none(self):
        assert retry_after_of(httpx.Response(429)) is None

    def test_an_object_without_headers_does_not_explode(self):
        """This runs inside an error path. Raising here would replace the real
        failure with an AttributeError about the failure."""

        class Bare:
            status_code = 429

        assert retry_after_of(Bare()) is None

    def test_blank_header_reads_as_absent(self):
        assert retry_after_of(httpx.Response(429, headers={"retry-after": "  "})) is None


def test_from_response_keeps_status_body_and_header():
    resp = httpx.Response(429, json={"error": "limited"}, headers={"retry-after": RETRY_AFTER})
    err = APIError.from_response(resp, "Request failed", json.loads(resp.text))
    assert (err.status_code, err.retry_after, err.response) == (
        429,
        RETRY_AFTER,
        {"error": "limited"},
    )


def test_the_signature_stays_backwards_compatible():
    """Every pre-existing call site passes three positional arguments and must
    keep working untouched."""
    err = APIError("boom", 502, {"error": "upstream"})
    assert (err.status_code, err.response, err.retry_after) == (502, {"error": "upstream"}, None)
