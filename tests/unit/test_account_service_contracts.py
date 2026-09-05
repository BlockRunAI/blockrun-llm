"""Public service entrypoints against an in-memory account gateway, never production."""

import json
from unittest.mock import patch

import httpx
import pytest

from blockrun_llm import (
    ImageClient,
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
from blockrun_llm.types import APIError, PaymentError

KEY = "brk_live_contract_fixture"
PHONE = "+12025550123"
# Each public method must reach its documented endpoint with account auth,
# preserve account failures, and never reinterpret a 402 as a wallet challenge.
CASES = [
    (ImageClient, "generate", ("test",), {}, "POST", "/v1/images/generations", {"prompt": "test"}),
    (
        VideoClient,
        "generate",
        ("test",),
        {"duration_seconds": 5},
        "POST",
        "/v1/videos/generations",
        {"duration_seconds": 5},
    ),
    (
        MusicClient,
        "generate",
        ("test music",),
        {},
        "POST",
        "/v1/audio/generations",
        {"prompt": "test music"},
    ),
    (SpeechClient, "generate", ("hello",), {}, "POST", "/v1/audio/speech", {"input": "hello"}),
    (SpeechClient, "sound_effect", ("rain",), {}, "POST", "/v1/audio/sound-effects", {}),
    (SpeechClient, "list_voices", (), {}, "GET", "/v1/audio/voices", {}),
    (
        VoiceClient,
        "call",
        (PHONE, "Read a test message"),
        {},
        "POST",
        "/v1/voice/call",
        {"to": PHONE},
    ),
    (VoiceClient, "get_status", ("fixture-call",), {}, "GET", "/v1/voice/call/fixture-call", {}),
    (PhoneClient, "lookup", (PHONE,), {}, "POST", "/v1/phone/lookup", {"phoneNumber": PHONE}),
    (PhoneClient, "lookup_fraud", (PHONE,), {}, "POST", "/v1/phone/lookup/fraud", {}),
    (
        PhoneClient,
        "buy_number",
        (),
        {"area_code": "202"},
        "POST",
        "/v1/phone/numbers/buy",
        {"areaCode": "202"},
    ),
    (PhoneClient, "renew_number", (PHONE,), {}, "POST", "/v1/phone/numbers/renew", {}),
    (PhoneClient, "list_numbers", (), {}, "POST", "/v1/phone/numbers/list", {}),
    (PhoneClient, "release_number", (PHONE,), {}, "POST", "/v1/phone/numbers/release", {}),
    (
        PortraitClient,
        "enroll",
        ("fixture", "https://example.com/test.png"),
        {},
        "POST",
        "/v1/portrait/enroll",
        {},
    ),
    (RealFaceClient, "init", ("fixture",), {}, "POST", "/v1/realface/init", {}),
    (RealFaceClient, "status", ("legacy_rf_123",), {}, "GET", "/v1/realface/status", {}),
    (
        RealFaceClient,
        "enroll",
        ("fixture", "https://example.com/test.png", "legacy_rf_123"),
        {},
        "POST",
        "/v1/realface/enroll",
        {"group_id": "legacy_rf_123"},
    ),
    (SearchClient, "search", ("test",), {}, "POST", "/v1/search", {"query": "test"}),
    (SurfClient, "get", ("market/ranking",), {}, "GET", "/v1/surf/market/ranking", {}),
    (
        SurfClient,
        "post",
        ("onchain/sql", {"query": "SELECT 1"}),
        {},
        "POST",
        "/v1/surf/onchain/sql",
        {"query": "SELECT 1"},
    ),
    (PriceClient, "price", ("crypto", "BTC-USD"), {}, "GET", "/v1/crypto/price/BTC-USD", {}),
    (PriceClient, "price", ("fx", "EUR-USD"), {}, "GET", "/v1/fx/price/EUR-USD", {}),
    (PriceClient, "price", ("commodity", "XAU-USD"), {}, "GET", "/v1/commodity/price/XAU-USD", {}),
    (
        PriceClient,
        "price",
        ("stocks", "AAPL"),
        {"market": "us"},
        "GET",
        "/v1/stocks/us/price/AAPL",
        {},
    ),
    (
        PriceClient,
        "history",
        ("crypto", "BTC-USD"),
        {"from_ts": 1, "to_ts": 2},
        "GET",
        "/v1/crypto/history/BTC-USD",
        {},
    ),
    (RpcClient, "call", ("solana", "getSlot"), {}, "POST", "/v1/rpc/solana", {"method": "getSlot"}),
    (RpcClient, "batch", ("base", [{"method": "eth_blockNumber"}]), {}, "POST", "/v1/rpc/base", {}),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0].__name__ + "." + c[1] + c[5] for c in CASES])
@pytest.mark.parametrize("status", [401, 402, 429])
def test_public_service_account_error_contract(case, status, monkeypatch):
    cls, method, args, kwargs, verb, path, expected_body = case
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    monkeypatch.delenv("BLOCKRUN_API_BASE_URL", raising=False)
    monkeypatch.setenv("BLOCKRUN_WALLET_KEY", "must-not-read-wallet")
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url.host == "api.blockrun.ai"
        assert request.url.path == path
        assert request.method == verb
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert "payment-signature" not in request.headers
        assert "x-payment" not in request.headers
        if expected_body:
            body = json.loads(request.content)
            assert all(body[k] == v for k, v in expected_body.items())
        return httpx.Response(
            status,
            json={"error": {"code": "fixture_limit", "message": "fixture"}},
            headers={"retry-after": "17", "payment-required": "must-not-sign"},
        )

    with patch("blockrun_llm.wallet.load_wallet", side_effect=AssertionError("wallet read")):
        client = cls()
    client._client.close()
    client._client = httpx.Client(
        headers=client._client.headers, transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(PaymentError if status == 402 else APIError) as failure:
            getattr(client, method)(*args, **kwargs)
        if status != 402:
            assert failure.value.status_code == status
        assert len(seen) == 1
    finally:
        client.close()
