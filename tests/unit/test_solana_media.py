"""Unit tests for the Solana media surface added in #16 (video/music/speech/
sound-effects/price/list_voices) plus the mid-poll re-sign payment-terms guard.

Payment flow is mocked at the httpx transport level (402 on the unsigned probe,
success once a PAYMENT-SIGNATURE is present); the x402 codec + signer are
stubbed so no wallet or network is needed — same approach as
test_solana_timeout_routing.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import httpx
import pytest

from blockrun_llm.solana_client import SolanaLLMClient, _assert_same_payment_terms
from blockrun_llm.types import (
    MusicResponse,
    PaymentError,
    SpeechResponse,
)


# ---------------------------------------------------------------------------
# _assert_same_payment_terms — the mid-poll re-sign guard
# ---------------------------------------------------------------------------


def _payload(amount: str, pay_to: str) -> SimpleNamespace:
    return SimpleNamespace(accepted=SimpleNamespace(amount=amount, pay_to=pay_to))


class TestPaymentTermsGuard:
    def test_same_terms_pass(self) -> None:
        # Identical amount + recipient (the normal stale-blockhash re-sign) is
        # allowed through with no exception.
        _assert_same_payment_terms(_payload("1000000", "WALLET_A"), "1000000", "WALLET_A")

    def test_amount_change_rejected(self) -> None:
        with pytest.raises(PaymentError, match="changed the payment terms"):
            _assert_same_payment_terms(_payload("9999999", "WALLET_A"), "1000000", "WALLET_A")

    def test_recipient_change_rejected(self) -> None:
        with pytest.raises(PaymentError, match="changed the payment terms"):
            _assert_same_payment_terms(_payload("1000000", "ATTACKER"), "1000000", "WALLET_A")

    def test_amount_type_coerced_before_compare(self) -> None:
        # int vs str for the same value must not trip the guard.
        _assert_same_payment_terms(_payload(1000000, "WALLET_A"), "1000000", "WALLET_A")


# ---------------------------------------------------------------------------
# Media dispatch — body construction + response parsing over the mocked flow
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_x402_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blockrun_llm.solana_client.decode_payment_required_header",
        lambda header: {"stub": True},
    )
    monkeypatch.setattr(
        "blockrun_llm.solana_client.encode_payment_signature_header",
        lambda payload: "stub-signature",
    )


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("blockrun_llm.cache.get_cached", lambda *a, **k: None)
    monkeypatch.setattr("blockrun_llm.cache.save_to_cache", lambda *a, **k: None)


def _make_client(handler: Any) -> SolanaLLMClient:
    with (
        mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
        mock.patch("blockrun_llm.solana_client._create_signer"),
    ):
        client = SolanaLLMClient(
            private_key="bogus_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
        )

    class _FakePayload:
        class accepted:
            amount = "1000000"
            pay_to = "GsbwXfJraMomNxBcpR3DBNxnKwZbyq7YCoDdSLDwzxdV"

    client._x402_client = mock.MagicMock()
    client._x402_client.create_payment_payload.return_value = _FakePayload()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client._address = "11111111111111111111111111111111"
    return client


def _paid_flow(calls: List[httpx.Request], ok_body: Dict[str, Any]):
    """402 on the unsigned probe, then ``ok_body`` once signed. Captures the
    signed request so tests can assert the forwarded JSON body + path."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                headers={"content-type": "application/json", "payment-required": "stub"},
                json={"error": "Payment Required"},
            )
        calls.append(request)
        return httpx.Response(200, json=ok_body, headers={"content-type": "application/json"})

    return handler


_MUSIC_OK = {"created": 1, "model": "minimax/music-2.5+", "data": [{"url": "https://cdn/x.mp3"}]}
_SPEECH_OK = {
    "created": 1,
    "model": "elevenlabs/flash-v2.5",
    "data": [{"url": "https://cdn/x.wav"}],
}


class TestMediaDispatch:
    def test_music_body_and_response(self) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _MUSIC_OK))
        resp = client.music("lo-fi beats")
        assert isinstance(resp, MusicResponse)
        assert resp.data[0].url == "https://cdn/x.mp3"
        assert calls[-1].url.path == "/api/v1/audio/generations"
        sent = json.loads(calls[-1].content)
        assert sent["model"] == "minimax/music-2.5+"
        assert sent["instrumental"] is True

    def test_speech_body_and_response(self) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _SPEECH_OK))
        resp = client.speech("hello world", voice="sarah")
        assert isinstance(resp, SpeechResponse)
        assert resp.data[0].url == "https://cdn/x.wav"
        assert calls[-1].url.path == "/api/v1/audio/speech"
        sent = json.loads(calls[-1].content)
        assert sent["input"] == "hello world"
        assert sent["voice"] == "sarah"

    def test_sound_effect_endpoint(self) -> None:
        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _SPEECH_OK))
        client.sound_effect("thunder clap")
        assert calls[-1].url.path == "/api/v1/audio/sound-effects"

    def test_list_voices_returns_list_not_envelope(self) -> None:
        # Regression: the gateway returns {"data": [...]}, and list_voices must
        # return the list, not the whole dict.
        voices = [{"id": "sarah"}, {"id": "adam"}]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": voices})

        client = _make_client(handler)
        assert client.list_voices() == voices


# ---------------------------------------------------------------------------
# Local validation — must reject before any HTTP / payment
# ---------------------------------------------------------------------------


class TestLocalValidation:
    def test_music_lyrics_with_instrumental_rejected(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))  # never reached
        with pytest.raises(ValueError, match="lyrics"):
            client.music("pop", instrumental=True, lyrics="la la la")

    def test_video_mutually_exclusive_image_and_face(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        with pytest.raises(ValueError, match="mutually exclusive"):
            client.video("a cat", image_url="https://x/y.png", real_face_asset_id="ta_abc")

    def test_video_bad_face_id_prefix(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        with pytest.raises(ValueError, match="ta_"):
            client.video("a cat", real_face_asset_id="not_a_valid_id")

    def test_portrait_enroll_requires_http_url(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        with pytest.raises(ValueError, match="image_url"):
            client.portrait_enroll("Alice", "ftp://bad/url")


# ---------------------------------------------------------------------------
# price() — missing "price" in a paid body must not raise a raw KeyError
# ---------------------------------------------------------------------------


class TestPriceRobustness:
    def test_missing_price_field_is_clean_error_not_keyerror(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "PAYMENT-SIGNATURE" not in request.headers:
                return httpx.Response(
                    402,
                    headers={"content-type": "application/json", "payment-required": "stub"},
                    json={"error": "Payment Required"},
                )
            # Paid 200 but the body is missing "price" — must surface as a
            # pydantic validation error, not a bare KeyError.
            return httpx.Response(200, json={"symbol": "BTCUSD"})

        client = _make_client(handler)
        with pytest.raises(Exception) as exc_info:
            client.price("crypto", "BTCUSD")
        assert not isinstance(exc_info.value, KeyError)


# ---------------------------------------------------------------------------
# Path-segment guard — LLM-controlled values can't escape the URL path
# ---------------------------------------------------------------------------


class TestPathSegmentGuard:
    def test_symbol_with_slash_rejected(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        with pytest.raises(ValueError, match="symbol"):
            client.price("crypto", "../../secret")

    def test_network_with_traversal_rejected(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        with pytest.raises(ValueError, match="network"):
            client.rpc("../evil", "eth_blockNumber")
