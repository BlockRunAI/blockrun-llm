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

# Solana x402 extras (x402[svm]) require Python >= 3.10; skip the whole module
# on 3.9, where they aren't installed and the codec stubs below have nothing to
# patch. Mirrors test_solana_timeout_routing.py.
pytest.importorskip("x402")
pytest.importorskip("solders")

from blockrun_llm.solana_client import (  # noqa: E402
    AsyncSolanaLLMClient,
    SolanaLLMClient,
    _assert_same_payment_terms,
)
from blockrun_llm.types import (  # noqa: E402
    APIError,
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


# ---------------------------------------------------------------------------
# poll_url host pinning — the signed PAYMENT-SIGNATURE must not go off-host
# ---------------------------------------------------------------------------


class TestPollUrlHostPin:
    def test_relative_poll_url_resolved_to_api_host(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        assert (
            client._absolute_url("/api/v1/videos/generations/JOB")
            == "https://sol.blockrun.ai/api/v1/videos/generations/JOB"
        )

    def test_absolute_same_host_passes(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        url = "https://sol.blockrun.ai/api/v1/videos/generations/JOB"
        assert client._absolute_url(url) == url

    def test_absolute_cross_host_rejected(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        with pytest.raises(APIError, match="off-host"):
            client._absolute_url("https://evil.example.com/api/v1/videos/generations/JOB")

    def test_absolute_http_downgrade_rejected(self) -> None:
        client = _make_client(lambda r: httpx.Response(500))
        with pytest.raises(APIError, match="off-host"):
            client._absolute_url("http://sol.blockrun.ai/api/v1/videos/generations/JOB")


# ---------------------------------------------------------------------------
# Mid-poll re-sign — end-to-end through the poll loop (sync + async parity)
# ---------------------------------------------------------------------------


def _make_async_client(handler: Any) -> AsyncSolanaLLMClient:
    with (
        mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
        mock.patch("blockrun_llm.solana_client._create_signer"),
    ):
        client = AsyncSolanaLLMClient(
            private_key="bogus_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
        )

    class _FakePayload:
        class accepted:
            amount = "1000000"
            pay_to = "GsbwXfJraMomNxBcpR3DBNxnKwZbyq7YCoDdSLDwzxdV"

    client._x402_client = mock.MagicMock()
    # Async _sign_payment awaits create_payment_payload — must return a coroutine.
    client._x402_client.create_payment_payload = mock.AsyncMock(return_value=_FakePayload())
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._address = "11111111111111111111111111111111"
    return client


def _resign_handler(signed_poll_codes: List[int]):
    """Drive a video job through the mid-poll re-sign path.

    probe → 402; signed POST → 202 + poll_url; each *signed* GET poll returns
    the next code from ``signed_poll_codes`` (402 = settlement failed, 200 =
    completed); an *unsigned* GET is the re-challenge and always hands back a
    fresh 402 payment-required so the client re-signs.
    """
    pr = {"content-type": "application/json", "payment-required": "stub"}
    completed = {
        "status": "completed",
        "created": 1,
        "model": "xai/grok-imagine-video",
        "data": [{"url": "https://cdn/v.mp4"}],
    }
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        has_sig = "PAYMENT-SIGNATURE" in request.headers
        if request.method == "POST":
            if not has_sig:  # unsigned probe
                return httpx.Response(402, headers=pr, json={"error": "Payment Required"})
            return httpx.Response(  # signed submit
                202,
                json={
                    "id": "JOB",
                    "poll_url": "/api/v1/videos/generations/JOB",
                    "status": "queued",
                },
            )
        if not has_sig:  # unsigned re-challenge → trigger a re-sign
            return httpx.Response(402, headers=pr, json={"error": "Payment Required"})
        code = signed_poll_codes[min(state["i"], len(signed_poll_codes) - 1)]
        state["i"] += 1
        if code == 200:
            return httpx.Response(200, json=completed, headers={"content-type": "application/json"})
        return httpx.Response(402, headers=pr, json={"error": "settlement failed"})

    return handler


_HELPER_KW: Dict[str, Any] = {
    "poll_budget_seconds": 5.0,
    "poll_interval_seconds": 0.001,
    "max_resigns": 2,
    "label": "Video generation",
}
_VIDEO_BODY = {"model": "xai/grok-imagine-video", "prompt": "a cat"}


class TestResignEndToEnd:
    def test_sync_resign_same_terms_then_completes(self) -> None:
        # poll 402 (stale blockhash) → re-challenge → re-sign (same terms, guard
        # passes) → next poll 200 completed.
        client = _make_client(_resign_handler([402, 200]))
        data = client._request_image_with_payment(
            "/v1/videos/generations", dict(_VIDEO_BODY), **_HELPER_KW
        )
        assert data["data"][0]["url"] == "https://cdn/v.mp4"

    def test_sync_resign_reprice_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A guard rejection on the re-signed challenge must propagate, NOT be
        # swallowed by the re-sign try/except and masked as a generic 402.
        monkeypatch.setattr(
            "blockrun_llm.solana_client._assert_same_payment_terms",
            mock.Mock(side_effect=PaymentError("repriced")),
        )
        client = _make_client(_resign_handler([402, 200]))
        with pytest.raises(PaymentError, match="repriced"):
            client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_HELPER_KW
            )

    async def test_async_resign_same_terms_then_completes(self) -> None:
        client = _make_async_client(_resign_handler([402, 200]))
        try:
            data = await client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_HELPER_KW
            )
            assert data["data"][0]["url"] == "https://cdn/v.mp4"
        finally:
            await client._client.aclose()

    async def test_async_resign_reprice_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Async parity with the sync guard: a re-price is rejected and the
        # PaymentError propagates out of the poll loop.
        monkeypatch.setattr(
            "blockrun_llm.solana_client._assert_same_payment_terms",
            mock.Mock(side_effect=PaymentError("repriced")),
        )
        client = _make_async_client(_resign_handler([402, 200]))
        try:
            with pytest.raises(PaymentError, match="repriced"):
                await client._request_image_with_payment(
                    "/v1/videos/generations", dict(_VIDEO_BODY), **_HELPER_KW
                )
        finally:
            await client._client.aclose()


# ---------------------------------------------------------------------------
# Proactive per-poll re-sign — keeps the settlement blockhash fresh even when
# NO poll ever 402s (the 1080p Seedance case: upstream status flaps
# completed<->in_progress for minutes and would otherwise settle a stale
# signature). Distinct from the on-402 re-sign guard tested above.
# ---------------------------------------------------------------------------


def _fresh_sig_handler(n_in_progress: int, poll_sigs: List[str]):
    """Video job that NEVER 402s on a poll: n_in_progress in-progress polls,
    then completed. Records the PAYMENT-SIGNATURE seen on every signed poll so a
    test can assert the proactive re-sign refreshed it each time."""
    completed = {
        "status": "completed",
        "created": 1,
        "model": "xai/grok-imagine-video",
        "data": [{"url": "https://cdn/v.mp4"}],
    }
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if "PAYMENT-SIGNATURE" not in request.headers:
                return httpx.Response(
                    402,
                    headers={"content-type": "application/json", "payment-required": "stub"},
                    json={"error": "Payment Required"},
                )
            return httpx.Response(
                202,
                json={
                    "id": "JOB",
                    "poll_url": "/api/v1/videos/generations/JOB",
                    "status": "queued",
                },
            )
        poll_sigs.append(request.headers.get("PAYMENT-SIGNATURE"))
        state["i"] += 1
        if state["i"] <= n_in_progress:
            return httpx.Response(
                202, json={"status": "in_progress"}, headers={"content-type": "application/json"}
            )
        return httpx.Response(200, json=completed, headers={"content-type": "application/json"})

    return handler


class TestProactiveResign:
    def test_sync_refreshes_signature_every_poll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import itertools

        counter = itertools.count()
        monkeypatch.setattr(
            "blockrun_llm.solana_client.encode_payment_signature_header",
            lambda payload: f"sig-{next(counter)}",
        )
        # Fire the proactive re-sign on every poll (0s freshness window).
        monkeypatch.setattr(SolanaLLMClient, "MEDIA_RESIGN_FRESH_SECONDS", 0.0)

        poll_sigs: List[str] = []
        client = _make_client(_fresh_sig_handler(3, poll_sigs))
        data = client._request_image_with_payment(
            "/v1/videos/generations", dict(_VIDEO_BODY), **_HELPER_KW
        )
        assert data["data"][0]["url"] == "https://cdn/v.mp4"
        # 3 in-progress + 1 completed, and every signed poll carried a DISTINCT
        # (freshly re-signed) signature — the completed poll never reused the
        # stale submit-time one.
        assert len(poll_sigs) == 4
        assert len(set(poll_sigs)) == 4, poll_sigs

    @pytest.mark.asyncio
    async def test_async_refreshes_signature_every_poll(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import itertools

        counter = itertools.count()
        monkeypatch.setattr(
            "blockrun_llm.solana_client.encode_payment_signature_header",
            lambda payload: f"sig-{next(counter)}",
        )
        monkeypatch.setattr(SolanaLLMClient, "MEDIA_RESIGN_FRESH_SECONDS", 0.0)

        poll_sigs: List[str] = []
        client = _make_async_client(_fresh_sig_handler(3, poll_sigs))
        try:
            data = await client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_HELPER_KW
            )
            assert data["data"][0]["url"] == "https://cdn/v.mp4"
            assert len(poll_sigs) == 4
            assert len(set(poll_sigs)) == 4, poll_sigs
        finally:
            await client._client.aclose()

    # max_resigns == 0 (the image path) must NOT proactively re-sign, even with a
    # 0s freshness window: every poll reuses the single submit-time signature so
    # the image flow is provably untouched by the video-only fix.
    _IMAGE_KW: Dict[str, Any] = {
        "poll_budget_seconds": 5.0,
        "poll_interval_seconds": 0.001,
        "max_resigns": 0,
        "label": "Image generation",
    }

    def test_sync_image_path_never_resigns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import itertools

        counter = itertools.count()
        monkeypatch.setattr(
            "blockrun_llm.solana_client.encode_payment_signature_header",
            lambda payload: f"sig-{next(counter)}",
        )
        monkeypatch.setattr(SolanaLLMClient, "MEDIA_RESIGN_FRESH_SECONDS", 0.0)

        poll_sigs: List[str] = []
        client = _make_client(_fresh_sig_handler(3, poll_sigs))
        data = client._request_image_with_payment(
            "/v1/images/generations", dict(_VIDEO_BODY), **self._IMAGE_KW
        )
        assert data["data"][0]["url"] == "https://cdn/v.mp4"
        # 3 in-progress + 1 completed, every poll carrying the SAME submit-time
        # signature — the proactive re-sign never fired for max_resigns == 0.
        assert len(poll_sigs) == 4
        assert len(set(poll_sigs)) == 1, poll_sigs

    @pytest.mark.asyncio
    async def test_async_image_path_never_resigns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import itertools

        counter = itertools.count()
        monkeypatch.setattr(
            "blockrun_llm.solana_client.encode_payment_signature_header",
            lambda payload: f"sig-{next(counter)}",
        )
        monkeypatch.setattr(SolanaLLMClient, "MEDIA_RESIGN_FRESH_SECONDS", 0.0)

        poll_sigs: List[str] = []
        client = _make_async_client(_fresh_sig_handler(3, poll_sigs))
        try:
            data = await client._request_image_with_payment(
                "/v1/images/generations", dict(_VIDEO_BODY), **self._IMAGE_KW
            )
            assert data["data"][0]["url"] == "https://cdn/v.mp4"
            assert len(poll_sigs) == 4
            assert len(set(poll_sigs)) == 1, poll_sigs
        finally:
            await client._client.aclose()


_IMAGE_OK = {"created": 1, "model": "openai/gpt-image-2", "data": [{"url": "https://cdn/x.png"}]}
_VIDEO_OK = {"created": 1, "model": "xai/grok-imagine-video", "data": [{"url": "https://cdn/x.mp4"}]}
_DATA_URI = "data:image/png;base64,AA=="


class TestSolanaImageQuality:
    """`quality` is a Solana-only latency/fidelity knob (openai/gpt-image-* on
    the gateway). The Base gateway has no such field and zod would silently
    strip it, which is why ImageClient deliberately rejects it — see
    test_image_parameter_validation.test_generate_rejects_quality_parameter.
    """

    def test_image_forwards_quality(self) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _IMAGE_OK))
        client.image("a cat", model="openai/gpt-image-2", quality="low")
        sent = json.loads(calls[-1].content)
        assert sent["quality"] == "low"

    def test_image_omits_quality_when_unset(self) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _IMAGE_OK))
        client.image("a cat")
        assert "quality" not in json.loads(calls[-1].content)

    def test_image_edit_forwards_quality(self) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _IMAGE_OK))
        client.image_edit("make it green", _DATA_URI, quality="high")
        sent = json.loads(calls[-1].content)
        assert sent["quality"] == "high"
        assert calls[-1].url.path == "/api/v1/images/image2image"

    @pytest.mark.parametrize("value", ["low", "medium", "high", "auto"])
    def test_image_accepts_every_gateway_quality(self, value: str) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _IMAGE_OK))
        client.image("a cat", model="openai/gpt-image-2", quality=value)
        assert json.loads(calls[-1].content)["quality"] == value

    def test_image_rejects_unknown_quality_before_paying(self) -> None:
        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _IMAGE_OK))
        with pytest.raises(ValueError, match="quality must be one of"):
            client.image("a cat", model="openai/gpt-image-2", quality="hd")
        assert calls == []  # rejected locally — no request, no payment

    def test_image_edit_rejects_unknown_quality_before_paying(self) -> None:
        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _IMAGE_OK))
        with pytest.raises(ValueError, match="quality must be one of"):
            client.image_edit("make it green", _DATA_URI, quality="ultra")
        assert calls == []


class TestSolanaVideoInputType:
    def test_video_forwards_input_type(self) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _VIDEO_OK))
        client.video(
            "the flower blooms",
            image_url="https://example.com/bud.jpg",
            last_frame_url="https://example.com/bloom.jpg",
            input_type="first_last_frame",
        )
        assert json.loads(calls[-1].content)["input_type"] == "first_last_frame"

    def test_video_omits_input_type_when_unset(self) -> None:
        import json

        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _VIDEO_OK))
        client.video("a calm lake")
        assert "input_type" not in json.loads(calls[-1].content)

    def test_video_rejects_unknown_input_type_before_paying(self) -> None:
        calls: List[httpx.Request] = []
        client = _make_client(_paid_flow(calls, _VIDEO_OK))
        with pytest.raises(ValueError, match="input_type must be one of"):
            client.video("x", input_type="img")
        assert calls == []


class TestSharedVideoBodyBuilder:
    """Sync and async video() share _build_video_body so they can't drift."""

    def test_input_type_reaches_body(self) -> None:
        body = SolanaLLMClient._build_video_body(
            "x",
            model=None,
            image_url=None,
            last_frame_url=None,
            reference_image_urls=None,
            real_face_asset_id=None,
            duration_seconds=None,
            aspect_ratio=None,
            resolution=None,
            generate_audio=None,
            seed=None,
            watermark=None,
            return_last_frame=None,
            input_type="text",
        )
        assert body["input_type"] == "text"

    def test_async_video_exposes_input_type(self) -> None:
        import inspect

        assert "input_type" in inspect.signature(AsyncSolanaLLMClient.video).parameters
