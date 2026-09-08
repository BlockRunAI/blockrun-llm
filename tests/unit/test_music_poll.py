"""Tests for the music-generation 202 + poll_url path.

Music is never fast: MiniMax takes one to three minutes per track and the
gateway answers 202 + poll_url once its inline window is over — which, since
2026-09-08, is at once. This client treated every non-200 as an error, so on
both rails a music request could not succeed at all; the enterprise ledger
showed 11 of 11 creates in 30 days answered 202 and this SDK raised
"API error: 202" for each. The image client already polls; music mirrors it.

``httpx.MockTransport`` keeps the network out. The poll interval is patched
to 0 so the loop spins instantly.
"""

from __future__ import annotations

import httpx
import pytest

from blockrun_llm import MusicClient
from blockrun_llm.types import APIError

from ..helpers import TEST_PRIVATE_KEY, build_payment_required_response

KEY = "brk_live_testkey"


def _wallet_client(transport: httpx.MockTransport) -> MusicClient:
    client = MusicClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=transport)
    return client


def _apikey_client(transport: httpx.MockTransport, monkeypatch: pytest.MonkeyPatch) -> MusicClient:
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    monkeypatch.delenv("BLOCKRUN_WALLET_KEY", raising=False)
    client = MusicClient()
    client._client = httpx.Client(transport=transport, headers=client._client.headers)
    return client


def _payment_required_402() -> httpx.Response:
    return httpx.Response(
        402,
        headers={
            "content-type": "application/json",
            "payment-required": build_payment_required_response(),
        },
        json={"error": "Payment Required", "price": {"amount": "0.1575"}},
    )


def _queued(job_id: str) -> httpx.Response:
    return httpx.Response(
        202,
        headers={"content-type": "application/json"},
        json={
            "id": job_id,
            "object": "audio.generation.job",
            "status": "queued",
            "model": "minimax/music-2.5+",
            "poll_url": f"/api/v1/audio/generations/{job_id}",
            "created": 1700000000,
        },
    )


def _completed(job_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/json", "x-payment-receipt": "0xabc"},
        json={
            "id": job_id,
            "object": "audio.generation.job",
            "status": "completed",
            "model": "minimax/music-2.5+",
            "created": 1700000000,
            "data": [{"url": "https://blockrun.ai/media/track.mp3", "duration_seconds": 182}],
            "payment": {"status": "settled"},
        },
    )


def test_music_wallet_rail_polls_to_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MusicClient, "MUSIC_POLL_INTERVAL_SECONDS", 0.0)
    calls: list[httpx.Request] = []
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path.endswith("/v1/audio/generations"):
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _payment_required_402()
            return _queued("mus_1")
        if request.method == "GET" and "/v1/audio/generations/mus_1" in request.url.path:
            polls["n"] += 1
            if polls["n"] == 1:
                return httpx.Response(
                    202,
                    headers={"content-type": "application/json"},
                    json={"id": "mus_1", "status": "in_progress"},
                )
            return _completed("mus_1")
        return httpx.Response(404)

    result = _wallet_client(httpx.MockTransport(handler)).generate("chill lo-fi beats")

    assert [c.method for c in calls] == ["POST", "POST", "GET", "GET"]
    # Every poll replays the signature the create was paid with; the job is
    # bound to that wallet and settles on the completed poll.
    assert calls[2].headers["PAYMENT-SIGNATURE"] == calls[1].headers["PAYMENT-SIGNATURE"]
    assert calls[3].headers["PAYMENT-SIGNATURE"] == calls[1].headers["PAYMENT-SIGNATURE"]
    assert result.data[0].url == "https://blockrun.ai/media/track.mp3"
    assert result.data[0].duration_seconds == 182
    assert result.txHash == "0xabc"


def test_music_api_key_rail_polls_on_first_202(monkeypatch: pytest.MonkeyPatch) -> None:
    # The account rail has already paid, so the 202 comes on the FIRST post
    # and the polls carry the key, not a signature.
    monkeypatch.setattr(MusicClient, "MUSIC_POLL_INTERVAL_SECONDS", 0.0)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return _queued("mus_2")
        if request.method == "GET" and "/v1/audio/generations/mus_2" in request.url.path:
            return _completed("mus_2")
        return httpx.Response(404)

    result = _apikey_client(httpx.MockTransport(handler), monkeypatch).generate("epic orchestral")

    assert [c.method for c in calls] == ["POST", "GET"]
    assert "PAYMENT-SIGNATURE" not in calls[1].headers
    assert calls[1].headers.get("authorization") == f"Bearer {KEY}"
    # The gateway's poll_url is /api/v1/...; api.blockrun.ai serves it at /v1/...
    assert calls[1].url.path == "/v1/audio/generations/mus_2"
    assert result.data[0].url == "https://blockrun.ai/media/track.mp3"


def test_music_poll_surfaces_upstream_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MusicClient, "MUSIC_POLL_INTERVAL_SECONDS", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _payment_required_402()
            return _queued("mus_3")
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "mus_3",
                "status": "failed",
                "error": "The operation was aborted due to timeout",
                "payment_status": "not_charged",
            },
        )

    with pytest.raises(APIError) as excinfo:
        _wallet_client(httpx.MockTransport(handler)).generate("waiting")
    assert "aborted due to timeout" in str(excinfo.value)


def test_music_poll_times_out_without_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MusicClient, "MUSIC_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(MusicClient, "MUSIC_POLL_BUDGET_SECONDS", 0.05)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _payment_required_402()
            return _queued("mus_4")
        return httpx.Response(
            202,
            headers={"content-type": "application/json"},
            json={"id": "mus_4", "status": "in_progress"},
        )

    with pytest.raises(APIError) as excinfo:
        _wallet_client(httpx.MockTransport(handler)).generate("forever")
    assert excinfo.value.status_code == 504
    assert "no payment was taken" in str(excinfo.value).lower()


def test_music_fast_path_unchanged() -> None:
    # A track that finishes inline still comes back as the legacy 200 shape.
    def handler(request: httpx.Request) -> httpx.Response:
        if "PAYMENT-SIGNATURE" not in request.headers:
            return _payment_required_402()
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-payment-receipt": "0xfast"},
            json={
                "created": 1700000000,
                "model": "minimax/music-2.5+",
                "data": [{"url": "https://blockrun.ai/media/fast.mp3"}],
            },
        )

    result = _wallet_client(httpx.MockTransport(handler)).generate("quick jingle")
    assert result.data[0].url == "https://blockrun.ai/media/fast.mp3"
    assert result.txHash == "0xfast"
