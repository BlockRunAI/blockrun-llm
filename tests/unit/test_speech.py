"""Unit tests for SpeechClient request construction and response parsing."""

import os

import pytest

from blockrun_llm import SpeechClient, SpeechResponse


@pytest.fixture
def client():
    # Deterministic dummy key — never actually signs against a live endpoint
    # in unit tests; we only exercise local request/response paths.
    os.environ.setdefault("BLOCKRUN_WALLET_KEY", "0x" + "11" * 32)
    return SpeechClient()


def test_generate_builds_speech_body(client, monkeypatch):
    captured = {}

    def fake_request(endpoint, body):
        captured["endpoint"] = endpoint
        captured["body"] = body
        return SpeechResponse(created=1, model=body["model"], data=[])

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    client.generate("Hello", voice="george", response_format="wav", speed=1.1)

    assert captured["endpoint"] == "/v1/audio/speech"
    assert captured["body"] == {
        "model": "elevenlabs/flash-v2.5",
        "input": "Hello",
        "voice": "george",
        "response_format": "wav",
        "speed": 1.1,
    }


def test_generate_omits_optional_fields(client, monkeypatch):
    captured = {}

    def fake_request(endpoint, body):
        captured["body"] = body
        return SpeechResponse(created=1, model=body["model"], data=[])

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    client.generate("Hi", model="elevenlabs/v3")

    assert captured["body"] == {"model": "elevenlabs/v3", "input": "Hi"}


def test_speak_is_generate_alias(client):
    assert SpeechClient.speak is SpeechClient.generate


def test_sound_effect_builds_body(client, monkeypatch):
    captured = {}

    def fake_request(endpoint, body):
        captured["endpoint"] = endpoint
        captured["body"] = body
        return SpeechResponse(created=1, model=body["model"], data=[])

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    client.sound_effect("rain on a tin roof", duration_seconds=5, prompt_influence=0.7)

    assert captured["endpoint"] == "/v1/audio/sound-effects"
    assert captured["body"] == {
        "model": "elevenlabs/sound-effects",
        "text": "rain on a tin roof",
        "duration_seconds": 5,
        "prompt_influence": 0.7,
    }


def test_speech_response_parses_payload():
    resp = SpeechResponse(
        created=1749000000,
        model="elevenlabs/flash-v2.5",
        data=[{"url": "https://cdn.example/a.mp3", "format": "mp3", "characters": 42}],
        txHash="0xabc",
    )
    assert resp.data[0].url == "https://cdn.example/a.mp3"
    assert resp.data[0].characters == 42
    assert resp.data[0].credits is None
    assert resp.txHash == "0xabc"


def test_get_wallet_address(client):
    addr = client.get_wallet_address()
    assert addr.startswith("0x")
    assert len(addr) == 42
