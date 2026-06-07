"""Unit tests for VideoClient.generate() parameter validation and body construction."""

import os
import pytest

from blockrun_llm import VideoClient
from blockrun_llm.types import VideoResponse


@pytest.fixture
def client():
    # Deterministic dummy key — never signs against a live endpoint in unit
    # tests; we only exercise local request/response paths.
    os.environ.setdefault("BLOCKRUN_WALLET_KEY", "0x" + "11" * 32)
    return VideoClient()


@pytest.fixture
def captured(client, monkeypatch):
    captured = {}

    def fake_submit(body, budget_seconds):
        captured["body"] = body
        captured["budget"] = budget_seconds
        return VideoResponse(created=1, model=body["model"], data=[])

    monkeypatch.setattr(client, "_submit_and_poll", fake_submit)
    return captured


def test_first_last_frame_body(client, captured):
    client.generate(
        "the flower blooms",
        model="bytedance/seedance-1.5-pro",
        image_url="https://example.com/bud.jpg",
        last_frame_url="https://example.com/bloom.jpg",
    )
    assert captured["body"]["image_url"] == "https://example.com/bud.jpg"
    assert captured["body"]["last_frame_url"] == "https://example.com/bloom.jpg"


def test_reference_images_body(client, captured):
    urls = ["https://example.com/1.jpg", "https://example.com/2.jpg"]
    client.generate(
        "the character from image 1 in the city from image 2",
        model="bytedance/seedance-2.0",
        reference_image_urls=urls,
    )
    assert captured["body"]["reference_image_urls"] == urls
    assert "image_url" not in captured["body"]


def test_token360_passthroughs(client, captured):
    client.generate(
        "a calm lake at dawn",
        model="bytedance/seedance-2.0",
        aspect_ratio="16:9",
        seed=42,
        watermark=False,
        return_last_frame=True,
    )
    body = captured["body"]
    assert body["aspect_ratio"] == "16:9"
    assert body["seed"] == 42
    assert body["watermark"] is False
    assert body["return_last_frame"] is True


def test_last_frame_requires_image_url(client):
    with pytest.raises(ValueError, match="requires image_url"):
        client.generate("x", last_frame_url="https://example.com/last.jpg")


def test_last_frame_excludes_real_face(client):
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.generate(
            "x",
            image_url="https://example.com/first.jpg",
            last_frame_url="https://example.com/last.jpg",
            real_face_asset_id="ta_abc123",
        )


def test_reference_images_exclude_other_image_inputs(client):
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.generate(
            "x",
            image_url="https://example.com/seed.jpg",
            reference_image_urls=["https://example.com/r.jpg"],
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.generate(
            "x",
            real_face_asset_id="ta_abc123",
            reference_image_urls=["https://example.com/r.jpg"],
        )


def test_reference_images_max_nine(client):
    with pytest.raises(ValueError, match="at most 9"):
        client.generate(
            "x",
            reference_image_urls=[f"https://example.com/{i}.jpg" for i in range(10)],
        )


def test_image_url_and_real_face_still_exclusive(client):
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.generate(
            "x",
            image_url="https://example.com/a.jpg",
            real_face_asset_id="ta_abc123",
        )
