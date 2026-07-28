"""
Unit tests for image editing (img2img) request shaping.

The production /v1/images/image2image endpoint accepts ``image`` as either a
single base64 data URI or an array of 1-4 data URIs (multi-image fusion).
These tests use httpx.MockTransport — no real network call ever happens — and
assert that the SDK passes ``image`` through unchanged for both shapes, and
that the 402 → sign → retry dance preserves it on the paid request.
"""

from __future__ import annotations

import json

import httpx

from blockrun_llm import ImageClient

from ..helpers import TEST_PRIVATE_KEY, build_payment_required_response

DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


def _image_edit_transport(calls: list[httpx.Request]) -> httpx.MockTransport:
    """First POST → 402 with payment requirements; retry with signature → 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                headers={
                    "content-type": "application/json",
                    "payment-required": build_payment_required_response(),
                },
                json={"error": "Payment Required", "price": {"amount": "0.04"}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "created": 1700000000,
                "data": [{"url": "https://blockrun.ai/img/out.png"}],
            },
        )

    return httpx.MockTransport(handler)


def _make_client(calls: list[httpx.Request]) -> ImageClient:
    client = ImageClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=_image_edit_transport(calls))
    return client


def test_edit_single_image_passes_string_through():
    calls: list[httpx.Request] = []
    client = _make_client(calls)

    result = client.edit("Make the sky purple", image=DATA_URI)

    # 402 dance = exactly two requests; signature only on the retry.
    assert len(calls) == 2
    assert "PAYMENT-SIGNATURE" not in calls[0].headers
    assert "PAYMENT-SIGNATURE" in calls[1].headers

    body = json.loads(calls[1].content)
    assert body["image"] == DATA_URI
    assert isinstance(body["image"], str)
    assert result.data[0].url == "https://blockrun.ai/img/out.png"


def test_edit_defaults_to_gpt_image_2():
    calls: list[httpx.Request] = []
    client = _make_client(calls)

    client.edit("Make the sky purple", image=DATA_URI)

    body = json.loads(calls[1].content)
    # Default edit model matches the production schema default.
    assert body["model"] == "openai/gpt-image-2"


def test_edit_multi_image_passes_list_through():
    calls: list[httpx.Request] = []
    client = _make_client(calls)

    images = [DATA_URI, DATA_URI]
    result = client.edit(
        "Place the logo on the t-shirt",
        image=images,
        model="google/nano-banana",
    )

    body = json.loads(calls[1].content)
    # The array must survive serialization as a JSON array, not a coerced string.
    assert body["image"] == images
    assert isinstance(body["image"], list)
    assert len(body["image"]) == 2
    assert body["model"] == "google/nano-banana"
    assert result.data[0].url == "https://blockrun.ai/img/out.png"
