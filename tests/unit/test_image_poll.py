"""Tests for the image-generation 202 + poll_url slow path.

Regression guard for the silent failure on slow models like
``openai/gpt-image-2`` and ``openai/dall-e-3``: pre-fix, the SDK treated
202 as success and tried to parse the job-stub JSON as an
``ImageResponse``, raising a confusing Pydantic ValidationError. Now the
client transparently polls until the upstream finishes.

These tests use ``httpx.MockTransport`` so no real network is ever
called. They also patch ``IMAGE_POLL_INTERVAL_SECONDS`` to 0 so the loop
spins instantly.
"""

from __future__ import annotations

import json
from typing import List

import httpx
import pytest

from blockrun_llm import ImageClient
from blockrun_llm.types import APIError, PaymentError

from ..helpers import TEST_PRIVATE_KEY, build_payment_required_response


def _make_client(transport: httpx.MockTransport) -> ImageClient:
    client = ImageClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=transport)
    return client


def _payment_required_402(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        402,
        headers={
            "content-type": "application/json",
            "payment-required": build_payment_required_response(),
        },
        json={"error": "Payment Required", "price": {"amount": "0.06"}},
    )


def test_image_generate_polls_to_completion_on_202(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt-image-2 routinely exceeds the 30s inline window → 202 + poll_url
    → SDK should poll the same URL with the same PAYMENT-SIGNATURE until
    status=completed, then return the image."""
    monkeypatch.setattr(ImageClient, "IMAGE_POLL_INTERVAL_SECONDS", 0.0)

    calls: List[httpx.Request] = []
    poll_state = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path

        if request.method == "POST" and path.endswith("/v1/images/generations"):
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _payment_required_402(request)
            # Signed POST → slow path 202 with poll_url
            return httpx.Response(
                202,
                headers={"content-type": "application/json"},
                json={
                    "id": "img_abc123",
                    "object": "image.generation.job",
                    "status": "queued",
                    "model": "openai/gpt-image-2",
                    "size": "1024x1024",
                    "n": 1,
                    "poll_url": "/api/v1/images/generations/img_abc123",
                    "created": 1700000000,
                },
            )

        if request.method == "GET" and "/v1/images/generations/img_abc123" in path:
            poll_state["count"] += 1
            # First poll: still in progress; second poll: completed.
            if poll_state["count"] == 1:
                return httpx.Response(
                    202,
                    headers={"content-type": "application/json"},
                    json={"id": "img_abc123", "status": "in_progress"},
                )
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "id": "img_abc123",
                    "object": "image.generation.job",
                    "status": "completed",
                    "model": "openai/gpt-image-2",
                    "created": 1700000000,
                    "data": [{"url": "https://blockrun.ai/img/abc.png"}],
                },
            )

        return httpx.Response(404)

    client = _make_client(httpx.MockTransport(handler))
    result = client.generate("古风汉服少女", model="openai/gpt-image-2", size="1024x1024")

    # 1 probe POST + 1 signed POST + 2 polls = 4 calls
    assert len(calls) == 4
    assert calls[0].method == "POST"
    assert "PAYMENT-SIGNATURE" not in calls[0].headers
    assert calls[1].method == "POST"
    assert "PAYMENT-SIGNATURE" in calls[1].headers
    # Both polls replay the same signature.
    assert calls[2].method == "GET"
    assert calls[3].method == "GET"
    assert calls[2].headers.get("PAYMENT-SIGNATURE")
    assert calls[2].headers["PAYMENT-SIGNATURE"] == calls[1].headers["PAYMENT-SIGNATURE"]
    assert calls[3].headers["PAYMENT-SIGNATURE"] == calls[1].headers["PAYMENT-SIGNATURE"]

    assert result.data[0].url == "https://blockrun.ai/img/abc.png"


def test_image_poll_surfaces_settlement_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the final poll's settlement fails on the facilitator
    (e.g. ``transaction_simulation_failed``), the SDK must surface the
    gateway's real reason instead of swallowing it."""
    monkeypatch.setattr(ImageClient, "IMAGE_POLL_INTERVAL_SECONDS", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/v1/images/generations"):
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _payment_required_402(request)
            return httpx.Response(
                202,
                headers={"content-type": "application/json"},
                json={
                    "id": "img_xyz",
                    "status": "queued",
                    "poll_url": "/api/v1/images/generations/img_xyz",
                    "created": 1700000000,
                    "model": "openai/gpt-image-2",
                    "size": "1024x1024",
                    "n": 1,
                },
            )
        # Settlement failed at the facilitator.
        return httpx.Response(
            402,
            headers={"content-type": "application/json"},
            json={
                "error": "Payment settlement failed",
                "details": "transaction_simulation_failed",
            },
        )

    client = _make_client(httpx.MockTransport(handler))

    with pytest.raises(PaymentError) as excinfo:
        client.generate("a red apple", model="openai/gpt-image-2")

    exc = excinfo.value
    assert exc.status_code == 402
    assert exc.response is not None
    assert exc.response.get("details") == "transaction_simulation_failed"
    assert "transaction_simulation_failed" in str(exc)


def test_image_poll_times_out_without_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll budget exhausted → APIError 504 + no settlement (so no
    charge). This is the customer-friendly contract: pay only when the
    image is delivered."""
    monkeypatch.setattr(ImageClient, "IMAGE_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(ImageClient, "IMAGE_POLL_BUDGET_SECONDS", 0.05)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/v1/images/generations"):
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _payment_required_402(request)
            return httpx.Response(
                202,
                headers={"content-type": "application/json"},
                json={
                    "id": "img_stuck",
                    "status": "queued",
                    "poll_url": "/api/v1/images/generations/img_stuck",
                    "created": 1700000000,
                    "model": "openai/gpt-image-2",
                    "size": "1024x1024",
                    "n": 1,
                },
            )
        # Always still in_progress — never completes.
        return httpx.Response(
            202,
            headers={"content-type": "application/json"},
            json={"id": "img_stuck", "status": "in_progress"},
        )

    client = _make_client(httpx.MockTransport(handler))

    with pytest.raises(APIError) as excinfo:
        client.generate("waiting forever", model="openai/gpt-image-2")

    exc = excinfo.value
    assert exc.status_code == 504
    assert "did not complete" in str(exc)
    assert "no payment was taken" in str(exc).lower()


def test_image_generate_fast_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: fast models that return 200 inline must still work
    identically — the poll path is only entered on 202."""
    monkeypatch.setattr(ImageClient, "IMAGE_POLL_INTERVAL_SECONDS", 0.0)

    calls: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if "PAYMENT-SIGNATURE" not in request.headers:
            return _payment_required_402(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "created": 1700000000,
                "data": [{"url": "https://blockrun.ai/img/fast.png"}],
            },
        )

    client = _make_client(httpx.MockTransport(handler))
    result = client.generate("fast model", model="google/nano-banana")

    assert len(calls) == 2  # No polling — went straight through.
    assert result.data[0].url == "https://blockrun.ai/img/fast.png"


def test_image_poll_surfaces_upstream_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the upstream generation fails (content policy, model error,
    etc.), the gateway flips ``status: failed`` on the poll. The SDK
    should raise APIError with the upstream reason — no settlement."""
    monkeypatch.setattr(ImageClient, "IMAGE_POLL_INTERVAL_SECONDS", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST":
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _payment_required_402(request)
            return httpx.Response(
                202,
                headers={"content-type": "application/json"},
                json={
                    "id": "img_bad",
                    "status": "queued",
                    "poll_url": "/api/v1/images/generations/img_bad",
                    "created": 1700000000,
                    "model": "openai/gpt-image-2",
                    "size": "1024x1024",
                    "n": 1,
                },
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "img_bad",
                "status": "failed",
                "error": "content policy violation",
            },
        )

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(APIError) as excinfo:
        client.generate("blocked prompt", model="openai/gpt-image-2")
    assert "content policy violation" in str(excinfo.value)
