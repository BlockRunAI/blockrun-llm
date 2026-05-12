"""
Unit tests for chat_completion_stream (sync + async).

We use httpx.MockTransport so no real network call ever happens. Two
scenarios are covered for each variant:

1. **Free-model path** — first POST returns 200 + ``text/event-stream``;
   we should iterate chunks without ever invoking the x402 signer.

2. **Paid-model path** — first POST returns 402 with payment requirements;
   we verify the signer fires, the retry sends a ``PAYMENT-SIGNATURE``
   header, and chunks come back on the second response.

Plus a couple of robustness tests: ``[DONE]`` terminator, malformed
chunks skipped, finish_reason on the final chunk.
"""

from __future__ import annotations

import json
from typing import Iterator, List

import httpx
import pytest

from blockrun_llm import AsyncLLMClient, ChatCompletionChunk, LLMClient
from blockrun_llm.types import PaymentError

from ..helpers import TEST_PRIVATE_KEY, build_payment_required_response


# ---------------------------------------------------------------------------
# Synthetic SSE bodies
# ---------------------------------------------------------------------------

def _sse_events(deltas: List[str], finish: str = "stop", model: str = "test/model") -> bytes:
    """Render a list of content deltas as raw SSE bytes ending with [DONE]."""
    lines: List[str] = []
    # First chunk — role only.
    lines.append(
        "data: " + json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
    )
    # Content chunks.
    for i, d in enumerate(deltas):
        lines.append(
            "data: " + json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}],
            })
        )
    # Final chunk with finish_reason.
    lines.append(
        "data: " + json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        })
    )
    lines.append("data: [DONE]")
    body = "\n\n".join(lines) + "\n\n"
    return body.encode("utf-8")


def _sse_with_garbage(deltas: List[str]) -> bytes:
    """Same as ``_sse_events`` but with a couple of malformed lines mixed in
    to verify the parser is tolerant."""
    base = _sse_events(deltas).decode("utf-8")
    # Insert a malformed chunk after the first content event.
    parts = base.split("\n\n")
    parts.insert(2, "data: {this is not valid json}")
    parts.insert(3, ": this is an SSE comment heartbeat")
    return ("\n\n".join(parts)).encode("utf-8")


# ---------------------------------------------------------------------------
# Mock transports
# ---------------------------------------------------------------------------

def _make_free_model_transport(sse_body: bytes, calls: List[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    return httpx.MockTransport(handler)


def _make_paid_model_transport(
    sse_body: bytes, calls: List[httpx.Request]
) -> httpx.MockTransport:
    """First call → 402 with valid payment-required header; second → 200 SSE."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                headers={
                    "content-type": "application/json",
                    "payment-required": build_payment_required_response(),
                },
                json={"error": "Payment Required", "price": {"amount": "0.001"}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------

class TestSyncStreaming:
    def test_free_model_streams_without_payment(self):
        calls: List[httpx.Request] = []
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(
            transport=_make_free_model_transport(_sse_events(["Hello", " world"]), calls)
        )

        chunks: List[ChatCompletionChunk] = list(
            client.chat_completion_stream(
                "nvidia/deepseek-v4-flash",
                [{"role": "user", "content": "hi"}],
                max_tokens=32,
            )
        )

        # Free path = one HTTP request total. No PAYMENT-SIGNATURE seen.
        assert len(calls) == 1
        assert "PAYMENT-SIGNATURE" not in calls[0].headers

        # First chunk carries role; subsequent carry content; last carries finish.
        roles = [c.choices[0].delta.role for c in chunks]
        contents = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        finishes = [c.choices[0].finish_reason for c in chunks if c.choices[0].finish_reason]

        assert roles[0] == "assistant"
        assert "".join(contents) == "Hello world"
        assert finishes == ["stop"]

    def test_paid_model_signs_and_retries(self):
        calls: List[httpx.Request] = []
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(
            transport=_make_paid_model_transport(_sse_events(["Paid"]), calls)
        )

        chunks = list(
            client.chat_completion_stream(
                "openai/gpt-5.5",
                [{"role": "user", "content": "hi"}],
                max_tokens=16,
            )
        )

        # 402 dance = exactly two HTTP requests.
        assert len(calls) == 2
        assert "PAYMENT-SIGNATURE" not in calls[0].headers
        assert "PAYMENT-SIGNATURE" in calls[1].headers
        # Session cost was tracked.
        assert client._session_calls == 1
        assert client._session_total_usd > 0
        # Streamed content arrives.
        assert "".join(
            c.choices[0].delta.content for c in chunks if c.choices[0].delta.content
        ) == "Paid"

    def test_malformed_chunks_dont_abort_stream(self):
        calls: List[httpx.Request] = []
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(
            transport=_make_free_model_transport(_sse_with_garbage(["A", "B"]), calls)
        )

        chunks = list(
            client.chat_completion_stream(
                "nvidia/deepseek-v4-flash",
                [{"role": "user", "content": "hi"}],
            )
        )
        # We should have gotten both deltas through, despite the garbage chunk.
        joined = "".join(
            c.choices[0].delta.content for c in chunks if c.choices[0].delta.content
        )
        assert joined == "AB"

    def test_paid_path_propagates_payment_rejected(self):
        """If the retry also returns 402, surface PaymentError."""
        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                402,
                headers={
                    "content-type": "application/json",
                    "payment-required": build_payment_required_response(),
                },
                json={"error": "Payment Required", "price": {"amount": "0.001"}},
            )

        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(PaymentError):
            list(
                client.chat_completion_stream(
                    "openai/gpt-5.5",
                    [{"role": "user", "content": "hi"}],
                )
            )
        # Probe + retry both got 402.
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------

class TestAsyncStreaming:
    @pytest.mark.asyncio
    async def test_async_free_model(self):
        calls: List[httpx.Request] = []
        client = AsyncLLMClient(private_key=TEST_PRIVATE_KEY)
        # Swap in mock transport (same pattern as sync).
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=_make_free_model_transport(_sse_events(["Hi", "!"]), calls)
        )

        chunks: List[ChatCompletionChunk] = []
        async for chunk in client.chat_completion_stream(
            "nvidia/deepseek-v4-flash",
            [{"role": "user", "content": "hi"}],
        ):
            chunks.append(chunk)

        assert len(calls) == 1
        assert "".join(
            c.choices[0].delta.content for c in chunks if c.choices[0].delta.content
        ) == "Hi!"
        await client.close()

    @pytest.mark.asyncio
    async def test_async_paid_model_signs_and_retries(self):
        calls: List[httpx.Request] = []
        client = AsyncLLMClient(private_key=TEST_PRIVATE_KEY)
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=_make_paid_model_transport(_sse_events(["X"]), calls)
        )

        chunks: List[ChatCompletionChunk] = []
        async for chunk in client.chat_completion_stream(
            "openai/gpt-5.5",
            [{"role": "user", "content": "hi"}],
        ):
            chunks.append(chunk)

        assert len(calls) == 2
        assert "PAYMENT-SIGNATURE" not in calls[0].headers
        assert "PAYMENT-SIGNATURE" in calls[1].headers
        await client.close()
