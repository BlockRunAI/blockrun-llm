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
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
    )
    # Content chunks.
    for i, d in enumerate(deltas):
        lines.append(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}],
                }
            )
        )
    # Final chunk with finish_reason.
    lines.append(
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
        )
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


def _make_paid_model_transport(sse_body: bytes, calls: List[httpx.Request]) -> httpx.MockTransport:
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
        assert (
            "".join(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
            == "Paid"
        )

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
        joined = "".join(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
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
        assert (
            "".join(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
            == "Hi!"
        )
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


# ---------------------------------------------------------------------------
# 5xx retry tests
# ---------------------------------------------------------------------------


def _make_flaky_free_transport(
    sse_body: bytes,
    fail_count: int,
    calls: List[httpx.Request],
    status: int = 503,
) -> httpx.MockTransport:
    """Returns ``status`` (default 503) for the first ``fail_count`` requests,
    then 200 + SSE on the next one. Used to verify retry-with-backoff logic."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= fail_count:
            return httpx.Response(
                status, headers={"content-type": "application/json"}, json={"error": "transient"}
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_body)

    return httpx.MockTransport(handler)


class TestStreamingRetries:
    """LLMClient._STREAM_5XX_BACKOFFS controls the retry policy. With three
    backoffs the SDK tries up to 4 times per phase before raising."""

    def test_recovers_after_two_503s(self, monkeypatch):
        # Zero out sleeps to keep tests fast.
        monkeypatch.setattr("time.sleep", lambda _s: None)

        calls: List[httpx.Request] = []
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(
            transport=_make_flaky_free_transport(_sse_events(["OK"]), fail_count=2, calls=calls)
        )
        chunks = list(
            client.chat_completion_stream(
                "nvidia/deepseek-v4-flash",
                [{"role": "user", "content": "hi"}],
            )
        )
        # 2 failed + 1 success
        assert len(calls) == 3
        assert any(c.choices[0].delta.content == "OK" for c in chunks)

    def test_raises_after_exhausting_retries(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _s: None)

        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                503,
                headers={"content-type": "application/json"},
                json={"error": "persistent"},
            )

        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        from blockrun_llm.types import APIError

        with pytest.raises(APIError):
            list(
                client.chat_completion_stream(
                    "nvidia/deepseek-v4-flash",
                    [{"role": "user", "content": "hi"}],
                )
            )
        # 1 + 3 backoffs == 4 probe attempts before raising.
        assert len(calls) == 1 + len(LLMClient._STREAM_5XX_BACKOFFS)

    def test_5xx_retry_also_works_after_payment(self, monkeypatch):
        """After signing a 402, subsequent 5xx on the retry stream should
        also trigger in-band retries before raising."""
        monkeypatch.setattr("time.sleep", lambda _s: None)

        calls: List[httpx.Request] = []
        body = _sse_events(["paid-OK"])

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            sig = request.headers.get("PAYMENT-SIGNATURE")
            if not sig:
                # Probe → 402 with payment requirements.
                return httpx.Response(
                    402,
                    headers={
                        "content-type": "application/json",
                        "payment-required": build_payment_required_response(),
                    },
                    json={"error": "Payment Required", "price": {"amount": "0.001"}},
                )
            # After payment: fail twice with 503, then succeed.
            paid_calls = sum(1 for c in calls if c.headers.get("PAYMENT-SIGNATURE"))
            if paid_calls <= 2:
                return httpx.Response(503, json={"error": "transient"})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        chunks = list(
            client.chat_completion_stream(
                "openai/gpt-5.5",
                [{"role": "user", "content": "hi"}],
            )
        )
        # 1 probe (402) + 2 paid-503 + 1 paid-200 == 4 total
        assert len(calls) == 4
        assert any(c.choices[0].delta.content == "paid-OK" for c in chunks)


# ---------------------------------------------------------------------------
# Fallback chain tests
# ---------------------------------------------------------------------------


class TestStreamingFallback:
    """``fallback_models`` walks the chain only on retriable pre-stream
    errors. Once a chunk is yielded, the upstream is committed."""

    def test_falls_back_to_next_model_on_503(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _s: None)

        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            body = request.read()
            import json as _json

            payload = _json.loads(body)
            if payload["model"] == "primary/bad":
                return httpx.Response(503, json={"error": "down"})
            # Fallback model succeeds.
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse_events(["FALLBACK"]),
            )

        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        chunks = list(
            client.chat_completion_stream(
                "primary/bad",
                [{"role": "user", "content": "hi"}],
                fallback_models=["fallback/good"],
            )
        )

        # 4 hits on primary (1 + 3 retries) all 503 → swap to fallback → 1 success
        assert len(calls) >= 5
        assert any(c.choices[0].delta.content == "FALLBACK" for c in chunks)

    def test_no_fallback_after_first_chunk(self, monkeypatch):
        """If the upstream successfully streams a few chunks then drops, we
        must NOT fall back — partial output has already gone to the caller."""
        monkeypatch.setattr("time.sleep", lambda _s: None)

        # Build SSE that's truncated (no [DONE]) so iter_lines simulates a
        # mid-stream connection drop via httpx parsing exception.
        truncated = (
            'data: {"id":"x","object":"chat.completion.chunk","created":1,'
            '"model":"primary/bad","choices":[{"index":0,"delta":{"role":"assistant"},'
            '"finish_reason":null}]}\n\n'
            'data: {"id":"x","object":"chat.completion.chunk","created":1,'
            '"model":"primary/bad","choices":[{"index":0,"delta":{"content":"par"},'
            '"finish_reason":null}]}\n\n'
        ).encode()

        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=truncated,
            )

        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        # Even with a fallback configured, the partial stream completes
        # naturally — no exception, no fallback. The fallback handler should
        # NEVER be invoked because we got valid chunks before the stream
        # ended.
        chunks = list(
            client.chat_completion_stream(
                "primary/bad",
                [{"role": "user", "content": "hi"}],
                fallback_models=["fallback/good"],
            )
        )
        # Exactly one upstream call: no fallback because partial chunks were
        # already yielded.
        assert len(calls) == 1
        contents = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        assert "par" in contents

    def test_non_retriable_error_does_not_fall_back(self, monkeypatch):
        """4xx (other than 402) must NOT trigger fallback — those are
        permanent client errors, not transient upstream issues."""
        monkeypatch.setattr("time.sleep", lambda _s: None)

        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(400, json={"error": "bad request"})

        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        from blockrun_llm.types import APIError

        with pytest.raises(APIError):
            list(
                client.chat_completion_stream(
                    "primary/bad",
                    [{"role": "user", "content": "hi"}],
                    fallback_models=["fallback/good"],
                )
            )
        # Single attempt; no retries (400 isn't 5xx), no fallback (400 isn't retriable).
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Streamed tool calls — regression for the archive-loop crash
# ---------------------------------------------------------------------------


def _sse_with_tool_call(model: str = "anthropic/claude-haiku-4-5") -> bytes:
    """SSE for a streamed tool call: role frame, a name frame, then argument-
    fragment frames (id/name absent — these used to fail the strict ToolCall
    schema), and a final finish=tool_calls frame with usage."""
    frames = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    lines = []
    for f in frames:
        f = {
            "id": "chatcmpl-tc",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            **f,
        }
        lines.append("data: " + json.dumps(f))
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


def _collect_tool_args(chunks: List[ChatCompletionChunk]) -> str:
    out: List[str] = []
    for c in chunks:
        if not c.choices:
            continue
        for tc in c.choices[0].delta.tool_calls or []:
            if tc.function and tc.function.arguments:
                out.append(tc.function.arguments)
    return "".join(out)


class TestStreamedToolCalls:
    """Streamed tool calls must parse + archive without crashing.

    The argument-fragment frames (id/name absent) used to fail the strict
    ToolCall schema, fall back to model_construct (leaving choices as dicts),
    then crash the archive loop with "'dict' object has no attribute 'delta'".
    The PAID path is used so cost_usd > 0 and the archive loop actually runs.
    """

    def test_sync_streamed_tool_call(self):
        calls: List[httpx.Request] = []
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(
            transport=_make_paid_model_transport(_sse_with_tool_call(), calls)
        )
        chunks = list(
            client.chat_completion_stream(
                "openai/gpt-5.5",
                [{"role": "user", "content": "weather?"}],
                max_tokens=64,
            )
        )
        tool_frames = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
        assert tool_frames, "expected streamed tool_call deltas"
        for c in tool_frames:
            assert hasattr(c.choices[0], "delta")  # parsed object, not a raw dict
        assert _collect_tool_args(chunks) == '{"city":"Paris"}'
        finishes = [
            c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
        ]
        assert finishes == ["tool_calls"]

    @pytest.mark.asyncio
    async def test_async_streamed_tool_call(self):
        calls: List[httpx.Request] = []
        client = AsyncLLMClient(private_key=TEST_PRIVATE_KEY)
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=_make_paid_model_transport(_sse_with_tool_call(), calls)
        )
        chunks: List[ChatCompletionChunk] = []
        async for chunk in client.chat_completion_stream(
            "openai/gpt-5.5",
            [{"role": "user", "content": "weather?"}],
        ):
            chunks.append(chunk)
        assert _collect_tool_args(chunks) == '{"city":"Paris"}'
        await client.close()

    def test_sync_streamed_tool_call_non_function_type(self):
        """A non-"function" tool ``type`` must still parse into a real object
        rather than re-trigger the strict-validation -> model_construct fallback
        (which would leave choices as raw dicts and crash consumers)."""
        frames = [
            {
                "id": "chatcmpl-tc",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "custom",  # non-"function" type
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-tc",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        sse = (
            "\n\n".join("data: " + json.dumps(f) for f in frames) + "\n\ndata: [DONE]\n\n"
        ).encode()
        calls: List[httpx.Request] = []
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=_make_paid_model_transport(sse, calls))
        chunks = list(
            client.chat_completion_stream(
                "openai/gpt-5.5",
                [{"role": "user", "content": "weather?"}],
                max_tokens=64,
            )
        )
        tool_frames = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
        assert tool_frames, "expected the non-'function' tool_call delta to parse"
        assert tool_frames[0].choices[0].delta.tool_calls[0].type == "custom"
        assert _collect_tool_args(chunks) == '{"city":"Paris"}'

    def test_sync_stream_archive_survives_model_construct_chunk_missing_id(self):
        """Archive-loop hardening: a frame that omits the required top-level
        ``id`` fails strict validation and is yielded via ``model_construct``
        (no ``id`` attribute). The stream-archiving loop must not crash reading
        ``chunk.id`` (old behaviour: ``AttributeError``); draining the generator
        runs the paid archive path end to end."""
        frames = [
            # Missing "id" -> model_construct, no .id attribute on the chunk.
            {
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-tc",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        sse = (
            "\n\n".join("data: " + json.dumps(f) for f in frames) + "\n\ndata: [DONE]\n\n"
        ).encode()
        calls: List[httpx.Request] = []
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=_make_paid_model_transport(sse, calls))
        # Must not raise: the archive loop reads chunk.id via the dict/attr-tolerant
        # accessor, so a model_construct'd chunk missing id is skipped, not fatal.
        chunks = list(
            client.chat_completion_stream(
                "openai/gpt-5.5",
                [{"role": "user", "content": "hi"}],
                max_tokens=64,
            )
        )
        assert len(chunks) == 2  # both frames yielded, stream drained cleanly
