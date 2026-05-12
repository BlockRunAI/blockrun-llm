"""
Unit tests for SolanaLLMClient.chat_completion_stream (sync only — async
isn't implemented for the Solana client yet).

We mock at the httpx transport level so no real wallet / RPC / network
is needed. The Solana payment-signing path uses the x402 SDK's SVM
client, which we patch with a small fake that returns a static encoded
payload — this isolates the SSE/retry logic from the cryptography.
"""

from __future__ import annotations

import json
from typing import List

import httpx
import pytest

# Skip the whole module if the solana extras aren't installed.
pytest.importorskip("x402")
pytest.importorskip("solders")

from blockrun_llm import ChatCompletionChunk, SolanaLLMClient
from blockrun_llm.types import APIError, PaymentError


# ---------------------------------------------------------------------------
# Helpers — synthetic SSE bodies (same shape Base tests use)
# ---------------------------------------------------------------------------

def _sse_events(deltas: List[str], finish: str = "stop", model: str = "test/model") -> bytes:
    lines: List[str] = []
    lines.append(
        "data: " + json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })
    )
    for d in deltas:
        lines.append(
            "data: " + json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}],
            })
        )
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
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


# A valid Solana keypair seed (32 bytes, base58-encoded). Hardcoded test value;
# never use in production.
TEST_SOLANA_KEY = "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE"  # 32 1-bytes


@pytest.fixture
def solana_client():
    """Build a SolanaLLMClient without going through the x402 SDK signer
    init (which needs real keys + an RPC). We monkey-patch the signer
    after construction by replacing the x402_client with a fake."""
    import unittest.mock as mock

    with mock.patch("blockrun_llm.solana_client.register_exact_svm_client"), \
         mock.patch("blockrun_llm.solana_client._create_signer"):
        client = SolanaLLMClient(
            private_key="bogus_not_used_because_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
        )

    # Stub the signing path: x402_client.create_payment_payload returns an
    # object with the right shape for the rest of the code.
    class _FakePayload:
        class accepted:
            amount = "1000000"  # 1 USDC in micro-units

    client._x402_client = mock.MagicMock()
    client._x402_client.create_payment_payload.return_value = _FakePayload()
    return client


def _patch_sse_helpers(monkeypatch):
    """Replace the x402 SDK's decode/encode functions with identity-ish
    stubs so our handler code can run without real x402 payloads."""
    monkeypatch.setattr(
        "blockrun_llm.solana_client.decode_payment_required_header",
        lambda header: {"stub": True},
    )
    monkeypatch.setattr(
        "blockrun_llm.solana_client.encode_payment_signature_header",
        lambda payload: "stub-signature",
    )


# ---------------------------------------------------------------------------
# Transport builders
# ---------------------------------------------------------------------------

def _free_transport(sse_body: bytes, calls: List[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_body
        )
    return httpx.MockTransport(handler)


def _paid_transport(sse_body: bytes, calls: List[httpx.Request]) -> httpx.MockTransport:
    """First call → 402; second call (with PAYMENT-SIGNATURE) → 200 SSE."""
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if "PAYMENT-SIGNATURE" not in request.headers:
            return httpx.Response(
                402,
                headers={
                    "content-type": "application/json",
                    "payment-required": "stub-payment-required-base64",
                },
                json={"error": "Payment Required"},
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_body
        )
    return httpx.MockTransport(handler)


def _flaky_transport(
    sse_body: bytes, fail_count: int, calls: List[httpx.Request], status: int = 503
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= fail_count:
            return httpx.Response(status, json={"error": "transient"})
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_body
        )
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSolanaStreaming:
    def test_free_model_streams_directly(self, solana_client, monkeypatch):
        calls: List[httpx.Request] = []
        solana_client._client = httpx.Client(
            transport=_free_transport(_sse_events(["Hello", " world"]), calls)
        )

        chunks = list(solana_client.chat_completion_stream(
            "nvidia/deepseek-v4-flash",
            [{"role": "user", "content": "hi"}],
        ))

        assert len(calls) == 1
        assert "PAYMENT-SIGNATURE" not in calls[0].headers
        content = "".join(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
        assert content == "Hello world"

    def test_paid_model_signs_and_retries(self, solana_client, monkeypatch):
        _patch_sse_helpers(monkeypatch)
        calls: List[httpx.Request] = []
        solana_client._client = httpx.Client(
            transport=_paid_transport(_sse_events(["Paid"]), calls)
        )

        chunks = list(solana_client.chat_completion_stream(
            "openai/gpt-5.5",
            [{"role": "user", "content": "hi"}],
        ))
        # 1 probe (402) + 1 paid (200) == 2 total
        assert len(calls) == 2
        assert "PAYMENT-SIGNATURE" not in calls[0].headers
        assert calls[1].headers["PAYMENT-SIGNATURE"] == "stub-signature"
        content = "".join(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
        assert content == "Paid"
        assert solana_client._session_calls == 1
        assert solana_client._last_call_cost > 0

    def test_retries_5xx_with_backoff(self, solana_client, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _s: None)
        calls: List[httpx.Request] = []
        solana_client._client = httpx.Client(
            transport=_flaky_transport(_sse_events(["OK"]), fail_count=2, calls=calls)
        )

        chunks = list(solana_client.chat_completion_stream(
            "nvidia/deepseek-v4-flash",
            [{"role": "user", "content": "hi"}],
        ))
        # 2 failed + 1 success
        assert len(calls) == 3
        assert any(c.choices[0].delta.content == "OK" for c in chunks)

    def test_raises_after_exhausting_retries(self, solana_client, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _s: None)
        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(503, json={"error": "persistent"})

        solana_client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(APIError):
            list(solana_client.chat_completion_stream(
                "nvidia/deepseek-v4-flash",
                [{"role": "user", "content": "hi"}],
            ))
        # 1 + 3 backoffs == 4 attempts
        assert len(calls) == 1 + len(SolanaLLMClient._STREAM_5XX_BACKOFFS)

    def test_fallback_models_walks_chain(self, solana_client, monkeypatch):
        _patch_sse_helpers(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            body = json.loads(request.read())
            if body["model"] == "primary/bad":
                return httpx.Response(503, json={"error": "down"})
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse_events(["FALLBACK"]),
            )

        solana_client._client = httpx.Client(transport=httpx.MockTransport(handler))

        chunks = list(solana_client.chat_completion_stream(
            "primary/bad",
            [{"role": "user", "content": "hi"}],
            fallback_models=["fallback/good"],
        ))
        # 4 calls to primary/bad all 503, then 1 to fallback/good
        assert len(calls) >= 5
        assert any(c.choices[0].delta.content == "FALLBACK" for c in chunks)

    def test_payment_rejected_raises_payment_error(self, solana_client, monkeypatch):
        _patch_sse_helpers(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        calls: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            # Always 402, even after signing.
            return httpx.Response(
                402,
                headers={
                    "content-type": "application/json",
                    "payment-required": "stub-payment-required-base64",
                },
                json={"error": "Payment Required"},
            )

        solana_client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(PaymentError):
            list(solana_client.chat_completion_stream(
                "openai/gpt-5.5",
                [{"role": "user", "content": "hi"}],
            ))
