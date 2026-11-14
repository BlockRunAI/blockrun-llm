"""A paid Solana request that answered 5xx is never paid for a second time.

sol.blockrun.ai settles in parallel with the upstream call, so a request can be
charged and still answer 5xx (the gateway logs "CHARGED BUT REQUEST FAILED").
The paid leg re-sends the SAME signed payment once on a 5xx, and the gateway,
still holding that payment's nonce claim, answers the resend with its bare
``402 PAYMENT_REPLAY``. Up to 1.22.2 the whole-request retry read that replay
as a concurrent nonce collision and signed a NEW payment: two settlements for
one call. These tests drive the real client and the real ``x402`` signer
against a fake gateway that settles every distinct authorization it receives,
and count the settlements.

Guarded with importorskip: the 3.9 CI job installs the SDK without the solana
extra.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest

pytest.importorskip("x402")
pytest.importorskip("solders")

from solders.hash import Hash
from solders.keypair import Keypair
from x402.mechanisms.svm.exact.client import ExactSvmScheme

import blockrun_llm.cache as cache_mod
from blockrun_llm.client import _SETTLED_ATTR
from blockrun_llm.solana_client import (
    AsyncSolanaLLMClient,
    SolanaLLMClient,
    _is_safe_resign_error,
)
from blockrun_llm.types import APIError, PaymentReplayError

TEST_BS58_KEY = (
    "433C7KFcM4y1ZEVdZYSH7wheSNAM384UcbgXEyD5FV7Q2HsQ1BwjEDx4GbBZUqPkZTVhFPyLyuZnzK8wCeAkU7wG"
)
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
REQUEST_ID = "8f1c0e1e-5a52-4d7e-9a43-4f0b1d1b2c3d"

# sol's literal replay body (chat/completions route.ts and payment-nonce.ts).
BARE_REPLAY = {
    "error": "Payment authorization already used",
    "message": "This payment signature was already redeemed. Sign a new payment for each request.",
    "code": "PAYMENT_REPLAY",
}
# A 5xx that says nothing about the payment (load balancer, "all providers
# exhausted" after the parallel settle).
BARE_502 = (502, {"error": "Bad gateway"})
# sol's Fable 5.2 past-the-45s-bound answer: charged, with its receipt.
CHARGED_502 = (
    502,
    {
        "error": "Upstream failed after payment",
        "code": "CHARGED_UPSTREAM_FAILED",
        "payment_status": "charged",
        "tx_hash": "5settledTx",
        "message": "Your payment settled (tx 5settledTx) before the model failed.",
    },
)
# sol's pre-settle refusal (upstream cooldown): nothing charged, header unspent.
NOT_CHARGED_503 = (
    503,
    {
        "error": "Model temporarily rate-limited upstream",
        "code": "UPSTREAM_RATE_LIMITED",
        "payment_status": "not_charged",
    },
)

CHAT_OK = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "openai/gpt-5.5",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}
SSE_OK = (
    b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)
MESSAGES = [{"role": "user", "content": "hi"}]


class _FakeRpc:
    def get_account_info(self, _mint: Any) -> Any:
        data = bytes(44) + bytes([6]) + bytes(37)
        return SimpleNamespace(value=SimpleNamespace(owner=TOKEN_PROGRAM, data=data))

    def get_latest_blockhash(self, *_args: Any, **_kwargs: Any) -> Any:
        fresh = Hash.from_bytes(bytes(Keypair().pubkey()))
        return SimpleNamespace(value=SimpleNamespace(blockhash=fresh, last_valid_block_height=1))


class Gateway:
    """Settles every distinct authorization; the first one then fails 5xx.

    A resend of an authorization it has already seen gets ``replay``. When
    ``first_claim_released`` is set the first authorization is handed back
    (sol's ``releasePaymentNonce`` on a pre-settle refusal), so its resend is
    served instead of refused.
    """

    def __init__(
        self,
        first: tuple[int, dict[str, Any]],
        *,
        stream: bool = False,
        replay: dict[str, Any] = BARE_REPLAY,
        first_settles: bool = True,
        first_claim_released: bool = False,
    ) -> None:
        self.first = first
        self.stream = stream
        self.replay = replay
        self.first_settles = first_settles
        self.first_claim_released = first_claim_released
        self.signed: list[str] = []
        self.settled: list[str] = []
        self.paid_requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        header = request.headers.get("PAYMENT-SIGNATURE")
        if header is None:
            return self._challenge(request)
        self.paid_requests += 1
        if header in self.signed:
            if self.first_claim_released and header == self.signed[0]:
                self.first_claim_released = False
                return self._served(request, header)
            return httpx.Response(402, json=self.replay)
        self.signed.append(header)
        if len(self.signed) == 1:
            if self.first_settles:
                self.settled.append(header)
            status, body = self.first
            return httpx.Response(
                status, json=body, headers={"x-blockrun-gateway-request-id": REQUEST_ID}
            )
        return self._served(request, header)

    def _served(self, request: httpx.Request, header: str) -> httpx.Response:
        if header not in self.settled:
            self.settled.append(header)
        if self.stream:
            return httpx.Response(
                200, content=SSE_OK, headers={"content-type": "text/event-stream"}
            )
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=CHAT_OK)
        return httpx.Response(200, json={"ok": True})

    def _challenge(self, request: httpx.Request) -> httpx.Response:
        challenge = {
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": NETWORK,
                    "amount": "10000",
                    "asset": USDC,
                    "payTo": str(Keypair().pubkey()),
                    "maxTimeoutSeconds": 300,
                    "extra": {"feePayer": str(Keypair().pubkey())},
                }
            ],
            "resource": {
                "url": str(request.url),
                "description": "paid call",
                "mimeType": "application/json",
            },
            "error": "Payment Required",
        }
        return httpx.Response(402, json=challenge)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache_mod, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")
    monkeypatch.delenv("BLOCKRUN_API_KEY", raising=False)
    monkeypatch.setattr(ExactSvmScheme, "_get_client", lambda self, network: _FakeRpc())
    monkeypatch.setattr("time.sleep", lambda _s: None)
    real_sleep = asyncio.sleep

    async def _no_wait(_s: float, *a: Any, **k: Any) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _no_wait)


def _sync(gw: Gateway) -> SolanaLLMClient:
    client = SolanaLLMClient(private_key=TEST_BS58_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(gw))
    return client


def _async(gw: Gateway) -> AsyncSolanaLLMClient:
    client = AsyncSolanaLLMClient(private_key=TEST_BS58_KEY)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(gw))
    return client


async def _drain(agen: Any) -> list[Any]:
    return [chunk async for chunk in agen]


# (id, stream?, sync call, async call). Every whole-request retry loop the
# client has: non-stream chat, stream chat, raw POST, raw GET.
SITES: list[tuple[str, bool, Callable[[Any], Any], Callable[[Any], Any]]] = [
    (
        "chat",
        False,
        lambda c: c.chat_completion("openai/gpt-5.5", MESSAGES),
        lambda c: c.chat_completion("openai/gpt-5.5", MESSAGES),
    ),
    (
        "stream",
        True,
        lambda c: list(c.chat_completion_stream("openai/gpt-5.5", MESSAGES)),
        lambda c: _drain(c.chat_completion_stream("openai/gpt-5.5", MESSAGES)),
    ),
    (
        "raw-post",
        False,
        lambda c: c.modal_sandbox_exec("sb-1", ["date"]),
        lambda c: c.modal_sandbox_exec("sb-1", ["date"]),
    ),
    (
        "raw-get",
        False,
        lambda c: c._get_with_payment_raw("/v1/pm/markets"),
        lambda c: c._get_with_payment_raw("/v1/pm/markets"),
    ),
]
SITE_IDS = [s[0] for s in SITES]


def _run(is_async: bool, gw: Gateway, site: tuple) -> Any:
    _, _, sync_call, async_call = site
    if is_async:
        return asyncio.run(async_call(_async(gw)))
    return sync_call(_sync(gw))


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("site", SITES, ids=SITE_IDS)
def test_replay_after_a_settled_5xx_is_not_paid_again(is_async: bool, site: tuple) -> None:
    """The double charge: settled, 502, resend refused as a bare replay."""
    gw = Gateway(BARE_502, stream=site[1])
    with pytest.raises(PaymentReplayError) as caught:
        _run(is_async, gw, site)
    assert len(gw.signed) == 1, "a second payment was signed for one call"
    assert len(gw.settled) == 1
    assert gw.paid_requests == 2  # the paid POST and its one same-signature resend
    exc = caught.value
    assert getattr(exc, _SETTLED_ATTR, False) is True
    assert _is_safe_resign_error(exc) is False
    assert exc.response["prior_attempt"] == {
        "status_code": 502,
        "gateway_request_id": REQUEST_ID,
    }
    assert REQUEST_ID in str(exc)
    assert "may already have been charged" in str(exc)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("site", SITES, ids=SITE_IDS)
def test_any_refusal_of_the_resend_is_terminal(is_async: bool, site: tuple) -> None:
    """A verify rejection of the RESEND is pre-broadcast for the resend only.
    The first attempt may have landed, so it is not a licence to re-sign."""
    expired = {
        "error": "Payment verification failed",
        "code": "PAYMENT_INVALID",
        "reason": "expired_signature",
    }
    gw = Gateway(BARE_502, stream=site[1], replay=expired)
    with pytest.raises(Exception) as caught:
        _run(is_async, gw, site)
    assert len(gw.signed) == 1
    assert _is_safe_resign_error(caught.value) is False


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("site", SITES, ids=SITE_IDS)
def test_a_5xx_that_reports_the_charge_is_surfaced_not_resent(is_async: bool, site: tuple) -> None:
    """The gateway's own receipt reaches the caller; nothing is resent."""
    gw = Gateway(CHARGED_502, stream=site[1])
    with pytest.raises(APIError) as caught:
        _run(is_async, gw, site)
    assert len(gw.signed) == 1
    assert gw.paid_requests == 1
    assert caught.value.status_code == 502
    assert caught.value.response["tx_hash"] == "5settledTx"
    assert caught.value.response["payment_status"] == "charged"


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("site", SITES, ids=SITE_IDS)
def test_a_transient_5xx_the_gateway_never_charged_still_recovers(
    is_async: bool, site: tuple
) -> None:
    """Honest retry kept: the same signature resent once, served, paid once."""
    gw = Gateway(BARE_502, stream=site[1], first_settles=False, first_claim_released=True)
    _run(is_async, gw, site)
    assert len(gw.signed) == 1
    assert len(gw.settled) == 1
    assert gw.paid_requests == 2


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("site", SITES, ids=SITE_IDS)
def test_a_proven_not_charged_5xx_keeps_the_collision_resign(is_async: bool, site: tuple) -> None:
    """`payment_status: not_charged` proves the first attempt moved nothing, so
    a replay of the resend is the ordinary concurrent collision and is re-signed."""
    gw = Gateway(NOT_CHARGED_503, stream=site[1], first_settles=False)
    _run(is_async, gw, site)
    assert len(gw.signed) == 2
    assert len(gw.settled) == 1


def test_a_bare_replay_with_no_prior_5xx_is_still_resigned() -> None:
    """The documented concurrent-collision re-sign is untouched."""
    state = {"n": 0}
    inner = Gateway(BARE_502)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("PAYMENT-SIGNATURE") is None:
            return inner._challenge(request)
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(402, json=BARE_REPLAY)
        return httpx.Response(200, json={"ok": True})

    client = SolanaLLMClient(private_key=TEST_BS58_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.modal_sandbox_exec("sb-1", ["date"]) == {"ok": True}
    assert state["n"] == 2


async def test_async_stream_error_after_payment_is_an_api_error() -> None:
    """The async client aliased the sync staticmethod `_raise_stream_error` as a
    plain function, so every async stream error raised TypeError instead."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("PAYMENT-SIGNATURE") is None:
            return Gateway(BARE_502)._challenge(request)
        return httpx.Response(400, json={"error": "Bad request", "code": "INVALID_REQUEST"})

    client = AsyncSolanaLLMClient(private_key=TEST_BS58_KEY)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(APIError) as caught:
        await _drain(client.chat_completion_stream("openai/gpt-5.5", MESSAGES))
    assert caught.value.status_code == 400


def test_signed_payload_decodes() -> None:
    """Sanity for the harness: the client really signs distinct payments."""
    gw = Gateway(NOT_CHARGED_503, first_settles=False)
    _sync(gw).modal_sandbox_exec("sb-1", ["date"])
    decoded = [json.loads(base64.b64decode(h)) for h in gw.signed]
    assert len({d["payload"]["transaction"] for d in decoded}) == 2
