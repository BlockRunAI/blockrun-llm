"""``PAYMENT_REPLAY``: the gateway refused a payment as already used.

Ticket SDK2 (blockrun ``specs/2026-11-06-paid-path-alignment.md``). The gateway
sends ``code: "PAYMENT_REPLAY"`` when the signed authorization was submitted
before. The earlier request may already have been served and charged, so every
paid call must:

* raise :class:`PaymentReplayError` (not a "check your wallet balance"
  :class:`PaymentError`), carrying the gateway's ``message`` / ``details`` and
  any ``poll_url`` / ``job_id`` / ``wallet_auth`` verbatim;
* sign nothing new. A fresh authorization for the same request can be a second
  charge.

The bodies below are the gateway's literal shapes from blockrun
``src/lib/payment-nonce.ts`` (``replayResponse``, ``recoverClaimedJob``) and
``src/app/api/v1/chat/completions/route.ts``.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx
import pytest

from blockrun_llm import (
    PAYMENT_REPLAY,
    ImageClient,
    LLMClient,
    MusicClient,
    PaymentError,
    PaymentReplayError,
    SearchClient,
    VideoClient,
    cache,
)
from blockrun_llm.validation import build_payment_rejected_error, is_payment_replay

from ..helpers import TEST_ACCOUNT, TEST_PRIVATE_KEY, build_payment_required_response

# 402, replayResponse() with no earlier job, and the chat route's nonce claim.
ALREADY_USED = {
    "error": "Payment authorization already used",
    "message": (
        "This authorization was already submitted and may already have been charged. "
        "Check the transaction on chain before signing a new one."
    ),
    "code": "PAYMENT_REPLAY",
    "payer": TEST_ACCOUNT.address,
}

# 402, recoverClaimedJob(): a settled authorization presented on a job POST.
WALLET_AUTH = {
    "timestamp": "1762800000",
    "message": "blockrun.ai wants you to prove ownership of the wallet that paid job ...",
    "headers": ["X-Wallet-Address", "X-Wallet-Timestamp", "X-Wallet-Signature"],
    "expires_in": 300,
}
ALREADY_SETTLED = {
    "error": "Payment authorization already settled",
    "code": "PAYMENT_REPLAY",
    "message": (
        "This authorization has settled and belongs to an earlier request. To get its "
        "poll_url, sign wallet_auth.message with the paying wallet (EIP-191) and resend "
        "this request with the headers in wallet_auth. A new authorization starts a "
        "separate billable request."
    ),
    "wallet_auth": WALLET_AUTH,
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")
    monkeypatch.delenv("BLOCKRUN_API_KEY", raising=False)
    monkeypatch.setattr("time.sleep", lambda _s: None)


class Gateway:
    """402 on every unpaid request; scripted answers for each paid one.

    Records every request, so a test can count signatures and challenges.
    """

    def __init__(self, paid: list[httpx.Response]) -> None:
        self.paid = list(paid)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("PAYMENT-SIGNATURE") is None:
            return httpx.Response(
                402,
                headers={"payment-required": build_payment_required_response()},
                json={"error": "Payment Required"},
            )
        if not self.paid:
            raise AssertionError(f"unexpected paid request {request.method} {request.url}")
        return self.paid.pop(0)

    @property
    def signatures(self) -> list[str]:
        return [
            r.headers["PAYMENT-SIGNATURE"]
            for r in self.requests
            if "PAYMENT-SIGNATURE" in r.headers
        ]

    @property
    def challenges(self) -> int:
        return sum(1 for r in self.requests if "PAYMENT-SIGNATURE" not in r.headers)


def _install(client: Any, gw: Gateway) -> Any:
    client._client = httpx.Client(transport=httpx.MockTransport(gw))
    return client


def _assert_one_authorization(gw: Gateway) -> None:
    """One challenge, one signature. Re-sending the SAME header is allowed
    (it can settle at most once); a second, different signature is not."""
    assert gw.challenges == 1
    assert len(set(gw.signatures)) == 1


def _assert_replay(exc: PaymentError, body: dict[str, Any]) -> None:
    assert isinstance(exc, PaymentReplayError)
    assert exc.code == PAYMENT_REPLAY == "PAYMENT_REPLAY"
    assert exc.status_code == 402
    assert exc.server_message == body["message"]
    assert body["message"] in str(exc)
    assert "wallet balance" not in str(exc).lower()
    assert exc.response is not None
    assert exc.response["code"] == "PAYMENT_REPLAY"
    assert exc.response["message"] == body["message"]


def _job_202(kind: str = "videos") -> httpx.Response:
    return httpx.Response(
        202,
        json={
            "id": "job_1",
            "status": "queued",
            "poll_url": f"/api/v1/{kind}/generations/job_1",
        },
    )


Run = Callable[[Gateway], Any]


def _chat(gw: Gateway) -> Any:
    client = _install(LLMClient(private_key=TEST_PRIVATE_KEY, payment_scheme="exact"), gw)
    return client.chat_completion("openai/gpt-5.4", [{"role": "user", "content": "hi"}])


def _search(gw: Gateway) -> Any:
    return _install(SearchClient(private_key=TEST_PRIVATE_KEY), gw).search("x402")


def _image(gw: Gateway) -> Any:
    return _install(ImageClient(private_key=TEST_PRIVATE_KEY), gw).generate("a cat")


def _music(gw: Gateway) -> Any:
    return _install(MusicClient(private_key=TEST_PRIVATE_KEY), gw).generate("lo-fi")


def _video(gw: Gateway) -> Any:
    return _install(VideoClient(private_key=TEST_PRIVATE_KEY), gw).generate("a wave")


@pytest.mark.parametrize("body", [ALREADY_USED, ALREADY_SETTLED], ids=["used", "settled"])
@pytest.mark.parametrize(
    "run",
    [_chat, _search, _image, _music, _video],
    ids=["chat", "search", "image", "music", "video"],
)
def test_replay_on_the_paid_request_raises_without_signing_again(
    run: Run, body: dict[str, Any]
) -> None:
    gw = Gateway([httpx.Response(402, json=body)])
    with pytest.raises(PaymentReplayError) as info:
        run(gw)
    _assert_replay(info.value, body)
    _assert_one_authorization(gw)


@pytest.mark.parametrize(
    "run",
    [_chat, _search, _image, _music, _video],
    ids=["chat", "search", "image", "music", "video"],
)
def test_wallet_auth_challenge_is_carried_verbatim(run: Run) -> None:
    gw = Gateway([httpx.Response(402, json=ALREADY_SETTLED)])
    with pytest.raises(PaymentReplayError) as info:
        run(gw)
    assert info.value.wallet_auth == WALLET_AUTH
    assert info.value.response is not None
    assert info.value.response["wallet_auth"] == WALLET_AUTH


def test_chat_replay_after_a_same_header_retry_is_not_paid_again() -> None:
    """The realistic replay: a 502 makes the SDK re-send the SAME header, the
    first send had been served, and the gateway answers PAYMENT_REPLAY."""
    gw = Gateway(
        [httpx.Response(502, json={"error": "Bad gateway"}), httpx.Response(402, json=ALREADY_USED)]
    )
    with pytest.raises(PaymentReplayError) as info:
        _chat(gw)
    _assert_replay(info.value, ALREADY_USED)
    assert info.value.payer == TEST_ACCOUNT.address
    assert len(gw.signatures) == 2  # the same header, sent twice
    _assert_one_authorization(gw)


@pytest.mark.parametrize("kind", ["image", "music"])
def test_media_poll_replay_raises_without_signing_again(kind: str) -> None:
    path = "images" if kind == "image" else "audio"
    gw = Gateway([_job_202(path), httpx.Response(402, json=ALREADY_USED)])
    with pytest.raises(PaymentReplayError) as info:
        _image(gw) if kind == "image" else _music(gw)
    _assert_replay(info.value, ALREADY_USED)
    _assert_one_authorization(gw)


def test_video_mid_poll_replay_is_not_treated_as_signature_expiry() -> None:
    """A mid-poll 402 used to be read as "the 600s window lapsed": the SDK
    fetched a fresh challenge and signed again. On PAYMENT_REPLAY that fresh
    signature can be a second charge."""
    gw = Gateway([_job_202(), httpx.Response(402, json=ALREADY_USED)])
    with pytest.raises(PaymentReplayError) as info:
        _video(gw)
    _assert_replay(info.value, ALREADY_USED)
    _assert_one_authorization(gw)


def test_video_mid_poll_expiry_still_re_signs() -> None:
    """Control: a non-replay mid-poll 402 keeps the documented re-sign."""
    expired = httpx.Response(402, json={"error": "Payment verification failed"})
    done = httpx.Response(
        200,
        json={
            "id": "job_1",
            "status": "completed",
            "created": 1700000000,
            "model": "xai/grok-imagine-video",
            "data": [{"url": "https://x/v.mp4"}],
        },
    )
    gw = Gateway([_job_202(), expired, done])
    _video(gw)
    assert gw.challenges == 2
    assert len(set(gw.signatures)) == 2


def test_replay_recovery_fields_are_kept() -> None:
    body = {
        "status": "in_progress",
        "object": "generation.job",
        "message": "An earlier request owns this authorization and may still be running.",
        "code": "PAYMENT_REPLAY",
        "payer": TEST_ACCOUNT.address,
        "id": "job_9",
        "job_id": "job_9",
        "poll_url": "/api/v1/videos/generations/job_9",
        "recoverable": True,
    }
    exc = build_payment_rejected_error(httpx.Response(402, json=body))
    assert isinstance(exc, PaymentReplayError)
    assert exc.poll_url == "/api/v1/videos/generations/job_9"
    assert exc.job_id == "job_9"
    assert exc.recoverable is True
    assert "/api/v1/videos/generations/job_9" in str(exc)


def test_replay_details_are_kept() -> None:
    exc = build_payment_rejected_error(
        httpx.Response(402, json={**ALREADY_USED, "details": "nonce_already_used"})
    )
    assert isinstance(exc, PaymentReplayError)
    assert exc.details == "nonce_already_used"
    assert "nonce_already_used" in str(exc)


def test_is_payment_replay_reads_flat_and_nested_codes() -> None:
    assert is_payment_replay(ALREADY_USED)
    assert is_payment_replay({"error": {"code": "payment_replay"}})
    assert not is_payment_replay({"error": "Payment verification failed"})
    assert not is_payment_replay(None)


def test_a_non_replay_402_passes_the_gateway_reason_through() -> None:
    """PAYMENT_MALFORMED used to read as "check your wallet balance"."""
    body = {
        "error": "Malformed payment",
        "message": "accepted.amount does not match the quoted price",
        "code": "PAYMENT_MALFORMED",
    }
    gw = Gateway([httpx.Response(402, json=body)])
    with pytest.raises(PaymentError) as info:
        _search(gw)
    assert not isinstance(info.value, PaymentReplayError)
    assert "Malformed payment" in str(info.value)
    assert "accepted.amount does not match the quoted price" in str(info.value)
    assert "wallet balance" not in str(info.value).lower()
    assert info.value.response is not None
    assert info.value.response["code"] == "PAYMENT_MALFORMED"


def test_video_mid_poll_refusal_after_re_sign_passes_the_reason_through() -> None:
    refused = {"error": "Payment verification failed", "details": "insufficient_funds"}
    gw = Gateway([_job_202(), httpx.Response(402, json=refused), httpx.Response(402, json=refused)])
    client = _install(VideoClient(private_key=TEST_PRIVATE_KEY), gw)
    client.MAX_POLL_RESIGNS = 1
    with pytest.raises(PaymentError) as info:
        client.generate("a wave")
    assert not isinstance(info.value, PaymentReplayError)
    assert "insufficient_funds" in str(info.value)


def test_solana_never_re_signs_a_replay_that_names_the_earlier_result() -> None:
    """sol's bare "Sign a new payment for each request" replay is a concurrent
    nonce collision on ANOTHER request, and stays re-signable. A replay naming
    the earlier request's result (poll_url / wallet_auth) is this request,
    already owned and possibly charged: never re-sign it."""
    from blockrun_llm.solana_client import _is_safe_resign_error

    sol_bare = {
        "error": "Payment authorization already used",
        "message": "This payment signature was already redeemed. Sign a new payment for each request.",
        "code": "PAYMENT_REPLAY",
    }
    assert _is_safe_resign_error(build_payment_rejected_error(httpx.Response(402, json=sol_bare)))
    for extra in (
        {"poll_url": "/api/v1/videos/generations/j"},
        {"job_id": "j"},
        {"recoverable": True},
        {"wallet_auth": WALLET_AUTH},
    ):
        exc = build_payment_rejected_error(httpx.Response(402, json={**sol_bare, **extra}))
        assert isinstance(exc, PaymentReplayError)
        assert _is_safe_resign_error(exc) is False
