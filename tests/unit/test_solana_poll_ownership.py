"""Solana media polls prove job ownership with an ed25519 signature, not a payment.

Server contract: blockrun-sol ``src/lib/poll-ownership.ts`` and the poll routes
under ``src/app/api/v1/{videos,audio,images}/generations/[id]``. The fake gateway
below enforces it the way those routes do: an x-poll-* proof must verify (with
an ed25519 implementation independent of the SDK's) over
``blockrun-poll:v1:<kind>:<job id>:<unix seconds>`` for the wallet that paid the
submit, or the poll gets the route's "no such job" answer.

The x402 codec and payment signer are stubbed as in test_solana_media.py; the
poll keypair is a real ``solders`` Keypair.
"""

from __future__ import annotations

import hashlib
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

import httpx
import pytest

pytest.importorskip("x402")
pytest.importorskip("solders")

from solders.keypair import Keypair

from blockrun_llm.poll_auth import (
    is_poll_ownership_refusal,
    poll_auth_headers,
    poll_auth_message,
    poll_job_id,
    poll_kind_for_url,
)
from blockrun_llm.solana_client import AsyncSolanaLLMClient, SolanaLLMClient
from blockrun_llm.types import APIError, MusicResponse, PaymentError

# ---------------------------------------------------------------------------
# An ed25519 verifier and base58 decoder that share no code with the SDK
# (RFC 8032 section 5.1.7, affine coordinates; slow but exact).
# ---------------------------------------------------------------------------

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _recover_x(y: int, sign: int) -> int | None:
    xx = (y * y - 1) * _inv(_D * y * y + 1) % _P
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - xx) % _P != 0:
        return None
    if x == 0 and sign:
        return None
    if x & 1 != sign:
        x = _P - x
    return x


def _add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    (x1, y1), (x2, y2) = p, q
    t = _D * x1 * x2 * y1 * y2 % _P
    return (
        (x1 * y2 + x2 * y1) * _inv(1 + t) % _P,
        (y1 * y2 + x1 * x2) * _inv(1 - t) % _P,
    )


def _mul(p: tuple[int, int], e: int) -> tuple[int, int]:
    q = (0, 1)
    while e:
        if e & 1:
            q = _add(q, p)
        p = _add(p, p)
        e >>= 1
    return q


_BY = 4 * _inv(5) % _P
_BASE = (_recover_x(_BY, 0) or 0, _BY)


def _decode_point(s: bytes) -> tuple[int, int] | None:
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= _P:
        return None
    x = _recover_x(y, sign)
    return None if x is None else (x, y)


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    a = _decode_point(public_key)
    r = _decode_point(signature[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(signature[:32] + public_key + message).digest(), "little")
    return _mul(_BASE, s) == _add(r, _mul(a, h % _L))


_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58decode(value: str) -> bytes:
    n = 0
    for ch in value:
        n = n * 58 + _B58.index(ch)
    body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return b"\0" * (len(value) - len(value.lstrip("1"))) + body


def proof_is_valid(
    headers: httpx.Headers, kind: str, job_id: str, wallet: str, *, skew_s: int = 300
) -> bool:
    """What the gateway checks: all three headers, a fresh timestamp, the
    submitting wallet, and an ed25519 signature over the exact message."""
    w = headers.get("x-poll-wallet")
    ts = headers.get("x-poll-timestamp")
    sig = headers.get("x-poll-signature")
    if not (w and ts and sig) or not ts.isdigit() or w != wallet:
        return False
    if abs(time.time() - int(ts)) > skew_s:
        return False
    message = f"blockrun-poll:v1:{kind}:{job_id}:{ts}".encode()
    return ed25519_verify(b58decode(w), message, b58decode(sig))


def has_proof(request: httpx.Request) -> bool:
    return any(h in request.headers for h in ("x-poll-wallet", "x-poll-signature"))


# ---------------------------------------------------------------------------
# The proof itself
# ---------------------------------------------------------------------------


class TestProof:
    def test_signature_verifies_over_the_exact_message(self) -> None:
        kp = Keypair()
        headers = httpx.Headers(poll_auth_headers(kp, "video", "job:abc/1", now_s=1_800_000_000))

        assert headers["x-poll-wallet"] == str(kp.pubkey())
        assert headers["x-poll-timestamp"] == "1800000000"
        pub, sig = b58decode(headers["x-poll-wallet"]), b58decode(headers["x-poll-signature"])
        assert len(pub) == 32 and len(sig) == 64
        assert ed25519_verify(pub, b"blockrun-poll:v1:video:job:abc/1:1800000000", sig)
        # The verifier is not vacuous: the same signature fails for any other
        # kind, job, timestamp, or version string.
        for other in (
            b"blockrun-poll:v1:image:job:abc/1:1800000000",
            b"blockrun-poll:v1:video:job:abc/2:1800000000",
            b"blockrun-poll:v1:video:job:abc/1:1800000001",
            b"blockrun-poll:v2:video:job:abc/1:1800000000",
        ):
            assert not ed25519_verify(pub, other, sig)

    def test_signature_matches_solders_for_a_fixed_key(self) -> None:
        # Deterministic ed25519: a fixed seed and message give fixed bytes.
        kp = Keypair.from_seed(bytes(range(32)))
        headers = poll_auth_headers(kp, "audio", "mus_1", now_s=1_700_000_000)
        assert poll_auth_message("audio", "mus_1", 1_700_000_000) == (
            "blockrun-poll:v1:audio:mus_1:1700000000"
        )
        assert ed25519_verify(
            bytes(kp.pubkey()),
            b"blockrun-poll:v1:audio:mus_1:1700000000",
            b58decode(headers["x-poll-signature"]),
        )

    def test_timestamp_defaults_to_now(self) -> None:
        headers = poll_auth_headers(Keypair(), "image", "img_1")
        assert abs(int(headers["x-poll-timestamp"]) - time.time()) <= 2

    @pytest.mark.parametrize(
        ("poll_url", "kind", "job_id"),
        [
            ("/api/v1/videos/generations/vid%2F1?duration=8&sig=abc", "video", "vid/1"),
            ("https://sol.blockrun.ai/api/v1/audio/generations/mus_9", "audio", "mus_9"),
            ("/api/v1/images/generations/img%3A7", "image", "img:7"),
            ("/v1/images/generations/img_2", "image", "img_2"),
        ],
    )
    def test_kind_and_job_id_from_poll_url(self, poll_url: str, kind: str, job_id: str) -> None:
        assert poll_kind_for_url(poll_url) == kind
        assert poll_job_id(poll_url) == job_id

    @pytest.mark.parametrize(
        "poll_url",
        ["/api/v1/audio/speech/x", "/api/v1/videos/x", "/api/v2/videos/generations/x", "/x"],
    )
    def test_other_paths_get_no_kind(self, poll_url: str) -> None:
        assert poll_kind_for_url(poll_url) is None

    def test_refusal_shapes(self) -> None:
        assert is_poll_ownership_refusal("video", 400, {"error": "Invalid job id"})
        assert is_poll_ownership_refusal("audio", 404, {"error": "Job not found"})
        assert is_poll_ownership_refusal("image", 404, {"error": "Job not found"})
        # A different error, or the right error on another route's status, is not one.
        assert not is_poll_ownership_refusal("video", 404, {"error": "Job not found"})
        assert not is_poll_ownership_refusal("video", 400, {"error": "Invalid duration"})
        assert not is_poll_ownership_refusal("image", 404, "Job not found")

    def test_real_wallet_key_gives_the_payer_keypair(self) -> None:
        kp = Keypair()
        with mock.patch("blockrun_llm.solana_client.register_exact_svm_client"):
            full = SolanaLLMClient(private_key=str(kp), rpc_url="http://test")
            seed_only = SolanaLLMClient(
                private_key=_b58encode(bytes(kp)[:32]), rpc_url="http://test"
            )
        assert full._poll_keypair.pubkey() == kp.pubkey()
        assert seed_only._poll_keypair.pubkey() == kp.pubkey()

    def test_account_rail_has_no_poll_keypair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLOCKRUN_API_KEY", "brk_live_testkey")
        assert SolanaLLMClient()._poll_keypair is None


def _b58encode(raw: bytes) -> str:
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    return "1" * (len(raw) - len(raw.lstrip(b"\0"))) + out


# ---------------------------------------------------------------------------
# Fake gateway
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_x402_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    # The challenge a payment is signed from travels through to the header the
    # client sends, so a test can tell WHICH 402 a payment answered.
    monkeypatch.setattr(
        "blockrun_llm.solana_client.decode_payment_required_header",
        lambda header: {"challenge": header},
    )
    monkeypatch.setattr(
        "blockrun_llm.solana_client.encode_payment_signature_header",
        lambda payload: f"paid-from:{payload.challenge['challenge']}",
    )


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("blockrun_llm.cache.get_cached", lambda *a, **k: None)
    monkeypatch.setattr("blockrun_llm.cache.save_to_cache", lambda *a, **k: None)


def _payload_for(challenge: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        accepted=SimpleNamespace(amount="1000000", pay_to="TREASURY"), challenge=challenge
    )


def _make_client(handler: Any, *, keypair: Keypair | None) -> SolanaLLMClient:
    with (
        mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
        mock.patch("blockrun_llm.solana_client._create_signer"),
    ):
        client = SolanaLLMClient(
            private_key="bogus_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
        )
    client._x402_client = mock.MagicMock()
    client._x402_client.create_payment_payload.side_effect = _payload_for
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client._address = "11111111111111111111111111111111"
    client._poll_keypair = keypair
    return client


def _make_async_client(handler: Any, *, keypair: Keypair | None) -> AsyncSolanaLLMClient:
    with (
        mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
        mock.patch("blockrun_llm.solana_client._create_signer"),
    ):
        client = AsyncSolanaLLMClient(
            private_key="bogus_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
        )
    client._x402_client = mock.MagicMock()
    client._x402_client.create_payment_payload = mock.AsyncMock(side_effect=_payload_for)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._address = "11111111111111111111111111111111"
    client._poll_keypair = keypair
    return client


def _signed_challenges(client: Any) -> list[str]:
    """Every challenge the wallet signed a payment from, in order."""
    return [
        c.args[0]["challenge"] for c in client._x402_client.create_payment_payload.call_args_list
    ]


def _json(status: int, body: dict[str, Any], **headers: str) -> httpx.Response:
    return httpx.Response(
        status, json=body, headers={"content-type": "application/json", **headers}
    )


_VIDEO_DONE = {
    "status": "completed",
    "created": 1,
    "model": "xai/grok-imagine-video",
    "data": [{"url": "https://cdn/v.mp4"}],
}


class FakeVideoGateway:
    """The sol video poll route: x-poll polls report status and never charge;
    a finished unpaid job answers 402 + challenge + ``status: completed``; a
    paid poll settles once; later polls answer ``already_settled``."""

    JOB = "vid/1"
    POLL_URL = "/api/v1/videos/generations/vid%2F1?duration=8&sig=abc"

    def __init__(
        self,
        wallet: str,
        *,
        pending: int = 2,
        refuse_proofs: bool = False,
        first_paid_poll: int = 200,
        fail_after_pending: bool = False,
        refuse_payments: bool = False,
    ) -> None:
        self.wallet = wallet
        self.pending = pending
        self.refuse_proofs = refuse_proofs
        self.first_paid_poll = first_paid_poll
        self.proof_polls: list[httpx.Request] = []
        self.paid_polls: list[httpx.Request] = []
        self.bare_polls: list[httpx.Request] = []
        self.settled = False
        self.claim_in_flight_once = first_paid_poll == 402
        self.fail_after_pending = fail_after_pending
        self.refuse_payments = refuse_payments

    def __call__(self, request: httpx.Request) -> httpx.Response:
        paid = "PAYMENT-SIGNATURE" in request.headers
        if request.method == "POST":
            if not paid:
                return _json(
                    402, {"error": "Payment Required"}, **{"payment-required": "submit-challenge"}
                )
            return _json(202, {"id": self.JOB, "poll_url": self.POLL_URL, "status": "queued"})

        # The id is sent percent-encoded, exactly as the POST minted it.
        assert request.url.raw_path.split(b"?")[0] == b"/api/v1/videos/generations/vid%2F1"
        if paid:
            assert not has_proof(request), "a paid poll carried an ownership proof too"
            self.paid_polls.append(request)
            if self.refuse_payments:
                return _json(
                    402,
                    {"error": "Payment verification failed", "details": "insufficient_funds"},
                )
            if self.claim_in_flight_once:
                # A concurrent poll holds the claim and then settles THIS
                # payment; the paid poll itself only hears "in progress".
                self.claim_in_flight_once = False
                self.settled = True
                return _json(402, {"error": "Payment settlement in progress"})
            # Legacy payment polls report progress until the job is done.
            if len(self.proof_polls) + len(self.paid_polls) <= self.pending:
                return _json(202, {"status": "in_progress", "payment_status": "verified"})
            self.settled = True
            return _json(200, _VIDEO_DONE, **{"x-payment-receipt": "SETTLE_TX"})

        if has_proof(request):
            self.proof_polls.append(request)
            if self.refuse_proofs or not proof_is_valid(
                request.headers, "video", self.JOB, self.wallet
            ):
                return _json(
                    400,
                    {
                        "error": "Invalid job id",
                        "details": "Pass the job id from POST back verbatim",
                    },
                )
            if len(self.proof_polls) <= self.pending:
                return _json(202, {"status": "in_progress", "payment_status": "not_charged"})
            if self.fail_after_pending:
                return _json(
                    200,
                    {"status": "failed", "error": "moderation", "payment_status": "not_charged"},
                )
            if self.settled:
                return _json(
                    200,
                    {**_VIDEO_DONE, "payment": {"status": "already_settled", "tx_hash": "TX"}},
                )
            return _json(
                402,
                {"status": "completed", "error": "Payment Required", "job_id": self.JOB},
                **{"payment-required": f"completion-challenge-{len(self.proof_polls)}"},
            )

        self.bare_polls.append(request)
        return _json(402, {"error": "Payment Required"}, **{"payment-required": "bare-challenge"})


_VIDEO_KW: dict[str, Any] = {
    "poll_budget_seconds": 5.0,
    "poll_interval_seconds": 0.001,
    "max_resigns": 2,
    "label": "Video generation",
}
_VIDEO_BODY = {"model": "xai/grok-imagine-video", "prompt": "a cat"}


# ---------------------------------------------------------------------------
# Video: status by proof, ONE payment at completion
# ---------------------------------------------------------------------------


class TestVideo:
    def test_pending_polls_carry_a_proof_and_completion_pays_once(self) -> None:
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=3)
        client = _make_client(gw, keypair=kp)

        data = client._request_image_with_payment(
            "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
        )

        assert data["data"][0]["url"] == "https://cdn/v.mp4"
        assert data["txHash"] == "SETTLE_TX"
        # 3 pending proofs + the one that found the clip finished and unpaid.
        assert len(gw.proof_polls) == 4
        for req in gw.proof_polls:
            assert "PAYMENT-SIGNATURE" not in req.headers
            assert proof_is_valid(req.headers, "video", "vid/1", str(kp.pubkey()))
        # Two payments in all: the submit, and one from the completion 402.
        assert _signed_challenges(client) == ["submit-challenge", "completion-challenge-4"]
        assert [r.headers["PAYMENT-SIGNATURE"] for r in gw.paid_polls] == [
            "paid-from:completion-challenge-4"
        ]
        assert gw.bare_polls == []
        # Booked once, for the clip.
        assert client._session_calls == 1
        assert client._session_total_usd == pytest.approx(1.0)

    def test_pending_polls_never_call_the_payment_signer(self) -> None:
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=6, fail_after_pending=True)
        client = _make_client(gw, keypair=kp)

        with pytest.raises(APIError, match="failed upstream: moderation"):
            client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
            )

        assert len(gw.proof_polls) == 7
        assert _signed_challenges(client) == ["submit-challenge"]
        assert gw.paid_polls == [] and gw.bare_polls == []
        assert client._session_calls == 0

    def test_settlement_in_flight_is_not_paid_twice(self) -> None:
        # The paid poll hears "settlement in progress"; the next proof poll
        # finds the job settled and delivers it without another payment.
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=1, first_paid_poll=402)
        client = _make_client(gw, keypair=kp)

        data = client._request_image_with_payment(
            "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
        )

        assert data["payment"]["status"] == "already_settled"
        assert _signed_challenges(client) == ["submit-challenge", "completion-challenge-2"]
        assert len(gw.paid_polls) == 1
        assert client._session_calls == 1

    def test_completion_payments_are_bounded_and_surface_the_settle_reason(self) -> None:
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=0, refuse_payments=True)
        client = _make_client(gw, keypair=kp)

        with pytest.raises(PaymentError, match="insufficient_funds"):
            client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
            )

        # 1 + max_resigns (2) completion payments, then stop.
        assert len(gw.paid_polls) == 3
        assert client._session_calls == 0

    def test_refused_proof_falls_back_once_to_payment_polls(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=2, refuse_proofs=True)
        client = _make_client(gw, keypair=kp)

        with caplog.at_level(logging.WARNING, logger="blockrun_llm.solana_client"):
            data = client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
            )

        assert data["data"][0]["url"] == "https://cdn/v.mp4"
        # One refused proof, then payment-signature polls only.
        assert len(gw.proof_polls) == 1
        assert len(gw.paid_polls) >= 1
        assert all(not has_proof(r) for r in gw.paid_polls)
        assert any("falling back to payment-signature polling" in m for m in caplog.messages)
        assert client._session_calls == 1

    def test_a_proof_from_another_wallet_is_refused_then_falls_back(self) -> None:
        # The gateway binds the job to the submit's payer; a proof by any other
        # key is "no such job", and the client recovers through payments.
        gw = FakeVideoGateway(str(Keypair().pubkey()), pending=0)
        client = _make_client(gw, keypair=Keypair())

        data = client._request_image_with_payment(
            "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
        )

        assert data["data"][0]["url"] == "https://cdn/v.mp4"
        assert len(gw.proof_polls) == 1 and len(gw.paid_polls) == 1

    def test_without_a_poll_keypair_polls_pay_as_before(self) -> None:
        gw = FakeVideoGateway("unused", pending=1)
        client = _make_client(gw, keypair=None)

        client._request_image_with_payment("/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW)

        assert gw.proof_polls == []
        assert len(gw.paid_polls) == 2

    def test_completion_challenge_that_reprices_is_refused(self) -> None:
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=0)
        client = _make_client(gw, keypair=kp)

        def reprice(challenge: dict[str, str]) -> SimpleNamespace:
            payload = _payload_for(challenge)
            if challenge["challenge"].startswith("completion"):
                payload.accepted.amount = "99000000"
            return payload

        client._x402_client.create_payment_payload.side_effect = reprice
        with pytest.raises(Exception, match="changed the payment terms"):
            client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
            )
        assert gw.paid_polls == []

    async def test_async_pending_proofs_then_one_payment(self) -> None:
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=3)
        client = _make_async_client(gw, keypair=kp)
        try:
            data = await client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
            )
        finally:
            await client._client.aclose()

        assert data["data"][0]["url"] == "https://cdn/v.mp4"
        assert len(gw.proof_polls) == 4
        assert all(
            proof_is_valid(r.headers, "video", "vid/1", str(kp.pubkey())) for r in gw.proof_polls
        )
        assert all("PAYMENT-SIGNATURE" not in r.headers for r in gw.proof_polls)
        assert _signed_challenges(client) == ["submit-challenge", "completion-challenge-4"]
        assert len(gw.paid_polls) == 1

    async def test_async_refused_proof_falls_back_once(self) -> None:
        kp = Keypair()
        gw = FakeVideoGateway(str(kp.pubkey()), pending=1, refuse_proofs=True)
        client = _make_async_client(gw, keypair=kp)
        try:
            data = await client._request_image_with_payment(
                "/v1/videos/generations", dict(_VIDEO_BODY), **_VIDEO_KW
            )
        finally:
            await client._client.aclose()

        assert data["data"][0]["url"] == "https://cdn/v.mp4"
        assert len(gw.proof_polls) == 1


# ---------------------------------------------------------------------------
# Image and music: paid at POST, polls never pay
# ---------------------------------------------------------------------------


class FakePaidAtPostGateway:
    """The sol image / music poll routes: the job was paid at POST, a proof
    poll returns status and results, a refused proof is 404 "Job not found"."""

    def __init__(
        self,
        kind: str,
        route: str,
        wallet: str,
        *,
        pending: int = 2,
        done: dict[str, Any],
        refuse_proofs: bool = False,
        batch_billed: bool = False,
    ) -> None:
        self.kind = kind
        self.route = route
        self.wallet = wallet
        self.pending = pending
        self.done = done
        self.refuse_proofs = refuse_proofs
        self.batch_billed = batch_billed
        self.proof_polls: list[httpx.Request] = []
        self.paid_polls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        paid = "PAYMENT-SIGNATURE" in request.headers
        if request.method == "POST":
            if not paid:
                return _json(
                    402, {"error": "Payment Required"}, **{"payment-required": "submit-challenge"}
                )
            return _json(
                202,
                {
                    "id": "job_1",
                    "poll_url": f"/api/v1/{self.route}/generations/job_1",
                    "status": "queued",
                },
                **{"x-payment-receipt": "SUBMIT_TX"},
            )
        assert request.url.path == f"/api/v1/{self.route}/generations/job_1"
        if paid:
            self.paid_polls.append(request)
            return _json(200, self.done)
        if has_proof(request):
            self.proof_polls.append(request)
            if self.refuse_proofs or not proof_is_valid(
                request.headers, self.kind, "job_1", self.wallet
            ):
                return _json(404, {"error": "Job not found"})
            if self.batch_billed:
                return _json(402, {"error": "Payment Required", "details": "batch payment"})
            if len(self.proof_polls) <= self.pending:
                return _json(202, {"status": "in_progress", "payment_status": "settled_optimistic"})
            return _json(200, self.done)
        return _json(402, {"error": "Payment Required"}, **{"payment-required": "bare-challenge"})


_IMAGE_DONE = {
    "status": "completed",
    "created": 1,
    "data": [{"url": "https://cdn/i.png"}],
}
_MUSIC_DONE = {
    "id": "job_1",
    "status": "completed",
    "created": 1,
    "model": "minimax/music-2.5+",
    "data": [{"url": "https://cdn/x.mp3", "duration_seconds": 30}],
    "payment": {"status": "settled_at_post"},
}
_IMAGE_KW: dict[str, Any] = {
    "poll_budget_seconds": 5.0,
    "poll_interval_seconds": 0.001,
    "max_resigns": 2,
    "label": "Image",
    "settled_at_submit": True,
}
_IMAGE_BODY = {"model": "openai/gpt-image-2", "prompt": "a cat"}


class TestImage:
    def test_proof_polls_never_pay(self) -> None:
        kp = Keypair()
        gw = FakePaidAtPostGateway("image", "images", str(kp.pubkey()), done=_IMAGE_DONE)
        client = _make_client(gw, keypair=kp)

        data = client._request_image_with_payment(
            "/v1/images/generations", dict(_IMAGE_BODY), **_IMAGE_KW
        )

        assert data["data"][0]["url"] == "https://cdn/i.png"
        assert len(gw.proof_polls) == 3 and gw.paid_polls == []
        assert all(
            proof_is_valid(r.headers, "image", "job_1", str(kp.pubkey())) for r in gw.proof_polls
        )
        assert _signed_challenges(client) == ["submit-challenge"]
        assert client._session_calls == 1  # the POST, booked once

    def test_batch_billed_402_falls_back_to_the_payment_path(self) -> None:
        kp = Keypair()
        gw = FakePaidAtPostGateway(
            "image", "images", str(kp.pubkey()), done=_IMAGE_DONE, batch_billed=True
        )
        client = _make_client(gw, keypair=kp)

        data = client._request_image_with_payment(
            "/v1/images/generations", dict(_IMAGE_BODY), **_IMAGE_KW
        )

        assert data["data"][0]["url"] == "https://cdn/i.png"
        assert len(gw.proof_polls) == 1 and len(gw.paid_polls) == 1

    def test_refused_proof_falls_back_once(self) -> None:
        kp = Keypair()
        gw = FakePaidAtPostGateway(
            "image", "images", str(kp.pubkey()), done=_IMAGE_DONE, refuse_proofs=True
        )
        client = _make_client(gw, keypair=kp)

        client._request_image_with_payment("/v1/images/generations", dict(_IMAGE_BODY), **_IMAGE_KW)

        assert len(gw.proof_polls) == 1 and len(gw.paid_polls) == 1

    async def test_async_proof_polls_never_pay(self) -> None:
        kp = Keypair()
        gw = FakePaidAtPostGateway("image", "images", str(kp.pubkey()), done=_IMAGE_DONE)
        client = _make_async_client(gw, keypair=kp)
        try:
            data = await client._request_image_with_payment(
                "/v1/images/generations", dict(_IMAGE_BODY), **_IMAGE_KW
            )
        finally:
            await client._client.aclose()

        assert data["data"][0]["url"] == "https://cdn/i.png"
        assert len(gw.proof_polls) == 3 and gw.paid_polls == []
        assert _signed_challenges(client) == ["submit-challenge"]


@pytest.fixture
def _fast_audio_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SolanaLLMClient, "AUDIO_POLL_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(SolanaLLMClient, "AUDIO_POLL_BUDGET_SECONDS", 5.0)


@pytest.mark.usefixtures("_fast_audio_polls")
class TestMusic:
    def test_slow_track_is_polled_with_proofs_and_never_pays(self) -> None:
        # Before this change the wallet rail returned the {id, poll_url} stub
        # and MusicResponse failed validation after the track was paid for.
        kp = Keypair()
        gw = FakePaidAtPostGateway("audio", "audio", str(kp.pubkey()), done=_MUSIC_DONE)
        client = _make_client(gw, keypair=kp)

        resp = client.music("lo-fi beats")

        assert isinstance(resp, MusicResponse)
        assert resp.data[0].url == "https://cdn/x.mp3"
        assert resp.txHash == "SUBMIT_TX"  # the POST's receipt, not a poll's
        assert len(gw.proof_polls) == 3 and gw.paid_polls == []
        assert all(
            proof_is_valid(r.headers, "audio", "job_1", str(kp.pubkey())) for r in gw.proof_polls
        )
        assert _signed_challenges(client) == ["submit-challenge"]
        assert client._session_calls == 1

    def test_refused_proof_falls_back_once(self, caplog: pytest.LogCaptureFixture) -> None:
        kp = Keypair()
        gw = FakePaidAtPostGateway(
            "audio", "audio", str(kp.pubkey()), done=_MUSIC_DONE, refuse_proofs=True
        )
        client = _make_client(gw, keypair=kp)

        with caplog.at_level(logging.WARNING, logger="blockrun_llm.solana_client"):
            resp = client.music("lo-fi beats")

        assert resp.data[0].url == "https://cdn/x.mp3"
        assert len(gw.proof_polls) == 1 and len(gw.paid_polls) == 1
        # The fallback poll carries a payment signed from the submit's own
        # challenge (verify-only on this route: never settled again).
        assert gw.paid_polls[0].headers["PAYMENT-SIGNATURE"] == "paid-from:submit-challenge"
        assert any("falling back" in m for m in caplog.messages)
        assert client._session_calls == 1

    async def test_async_slow_track_is_polled_with_proofs(self) -> None:
        kp = Keypair()
        gw = FakePaidAtPostGateway("audio", "audio", str(kp.pubkey()), done=_MUSIC_DONE)
        client = _make_async_client(gw, keypair=kp)
        try:
            resp = await client.music("lo-fi beats")
        finally:
            await client._client.aclose()

        assert resp.data[0].url == "https://cdn/x.mp3"
        assert resp.txHash == "SUBMIT_TX"
        assert len(gw.proof_polls) == 3 and gw.paid_polls == []
        assert _signed_challenges(client) == ["submit-challenge"]

    def test_inline_track_is_unchanged(self) -> None:
        kp = Keypair()

        def handler(request: httpx.Request) -> httpx.Response:
            if "PAYMENT-SIGNATURE" not in request.headers:
                return _json(402, {"error": "x"}, **{"payment-required": "submit-challenge"})
            return _json(200, _MUSIC_DONE)

        resp = _make_client(handler, keypair=kp).music("lo-fi beats")
        assert resp.data[0].url == "https://cdn/x.mp3"


# ---------------------------------------------------------------------------
# Rails that never send a proof
# ---------------------------------------------------------------------------


class TestOtherRails:
    def test_base_music_polls_keep_the_payment_signature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from blockrun_llm import MusicClient

        from ..helpers import TEST_PRIVATE_KEY, build_payment_required_response

        monkeypatch.setattr(MusicClient, "MUSIC_POLL_INTERVAL_SECONDS", 0.0)
        polls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                if "PAYMENT-SIGNATURE" not in request.headers:
                    return _json(
                        402,
                        {"error": "Payment Required"},
                        **{"payment-required": build_payment_required_response()},
                    )
                return _json(
                    202,
                    {"id": "m1", "status": "queued", "poll_url": "/api/v1/audio/generations/m1"},
                )
            polls.append(request)
            if len(polls) == 1:
                return _json(202, {"id": "m1", "status": "in_progress"})
            return _json(200, _MUSIC_DONE)

        client = MusicClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        client.generate("chill")

        assert len(polls) == 2
        assert all("PAYMENT-SIGNATURE" in r.headers and not has_proof(r) for r in polls)

    def test_account_rail_polls_carry_no_proof(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLOCKRUN_API_KEY", "brk_live_testkey")
        client = SolanaLLMClient()
        polls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return _json(
                    202,
                    {
                        "id": "job_1",
                        "status": "queued",
                        "poll_url": "/api/v1/images/generations/job_1",
                    },
                )
            polls.append(request)
            return _json(200, _IMAGE_DONE)

        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), headers=client._client.headers
        )
        client._request_image_with_payment("/v1/images/generations", dict(_IMAGE_BODY), **_IMAGE_KW)
        assert len(polls) == 1 and not has_proof(polls[0])
        assert "PAYMENT-SIGNATURE" not in polls[0].headers
