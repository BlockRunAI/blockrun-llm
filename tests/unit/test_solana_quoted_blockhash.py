"""Solana payments sign on the blockhash the 402 quotes.

sol.blockrun.ai's Modal create 402 quotes ``extra.recentBlockhash`` with
``lastValidBlockHeight`` and an HMAC ``blockrunCreateProof``. The gateway
checks that a payment has enough blocks of validity left only when the signed
transaction's blockhash equals the proof's AND ``accepted.extra`` carries the
proof back. These tests drive the real client and the real ``x402`` signer
end to end; only the Solana RPC and the gateway are faked.

The quoted hash is a fresh random value distinct from the one the fake RPC
returns, so a client that fetches its own hash cannot pass.

Guarded with importorskip: the 3.9 CI job installs the SDK without the solana
extra.
"""

from __future__ import annotations

import base64
import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("x402")
pytest.importorskip("solders")

from solders.hash import Hash
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from x402.mechanisms.svm.exact.client import ExactSvmScheme

import blockrun_llm.cache as cache_mod
import blockrun_llm.solana_client as solana_client_mod
from blockrun_llm.solana_client import AsyncSolanaLLMClient, SolanaLLMClient

TEST_BS58_KEY = (
    "433C7KFcM4y1ZEVdZYSH7wheSNAM384UcbgXEyD5FV7Q2HsQ1BwjEDx4GbBZUqPkZTVhFPyLyuZnzK8wCeAkU7wG"
)
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
RESOURCE = "https://sol.blockrun.ai/api/v1/modal/sandbox/create"
FEE_PAYER = str(Keypair().pubkey())
PAY_TO = str(Keypair().pubkey())


def _new_hash() -> str:
    # A blockhash is 32 base58 bytes, the same shape as a pubkey.
    return str(Hash.from_bytes(bytes(Keypair().pubkey())))


class _FakeRpc:
    """Stands in for the SVM scheme's RPC client: mint lookup and blockhash."""

    def __init__(self, blockhash: str) -> None:
        self.blockhash = blockhash
        self.blockhash_calls = 0

    def get_account_info(self, _mint: Pubkey) -> Any:
        data = bytes(44) + bytes([6]) + bytes(37)
        return SimpleNamespace(value=SimpleNamespace(owner=TOKEN_PROGRAM, data=data))

    def get_latest_blockhash(self, *_args: Any, **_kwargs: Any) -> Any:
        self.blockhash_calls += 1
        return SimpleNamespace(
            value=SimpleNamespace(
                blockhash=Hash.from_string(self.blockhash), last_valid_block_height=1
            )
        )


def _modal_402(extra: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        402,
        json={
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": NETWORK,
                    "amount": "10000",
                    "asset": USDC,
                    "payTo": PAY_TO,
                    "maxTimeoutSeconds": 300,
                    "extra": extra,
                }
            ],
            "resource": {
                "url": RESOURCE,
                "description": "Create a sandbox",
                "mimeType": "application/json",
            },
            "error": "Payment Required",
        },
    )


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """A wallet-mode client whose RPC and gateway are fakes.

    Returns ``(client, rpc, payments)``: ``payments`` collects every decoded
    ``PAYMENT-SIGNATURE`` the client sent.
    """
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache_mod, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")
    rpc = _FakeRpc(_new_hash())
    monkeypatch.setattr(ExactSvmScheme, "_get_client", lambda self, network: rpc)

    payments: list[dict[str, Any]] = []
    state: dict[str, Any] = {"extra": None}

    def handler(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("PAYMENT-SIGNATURE")
        if header is None:
            return _modal_402(state["extra"])
        payments.append(json.loads(base64.b64decode(header)))
        return httpx.Response(200, json={"sandbox_id": f"sb-{len(payments)}"})

    client = SolanaLLMClient(private_key=TEST_BS58_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, rpc, payments, state


def _signed_blockhash(payment: dict[str, Any]) -> str:
    tx = VersionedTransaction.from_bytes(base64.b64decode(payment["payload"]["transaction"]))
    return str(tx.message.recent_blockhash)


def test_signs_on_the_quoted_blockhash_and_echoes_the_proof(harness):
    client, rpc, payments, state = harness
    quoted = _new_hash()
    assert quoted != rpc.blockhash
    extra = {
        "feePayer": FEE_PAYER,
        "description": "Modal API request",
        "mimeType": "application/json",
        "resource": RESOURCE,
        "recentBlockhash": quoted,
        "lastValidBlockHeight": "447122559",
        "blockrunCreateProof": "eyJibG9ja2hhc2giOiJ4In0."
        "d40a8275c6d49e4fce6060d2787fc4fb14f97da3974b2a0846a027917e35a34b",
    }
    state["extra"] = extra

    client.modal_sandbox_create(timeout=300)

    assert len(payments) == 1
    payment = payments[0]
    assert _signed_blockhash(payment) == quoted
    assert rpc.blockhash_calls == 0
    accepted = payment["accepted"]
    assert accepted["extra"] == extra
    assert accepted["amount"] == "10000"
    assert accepted["payTo"] == PAY_TO


async def test_async_client_signs_on_the_quoted_blockhash(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache_mod, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")
    rpc = _FakeRpc(_new_hash())
    monkeypatch.setattr(ExactSvmScheme, "_get_client", lambda self, network: rpc)
    quoted = _new_hash()
    extra = {"feePayer": FEE_PAYER, "recentBlockhash": quoted, "lastValidBlockHeight": "1"}
    payments: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("PAYMENT-SIGNATURE")
        if header is None:
            return _modal_402(extra)
        payments.append(json.loads(base64.b64decode(header)))
        return httpx.Response(200, json={"sandbox_id": "sb-async"})

    client = AsyncSolanaLLMClient(private_key=TEST_BS58_KEY)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.modal_sandbox_create(timeout=300)

    assert len(payments) == 1
    assert _signed_blockhash(payments[0]) == quoted
    assert rpc.blockhash_calls == 0
    assert payments[0]["accepted"]["extra"] == extra


def test_falls_back_to_its_own_blockhash_when_the_402_quotes_none(harness):
    client, rpc, payments, state = harness
    state["extra"] = {"feePayer": FEE_PAYER}

    client.modal_sandbox_create(timeout=300)

    assert len(payments) == 1
    assert _signed_blockhash(payments[0]) == rpc.blockhash
    assert rpc.blockhash_calls == 1


def test_falls_back_when_the_quote_is_not_a_blockhash(harness):
    client, rpc, payments, state = harness
    state["extra"] = {"feePayer": FEE_PAYER, "recentBlockhash": "not-a-blockhash"}

    client.modal_sandbox_create(timeout=300)

    assert _signed_blockhash(payments[0]) == rpc.blockhash
    assert rpc.blockhash_calls == 1


def test_installed_x402_signs_on_the_quote():
    """The declared floor is what CI installs at least; a release that drops
    the behaviour would also trip the client's runtime warning."""
    from x402.mechanisms.svm.exact import client as svm_exact_client

    assert hasattr(svm_exact_client, "resolve_blockhash")


def test_warns_once_when_x402_ignores_the_quote(monkeypatch, caplog):
    from x402.mechanisms.svm.exact import client as svm_exact_client

    monkeypatch.setattr(solana_client_mod, "_quoted_blockhash_warned", False)
    monkeypatch.delattr(svm_exact_client, "resolve_blockhash")
    with caplog.at_level(logging.WARNING, logger=solana_client_mod.logger.name):
        solana_client_mod._warn_if_x402_ignores_quoted_blockhash()
        solana_client_mod._warn_if_x402_ignores_quoted_blockhash()
    warnings = [r for r in caplog.records if "ignores the one the 402 quotes" in r.getMessage()]
    assert len(warnings) == 1


def test_no_warning_on_a_current_x402(monkeypatch, caplog):
    monkeypatch.setattr(solana_client_mod, "_quoted_blockhash_warned", False)
    with caplog.at_level(logging.WARNING, logger=solana_client_mod.logger.name):
        solana_client_mod._warn_if_x402_ignores_quoted_blockhash()
    assert not [r for r in caplog.records if "ignores the one the 402 quotes" in r.getMessage()]


# --- Media poll re-signs never reuse the challenge's quote -------------------
#
# A video's payment polls keep the signature fresh by re-signing the submit
# challenge every MEDIA_RESIGN_FRESH_SECONDS. sol quotes `recentBlockhash` on
# every 402 (x402-solana.ts create402ResponseParts), and x402 >= 2.17 signs on
# a quote, so re-signing the original challenge reproduced the same aging hash
# every time: the "fresh" signature died with the submit's blockhash.


def _video_job_handler(extra: dict[str, Any], poll_hashes: list[str], *, in_progress: int):
    state = {"polls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("PAYMENT-SIGNATURE")
        if request.method == "POST":
            if header is None:
                return _modal_402(extra)
            return httpx.Response(
                202,
                json={
                    "id": "JOB",
                    "poll_url": "/api/v1/videos/generations/JOB",
                    "status": "queued",
                },
            )
        assert header is not None, "payment-signature polling expected"
        poll_hashes.append(_signed_blockhash(json.loads(base64.b64decode(header))))
        state["polls"] += 1
        if state["polls"] <= in_progress:
            return httpx.Response(202, json={"status": "in_progress"})
        return httpx.Response(
            200,
            json={"status": "completed", "created": 1, "data": [{"url": "https://cdn/v.mp4"}]},
        )

    return handler


_POLL_KW: dict[str, Any] = {
    "poll_budget_seconds": 5.0,
    "poll_interval_seconds": 0.001,
    "max_resigns": 2,
    "label": "Video generation",
}


def test_poll_resign_signs_on_a_fresh_blockhash_not_the_quote(harness, monkeypatch):
    client, rpc, _payments, _state = harness
    quoted = _new_hash()
    extra = {"feePayer": FEE_PAYER, "recentBlockhash": quoted, "lastValidBlockHeight": "1"}
    poll_hashes: list[str] = []
    client._client = httpx.Client(
        transport=httpx.MockTransport(_video_job_handler(extra, poll_hashes, in_progress=2))
    )
    client._poll_keypair = None  # a gateway without poll proofs: payment polls
    monkeypatch.setattr(SolanaLLMClient, "MEDIA_RESIGN_FRESH_SECONDS", 0.0)

    data = client._request_image_with_payment(
        "/v1/videos/generations", {"model": "xai/grok-imagine-video", "prompt": "a cat"}, **_POLL_KW
    )

    assert data["data"][0]["url"] == "https://cdn/v.mp4"
    assert len(poll_hashes) == 3
    # Every poll carried a re-signed payment, none of it on the submit's quote.
    assert quoted not in poll_hashes, poll_hashes
    assert set(poll_hashes) == {rpc.blockhash}


async def test_async_poll_resign_signs_on_a_fresh_blockhash_not_the_quote(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache_mod, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")
    rpc = _FakeRpc(_new_hash())
    monkeypatch.setattr(ExactSvmScheme, "_get_client", lambda self, network: rpc)
    monkeypatch.setattr(SolanaLLMClient, "MEDIA_RESIGN_FRESH_SECONDS", 0.0)
    quoted = _new_hash()
    extra = {"feePayer": FEE_PAYER, "recentBlockhash": quoted, "lastValidBlockHeight": "1"}
    poll_hashes: list[str] = []

    client = AsyncSolanaLLMClient(private_key=TEST_BS58_KEY)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_video_job_handler(extra, poll_hashes, in_progress=2))
    )
    client._poll_keypair = None

    await client._request_image_with_payment(
        "/v1/videos/generations", {"model": "xai/grok-imagine-video", "prompt": "a cat"}, **_POLL_KW
    )

    assert len(poll_hashes) == 3
    assert quoted not in poll_hashes, poll_hashes
    assert set(poll_hashes) == {rpc.blockhash}


def test_stripping_the_quote_keeps_the_payment_terms():
    from x402.http.utils import decode_payment_required_header

    response = _modal_402(
        {
            "feePayer": FEE_PAYER,
            "recentBlockhash": _new_hash(),
            "lastValidBlockHeight": "9",
            "blockrunCreateProof": "p",
        }
    )
    challenge = decode_payment_required_header(base64.b64encode(response.content).decode())
    stripped = solana_client_mod._without_quoted_blockhash(challenge)
    original, after = challenge.accepts[0], stripped.accepts[0]
    assert after.extra == {"feePayer": FEE_PAYER}
    assert (after.amount, after.pay_to, after.asset, after.network) == (
        original.amount,
        original.pay_to,
        original.asset,
        original.network,
    )
    # The original challenge is not mutated: the submit already signed on it.
    assert "recentBlockhash" in original.extra
