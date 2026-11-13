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
