"""Paid calls with side effects or live state are never served from cache.

The local cache answers a repeated (endpoint, body) without calling the
gateway. That is right for read-only data and wrong for anything else: a
second identical Modal ``exec`` returned the first run's output with no
gateway call and no payment, and a trade-status poll would never change.
Unknown endpoints now default to no cache; read-only families are listed.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

import blockrun_llm.cache as cache_mod
from blockrun_llm import LLMClient

from ..helpers import TEST_PRIVATE_KEY, build_payment_required_response

STATEFUL = [
    ("/v1/modal/sandbox/create", {"timeout": 300}),
    ("/v1/modal/sandbox/exec", {"sandbox_id": "sb-1", "command": ["date"]}),
    ("/v1/modal/sandbox/status", {"sandbox_id": "sb-1"}),
    ("/v1/modal/sandbox/terminate", {"sandbox_id": "sb-1"}),
    ("/v1/zerox/gasless/submit", {"trade": {"type": "settler_metatransaction"}}),
    ("/v1/zerox/gasless/status/0xabc", {}),
    ("/v1/zerox/swap/allowance-holder/quote", {"sellToken": "USDC"}),
    ("/v1/onramp/token", {"address": "0x1", "network": "base", "asset": "USDC"}),
    ("/v1/some-future-route", {"x": 1}),
]

READ_ONLY = [
    ("/v1/exa/search", {"query": "q"}),
    ("/v1/defillama/protocols", {}),
    ("/v1/pm/markets", {}),
    ("/v1/search", {"query": "q"}),
    ("/v1/partner/x/user", {"username": "a"}),
]


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache_mod, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")
    return tmp_path


@pytest.mark.parametrize("endpoint,body", STATEFUL)
def test_stateful_paid_response_is_never_a_cache_hit(isolated_cache, endpoint, body):
    cache_mod.save_to_cache(endpoint, body, {"result": "first"}, cost_usd=0.01)
    assert cache_mod.get_cached(endpoint, body) is None


@pytest.mark.parametrize("endpoint,body", READ_ONLY)
def test_read_only_paid_response_is_still_cached(isolated_cache, endpoint, body):
    cache_mod.save_to_cache(endpoint, body, {"result": "first"}, cost_usd=0.01)
    assert cache_mod.get_cached(endpoint, body) == {"result": "first"}


def test_stateful_paid_call_still_logs_its_cost(isolated_cache):
    """Not caching must not drop the cost-log row."""
    cache_mod.save_to_cache("/v1/modal/sandbox/exec", {"a": 1}, {"ok": True}, cost_usd=0.01)
    rows = (isolated_cache / "cost_log.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["cost_usd"] == 0.01


def _base_gateway(payments: list[str]):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        sig = request.headers.get("PAYMENT-SIGNATURE")
        if sig is None:
            return httpx.Response(
                402,
                headers={"payment-required": build_payment_required_response(amount="10000")},
                json={"error": "Payment Required"},
            )
        payments.append(sig)
        return httpx.Response(200, json={"stdout": f"run {len(payments)}"})

    return handler, calls


def test_base_client_pays_and_calls_for_each_identical_modal_exec(isolated_cache):
    payments: list[str] = []
    handler, calls = _base_gateway(payments)
    client = LLMClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    first = client.modal_sandbox_exec("sb-1", ["date"])
    second = client.modal_sandbox_exec("sb-1", ["date"])

    assert len(payments) == 2
    assert calls["n"] == 4  # probe + paid, twice
    assert first["stdout"] == "run 1" and second["stdout"] == "run 2"


# ── Solana ─────────────────────────────────────────────────────────────────

# The Solana half needs the solana extra, which the 3.9 CI job does not
# install; skip it there without skipping the Base tests above.
try:
    from solders.hash import Hash
    from solders.keypair import Keypair
    from x402.mechanisms.svm.exact.client import ExactSvmScheme

    from blockrun_llm.solana_client import AsyncSolanaLLMClient, SolanaLLMClient

    HAS_SOLANA = True
except ImportError:
    HAS_SOLANA = False

needs_solana = pytest.mark.skipif(not HAS_SOLANA, reason="solana extra not installed")

SOL_KEY = "433C7KFcM4y1ZEVdZYSH7wheSNAM384UcbgXEyD5FV7Q2HsQ1BwjEDx4GbBZUqPkZTVhFPyLyuZnzK8wCeAkU7wG"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
FEE_PAYER = str(Keypair().pubkey()) if HAS_SOLANA else ""
PAY_TO = str(Keypair().pubkey()) if HAS_SOLANA else ""


class _FakeRpc:
    def get_account_info(self, _mint: Any) -> Any:
        from types import SimpleNamespace

        data = bytes(44) + bytes([6]) + bytes(37)
        return SimpleNamespace(value=SimpleNamespace(owner=TOKEN_PROGRAM, data=data))

    def get_latest_blockhash(self, *_a: Any, **_k: Any) -> Any:
        from types import SimpleNamespace

        h = Hash.from_bytes(bytes(Keypair().pubkey()))
        return SimpleNamespace(value=SimpleNamespace(blockhash=h, last_valid_block_height=1))


def _sol_402() -> httpx.Response:
    return httpx.Response(
        402,
        json={
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
                    "amount": "10000",
                    "asset": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "payTo": PAY_TO,
                    "maxTimeoutSeconds": 300,
                    "extra": {"feePayer": FEE_PAYER},
                }
            ],
            "resource": {"url": "https://sol.blockrun.ai/api/v1/modal/sandbox/exec"},
        },
    )


def _sol_gateway(payments: list[dict[str, Any]]):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        sig = request.headers.get("PAYMENT-SIGNATURE")
        if sig is None:
            return _sol_402()
        payments.append(json.loads(base64.b64decode(sig)))
        return httpx.Response(200, json={"stdout": f"run {len(payments)}"})

    return handler, calls


@needs_solana
def test_solana_client_pays_and_calls_for_each_identical_modal_exec(isolated_cache, monkeypatch):
    monkeypatch.setattr(ExactSvmScheme, "_get_client", lambda self, network: _FakeRpc())
    payments: list[dict[str, Any]] = []
    handler, calls = _sol_gateway(payments)
    client = SolanaLLMClient(private_key=SOL_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    first = client.modal_sandbox_exec("sb-1", ["date"])
    second = client.modal_sandbox_exec("sb-1", ["date"])

    assert len(payments) == 2
    assert calls["n"] == 4
    assert first["stdout"] == "run 1" and second["stdout"] == "run 2"


@needs_solana
async def test_async_solana_client_pays_and_calls_for_each_identical_modal_exec(
    isolated_cache, monkeypatch
):
    monkeypatch.setattr(ExactSvmScheme, "_get_client", lambda self, network: _FakeRpc())
    payments: list[dict[str, Any]] = []
    handler, calls = _sol_gateway(payments)
    client = AsyncSolanaLLMClient(private_key=SOL_KEY)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    first = await client.modal_sandbox_exec("sb-1", ["date"])
    second = await client.modal_sandbox_exec("sb-1", ["date"])

    assert len(payments) == 2
    assert calls["n"] == 4
    assert first["stdout"] == "run 1" and second["stdout"] == "run 2"
