"""Unit tests for the DefiLlama / 0x DEX / Modal passthrough methods."""

import os
import pytest

from blockrun_llm import LLMClient


@pytest.fixture
def client():
    # Deterministic dummy key — never signs against a live endpoint in unit
    # tests; we only exercise local request/path construction.
    os.environ.setdefault("BLOCKRUN_WALLET_KEY", "0x" + "11" * 32)
    return LLMClient()


@pytest.fixture
def captured(client, monkeypatch):
    captured = {}

    def fake_get(endpoint, params=None):
        captured["method"] = "GET"
        captured["endpoint"] = endpoint
        captured["params"] = params
        return {"ok": True}

    def fake_post(endpoint, body):
        captured["method"] = "POST"
        captured["endpoint"] = endpoint
        captured["body"] = body
        return {"ok": True}

    monkeypatch.setattr(client, "_get_with_payment_raw", fake_get)
    monkeypatch.setattr(client, "_request_with_payment_raw", fake_post)
    return captured


# ── DefiLlama ────────────────────────────────────────────────────────────


def test_defi_generic_path_and_params(client, captured):
    client.defi("yields", chain="Base")
    assert captured["method"] == "GET"
    assert captured["endpoint"] == "/v1/defillama/yields"
    assert captured["params"] == {"chain": "Base"}


def test_defi_conveniences(client, captured):
    client.defi_protocols()
    assert captured["endpoint"] == "/v1/defillama/protocols"
    client.defi_protocol("aave")
    assert captured["endpoint"] == "/v1/defillama/protocol/aave"
    client.defi_chains()
    assert captured["endpoint"] == "/v1/defillama/chains"


def test_defi_prices_joins_coin_list(client, captured):
    client.defi_prices(["coingecko:bitcoin", "base:0xabc"])
    assert captured["endpoint"] == "/v1/defillama/prices/coingecko:bitcoin,base:0xabc"
    client.defi_prices("coingecko:ethereum")
    assert captured["endpoint"] == "/v1/defillama/prices/coingecko:ethereum"


# ── 0x DEX ───────────────────────────────────────────────────────────────


def test_dex_get_with_params(client, captured):
    client.dex_quote(chainId=8453, sellToken="0xa", buyToken="0xb", sellAmount="1000")
    assert captured["method"] == "GET"
    assert captured["endpoint"] == "/v1/zerox/quote"
    assert captured["params"]["chainId"] == 8453


def test_dex_gasless_submit_is_post(client, captured):
    client.dex_gasless_submit({"trade": {"signature": "0xsig"}})
    assert captured["method"] == "POST"
    assert captured["endpoint"] == "/v1/zerox/gasless/submit"
    assert captured["body"] == {"trade": {"signature": "0xsig"}}


def test_dex_gasless_status_embeds_hash(client, captured):
    client.dex_gasless_status("0xtradehash")
    assert captured["endpoint"] == "/v1/zerox/gasless/status/0xtradehash"


def test_dex_chain_discovery(client, captured):
    client.dex_chains()
    assert captured["endpoint"] == "/v1/zerox/swap/chains"
    client.dex_gasless_chains()
    assert captured["endpoint"] == "/v1/zerox/gasless/chains"


# ── Modal ────────────────────────────────────────────────────────────────


def test_modal_create_exec_lifecycle(client, captured):
    client.modal_sandbox_create(image="python:3.11", gpu="T4")
    assert captured["method"] == "POST"
    assert captured["endpoint"] == "/v1/modal/sandbox/create"
    assert captured["body"] == {"image": "python:3.11", "gpu": "T4"}

    client.modal_sandbox_exec("sb_123", ["python", "-c", "print(1)"])
    assert captured["endpoint"] == "/v1/modal/sandbox/exec"
    assert captured["body"]["sandbox_id"] == "sb_123"
    assert captured["body"]["command"] == ["python", "-c", "print(1)"]

    client.modal_sandbox_status("sb_123")
    assert captured["endpoint"] == "/v1/modal/sandbox/status"

    client.modal_sandbox_terminate("sb_123")
    assert captured["endpoint"] == "/v1/modal/sandbox/terminate"
    assert captured["body"] == {"sandbox_id": "sb_123"}
