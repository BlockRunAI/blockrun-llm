"""Unit tests for RpcClient request construction and response parsing."""

import os
import httpx
import pytest

from blockrun_llm import RpcClient, RpcResponse, SUPPORTED_NETWORKS, NETWORK_ALIASES


@pytest.fixture
def client():
    # Deterministic dummy key — never actually signs against a live endpoint
    # in unit tests; we only exercise local request/response paths.
    os.environ.setdefault("BLOCKRUN_WALLET_KEY", "0x" + "11" * 32)
    return RpcClient()


def _headers(**extra):
    base = {"x-network": "ethereum", "x-cache": "MISS"}
    base.update(extra)
    return httpx.Headers(base)


def test_call_builds_jsonrpc_body(client, monkeypatch):
    captured = {}

    def fake_request(network, body):
        captured["network"] = network
        captured["body"] = body
        return {"jsonrpc": "2.0", "id": 1, "result": "0x10"}, _headers()

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    resp = client.call("ethereum", "eth_blockNumber")

    assert captured["network"] == "ethereum"
    assert captured["body"] == {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber"}
    assert resp.result == "0x10"
    assert resp.network == "ethereum"
    assert resp.cache_hit is False


def test_call_includes_params_and_custom_id(client, monkeypatch):
    captured = {}

    def fake_request(network, body):
        captured["body"] = body
        return {"jsonrpc": "2.0", "id": body["id"], "result": "0x0"}, _headers()

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    client.call("base", "eth_getBalance", ["0xabc", "latest"], id="bal-1")

    assert captured["body"] == {
        "jsonrpc": "2.0",
        "id": "bal-1",
        "method": "eth_getBalance",
        "params": ["0xabc", "latest"],
    }


def test_call_surfaces_cache_hit_and_tx_hash(client, monkeypatch):
    def fake_request(network, body):
        return (
            {"jsonrpc": "2.0", "id": 1, "result": "0x1"},
            _headers(**{"x-cache": "HIT", "x-payment-receipt": "0xdeadbeef"}),
        )

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    resp = client.call("ethereum", "eth_chainId")
    assert resp.cache_hit is True
    assert resp.tx_hash == "0xdeadbeef"


def test_call_parses_jsonrpc_error(client, monkeypatch):
    def fake_request(network, body):
        return (
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "no method"}},
            _headers(),
        )

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    resp = client.call("ethereum", "eth_bogus")
    assert resp.result is None
    assert resp.error is not None
    assert resp.error.code == -32601
    assert resp.error.message == "no method"


def test_batch_fills_jsonrpc_and_ids(client, monkeypatch):
    captured = {}

    def fake_request(network, body):
        captured["body"] = body
        return [
            {"jsonrpc": "2.0", "id": 1, "result": "0x10"},
            {"jsonrpc": "2.0", "id": 7, "result": "0x3b9aca00"},
        ], _headers()

    monkeypatch.setattr(client, "_request_with_payment", fake_request)

    out = client.batch(
        "polygon",
        [{"method": "eth_blockNumber"}, {"method": "eth_gasPrice", "id": 7}],
    )

    assert captured["body"] == [
        {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber"},
        {"jsonrpc": "2.0", "id": 7, "method": "eth_gasPrice"},
    ]
    assert len(out) == 2
    assert all(isinstance(r, RpcResponse) for r in out)
    assert out[1].id == 7


def test_batch_rejects_empty_and_missing_method(client):
    with pytest.raises(ValueError, match="at least one"):
        client.batch("ethereum", [])
    with pytest.raises(ValueError, match="missing 'method'"):
        client.batch("ethereum", [{"params": []}])


def test_network_registry_mirrors_backend():
    # 40 curated chains, 29 EVM + 11 non-EVM (backend src/lib/tatum.ts)
    assert len(SUPPORTED_NETWORKS) == 40
    assert len(SUPPORTED_NETWORKS) == len(set(SUPPORTED_NETWORKS))
    for must in ("ethereum", "base", "solana", "bitcoin", "ripple", "sui"):
        assert must in SUPPORTED_NETWORKS
    # Aliases resolve to curated keys
    for alias, canonical in NETWORK_ALIASES.items():
        assert canonical in SUPPORTED_NETWORKS, f"{alias} -> {canonical} not curated"
    assert NETWORK_ALIASES["xrpl"] == "ripple"
    assert NETWORK_ALIASES["sol"] == "solana"


def test_get_wallet_address(client):
    addr = client.get_wallet_address()
    assert addr.startswith("0x")
    assert len(addr) == 42
