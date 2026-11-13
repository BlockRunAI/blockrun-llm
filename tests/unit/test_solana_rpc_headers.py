"""Custom Solana RPC headers reach the RPC on a mainnet payment.

``SolanaLLMClient(rpc_url=..., rpc_headers=...)`` pre-registers a header-
carrying RPC client on the x402 SVM schemes. x402 2.x looks clients up by the
CAIP-2 id, and the pre-registered keys used to be legacy names only, so every
mainnet payment built a fresh header-less client: the API key never reached
the mint lookup or the blockhash fetch. This drives the real lookup path
against a local JSON-RPC server and checks the header on every RPC call.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import pytest

pytest.importorskip("x402")
pytest.importorskip("solana")

from solders.keypair import Keypair

import blockrun_llm.cache as cache_mod
from blockrun_llm.solana_client import AsyncSolanaLLMClient, SolanaLLMClient

SOL_KEY = "433C7KFcM4y1ZEVdZYSH7wheSNAM384UcbgXEyD5FV7Q2HsQ1BwjEDx4GbBZUqPkZTVhFPyLyuZnzK8wCeAkU7wG"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
FEE_PAYER = str(Keypair().pubkey())
PAY_TO = str(Keypair().pubkey())
BLOCKHASH = str(Keypair().pubkey())  # 32 base58 bytes, same shape as a hash


def _rpc_result(method: str) -> Any:
    if method == "getAccountInfo":
        data = base64.b64encode(bytes(44) + bytes([6]) + bytes(37)).decode()
        return {
            "context": {"slot": 1},
            "value": {
                "data": [data, "base64"],
                "executable": False,
                "lamports": 1461600,
                "owner": TOKEN_PROGRAM,
                "rentEpoch": 0,
                "space": 82,
            },
        }
    if method == "getLatestBlockhash":
        return {
            "context": {"slot": 1},
            "value": {"blockhash": BLOCKHASH, "lastValidBlockHeight": 100},
        }
    raise AssertionError(f"unexpected RPC method {method}")


@pytest.fixture
def rpc_server():
    """A local JSON-RPC server recording (method, x-api-key) per request."""
    seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((body["method"], self.headers.get("x-api-key")))
            out = json.dumps(
                {"jsonrpc": "2.0", "id": body.get("id"), "result": _rpc_result(body["method"])}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *_args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()
    server.server_close()


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache_mod, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")


def _gateway(payments: list[dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        sig = request.headers.get("PAYMENT-SIGNATURE")
        if sig is None:
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
                            # No recentBlockhash: the client fetches its own.
                            "extra": {"feePayer": FEE_PAYER},
                        }
                    ],
                    "resource": {"url": "https://sol.blockrun.ai/api/v1/modal/sandbox/create"},
                },
            )
        payments.append(json.loads(base64.b64decode(sig)))
        return httpx.Response(200, json={"sandbox_id": "sb-1"})

    return handler


def _assert_headers_reached_rpc(seen: list[tuple[str, str | None]]) -> None:
    methods = [m for m, _ in seen]
    assert "getAccountInfo" in methods and "getLatestBlockhash" in methods
    assert all(key == "secret-key" for _, key in seen), seen


def test_headers_reach_mint_lookup_and_blockhash_fetch(rpc_server, isolated_cache):
    url, seen = rpc_server
    payments: list[dict[str, Any]] = []
    client = SolanaLLMClient(
        private_key=SOL_KEY, rpc_url=url, rpc_headers={"x-api-key": "secret-key"}
    )
    client._client = httpx.Client(transport=httpx.MockTransport(_gateway(payments)))

    client.modal_sandbox_create(timeout=300)

    assert len(payments) == 1
    _assert_headers_reached_rpc(seen)


async def test_async_client_headers_reach_the_rpc(rpc_server, isolated_cache):
    url, seen = rpc_server
    payments: list[dict[str, Any]] = []
    client = AsyncSolanaLLMClient(
        private_key=SOL_KEY, rpc_url=url, rpc_headers={"x-api-key": "secret-key"}
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_gateway(payments)))

    await client.modal_sandbox_create(timeout=300)

    assert len(payments) == 1
    _assert_headers_reached_rpc(seen)
