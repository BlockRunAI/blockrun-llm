"""Exercise the public Anthropic wrapper without a wallet or a real upstream."""

import json

import httpx
import pytest

from blockrun_llm import AnthropicClient

pytest.importorskip("anthropic")

import blockrun_llm.anthropic_client as ac

KEY = "brk_live_synthetic_customer_test"
WALLET = "0x" + "01" * 32


def test_anthropic_account_never_loads_wallet_or_replays_failed_post(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    monkeypatch.setenv("BLOCKRUN_WALLET_KEY", "must-not-parse-this")
    calls = []

    def handle(_self, request):
        calls.append(request)
        return httpx.Response(500, json={"error": {"type": "api_error", "message": "failed"}})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    client = AnthropicClient()
    try:
        assert client.payment_mode == "apikey"
        import anthropic

        with pytest.raises(anthropic.InternalServerError):
            client.messages.create(
                model="anthropic/claude-sonnet-4.6",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert len(calls) == 1
        assert str(calls[0].url) == "https://api.blockrun.ai/v1/messages"
        assert calls[0].headers["x-api-key"] == KEY
        assert "payment-signature" not in calls[0].headers
    finally:
        client.close()


def test_anthropic_explicit_wallet_overrides_env_key(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    client = AnthropicClient(private_key=WALLET)
    try:
        assert client.payment_mode == "wallet"
        assert str(client.base_url) == "https://blockrun.ai/api/"
    finally:
        client.close()


@pytest.mark.parametrize("mode", ["wallet", "apikey"])
def test_no_rail_replays_a_settled_post(monkeypatch, mode):
    """One caller request must sign at most one payment.

    The x402 transport signs a *fresh* payment for every 402 it sees, so an SDK
    retry after the gateway already settled signs and settles again. At the
    Anthropic default of 2 that is three on-chain transfers for one
    messages.create(), which is why max_retries has to be pinned on both rails
    and not just the account one.
    """
    signed = []

    class Base(httpx.BaseTransport):
        def handle_request(self, request):
            if request.headers.get("PAYMENT-SIGNATURE") is None and mode == "wallet":
                body = {"x402Version": 2, "accepts": [{}]}
                return httpx.Response(
                    402, json=body, headers={"payment-required": json.dumps(body)}
                )
            signed.append(request)
            return httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}})

        def close(self):
            pass

    if mode == "wallet":
        monkeypatch.delenv("BLOCKRUN_API_KEY", raising=False)
        monkeypatch.setattr(ac, "parse_payment_required", json.loads)
        monkeypatch.setattr(
            ac,
            "extract_payment_details",
            lambda p: {"recipient": "0x1", "amount": "1000", "network": "eip155:8453"},
        )
        monkeypatch.setattr(ac, "create_payment_payload", lambda **kw: "signed-payload")
        client = AnthropicClient(private_key=WALLET)
    else:
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        client = AnthropicClient()

    try:
        assert client._client.max_retries == 0
        # Swap in the counting transport, keeping the x402 wrapper on the
        # wallet rail so a retry would really re-sign.
        inner = client._client._client
        if mode == "wallet":
            inner._transport = ac._BlockRunX402Transport(
                account=ac.Account.from_key(WALLET),
                api_url="https://blockrun.ai/api",
                base_transport=Base(),
            )
        else:
            inner._transport = Base()

        import anthropic

        with pytest.raises(anthropic.InternalServerError):
            client.messages.create(
                model="anthropic/claude-sonnet-4.6",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert len(signed) == 1, f"{mode} rail submitted {len(signed)} payments for one request"
    finally:
        client.close()


def test_max_retries_stays_an_explicit_caller_choice(monkeypatch):
    monkeypatch.delenv("BLOCKRUN_API_KEY", raising=False)
    client = AnthropicClient(private_key=WALLET, max_retries=3)
    try:
        assert client._client.max_retries == 3
    finally:
        client.close()
