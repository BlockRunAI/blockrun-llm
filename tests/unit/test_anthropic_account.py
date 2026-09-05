"""Exercise the public Anthropic wrapper without a wallet or a real upstream."""

import httpx
import pytest

from blockrun_llm import AnthropicClient

pytest.importorskip("anthropic")

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
