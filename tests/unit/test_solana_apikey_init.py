"""An account credential must never initialize either Solana signer."""

import pytest

from blockrun_llm import AsyncSolanaLLMClient, SolanaLLMClient, solana_client


@pytest.mark.parametrize("client_class", [SolanaLLMClient, AsyncSolanaLLMClient])
@pytest.mark.parametrize("explicit", [True, False])
def test_account_client_does_not_construct_signer(monkeypatch, client_class, explicit):
    key = "brk_live_AccountAcceptanceFixture123456"
    monkeypatch.setenv("BLOCKRUN_API_KEY", key)
    monkeypatch.setenv("SOLANA_WALLET_KEY", "invalid-leftover-wallet")

    def forbidden(*args, **kwargs):
        pytest.fail("API account initialized a wallet signer")

    monkeypatch.setattr(solana_client, "_create_signer", forbidden)
    client = client_class(private_key=key if explicit else None)
    assert client.payment_mode == "apikey"
    assert client._private_key is None
    assert client._client.headers["Authorization"] == f"Bearer {key}"
    assert client._x402_client is None
    if client_class is SolanaLLMClient:
        client.close()
    else:
        import asyncio

        asyncio.run(client.close())
