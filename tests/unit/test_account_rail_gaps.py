"""The account rail's remaining edges: wallet-keyed listings and Solana polling.

Every case here is a place where an API-key client used to reach code that
assumes a wallet exists. The failures were never wrong answers — they were
`AttributeError: 'NoneType' object has no attribute 'address'`, a 402 reported
as a generic HTTP error, and a poll loop pointed at a URL the gateway answers
with `wrong_host`. All three look like SDK bugs to the caller, which is exactly
what the account rail is not supposed to feel like.
"""

from __future__ import annotations

import httpx
import pytest

from blockrun_llm import PortraitClient, RealFaceClient, solana_client
from blockrun_llm.apikey import DEFAULT_API_KEY_URL
from blockrun_llm.types import APIError, PaymentError

KEY = "brk_live_account_rail_fixture"
WALLET = "0x" + "01" * 40
# Throwaway base58 keypair (seed = bytes(range(32))), only ever used to reach
# the wallet-rail branch of _absolute_url. Never funded, never signs anything.
_SOLANA_KEY = (
    "1GMkH3brNXiNNs1tiFZHu4yZSRrzJwxi5wB9bHFtMikjwpAW9DMZzU2Pqakc5it8X3N5vPmqdN7KF4CCUpmKhq"
)


def _mock(client: PortraitClient | RealFaceClient, status: int) -> None:
    client._client.close()
    client._client = httpx.Client(
        headers=client._client.headers,
        transport=httpx.MockTransport(lambda r: httpx.Response(status, json={"error": "fixture"})),
    )


@pytest.mark.parametrize(
    "cls,method",
    [(PortraitClient, "list_portraits"), (RealFaceClient, "list_realfaces")],
)
class TestWalletKeyedListings:
    """These endpoints are keyed by wallet address, which the account rail has
    none of. Both classes have to say so the same way — the pair drifted once
    already, with only one of them growing the 402 handling."""

    def test_no_address_argument_names_the_helper(self, monkeypatch, cls, method):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        client = cls()
        try:
            with pytest.raises(ValueError, match=f"{method}\\(\\) is wallet-only"):
                getattr(client, method)()
        finally:
            client.close()

    def test_402_is_a_credit_refusal_not_a_generic_http_error(self, monkeypatch, cls, method):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        client = cls()
        _mock(client, 402)
        try:
            with pytest.raises(PaymentError, match="no credit left"):
                getattr(client, method)(WALLET)
        finally:
            client.close()

    def test_other_failures_stay_api_errors(self, monkeypatch, cls, method):
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        client = cls()
        _mock(client, 500)
        try:
            with pytest.raises(APIError) as failure:
                getattr(client, method)(WALLET)
            assert failure.value.status_code == 500
        finally:
            client.close()

    def test_wallet_rail_still_defaults_to_its_own_address(self, monkeypatch, cls, method):
        monkeypatch.delenv("BLOCKRUN_API_KEY", raising=False)
        seen = []
        client = cls(private_key="0x" + "ac" * 32)
        client._client.close()

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(500, json={"error": "fixture"})

        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(APIError):
                getattr(client, method)()
            assert client.account.address in seen[0]
        finally:
            client.close()


@pytest.mark.parametrize(
    "client_class", [solana_client.SolanaLLMClient, solana_client.AsyncSolanaLLMClient]
)
class TestSolanaAccountPolling:
    def test_poll_url_drops_the_gateway_api_prefix(self, monkeypatch, client_class):
        """The gateway mints /api/v1/... against its own host. api.blockrun.ai
        serves that route at /v1/... and answers /api/v1/... with wrong_host, so
        leaving the prefix on makes every slow media job poll a dead URL until
        its budget runs out."""
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        client = client_class(private_key=KEY)
        resolved = client._absolute_url("/api/v1/images/generations/job_1")
        assert resolved == f"{DEFAULT_API_KEY_URL}/v1/images/generations/job_1"

    def test_poll_url_refuses_a_foreign_origin(self, monkeypatch, client_class):
        """The Authorization header rides on the client's default headers, so a
        gateway response pointing the poll loop elsewhere would hand the key to
        that host."""
        monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
        client = client_class(private_key=KEY)
        for hostile in ("https://elsewhere.example/x", "//elsewhere.example/x"):
            with pytest.raises(ValueError, match="origin"):
                client._absolute_url(hostile)

    def test_wallet_rail_poll_url_is_unchanged(self, monkeypatch, client_class):
        monkeypatch.delenv("BLOCKRUN_API_KEY", raising=False)
        pytest.importorskip("x402")
        client = client_class(private_key=_SOLANA_KEY)
        assert (
            client._absolute_url("/api/v1/images/generations/job_1")
            == "https://sol.blockrun.ai/api/v1/images/generations/job_1"
        )


def test_solana_account_402_on_the_media_probe_is_a_credit_refusal(monkeypatch):
    """Before the guard the probe fell into the x402 branch, which on this rail
    has no signer to reach for and — without the optional SDK installed — no
    decoder either."""
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    client = solana_client.SolanaLLMClient(private_key=KEY)
    client._client.close()
    client._client = httpx.Client(
        headers={"Authorization": f"Bearer {KEY}"},
        transport=httpx.MockTransport(
            lambda r: httpx.Response(402, json={"error": "insufficient_credit"})
        ),
    )
    try:
        with pytest.raises(PaymentError, match="no credit left"):
            client.image("a cat")
    finally:
        client.close()
