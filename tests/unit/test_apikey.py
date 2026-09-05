"""The API-key rail: precedence, routing, and the things it must refuse.

The precedence rule gets its own test class because it decides whether a call
spends prepaid credit or on-chain USDC, and that is not a difference anyone
wants to discover from an invoice.
"""

from __future__ import annotations

import httpx
import pytest

from blockrun_llm import LLMClient
from blockrun_llm.apikey import (
    DEFAULT_API_KEY_URL,
    ENV_API_KEY,
    ENV_API_KEY_URL,
    PAYMENT_MODE_API_KEY,
    PAYMENT_MODE_WALLET,
    api_key_base_url,
    auth_headers,
    is_api_key,
    resolve_api_key,
    resolve_poll_url,
)
from blockrun_llm.image import ImageClient
from blockrun_llm.types import PaymentError

API_KEY = "brk_live_TESTKEYTESTKEYTESTKEY"
WALLET_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
WALLET_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"


class TestIsAPIKey:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("brk_live_abc", True),
            ("  brk_test_abc  ", True),
            (WALLET_KEY, False),
            ("", False),
            ("sk-abc", False),
            (None, False),
        ],
    )
    def test_prefix(self, value, expected):
        assert is_api_key(value) is expected


class TestPrecedence:
    def test_explicit_key_wins_over_everything(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEY, "brk_live_fromenv")
        assert resolve_api_key(API_KEY) == API_KEY

    def test_explicit_wallet_key_opts_out_of_the_env_key(self, monkeypatch):
        """An explicit wallet key is a deliberate choice of the x402 rail."""
        monkeypatch.setenv(ENV_API_KEY, API_KEY)
        assert resolve_api_key(WALLET_KEY) is None

    def test_env_key_beats_the_wallet_env_vars(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEY, API_KEY)
        monkeypatch.setenv("BLOCKRUN_WALLET_KEY", WALLET_KEY)
        assert resolve_api_key(None) == API_KEY

    def test_no_key_anywhere(self):
        assert resolve_api_key(None) is None

    def test_a_non_brk_env_value_is_not_a_key(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEY, "not-a-key")
        with pytest.raises(ValueError, match="BLOCKRUN_API_KEY"):
            resolve_api_key(None)


class TestClientConstruction:
    def test_api_key_client(self):
        client = LLMClient(private_key=API_KEY)
        assert client.payment_mode == PAYMENT_MODE_API_KEY
        assert client.api_url == DEFAULT_API_KEY_URL
        assert client.account is None
        assert client.get_wallet_address() == ""

    def test_wallet_client_is_unchanged(self):
        """A wallet client must be exactly what it was before this feature."""
        client = LLMClient(private_key=WALLET_KEY)
        assert client.payment_mode == PAYMENT_MODE_WALLET
        assert client.api_url == LLMClient.DEFAULT_API_URL
        assert client.get_wallet_address() == WALLET_ADDRESS

    def test_env_key_beats_wallet_env(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEY, API_KEY)
        monkeypatch.setenv("BLOCKRUN_WALLET_KEY", WALLET_KEY)
        assert LLMClient().payment_mode == PAYMENT_MODE_API_KEY

    def test_x402_env_url_does_not_retarget_an_api_key_client(self, monkeypatch):
        """BLOCKRUN_API_URL names an x402 gateway. Following it would send the
        key to a host configured for a different rail."""
        monkeypatch.setenv("BLOCKRUN_API_URL", "https://private-x402.example.com/api")
        assert LLMClient(private_key=API_KEY).api_url == DEFAULT_API_KEY_URL

    def test_api_key_url_override(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEY_URL, "https://api.staging.example.com/")
        assert api_key_base_url(None) == "https://api.staging.example.com"

    def test_every_request_carries_the_key(self):
        client = LLMClient(private_key=API_KEY)
        assert client._client.headers.get("authorization") == f"Bearer {API_KEY}"

    def test_wallet_client_sends_no_authorization(self):
        client = LLMClient(private_key=WALLET_KEY)
        assert "authorization" not in client._client.headers


class TestAuthHeaders:
    def test_key(self):
        assert auth_headers("brk_live_x") == {"Authorization": "Bearer brk_live_x"}

    def test_no_key_is_empty_so_call_sites_can_be_unconditional(self):
        assert auth_headers(None) == {}


class TestPollURL:
    """``poll_url`` is minted by the x402 gateway relative to ITS host, so it
    arrives as ``/api/v1/...``. api.blockrun.ai serves that route at ``/v1/...``
    and answers ``/api/v1/...`` with ``wrong_host`` — an unstripped prefix is an
    async job polling a 404 until its budget runs out."""

    def test_account_rail_strips_the_api_prefix(self):
        got = resolve_poll_url(
            "/api/v1/images/generations/job_1", DEFAULT_API_KEY_URL, "brk_live_x"
        )
        assert got == f"{DEFAULT_API_KEY_URL}/v1/images/generations/job_1"

    def test_wallet_rail_keeps_it(self):
        got = resolve_poll_url("/api/v1/images/generations/job_1", "https://blockrun.ai/api", None)
        assert got == "https://blockrun.ai/api/v1/images/generations/job_1"

    @pytest.mark.parametrize("url", ["https://elsewhere.example/x", "//elsewhere.example/x"])
    def test_account_rejects_foreign_poll_origin(self, url):
        with pytest.raises(ValueError, match="origin"):
            resolve_poll_url(url, DEFAULT_API_KEY_URL, "brk_x")

    def test_same_origin_signed_url_preserves_query(self):
        url = DEFAULT_API_KEY_URL + "/v1/videos/generations/job?token=a%2Fb&signature=x"
        assert resolve_poll_url(url, DEFAULT_API_KEY_URL, "brk_x") == url


class TestRequests:
    """One request, key attached, no 402 round trip."""

    def test_chat_sends_bearer_and_skips_the_402_dance(self, monkeypatch):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["count"] = seen.get("count", 0) + 1
            seen["auth"] = request.headers.get("authorization")
            seen["path"] = request.url.path
            seen["payment_sig"] = request.headers.get("payment-signature")
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "openai/gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "4"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        client = LLMClient(private_key=API_KEY)
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), headers=auth_headers(API_KEY)
        )

        assert client.chat("openai/gpt-4o", "2+2?") == "4"
        assert seen["count"] == 1, "the account rail must not make a 402 round trip"
        assert seen["auth"] == f"Bearer {API_KEY}"
        # No "/api" inserted: the endpoint constants are already /v1/...
        assert seen["path"] == "/v1/chat/completions"
        assert seen["payment_sig"] is None

    def test_402_is_a_credit_refusal_not_a_challenge(self):
        """The old path would answer 'no wallet configured', which sends the
        reader hunting a wallet problem they do not have."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402,
                json={"error": {"type": "insufficient_quota", "code": "BALANCE_EXHAUSTED"}},
            )

        client = LLMClient(private_key=API_KEY)
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), headers=auth_headers(API_KEY)
        )

        with pytest.raises(PaymentError) as exc:
            client.chat("openai/gpt-4o", "hi")
        message = str(exc.value)
        assert "user.blockrun.ai" in message, "does not say where to top up"
        assert "BALANCE_EXHAUSTED" in message, "drops the gateway's own reason"
        assert "no wallet" not in message.lower(), "blames a wallet the caller does not have"

    def test_image_async_202_on_the_first_post(self):
        """The wallet rail only ever sees a 202 after the signed retry, so
        without the account-rail branch every slow model raised 'API error: 202'."""
        polls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("authorization") == f"Bearer {API_KEY}"
            assert request.headers.get("payment-signature") is None
            if request.method == "POST":
                return httpx.Response(
                    202,
                    json={
                        "id": "img_1",
                        "status": "queued",
                        # Minted by the gateway, so it carries the /api prefix.
                        "poll_url": "/api/v1/images/generations/img_1",
                    },
                )
            polls["n"] += 1
            if polls["n"] < 2:
                return httpx.Response(202, json={"status": "in_progress"})
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "created": 1,
                    "data": [{"url": "https://cdn.example/i.png"}],
                },
            )

        client = ImageClient(private_key=API_KEY)
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), headers=auth_headers(API_KEY)
        )
        client.IMAGE_POLL_INTERVAL_SECONDS = 0.0

        resp = client.generate("a red cube")
        assert resp.data[0].url == "https://cdn.example/i.png"
        assert polls["n"] == 2


class TestWalletOnlyHelpers:
    """Returning 0 from get_balance would be indistinguishable from an empty
    wallet, and an agent gating on it would stop calling a funded account."""

    def test_get_balance_refuses(self):
        client = LLMClient(private_key=API_KEY)
        with pytest.raises(ValueError, match="user.blockrun.ai"):
            client.get_balance()

    def test_onramp_refuses(self):
        client = LLMClient(private_key=API_KEY)
        with pytest.raises(ValueError, match="wallet-only"):
            client.onramp(WALLET_ADDRESS)

    def test_get_wallet_address_is_empty(self):
        assert LLMClient(private_key=API_KEY).get_wallet_address() == ""


class TestSetupAgentWallet:
    def test_uses_the_key_without_minting_a_wallet(self, monkeypatch, tmp_path):
        """A skill calls this unconditionally; with a key configured it must not
        write a private key to disk for a wallet that will never sign."""
        monkeypatch.setenv(ENV_API_KEY, API_KEY)
        monkeypatch.setenv("HOME", str(tmp_path))

        from blockrun_llm import setup_agent_wallet

        client = setup_agent_wallet()
        assert client.payment_mode == PAYMENT_MODE_API_KEY
        assert client.get_wallet_address() == ""
        assert not (tmp_path / ".blockrun" / ".session").exists()


@pytest.mark.parametrize("bad_key", ["", "   ", "not-a-key"])
def test_invalid_env_never_selects_a_wallet(monkeypatch, bad_key):
    monkeypatch.setenv(ENV_API_KEY, bad_key)
    monkeypatch.setenv("BLOCKRUN_WALLET_KEY", WALLET_KEY)
    with pytest.raises(ValueError, match="BLOCKRUN_API_KEY"):
        LLMClient()
    # Explicit wallet selection remains available even with a broken env key.
    with LLMClient(private_key=WALLET_KEY) as client:
        assert client.payment_mode == PAYMENT_MODE_WALLET


def test_rotating_env_only_affects_new_clients(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, API_KEY)
    with LLMClient() as first:
        monkeypatch.setenv(ENV_API_KEY, "brk_test_second")
        with LLMClient() as second:
            assert first._client.headers["authorization"] == f"Bearer {API_KEY}"
            assert second._client.headers["authorization"] == "Bearer brk_test_second"
        monkeypatch.delenv(ENV_API_KEY)
        with LLMClient(private_key=WALLET_KEY) as wallet:
            assert "authorization" not in wallet._client.headers
