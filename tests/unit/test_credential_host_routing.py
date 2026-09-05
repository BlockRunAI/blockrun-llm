"""One rule, checked on every client: the credential decides the host.

    API key      -> api.blockrun.ai   (prepaid credit, bearer auth)
    Solana key   -> sol.blockrun.ai   (x402 on SVM)
    Base key     -> blockrun.ai       (x402 on EVM)

Sending a credential to the wrong front door is not a 404 — an API key handed
to an x402 host is a key disclosed to a host that never needed it, and a wallet
pointed at the account rail signs nothing and gets a bearer-auth rejection.
The table is parametrized over every exported client so a new one cannot quietly
skip the rule.
"""

from __future__ import annotations

import pytest

from blockrun_llm import (
    AnthropicClient,
    AsyncLLMClient,
    AsyncSolanaLLMClient,
    ImageClient,
    LLMClient,
    MusicClient,
    PhoneClient,
    PortraitClient,
    PriceClient,
    RealFaceClient,
    RpcClient,
    SearchClient,
    SolanaLLMClient,
    SpeechClient,
    SurfClient,
    VideoClient,
    VoiceClient,
)
from blockrun_llm.apikey import DEFAULT_API_KEY_URL, ENV_API_KEY, ENV_API_KEY_URL

API_KEY = "brk_live_host_routing_fixture"
BASE_KEY = "0x" + "ac" * 32
# Throwaway base58 keypair (seed = bytes(range(32))). Never funded.
SOLANA_KEY = (
    "1GMkH3brNXiNNs1tiFZHu4yZSRrzJwxi5wB9bHFtMikjwpAW9DMZzU2Pqakc5it8X3N5vPmqdN7KF4CCUpmKhq"
)

SOLANA_CLIENTS = [SolanaLLMClient, AsyncSolanaLLMClient]
BASE_CLIENTS = [
    LLMClient,
    AsyncLLMClient,
    ImageClient,
    VideoClient,
    MusicClient,
    SpeechClient,
    VoiceClient,
    PhoneClient,
    PortraitClient,
    RealFaceClient,
    RpcClient,
    SearchClient,
    SurfClient,
    PriceClient,
    AnthropicClient,
]
ALL_CLIENTS = BASE_CLIENTS + SOLANA_CLIENTS


def host_of(client) -> str:
    for attr in ("api_url", "_api_url"):
        value = getattr(client, attr, None)
        if value:
            return str(value).rstrip("/")
    raise AssertionError(f"{type(client).__name__} exposes no resolved host")


def build(cls, credential, **kwargs):
    if cls is AnthropicClient:
        pytest.importorskip("anthropic")
    if cls in SOLANA_CLIENTS and credential is BASE_KEY:
        pytest.skip("Solana clients take a Solana key, not a Base one")
    return cls(credential, **kwargs)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        ENV_API_KEY,
        ENV_API_KEY_URL,
        "BLOCKRUN_API_URL",
        "BLOCKRUN_WALLET_KEY",
        "BASE_CHAIN_WALLET_KEY",
        "SOLANA_WALLET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("cls", ALL_CLIENTS, ids=lambda c: c.__name__)
def test_an_api_key_always_reaches_the_account_rail(cls):
    client = build(cls, API_KEY)
    assert host_of(client) == DEFAULT_API_KEY_URL


@pytest.mark.parametrize("cls", SOLANA_CLIENTS, ids=lambda c: c.__name__)
def test_a_solana_wallet_reaches_the_solana_gateway(cls):
    pytest.importorskip("x402")
    assert host_of(build(cls, SOLANA_KEY)) == "https://sol.blockrun.ai/api"


@pytest.mark.parametrize("cls", BASE_CLIENTS, ids=lambda c: c.__name__)
def test_a_base_wallet_reaches_the_base_gateway(cls):
    assert host_of(build(cls, BASE_KEY)) == "https://blockrun.ai/api"


@pytest.mark.parametrize("cls", ALL_CLIENTS, ids=lambda c: c.__name__)
def test_an_x402_gateway_override_never_captures_an_api_key(cls, monkeypatch):
    """BLOCKRUN_API_URL names an x402 host. A developer pointing it at a private
    deployment must not have an API-key client follow it there and hand over the
    key — which is why the account rail has its own variable."""
    monkeypatch.setenv("BLOCKRUN_API_URL", "https://x402-deployment.internal/api")
    assert host_of(build(cls, API_KEY)) == DEFAULT_API_KEY_URL


@pytest.mark.parametrize("cls", ALL_CLIENTS, ids=lambda c: c.__name__)
def test_the_account_rail_has_its_own_override(cls, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY_URL, "https://staging.blockrun.ai")
    assert host_of(build(cls, API_KEY)) == "https://staging.blockrun.ai"


@pytest.mark.parametrize("cls", ALL_CLIENTS, ids=lambda c: c.__name__)
def test_an_explicit_api_url_wins_on_every_client(cls):
    """The Solana constructors default api_url to the SVM gateway rather than
    None, so an account-rail client there has to tell "the caller typed this"
    apart from "nobody passed anything" — otherwise it either ignores the
    argument or sends the key to sol.blockrun.ai."""
    assert host_of(build(cls, API_KEY, api_url="https://custom.example")) == (
        "https://custom.example"
    )


@pytest.mark.parametrize("cls", SOLANA_CLIENTS, ids=lambda c: c.__name__)
def test_the_solana_default_never_leaks_onto_the_account_rail(cls):
    """Passing the default explicitly is indistinguishable from not passing it,
    and it must resolve to the account rail either way."""
    assert host_of(build(cls, API_KEY, api_url="https://sol.blockrun.ai/api")) == (
        DEFAULT_API_KEY_URL
    )
