import asyncio
from unittest.mock import patch

import httpx
import pytest

from blockrun_llm import (
    APIClient,
    AsyncAPIClient,
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
from blockrun_llm.api_key import resolve_api_auth
from blockrun_llm.types import APIError

KEY = "brk_live_test_account"
WALLET = "0x" + "01" * 32
CHAT = {
    "created": 1,
    "id": "chat-1",
    "model": "openai/gpt-5.2",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
CLASSES = [
    LLMClient,
    AsyncLLMClient,
    SolanaLLMClient,
    AsyncSolanaLLMClient,
    ImageClient,
    VideoClient,
    MusicClient,
    SpeechClient,
    VoiceClient,
    PhoneClient,
    PortraitClient,
    PriceClient,
    SearchClient,
    SurfClient,
    RpcClient,
    RealFaceClient,
]


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    monkeypatch.delenv("BLOCKRUN_API_BASE_URL", raising=False)
    # An API client must not consult or create either wallet.
    with (
        patch("blockrun_llm.wallet.load_wallet", side_effect=AssertionError("wallet accessed")),
        patch(
            "blockrun_llm.solana_wallet.load_solana_wallet",
            side_effect=AssertionError("Solana wallet accessed"),
        ),
    ):
        yield


def wire(client, handler):
    client._client.close()
    client._client = httpx.Client(auth=client._api_auth, transport=httpx.MockTransport(handler))
    return client


@pytest.mark.parametrize("cls", CLASSES)
def test_no_wallet_needed(cls):
    client = cls()
    assert client.auth_mode == "api-key"
    http = client._client
    if isinstance(http, httpx.AsyncClient):
        asyncio.run(http.aclose())
    else:
        http.close()


@pytest.mark.parametrize("cls", [LLMClient, SolanaLLMClient])
def test_native_chat(cls):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=CHAT)

    client = wire(cls(), handler)
    try:
        assert client.chat("openai/gpt-5.2", "hi") == "ok"
        assert str(seen[0].url) == "https://api.blockrun.ai/v1/chat/completions"
        assert seen[0].headers["authorization"] == f"Bearer {KEY}"
        assert "payment-signature" not in seen[0].headers
    finally:
        client.close()


@pytest.mark.parametrize("status", [401, 402, 429])
def test_account_errors_never_sign_or_replay(status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            status,
            json={"error": {"code": "quota", "message": KEY}},
            headers={"payment-required": "never-sign", "retry-after": "10"},
        )

    client = wire(LLMClient(), handler)
    with client, pytest.raises(APIError) as error:
        client.chat("openai/gpt-5.2", "hi")
    assert error.value.status_code == status
    assert error.value.retry_after == "10"
    assert error.value.response["code"] == "quota"
    assert KEY not in str(error.value.response)
    assert len(seen) == 1


@pytest.mark.parametrize("cls", [AsyncLLMClient, AsyncSolanaLLMClient])
@pytest.mark.asyncio
async def test_async_chat_and_quota(cls):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200 if len(seen) == 1 else 402, json=CHAT)

    client = cls()
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        auth=client._api_auth, transport=httpx.MockTransport(handler)
    )
    async with client:
        assert await client.chat("openai/gpt-5.2", "hi") == "ok"
        with pytest.raises(APIError) as error:
            await client.chat("openai/gpt-5.2", "hi")
        assert error.value.status_code == 402
    assert len(seen) == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/job",
        "https://api.blockrun.ai:444/job",
        "http://api.blockrun.ai/job",
        "https://u:p@api.blockrun.ai/job",
    ],
)
def test_no_credential_forwarding(url):
    auth = resolve_api_auth(KEY, None, None)
    with pytest.raises(ValueError, match="origin"):
        auth.resolve_url(url)


def test_explicit_wallet_and_validation():
    client = LLMClient(private_key=WALLET)
    assert client.auth_mode == "wallet"
    client.close()
    with pytest.raises(ValueError, match="either"):
        LLMClient(api_key=KEY, private_key=WALLET)
    with pytest.raises(ValueError, match="Invalid BlockRun API key"):
        LLMClient(api_key="bad")
    with LLMClient() as client, pytest.raises(ValueError, match="requires a wallet"):
        client.get_wallet_address()


def test_account_and_wallet_clients_keep_separate_credentials_after_env_change(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=CHAT)

    account = wire(LLMClient(), handler)
    wallet = wire(LLMClient(private_key=WALLET), handler)
    address = wallet.get_wallet_address()
    monkeypatch.setenv("BLOCKRUN_API_KEY", "brk_live_different_account")
    monkeypatch.setenv("BLOCKRUN_API_BASE_URL", "https://different.example")
    with account, wallet:
        assert account.chat("openai/gpt-5.2", "hi") == "ok"
        assert wallet.chat("openai/gpt-5.2", "hi") == "ok"
        assert wallet.get_wallet_address() == address
    assert str(seen[0].url) == "https://api.blockrun.ai/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    assert "payment-signature" not in seen[0].headers
    assert str(seen[1].url) == "https://blockrun.ai/api/v1/chat/completions"
    assert "authorization" not in seen[1].headers


@pytest.mark.parametrize("key", ["", "   ", "bad"])
def test_invalid_env_key_never_falls_back_but_explicit_wallet_still_works(key, monkeypatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", key)
    with pytest.raises(ValueError, match="Invalid BlockRun API key"):
        LLMClient()
    with LLMClient(private_key=WALLET) as client:
        assert client.auth_mode == "wallet"


@pytest.mark.parametrize("cls", [ImageClient, VideoClient, MusicClient])
def test_media_first_response_polled_without_payment(cls, monkeypatch):
    monkeypatch.setattr("blockrun_llm.api_key.time.sleep", lambda _: None)
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(
                202,
                json={
                    "id": "job-1",
                    "status": "queued",
                    "poll_url": "/api/v1/videos/generations/job-1",
                },
            )
        return httpx.Response(
            200,
            json={
                "created": 1,
                "model": "test/model",
                "status": "completed",
                "data": [{"url": "https://cdn.example/output"}],
            },
        )

    client = wire(cls(), handler)
    try:
        result = client.generate("a cat")
        assert result.data[0].url == "https://cdn.example/output"
        assert [r.method for r in seen] == ["POST", "GET"]
        assert str(seen[1].url) == "https://api.blockrun.ai/v1/videos/generations/job-1"
        assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in seen)
    finally:
        client._client.close()


def test_account_does_not_read_wallet_response_cache():
    client = wire(LLMClient(), lambda r: httpx.Response(200, json={"answer": "account"}))
    with (
        client,
        patch(
            "blockrun_llm.cache.get_cached",
            side_effect=AssertionError("shared wallet cache accessed"),
        ),
    ):
        assert client._request_with_payment_raw("/v1/exa/answer", {"query": "hi"}) == {
            "answer": "account"
        }


def test_generic_responses_and_stream():
    seen = []

    def handler(request):
        seen.append(request)
        return (
            httpx.Response(200, text="data: [DONE]\n\n")
            if len(seen) == 2
            else httpx.Response(200, json={"id": "r-1"})
        )

    with APIClient(api_url="https://api.blockrun.ai/v1/") as client:
        client._client.close()
        client._client = httpx.Client(auth=client._auth, transport=httpx.MockTransport(handler))
        assert client.post("/v1/responses", {"input": "hi"})["id"] == "r-1"
        with client.stream("/v1/responses", {"input": "hi"}) as response:
            assert list(response.iter_lines()) == ["data: [DONE]", ""]
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.asyncio
async def test_async_generic_poll():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return (
            httpx.Response(202, json={"status": "queued", "poll_url": "/v1/responses/r-1"})
            if count == 1
            else httpx.Response(200, json={"id": "r-1", "status": "completed"})
        )

    async with AsyncAPIClient() as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            auth=client._auth, transport=httpx.MockTransport(handler)
        )
        assert (await client.poll("/v1/responses", interval_seconds=0))["status"] == "completed"
    assert count == 2


@pytest.mark.parametrize("status", [402, 429, 500, 502])
def test_anthropic_account_quota_is_native_error_without_replay(status):
    anthropic = pytest.importorskip("anthropic")
    from blockrun_llm import AnthropicClient

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            status, json={"error": {"type": "api_error", "message": "uncertain completion"}}
        )

    client = AnthropicClient()
    client._client._client.close()
    client._client._client = httpx.Client(
        auth=client._api_auth, transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(anthropic.APIStatusError) as error:
            client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert error.value.status_code == status
        assert len(seen) == 1
        assert seen[0].headers["authorization"] == f"Bearer {KEY}"
        assert str(seen[0].url) == "https://api.blockrun.ai/v1/messages"
    finally:
        client.close()


@pytest.mark.parametrize("cls", [LLMClient, SolanaLLMClient])
def test_native_account_stream(cls):
    import json

    chunk = {
        "id": "s-1",
        "created": 1,
        "model": "openai/gpt-5.2",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    client = wire(
        cls(), lambda r: httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
    )
    try:
        chunks = list(
            client.chat_completion_stream("openai/gpt-5.2", [{"role": "user", "content": "hi"}])
        )
        assert chunks[0].choices[0].delta.content == "hello"
    finally:
        client.close()


@pytest.mark.parametrize(
    "saved,base,expected",
    [
        (None, False, "solana"),
        (None, True, "base"),
        ("base", False, "base"),
        ("solana", True, "solana"),
    ],
)
def test_setup_preserves_wallet_preference(saved, base, expected, monkeypatch, tmp_path):
    from blockrun_llm import setup_agent_client

    monkeypatch.delenv("BLOCKRUN_API_KEY")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    if saved:
        (tmp_path / ".blockrun").mkdir()
        (tmp_path / ".blockrun/payment-chain").write_text(saved)
    with (
        patch("blockrun_llm.wallet.load_wallet", return_value=WALLET if base else None),
        patch("blockrun_llm.solana_wallet.load_solana_wallet", return_value=None),
        patch("blockrun_llm.wallet.setup_agent_wallet", return_value="base"),
        patch("blockrun_llm.solana_wallet.setup_agent_solana_wallet", return_value="solana"),
    ):
        assert setup_agent_client() == expected


@pytest.mark.parametrize("cls", [LLMClient, AsyncLLMClient, SolanaLLMClient, AsyncSolanaLLMClient])
def test_account_rejects_unenforceable_wallet_limits(cls):
    with pytest.raises(ValueError, match="Wallet spend limits"):
        cls(max_cost_per_call=0.01)


def test_solana_account_context_manager_closes_client_even_on_error(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", KEY)
    client = SolanaLLMClient()
    with pytest.raises(RuntimeError, match="consumer failure"), client as active:
        assert active is client
        assert active.auth_mode == "api-key"
        raise RuntimeError("consumer failure")
    assert client._client.is_closed


@pytest.mark.parametrize("status", [502, 503, 504, 522, 524])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_account_poll_recovers_same_job_after_gateway_hiccup(status, asynchronous):
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url)))
        if len(seen) == 1:
            return httpx.Response(
                202, json={"status": "queued", "poll_url": "/v1/jobs/existing?token=signed"}
            )
        if len(seen) == 2:
            return httpx.Response(status, json={"error": {"message": "temporary gateway failure"}})
        return httpx.Response(200, json={"status": "completed", "id": "existing"})

    async def run_async():
        async with AsyncAPIClient() as client:
            await client._client.aclose()
            client._client = httpx.AsyncClient(
                auth=client._auth, transport=httpx.MockTransport(handler)
            )
            return await client.poll("/v1/jobs", interval_seconds=0)

    if asynchronous:
        result = asyncio.run(run_async())
    else:
        with APIClient() as client:
            client._client.close()
            client._client = httpx.Client(auth=client._auth, transport=httpx.MockTransport(handler))
            result = client.poll("/v1/jobs", interval_seconds=0)
    assert result["status"] == "completed"
    assert [method for method, _ in seen] == ["POST", "GET", "GET"]
    assert seen[1] == seen[2]
