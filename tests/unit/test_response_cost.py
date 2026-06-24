"""ChatResponse carries the real per-call x402 charge.

These lock in that the client attaches ``cost_usd`` to every chat completion so
downstream consumers (e.g. blockrun-litellm) can report the actual wallet
deduction instead of a token×list-price estimate. The free/200-first path must
report exactly 0.0 (not a stale prior charge).
"""

from unittest.mock import MagicMock, patch

from blockrun_llm import LLMClient

TEST_KEY = "0x" + "1" * 64

_CHAT_BODY = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "openai/gpt-5.5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
}


@patch("blockrun_llm.client.httpx.Client")
def test_free_call_attaches_zero_cost(mock_client_class):
    """A 200 on the first attempt = no payment required => cost_usd == 0.0."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    resp200 = MagicMock(status_code=200)
    resp200.json.return_value = _CHAT_BODY
    mock_client.post.return_value = resp200

    client = LLMClient(private_key=TEST_KEY)
    result = client.chat_completion(
        model="openai/gpt-5.5", messages=[{"role": "user", "content": "hi"}]
    )

    assert result.cost_usd == 0.0
    assert result.settlement is None
    # cost_usd survives model_dump so the OpenAI-shaped payload carries it.
    assert result.model_dump(exclude_none=True)["cost_usd"] == 0.0


def test_chatresponse_cost_fields_default_none():
    from blockrun_llm.types import ChatResponse

    r = ChatResponse(id="x", object="chat.completion", created=0, model="m", choices=[])
    assert r.cost_usd is None and r.settlement is None
    # absent by default (exclude_none) — no phantom $0 on objects we didn't charge
    assert "cost_usd" not in r.model_dump(exclude_none=True)
