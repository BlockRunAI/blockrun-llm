"""Golden conformance and SDK plumbing for Router Core V3."""

from __future__ import annotations

from typing import Any

import pytest

from blockrun_llm.client import AsyncLLMClient, LLMClient
from blockrun_llm.router_v3 import ROUTER_CORE_COMMIT, message_routing_inputs, route
from blockrun_llm.solana_client import AsyncSolanaLLMClient, SolanaLLMClient
from blockrun_llm.types import ChatResponse


def _price(input_price: float, output_price: float) -> dict[str, float]:
    return {"input_price": input_price, "output_price": output_price, "flat_price": 0}


# Current public catalog subset used by the official Router Core golden cases.
PRICING = {
    "google/gemini-2.5-flash": _price(0.3, 2.5),
    "google/gemini-3-flash-preview": _price(0.5, 3),
    "google/gemini-3.5-flash": _price(1.5, 9),
    "google/gemini-3.1-pro": _price(2, 12),
    "google/gemini-3.1-flash-lite": _price(0.25, 1.5),
    "google/gemini-2.5-flash-lite": _price(0.1, 0.4),
    "deepseek/deepseek-chat": _price(0.2, 0.4),
    "deepseek/deepseek-reasoner": _price(0.2, 0.4),
    "deepseek/deepseek-v4-pro": _price(0.435, 0.87),
    "openai/gpt-5.4-nano": _price(0.2, 1.25),
    "openai/gpt-5-mini": _price(0.25, 2),
    "openai/gpt-5.3-codex": _price(1.75, 14),
    "openai/gpt-4o-mini": _price(0.15, 0.6),
    "openai/gpt-4.1": _price(2, 8),
    "openai/o4-mini": _price(1.1, 4.4),
    "openai/o3": _price(2, 8),
    "anthropic/claude-haiku-4.5": _price(1, 5),
    "anthropic/claude-sonnet-5": _price(3, 15),
    "anthropic/claude-sonnet-4.6": _price(3, 15),
    "anthropic/claude-opus-4.8": _price(5, 25),
    "anthropic/claude-opus-5": _price(5, 25),
    "xai/grok-4.5": _price(2.5, 9),
    "xai/grok-4.3": _price(1.5, 4),
    "moonshot/kimi-k3": _price(3, 15),
    "zai/glm-5.2": _price(1.4, 4.4),
}


TOOLS = {
    "terminal": [
        {"type": "function", "function": {"name": "terminalExec"}},
        {"type": "function", "function": {"name": "terminalInspect"}},
        {"type": "function", "function": {"name": "terminalSendKeys"}},
    ],
    "order": [
        {"type": "function", "function": {"name": "get_order"}},
        {"type": "function", "function": {"name": "update_address"}},
    ],
    "weather": [{"type": "function", "function": {"name": "get_weather"}}],
    "web": [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "web_fetch"}},
    ],
}


@pytest.mark.parametrize(
    ("prompt", "kwargs", "expected"),
    [
        (
            "Explain why the sky is blue in two sentences.",
            {},
            ("google/gemini-2.5-flash", "SIMPLE", "chat"),
        ),
        (
            "从这段文字中提取姓名和公司，并返回 JSON：Ada works at BlockRun。",
            {"requires_structured_output": True},
            ("google/gemini-2.5-flash", "MEDIUM", "extraction"),
        ),
        (
            "Refactor this TypeScript function to avoid the race condition and return a patch.",
            {},
            ("google/gemini-3-flash-preview", "MEDIUM", "code_edit"),
        ),
        (
            "Debug the failing Python tests, identify the regression, edit the files, and verify the fix.",
            {},
            ("openai/gpt-4o-mini", "MEDIUM", "debug"),
        ),
        (
            "Prove the theorem formally and derive every step.",
            {},
            ("deepseek/deepseek-v4-pro", "REASONING", "reasoning"),
        ),
        (
            "Which option is correct?\nA. Mercury\nB. Venus\nC. Earth\nD. Mars",
            {},
            ("google/gemini-3-flash-preview", "REASONING", "reasoning_mcq"),
        ),
        (
            "A shop sells 3 books at $12 each with a 25% discount. How much is the total?",
            {},
            ("deepseek/deepseek-v4-pro", "REASONING", "reasoning_math"),
        ),
        (
            "Inspect the repository, edit the TypeScript files, run tests, and fix the bug.",
            {"tools": TOOLS["terminal"], "tool_choice": "required"},
            ("openai/gpt-5-mini", "SIMPLE", "code_agent"),
        ),
        (
            "Check my order status and update its delivery address.",
            {"tools": TOOLS["order"], "tool_choice": "required"},
            ("openai/gpt-5-mini", "SIMPLE", "tool_agent"),
        ),
        (
            "Get the weather for Tokyo, Paris, and London simultaneously.",
            {"tools": TOOLS["weather"], "tool_choice": "required"},
            ("anthropic/claude-opus-4.8", "SIMPLE", "tool_agent_parallel"),
        ),
        (
            "Using multiple public sources, identify the person described by the following clues and return one exact best-supported answer: they founded a company after 2010, later joined another lab, and published work in 2024.",
            {"tools": TOOLS["web"], "tool_choice": "required"},
            ("anthropic/claude-sonnet-5", "MEDIUM", "tool_agent"),
        ),
        (
            "Read this screenshot and explain the error.",
            {"has_vision": True},
            ("google/gemini-2.5-flash", "SIMPLE", "vision"),
        ),
        (
            "Review this production payment implementation for security vulnerabilities.",
            {"routing_profile": "premium"},
            ("google/gemini-2.5-flash", "SIMPLE", "chat"),
        ),
        (
            "Summarize this note in one sentence.",
            {"routing_profile": "eco"},
            ("google/gemini-3.1-flash-lite", "SIMPLE", "chat"),
        ),
    ],
)
def test_router_v3_matches_core_golden_after_catalog_filter(
    prompt: str, kwargs: dict[str, Any], expected: tuple[str, str, str]
) -> None:
    decision = route(prompt, None, 512, PRICING, **kwargs)
    assert (decision["model"], decision["tier"], decision["task_type"]) == expected
    assert decision["router_version"] == "v3-portfolio"
    assert ROUTER_CORE_COMMIT == "d4308049348e11e17ed08a254676a34949be80f9"


def _response(model: str) -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-test",
        created=1,
        model=model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    )


def test_base_auto_alias_routes_full_agent_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object.__new__(LLMClient)
    client._model_pricing_cache = PRICING
    seen: dict[str, Any] = {}

    def fake_request(_endpoint: str, body: dict[str, Any]) -> ChatResponse:
        seen.update(body)
        return _response(body["model"])

    monkeypatch.setattr(client, "_request_with_payment", fake_request)
    result = client.chat_completion(
        "blockrun/auto",
        [{"role": "user", "content": "Check my order status and update its delivery address."}],
        tools=TOOLS["order"],
        tool_choice="required",
        max_tokens=512,
    )
    assert seen["model"] == "openai/gpt-5-mini"
    assert seen["tools"] == TOOLS["order"]
    assert result.routing and result.routing["task_type"] == "tool_agent"


def test_solana_auto_alias_uses_solana_payment_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object.__new__(SolanaLLMClient)
    client._model_pricing_cache = PRICING
    seen: dict[str, Any] = {}

    def fake_request(
        _endpoint: str, body: dict[str, Any], timeout: float | None = None
    ) -> ChatResponse:
        del timeout
        seen.update(body)
        return _response(body["model"])

    monkeypatch.setattr(client, "_request_with_payment", fake_request)
    result = client.chat_completion(
        "blockrun/auto",
        [
            {
                "role": "user",
                "content": "Get the weather for Tokyo, Paris, and London simultaneously.",
            }
        ],
        tools=TOOLS["weather"],
        tool_choice="required",
        max_tokens=512,
    )
    assert seen["model"] == "anthropic/claude-opus-4.8"
    assert result.routing and result.routing["cost_estimate"] >= 0.001


@pytest.mark.parametrize("client_type", [AsyncLLMClient, AsyncSolanaLLMClient])
async def test_async_auto_alias_routes_before_payment(
    monkeypatch: pytest.MonkeyPatch, client_type: type[AsyncLLMClient | AsyncSolanaLLMClient]
) -> None:
    client = object.__new__(client_type)

    async def fake_pricing() -> dict[str, dict[str, float]]:
        return PRICING

    async def fake_request(_endpoint: str, body: dict[str, Any], **_kwargs: Any) -> ChatResponse:
        return _response(body["model"])

    monkeypatch.setattr(client, "_get_model_pricing", fake_pricing)
    monkeypatch.setattr(client, "_request_with_payment", fake_request)
    result = await client.chat_completion(
        "blockrun/auto",
        [{"role": "user", "content": "Check my order status and update its delivery address."}],
        tools=TOOLS["order"],
        tool_choice="required",
        max_tokens=512,
    )
    assert result.model == "openai/gpt-5-mini"
    assert result.routing and result.routing["task_type"] == "tool_agent"


def test_message_extraction_uses_latest_user_and_preserves_vision() -> None:
    prompt, system, vision = message_routing_inputs(
        [
            {"role": "system", "content": "Be safe"},
            {"role": "user", "content": "old task"},
            {"role": "assistant", "content": "done"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "read this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                ],
            },
        ]
    )
    assert prompt == "read this"
    assert system == "Be safe"
    assert vision is True
