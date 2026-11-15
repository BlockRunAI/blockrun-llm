"""decisions() and the xAI-only search guard.

decisions(): typed judgments on POST /v1/decisions (OpenAI Decisions-compatible).
Search: the gateway serves search on xai/* models only and answers any other
model with a non-transient 400 SEARCH_UNSUPPORTED, so an explicit non-xAI
model is refused before anything is sent, and routed calls stay on xai/*.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import blockrun_llm.cache as cache_mod
from blockrun_llm import DECISIONS_DEFAULT_MODEL, DecisionsResponse, LLMClient
from blockrun_llm.decisions import build_decisions_body
from blockrun_llm.router_adapter import (
    build_model_pricing,
    requests_search,
    require_search_model,
    route_with_catalog,
)

from ..helpers import TEST_PRIVATE_KEY, build_payment_required_response

QUESTIONS = [
    {"type": "predicate", "name": "urgent", "instructions": "Does this need a reply today?"},
    {
        "type": "choice",
        "name": "team",
        "instructions": "Which team?",
        "choices": [{"value": "billing"}, {"value": "none"}],
    },
    {
        "type": "score",
        "name": "anger",
        "instructions": "How upset?",
        "levels": [{"label": "calm"}, {"label": "furious"}],
    },
]

# The shape gpt-6-luna answered with on sol.blockrun.ai, 2026-11-15.
ANSWER = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "urgent", "probability": 0.85},
        {
            "type": "choice",
            "name": "team",
            "choice": "billing",
            "confidence": 1,
            "probabilities": [
                {"value": "billing", "probability": 1},
                {"value": "none", "probability": 0},
            ],
        },
        {
            "type": "score",
            "name": "anger",
            "score": 1,
            "confidence": 1,
            "probabilities": [
                {"value": 0, "label": "calm", "probability": 0},
                {"value": 1, "label": "furious", "probability": 1},
            ],
        },
    ],
    "usage": {"input_tokens": 386, "output_tokens": 0, "total_tokens": 386},
}


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cache_mod, "COST_LOG_PATH", tmp_path / "cost_log.jsonl")
    return tmp_path


# ── decisions() ───────────────────────────────────────────────────────────


class TestBuildDecisionsBody:
    def test_defaults_the_model_and_forwards_the_rest(self):
        body = build_decisions_body("text", QUESTIONS, metadata={"ticket": "42"})
        assert body == {
            "model": DECISIONS_DEFAULT_MODEL,
            "input": "text",
            "questions": QUESTIONS,
            "metadata": {"ticket": "42"},
        }

    def test_keeps_a_callers_model(self):
        assert build_decisions_body("t", QUESTIONS, model="openai/gpt-6-luna")["model"] == (
            "openai/gpt-6-luna"
        )

    @pytest.mark.parametrize(
        "input_,questions,match",
        [
            ("", QUESTIONS, "input"),
            ([], QUESTIONS, "input"),
            ("t", [], "questions"),
            ("t", [{"type": "rank", "name": "a", "instructions": "x"}], "type"),
            ("t", [{"type": "predicate", "name": "", "instructions": "x"}], "name"),
            ("t", [{"type": "predicate", "name": "a", "instructions": ""}], "instructions"),
            ("t", [QUESTIONS[0], QUESTIONS[0]], "twice"),
            ("t", [{"type": "choice", "name": "a", "instructions": "x", "choices": []}], "choices"),
            ("t", [{"type": "score", "name": "a", "instructions": "x"}], "levels"),
        ],
    )
    def test_refuses_a_malformed_request(self, input_: Any, questions: Any, match: str):
        with pytest.raises(ValueError, match=match):
            build_decisions_body(input_, questions)


def _wallet_gateway(seen: list[httpx.Request], answer: dict[str, Any] = ANSWER):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("PAYMENT-SIGNATURE") is None:
            return httpx.Response(
                402,
                headers={"payment-required": build_payment_required_response(amount="1069")},
                json={"error": "Payment Required"},
            )
        return httpx.Response(200, json=answer)

    return handler


def test_llmclient_decisions_pays_the_quote_and_returns_typed_answers():
    seen: list[httpx.Request] = []
    client = LLMClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(_wallet_gateway(seen)))

    result = client.decisions("My card was charged twice.", QUESTIONS)

    assert isinstance(result, DecisionsResponse)
    assert result.answers == ANSWER["answers"]
    assert result.usage and result.usage["output_tokens"] == 0
    assert [str(r.url) for r in seen] == ["https://blockrun.ai/api/v1/decisions"] * 2
    sent = json.loads(seen[0].content)
    assert sent["model"] == "gpt-6-luna" and sent["questions"] == QUESTIONS
    assert json.loads(seen[1].content) == sent
    assert client.get_spending()["calls"] == 1


def test_decisions_is_never_served_from_cache():
    seen: list[httpx.Request] = []
    client = LLMClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(_wallet_gateway(seen)))

    client.decisions("same", QUESTIONS)
    client.decisions("same", QUESTIONS)

    assert len([r for r in seen if r.headers.get("PAYMENT-SIGNATURE")]) == 2


def test_decisions_sends_nothing_for_a_malformed_request():
    seen: list[httpx.Request] = []
    client = LLMClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(_wallet_gateway(seen)))
    with pytest.raises(ValueError, match="questions"):
        client.decisions("text", [])
    assert seen == []


# ── search guard ──────────────────────────────────────────────────────────


class TestRequestsSearch:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, False),
            ({"search": True}, True),
            ({"search": False}, False),
            ({"search_parameters": {"mode": "on"}}, True),
            ({"search_parameters": {"mode": "off"}}, False),
            ({"search": True, "search_parameters": {"mode": "off"}}, False),
            ({"tools": [{"type": "x_search"}]}, True),
            ({"tools": [{"type": "web_search"}]}, True),
            ({"tools": [{"type": "function", "function": {"name": "f"}}]}, False),
        ],
    )
    def test_detection(self, kwargs: dict[str, Any], expected: bool):
        assert requests_search(**kwargs) is expected

    def test_require_search_model_refuses_a_non_xai_model(self):
        with pytest.raises(ValueError, match="openai/gpt-5.2"):
            require_search_model(["openai/gpt-5.2"], True)
        require_search_model(["xai/grok-4.7"], True)
        require_search_model(["openai/gpt-5.2"], None, {"mode": "off"})


@pytest.mark.parametrize("model", ["openai/gpt-5.2", "anthropic/claude-sonnet-5"])
def test_chat_completion_refuses_search_on_a_non_xai_model_before_sending(model: str):
    seen: list[httpx.Request] = []
    client = LLMClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(_wallet_gateway(seen)))
    with pytest.raises(ValueError, match="xai/"):
        client.chat_completion(model, [{"role": "user", "content": "news?"}], search=True)
    assert seen == []


def test_chat_completion_refuses_a_non_xai_fallback_with_search():
    client = LLMClient(private_key=TEST_PRIVATE_KEY)
    with pytest.raises(ValueError, match="openai/gpt-5.2"):
        client.chat_completion(
            "xai/grok-4.7",
            [{"role": "user", "content": "news?"}],
            search=True,
            fallback_models=["openai/gpt-5.2"],
        )


# The catalog rows the search fallback reads (categories matter).
SEARCH_CATALOG = [
    {
        "id": "google/gemini-2.5-flash",
        "categories": ["chat"],
        "pricing": {"input": 0.15, "output": 0.6},
    },
    {
        "id": "google/gemini-3.1-pro",
        "categories": ["chat"],
        "pricing": {"input": 1.25, "output": 10},
    },
    {
        "id": "anthropic/claude-opus-4.7",
        "categories": ["chat"],
        "pricing": {"input": 5, "output": 25},
    },
    {
        "id": "deepseek/deepseek-reasoner",
        "categories": ["chat"],
        "pricing": {"input": 0.28, "output": 0.42},
    },
    {"id": "xai/grok-4.7", "categories": ["chat", "search"], "pricing": {"input": 3, "output": 15}},
    {
        "id": "xai/grok-4.5",
        "categories": ["chat", "search"],
        "pricing": {"input": 0.2, "output": 0.5},
    },
    {
        "id": "xai/grok-build-0.1",
        "categories": ["chat", "coding"],
        "pricing": {"input": 0.2, "output": 1},
    },
    {
        "id": "xai/grok-old",
        "categories": ["chat", "search"],
        "available": False,
        "pricing": {"input": 0.1, "output": 0.1},
    },
]


class TestSearchRouting:
    def test_pricing_records_the_catalogs_search_models(self):
        pricing = build_model_pricing(SEARCH_CATALOG)
        # Tagged, available xai/* chat models only.
        assert sorted(pricing.search_models) == ["xai/grok-4.5", "xai/grok-4.7"]  # type: ignore[attr-defined]

    def test_without_search_nothing_changes(self):
        pricing = build_model_pricing(SEARCH_CATALOG)
        decision = route_with_catalog("What is the capital of France?", None, 256, pricing)
        assert not decision["model"].startswith("xai/")

    @pytest.mark.parametrize("profile", ["eco", "auto"])
    def test_routes_to_the_cheapest_search_model(self, profile: str):
        pricing = build_model_pricing(SEARCH_CATALOG)
        decision = route_with_catalog(
            "Latest AI news?", None, 256, pricing, routing_profile=profile, search=True
        )
        assert decision["model"] == "xai/grok-4.5"
        assert decision["candidates"] == ["xai/grok-4.5", "xai/grok-4.7"]
        assert decision["fallbacks"] == ["xai/grok-4.7"]
        assert "search runs on xai/* only" in decision["reasoning"]

    def test_premium_routes_to_the_most_capable_search_model(self):
        pricing = build_model_pricing(SEARCH_CATALOG)
        decision = route_with_catalog(
            "Latest AI news?", None, 256, pricing, routing_profile="premium", search=True
        )
        assert decision["model"] == "xai/grok-4.7"

    def test_free_profile_with_search_is_refused(self):
        pricing = build_model_pricing(SEARCH_CATALOG)
        with pytest.raises(ValueError, match="free profile"):
            route_with_catalog("news?", None, 256, pricing, routing_profile="free", search=True)

    def test_a_catalog_without_xai_is_refused(self):
        pricing = build_model_pricing([r for r in SEARCH_CATALOG if not r["id"].startswith("xai/")])
        with pytest.raises(ValueError, match="no available xai"):
            route_with_catalog("news?", None, 256, pricing, search=True)

    def test_smart_chat_completion_with_search_calls_an_xai_model(self):
        seen: list[httpx.Request] = []
        chat_ok = {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "xai/grok-4.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        client._client = httpx.Client(transport=httpx.MockTransport(_wallet_gateway(seen, chat_ok)))
        client._model_pricing_cache = build_model_pricing(SEARCH_CATALOG)

        result = client.smart_chat_completion(
            [{"role": "user", "content": "Latest AI news?"}], search=True
        )

        assert result.model == "xai/grok-4.5"
        assert json.loads(seen[0].content)["model"] == "xai/grok-4.5"
