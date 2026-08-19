"""
Routing surface parity across the four clients.

Base and Solana, sync and async, must expose the same routing: route(),
smart_chat(), smart_chat_completion(), the blockrun/* virtual model ids, and a
ranked fallback chain on the ordinary chat paths. Before 1.12.0 the Solana
clients had none of it and the Base clients had no message-list routing, so a
Solana user got no routing at all and an agent transcript could not be routed.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from blockrun_llm import AsyncLLMClient, AsyncSolanaLLMClient, LLMClient, SolanaLLMClient
from blockrun_llm.router_adapter import build_model_pricing, routing_profile_for_model

CLIENTS = [LLMClient, AsyncLLMClient, SolanaLLMClient, AsyncSolanaLLMClient]

CATALOG = [
    {"id": "google/gemini-2.5-flash", "pricing": {"input": 0.15, "output": 0.6}},
    {"id": "google/gemini-3.5-flash", "pricing": {"input": 0.5, "output": 3}},
    {"id": "google/gemini-3-flash-preview", "pricing": {"input": 0.5, "output": 3}},
    {"id": "google/gemini-3.1-flash-lite", "pricing": {"input": 0.25, "output": 1.5}},
    {"id": "google/gemini-3.1-pro", "pricing": {"input": 1.25, "output": 10}},
    {"id": "anthropic/claude-opus-4.7", "pricing": {"input": 5, "output": 25}},
    {"id": "anthropic/claude-sonnet-5", "pricing": {"input": 3, "output": 15}},
    {"id": "openai/gpt-5-mini", "pricing": {"input": 0.25, "output": 2}},
    {"id": "openai/gpt-5.3-codex", "pricing": {"input": 1.75, "output": 14}},
    {"id": "deepseek/deepseek-v4-pro", "pricing": {"input": 0.435, "output": 0.87}},
    {"id": "moonshot/kimi-k2.7", "pricing": {"input": 0.95, "output": 4}},
    {"id": "nvidia/step-3.7-flash", "pricing": {"input": 0, "output": 0}},
    {"id": "nvidia/mistral-nemotron", "pricing": {"input": 0, "output": 0}},
    {"id": "nvidia/nemotron-nano-9b-v2", "pricing": {"input": 0, "output": 0}},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "pricing": {"input": 0, "output": 0}},
    # Unavailable rows must never win routing.
    {"id": "dead/model", "pricing": {"input": 0.01, "output": 0.01}, "available": False},
]


class TestSurfaceParity:
    @pytest.mark.parametrize("client", CLIENTS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("method", ["route", "smart_chat", "smart_chat_completion"])
    def test_every_client_exposes_the_routing_surface(self, client, method):
        assert hasattr(client, method), f"{client.__name__} is missing {method}()"

    @pytest.mark.parametrize("client", CLIENTS, ids=lambda c: c.__name__)
    def test_the_ordinary_chat_paths_accept_a_fallback_chain(self, client):
        # Routing hands back a ranked chain; it is useless if chat() cannot walk it.
        for method in ("chat", "chat_completion"):
            params = inspect.signature(getattr(client, method)).parameters
            assert "fallback_models" in params, f"{client.__name__}.{method}"

    @pytest.mark.parametrize("client", CLIENTS, ids=lambda c: c.__name__)
    def test_routing_profile_is_selectable_everywhere(self, client):
        for method in ("route", "smart_chat", "smart_chat_completion"):
            params = inspect.signature(getattr(client, method)).parameters
            assert "routing_profile" in params, f"{client.__name__}.{method}"


class TestVirtualModelIds:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("blockrun/auto", "auto"),
            ("blockrun/eco", "eco"),
            ("blockrun/premium", "premium"),
            ("BLOCKRUN/AUTO", "auto"),
            ("google/gemini-2.5-flash", None),
            ("blockrun/nonsense", None),
        ],
    )
    def test_only_the_three_profiles_are_virtual(self, model, expected):
        assert routing_profile_for_model(model) == expected

    def test_chat_completion_routes_a_virtual_id_instead_of_calling_it(self):
        client = LLMClient(private_key="0x" + "11" * 32)
        with (
            patch.object(LLMClient, "list_models", return_value=CATALOG),
            patch.object(LLMClient, "chat_completion", wraps=client.chat_completion) as spy,
            patch.object(LLMClient, "_request_with_payment") as request,
        ):
            request.return_value = None
            try:
                client.chat_completion("blockrun/auto", [{"role": "user", "content": "hi"}])
            except Exception:  # the transport is stubbed; routing is the subject
                pass
            # Re-entered through the routed path with a concrete model.
            routed = [call.args[0] for call in spy.call_args_list if call.args]
            assert "blockrun/auto" in routed
            assert any(m != "blockrun/auto" for m in routed), "never resolved to a real model"


class TestPricingMap:
    def test_skips_rows_the_catalog_marks_unavailable(self):
        pricing = build_model_pricing(CATALOG)

        assert "dead/model" not in pricing
        assert pricing["google/gemini-2.5-flash"] == {
            "input_price": 0.15,
            "output_price": 0.6,
            "flat_price": 0.0,
        }

    def test_accepts_the_legacy_top_level_price_shape(self):
        pricing = build_model_pricing([{"id": "a/b", "inputPrice": 1, "outputPrice": 2}])

        assert pricing["a/b"]["input_price"] == 1
        assert pricing["a/b"]["output_price"] == 2


class TestDecisionsMatchAcrossChains:
    """Base and Solana share one engine: same catalog in, same model out.

    Only the cost floor differs — Base signs a $0.002 minimum, Solana $0.001.
    """

    @pytest.mark.parametrize(
        "prompt",
        [
            "Summarize this changelog entry in one line",
            "Prove that the square root of 2 is irrational, step by step",
            "Refactor this TypeScript function to use async/await",
        ],
    )
    def test_same_model_on_both_chains(self, prompt):
        base = LLMClient(private_key="0x" + "11" * 32)
        solana = SolanaLLMClient.__new__(SolanaLLMClient)
        solana._model_pricing_cache = build_model_pricing(CATALOG)

        with patch.object(LLMClient, "list_models", return_value=CATALOG):
            base_decision = base.route(prompt)
        solana_decision = SolanaLLMClient.route(solana, prompt)

        assert base_decision.model == solana_decision.model
        assert base_decision.tier == solana_decision.tier
        assert base_decision.task_type == solana_decision.task_type
        assert base_decision.candidates == solana_decision.candidates
        # Chain-specific payment floor, same routing.
        assert base_decision.cost_estimate >= solana_decision.cost_estimate

    def test_free_profile_is_free_on_solana_too(self):
        solana = SolanaLLMClient.__new__(SolanaLLMClient)
        solana._model_pricing_cache = build_model_pricing(CATALOG)

        decision = SolanaLLMClient.route(solana, "What is 2+2?", routing_profile="free")

        assert decision.cost_estimate == 0
        for model in [decision.model, *decision.fallbacks]:
            assert solana._model_pricing_cache[model]["input_price"] == 0
            assert solana._model_pricing_cache[model]["output_price"] == 0


class TestRetriableStatuses:
    """A saturated upstream must hand the turn to the next ranked model.

    Observed live: a rate-limited free model answered 429 and the three
    remaining free models in the chain were never tried, because 429 was not in
    the retriable set. The TypeScript adapter has always treated it as
    transient — same upstream saturated, next model is a different upstream.
    """

    @pytest.mark.parametrize("status", [429, 502, 503, 504, 522, 524])
    def test_saturation_and_availability_errors_walk_the_chain(self, status):
        from blockrun_llm.client import _should_fallback
        from blockrun_llm.solana_client import _should_fallback_solana
        from blockrun_llm.types import APIError

        exc = APIError(f"API error: {status}", status_code=status)

        assert _should_fallback(exc), f"Base refuses to fall back on {status}"
        assert _should_fallback_solana(exc), f"Solana refuses to fall back on {status}"

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_do_not_walk_the_chain(self, status):
        from blockrun_llm.client import _should_fallback
        from blockrun_llm.solana_client import _should_fallback_solana
        from blockrun_llm.types import APIError

        exc = APIError(f"API error: {status}", status_code=status)

        assert not _should_fallback(exc)
        assert not _should_fallback_solana(exc)

    def test_a_settled_payment_is_never_retried(self):
        # The next model would sign a second transfer for one call.
        from blockrun_llm.client import _mark_settled, _should_fallback
        from blockrun_llm.solana_client import _should_fallback_solana
        from blockrun_llm.types import APIError

        exc = APIError("API error: 503", status_code=503)
        _mark_settled(exc)

        assert not _should_fallback(exc)
        assert not _should_fallback_solana(exc)
