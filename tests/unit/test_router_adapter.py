"""
Tests for the BlockRun host glue around Router Core.

These cover what ``router_adapter`` adds on top of the product-neutral core:
catalog id resolution, the x402 payment floor, capacity filtering against the
whole conversation, and the SDK-only ``free`` profile.
"""

from __future__ import annotations

import pytest

from blockrun_llm.router import route
from blockrun_llm.router_adapter import (
    BASE_MINIMUM_PAYMENT_USD,
    FREE_TIERS,
    routing_profile_for_model,
    routing_text,
)
from blockrun_llm.router_core import DEFAULT_ROUTING_CONFIG
from blockrun_llm.types import RoutingDecision

# The free chat models that answer as themselves, not via a gateway redirect.
# Verified with a two-pass model-echo probe on 2026-08-31; keep in step with
# router_adapter.FREE_TIERS.
FREE_MODELS = [
    "nvidia/nemotron-3.5-lightning",
    "nvidia/nemotron-3-nano-30b",
    "nvidia/llama-3.2-11b-vision",
    "cohere/north-mini-code",
    "poolside/laguna-xs-2.1",
]


def _price(input_price: float, output_price: float, flat_price: float = 0) -> dict[str, float]:
    return {
        "input_price": input_price,
        "output_price": output_price,
        "flat_price": flat_price,
    }


CATALOG = {
    "google/gemini-2.5-flash": _price(0.15, 0.6),
    "google/gemini-2.5-flash-lite": _price(0.1, 0.4),
    "google/gemini-3.5-flash": _price(0.5, 3),
    "google/gemini-3-flash-preview": _price(0.5, 3),
    "google/gemini-3.1-flash-lite": _price(0.25, 1.5),
    "google/gemini-3.1-pro": _price(1.25, 10),
    "openai/gpt-5.4-nano": _price(0.2, 1.25),
    "openai/gpt-5-mini": _price(0.25, 2),
    "openai/gpt-5.3-codex": _price(1.75, 14),
    "anthropic/claude-opus-4.7": _price(5, 25),
    "anthropic/claude-sonnet-5": _price(3, 15),
    "anthropic/claude-fable-5": _price(10, 50),
    "deepseek/deepseek-chat": _price(0.2, 0.4),
    "deepseek/deepseek-v4-pro": _price(0.435, 0.87),
    "moonshot/kimi-k2.7": _price(0.95, 4),
    "xai/grok-4-1-fast-reasoning": _price(0.2, 0.5),
    "xai/grok-4-fast-non-reasoning": _price(0.2, 0.5),
    **{model: _price(0, 0) for model in FREE_MODELS},
}


class TestCatalogResolution:
    def test_heads_eco_with_the_gateway_native_free_tier(self):
        # Since d7bc10c the chains carry gateway-native nvidia/* ids directly,
        # so the adapter's free/*->nvidia/* mapping branch is dormant with the
        # current pin. It stays because pins move independently; the dropped-
        # unpriced-ids test below keeps the drop path honest. (Mirrors the
        # TypeScript SDK's retargeting of the same guard.)
        catalog = {**CATALOG, "nvidia/nemotron-3.5-lightning": _price(0, 0)}

        decision = route("hi", None, 512, catalog, "eco")

        assert "nvidia/nemotron-3.5-lightning" in [decision["model"], *decision["fallbacks"]]
        assert not any(
            model.startswith("free/") for model in [decision["model"], *decision["fallbacks"]]
        )

    def test_drops_free_ids_the_catalog_cannot_price(self):
        # No nvidia/gpt-oss-* rows here: those ids are hidden from /v1/models,
        # and an unmapped free/* id would draw a hard, non-transient 400.
        decision = route("hi", None, 512, CATALOG, "eco")

        assert not any(
            model.startswith("free/") for model in [decision["model"], *decision["fallbacks"]]
        )
        assert decision["model"] in CATALOG

    def test_candidates_lead_with_the_selected_model_and_fallbacks_follow(self):
        decision = route("What is 2+2?", None, 512, CATALOG)

        assert decision["candidates"][0] == decision["model"]
        assert decision["fallbacks"] == decision["candidates"][1:]
        assert decision["model"] not in decision["fallbacks"]


class TestCostMetadata:
    def test_applies_the_base_chain_payment_floor_to_paid_models(self):
        decision = route("What is 2+2?", None, 16, CATALOG)

        assert decision["cost_estimate"] == pytest.approx(BASE_MINIMUM_PAYMENT_USD)

    def test_never_floors_a_free_model_up_to_the_paid_minimum(self):
        decision = route("What is 2+2?", None, 512, CATALOG, "free")

        assert decision["cost_estimate"] == 0
        assert decision["savings"] == pytest.approx(1.0)

    def test_premium_profile_reports_no_savings(self):
        decision = route("Design a distributed ledger", None, 1024, CATALOG, "premium")

        assert decision["savings"] == 0


class TestCapacityFiltering:
    def test_drops_candidates_that_cannot_hold_the_full_conversation(self):
        # 8k output is above several small-output models' ceiling.
        decision = route("Explain this architecture", None, 20_000, CATALOG)

        assert "xai/grok-4-fast-non-reasoning" not in decision["candidates"]

    def test_keeps_models_absent_from_the_capability_snapshot(self):
        catalog = {**CATALOG, "acme/experimental-1": _price(0.1, 0.1)}
        config = {
            **DEFAULT_ROUTING_CONFIG,
            "strategy": "rules",
            "tiers": {
                tier: {"primary": "acme/experimental-1", "fallback": []}
                for tier in DEFAULT_ROUTING_CONFIG["tiers"]
            },
        }
        from blockrun_llm.router_adapter import route_with_catalog

        decision = route_with_catalog("hi", None, 512, catalog, config=config)

        assert decision["model"] == "acme/experimental-1"


class TestFreeProfile:
    @pytest.mark.parametrize(
        "prompt",
        [
            "What is 2+2?",
            "Prove the theorem step by step using mathematical induction",
            "Refactor this TypeScript function and explain the tradeoffs",
            "A" * 5_000,
        ],
    )
    def test_never_selects_a_billable_model(self, prompt):
        decision = route(prompt, None, 512, CATALOG, "free")

        for model in [decision["model"], *decision["fallbacks"]]:
            assert CATALOG[model]["input_price"] == 0
            assert CATALOG[model]["output_price"] == 0

    def test_every_free_tier_keeps_real_fallback_depth(self):
        # Membership alone did not catch the 2026-08 rot: the table stayed
        # internally consistent while the gateway retired four of its five ids,
        # leaving every tier on one model with no fallback. Depth is the signal
        # that survives that, so assert it per tier and across the table.
        for name, tier in FREE_TIERS.items():
            candidates = [tier["primary"], *tier["fallback"]]
            assert len(set(candidates)) >= 3, f"{name} has no fallback depth: {candidates}"

        used = {m for tier in FREE_TIERS.values() for m in [tier["primary"], *tier["fallback"]]}
        assert used == set(FREE_MODELS), sorted(set(FREE_MODELS) ^ used)

    def test_every_free_tier_entry_is_live_in_the_catalog(self):
        # The previous hand-maintained table rotted silently when NVIDIA EOL'd
        # its early free lineup; this asserts the replacement points at models
        # the catalog still prices.
        for tier in FREE_TIERS.values():
            for model in [tier["primary"], *tier["fallback"]]:
                assert model in FREE_MODELS, model

    def test_uses_the_rules_strategy_so_paid_evidence_models_cannot_leak_in(self):
        decision = route(
            "Fix the TypeScript payment retry bug, run tests, and update the patch.",
            None,
            4096,
            CATALOG,
            "free",
        )

        assert decision["method"] == "rules"
        assert "openai/gpt-5.3-codex" not in decision["candidates"]


class TestSdkDecisionShape:
    def test_the_decision_parses_into_the_public_pydantic_model(self):
        decision = route(
            "Which answer is correct?\nA. One\nB. Two\nC. Three\nD. Four", None, 512, CATALOG
        )

        parsed = RoutingDecision(**decision)

        assert parsed.model == decision["model"]
        assert parsed.method == "portfolio"
        assert parsed.router_version == "v3-portfolio"
        assert parsed.task_type == "reasoning_mcq"
        assert parsed.candidates[0] == parsed.model
        assert parsed.candidate_scores
        assert parsed.profile == "auto"

    def test_the_free_profile_decision_also_parses(self):
        parsed = RoutingDecision(**route("hi", None, 512, CATALOG, "free"))

        assert parsed.method == "rules"
        assert parsed.task_type is None


class TestRoutingText:
    def test_reads_the_whole_transcript_for_capacity_and_the_last_user_turn(self):
        view = routing_text(
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "and now?"},
            ]
        )

        assert view["prompt"] == "and now?"
        assert view["system_prompt"] == "You are terse."
        assert view["conversation_chars"] == len("You are terse.") + len("hello") + 2 + len(
            "and now?"
        )
        assert view["has_vision"] is False

    def test_detects_image_parts(self):
        view = routing_text(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    ],
                }
            ]
        )

        assert view["has_vision"] is True
        assert view["conversation_chars"] == len("what is this?")


class TestVirtualModelIds:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("blockrun/auto", "auto"),
            ("BlockRun/Eco", "eco"),
            ("blockrun/premium", "premium"),
            ("google/gemini-3.5-flash", None),
        ],
    )
    def test_maps_virtual_ids_to_profiles(self, model, expected):
        assert routing_profile_for_model(model) == expected
