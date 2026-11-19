"""
The routing cost estimate includes the gateway's per-token platform fee (spec PF-8).

blockrun.ai and sol.blockrun.ai charge per-token paid chat its list
``pricing.input`` / ``pricing.output`` rates times ``1 + platform_fee_percent /
100``. The fee rides on each ``/v1/models`` chat row: inside ``pricing`` on both
gateways, and at the row's top level on blockrun.ai. Older gateways and the
enterprise gateway do not send it, so absent means 0. Free, flat, media and
Decisions rows never pay it.
"""

from __future__ import annotations

import math

import pytest

from blockrun_llm.router_adapter import (
    BASE_MINIMUM_PAYMENT_USD,
    build_model_pricing,
    platform_fee_percent,
    route_with_catalog,
    token_cost_usd,
)
from blockrun_llm.router_core import DEFAULT_MODEL_CAPABILITIES

FEE = 5  # what the rows below publish; the SDK reads it, it never assumes it


class TestPlatformFeePercent:
    def test_reads_the_field_inside_pricing(self):
        row = {
            "id": "a/b",
            "billing_mode": "paid",
            "pricing": {"input": 1, "output": 2, "platform_fee_percent": FEE},
        }
        assert platform_fee_percent(row) == FEE

    def test_falls_back_to_the_top_level_field(self):
        row = {
            "id": "a/b",
            "billing_mode": "paid",
            "platform_fee_percent": FEE,
            "pricing": {"input": 1, "output": 2},
        }
        assert platform_fee_percent(row) == FEE

    def test_prefers_the_nested_field_over_the_top_level_one(self):
        row = {
            "id": "a/b",
            "billing_mode": "paid",
            "platform_fee_percent": 9,
            "pricing": {"input": 1, "output": 2, "platform_fee_percent": 3},
        }
        assert platform_fee_percent(row) == 3

    def test_is_zero_when_the_gateway_does_not_send_it(self):
        # Older gateways, and enterprise (PF-9), publish no fee.
        assert platform_fee_percent({"id": "a/b", "pricing": {"input": 1, "output": 2}}) == 0
        assert platform_fee_percent({"id": "a/b", "inputPrice": 1, "outputPrice": 2}) == 0

    @pytest.mark.parametrize("bad", [None, "5", True, -1, math.nan, math.inf, [5]])
    def test_an_unusable_value_is_zero(self, bad):
        row = {
            "id": "a/b",
            "billing_mode": "paid",
            "pricing": {"input": 1, "platform_fee_percent": bad},
        }
        assert platform_fee_percent(row) == 0

    def test_an_unusable_nested_value_still_falls_back_to_the_top_level(self):
        row = {
            "id": "a/b",
            "billing_mode": "paid",
            "platform_fee_percent": FEE,
            "pricing": {"input": 1, "platform_fee_percent": "oops"},
        }
        assert platform_fee_percent(row) == FEE

    @pytest.mark.parametrize(
        "row",
        [
            # Free rows publish 0 at top level and nothing nested.
            {
                "id": "nvidia/x",
                "billing_mode": "free",
                "platform_fee_percent": FEE,
                "pricing": {"input": 0, "output": 0, "platform_fee_percent": FEE},
            },
            # Flat rows: the flat price is final.
            {
                "id": "openai/gpt-oss-20b",
                "billing_mode": "flat",
                "platform_fee_percent": FEE,
                "pricing": {"flat": 0.002, "platform_fee_percent": FEE},
            },
            {"id": "a/flat", "platform_fee_percent": FEE, "pricing": {"flat": 0.002}},
            # Media rows.
            {
                "id": "openai/gpt-image-2",
                "billing_mode": "per_image",
                "pricing": {"per_image": 0.04, "platform_fee_percent": FEE},
            },
            # Decisions bill per token but pay no platform fee (PF-5).
            {
                "id": "openai/gpt-6-luna-decisions",
                "billing_mode": "per_token",
                "pricing": {"input": 0.1, "output": 0, "platform_fee_percent": FEE},
            },
        ],
        ids=["free", "flat", "flat-no-billing-mode", "media", "decisions"],
    )
    def test_never_applies_outside_per_token_paid_chat(self, row):
        assert platform_fee_percent(row) == 0


class TestPricingMap:
    def test_entries_carry_the_fee(self):
        pricing = build_model_pricing(
            [
                {
                    "id": "base/row",
                    "billing_mode": "paid",
                    "platform_fee_percent": FEE,
                    "pricing": {"input": 1, "output": 2, "platform_fee_percent": FEE},
                },
                {
                    "id": "sol/row",
                    "billing_mode": "paid",
                    "pricing": {"input": 1, "output": 2, "platform_fee_percent": FEE},
                },
                {
                    "id": "enterprise/row",
                    "billing_mode": "paid",
                    "pricing": {"input": 1, "output": 2},
                },
                {
                    "id": "nvidia/free",
                    "billing_mode": "free",
                    "platform_fee_percent": 0,
                    "pricing": {"input": 0, "output": 0},
                },
            ]
        )
        assert pricing["base/row"]["platform_fee_percent"] == FEE
        assert pricing["sol/row"]["platform_fee_percent"] == FEE
        assert pricing["enterprise/row"]["platform_fee_percent"] == 0
        assert pricing["nvidia/free"]["platform_fee_percent"] == 0
        # The list rates are untouched: the fee is applied to the estimate only.
        assert pricing["base/row"]["input_price"] == 1
        assert pricing["base/row"]["output_price"] == 2


class TestTokenCost:
    def test_multiplies_the_list_cost_by_the_fee(self):
        entry = {"input_price": 2, "output_price": 10, "flat_price": 0, "platform_fee_percent": FEE}
        list_cost = (1_000 * 2 + 500 * 10) / 1_000_000
        assert token_cost_usd(entry, 1_000, 500) == pytest.approx(list_cost * (1 + FEE / 100))

    def test_no_fee_field_is_the_list_cost(self):
        entry = {"input_price": 2, "output_price": 10, "flat_price": 0}
        assert token_cost_usd(entry, 1_000, 500) == pytest.approx((1_000 * 2 + 500 * 10) / 1e6)

    def test_flat_and_unpriced_entries_are_not_token_priced(self):
        assert (
            token_cost_usd({"flat_price": 0.002, "platform_fee_percent": FEE}, 1_000, 500) is None
        )
        assert token_cost_usd(None, 1_000, 500) is None
        assert token_cost_usd({"input_price": math.nan, "output_price": 1}, 1_000, 500) is None


PROMPT = "Design a distributed ledger with Byzantine fault tolerance and prove liveness"


def _catalog(**row_extra):
    """Every model the router can rank, priced alike, so whatever it picks is priced."""
    return [
        {
            "id": model,
            "billing_mode": "paid",
            **row_extra.get("top", {}),
            "pricing": {"input": 2, "output": 10, **row_extra.get("nested", {})},
        }
        for model in DEFAULT_MODEL_CAPABILITIES
    ]


def _route(catalog, max_tokens=4_000, profile="auto"):
    return route_with_catalog(
        PROMPT,
        None,
        max_tokens,
        build_model_pricing(catalog),
        routing_profile=profile,
        minimum_payment_usd=BASE_MINIMUM_PAYMENT_USD,
    )


def _list_cost(max_tokens):
    prompt_tokens = math.ceil(len(f" {PROMPT}") / 4)
    return (prompt_tokens * 2 + max_tokens * 10) / 1_000_000


class TestCostEstimate:
    def test_paid_chat_estimate_includes_the_nested_fee(self):
        decision = _route(_catalog(nested={"platform_fee_percent": FEE}))
        assert decision["cost_estimate"] == pytest.approx(_list_cost(4_000) * (1 + FEE / 100))

    def test_top_level_fee_is_used_when_pricing_has_none(self):
        decision = _route(_catalog(top={"platform_fee_percent": FEE}))
        assert decision["cost_estimate"] == pytest.approx(_list_cost(4_000) * (1 + FEE / 100))

    def test_absent_fee_estimates_at_list_price(self):
        # Older gateways and enterprise send no fee: no hidden margin is added.
        decision = _route(_catalog())
        assert decision["cost_estimate"] == pytest.approx(_list_cost(4_000))

    def test_the_fee_does_not_change_which_model_is_picked(self):
        with_fee = _route(_catalog(nested={"platform_fee_percent": FEE}))
        without = _route(_catalog())
        assert with_fee["model"] == without["model"]
        assert with_fee["fallbacks"] == without["fallbacks"]

    def test_free_models_stay_at_zero(self):
        catalog = [
            {
                "id": model,
                "billing_mode": "free",
                "platform_fee_percent": 0,
                "pricing": {"input": 0, "output": 0},
            }
            for model in (
                "nvidia/nemotron-3.5-lightning",
                "nvidia/nemotron-3-ultra-550b",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                "nvidia/nemotron-3.5-bolt",
                "poolside/laguna-xs-2.1",
            )
        ]
        assert _route(catalog, profile="free")["cost_estimate"] == 0

    def test_a_small_paid_call_is_still_floored(self):
        decision = _route(_catalog(nested={"platform_fee_percent": FEE}), max_tokens=1)
        assert decision["cost_estimate"] == pytest.approx(BASE_MINIMUM_PAYMENT_USD)
