"""
Regression tests for the Router Core re-sync to upstream ``273c2af``.

1:1 ports of upstream ``hardening.test.ts`` (the 2026-11 router-core audit:
inputs that produced a wrong classification, a NaN ranking, an unbounded scan
or a fabricated figure) plus the 2026-11-08 capability snapshot checks. The
upstream cases added before ``8773a99`` live in ``test_router_core.py``;
``test_router_core_snapshot.py`` holds the full-decision parity.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from blockrun_llm.router_core import (
    DEFAULT_MODEL_CAPABILITIES,
    DEFAULT_ROUTING_CONFIG,
    RulesStrategy,
    calculate_model_cost,
    classify_by_rules,
    route,
)
from blockrun_llm.router_core.selector import select_model
from blockrun_llm.router_core.tokens import sample_for_scan, scan_limit

SCORING = DEFAULT_ROUTING_CONFIG["scoring"]


def _price(input_price: float, output_price: float) -> dict[str, float]:
    return {"input_price": input_price, "output_price": output_price}


# ─── model-capabilities: 2026-11-08 sync ───


def test_snapshot_retains_a_routed_model_absent_from_the_listing():
    # kimi-k3 left the 2026-11-08 listing but is still served and still routed.
    assert DEFAULT_MODEL_CAPABILITIES["moonshot/kimi-k3"]["supports_tools"] is True


@pytest.mark.parametrize(
    "model",
    ["deepseek/deepseek-chat", "deepseek/deepseek-reasoner", "deepseek/deepseek-v4.1-flash"],
)
def test_snapshot_fails_closed_on_unprobed_vision_tags(model):
    assert DEFAULT_MODEL_CAPABILITIES[model]["supports_vision"] is False


# ─── hardening.test.ts: 2026-11 audit regressions ───

_HARDENING_PRICING = {
    "anthropic/claude-sonnet-5": _price(3, 15),
    "anthropic/claude-opus-4.8": _price(5, 25),
    "openai/gpt-5.3-codex": _price(1.75, 14),
    "openai/gpt-5-mini": _price(0.25, 2),
    "google/gemini-3.5-flash": _price(1.5, 9),
    "google/gemini-2.5-flash": _price(0.3, 2.5),
    "deepseek/deepseek-v4-pro": _price(0.435, 0.87),
    "moonshot/kimi-k3": _price(3, 15),
}


def _hardening(**extra):
    return {"config": DEFAULT_ROUTING_CONFIG, "model_pricing": _HARDENING_PRICING, **extra}


@pytest.mark.parametrize(
    ("prompt", "task_type"),
    [
        ("请修复这个报错", "debug"),
        ("请重构这个模块", "code_edit"),
        ("提取所有字段", "extraction"),
        ("证明这个定理", "reasoning"),
    ],
)
def test_chinese_task_keywords_classify(prompt, task_type):
    assert route(prompt, None, 512, _hardening())["task_type"] == task_type


class TestCodeAgentDetection:
    WEB = {"has_tools": True, "requires_tools": True, "tool_count": 1, "tool_names": ["web_search"]}

    @pytest.mark.parametrize(
        "prompt",
        [
            "look up current java coffee bean prices online",
            "search the web for python snake habitat facts",
        ],
    )
    def test_bare_language_name_in_a_web_lookup_is_not_code(self, prompt):
        decision = route(prompt, None, 512, _hardening(**self.WEB))
        assert decision["task_type"] == "tool_agent"
        assert decision["model"] != "openai/gpt-5.3-codex"

    @pytest.mark.parametrize(
        "prompt",
        [
            "Fix the TypeScript payment retry bug, run tests, and update the patch.",
            "Write a python script that parses the CSV and run it",
            "```ts\nconst x = 1\n``` refactor this",
        ],
    )
    def test_real_code_work_with_tools_stays_code_agent(self, prompt):
        decision = route(prompt, None, 512, _hardening(has_tools=True, requires_tools=True))
        assert decision["task_type"] == "code_agent"


class TestParallelToolDetection:
    def test_does_not_read_add_out_of_address(self):
        decision = route(
            "Please update my address to 12 Main St and send me a confirmation",
            None,
            512,
            _hardening(
                has_tools=True,
                requires_tools=True,
                tool_count=2,
                tool_names=["add_item", "send_email"],
            ),
        )
        assert decision["task_type"] == "tool_agent"

    def test_still_detects_two_operation_verbs(self):
        decision = route(
            "Add milk to my list and delete the eggs entry",
            None,
            512,
            _hardening(
                has_tools=True,
                requires_tools=True,
                tool_count=2,
                tool_names=["add_task", "delete_task"],
            ),
        )
        assert decision["task_type"] == "tool_agent_parallel"


def test_shouted_words_are_not_reservation_ids():
    airline = _hardening(
        has_tools=True,
        requires_tools=True,
        tool_count=3,
        tool_names=["get_reservation_details", "cancel_reservation", "search_flights"],
    )
    assert "agentRisk=standard" in route("CANCEL MY FLIGHT please", None, 512, airline)["reasoning"]
    two_ids = route("Cancel reservations ZFA04Y and 4WQ150", None, 512, airline)
    assert "agentRisk=high" in two_ids["reasoning"]


class TestRankingRobustness:
    def test_unpriced_candidate_does_not_win_on_cost(self):
        pricing = dict(_HARDENING_PRICING)
        del pricing["deepseek/deepseek-v4-pro"]
        decision = route(
            "Return the pricier of my two orders and exchange the other one for a larger size",
            None,
            1024,
            _hardening(
                model_pricing=pricing,
                has_tools=True,
                requires_tools=True,
                tool_count=4,
                tool_names=[
                    "get_order_details",
                    "return_delivered_order_items",
                    "exchange_delivered_order_items",
                    "get_product_details",
                ],
            ),
        )
        unpriced = next(
            row
            for row in decision["candidate_scores"]
            if row["model"] == "deepseek/deepseek-v4-pro"
        )
        assert unpriced["cost"] == 0

    def test_ignores_a_malformed_performance_profile(self):
        decision = route(
            "Summarize the plot of Hamlet in two sentences.",
            None,
            512,
            _hardening(
                model_performance={
                    "google/gemini-2.5-flash": {
                        "measured_at": "2026-11-01T00:00:00Z",
                        "latency_ms": 900,
                        "output_tokens_per_second": math.nan,
                        "samples": math.nan,
                    }
                }
            ),
        )
        for row in decision["candidate_scores"]:
            assert math.isfinite(row["score"])


class TestCostFields:
    TIERS = {
        tier: {"primary": "acme/nan", "fallback": []}
        for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
    }
    BAD = {"acme/nan": {"input_price": math.nan, "output_price": 1}}

    def test_non_finite_price_is_unknown_not_nan(self):
        decision = select_model("SIMPLE", 1, "rules", "", self.TIERS, self.BAD, 1000, 1000)
        assert math.isfinite(decision["cost_estimate"])
        assert decision["savings"] == 0
        cost = calculate_model_cost("acme/nan", self.BAD, 1000, 1000)
        assert math.isfinite(cost["cost_estimate"])
        assert cost["savings"] == 0

    def test_no_savings_for_an_unpriced_model(self):
        assert calculate_model_cost("acme/unknown", {}, 1000, 1000)["savings"] == 0


class TestPromotions:
    NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)

    @staticmethod
    def _config(start: str, end: str):
        return {
            **DEFAULT_ROUTING_CONFIG,
            "strategy": "rules",
            "promotions": [
                {
                    "name": "typo",
                    "start_date": start,
                    "end_date": end,
                    "tier_overrides": {"SIMPLE": {"primary": "acme/promo"}},
                }
            ],
        }

    @pytest.mark.parametrize(
        ("start", "end"), [("2026-13-01", "2026-12-01"), ("2026-11-01", "soon")]
    )
    def test_ignores_unparseable_dates(self, start, end):
        decision = RulesStrategy().route(
            "hi",
            None,
            64,
            {
                "config": self._config(start, end),
                "model_pricing": _HARDENING_PRICING,
                "now": self.NOW,
            },
        )
        assert decision["model"] != "acme/promo"

    def test_applies_a_valid_window(self):
        decision = RulesStrategy().route(
            "hi",
            None,
            64,
            {
                "config": self._config("2026-11-01", "2026-11-15"),
                "model_pricing": _HARDENING_PRICING,
                "now": self.NOW,
            },
        )
        assert decision["tier_configs"][decision["tier"]]["primary"] == "acme/promo"


class TestScanBounds:
    def test_bounds_the_sample_at_the_smallest_limit(self):
        long = "x" * 100_000
        assert len(sample_for_scan(long, 1)) <= 2
        assert len(sample_for_scan(long, scan_limit(0))) <= 2

    def test_non_numeric_truncation_uses_the_default_bound(self):
        assert scan_limit(math.nan) == 8_000
        assert len(sample_for_scan("x" * 100_000, scan_limit(math.nan))) <= 8_001


class TestRulesSignals:
    @staticmethod
    def _joined(prompt: str) -> str:
        return " | ".join(classify_by_rules(prompt, None, 20, SCORING)["signals"])

    def test_sentence_ending_in_a_number_is_not_multi_step(self):
        assert "multi-step" not in self._joined("I was born in 1990. How old am I?")

    def test_numbered_list_is_multi_step(self):
        assert "multi-step" in self._joined("Do this:\n1. install it\n2. configure it")

    def test_counts_full_width_question_marks(self):
        assert "4 questions" in self._joined("这是什么？为什么？怎么做？什么时候？")


def test_japanese_extraction_does_not_get_the_mandarin_band():
    decision = route(
        "次の文章から氏名と住所を抽出して、JSONで出力してください。山田太郎、東京都。",
        None,
        512,
        _hardening(requires_structured_output=True),
    )
    assert decision["task_type"] == "extraction"
    kimi = next(
        (row for row in decision["candidate_scores"] if row["model"] == "moonshot/kimi-k3"), None
    )
    if kimi is not None:
        assert kimi["quality"] < 1
