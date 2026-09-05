"""
Router Core — deterministic, constraint-first model routing.

Python port of `@blockrun/router-core <https://github.com/BlockRunAI/router-core>`_
(upstream commit ``5ee7c23``), the same routing engine the TypeScript SDK and
the BlockRun gateway use. The package is deliberately product-neutral: task
classification, hard capability filtering, portfolio scoring, ordered
fallbacks, and routing configuration. It contains no wallet, gateway client,
proxy server, agent loop, payment handling or telemetry transport — the SDK
supplies those through :mod:`blockrun_llm.router_adapter`.

Hosts provide request capabilities and current model pricing, and may override
model capability and performance observations without adding a network call on
the routing hot path.

Usage::

    from blockrun_llm.router_core import DEFAULT_ROUTING_CONFIG, route

    decision = route(prompt, system_prompt, max_output_tokens, {
        "config": DEFAULT_ROUTING_CONFIG,
        "model_pricing": pricing,
        "has_tools": True,
        "requires_tools": True,
    })

Routing is local and deterministic for identical inputs, configuration, model
metadata, and time.
"""

from __future__ import annotations

from .config import DEFAULT_ROUTING_CONFIG
from .model_capabilities import DEFAULT_MODEL_CAPABILITIES
from .model_profiles import HISTORICAL_MODEL_PROFILES, LIVE_MODEL_PROFILES
from .portfolio import PortfolioStrategy, classify_task
from .rules import classify_by_rules
from .selector import (
    calculate_model_cost,
    filter_by_exclude_list,
    filter_by_tool_calling,
    filter_by_vision,
    filter_candidates_by_capacity,
    get_fallback_chain,
    get_fallback_chain_filtered,
)
from .strategy import (
    RouterStrategy,
    RulesStrategy,
    apply_unavailable_models,
    get_strategy,
    register_strategy,
)
from .tool_intent import infer_tool_requirement
from .types import (
    Capacity,
    ModelCapabilities,
    ModelPerformanceProfile,
    ModelPricing,
    RouterOptions,
    RoutingConfig,
    RoutingDecision,
    RoutingProfile,
    TaskType,
    Tier,
    TierConfig,
)

# Registered here instead of in strategy.py so PortfolioStrategy can reuse the
# stable RulesStrategy without introducing a module cycle.
register_strategy(PortfolioStrategy())


def route(
    prompt: str,
    system_prompt: str | None,
    max_output_tokens: int,
    options: RouterOptions,
) -> RoutingDecision:
    """Route a request to the cheapest capable model.

    Delegates to the configured strategy (``PortfolioStrategy`` by default).
    """
    strategy = get_strategy(options["config"].get("strategy") or "portfolio")
    return strategy.route(prompt, system_prompt, max_output_tokens, options)


__all__ = [
    "DEFAULT_MODEL_CAPABILITIES",
    "DEFAULT_ROUTING_CONFIG",
    "HISTORICAL_MODEL_PROFILES",
    "LIVE_MODEL_PROFILES",
    "Capacity",
    "ModelCapabilities",
    "ModelPerformanceProfile",
    "ModelPricing",
    "PortfolioStrategy",
    "RouterOptions",
    "RouterStrategy",
    "RoutingConfig",
    "RoutingDecision",
    "RoutingProfile",
    "RulesStrategy",
    "TaskType",
    "Tier",
    "TierConfig",
    "apply_unavailable_models",
    "calculate_model_cost",
    "classify_by_rules",
    "classify_task",
    "filter_by_exclude_list",
    "filter_by_tool_calling",
    "filter_by_vision",
    "filter_candidates_by_capacity",
    "get_fallback_chain",
    "get_fallback_chain_filtered",
    "get_strategy",
    "infer_tool_requirement",
    "register_strategy",
    "route",
]
