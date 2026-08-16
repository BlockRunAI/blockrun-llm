"""
Smart Router for BlockRun LLM SDK

Thin compatibility shim over :mod:`blockrun_llm.router_core` — the Python port
of `@blockrun/router-core <https://github.com/BlockRunAI/router-core>`_, the
same routing engine the TypeScript SDK and the BlockRun gateway run.

Routing decisions are local and deterministic (<1ms, no extra model call): the
core classifies the task shape, applies capability constraints as hard filters,
and ranks an ordered candidate portfolio; :mod:`blockrun_llm.router_adapter`
then resolves that ranking against the live catalog.

Until 1.10.1 this module carried its own hand-maintained tier tables and a
14-dimension scorer. Those have been replaced by the shared core, so tier
configuration now lives in :data:`blockrun_llm.router_core.DEFAULT_ROUTING_CONFIG`
(and, for the SDK-only ``free`` profile, in
:data:`blockrun_llm.router_adapter.FREE_TIERS`).

Usage:
    from blockrun_llm import LLMClient

    client = LLMClient()
    result = client.smart_chat("What is 2+2?")
    print(result.response)   # '4'
    print(result.model)      # 'google/gemini-3.5-flash'
    print(f"Saved {result.routing.savings * 100:.0f}%")
"""

from __future__ import annotations

from collections.abc import Mapping

from .router_adapter import (
    BASE_MINIMUM_PAYMENT_USD,
    FREE_TIERS,
    ResolvedRoutingDecision,
    route_with_catalog,
)
from .router_core import DEFAULT_ROUTING_CONFIG
from .router_core import classify_by_rules as _classify_by_rules
from .router_core.types import ModelPricing, ScoringResult, Tier, TierConfig
from .types import RoutingProfile

#: Back-compat alias — this module used to define its own decision TypedDict.
RoutingDecision = ResolvedRoutingDecision

__all__ = [
    "DEFAULT_ROUTING_CONFIG",
    "FREE_TIERS",
    "ModelPricing",
    "ResolvedRoutingDecision",
    "RoutingDecision",
    "RoutingProfile",
    "ScoringResult",
    "Tier",
    "TierConfig",
    "classify_by_rules",
    "route",
]


def classify_by_rules(
    prompt: str,
    system_prompt: str | None = None,
    estimated_tokens: int | None = None,
) -> ScoringResult:
    """Classify a prompt into a tier with the shared 15-dimension scorer.

    ``estimated_tokens`` defaults to the ~4-chars-per-token estimate the router
    itself uses.
    """
    if estimated_tokens is None:
        full_text = f"{system_prompt or ''} {prompt}"
        estimated_tokens = -(-len(full_text) // 4)  # ceil
    return _classify_by_rules(
        prompt, system_prompt, estimated_tokens, DEFAULT_ROUTING_CONFIG["scoring"]
    )


def route(
    prompt: str,
    system_prompt: str | None,
    max_output_tokens: int,
    model_pricing: Mapping[str, ModelPricing],
    routing_profile: RoutingProfile = "auto",
    *,
    minimum_payment_usd: float = BASE_MINIMUM_PAYMENT_USD,
) -> ResolvedRoutingDecision:
    """
    Route a request to the cheapest capable model.

    Args:
        prompt: User message
        system_prompt: Optional system prompt
        max_output_tokens: Max tokens to generate
        model_pricing: Dict of model_id -> {"input_price": x, "output_price": y,
            "flat_price": z}, as built from ``/v1/models``
        routing_profile: "free" | "eco" | "auto" | "premium"
        minimum_payment_usd: x402 per-request floor applied to the cost
            estimate; defaults to the Base chain's $0.002

    Returns:
        The routing decision: selected ``model``, the ordered ``fallbacks``
        chain, ``tier``, ``confidence``, ``method``, ``reasoning``, cost
        metadata, plus the portfolio's ``candidates`` / ``candidate_scores`` /
        ``task_type`` when the portfolio strategy ran.
    """
    return route_with_catalog(
        prompt,
        system_prompt,
        max_output_tokens,
        model_pricing,
        routing_profile=routing_profile,
        minimum_payment_usd=minimum_payment_usd,
    )
