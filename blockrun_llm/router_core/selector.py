"""
Tier → Model Selection

Python port of ``@blockrun/router-core`` ``selector.ts``.

Maps a classification tier to the cheapest capable model and builds
RoutingDecision metadata with cost estimates and savings.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from .types import Capacity, Method, ModelPricing, RoutingDecision, Tier, TierConfig

# The savings baseline is a price anchor, not "the current flagship" — it is
# deliberately NOT bumped every time a new Opus ships. Opus 4.7, 4.8 and 5 all
# bill $5/$25, so moving it would change no reported number while breaking
# comparability with historical journal entries. Only move it if the Opus tier
# itself is repriced.
BASELINE_MODEL_ID = "anthropic/claude-opus-4.7"

# Hardcoded fallback: Claude Opus 4.7 pricing (per 1M tokens), used when the
# baseline model is absent from the dynamic pricing map.
BASELINE_INPUT_PRICE = 5.0
BASELINE_OUTPUT_PRICE = 25.0

# Server-side margin applied to all x402 payments (must match the blockrun
# server's MARGIN_PERCENT).
SERVER_MARGIN_PERCENT = 5
# Minimum payment enforced by the CDP Facilitator (must match the blockrun
# server's MIN_PAYMENT_USD).
MIN_PAYMENT_USD = 0.001


def _flat_price(pricing: ModelPricing | None) -> float | None:
    """Active promo flat price, or ``None`` for per-token billing.

    The catalog reports ``flat_price: 0`` for per-token models where the
    TypeScript host omits the field, so a falsy value means "not flat".
    """
    if not pricing:
        return None
    flat = pricing.get("flat_price")
    return float(flat) if flat else None


def _baseline_cost(
    model_pricing: Mapping[str, ModelPricing],
    estimated_input_tokens: int,
    max_output_tokens: int,
) -> float:
    """What the premium reference model would cost for the same request."""
    opus_pricing = model_pricing.get(BASELINE_MODEL_ID)
    opus_input_price = (opus_pricing or {}).get("input_price", BASELINE_INPUT_PRICE)
    opus_output_price = (opus_pricing or {}).get("output_price", BASELINE_OUTPUT_PRICE)
    baseline_input = (estimated_input_tokens / 1_000_000) * opus_input_price
    baseline_output = (max_output_tokens / 1_000_000) * opus_output_price
    return baseline_input + baseline_output


def _savings(cost_estimate: float, baseline_cost: float, routing_profile: str | None) -> float:
    # Premium profile doesn't calculate savings (it's about quality, not cost).
    if routing_profile == "premium":
        return 0.0
    if baseline_cost > 0:
        return max(0.0, (baseline_cost - cost_estimate) / baseline_cost)
    return 0.0


def select_model(
    tier: Tier,
    confidence: float,
    method: Method,
    reasoning: str,
    tier_configs: Mapping[str, TierConfig],
    model_pricing: Mapping[str, ModelPricing],
    estimated_input_tokens: int,
    max_output_tokens: int,
    routing_profile: str | None = None,
    agentic_score: float | None = None,
) -> RoutingDecision:
    """Select the primary model for a tier and build the RoutingDecision."""
    tier_config = tier_configs[tier]
    model = tier_config["primary"]
    pricing = model_pricing.get(model)

    flat = _flat_price(pricing)
    if flat is not None:
        cost_estimate = flat
    else:
        input_price = (pricing or {}).get("input_price", 0)
        output_price = (pricing or {}).get("output_price", 0)
        cost_estimate = (estimated_input_tokens / 1_000_000) * input_price + (
            max_output_tokens / 1_000_000
        ) * output_price

    baseline_cost = _baseline_cost(model_pricing, estimated_input_tokens, max_output_tokens)

    decision: RoutingDecision = {
        "model": model,
        "tier": tier,
        "confidence": confidence,
        "method": method,
        "reasoning": reasoning,
        "cost_estimate": cost_estimate,
        "baseline_cost": baseline_cost,
        "savings": _savings(cost_estimate, baseline_cost, routing_profile),
    }
    if agentic_score is not None:
        decision["agentic_score"] = agentic_score
    return decision


def get_fallback_chain(tier: Tier, tier_configs: Mapping[str, TierConfig]) -> list[str]:
    """Get the ordered fallback chain for a tier: ``[primary, *fallbacks]``."""
    config = tier_configs[tier]
    return [config["primary"], *config["fallback"]]


def calculate_model_cost(
    model: str,
    model_pricing: Mapping[str, ModelPricing],
    estimated_input_tokens: int,
    max_output_tokens: int,
    routing_profile: str | None = None,
) -> dict[str, float]:
    """Calculate cost for a specific model (used when a fallback model is used).

    Includes the server margin and the facilitator minimum so the estimate
    matches the actual x402 charge.
    """
    pricing = model_pricing.get(model)

    flat = _flat_price(pricing)
    if flat is not None:
        # Active promo: fixed cost per request
        cost_estimate = max(flat * (1 + SERVER_MARGIN_PERCENT / 100), MIN_PAYMENT_USD)
    else:
        # Defensive: guard against undefined price fields (not just absent pricing)
        input_price = (pricing or {}).get("input_price", 0)
        output_price = (pricing or {}).get("output_price", 0)
        input_cost = (estimated_input_tokens / 1_000_000) * input_price
        output_cost = (max_output_tokens / 1_000_000) * output_price
        cost_estimate = max(
            (input_cost + output_cost) * (1 + SERVER_MARGIN_PERCENT / 100), MIN_PAYMENT_USD
        )

    baseline_cost = _baseline_cost(model_pricing, estimated_input_tokens, max_output_tokens)
    return {
        "cost_estimate": cost_estimate,
        "baseline_cost": baseline_cost,
        "savings": _savings(cost_estimate, baseline_cost, routing_profile),
    }


def filter_by_tool_calling(
    models: list[str],
    has_tools: bool,
    supports_tool_calling: Callable[[str], bool],
) -> list[str]:
    """Keep only models that support tool calling when the request has tools.

    When every model lacks tool calling the full list is returned unchanged —
    better to let the API error than to produce an empty chain.
    """
    if not has_tools:
        return models
    filtered = [model for model in models if supports_tool_calling(model)]
    return filtered if filtered else models


def filter_by_vision(
    models: list[str],
    has_vision: bool,
    supports_vision: Callable[[str], bool],
) -> list[str]:
    """Keep only vision-capable models when the request carries images.

    Same empty-chain safety net as :func:`filter_by_tool_calling`.
    """
    if not has_vision:
        return models
    filtered = [model for model in models if supports_vision(model)]
    return filtered if filtered else models


def filter_by_exclude_list(models: list[str], exclude_list: Iterable[str]) -> list[str]:
    """Remove user-excluded models, with the same empty-chain safety net."""
    excluded = set(exclude_list)
    if not excluded:
        return models
    filtered = [model for model in models if model not in excluded]
    return filtered if filtered else models


def get_fallback_chain_filtered(
    tier: Tier,
    tier_configs: Mapping[str, TierConfig],
    estimated_total_tokens: int,
    get_context_window: Callable[[str], int | None],
) -> list[str]:
    """Get the tier's fallback chain filtered by context length.

    Models with an unknown context window are kept (let the API reject them),
    and an entirely filtered-out chain falls back to the full chain.
    """
    full_chain = get_fallback_chain(tier, tier_configs)

    filtered = []
    for model_id in full_chain:
        context_window = get_context_window(model_id)
        # Unknown model - include it (let API reject if needed)
        # Add 10% buffer for safety
        if context_window is None or context_window >= estimated_total_tokens * 1.1:
            filtered.append(model_id)

    return filtered if filtered else full_chain


def filter_candidates_by_capacity(
    models: list[str],
    estimated_input_tokens: int,
    requested_output_tokens: int,
    get_capabilities: Callable[[str], Capacity | None],
) -> list[str]:
    """Filter an already-ranked candidate list by context and output capacity.

    Unlike :func:`get_fallback_chain_filtered` this supports the V3 portfolio
    order and returns an empty list when nothing fits.
    """
    filtered = []
    for model_id in models:
        capabilities = get_capabilities(model_id)
        if not capabilities:
            filtered.append(model_id)
            continue
        if (
            capabilities["context_window"]
            >= (estimated_input_tokens + requested_output_tokens) * 1.1
            and capabilities["max_output"] >= requested_output_tokens
        ):
            filtered.append(model_id)
    return filtered
