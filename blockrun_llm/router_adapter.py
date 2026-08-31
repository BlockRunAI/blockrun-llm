"""
Host glue between the BlockRun catalog and :mod:`blockrun_llm.router_core`.

Python port of the TypeScript SDK's ``src/router-adapter.ts``. Router Core is
deliberately product-neutral, so everything BlockRun-specific lives here:

* catalog id resolution (the router's ``free/*`` namespace vs the gateway's
  ``nvidia/*`` ids),
* the x402 per-request payment floors used for cost metadata,
* capacity filtering against the full conversation, not just the last message,
* the SDK-only ``free`` routing profile, which Router Core does not model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .router_core import (
    DEFAULT_MODEL_CAPABILITIES,
    DEFAULT_ROUTING_CONFIG,
    calculate_model_cost,
    filter_candidates_by_capacity,
    get_fallback_chain,
    route,
)
from .router_core.types import (
    Capacity,
    ModelPricing,
    RouterOptions,
    RoutingConfig,
    RoutingDecision,
    TierConfig,
)


class ResolvedRoutingDecision(RoutingDecision, total=False):
    """A Router Core decision resolved against the live BlockRun catalog.

    Adds ``fallbacks`` — the remaining candidates in ranked order, which
    ``chat()`` walks when an upstream fails transiently.
    """

    fallbacks: list[str]


#: Virtual model ids that select a routing profile instead of a concrete model.
AUTO_ROUTING_PROFILES: Mapping[str, str] = {
    "blockrun/auto": "auto",
    "blockrun/eco": "eco",
    "blockrun/premium": "premium",
}

# x402 per-request payment floors, used only for cost METADATA (the real charge
# is always the gateway's 402 quote). Free models settle at $0 and are never
# floored.
BASE_MINIMUM_PAYMENT_USD = 0.002
SOLANA_MINIMUM_PAYMENT_USD = 0.001

#: The BlockRun free tier is a gateway concept, not a Router Core profile: the
#: core's tiers rank paid models by task affinity, and its evidence candidates
#: are paid ids. ``routing_profile="free"`` therefore routes on the rules
#: strategy over this tier table, and the adapter additionally drops any
#: candidate the catalog does not price at $0.
#:
#: Refreshed 2026-08-31. Every id below was verified by asking the gateway for
#: it twice and reading back the ``model`` field of the reply, because a 200 is
#: not proof: blockrun server-redirects a retired free id to a live one, so a
#: dead rung answers normally while quietly serving something else. That is the
#: shape that defeats a host's ``/exclude``, and it is why the previous table
#: went stale unnoticed. Substituting on 2026-08-31, hence absent here:
#: ``step-3.7-flash``, ``nemotron-nano-9b-v2``, ``nemotron-nano-12b-v2-vl`` and
#: ``mistral-nemotron`` (retired upstream 2026-08-30), plus ``nemotron-3-ultra-550b``
#: and ``nemotron-3-nano-omni-30b-a3b-reasoning`` — the latter two still list at
#: $0 in ``/v1/models`` but both answer as ``nemotron-3-nano-30b``.
#:
#: The table is no longer NVIDIA-only: ``cohere/north-mini-code`` and
#: ``poolside/laguna-xs-2.1`` serve at $0 and carry the free coding load.
#: ``gpt-oss-120b/20b`` stay out — proxy-only ids with no catalog price, under
#: the NVIDIA free tier's prompt-retention policy.
FREE_TIERS: dict[str, TierConfig] = {
    "SIMPLE": {
        # Fastest free model (~121 tok/s), and latency is the only axis that
        # separates free rungs — they all cost $0.
        "primary": "nvidia/nemotron-3-nano-30b",  # 131K ctx
        "fallback": [
            "nvidia/nemotron-3.5-lightning",
            "nvidia/llama-3.2-11b-vision",
            "poolside/laguna-xs-2.1",
        ],
    },
    "MEDIUM": {
        "primary": "nvidia/nemotron-3.5-lightning",  # 1M ctx — free tier flagship
        "fallback": [
            "nvidia/nemotron-3-nano-30b",
            "poolside/laguna-xs-2.1",
            "cohere/north-mini-code",
        ],
    },
    "COMPLEX": {
        # Only free model above 256K, so it absorbs long inputs; the vision rung
        # behind it absorbs multi-modal ones.
        "primary": "nvidia/nemotron-3.5-lightning",
        "fallback": [
            "cohere/north-mini-code",  # 256K ctx
            "nvidia/nemotron-3-nano-30b",
            "nvidia/llama-3.2-11b-vision",  # free vision
        ],
    },
    "REASONING": {
        "primary": "nvidia/nemotron-3.5-lightning",
        "fallback": [
            "nvidia/nemotron-3-nano-30b",
            "cohere/north-mini-code",
            "poolside/laguna-xs-2.1",
        ],
    },
}


def build_model_pricing(models: list[dict[str, Any]]) -> dict[str, ModelPricing]:
    """Build the router's pricing map from a ``/v1/models`` payload.

    Shared by every client (Base and Solana, sync and async) so the four copies
    cannot drift. Rows the catalog marks unavailable are skipped: a model that
    cannot serve a request must not win routing, since every call to it would
    fail with a non-transient error.

    The catalog uses the nested ``pricing.input`` / ``pricing.output`` shape;
    older snapshots used top-level ``inputPrice`` / ``outputPrice``. Both are
    accepted so the SDK keeps working through backend transitions.
    """
    pricing: dict[str, ModelPricing] = {}
    for model in models:
        if model.get("available") is False:
            continue
        model_id = model.get("id", "")
        if not model_id:
            continue
        block = model.get("pricing") or {}
        input_price = block.get("input", model.get("inputPrice", model.get("input_price", 0)))
        output_price = block.get("output", model.get("outputPrice", model.get("output_price", 0)))
        flat_price = block.get("flat", model.get("flatPrice", model.get("flat_price", 0)))
        pricing[model_id] = {
            "input_price": float(input_price or 0),
            "output_price": float(output_price or 0),
            "flat_price": float(flat_price or 0),
        }
    return pricing


def routing_profile_for_model(model: str) -> str | None:
    """Map a ``blockrun/auto``-style virtual model id to a routing profile."""
    return AUTO_ROUTING_PROFILES.get(model.lower())


def _is_free(pricing: ModelPricing | None) -> bool:
    if pricing is None:
        return False
    return (
        pricing.get("input_price", 0) == 0
        and pricing.get("output_price", 0) == 0
        and not pricing.get("flat_price")
    )


def _capacity(model_id: str) -> Capacity | None:
    capabilities = DEFAULT_MODEL_CAPABILITIES.get(model_id)
    if capabilities is None:
        return None
    return {
        "context_window": capabilities["context_window"],
        "max_output": capabilities["max_output_tokens"],
    }


def _free_config(config: RoutingConfig) -> RoutingConfig:
    """A rules-only config whose every profile lands on the free tier table."""
    free_config: RoutingConfig = dict(config)  # type: ignore[assignment]
    free_config["strategy"] = "rules"
    free_config["tiers"] = FREE_TIERS
    free_config["eco_tiers"] = FREE_TIERS
    free_config["premium_tiers"] = FREE_TIERS
    free_config["agentic_tiers"] = FREE_TIERS
    # Promotions promote paid models; they must never leak into the free tier.
    free_config["promotions"] = []
    return free_config


def routing_text(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract the routing view of a chat transcript.

    Returns ``prompt`` (last user text), ``system_prompt``, ``conversation_chars``
    (the FULL transcript size — capacity checks must see the whole conversation,
    not just the last user message) and ``has_vision``.
    """
    system_parts = [
        message["content"]
        for message in messages
        if message.get("role") == "system" and isinstance(message.get("content"), str)
    ]
    system_prompt = "\n".join(system_parts) or None

    last_user = next(
        (
            message["content"]
            for message in reversed(list(messages))
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        None,
    )
    last_text = next(
        (
            message["content"]
            for message in reversed(list(messages))
            if isinstance(message.get("content"), str)
        ),
        None,
    )

    conversation_chars = 0
    has_vision = False
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            conversation_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") in ("image_url", "image"):
                    has_vision = True
                text = part.get("text")
                if isinstance(text, str):
                    conversation_chars += len(text)

    return {
        "prompt": last_user if last_user is not None else (last_text or ""),
        "system_prompt": system_prompt,
        "conversation_chars": conversation_chars,
        "has_vision": has_vision,
    }


def route_with_catalog(
    prompt: str,
    system_prompt: str | None,
    max_output_tokens: int,
    model_pricing: Mapping[str, ModelPricing],
    *,
    routing_profile: str | None = None,
    requires_structured_output: bool = False,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any = None,
    minimum_payment_usd: float = SOLANA_MINIMUM_PAYMENT_USD,
    conversation_chars: int | None = None,
    has_vision: bool = False,
    config: RoutingConfig | None = None,
    now: Any = None,
) -> ResolvedRoutingDecision:
    """Route a request and resolve the ranking against the live catalog.

    ``routing_profile`` accepts Router Core's ``"eco" | "auto" | "premium"``
    plus the SDK-only ``"free"``.
    """
    tool_list = list(tools or [])
    if tool_choice == "none":
        requires_tools: bool | None = False
    elif tool_choice == "required" or isinstance(tool_choice, Mapping):
        requires_tools = True
    else:
        requires_tools = None

    is_free_profile = routing_profile == "free"
    active_config = config or DEFAULT_ROUTING_CONFIG
    if is_free_profile:
        active_config = _free_config(active_config)
    core_profile = None if is_free_profile else routing_profile

    options: RouterOptions = {
        "config": active_config,
        "model_pricing": model_pricing,
        "routing_profile": core_profile,  # type: ignore[typeddict-item]
        "has_tools": len(tool_list) > 0,
        "tool_count": len(tool_list),
        "tool_names": [
            tool.get("function", {}).get("name", "")
            for tool in tool_list
            if isinstance(tool.get("function"), Mapping)
        ],
        "has_vision": has_vision,
        "requires_structured_output": requires_structured_output,
    }
    if requires_tools is not None:
        options["requires_tools"] = requires_tools
    if now is not None:
        options["now"] = now

    decision = route(prompt, system_prompt, max_output_tokens, options)

    # Turn the ranking into a gateway-callable list. The ranking is trusted
    # as-is — including ids withheld from /v1/models (e.g. moonshot/kimi-k2.7),
    # which the gateway serves by direct id — with one exception: the router
    # names its free tier `free/<model>`, a namespace resolved by ClawRouter's
    # proxy. The gateway's ids are `nvidia/<model>`, and an unmapped `free/*` id
    # draws a hard 400 (non-transient, so the fallback chain would never
    # engage). Map `free/*` to its catalog-listed `nvidia/*` id and drop it when
    # there is none (the proxy-only gpt-oss pair).
    tier_configs = decision.get("tier_configs") or active_config["tiers"]
    ranked = decision.get("candidates") or [
        decision["model"],
        *get_fallback_chain(decision["tier"], tier_configs),
    ]
    callable_models: list[str] = []
    for model_id in ranked:
        if not model_id.startswith("free/"):
            resolved: str | None = model_id
        else:
            nvidia_id = f"nvidia/{model_id[5:]}"
            resolved = nvidia_id if nvidia_id in model_pricing else None
        if resolved and resolved not in callable_models:
            callable_models.append(resolved)

    if is_free_profile:
        # Belt and braces: the free profile must never emit a billable model,
        # even if a host config or promotion smuggles one into the tier table.
        free_only = [
            model_id for model_id in callable_models if _is_free(model_pricing.get(model_id))
        ]
        if free_only:
            callable_models = free_only

    # Capacity check against the FULL conversation, not just the routing prompt
    # — an agent transcript can be 100x the last user message, and a context
    # overflow is a non-transient 400 the fallback chain won't save. Models
    # unknown to the capability snapshot are kept (benefit of the doubt).
    estimated_input_tokens = math.ceil(
        max(conversation_chars or 0, len(f"{system_prompt or ''} {prompt}")) / 4
    )
    fitting = filter_candidates_by_capacity(
        callable_models, estimated_input_tokens, max_output_tokens, _capacity
    )
    available_candidates = fitting if fitting else callable_models

    # If nothing survived (a chain of proxy-only free ids), call the router's
    # pick as-is so the gateway's real error surfaces rather than an invented
    # one here.
    model = available_candidates[0] if available_candidates else decision["model"]

    costs = calculate_model_cost(
        model, model_pricing, estimated_input_tokens, max_output_tokens, routing_profile
    )
    # Free models settle at $0 (no payment is signed) — never floor them up to
    # the paid minimum. Detected from the catalog pricing, because Router Core's
    # calculate_model_cost applies its own internal floor even to $0 models.
    entry = model_pricing.get(model)
    is_free = _is_free(entry) if entry is not None else False
    cost_estimate = 0.0 if is_free else max(costs["cost_estimate"], minimum_payment_usd)
    baseline_cost = costs["baseline_cost"]
    if routing_profile == "premium" or baseline_cost <= 0:
        savings = 0.0
    elif entry is not None:
        savings = max(0.0, (baseline_cost - cost_estimate) / baseline_cost)
    else:
        savings = decision["savings"]

    resolved_decision: ResolvedRoutingDecision = dict(decision)  # type: ignore[assignment]
    resolved_decision["baseline_cost"] = baseline_cost
    resolved_decision["cost_estimate"] = cost_estimate
    resolved_decision["savings"] = savings
    resolved_decision["model"] = model
    if model != decision["model"]:
        resolved_decision["reasoning"] = f"{decision['reasoning']} | catalog fallback: {model}"
    resolved_decision["candidates"] = available_candidates
    if "candidate_scores" in decision:
        resolved_decision["candidate_scores"] = [
            score
            for score in decision["candidate_scores"]
            if score["model"] in available_candidates
        ]
    # `fallbacks` is the SDK's runtime retry chain: every remaining candidate in
    # ranked order, which chat() walks on a transient upstream failure.
    resolved_decision["fallbacks"] = available_candidates[1:]
    return resolved_decision
