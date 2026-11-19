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
import re
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
from .router_core.selector import usable_pricing
from .router_core.types import (
    Capacity,
    ModelPricing,
    RouterOptions,
    RoutingConfig,
    RoutingDecision,
    TierConfig,
)


class CatalogPricing(ModelPricing, total=False):
    """A Router Core price entry plus the gateway's per-token platform fee.

    ``platform_fee_percent`` is what an x402 call to a per-token paid chat row
    pays on top of its list ``input`` / ``output`` rates (spec PF-8). Router
    Core ignores it; :func:`route_with_catalog` reads it for ``cost_estimate``.
    Absent means 0.
    """

    platform_fee_percent: float


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
#: $0 in ``/v1/models`` but both answered as ``nemotron-3-nano-30b``.
#:
#: 2026-09-29: ``nemotron-3-nano-30b`` itself is gone. NVIDIA deprovisioned it for
#: blockrun's account on 2026-09-08, it left ``/v1/models``, and the gateway now
#: redirects it to nano-omni. Every slot it held goes to nano-omni: that is the
#: model those slots were actually being served by, it is the same family and size
#: (30B-A3B), and with nano-30b gone the reason it was excluded above no longer
#: applies (a model-echo probe that day got nano-omni's own NIM deployment back).
#: The router adapter was already dropping nano-30b at runtime, because it is not
#: in the catalog, so SIMPLE had been opening on its first fallback.
#:
#: 2026-11-08: two rungs had gone the same way. ``nvidia/llama-3.2-11b-vision``
#: left ``/v1/models`` and ``cohere/north-mini-code`` lists ``available: false``;
#: both still answer 200, served by a different model (nano-omni or
#: ``nemotron-3-super-120b``). So the "free vision" rung was not Llama at all.
#: Replacements were checked on both gateways (Base and Solana), each id asked
#: twice with the reply's ``model`` field read back:
#: ``nvidia/nemotron-3.5-bolt`` is the vision rung — it named a solid
#: red/blue/green 64x64 PNG correctly 9/9 on each chain, serving as itself —
#: and ``nvidia/nemotron-3-ultra-550b`` (1M ctx, now answering as itself; it
#: was a substitute on 2026-08-31) takes north-mini-code's long-context and
#: coding slots. ``poolside/laguna-xs-2.1`` serves as itself on Base but is
#: substituted on Solana, so it is the last rung wherever it appears.
#: ``gpt-oss-120b/20b`` stay out — proxy-only ids with no catalog price, under
#: the NVIDIA free tier's prompt-retention policy.
FREE_TIERS: dict[str, TierConfig] = {
    "SIMPLE": {
        # Was nemotron-3-nano-30b, the fastest free model, until its 2026-09-08
        # delisting. Latency is the only axis that separates free rungs (they
        # all cost $0); nano-omni answered in 1.4s on blockrun's direct probe.
        "primary": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",  # 256K ctx
        "fallback": [
            "nvidia/nemotron-3.5-lightning",
            "nvidia/nemotron-3.5-bolt",  # free vision, verified on both chains
            "nvidia/nemotron-3-ultra-550b",
            "poolside/laguna-xs-2.1",
        ],
    },
    "MEDIUM": {
        "primary": "nvidia/nemotron-3.5-lightning",  # 1M ctx — free tier flagship
        "fallback": [
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "nvidia/nemotron-3-ultra-550b",
            "nvidia/nemotron-3.5-bolt",
            "poolside/laguna-xs-2.1",
        ],
    },
    "COMPLEX": {
        # The two 1M-context free models absorb long inputs; the vision rung
        # behind them absorbs multi-modal ones.
        "primary": "nvidia/nemotron-3.5-lightning",
        "fallback": [
            "nvidia/nemotron-3-ultra-550b",  # 1M ctx
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "nvidia/nemotron-3.5-bolt",  # free vision
        ],
    },
    "REASONING": {
        "primary": "nvidia/nemotron-3.5-lightning",
        "fallback": [
            "nvidia/nemotron-3-ultra-550b",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "poolside/laguna-xs-2.1",
        ],
    },
}


class ModelPricingMap(dict):  # type: ignore[type-arg]
    """The router's pricing map, plus the catalog's search-capable models.

    ``search_models`` lists the available ``xai/*`` chat models the catalog
    tags ``search`` (every available ``xai/*`` chat model when none is
    tagged). The gateway serves search on ``xai/*`` ids only.
    """

    search_models: list[str]


def requests_search(
    search: bool | None = None,
    search_parameters: Mapping[str, Any] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Whether a call asks for xAI search.

    An ``x_search`` / ``web_search`` tool, or ``search_parameters`` whose
    ``mode`` is not ``"off"``, or the ``search=True`` shortcut when no
    ``search_parameters`` are given (the precedence the request body follows).
    The gateway serves search on ``xai/*`` models only and answers any other
    model with a non-transient 400 ``SEARCH_UNSUPPORTED``.
    """
    if any(tool.get("type") in ("x_search", "web_search") for tool in tools or []):
        return True
    if search_parameters is not None:
        return search_parameters.get("mode") != "off"
    return search is True


def require_search_model(
    models: Sequence[str],
    search: bool | None = None,
    search_parameters: Mapping[str, Any] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Refuse, before anything is sent or paid, a search call on a non-xAI model."""
    if not requests_search(search, search_parameters, tools):
        return
    wrong = [model for model in models if not model.startswith("xai/")]
    if wrong:
        raise ValueError(
            "Search (search=True, search_parameters, or an x_search / web_search tool) runs on "
            f"xai/* models only; the gateway refuses {', '.join(wrong)} with a 400. "
            "Use an xai/* model (e.g. xai/grok-4.7), smart_chat_completion(..., search=True), "
            "or drop the search options."
        )


def _search_fallback_chain(
    model_pricing: Mapping[str, ModelPricing], routing_profile: str | None
) -> list[str]:
    """Search models when the route has none: cheapest first (premium: priciest first)."""
    pool = getattr(model_pricing, "search_models", None)
    if pool is None:
        pool = [model_id for model_id in model_pricing if model_id.startswith("xai/")]
    pool = [model_id for model_id in pool if model_id in model_pricing]

    def price(model_id: str) -> float:
        entry = model_pricing[model_id]
        return entry["input_price"] + entry["output_price"]

    # Ties go to the newer id (the larger one in version order).
    by_id = sorted(pool, key=_version_key, reverse=True)
    if routing_profile == "premium":
        return sorted(by_id, key=price, reverse=True)
    return sorted(by_id, key=price)


def _version_key(model_id: str) -> list[Any]:
    return [
        (0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", model_id)
    ]


def _fee_percent(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def platform_fee_percent(model: Mapping[str, Any]) -> float:
    """The platform fee percent a ``/v1/models`` row's per-token price pays on top.

    The gateways charge per-token paid chat its list rates multiplied by
    ``1 + platform_fee_percent / 100`` (spec PF-8). The field is read from
    ``pricing.platform_fee_percent`` (both gateways), then from the row's top
    level (blockrun.ai also publishes it there). It is 0 when neither is a
    finite, non-negative number — older gateways and the enterprise gateway do
    not send it — and 0 for any row that is not a per-token paid chat row:
    free rows, flat rows, media and Decisions never pay it.
    """
    billing_mode = model.get("billing_mode")
    if billing_mode is not None and billing_mode != "paid":
        return 0.0
    block = model.get("pricing")
    if isinstance(block, Mapping):
        if block.get("flat"):
            return 0.0
        nested = _fee_percent(block.get("platform_fee_percent"))
        if nested is not None:
            return nested
    if model.get("flatPrice") or model.get("flat_price"):
        return 0.0
    top_level = _fee_percent(model.get("platform_fee_percent"))
    return top_level if top_level is not None else 0.0


def token_cost_usd(
    pricing: ModelPricing | None, estimated_input_tokens: int, max_output_tokens: int
) -> float | None:
    """Estimated USD for a per-token entry, platform fee included.

    ``None`` for a flat entry, a missing one, or one with an unusable price —
    those keep Router Core's own estimate.
    """
    entry = usable_pricing(pricing)
    if entry is None or entry.get("flat_price"):
        return None
    list_cost = (
        estimated_input_tokens * entry.get("input_price", 0)
        + max_output_tokens * entry.get("output_price", 0)
    ) / 1_000_000
    fee = _fee_percent(dict(entry).get("platform_fee_percent")) or 0.0
    return list_cost * (1 + fee / 100)


def build_model_pricing(models: list[dict[str, Any]]) -> dict[str, CatalogPricing]:
    """Build the router's pricing map from a ``/v1/models`` payload.

    Shared by every client (Base and Solana, sync and async) so the four copies
    cannot drift. Rows the catalog marks unavailable are skipped: a model that
    cannot serve a request must not win routing, since every call to it would
    fail with a non-transient error.

    The catalog uses the nested ``pricing.input`` / ``pricing.output`` shape;
    older snapshots used top-level ``inputPrice`` / ``outputPrice``. Both are
    accepted so the SDK keeps working through backend transitions. Each entry
    also carries the row's ``platform_fee_percent`` (see
    :func:`platform_fee_percent`), so ``cost_estimate`` matches what the
    gateway quotes for per-token paid chat.
    """
    pricing = ModelPricingMap()
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
            "platform_fee_percent": platform_fee_percent(model),
        }
    xai_chat = [
        model
        for model in models
        if str(model.get("id", "")).startswith("xai/")
        and model["id"] in pricing
        and "chat" in (model.get("categories") or [])
    ]
    tagged = [model for model in xai_chat if "search" in (model.get("categories") or [])]
    pricing.search_models = [model["id"] for model in (tagged or xai_chat)]
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
    search: bool = False,
    config: RoutingConfig | None = None,
    now: Any = None,
) -> ResolvedRoutingDecision:
    """Route a request and resolve the ranking against the live catalog.

    ``routing_profile`` accepts Router Core's ``"eco" | "auto" | "premium"``
    plus the SDK-only ``"free"``. ``search`` (see :func:`requests_search`)
    keeps only ``xai/*`` candidates, the only models the gateway serves search
    on.
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

    # Search runs on xai/* only: the gateway refuses any other model with a
    # non-transient 400, so a non-xai pick (or fallback) would fail outright.
    # Keep the router's order among the xai/* candidates. When the route has
    # none (the common case: the router seldom ranks xAI), fall back to the
    # catalog's search-capable xAI models, cheapest first (premium: most
    # capable first). Mirrors the TypeScript SDK's routeWithCatalog.
    search_fallback = False
    if search:
        if is_free_profile:
            raise ValueError(
                "The free profile has no search-capable model: search runs on paid xai/* "
                "models only. Use routing_profile='eco' (or another) or drop the search options."
            )
        searchable = [model_id for model_id in callable_models if model_id.startswith("xai/")]
        if not searchable:
            searchable = _search_fallback_chain(model_pricing, routing_profile)
            search_fallback = True
        if not searchable:
            raise ValueError(
                "Search runs on xai/* models only, and the catalog lists no available xai/* "
                "chat model. Call an xai/* model directly or drop the search options."
            )
        callable_models = searchable

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
    # A per-token row's estimate is its list cost times (1 + platform_fee_percent
    # / 100), the fee the catalog publishes per row (0 when absent). Router
    # Core's calculate_model_cost instead multiplies by a hardcoded server
    # margin, which is not the per-row fee (enterprise and older gateways charge
    # none). Flat and unpriced entries keep Router Core's estimate.
    token_cost = token_cost_usd(model_pricing.get(model), estimated_input_tokens, max_output_tokens)
    if token_cost is not None:
        costs["cost_estimate"] = token_cost
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
    if search_fallback:
        which = "most capable" if routing_profile == "premium" else "cheapest"
        resolved_decision["reasoning"] = (
            f"{decision['reasoning']} | search runs on xai/* only; the route had none, "
            f"so the {which} search-capable xAI model: {model}"
        )
    elif model != decision["model"]:
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
