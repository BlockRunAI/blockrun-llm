"""
Router Strategy Registry

Python port of ``@blockrun/router-core`` ``strategy.ts``.

Pluggable strategy system for request routing.
Default: RulesStrategy — identical to the original inline route() logic, <1ms.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from ._js import as_utc, js_regex, parse_date, to_fixed
from .rules import classify_by_rules
from .selector import select_model
from .types import (
    TIER_RANK,
    Profile,
    Promotion,
    RouterOptions,
    RoutingDecision,
    Tier,
    TierConfig,
)

_STRUCTURED_OUTPUT = js_regex(r"json|structured|schema", ignorecase=True)


class RouterStrategy(Protocol):
    """Interface implemented by every routing strategy."""

    name: str

    def route(
        self,
        prompt: str,
        system_prompt: str | None,
        max_output_tokens: int,
        options: RouterOptions,
    ) -> RoutingDecision: ...


def sample_prompt(value: str, scan_limit: int) -> str:
    """Sample both ends of a long prompt, keeping instructions at either edge."""
    if len(value) <= scan_limit:
        return value
    prefix_length = math.ceil(scan_limit / 2)
    suffix_length = scan_limit - prefix_length
    suffix = value[-suffix_length:] if suffix_length else value
    return f"{value[:prefix_length]}\n{suffix}"


def scan_limit_for(options: RouterOptions) -> int:
    return max(1, min(8_000, options["config"]["classifier"]["prompt_truncation_chars"]))


def apply_unavailable_models(
    tier_configs: dict[str, TierConfig],
    unavailable_models: Sequence[str] | None,
) -> dict[str, TierConfig]:
    """Remove host-declared-dead models from every tier chain.

    Promotes the first surviving rung to primary. A tier whose chain is
    entirely dead keeps its original config — the router has nothing live to
    offer there, and inventing a model would hide the outage from the host
    that reported it.
    """
    if not unavailable_models:
        return tier_configs
    dead = set(unavailable_models)
    result = tier_configs
    for tier, config in tier_configs.items():
        alive = [model for model in [config["primary"], *config["fallback"]] if model not in dead]
        if not alive or (
            alive[0] == config["primary"] and len(alive) == len(config["fallback"]) + 1
        ):
            continue
        if result is tier_configs:
            result = dict(tier_configs)
        result[tier] = {"primary": alive[0], "fallback": alive[1:]}
    return result


def apply_promotions(
    tier_configs: dict[str, TierConfig],
    promotions: list[Promotion] | None,
    profile: Profile,
    now: datetime | None = None,
) -> dict[str, TierConfig]:
    """Apply active time-windowed promotions to tier configs.

    Returns a new tier-config mapping with promotion overrides merged in.
    Expired or not-yet-active promotions are ignored.
    """
    if not promotions:
        return tier_configs

    current = now if now is not None else as_utc(None)
    result = tier_configs
    for promo in promotions:
        start = parse_date(promo.get("start_date", ""))
        end = parse_date(promo.get("end_date", ""))
        if start is None or end is None:
            continue
        if current < start or current >= end:
            continue

        profiles = promo.get("profiles")
        if profiles and profile not in profiles:
            continue

        # Shallow-clone on first mutation
        if result is tier_configs:
            result = {tier: copy.copy(config) for tier, config in tier_configs.items()}

        for tier, override in promo.get("tier_overrides", {}).items():
            if tier not in result:
                continue
            primary = override.get("primary")
            fallback = override.get("fallback")
            if primary:
                result[tier]["primary"] = primary
            if fallback:
                result[tier]["fallback"] = fallback

    return result


class RulesStrategy:
    """Rules-based routing strategy.

    Attaches ``tier_configs`` and ``profile`` to the decision for downstream use.
    """

    name = "rules"

    def route(
        self,
        prompt: str,
        system_prompt: str | None,
        max_output_tokens: int,
        options: RouterOptions,
    ) -> RoutingDecision:
        config = options["config"]
        model_pricing = options["model_pricing"]

        # Estimate input tokens (~4 chars per token)
        full_text = f"{system_prompt or ''} {prompt}"
        estimated_tokens = math.ceil(len(full_text) / 4)
        scan_limit = scan_limit_for(options)
        scanned_prompt = sample_prompt(prompt, scan_limit)
        scanned_system_prompt = sample_prompt(system_prompt, scan_limit) if system_prompt else None

        # --- Rule-based classification (runs first to get agentic_score) ---
        rule_result = classify_by_rules(
            scanned_prompt, scanned_system_prompt, estimated_tokens, config["scoring"]
        )

        # --- Select tier configs based on routing profile ---
        routing_profile = options.get("routing_profile")
        profile: Profile
        if routing_profile == "eco":
            # `eco_tiers: None` explicitly disables the special eco tier set
            # while keeping eco routing semantics. Fall back to regular tiers
            # instead of dropping into auto routing (which could select agentic
            # tiers).
            eco_tiers = config.get("eco_tiers")
            tier_configs = eco_tiers if eco_tiers else config["tiers"]
            profile_suffix = " | eco" if eco_tiers else " | eco (default tiers)"
            profile = "eco"
        elif routing_profile == "premium":
            # `premium_tiers: None` disables the premium-specific tier set but
            # the request is still a premium-profile request, so use regular
            # tiers while preserving premium metadata/cost semantics.
            premium_tiers = config.get("premium_tiers")
            tier_configs = premium_tiers if premium_tiers else config["tiers"]
            profile_suffix = " | premium" if premium_tiers else " | premium (default tiers)"
            profile = "premium"
        else:
            # Auto profile (or unset): intelligent routing with agentic detection.
            #
            # `agentic_mode` semantics:
            #   - True  -> force agentic tiers (ignore heuristics)
            #   - False -> disable agentic tiers entirely (even if tools present)
            #   - unset -> auto-detect via heuristics (tools present OR high
            #     agentic score)
            agentic_score = rule_result.get("agentic_score", 0) or 0
            is_auto_agentic = agentic_score >= 0.5
            agentic_mode_setting = config["overrides"].get("agentic_mode")
            requires_tools = options.get("requires_tools")
            has_tools_in_request = (
                requires_tools if requires_tools is not None else options.get("has_tools", False)
            )
            agentic_tiers = config.get("agentic_tiers")
            if agentic_mode_setting is False:
                # Explicitly disabled — never use agentic tiers
                use_agentic_tiers = False
            elif agentic_mode_setting is True:
                # Explicitly enabled — use agentic tiers if available
                use_agentic_tiers = agentic_tiers is not None
            else:
                use_agentic_tiers = bool(
                    (has_tools_in_request or is_auto_agentic) and agentic_tiers is not None
                )
            if use_agentic_tiers and agentic_tiers is not None:
                tier_configs = agentic_tiers
                profile_suffix = f" | agentic{' (tools)' if has_tools_in_request else ''}"
                profile = "agentic"
            else:
                tier_configs = config["tiers"]
                profile_suffix = ""
                profile = "auto"

        # Apply time-windowed promotions
        now = as_utc(options.get("now"))
        tier_configs = apply_promotions(tier_configs, config.get("promotions"), profile, now)

        # Hard-remove models the host has observed dead at the gateway. After
        # promotions, so a promo cannot resurrect a rung the host just killed.
        tier_configs = apply_unavailable_models(tier_configs, options.get("unavailable_models"))

        agentic_score_value = rule_result.get("agentic_score")

        # --- Override: large context → force COMPLEX ---
        force_complex_at = config["overrides"]["max_tokens_force_complex"]
        if estimated_tokens > force_complex_at:
            decision = select_model(
                "COMPLEX",
                0.95,
                "rules",
                f"Input exceeds {force_complex_at} tokens{profile_suffix}",
                tier_configs,
                model_pricing,
                estimated_tokens,
                max_output_tokens,
                routing_profile,
                agentic_score_value,
            )
            decision["tier_configs"] = tier_configs
            decision["profile"] = profile
            return decision

        # Structured output detection
        has_structured_output = options.get("requires_structured_output") is True or (
            bool(_STRUCTURED_OUTPUT.search(scanned_system_prompt))
            if scanned_system_prompt
            else False
        )

        tier: Tier
        signals = ", ".join(rule_result.get("signals", []))
        reasoning = f"score={to_fixed(rule_result['score'], 2)} | {signals}"

        if rule_result.get("tier") is not None:
            tier = rule_result["tier"]  # type: ignore[assignment]
            confidence = rule_result["confidence"]
        else:
            # Ambiguous — default to configurable tier (no external API call)
            tier = config["overrides"]["ambiguous_default_tier"]
            confidence = 0.5
            reasoning += f" | ambiguous -> default: {tier}"

        # Apply structured output minimum tier
        if has_structured_output:
            min_tier = config["overrides"]["structured_output_min_tier"]
            if TIER_RANK[tier] < TIER_RANK[min_tier]:
                reasoning += f" | upgraded to {min_tier} (structured output)"
                tier = min_tier

        # Add routing profile suffix to reasoning
        reasoning += profile_suffix

        decision = select_model(
            tier,
            confidence,
            "rules",
            reasoning,
            tier_configs,
            model_pricing,
            estimated_tokens,
            max_output_tokens,
            routing_profile,
            agentic_score_value,
        )
        decision["tier_configs"] = tier_configs
        decision["profile"] = profile
        return decision


# --- Strategy Registry ---

_registry: dict[str, RouterStrategy] = {"rules": RulesStrategy()}


def get_strategy(name: str) -> RouterStrategy:
    strategy = _registry.get(name)
    if strategy is None:
        raise ValueError(f"Unknown routing strategy: {name}")
    return strategy


def register_strategy(strategy: RouterStrategy) -> None:
    _registry[strategy.name] = strategy
