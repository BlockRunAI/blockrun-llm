"""
Router Core types — Python port of ``@blockrun/router-core`` ``types.ts``.

Four classification tiers — REASONING is distinct from COMPLEX because
reasoning tasks need different models (o3, gemini-pro) than general complex
tasks (gpt-4o, sonnet-4).

Scoring uses weighted float dimensions with sigmoid confidence calibration.

Field names are snake_case (the upstream TypeScript uses camelCase); the
mapping is 1:1 and mechanical, e.g. ``costEstimate`` -> ``cost_estimate``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

Tier = Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]

TaskType = Literal[
    "chat",
    "extraction",
    "code_edit",
    "code_agent",
    "tool_agent",
    "tool_agent_parallel",
    "debug",
    "reasoning",
    "reasoning_mcq",
    "reasoning_math",
    "long_context",
    "vision",
]

Profile = Literal["auto", "eco", "premium", "agentic"]

RoutingProfile = Literal["eco", "auto", "premium"]

Method = Literal["rules", "llm", "portfolio"]

#: Ordering used by the structured-output minimum-tier override.
TIER_RANK: dict[str, int] = {"SIMPLE": 0, "MEDIUM": 1, "COMPLEX": 2, "REASONING": 3}

TIERS: tuple[str, ...] = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")


class ModelPricing(TypedDict, total=False):
    """Catalog prices per 1M tokens.

    ``flat_price`` overrides token pricing when present and non-zero (the
    BlockRun catalog reports ``0`` rather than omitting the field, so falsy
    means "per-token billing" here, matching the TypeScript ``undefined``).
    """

    input_price: float
    output_price: float
    flat_price: float


class ModelCapabilities(TypedDict):
    context_window: int
    max_output_tokens: int
    supports_tools: bool
    supports_vision: bool


class Capacity(TypedDict):
    """Narrow capability view used by :func:`filter_candidates_by_capacity`."""

    context_window: int
    max_output: int


class ModelPerformanceProfile(TypedDict, total=False):
    measured_at: str
    #: Gateway end-to-end latency for the benchmark workload.
    latency_ms: float
    #: Tail latency is more relevant than mean latency for urgent requests.
    p95_latency_ms: float
    output_tokens_per_second: float
    #: External intelligence index when one was available; not task success.
    intelligence_index: float
    #: Failure fraction observed in the same benchmark run.
    error_rate: float
    #: Number of sampled calls behind the observation.
    samples: int


class TierConfig(TypedDict):
    primary: str
    fallback: list[str]


class DimensionScore(TypedDict):
    name: str
    score: float
    signal: str | None


class ScoringResult(TypedDict, total=False):
    #: weighted float (roughly [-0.3, 0.4])
    score: float
    #: ``None`` = ambiguous, needs fallback classifier
    tier: Tier | None
    #: sigmoid-calibrated [0, 1]
    confidence: float
    signals: list[str]
    #: 0-1 agentic task score for auto-switching to agentic tiers
    agentic_score: float
    #: per-dimension breakdown for /debug
    dimensions: list[DimensionScore]


class CandidateScore(TypedDict):
    model: str
    score: float
    quality: float
    cost: float
    speed: float
    reliability: float


class _RoutingDecisionRequired(TypedDict):
    model: str
    tier: Tier
    confidence: float
    method: Method
    reasoning: str
    cost_estimate: float
    baseline_cost: float
    savings: float  # 0-1 percentage


class RoutingDecision(_RoutingDecisionRequired, total=False):
    #: 0-1 agentic task score (present when tier routing used)
    agentic_score: float
    #: Which tier configs were used (auto/eco/premium/agentic)
    tier_configs: dict[str, TierConfig]
    #: Which routing profile was applied
    profile: Profile
    #: Ordered, capability-eligible candidates. The first entry is ``model``.
    candidates: list[str]
    #: Explainable request classification used by the portfolio router.
    task_type: TaskType
    #: Router implementation that made the selection.
    router_version: Literal["v2-rules", "v3-portfolio"]
    #: Explainable local portfolio score breakdown, ordered with ``candidates``.
    candidate_scores: list[CandidateScore]


class TokenCountThresholds(TypedDict):
    simple: int
    complex: int


class TierBoundaries(TypedDict):
    simple_medium: float
    medium_complex: float
    complex_reasoning: float


class ScoringConfig(TypedDict):
    token_count_thresholds: TokenCountThresholds
    code_keywords: list[str]
    reasoning_keywords: list[str]
    simple_keywords: list[str]
    technical_keywords: list[str]
    creative_keywords: list[str]
    imperative_verbs: list[str]
    constraint_indicators: list[str]
    output_format_keywords: list[str]
    reference_keywords: list[str]
    negation_keywords: list[str]
    domain_specific_keywords: list[str]
    agentic_task_keywords: list[str]
    dimension_weights: dict[str, float]
    tier_boundaries: TierBoundaries
    confidence_steepness: float
    confidence_threshold: float


class ClassifierConfig(TypedDict):
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    prompt_truncation_chars: int
    cache_ttl_ms: int


class OverridesConfig(TypedDict, total=False):
    max_tokens_force_complex: int
    structured_output_min_tier: Tier
    ambiguous_default_tier: Tier
    #: ``True`` forces agentic tiers, ``False`` disables them, absent = auto-detect.
    agentic_mode: bool | None


class PortfolioBandWeights(TypedDict):
    quality: float
    capability: float
    cost: float
    speed: float
    reliability: float
    legacy: float


class HighStakesBoost(TypedDict):
    quality: float
    reliability: float


class AffinityFloorGap(TypedDict):
    auto: float
    eco: float
    premium: float


class PortfolioConfig(TypedDict):
    auto: PortfolioBandWeights
    eco: PortfolioBandWeights
    premium: PortfolioBandWeights
    high_stakes_boost: HighStakesBoost
    latency_sensitive_speed_boost: float
    #: A candidate materially below the best task affinity cannot win on cost alone.
    affinity_floor_gap: AffinityFloorGap


class PromotionTierOverride(TypedDict, total=False):
    primary: str
    fallback: list[str]


class Promotion(TypedDict, total=False):
    """Time-windowed promotion that temporarily overrides tier routing.

    Active promotions are auto-applied; expired ones are ignored at runtime.
    """

    #: Human-readable label (e.g. "GLM-5 Launch Promo")
    name: str
    #: ISO date string, promotion starts (inclusive). e.g. "2026-04-01"
    start_date: str
    #: ISO date string, promotion ends (exclusive). e.g. "2026-04-15"
    end_date: str
    #: Partial tier overrides merged into the active tier configs.
    tier_overrides: dict[str, PromotionTierOverride]
    #: Which profiles this applies to. Default: all profiles.
    profiles: list[Profile]


class ShadowConfig(TypedDict, total=False):
    strategy: Literal["rules", "portfolio"]
    sample_rate: float


class _RoutingConfigRequired(TypedDict):
    version: str
    classifier: ClassifierConfig
    scoring: ScoringConfig
    tiers: dict[str, TierConfig]
    overrides: OverridesConfig


class RoutingConfig(_RoutingConfigRequired, total=False):
    #: Enables a one-line rollback to the established V2 rules selector.
    strategy: Literal["rules", "portfolio"]
    #: Locally recompute a comparison strategy without changing the served model.
    shadow: ShadowConfig
    #: Calibratable local portfolio scoring weights; relative, not probabilities.
    portfolio: PortfolioConfig
    #: Tier configs for agentic mode. ``None`` disables agentic tier selection.
    agentic_tiers: dict[str, TierConfig] | None
    #: Tier configs for eco profile. ``None`` falls back to ``tiers``.
    eco_tiers: dict[str, TierConfig] | None
    #: Tier configs for premium profile. ``None`` falls back to ``tiers``.
    premium_tiers: dict[str, TierConfig] | None
    #: Time-windowed promotions that temporarily override tier routing.
    promotions: list[Promotion]


class _RouterOptionsRequired(TypedDict):
    config: RoutingConfig
    model_pricing: Mapping[str, ModelPricing]


class RouterOptions(_RouterOptionsRequired, total=False):
    """Per-request routing inputs."""

    #: Host-provided capability snapshot; overrides the core's built-in one.
    model_capabilities: Mapping[str, ModelCapabilities]
    routing_profile: RoutingProfile | None
    has_tools: bool
    #: Number of tool definitions visible to the model on this turn.
    tool_count: int
    #: Local tool identifiers, used only for request/tool intent matching.
    tool_names: Sequence[str]
    #: Tools are attached by the host and this turn needs to use them.
    requires_tools: bool | None
    has_vision: bool
    #: ``response_format`` / JSON schema requires reliable structured output.
    requires_structured_output: bool
    #: Model ids the host has observed to be unavailable at the gateway (a
    #: 400/404/410 on a direct call, a provider EOL). Hard-removed from every
    #: chain before selection and never restored by an eligibility fail-open —
    #: the operational kill-switch for a dead chain rung, usable the moment the
    #: host observes the failure instead of waiting on a core release and two
    #: consumer repins. Distinct from user-preference exclusion
    #: (``filter_by_exclude_list``), which deliberately fail-opens rather than
    #: empty a chain.
    unavailable_models: Sequence[str]
    #: Override current time for promotion window checks (for testing). Naive
    #: values are read as UTC. ``datetime.datetime``.
    now: object
    #: Fresh gateway performance observations, injected off the hot path.
    model_performance: Mapping[str, ModelPerformanceProfile]
