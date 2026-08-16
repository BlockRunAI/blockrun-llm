"""
Model-performance priors consumed by the portfolio router.

Python port of ``@blockrun/router-core`` ``model-profiles.ts``.

The entries below are a small, auditable seed extracted from the 2026-03-16
BlockRun performance run. They are deliberately weak priors: live data injected
by the host should replace them through configuration before a release.
Historical numbers must never be presented as a current provider SLA or as
task-quality measurements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .types import ModelPerformanceProfile

_GENERATED_PATH = Path(__file__).with_name("model_profiles.generated.json")

#: camelCase (upstream JSON) -> snake_case (this port).
_FIELD_ALIASES = {
    "measuredAt": "measured_at",
    "latencyMs": "latency_ms",
    "p95LatencyMs": "p95_latency_ms",
    "outputTokensPerSecond": "output_tokens_per_second",
    "intelligenceIndex": "intelligence_index",
    "errorRate": "error_rate",
    "samples": "samples",
}


def _normalize(raw: Mapping[str, Any]) -> ModelPerformanceProfile:
    """Accept either the upstream camelCase JSON or already-ported keys."""
    profile: dict[str, Any] = {}
    for key, value in raw.items():
        profile[_FIELD_ALIASES.get(key, key)] = value
    return profile  # type: ignore[return-value]


def _load_generated() -> Mapping[str, ModelPerformanceProfile]:
    try:
        with _GENERATED_PATH.open(encoding="utf-8") as handle:
            payload: dict[str, dict[str, Any]] = json.load(handle)
    except (OSError, ValueError):
        # A missing or corrupt asset must not take routing down: these are
        # weak priors, and the router already handles an absent observation.
        return MappingProxyType({})
    return MappingProxyType({model: _normalize(raw) for model, raw in payload.items()})


#: Generated from benchmark files that satisfy the uncached-inference
#: invariant. These are weak performance priors (speed/reliability), never
#: task-quality labels.
LIVE_MODEL_PROFILES: Mapping[str, ModelPerformanceProfile] = _load_generated()

HISTORICAL_MODEL_PROFILES: Mapping[str, ModelPerformanceProfile] = MappingProxyType(
    {
        "anthropic/claude-haiku-4.5": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 2305,
            "output_tokens_per_second": 140.6,
        },
        "anthropic/claude-opus-4.6": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 2139,
            "output_tokens_per_second": 119.7,
        },
        "anthropic/claude-sonnet-4.6": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 2110,
            "output_tokens_per_second": 121.3,
        },
        "deepseek/deepseek-chat": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1431,
            "output_tokens_per_second": 179.2,
            "intelligence_index": 32,
        },
        "google/gemini-2.5-flash": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1238,
            "output_tokens_per_second": 207.6,
            "intelligence_index": 20,
        },
        "google/gemini-2.5-flash-lite": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1353,
            "output_tokens_per_second": 192.5,
            "intelligence_index": 20,
        },
        "google/gemini-2.5-pro": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1294,
            "output_tokens_per_second": 197.8,
        },
        "google/gemini-3.1-pro": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1609,
            "output_tokens_per_second": 167.2,
        },
        "moonshot/kimi-k2.5": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1646,
            "output_tokens_per_second": 155.7,
        },
        "openai/gpt-4o-mini": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 2764,
            "output_tokens_per_second": 92.8,
        },
        "openai/gpt-5.3-codex": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 7935,
            "output_tokens_per_second": 32.3,
        },
        "xai/grok-4-1-fast-non-reasoning": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1244,
            "output_tokens_per_second": 205.8,
            "intelligence_index": 41,
        },
        "xai/grok-4-1-fast-reasoning": {
            "measured_at": "2026-03-16T13:50:48Z",
            "latency_ms": 1454,
            "output_tokens_per_second": 176.2,
            "intelligence_index": 41,
        },
    }
)
