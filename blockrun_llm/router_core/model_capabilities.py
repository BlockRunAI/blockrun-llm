"""
Model capabilities used for hard routing constraints.

Python port of ``@blockrun/router-core`` ``model-capabilities.ts``.

Hosts may inject fresher values through ``RouterOptions["model_capabilities"]``.
Keeping a small built-in snapshot makes the core safe and useful when a
product catalog is temporarily unavailable, without importing product code.

GENERATED upstream by ``scripts/sync-model-capabilities.mjs`` from the public
catalog (GET https://blockrun.ai/api/v1/models) on 2026-08-31; ``supports_tools``
comes from a live function-calling probe. Re-sync from ``model-capabilities.ts``
rather than editing by hand — a hand edit is lost on the next sync.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .types import ModelCapabilities

DEFAULT_MODEL_CAPABILITIES: Mapping[str, ModelCapabilities] = MappingProxyType(
    {
        "anthropic/claude-fable-5": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        # override: The public catalog's `categories` omit "vision" for this Anthropic model even
        # though the gateway accepts image input for it (the prior hand-maintained snapshot had
        # it, and Anthropic's model card lists it). Without this the vision filter would silently
        # drop it — reported against the catalog; remove once the categories carry vision.
        "anthropic/claude-haiku-4.5": {
            "context_window": 200_000,
            "max_output_tokens": 64_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-opus-4.5": {
            "context_window": 200_000,
            "max_output_tokens": 64_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-opus-4.7": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-opus-4.8": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-opus-5": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-sonnet-4.5": {
            "context_window": 200_000,
            "max_output_tokens": 64_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        # override: The public catalog's `categories` omit "vision" for this Anthropic model even
        # though the gateway accepts image input for it (the prior hand-maintained snapshot had
        # it, and Anthropic's model card lists it). Without this the vision filter would silently
        # drop it — reported against the catalog; remove once the categories carry vision.
        "anthropic/claude-sonnet-4.6": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-sonnet-5": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        # supportsTools: not probed — fails closed
        "cohere/north-mini-code": {
            "context_window": 256_000,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "deepseek/deepseek-chat": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "deepseek/deepseek-reasoner": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "deepseek/deepseek-v4-pro": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "google/gemini-2.5-flash": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "google/gemini-2.5-flash-lite": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "google/gemini-2.5-pro": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "google/gemini-3-flash-preview": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "google/gemini-3.1-flash-lite": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "google/gemini-3.1-pro": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "google/gemini-3.5-flash": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "google/gemini-3.5-flash-lite": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "google/gemini-3.6-flash": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "minimax/minimax-m2.7": {
            "context_window": 204_800,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "minimax/minimax-m3": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "moonshot/kimi-k3": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        # supportsTools: not probed — fails closed
        "nvidia/llama-3.2-11b-vision": {
            "context_window": 128_000,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": True,
        },
        # supportsTools: not probed — fails closed
        "nvidia/nemotron-3-nano-30b": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": {
            "context_window": 256_000,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": True,
        },
        # supportsTools: not probed — fails closed
        "nvidia/nemotron-3-ultra-550b": {
            "context_window": 1_000_000,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        # supportsTools: not probed — fails closed
        "nvidia/nemotron-3.5-lightning": {
            "context_window": 1_000_000,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "openai/chat-latest": {
            "context_window": 128_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-4.1": {
            "context_window": 128_000,
            "max_output_tokens": 32_768,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-4.1-mini": {
            "context_window": 128_000,
            "max_output_tokens": 32_768,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-4.1-nano": {
            "context_window": 128_000,
            "max_output_tokens": 32_768,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-4o": {
            "context_window": 128_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-4o-mini": {
            "context_window": 128_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-5-mini": {
            "context_window": 200_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-5.2": {
            "context_window": 400_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        # supportsTools: not probed — fails closed
        "openai/gpt-5.2-pro": {
            "context_window": 400_000,
            "max_output_tokens": 128_000,
            "supports_tools": False,
            "supports_vision": True,
        },
        # supportsTools: gateway unavailable at probe time — fails closed; override: 2026-08-29
        # probe: every request (6 plain + 3 tool attempts) returned a gateway 500, so the probe
        # measured an incident, not the model. Codex's function calling is established by the
        # 2026-07 Terminal-Bench / tau2 calibration trajectories in portfolio.ts. Hosts observing
        # the 500s should drop it with RouterOptions.unavailableModels rather than this snapshot
        # claiming the model cannot call tools.
        "openai/gpt-5.3-codex": {
            "context_window": 400_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-5.4": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-5.4-mini": {
            "context_window": 400_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-5.4-nano": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        # supportsTools: not probed — fails closed
        "openai/gpt-5.4-pro": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": False,
            "supports_vision": True,
        },
        "openai/gpt-5.5": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        # supportsTools: not probed — fails closed
        "openai/gpt-5.5-pro": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": False,
            "supports_vision": True,
        },
        "openai/gpt-5.6-luna": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-5.6-luna-pro": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": False,
            "supports_vision": True,
        },
        "openai/gpt-5.6-sol": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-5.6-sol-pro": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-5.6-terra": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-5.6-terra-pro": {
            "context_window": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/o1": {
            "context_window": 200_000,
            "max_output_tokens": 100_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/o3": {
            "context_window": 200_000,
            "max_output_tokens": 100_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/o3-mini": {
            "context_window": 128_000,
            "max_output_tokens": 100_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/o4-mini": {
            "context_window": 128_000,
            "max_output_tokens": 100_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        # supportsTools: not probed — fails closed
        "poolside/laguna-xs-2.1": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "qwen/qwen3.7-flash": {
            "context_window": 1_000_000,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "qwen/qwen3.7-max": {
            "context_window": 1_000_000,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "qwen/qwen3.7-plus": {
            "context_window": 1_000_000,
            "max_output_tokens": 131_072,
            "supports_tools": True,
            "supports_vision": False,
        },
        "tencent/hy3": {
            "context_window": 262_144,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xai/grok-4.3": {
            "context_window": 1_000_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
        },
        "xai/grok-4.5": {
            "context_window": 500_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
        },
        "xai/grok-build-0.1": {
            "context_window": 256_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xiaomi/mimo-v2.5-pro": {
            "context_window": 1_048_576,
            "max_output_tokens": 131_072,
            "supports_tools": True,
            "supports_vision": False,
        },
        "zai/glm-5": {
            "context_window": 200_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "zai/glm-5-turbo": {
            "context_window": 200_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "zai/glm-5.1": {
            "context_window": 200_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "zai/glm-5.2": {
            "context_window": 1_000_000,
            "max_output_tokens": 131_072,
            "supports_tools": True,
            "supports_vision": False,
        },
        "zai/glm-5.3": {
            "context_window": 1_000_000,
            "max_output_tokens": 131_072,
            "supports_tools": True,
            "supports_vision": False,
        },
        "zai/glm-5.3-flash": {
            "context_window": 1_000_000,
            "max_output_tokens": 131_072,
            "supports_tools": True,
            "supports_vision": True,
        },
    }
)
