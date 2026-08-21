"""
Model capabilities used for hard routing constraints.

Python port of ``@blockrun/router-core`` ``model-capabilities.ts``.

Hosts may inject fresher values through ``RouterOptions["model_capabilities"]``.
Keeping a small built-in snapshot makes the core safe and useful when a
product catalog is temporarily unavailable, without importing product code.
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
        "anthropic/claude-haiku-4.5": {
            "context_window": 200_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-opus-4.6": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
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
        "anthropic/claude-sonnet-4.6": {
            "context_window": 200_000,
            "max_output_tokens": 64_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "anthropic/claude-sonnet-5": {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "deepseek/deepseek-chat": {
            "context_window": 1_000_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": False,
        },
        "deepseek/deepseek-reasoner": {
            "context_window": 1_000_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": False,
        },
        "deepseek/deepseek-v4-pro": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "free/deepseek-v4-flash": {
            "context_window": 1_000_000,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "free/seed-oss-36b": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "google/gemini-2.5-flash": {
            "context_window": 1_000_000,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "google/gemini-2.5-flash-lite": {
            "context_window": 1_000_000,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "google/gemini-2.5-pro": {
            "context_window": 1_050_000,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "google/gemini-3-flash-preview": {
            "context_window": 1_000_000,
            "max_output_tokens": 65_536,
            "supports_tools": False,
            "supports_vision": True,
        },
        "google/gemini-3.1-flash-lite": {
            "context_window": 1_000_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": False,
        },
        "google/gemini-3.1-pro": {
            "context_window": 1_050_000,
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
        "moonshot/kimi-k2.5": {
            "context_window": 262_144,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
        },
        "moonshot/kimi-k2.6": {
            "context_window": 262_144,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "moonshot/kimi-k2.7": {
            "context_window": 262_144,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "moonshot/kimi-k3": {
            "context_window": 1_048_576,
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": True,
        },
        "nvidia/nemotron-nano-9b-v2": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "nvidia/step-3.7-flash": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": False,
            "supports_vision": False,
        },
        "openai/gpt-4.1": {
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
            "max_output_tokens": 65_536,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-5.3-codex": {
            "context_window": 400_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-5.4": {
            "context_window": 400_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": True,
        },
        "openai/gpt-5.4-nano": {
            "context_window": 1_050_000,
            "max_output_tokens": 32_768,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/gpt-5.5": {
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
        "openai/o3": {
            "context_window": 200_000,
            "max_output_tokens": 100_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "openai/o4-mini": {
            "context_window": 128_000,
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
        "xai/grok-3-mini": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xai/grok-4-0709": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xai/grok-4-1-fast-non-reasoning": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xai/grok-4-1-fast-reasoning": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xai/grok-4-fast-non-reasoning": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xai/grok-4-fast-reasoning": {
            "context_window": 131_072,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": False,
        },
        "xai/grok-4.5": {
            "context_window": 500_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
        },
        "zai/glm-5.1": {
            "context_window": 200_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_vision": False,
        },
        "zai/glm-5.2": {
            "context_window": 1_000_000,
            "max_output_tokens": 262_144,
            "supports_tools": True,
            "supports_vision": False,
        },
    }
)
