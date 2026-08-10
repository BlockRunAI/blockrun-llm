"""BlockRun Router Core V3 native Python adapter.

This module mirrors the product-neutral Router Core portfolio contract pinned
at ``d4308049348e11e17ed08a254676a34949be80f9``.  It deliberately contains no
network or payment code: callers provide the current gateway catalog and the
router returns one capability-eligible primary plus ordered fallbacks.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal, TypedDict

from .router import classify_by_rules

ROUTER_CORE_COMMIT = "d4308049348e11e17ed08a254676a34949be80f9"
ROUTER_VERSION = "v3-portfolio"

Tier = Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]
RoutingProfile = Literal["free", "eco", "auto", "premium"]
ROUTER_ALIASES: dict[str, RoutingProfile] = {
    "blockrun/auto": "auto",
    "blockrun/eco": "eco",
    "blockrun/premium": "premium",
}
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


class RoutingDecision(TypedDict):
    model: str
    tier: Tier
    confidence: float
    method: Literal["portfolio"]
    reasoning: str
    cost_estimate: float
    baseline_cost: float
    savings: float
    profile: str
    task_type: TaskType
    router_version: Literal["v3-portfolio"]
    candidates: list[str]
    candidate_scores: list[dict[str, float | str]]
    fallbacks: list[str]


def routing_profile_for_model(model: str) -> RoutingProfile | None:
    """Return the Router profile represented by a public model alias."""

    return ROUTER_ALIASES.get(model.strip().lower())


def message_routing_inputs(messages: list[dict[str, Any]]) -> tuple[str, str | None, bool]:
    """Extract bounded text and vision signals from OpenAI-style messages.

    The latest user turn is the routing prompt.  System/developer messages are
    included as instructions, while assistant/tool history is intentionally not
    reclassified as a new user task.  This keeps routing local and deterministic
    even for long agent transcripts.
    """

    user_parts: list[str] = []
    system_parts: list[str] = []
    has_vision = False

    def content_text(content: Any) -> str:
        nonlocal has_vision
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", ""))
            if kind in {"image_url", "input_image", "image"}:
                has_vision = True
            text = item.get("text") or item.get("input_text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)

    for message in messages:
        role = str(message.get("role", ""))
        text = content_text(message.get("content"))
        if role in {"system", "developer"} and text:
            system_parts.append(text)
        elif role == "user" and text:
            user_parts.append(text)

    prompt = user_parts[-1] if user_parts else ""
    return prompt, "\n".join(system_parts) or None, has_vision


TierConfig = dict[Tier, dict[str, Any]]

AUTO_TIERS: TierConfig = {
    "SIMPLE": {
        "primary": "google/gemini-2.5-flash",
        "fallback": [
            "google/gemini-3-flash-preview",
            "deepseek/deepseek-chat",
            "moonshot/kimi-k2.5",
            "google/gemini-3.1-flash-lite",
            "google/gemini-2.5-flash-lite",
            "openai/gpt-5.4-nano",
            "xai/grok-4-fast-non-reasoning",
            "free/gpt-oss-120b",
        ],
    },
    "MEDIUM": {
        "primary": "moonshot/kimi-k2.7",
        "fallback": [
            "moonshot/kimi-k2.6",
            "moonshot/kimi-k2.5",
            "google/gemini-3-flash-preview",
            "deepseek/deepseek-chat",
            "google/gemini-2.5-flash",
            "google/gemini-3.1-flash-lite",
            "google/gemini-2.5-flash-lite",
            "xai/grok-4-1-fast-non-reasoning",
            "xai/grok-3-mini",
        ],
    },
    "COMPLEX": {
        "primary": "google/gemini-3.1-pro",
        "fallback": [
            "google/gemini-3-flash-preview",
            "xai/grok-4-0709",
            "google/gemini-2.5-pro",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-sonnet-4.6",
            "deepseek/deepseek-chat",
            "google/gemini-2.5-flash",
            "openai/gpt-5.6-terra",
            "openai/gpt-5.5",
            "openai/gpt-5.4",
        ],
    },
    "REASONING": {
        "primary": "xai/grok-4-1-fast-reasoning",
        "fallback": [
            "xai/grok-4-fast-reasoning",
            "deepseek/deepseek-reasoner",
            "deepseek/deepseek-v4-pro",
            "openai/o4-mini",
            "openai/o3",
        ],
    },
}

ECO_TIERS: TierConfig = {
    "SIMPLE": {
        "primary": "free/gpt-oss-120b",
        "fallback": [
            "free/gpt-oss-20b",
            "free/deepseek-v4-flash",
            "google/gemini-3.1-flash-lite",
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash-lite",
            "xai/grok-4-fast-non-reasoning",
        ],
    },
    "MEDIUM": {
        "primary": "google/gemini-3.1-flash-lite",
        "fallback": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash-lite",
            "xai/grok-4-fast-non-reasoning",
            "google/gemini-2.5-flash",
        ],
    },
    "COMPLEX": {
        "primary": "google/gemini-3.1-flash-lite",
        "fallback": [
            "google/gemini-2.5-flash-lite",
            "xai/grok-4-0709",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-chat",
        ],
    },
    "REASONING": {
        "primary": "xai/grok-4-1-fast-reasoning",
        "fallback": [
            "xai/grok-4-fast-reasoning",
            "deepseek/deepseek-reasoner",
            "deepseek/deepseek-v4-pro",
        ],
    },
}

PREMIUM_TIERS: TierConfig = {
    "SIMPLE": {
        "primary": "moonshot/kimi-k2.7",
        "fallback": [
            "moonshot/kimi-k2.6",
            "moonshot/kimi-k2.5",
            "google/gemini-2.5-flash",
            "anthropic/claude-haiku-4.5",
            "google/gemini-2.5-flash-lite",
            "deepseek/deepseek-chat",
        ],
    },
    "MEDIUM": {
        "primary": "openai/gpt-5.3-codex",
        "fallback": [
            "moonshot/kimi-k2.7",
            "moonshot/kimi-k2.6",
            "moonshot/kimi-k2.5",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "xai/grok-4-0709",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-sonnet-4.6",
        ],
    },
    "COMPLEX": {
        "primary": "anthropic/claude-fable-5",
        "fallback": [
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.6",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-sonnet-4.6",
            "xai/grok-4.5",
            "xai/grok-4-0709",
            "moonshot/kimi-k2.7",
            "moonshot/kimi-k2.6",
            "moonshot/kimi-k2.5",
            "openai/gpt-5.6-terra",
            "openai/gpt-5.5",
            "openai/gpt-5.4",
            "openai/gpt-5.3-codex",
            "deepseek/deepseek-chat",
            "free/gpt-oss-120b",
        ],
    },
    "REASONING": {
        "primary": "anthropic/claude-sonnet-4.6",
        "fallback": [
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.6",
            "xai/grok-4-1-fast-reasoning",
            "openai/o4-mini",
            "openai/o3",
        ],
    },
}

AGENTIC_TIERS: TierConfig = {
    "SIMPLE": {
        "primary": "openai/gpt-4o-mini",
        "fallback": [
            "moonshot/kimi-k2.5",
            "anthropic/claude-haiku-4.5",
            "xai/grok-4-1-fast-non-reasoning",
        ],
    },
    "MEDIUM": {
        "primary": "moonshot/kimi-k2.7",
        "fallback": [
            "moonshot/kimi-k2.6",
            "moonshot/kimi-k2.5",
            "xai/grok-4-1-fast-non-reasoning",
            "openai/gpt-4o-mini",
            "anthropic/claude-haiku-4.5",
            "deepseek/deepseek-chat",
        ],
    },
    "COMPLEX": {
        "primary": "anthropic/claude-sonnet-4.6",
        "fallback": [
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.6",
            "xai/grok-4-0709",
            "moonshot/kimi-k2.7",
            "moonshot/kimi-k2.5",
            "openai/gpt-5.6-terra",
            "openai/gpt-5.5",
            "openai/gpt-5.4",
            "deepseek/deepseek-chat",
            "free/gpt-oss-120b",
        ],
    },
    "REASONING": {
        "primary": "anthropic/claude-sonnet-4.6",
        "fallback": [
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.6",
            "xai/grok-4-1-fast-reasoning",
            "deepseek/deepseek-reasoner",
        ],
    },
}

EVIDENCE_CANDIDATES: dict[TaskType, list[str]] = {
    "code_agent": [
        "openai/gpt-5.3-codex",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5-mini",
        "google/gemini-3.5-flash",
        "moonshot/kimi-k3",
        "deepseek/deepseek-v4-pro",
    ],
    "tool_agent": [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-5",
        "openai/gpt-5-mini",
        "openai/gpt-4.1",
        "openai/gpt-4o-mini",
        "google/gemini-3.5-flash",
        "openai/gpt-5.3-codex",
        "moonshot/kimi-k3",
        "deepseek/deepseek-v4-pro",
    ],
    "tool_agent_parallel": [
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5-mini",
        "openai/gpt-4.1",
        "openai/gpt-4o-mini",
        "xai/grok-4.5",
        "google/gemini-3.5-flash",
        "deepseek/deepseek-v4-pro",
    ],
    "long_context": [
        "google/gemini-3.1-pro",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-max",
        "zai/glm-5.2",
        "google/gemini-3.5-flash",
    ],
    "reasoning_mcq": [
        "google/gemini-3-flash-preview",
        "google/gemini-3.5-flash",
        "xai/grok-4.5",
        "anthropic/claude-sonnet-5",
        "deepseek/deepseek-v4-pro",
    ],
    "reasoning_math": [
        "google/gemini-3.5-flash",
        "xai/grok-4.5",
        "anthropic/claude-sonnet-5",
        "deepseek/deepseek-v4-pro",
        "moonshot/kimi-k3",
    ],
}

NO_TOOL_MODELS = {
    "free/deepseek-v4-flash",
    "free/gpt-oss-120b",
    "free/gpt-oss-20b",
    "free/seed-oss-36b",
    "google/gemini-3-flash-preview",
}

VISION_MODELS = {
    "anthropic/claude-fable-5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-5",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro",
    "google/gemini-3.5-flash",
    "moonshot/kimi-k2.5",
    "moonshot/kimi-k2.6",
    "moonshot/kimi-k2.7",
    "moonshot/kimi-k3",
    "openai/gpt-4.1",
    "openai/gpt-5.4",
    "openai/gpt-5.5",
    "openai/gpt-5.6-terra",
    "xai/grok-4.5",
}

OUTPUT_LIMITS = {
    "anthropic/claude-haiku-4.5": 8_192,
    "deepseek/deepseek-chat": 8_192,
    "deepseek/deepseek-reasoner": 8_192,
    "google/gemini-3.1-flash-lite": 8_192,
    "xai/grok-3-mini": 16_384,
    "xai/grok-4-0709": 16_384,
    "xai/grok-4-1-fast-non-reasoning": 16_384,
    "xai/grok-4-1-fast-reasoning": 16_384,
    "xai/grok-4-fast-non-reasoning": 16_384,
    "xai/grok-4-fast-reasoning": 16_384,
    "xai/grok-4.5": 16_384,
}

CONTEXT_LIMITS = {
    "openai/gpt-4.1": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-5-mini": 200_000,
    "openai/gpt-5.3-codex": 400_000,
    "moonshot/kimi-k2.5": 262_144,
    "moonshot/kimi-k2.6": 262_144,
    "moonshot/kimi-k2.7": 262_144,
    "xai/grok-3-mini": 131_072,
    "xai/grok-4-0709": 131_072,
    "xai/grok-4-1-fast-non-reasoning": 131_072,
    "xai/grok-4-1-fast-reasoning": 131_072,
    "xai/grok-4-fast-non-reasoning": 131_072,
    "xai/grok-4-fast-reasoning": 131_072,
}

PORTFOLIO_WEIGHTS = {
    "auto": {
        "quality": 0.47,
        "capability": 0.20,
        "cost": 0.18,
        "speed": 0.07,
        "reliability": 0.03,
        "legacy": 0.05,
    },
    "eco": {
        "quality": 0.36,
        "capability": 0.20,
        "cost": 0.28,
        "speed": 0.10,
        "reliability": 0.04,
        "legacy": 0.02,
    },
    "premium": {
        "quality": 0.58,
        "capability": 0.20,
        "cost": 0.08,
        "speed": 0.06,
        "reliability": 0.06,
        "legacy": 0.02,
    },
}
AFFINITY_FLOOR = {"auto": 0.10, "eco": 0.22, "premium": 0.05}


def _sample(text: str, limit: int = 8_000) -> str:
    if len(text) <= limit:
        return text
    first = math.ceil(limit / 2)
    return f"{text[:first]}\n{text[-(limit - first) :]}"


def _infer_tool_requirement(prompt: str, system: str | None, tool_choice: Any) -> bool:
    if tool_choice == "none":
        return False
    if tool_choice == "required" or isinstance(tool_choice, dict):
        return True
    # System prompts usually describe every tool a host exposes and are not
    # evidence that the user requested an action on this turn.
    del system
    text = prompt
    return bool(
        re.search(
            r"\b(?:get|fetch|search|look up|check|update|change|create|delete|cancel|book|send|run|execute|open|read|write|edit|deploy|install)\b|"
            r"(?:查询|搜索|查看|获取|更新|修改|创建|删除|取消|预订|发送|执行|打开|读取|写入|部署|安装)",
            text,
            re.IGNORECASE,
        )
    )


def _parallel(prompt: str, needs_tools: bool, tool_names: list[str]) -> bool:
    if not needs_tools or not tool_names:
        return False
    if re.search(
        r"\b(?:in parallel|simultaneously|concurrently|for each|each of|every one|both|(?:two|three|multiple|several)\s+(?:cities|locations|items|tasks|orders|users|files))\b|"
        r"并行|同时|分别|每个|各自|(?:两个|三个|多个)(?:城市|地点|项目|任务|订单|用户|文件)",
        prompt,
        re.IGNORECASE,
    ):
        return True
    lookup = re.search(
        r"\b(?:weather|climate|temperature|news|report)\b|天气|气象|温度|新闻|报告",
        prompt,
        re.IGNORECASE,
    )
    return bool(
        lookup
        and (
            len(re.findall(r"[,，]", prompt)) >= 2
            or re.search(r"\band\b|以及|和|、", prompt, re.IGNORECASE)
        )
    )


def _task_features(
    prompt: str,
    system_prompt: str | None,
    tools: list[dict[str, Any]],
    tool_choice: Any,
    requires_structured_output: bool,
    has_vision: bool,
) -> dict[str, Any]:
    scanned = _sample(prompt)
    scanned_system = _sample(system_prompt or "")
    full = f"{scanned_system} {scanned}"
    lower = scanned.lower()
    estimated = math.ceil(len(f"{system_prompt or ''} {prompt}") / 4)
    names = [str(tool.get("function", {}).get("name", "")).lower() for tool in tools]
    has_code = bool(
        re.search(
            r"```|\b(?:typescript|javascript|python|rust|java|sql|stack trace|traceback|exception)\b|\.(?:ts|tsx|js|py|go|rs)\b",
            scanned,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:implement|refactor|debug|write|edit|modify|create|define|review|fix)\b.{0,48}\b(?:api|function|class|method)\b",
            scanned,
            re.IGNORECASE | re.DOTALL,
        )
    )
    needs_tools = bool(tools) and _infer_tool_requirement(scanned, scanned_system, tool_choice)
    likely_parallel = _parallel(scanned, needs_tools, names)
    airline = any(
        re.search(r"flight|reservation|airport|baggage|passenger", name) for name in names
    )
    retail = any(re.search(r"order|product|item|return|exchange|address", name) for name in names)
    web = any(re.fullmatch(r"web_?search|web_?fetch", name) for name in names)
    domain = (
        "airline"
        if airline and not retail
        else "retail"
        if retail and not airline
        else "web_research"
        if web
        else "other"
    )
    deep_research = domain == "web_research" and bool(
        re.search(
            r"exact answer|best-supported answer|following clues|multiple public sources|精确答案|多个公开来源",
            full,
            re.IGNORECASE,
        )
        or (
            len(prompt) >= 320
            and re.search(
                r"identify|who is|who was|find the person|找出|识别|是谁", full, re.IGNORECASE
            )
        )
    )
    terminal = any(
        re.fullmatch(r"terminalexec|terminalinspect|terminalsendkeys", name) for name in names
    )
    high_stakes = bool(
        re.search(
            r"\b(?:production|security|payment|legal|medical|financial|audit)\b|生产|安全|支付|法律|医疗|财务|审计",
            full,
            re.IGNORECASE,
        )
    )
    risk = "standard"
    if needs_tools and re.search(r"\b(?:return|exchange)\b|退货|换货", scanned, re.IGNORECASE):
        risk = "high"
    if (
        needs_tools
        and domain == "airline"
        and re.search(
            r"\b(?:cheapest|lowest price|all reservations|every passenger)\b|最便宜|所有",
            scanned,
            re.IGNORECASE,
        )
    ):
        risk = "complex_high"
    if terminal and re.search(
        r"\b(?:multiple|several)\s+(?:scripts?|files?)\b|fix all the issues|pipeline.*(?:fail|fix)",
        scanned,
        re.IGNORECASE | re.DOTALL,
    ):
        risk = "complex_high"
    multiple_choice = len(re.findall(r"(?:^|\n)\s*[A-D][.)]\s+", scanned, re.IGNORECASE))
    numeric = len(re.findall(r"-?\d+(?:[.,]\d+)?", scanned))
    compact_math = (
        not has_code
        and len(prompt) < 2_500
        and numeric >= 2
        and bool(
            re.search(
                r"[+×÷=%$€£¥]|\b(?:total|each|per|times|half|twice|percent|how many|how much|calculate)\b",
                scanned,
                re.IGNORECASE,
            )
            or re.search(r"[?？]\s*$", scanned)
            or numeric >= 3
        )
    )
    task: TaskType = "chat"
    if has_vision:
        task = "vision"
    elif estimated > 80_000:
        task = "long_context"
    elif needs_tools and (
        has_code
        or (terminal and re.search(r"\b(?:file|script|server|endpoint)\b", scanned, re.IGNORECASE))
    ):
        task = "code_agent"
    elif needs_tools and likely_parallel:
        task = "tool_agent_parallel"
    elif needs_tools:
        task = "tool_agent"
    elif multiple_choice >= 3:
        task = "reasoning_mcq"
    elif compact_math:
        task = "reasoning_math"
    elif re.search(
        r"\b(?:bug|debug|error|failure|failing|regression|crash|修复|报错|错误|调试)\b",
        lower,
        re.IGNORECASE,
    ):
        task = "debug"
    elif has_code or re.search(
        r"\b(?:refactor|implement|patch|edit|rewrite|重构|实现|修改)\b", lower, re.IGNORECASE
    ):
        task = "code_edit"
    elif requires_structured_output or re.search(
        r"\b(?:extract|json|schema|csv|字段|提取)\b", lower, re.IGNORECASE
    ):
        task = "extraction"
    elif re.search(
        r"\b(?:prove|derive|theorem|formal|mathematical|reasoning|证明|推导|定理|数学)\b",
        lower,
        re.IGNORECASE,
    ):
        task = "reasoning"
    return {
        "task_type": task,
        "estimated_input_tokens": estimated,
        "needs_tools": needs_tools,
        "needs_vision": has_vision,
        "needs_structured_output": requires_structured_output,
        "language": "zh" if re.search(r"[\u3400-\u9fff]", full) else "other",
        "domain": domain,
        "deep_research": deep_research,
        "risk": risk,
        "terminal": terminal,
        "terminal_safety": terminal and high_stakes,
    }


def _affinity(model_id: str, features: dict[str, Any]) -> float:
    name = model_id.split("/", 1)[-1].lower()
    task = features["task_type"]
    domain = features["domain"]
    risk = features["risk"]
    terminal = features["terminal"]
    safety = features["terminal_safety"]
    base = 0.68

    def score(mapping: dict[str, float]) -> float:
        return max(base, mapping.get(name, 0.0))

    if task == "code_agent":
        if terminal and risk == "complex_high":
            return score(
                {
                    "claude-sonnet-5": 1,
                    "gpt-5.3-codex": 0.87,
                    "gpt-5-mini": 0.78,
                    "gemini-3.5-flash": 0.76,
                }
            )
        return score(
            {
                "gpt-5.3-codex": 1,
                "claude-sonnet-5": 0.98,
                "gpt-5-mini": 0.96,
                "gemini-3.5-flash": 0.92,
                "kimi-k3": 0.9,
                "deepseek-v4-pro": 0.88,
                "glm-5.2": 0.88,
            }
        )
    if task == "tool_agent":
        if terminal and risk == "complex_high":
            return score(
                {
                    "claude-sonnet-5": 1,
                    "gpt-5.3-codex": 0.87,
                    "gpt-5-mini": 0.78,
                    "gemini-3.5-flash": 0.76,
                }
            )
        if terminal and not safety:
            return score(
                {
                    "gpt-5-mini": 1,
                    "gpt-5.3-codex": 0.98,
                    "claude-sonnet-5": 0.9,
                    "gemini-3.5-flash": 0.89,
                }
            )
        if domain == "web_research":
            return score(
                {
                    "claude-sonnet-5": 1,
                    "gpt-5-mini": 0.88,
                    "gemini-3.5-flash": 0.84 if features["deep_research"] else 0.86,
                    "claude-opus-5": 0.8 if features["deep_research"] else 0.84,
                    "claude-opus-4.8": 0.78 if features["deep_research"] else 0.82,
                }
            )
        if domain in {"retail", "airline"} and risk == "standard":
            return score({"gpt-5-mini": 1, "claude-sonnet-5": 0.9, "gemini-3.5-flash": 0.82})
        if domain == "retail" and risk != "standard":
            return score(
                {
                    "deepseek-v4-pro": 1,
                    "claude-sonnet-5": 0.88,
                    "gemini-3.5-flash": 0.82,
                    "gpt-5-mini": 0.76,
                }
            )
        if domain == "airline" and risk == "complex_high":
            return score({"claude-sonnet-5": 1, "gpt-5-mini": 0.78, "gemini-3.5-flash": 0.76})
        return score(
            {
                "claude-sonnet-5": 1,
                "gemini-3.5-flash": 0.88,
                "gpt-5.3-codex": 0.87,
                "kimi-k3": 0.85,
                "gpt-5-mini": 0.84,
                "deepseek-v4-pro": 0.82,
            }
        )
    if task == "tool_agent_parallel":
        if terminal:
            return score(
                {
                    "gpt-5-mini": 1,
                    "gpt-5.3-codex": 0.98,
                    "claude-sonnet-5": 0.92,
                    "gemini-3.5-flash": 0.88,
                }
            )
        if domain in {"retail", "airline"}:
            return score(
                {
                    "deepseek-v4-pro": 1,
                    "claude-sonnet-5": 0.88,
                    "claude-opus-4.8": 0.84,
                    "gpt-5-mini": 0.78,
                }
            )
        return score(
            {
                "claude-opus-4.8": 1,
                "claude-sonnet-5": 0.84,
                "grok-4.5": 0.82,
                "gemini-3.5-flash": 0.8,
                "deepseek-v4-pro": 0.78,
            }
        )
    if task in {"code_edit", "debug"}:
        return score(
            {
                "gpt-5.3-codex": 1,
                "claude-sonnet-4.6": 0.94,
                "glm-5.2": 0.9,
                "kimi-k2.7": 0.86,
                "deepseek-v4-pro": 0.86,
            }
        )
    if task == "reasoning":
        return score(
            {
                "claude-sonnet-5": 0.98,
                "claude-sonnet-4.6": 0.98,
                "deepseek-v4-pro": 0.95,
                "grok-4.5": 0.94,
                "gemini-3.1-pro": 0.92,
                "gemini-3.5-flash": 0.92,
            }
        )
    if task == "reasoning_mcq":
        return score(
            {
                "gemini-3-flash-preview": 1,
                "gemini-3.5-flash": 0.91,
                "grok-4.5": 0.9,
                "claude-sonnet-5": 0.88,
                "deepseek-v4-pro": 0.84,
            }
        )
    if task == "reasoning_math":
        return score(
            {
                "gemini-3.5-flash": 1,
                "grok-4.5": 0.93,
                "claude-sonnet-5": 0.9,
                "deepseek-v4-pro": 0.9,
                "kimi-k3": 0.9,
                "kimi-k2.7": 0.84,
            }
        )
    if task == "vision":
        return score(
            {
                "gemini-3.1-pro": 0.96,
                "qwen3.7-max": 0.9,
                "claude-sonnet-4.6": 0.9,
                "kimi-k2.7": 0.9,
                "grok-4.3": 0.9,
            }
        )
    if task == "long_context":
        return score(
            {
                "gemini-3.1-pro": 1,
                "qwen3.7-max": 0.89,
                "glm-5.2": 0.89,
                "gemini-3.5-flash": 0.88,
                "deepseek-v4-pro": 0.85,
            }
        )
    if task == "extraction":
        kimi = 1 if features["language"] == "zh" else 0.9
        return score(
            {
                "gemini-3.5-flash": 0.9,
                "gemini-2.5-flash": 0.9,
                "gpt-4o-mini": 0.9,
                "claude-sonnet-5": 0.9,
                "claude-sonnet-4.6": 0.9,
                "kimi-k3": kimi,
                "kimi-k2.7": kimi,
            }
        )
    return score(
        {"gemini-3.5-flash": 0.86, "gemini-2.5-flash": 0.86, "kimi-k3": 0.86, "kimi-k2.7": 0.86}
    )


def _eligible(model: str, features: dict[str, Any], max_output_tokens: int) -> bool:
    if features["needs_tools"] and model in NO_TOOL_MODELS:
        return False
    if features["needs_vision"] and model not in VISION_MODELS:
        return False
    if features["needs_structured_output"] and model in NO_TOOL_MODELS:
        return False
    if OUTPUT_LIMITS.get(model, 65_536) < max_output_tokens:
        return False
    context = CONTEXT_LIMITS.get(model, 1_000_000)
    return bool(context >= (features["estimated_input_tokens"] + max_output_tokens) * 1.1)


def _cost(
    model: str, pricing: dict[str, dict[str, float]], input_tokens: int, output_tokens: int
) -> float:
    price = pricing.get(model)
    if not price:
        return math.inf
    flat = float(price.get("flat_price", 0) or 0)
    if flat:
        return flat
    return (
        input_tokens * float(price.get("input_price", 0) or 0)
        + output_tokens * float(price.get("output_price", 0) or 0)
    ) / 1_000_000


def route(
    prompt: str,
    system_prompt: str | None,
    max_output_tokens: int,
    model_pricing: dict[str, dict[str, float]],
    routing_profile: RoutingProfile = "auto",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    requires_structured_output: bool = False,
    has_vision: bool = False,
    minimum_payment_usd: float = 0.001,
) -> RoutingDecision:
    """Return a deterministic V3 portfolio decision with no model call."""

    tools = tools or []
    features = _task_features(
        prompt,
        system_prompt,
        tools,
        tool_choice,
        requires_structured_output,
        has_vision,
    )
    estimated = features["estimated_input_tokens"]
    rules = classify_by_rules(prompt, system_prompt, estimated)
    tier: Tier = rules["tier"] or "MEDIUM"
    if estimated > 100_000:
        tier = "COMPLEX"
    if features["task_type"] in {"reasoning_mcq", "reasoning_math"} and tier in {
        "SIMPLE",
        "MEDIUM",
    }:
        tier = "REASONING"

    normalized_profile = "eco" if routing_profile == "free" else routing_profile
    profile_name = normalized_profile if normalized_profile in {"eco", "premium"} else "auto"
    if profile_name == "eco":
        tiers = ECO_TIERS
        decision_profile = "eco"
    elif profile_name == "premium":
        tiers = PREMIUM_TIERS
        decision_profile = "premium"
    elif features["needs_tools"] or float(rules.get("agentic_score", 0)) >= 0.5:
        tiers = AGENTIC_TIERS
        decision_profile = "agentic"
    else:
        tiers = AUTO_TIERS
        decision_profile = "auto"

    configured = [tiers[tier]["primary"], *tiers[tier]["fallback"]]
    chain = list(dict.fromkeys([*configured, *EVIDENCE_CANDIDATES.get(features["task_type"], [])]))
    chain = [model for model in chain if model in model_pricing]
    eligible = [model for model in chain if _eligible(model, features, max_output_tokens)]
    available = eligible or chain
    if not available:
        raise ValueError("Router found no model present in the current BlockRun catalog")

    affinities = {model: _affinity(model, features) for model in available}
    best_affinity = max(affinities.values())
    specific = [model for model in available if affinities[model] > 0.68]
    pool = specific or [available[0]]
    gap = AFFINITY_FLOOR[profile_name]
    if features["terminal"]:
        gap = max(gap, 0.15 if features["terminal_safety"] else 0.12)
    candidates = [model for model in pool if affinities[model] >= best_affinity - gap]
    raw_costs = [_cost(model, model_pricing, estimated, max_output_tokens) for model in candidates]
    finite = [cost for cost in raw_costs if math.isfinite(cost)]
    min_cost = min(finite) if finite else 0
    max_cost = max(finite) if finite else 1
    weights = PORTFOLIO_WEIGHTS[profile_name]
    ranked_entries: list[dict[str, float | str]] = []
    for index, model in enumerate(candidates):
        raw = _cost(model, model_pricing, estimated, max_output_tokens)
        cost_score = (
            1 - (raw - min_cost) / (max_cost - min_cost)
            if math.isfinite(raw) and max_cost > min_cost
            else 0.5
        )
        legacy = 1 - index / max(1, len(candidates) - 1)
        quality_weight = weights["quality"] + (
            0.08
            if re.search(
                r"production|security|payment|legal|medical|financial|audit|生产|安全|支付|法律|医疗|财务|审计",
                f"{system_prompt or ''} {prompt}",
                re.IGNORECASE,
            )
            else 0
        )
        score_value = (
            affinities[model] * quality_weight
            + weights["capability"]
            + cost_score * weights["cost"]
            + 0.5 * weights["speed"]
            + 1.0 * weights["reliability"]
            + legacy * weights["legacy"]
        )
        ranked_entries.append(
            {
                "model": model,
                "score": score_value,
                "quality": affinities[model],
                "cost": cost_score,
                "speed": 0.5,
                "reliability": 1.0,
            }
        )
    ranked_entries.sort(key=lambda entry: float(entry["score"]), reverse=True)
    scored = [str(entry["model"]) for entry in ranked_entries]
    ranked = [*scored, *[model for model in available if model not in scored]]
    model = ranked[0]

    raw_selected = _cost(model, model_pricing, estimated, max_output_tokens)
    selected_price = model_pricing[model]
    if selected_price.get("flat_price"):
        estimated_cost = max(float(selected_price["flat_price"]), minimum_payment_usd)
    else:
        estimated_cost = max(raw_selected * 1.05, minimum_payment_usd)
    baseline = (estimated * 5.0 + max_output_tokens * 30.0) / 1_000_000
    savings = (
        0.0
        if profile_name == "premium" or baseline <= 0
        else max(0.0, (baseline - estimated_cost) / baseline)
    )
    confidence = (
        0.95 if estimated > 100_000 else float(rules["confidence"] if rules["tier"] else 0.5)
    )
    return {
        "model": model,
        "tier": tier,
        "confidence": confidence,
        "method": "portfolio",
        "reasoning": f"score={rules['score']:.2f} | v3 task={features['task_type']} candidates={len(ranked)}",
        "cost_estimate": estimated_cost,
        "baseline_cost": baseline,
        "savings": savings,
        "profile": decision_profile,
        "task_type": features["task_type"],
        "router_version": "v3-portfolio",
        "candidates": ranked,
        "candidate_scores": ranked_entries,
        "fallbacks": ranked[1:],
    }
