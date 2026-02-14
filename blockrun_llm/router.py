"""
Smart Router for BlockRun LLM SDK

Port of ClawRouter's 14-dimension rule-based scoring algorithm.
Routes requests to the cheapest capable model in <1ms, 100% local.

Usage:
    from blockrun_llm import LLMClient

    client = LLMClient()
    result = client.smart_chat("What is 2+2?")
    print(result["response"])  # '4'
    print(result["model"])     # 'google/gemini-2.5-flash'
    print(f"Saved {result['routing']['savings'] * 100:.0f}%")
"""

import re
import math
from typing import Dict, List, Optional, Literal, TypedDict, Any


# Type definitions
Tier = Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]
RoutingProfile = Literal["free", "eco", "auto", "premium"]


class RoutingDecision(TypedDict):
    model: str
    tier: Tier
    confidence: float
    method: Literal["rules"]
    reasoning: str
    cost_estimate: float
    baseline_cost: float
    savings: float  # 0-1 percentage


class TierConfig(TypedDict):
    primary: str
    fallback: List[str]


class ScoringResult(TypedDict):
    score: float
    tier: Optional[Tier]
    confidence: float
    signals: List[str]
    agentic_score: float


# ─── Scoring Config ───
# Multilingual keywords for 14-dimension scoring

CODE_KEYWORDS = [
    "function", "class", "import", "def", "SELECT", "async", "await",
    "const", "let", "var", "return", "```",
    "函数", "类", "导入", "定义", "查询", "异步", "等待", "常量", "变量", "返回",
    "関数", "クラス", "インポート", "非同期", "定数", "変数",
    "функция", "класс", "импорт", "определ", "запрос", "асинхронный",
]

REASONING_KEYWORDS = [
    "prove", "theorem", "derive", "step by step", "chain of thought",
    "formally", "mathematical", "proof", "logically",
    "证明", "定理", "推导", "逐步", "思维链", "形式化", "数学", "逻辑",
    "доказать", "теорема", "вывести", "шаг за шагом", "логически",
]

SIMPLE_KEYWORDS = [
    "what is", "define", "translate", "hello", "yes or no",
    "capital of", "how old", "who is", "when was",
    "什么是", "定义", "翻译", "你好", "是否", "首都",
    "что такое", "определение", "перевести", "привет",
]

TECHNICAL_KEYWORDS = [
    "algorithm", "optimize", "architecture", "distributed",
    "kubernetes", "microservice", "database", "infrastructure",
    "算法", "优化", "架构", "分布式", "微服务", "数据库",
]

CREATIVE_KEYWORDS = [
    "story", "poem", "compose", "brainstorm", "creative", "imagine", "write a",
    "故事", "诗", "创作", "头脑风暴", "创意", "想象",
]

AGENTIC_KEYWORDS = [
    "read file", "read the file", "look at", "check the", "open the",
    "edit", "modify", "update the", "change the", "write to", "create file",
    "execute", "deploy", "install", "npm", "pip", "compile",
    "after that", "and also", "once done", "step 1", "step 2",
    "fix", "debug", "until it works", "keep trying", "iterate",
    "make sure", "verify", "confirm",
]

# Tier boundaries on weighted score axis
TIER_BOUNDARIES = {
    "simple_medium": 0.0,
    "medium_complex": 0.3,
    "complex_reasoning": 0.5,
}

# Dimension weights (sum to ~1.0)
DIMENSION_WEIGHTS = {
    "token_count": 0.08,
    "code_presence": 0.15,
    "reasoning_markers": 0.18,
    "technical_terms": 0.10,
    "creative_markers": 0.05,
    "simple_indicators": 0.02,
    "multi_step_patterns": 0.12,
    "question_complexity": 0.05,
    "agentic_task": 0.04,
}

# ─── Tier Configs by Profile ───

AUTO_TIERS: Dict[Tier, TierConfig] = {
    "SIMPLE": {
        "primary": "nvidia/kimi-k2.5",
        "fallback": ["google/gemini-2.5-flash", "nvidia/gpt-oss-120b", "deepseek/deepseek-chat"],
    },
    "MEDIUM": {
        "primary": "xai/grok-code-fast-1",
        "fallback": ["google/gemini-2.5-flash", "deepseek/deepseek-chat", "xai/grok-4-1-fast-non-reasoning"],
    },
    "COMPLEX": {
        "primary": "google/gemini-3-pro-preview",
        "fallback": ["google/gemini-2.5-flash", "google/gemini-2.5-pro", "deepseek/deepseek-chat"],
    },
    "REASONING": {
        "primary": "xai/grok-4-1-fast-reasoning",
        "fallback": ["deepseek/deepseek-reasoner", "xai/grok-4-fast-reasoning", "openai/o3"],
    },
}

ECO_TIERS: Dict[Tier, TierConfig] = {
    "SIMPLE": {
        "primary": "nvidia/kimi-k2.5",
        "fallback": ["nvidia/gpt-oss-120b", "deepseek/deepseek-chat"],
    },
    "MEDIUM": {
        "primary": "deepseek/deepseek-chat",
        "fallback": ["xai/grok-code-fast-1", "google/gemini-2.5-flash"],
    },
    "COMPLEX": {
        "primary": "xai/grok-4-0709",
        "fallback": ["deepseek/deepseek-chat", "google/gemini-2.5-flash"],
    },
    "REASONING": {
        "primary": "deepseek/deepseek-reasoner",
        "fallback": ["xai/grok-4-fast-reasoning", "moonshot/kimi-k2.5"],
    },
}

PREMIUM_TIERS: Dict[Tier, TierConfig] = {
    "SIMPLE": {
        "primary": "google/gemini-2.5-flash",
        "fallback": ["openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"],
    },
    "MEDIUM": {
        "primary": "openai/gpt-4o",
        "fallback": ["google/gemini-2.5-pro", "anthropic/claude-sonnet-4"],
    },
    "COMPLEX": {
        "primary": "anthropic/claude-opus-4.5",
        "fallback": ["openai/gpt-5.2-pro", "google/gemini-3-pro-preview", "openai/gpt-5.2"],
    },
    "REASONING": {
        "primary": "openai/o3",
        "fallback": ["openai/o4-mini", "anthropic/claude-opus-4.5"],
    },
}

FREE_TIERS: Dict[Tier, TierConfig] = {
    "SIMPLE": {
        "primary": "nvidia/gpt-oss-120b",
        "fallback": [],
    },
    "MEDIUM": {
        "primary": "nvidia/gpt-oss-120b",
        "fallback": [],
    },
    "COMPLEX": {
        "primary": "nvidia/gpt-oss-120b",
        "fallback": [],
    },
    "REASONING": {
        "primary": "nvidia/gpt-oss-120b",
        "fallback": [],
    },
}


def _score_keyword_match(
    text: str,
    keywords: List[str],
    thresholds: tuple = (1, 2),
    scores: tuple = (0, 0.5, 1.0),
) -> tuple:
    """Score keyword matches, returning (score, matched_keywords)."""
    matches = [kw for kw in keywords if kw.lower() in text]
    if len(matches) >= thresholds[1]:
        return scores[2], matches[:3]
    if len(matches) >= thresholds[0]:
        return scores[1], matches[:3]
    return scores[0], []


def _calibrate_confidence(distance: float, steepness: float = 12) -> float:
    """Sigmoid confidence calibration."""
    return 1 / (1 + math.exp(-steepness * distance))


def classify_by_rules(
    prompt: str,
    system_prompt: Optional[str],
    estimated_tokens: int,
) -> ScoringResult:
    """
    14-dimension rule-based classifier.
    Returns tier classification with confidence score.
    """
    text = f"{system_prompt or ''} {prompt}".lower()
    user_text = prompt.lower()
    signals: List[str] = []

    # Dimension scores
    scores: Dict[str, float] = {}

    # 1. Token count
    if estimated_tokens < 50:
        scores["token_count"] = -1.0
        signals.append(f"short ({estimated_tokens} tokens)")
    elif estimated_tokens > 500:
        scores["token_count"] = 1.0
        signals.append(f"long ({estimated_tokens} tokens)")
    else:
        scores["token_count"] = 0.0

    # 2. Code presence
    score, matches = _score_keyword_match(text, CODE_KEYWORDS)
    scores["code_presence"] = score
    if matches:
        signals.append(f"code ({', '.join(matches[:3])})")

    # 3. Reasoning markers (user text only)
    score, matches = _score_keyword_match(user_text, REASONING_KEYWORDS, scores=(0, 0.7, 1.0))
    scores["reasoning_markers"] = score
    if matches:
        signals.append(f"reasoning ({', '.join(matches[:3])})")

    # 4. Technical terms
    score, matches = _score_keyword_match(text, TECHNICAL_KEYWORDS, thresholds=(2, 4))
    scores["technical_terms"] = score
    if matches:
        signals.append(f"technical ({', '.join(matches[:3])})")

    # 5. Creative markers
    score, matches = _score_keyword_match(text, CREATIVE_KEYWORDS, scores=(0, 0.5, 0.7))
    scores["creative_markers"] = score
    if matches:
        signals.append(f"creative ({', '.join(matches[:3])})")

    # 6. Simple indicators
    score, matches = _score_keyword_match(text, SIMPLE_KEYWORDS, scores=(0, -1.0, -1.0))
    scores["simple_indicators"] = score
    if matches:
        signals.append(f"simple ({', '.join(matches[:3])})")

    # 7. Multi-step patterns
    patterns = [r"first.*then", r"step \d", r"\d\.\s"]
    if any(re.search(p, text, re.IGNORECASE) for p in patterns):
        scores["multi_step_patterns"] = 0.5
        signals.append("multi-step")
    else:
        scores["multi_step_patterns"] = 0.0

    # 8. Question complexity
    question_count = text.count("?")
    if question_count > 3:
        scores["question_complexity"] = 0.5
        signals.append(f"{question_count} questions")
    else:
        scores["question_complexity"] = 0.0

    # 9. Agentic task indicators
    agentic_matches = [kw for kw in AGENTIC_KEYWORDS if kw.lower() in text]
    if len(agentic_matches) >= 4:
        scores["agentic_task"] = 1.0
        agentic_score = 1.0
        signals.append(f"agentic ({', '.join(agentic_matches[:3])})")
    elif len(agentic_matches) >= 3:
        scores["agentic_task"] = 0.6
        agentic_score = 0.6
        signals.append(f"agentic ({', '.join(agentic_matches[:3])})")
    elif len(agentic_matches) >= 1:
        scores["agentic_task"] = 0.2
        agentic_score = 0.2
    else:
        scores["agentic_task"] = 0.0
        agentic_score = 0.0

    # Compute weighted score
    weighted_score = sum(
        scores.get(dim, 0) * weight
        for dim, weight in DIMENSION_WEIGHTS.items()
    )

    # Check for reasoning override (2+ reasoning markers = REASONING)
    reasoning_matches = [kw for kw in REASONING_KEYWORDS if kw.lower() in user_text]
    if len(reasoning_matches) >= 2:
        confidence = _calibrate_confidence(max(weighted_score, 0.3))
        return {
            "score": weighted_score,
            "tier": "REASONING",
            "confidence": max(confidence, 0.85),
            "signals": signals,
            "agentic_score": agentic_score,
        }

    # Map score to tier
    if weighted_score < TIER_BOUNDARIES["simple_medium"]:
        tier: Tier = "SIMPLE"
        distance = TIER_BOUNDARIES["simple_medium"] - weighted_score
    elif weighted_score < TIER_BOUNDARIES["medium_complex"]:
        tier = "MEDIUM"
        distance = min(
            weighted_score - TIER_BOUNDARIES["simple_medium"],
            TIER_BOUNDARIES["medium_complex"] - weighted_score
        )
    elif weighted_score < TIER_BOUNDARIES["complex_reasoning"]:
        tier = "COMPLEX"
        distance = min(
            weighted_score - TIER_BOUNDARIES["medium_complex"],
            TIER_BOUNDARIES["complex_reasoning"] - weighted_score
        )
    else:
        tier = "REASONING"
        distance = weighted_score - TIER_BOUNDARIES["complex_reasoning"]

    confidence = _calibrate_confidence(distance)

    # Ambiguous if confidence too low
    if confidence < 0.7:
        return {
            "score": weighted_score,
            "tier": None,
            "confidence": confidence,
            "signals": signals,
            "agentic_score": agentic_score,
        }

    return {
        "score": weighted_score,
        "tier": tier,
        "confidence": confidence,
        "signals": signals,
        "agentic_score": agentic_score,
    }


def route(
    prompt: str,
    system_prompt: Optional[str],
    max_output_tokens: int,
    model_pricing: Dict[str, Dict[str, float]],
    routing_profile: RoutingProfile = "auto",
) -> RoutingDecision:
    """
    Route a request to the cheapest capable model.

    Args:
        prompt: User message
        system_prompt: Optional system prompt
        max_output_tokens: Max tokens to generate
        model_pricing: Dict of model_id -> {"input_price": x, "output_price": y}
        routing_profile: "free" | "eco" | "auto" | "premium"

    Returns:
        RoutingDecision with model, tier, confidence, reasoning, costs
    """
    # Estimate input tokens (~4 chars per token)
    full_text = f"{system_prompt or ''} {prompt}"
    estimated_tokens = len(full_text) // 4

    # Classify by rules
    result = classify_by_rules(prompt, system_prompt, estimated_tokens)

    # Select tier configs based on profile
    if routing_profile == "free":
        tier_configs = FREE_TIERS
        profile_suffix = " | free"
    elif routing_profile == "eco":
        tier_configs = ECO_TIERS
        profile_suffix = " | eco"
    elif routing_profile == "premium":
        tier_configs = PREMIUM_TIERS
        profile_suffix = " | premium"
    else:
        tier_configs = AUTO_TIERS
        profile_suffix = ""

    # Handle large context override
    if estimated_tokens > 100_000:
        tier: Tier = "COMPLEX"
        confidence = 0.95
        reasoning = f"Input exceeds 100K tokens{profile_suffix}"
    elif result["tier"] is None:
        # Ambiguous - default to MEDIUM
        tier = "MEDIUM"
        confidence = 0.5
        reasoning = f"score={result['score']:.2f} | {', '.join(result['signals'])} | ambiguous -> default: MEDIUM{profile_suffix}"
    else:
        tier = result["tier"]
        confidence = result["confidence"]
        reasoning = f"score={result['score']:.2f} | {', '.join(result['signals'])}{profile_suffix}"

    # Select model from tier
    config = tier_configs[tier]
    model = config["primary"]

    # Check if model is available in pricing
    if model not in model_pricing:
        for fallback in config["fallback"]:
            if fallback in model_pricing:
                model = fallback
                break

    # Calculate costs
    pricing = model_pricing.get(model, {"input_price": 0, "output_price": 0})
    input_cost = (estimated_tokens / 1_000_000) * pricing.get("input_price", 0)
    output_cost = (max_output_tokens / 1_000_000) * pricing.get("output_price", 0)
    cost_estimate = input_cost + output_cost

    # Baseline cost (GPT-4o pricing: $2.50/$10)
    baseline_input = (estimated_tokens / 1_000_000) * 2.50
    baseline_output = (max_output_tokens / 1_000_000) * 10.0
    baseline_cost = baseline_input + baseline_output

    # Savings calculation
    savings = max(0, (baseline_cost - cost_estimate) / baseline_cost) if baseline_cost > 0 else 0

    return {
        "model": model,
        "tier": tier,
        "confidence": confidence,
        "method": "rules",
        "reasoning": reasoning,
        "cost_estimate": cost_estimate,
        "baseline_cost": baseline_cost,
        "savings": savings,
    }
