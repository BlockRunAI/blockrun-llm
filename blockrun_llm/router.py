"""
Smart Router for BlockRun LLM SDK

Port of ClawRouter's 14-dimension rule-based scoring algorithm.
Routes requests to the cheapest capable model in <1ms, 100% local.

Usage:
    from blockrun_llm import LLMClient

    client = LLMClient()
    result = client.smart_chat("What is 2+2?")
    print(result["response"])  # '4'
    print(result["model"])     # 'moonshot/kimi-k2.6' (AUTO Simple picks here)
    print(f"Saved {result['routing']['savings'] * 100:.0f}%")
"""

import re
import math
from typing import Dict, List, Optional, Literal, TypedDict


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
    fallbacks: List[str]  # remaining models in tier order, for runtime fallback


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
    "function",
    "class",
    "import",
    "def",
    "SELECT",
    "async",
    "await",
    "const",
    "let",
    "var",
    "return",
    "```",
    "函数",
    "类",
    "导入",
    "定义",
    "查询",
    "异步",
    "等待",
    "常量",
    "变量",
    "返回",
    "関数",
    "クラス",
    "インポート",
    "非同期",
    "定数",
    "変数",
    "функция",
    "класс",
    "импорт",
    "определ",
    "запрос",
    "асинхронный",
]

REASONING_KEYWORDS = [
    "prove",
    "theorem",
    "derive",
    "step by step",
    "chain of thought",
    "formally",
    "mathematical",
    "proof",
    "logically",
    "证明",
    "定理",
    "推导",
    "逐步",
    "思维链",
    "形式化",
    "数学",
    "逻辑",
    "доказать",
    "теорема",
    "вывести",
    "шаг за шагом",
    "логически",
]

SIMPLE_KEYWORDS = [
    "what is",
    "define",
    "translate",
    "hello",
    "yes or no",
    "capital of",
    "how old",
    "who is",
    "when was",
    "什么是",
    "定义",
    "翻译",
    "你好",
    "是否",
    "首都",
    "что такое",
    "определение",
    "перевести",
    "привет",
]

TECHNICAL_KEYWORDS = [
    "algorithm",
    "optimize",
    "architecture",
    "distributed",
    "kubernetes",
    "microservice",
    "database",
    "infrastructure",
    "算法",
    "优化",
    "架构",
    "分布式",
    "微服务",
    "数据库",
]

CREATIVE_KEYWORDS = [
    "story",
    "poem",
    "compose",
    "brainstorm",
    "creative",
    "imagine",
    "write a",
    "故事",
    "诗",
    "创作",
    "头脑风暴",
    "创意",
    "想象",
]

AGENTIC_KEYWORDS = [
    "read file",
    "read the file",
    "look at",
    "check the",
    "open the",
    "edit",
    "modify",
    "update the",
    "change the",
    "write to",
    "create file",
    "execute",
    "deploy",
    "install",
    "npm",
    "pip",
    "compile",
    "after that",
    "and also",
    "once done",
    "step 1",
    "step 2",
    "fix",
    "debug",
    "until it works",
    "keep trying",
    "iterate",
    "make sure",
    "verify",
    "confirm",
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
        # moonshot/kimi-k2.6 is Moonshot's flagship (256K context, vision +
        # reasoning_content). kimi-k2.5 is hidden in the catalog (superseded)
        # so it no longer appears in /v1/models pricing — routing here would
        # silently fall back. k2.5 retained as fallback for clients that
        # explicitly pricing-pin to it.
        "primary": "moonshot/kimi-k2.6",
        "fallback": [
            "moonshot/kimi-k2.5",
            "google/gemini-2.5-flash-lite",
            "deepseek/deepseek-chat",
            "nvidia/llama-4-maverick",
        ],
    },
    "MEDIUM": {
        "primary": "google/gemini-2.5-flash",
        "fallback": [
            "deepseek/deepseek-chat",
            "nvidia/llama-4-maverick",
        ],
    },
    "COMPLEX": {
        "primary": "google/gemini-3.1-pro",
        "fallback": [
            "google/gemini-3.5-flash",
            "google/gemini-3-flash-preview",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-chat",
        ],
    },
    "REASONING": {
        # deepseek/deepseek-reasoner is V4 Flash thinking ($0.20/$0.40, 1M ctx)
        # — the cheapest production-grade reasoner. deepseek/deepseek-v4-pro
        # ($0.435/$0.87 — the 75% launch promo became DeepSeek's permanent
        # list price after 2026-05-31; MMLU-Pro 87.5, GPQA 90.1, SWE-bench
        # 80.6) is the strongest open-weight reasoner we serve; first
        # fallback when V4 Flash thinking is unavailable.
        "primary": "deepseek/deepseek-reasoner",
        "fallback": ["deepseek/deepseek-v4-pro", "openai/o3", "openai/o3-mini"],
    },
}

ECO_TIERS: Dict[Tier, TierConfig] = {
    "SIMPLE": {
        # See AUTO_TIERS note: kimi-k2.6 is the catalog flagship. kimi-k2.5
        # is hidden so the SDK no longer sees its pricing.
        "primary": "moonshot/kimi-k2.6",
        "fallback": ["moonshot/kimi-k2.5", "deepseek/deepseek-chat", "nvidia/llama-4-maverick"],
    },
    "MEDIUM": {
        # deepseek/deepseek-chat is V4 Flash non-thinking ($0.20/$0.40, 1M ctx
        # — DeepSeek upstream now serves the legacy alias as V4 Flash chat).
        "primary": "deepseek/deepseek-chat",
        "fallback": ["google/gemini-2.5-flash-lite", "google/gemini-2.5-flash"],
    },
    "COMPLEX": {
        # 2026-06-06: the whole GLM flat-rate promo family ended (glm-5 now
        # $0.60/$1.92 per-token), so no GLM earns a cheap-fallback slot here
        # anymore — the per-token chain below already covers every price
        # point (v4-pro $0.435/$0.87 is both cheaper and stronger).
        "primary": "google/gemini-2.5-pro",
        "fallback": [
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-chat",
            "google/gemini-2.5-flash",
        ],
    },
    "REASONING": {
        # V4 Flash thinking ($0.20/$0.40) preferred over V4 Pro ($0.435/$0.87)
        # in eco mode — V4 Pro retained as fallback for harder reasoning.
        "primary": "deepseek/deepseek-reasoner",
        "fallback": ["deepseek/deepseek-v4-pro", "openai/o3-mini"],
    },
}

PREMIUM_TIERS: Dict[Tier, TierConfig] = {
    "SIMPLE": {
        "primary": "google/gemini-2.5-flash",
        "fallback": ["openai/gpt-5.4-nano", "anthropic/claude-haiku-4.5"],
    },
    "MEDIUM": {
        "primary": "openai/gpt-5.5",
        "fallback": ["openai/gpt-5.4", "google/gemini-2.5-pro", "anthropic/claude-sonnet-4.6"],
    },
    "COMPLEX": {
        # claude-opus-4.8 (1M context, agentic coding + adaptive thinking) is
        # Anthropic's strongest current Claude. opus-4.7/4.5 retained as
        # fallbacks for clients pricing-pinned to them.
        "primary": "anthropic/claude-opus-4.8",
        "fallback": [
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.5",
            "openai/gpt-5.2-pro",
            "google/gemini-3.1-pro",
            "openai/gpt-5.2",
        ],
    },
    "REASONING": {
        "primary": "openai/o3",
        "fallback": ["openai/o1", "anthropic/claude-opus-4.8"],
    },
}

FREE_TIERS: Dict[Tier, TierConfig] = {
    # NVIDIA free tier refresh 2026-04-28: retired nvidia/gpt-oss-120b and
    # nvidia/gpt-oss-20b (NVIDIA's free build.nvidia.com tier reserves the
    # right to use prompts/outputs for service improvement, conflicting with
    # our data-privacy policy). Added nvidia/deepseek-v4-pro and
    # nvidia/deepseek-v4-flash (1M context); v4-pro currently hidden because
    # NVIDIA's NIM deployment for it is hung — backend MODEL_REDIRECTS sends
    # callers to v4-flash transparently. nvidia/deepseek-v3.2 is also hidden
    # for the same hang. Primaries here are pinned to visible models so the
    # Python pricing dict (built from /v1/models) can resolve them.
    #
    # 2026-06-07 sweep (live-probed every visible free model):
    # - nvidia/qwen3-next-80b-a3b-thinking hit NVIDIA END-OF-LIFE 2026-05-21
    #   (HTTP 410 Gone; backend marks it hidden + unavailable and redirects to
    #   llama-4-maverick). Dropped as COMPLEX/REASONING primary.
    # - nvidia/mistral-small-4-119b is timing out upstream (3/3 probes >60s).
    #   Dropped as SIMPLE primary and from all fallback chains.
    # - nvidia/deepseek-v4-flash RECOVERED from the 05-09 NIM regression
    #   (896ms probe) — reinstated as SIMPLE primary (1M context, fastest
    #   capable free chat).
    # - nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (681ms, 256K ctx,
    #   explicit reasoning + vision) takes the REASONING primary.
    # - nvidia/qwen3-coder-480b (871ms, 480B MoE) takes the COMPLEX primary.
    "SIMPLE": {
        "primary": "nvidia/deepseek-v4-flash",
        "fallback": ["nvidia/llama-4-maverick"],
    },
    "MEDIUM": {
        "primary": "nvidia/llama-4-maverick",
        "fallback": ["nvidia/qwen3-coder-480b", "nvidia/deepseek-v4-flash"],
    },
    "COMPLEX": {
        "primary": "nvidia/qwen3-coder-480b",
        "fallback": ["nvidia/llama-4-maverick", "nvidia/deepseek-v4-flash"],
    },
    "REASONING": {
        "primary": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "fallback": ["nvidia/llama-4-maverick", "nvidia/deepseek-v4-flash"],
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
    weighted_score = sum(scores.get(dim, 0) * weight for dim, weight in DIMENSION_WEIGHTS.items())

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
            TIER_BOUNDARIES["medium_complex"] - weighted_score,
        )
    elif weighted_score < TIER_BOUNDARIES["complex_reasoning"]:
        tier = "COMPLEX"
        distance = min(
            weighted_score - TIER_BOUNDARIES["medium_complex"],
            TIER_BOUNDARIES["complex_reasoning"] - weighted_score,
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

    # Build runtime fallback chain — every model in the tier other than the
    # chosen one, in tier-defined order, filtered to those with known pricing.
    # chat_completion() walks this list on timeout / 5xx so a hung upstream
    # does not break smart_chat.
    ordered = [config["primary"], *config["fallback"]]
    fallbacks = [m for m in ordered if m != model and m in model_pricing]

    # Calculate costs. Flat-billed models (ZAI GLM-5 family) charge a fixed
    # USD/call regardless of token count; honor that instead of computing
    # per-token cost as zero.
    pricing = model_pricing.get(model, {"input_price": 0, "output_price": 0, "flat_price": 0})
    flat_price = pricing.get("flat_price", 0)
    if flat_price:
        cost_estimate = float(flat_price)
    else:
        input_cost = (estimated_tokens / 1_000_000) * pricing.get("input_price", 0)
        output_cost = (max_output_tokens / 1_000_000) * pricing.get("output_price", 0)
        cost_estimate = input_cost + output_cost

    # Baseline cost (GPT-5.5 pricing: $5.00/$30)
    baseline_input = (estimated_tokens / 1_000_000) * 5.00
    baseline_output = (max_output_tokens / 1_000_000) * 30.0
    baseline_cost = baseline_input + baseline_output

    # Savings calculation
    savings = max(0, (baseline_cost - cost_estimate) / baseline_cost) if baseline_cost > 0 else 0

    return {
        "model": model,
        "fallbacks": fallbacks,
        "tier": tier,
        "confidence": confidence,
        "method": "rules",
        "reasoning": reasoning,
        "cost_estimate": cost_estimate,
        "baseline_cost": baseline_cost,
        "savings": savings,
    }
