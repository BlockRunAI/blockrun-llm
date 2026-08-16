"""
Rule-Based Classifier (v2 — Weighted Scoring)

Python port of ``@blockrun/router-core`` ``rules.ts``.

Scores a request across 15 weighted dimensions and maps the aggregate score to
a tier using configurable boundaries. Confidence is calibrated via sigmoid —
low confidence triggers the fallback classifier.

Handles 70-80% of requests in < 1ms with zero cost.
"""

from __future__ import annotations

import math

from ._js import js_regex
from .types import DimensionScore, ScoringConfig, ScoringResult, Tier, TokenCountThresholds

_MULTI_STEP_PATTERNS = [
    js_regex(r"first.*then", ignorecase=True),
    js_regex(r"step \d", ignorecase=True),
    js_regex(r"\d\.\s"),
]
_QUESTION_MARK = js_regex(r"\?")


# ─── Dimension Scorers ───
# Each returns a score in [-1, 1] and an optional signal string.


def _score_token_count(
    estimated_tokens: int,
    thresholds: TokenCountThresholds,
) -> DimensionScore:
    if estimated_tokens < thresholds["simple"]:
        return {
            "name": "tokenCount",
            "score": -1.0,
            "signal": f"short ({estimated_tokens} tokens)",
        }
    if estimated_tokens > thresholds["complex"]:
        return {"name": "tokenCount", "score": 1.0, "signal": f"long ({estimated_tokens} tokens)"}
    return {"name": "tokenCount", "score": 0, "signal": None}


def _score_keyword_match(
    text: str,
    keywords: list[str],
    name: str,
    signal_label: str,
    thresholds: tuple[int, int],
    scores: tuple[float, float, float],
) -> DimensionScore:
    """``thresholds`` is ``(low, high)``; ``scores`` is ``(none, low, high)``."""
    low_threshold, high_threshold = thresholds
    none_score, low_score, high_score = scores
    matches = [keyword for keyword in keywords if keyword.lower() in text]
    if len(matches) >= high_threshold:
        return {
            "name": name,
            "score": high_score,
            "signal": f"{signal_label} ({', '.join(matches[:3])})",
        }
    if len(matches) >= low_threshold:
        return {
            "name": name,
            "score": low_score,
            "signal": f"{signal_label} ({', '.join(matches[:3])})",
        }
    return {"name": name, "score": none_score, "signal": None}


def _score_multi_step(text: str) -> DimensionScore:
    if any(pattern.search(text) for pattern in _MULTI_STEP_PATTERNS):
        return {"name": "multiStepPatterns", "score": 0.5, "signal": "multi-step"}
    return {"name": "multiStepPatterns", "score": 0, "signal": None}


def _score_question_complexity(prompt: str) -> DimensionScore:
    count = len(_QUESTION_MARK.findall(prompt))
    if count > 3:
        return {"name": "questionComplexity", "score": 0.5, "signal": f"{count} questions"}
    return {"name": "questionComplexity", "score": 0, "signal": None}


def _score_agentic_task(text: str, keywords: list[str]) -> tuple[DimensionScore, float]:
    """Score agentic task indicators.

    Returns ``(dimension, agentic_score)`` where the 0-1 agentic score is based
    on keyword matches: 4+ matches = 1.0 (high agentic), 3 = 0.6 (moderate,
    triggers auto-agentic mode), 1-2 = 0.2 (low). Thresholds were raised
    because common keywords were pruned from the list.
    """
    match_count = 0
    signals: list[str] = []

    for keyword in keywords:
        if keyword.lower() in text:
            match_count += 1
            if len(signals) < 3:
                signals.append(keyword)

    if match_count >= 4:
        return (
            {"name": "agenticTask", "score": 1.0, "signal": f"agentic ({', '.join(signals)})"},
            1.0,
        )
    if match_count >= 3:
        return (
            {"name": "agenticTask", "score": 0.6, "signal": f"agentic ({', '.join(signals)})"},
            0.6,
        )
    if match_count >= 1:
        return (
            {
                "name": "agenticTask",
                "score": 0.2,
                "signal": f"agentic-light ({', '.join(signals)})",
            },
            0.2,
        )

    return ({"name": "agenticTask", "score": 0, "signal": None}, 0.0)


# ─── Main Classifier ───


def classify_by_rules(
    prompt: str,
    system_prompt: str | None,
    estimated_tokens: int,
    config: ScoringConfig,
) -> ScoringResult:
    """Classify a request into a tier with calibrated confidence."""
    # Score against user prompt only — system prompts contain boilerplate
    # keywords (tool definitions, skill descriptions, behavioral rules) that
    # dominate scoring and make every request score identically.
    user_text = prompt.lower()

    # Score the base dimensions against user text only; the agentic dimension is
    # appended below, so the scored total is one more than this list.
    dimensions: list[DimensionScore] = [
        # Token count uses total estimated tokens (system + user) — context size
        # matters for model selection.
        _score_token_count(estimated_tokens, config["token_count_thresholds"]),
        _score_keyword_match(
            user_text, config["code_keywords"], "codePresence", "code", (1, 2), (0, 0.5, 1.0)
        ),
        _score_keyword_match(
            user_text,
            config["reasoning_keywords"],
            "reasoningMarkers",
            "reasoning",
            (1, 2),
            (0, 0.7, 1.0),
        ),
        _score_keyword_match(
            user_text,
            config["technical_keywords"],
            "technicalTerms",
            "technical",
            (2, 4),
            (0, 0.5, 1.0),
        ),
        _score_keyword_match(
            user_text,
            config["creative_keywords"],
            "creativeMarkers",
            "creative",
            (1, 2),
            (0, 0.5, 0.7),
        ),
        _score_keyword_match(
            user_text,
            config["simple_keywords"],
            "simpleIndicators",
            "simple",
            (1, 2),
            (0, -1.0, -1.0),
        ),
        _score_multi_step(user_text),
        _score_question_complexity(prompt),
        # 6 new dimensions
        _score_keyword_match(
            user_text,
            config["imperative_verbs"],
            "imperativeVerbs",
            "imperative",
            (1, 2),
            (0, 0.3, 0.5),
        ),
        _score_keyword_match(
            user_text,
            config["constraint_indicators"],
            "constraintCount",
            "constraints",
            (1, 3),
            (0, 0.3, 0.7),
        ),
        _score_keyword_match(
            user_text,
            config["output_format_keywords"],
            "outputFormat",
            "format",
            (1, 2),
            (0, 0.4, 0.7),
        ),
        _score_keyword_match(
            user_text,
            config["reference_keywords"],
            "referenceComplexity",
            "references",
            (1, 2),
            (0, 0.3, 0.5),
        ),
        _score_keyword_match(
            user_text,
            config["negation_keywords"],
            "negationComplexity",
            "negation",
            (2, 3),
            (0, 0.3, 0.5),
        ),
        _score_keyword_match(
            user_text,
            config["domain_specific_keywords"],
            "domainSpecificity",
            "domain-specific",
            (1, 2),
            (0, 0.5, 0.8),
        ),
    ]

    # Score agentic task indicators — user prompt only. The system prompt
    # describes assistant behavior, not the user's intent: a coding assistant
    # system prompt with "edit files" / "fix bugs" should NOT force every
    # request into agentic mode.
    agentic_dimension, agentic_score = _score_agentic_task(
        user_text, config["agentic_task_keywords"]
    )
    dimensions.append(agentic_dimension)

    signals = [dimension["signal"] for dimension in dimensions if dimension["signal"] is not None]

    weights = config["dimension_weights"]
    weighted_score = sum(
        dimension["score"] * weights.get(dimension["name"], 0) for dimension in dimensions
    )

    # Count reasoning markers for override — only the USER prompt, so a system
    # prompt saying "step by step" cannot force REASONING for simple queries.
    reasoning_matches = [
        keyword for keyword in config["reasoning_keywords"] if keyword.lower() in user_text
    ]

    # Direct reasoning override: 2+ reasoning markers = high confidence REASONING
    if len(reasoning_matches) >= 2:
        confidence = _calibrate_confidence(
            max(weighted_score, 0.3),  # ensure positive for confidence calc
            config["confidence_steepness"],
        )
        return {
            "score": weighted_score,
            "tier": "REASONING",
            "confidence": max(confidence, 0.85),
            "signals": signals,
            "agentic_score": agentic_score,
            "dimensions": dimensions,
        }

    # Map weighted score to tier using boundaries
    boundaries = config["tier_boundaries"]
    simple_medium = boundaries["simple_medium"]
    medium_complex = boundaries["medium_complex"]
    complex_reasoning = boundaries["complex_reasoning"]
    tier: Tier
    if weighted_score < simple_medium:
        tier = "SIMPLE"
        distance_from_boundary = simple_medium - weighted_score
    elif weighted_score < medium_complex:
        tier = "MEDIUM"
        distance_from_boundary = min(
            weighted_score - simple_medium, medium_complex - weighted_score
        )
    elif weighted_score < complex_reasoning:
        tier = "COMPLEX"
        distance_from_boundary = min(
            weighted_score - medium_complex, complex_reasoning - weighted_score
        )
    else:
        tier = "REASONING"
        distance_from_boundary = weighted_score - complex_reasoning

    # Calibrate confidence via sigmoid of distance from nearest boundary
    confidence = _calibrate_confidence(distance_from_boundary, config["confidence_steepness"])

    # If confidence is below threshold → ambiguous
    if confidence < config["confidence_threshold"]:
        return {
            "score": weighted_score,
            "tier": None,
            "confidence": confidence,
            "signals": signals,
            "agentic_score": agentic_score,
            "dimensions": dimensions,
        }

    return {
        "score": weighted_score,
        "tier": tier,
        "confidence": confidence,
        "signals": signals,
        "agentic_score": agentic_score,
        "dimensions": dimensions,
    }


def _calibrate_confidence(distance: float, steepness: float) -> float:
    """Sigmoid confidence calibration onto the [0.5, 1.0] range."""
    try:
        return 1 / (1 + math.exp(-steepness * distance))
    except OverflowError:
        # JS evaluates exp() to Infinity here and collapses to 0; Python raises.
        return 0.0
