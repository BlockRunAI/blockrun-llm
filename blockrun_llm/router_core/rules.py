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
import re
import unicodedata
from dataclasses import dataclass

from ._js import js_regex
from .types import DimensionScore, ScoringConfig, ScoringResult, Tier, TokenCountThresholds

_MULTI_STEP_PATTERNS = [
    js_regex(r"first.*then", ignorecase=True),
    js_regex(r"step \d", ignorecase=True),
    js_regex(r"\d\.\s"),
]
_QUESTION_MARK = js_regex(r"\?")


# ─── Keyword Matching ───
#
# A plain substring test counted `class` in "classic", `story` in "history",
# `edit` in "credit" and `fix` in "prefix", so everyday prose picked up code,
# creative and agentic signals it did not carry. Alphabetic keywords now have
# to start at a word boundary. Short ASCII keywords additionally have to end at
# one, after an optional inflection, graded by how much a stray suffix can
# change the word: a three-letter keyword takes only the English endings that
# cannot spell another word (fix → fixes, never pip → pipe or def → defer); a
# four-to-six-letter keyword also takes the common German and Romance endings,
# because the lists spell those languages in ASCII too (beweis → beweise,
# klasse → klassen); anything longer stays trailing-open like the substring
# test it replaces, since its false-positive surface is small and its
# inflection surface is large. Non-ASCII alphabetic keywords keep a
# trailing-open match because several are deliberate stems (`оптимизаци`,
# `рассуждени`). CJK, Hangul and Arabic keywords keep the substring test: those
# scripts have no word boundaries, or attach particles and articles directly
# to the word.

_UNSEGMENTED_SCRIPT = re.compile(
    "[\u0600-\u06ff\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]"
)
_ASCII_PHRASE = re.compile(r"[a-z0-9' -]+")
_JS_REGEX_SPECIALS = re.compile(r"[.*+?^${}()|[\]\\]")

# A word boundary is a transition out of the segmented scripts. A CJK
# character next to a Latin keyword ("请step by step证明") is a boundary, not
# part of the word: mixed-script prompts carry no spaces between scripts.
#
# Upstream spells the class `[\p{Script=Latin}\p{Script=Cyrillic}
# \p{Script=Greek}\p{N}]`, which the `re` module cannot express. These are its
# code-point ranges (hex, inclusive), enumerated from Node 22's V8 (Unicode
# 16.0) so the boundary does not move with the interpreter's unicodedata.
_SEGMENTED_WORD_RANGES = (
    "30-39 41-5A 61-7A AA B2-B3 B9-BA BC-BE C0-D6 D8-F6 F8-2B8 2E0-2E4 370-373 375-377 "
    "37A-37D 37F 384 386 388-38A 38C 38E-3A1 3A3-3E1 3F0-484 487-52F 660-669 6F0-6F9 "
    "7C0-7C9 966-96F 9E6-9EF 9F4-9F9 A66-A6F AE6-AEF B66-B6F B72-B77 BE6-BF2 C66-C6F "
    "C78-C7E CE6-CEF D58-D5E D66-D78 DE6-DEF E50-E59 ED0-ED9 F20-F33 1040-1049 1090-1099 "
    "1369-137C 16EE-16F0 17E0-17E9 17F0-17F9 1810-1819 1946-194F 19D0-19DA 1A80-1A89 "
    "1A90-1A99 1B50-1B59 1BB0-1BB9 1C40-1C49 1C50-1C59 1C80-1C8A 1D00-1DBF 1E00-1F15 "
    "1F18-1F1D 1F20-1F45 1F48-1F4D 1F50-1F57 1F59 1F5B 1F5D 1F5F-1F7D 1F80-1FB4 1FB6-1FC4 "
    "1FC6-1FD3 1FD6-1FDB 1FDD-1FEF 1FF2-1FF4 1FF6-1FFE 2070-2071 2074-2079 207F-2089 "
    "2090-209C 2126 212A-212B 2132 214E 2150-2189 2460-249B 24EA-24FF 2776-2793 2C60-2C7F "
    "2CFD 2DE0-2DFF 3007 3021-3029 3038-303A 3192-3195 3220-3229 3248-324F 3251-325F "
    "3280-3289 32B1-32BF A620-A629 A640-A69F A6E6-A6EF A722-A787 A78B-A7CD A7D0-A7D1 A7D3 "
    "A7D5-A7DC A7F2-A7FF A830-A835 A8D0-A8D9 A900-A909 A9D0-A9D9 A9F0-A9F9 AA50-AA59 "
    "AB30-AB5A AB5C-AB69 ABF0-ABF9 FB00-FB06 FE2E-FE2F FF10-FF19 FF21-FF3A FF41-FF5A "
    "10107-10133 10140-1018E 101A0 102E1-102FB 10320-10323 10341 1034A 103D1-103D5 "
    "104A0-104A9 10780-10785 10787-107B0 107B2-107BA 10858-1085F 10879-1087F 108A7-108AF "
    "108FB-108FF 10916-1091B 109BC-109BD 109C0-109CF 109D2-109FF 10A40-10A48 10A7D-10A7E "
    "10A9D-10A9F 10AEB-10AEF 10B58-10B5F 10B78-10B7F 10BA9-10BAF 10CFA-10CFF 10D30-10D39 "
    "10D40-10D49 10E60-10E7E 10F1D-10F26 10F51-10F54 10FC5-10FCB 11052-1106F 110F0-110F9 "
    "11136-1113F 111D0-111D9 111E1-111F4 112F0-112F9 11450-11459 114D0-114D9 11650-11659 "
    "116C0-116C9 116D0-116E3 11730-1173B 118E0-118F2 11950-11959 11BF0-11BF9 11C50-11C6C "
    "11D50-11D59 11DA0-11DA9 11F50-11F59 11FC0-11FD4 12400-1246E 16130-16139 16A60-16A69 "
    "16AC0-16AC9 16B50-16B59 16B5B-16B61 16D70-16D79 16E80-16E96 1CCF0-1CCF9 1D200-1D245 "
    "1D2C0-1D2D3 1D2E0-1D2F3 1D360-1D378 1D7CE-1D7FF 1DF00-1DF1E 1DF25-1DF2A 1E030-1E06D "
    "1E08F 1E140-1E149 1E2F0-1E2F9 1E4F0-1E4F9 1E5F1-1E5FA 1E8C7-1E8CF 1E950-1E959 "
    "1EC71-1ECAB 1ECAD-1ECAF 1ECB1-1ECB4 1ED01-1ED2D 1ED2F-1ED3D 1F100-1F10C 1FBF0-1FBF9"
)
_SEGMENTED_WORD_CHAR = "".join(
    (
        f"\\U{int(span, 16):08X}"
        if "-" not in span
        else "\\U{:08X}-\\U{:08X}".format(*(int(end, 16) for end in span.split("-")))
    )
    for span in _SEGMENTED_WORD_RANGES.split()
)
_BOUNDARY_BEFORE = f"(?<![{_SEGMENTED_WORD_CHAR}])"
_BOUNDARY_AFTER = f"(?![{_SEGMENTED_WORD_CHAR}])"
_BOUNDED_LAST_WORD_LENGTH = 6
_SHORT_SUFFIXES = "(?:s|ed|ing)?"
_SIBILANT_PLURAL = "es"
_MID_SUFFIXES = (
    "(?:s|es|ed|d|ing|er|ers|ly|ment|ments|ation|ations|e|en|n|a|as|o|os|em|et|ung|ungen)?"
)
_DROPPED_E_SUFFIXES = "(?:ing|ed|es|er|ers|ation|ations)"
_DOUBLED_CONSONANT_SUFFIXES = "(?:ing|ed|er|ers)"


@dataclass(frozen=True)
class _KeywordMatcher:
    #: Cheap substring pre-check; the regex only runs when this is present.
    needle: str
    pattern: re.Pattern[str] | None


_matcher_cache: dict[str, _KeywordMatcher] = {}


def _escape_reg_exp(value: str) -> str:
    """Port of the upstream ``escapeRegExp`` (escapes only regex metacharacters)."""
    return _JS_REGEX_SPECIALS.sub(lambda match: f"\\{match.group(0)}", value)


def _is_word_char(char: str) -> bool:
    """Port of ``/[\\p{L}\\p{N}]/u``."""
    return unicodedata.category(char)[0] in ("L", "N")


def _keyword_matcher(keyword: str) -> _KeywordMatcher:
    cached = _matcher_cache.get(keyword)
    if cached is not None:
        return cached

    kw = keyword.lower()
    if not kw or _UNSEGMENTED_SCRIPT.search(kw):
        matcher = _KeywordMatcher(needle=kw, pattern=None)
    else:
        lead = _BOUNDARY_BEFORE if _is_word_char(kw[0]) else ""
        escaped = _escape_reg_exp(kw)
        body = escaped
        trail = ""
        needle = kw
        last_word = kw[kw.rfind(" ") + 1 :]
        if (
            _is_word_char(kw[-1])
            and _ASCII_PHRASE.fullmatch(kw)
            and len(last_word) <= _BOUNDED_LAST_WORD_LENGTH
        ):
            # ASCII phrases contain no regex metacharacters, so `escaped` slices
            # align with `kw` slices below.
            forms: list[str] = []
            if kw[-1].isdigit():
                # "step 1" must not absorb "step 10".
                forms.append(escaped)
            elif len(last_word) <= 3:
                forms.append(f"{escaped}{_SHORT_SUFFIXES}")
                if kw.endswith(("x", "s", "z", "ch", "sh")):
                    forms.append(f"{escaped}{_SIBILANT_PLURAL}")
            else:
                forms.append(f"{escaped}{_MID_SUFFIXES}")
                if kw.endswith("e"):
                    forms.append(f"{escaped[:-1]}{_DROPPED_E_SUFFIXES}")
                    needle = kw[:-1]
                elif kw.endswith("y"):
                    forms.append(f"{escaped[:-1]}(?:ies|ied)")
                    needle = kw[:-1]
                elif len(kw) >= 2 and kw[-2] in "aeiou" and kw[-1] in "bdgmnprt":
                    # debug → debugging, plan → planned
                    forms.append(f"{escaped}{kw[-1]}{_DOUBLED_CONSONANT_SUFFIXES}")
            body = f"(?:{'|'.join(forms)})"
            trail = _BOUNDARY_AFTER
        matcher = _KeywordMatcher(needle=needle, pattern=re.compile(f"{lead}{body}{trail}"))
    _matcher_cache[keyword] = matcher
    return matcher


def _matches_keyword(text: str, keyword: str) -> bool:
    """Whether lower-cased ``text`` contains ``keyword`` as a word (or as a substring for CJK)."""
    matcher = _keyword_matcher(keyword)
    if matcher.needle not in text:
        return False
    return True if matcher.pattern is None else matcher.pattern.search(text) is not None


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
    matches = [keyword for keyword in keywords if _matches_keyword(text, keyword)]
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
        if _matches_keyword(text, keyword):
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
        keyword for keyword in config["reasoning_keywords"] if _matches_keyword(user_text, keyword)
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
