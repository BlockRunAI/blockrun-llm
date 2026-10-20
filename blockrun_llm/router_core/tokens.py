"""
Request-size helpers shared by the rules and portfolio strategies.

Python port of ``@blockrun/router-core`` ``tokens.ts``.

The router never tokenizes — it is on the hot path and model-agnostic — so
sizes are estimates. The estimate only has to be honest enough for the
decisions that consume it: the long-context task type, the COMPLEX override
and the per-model capacity filter.
"""

from __future__ import annotations

import math
import re

# Han, kana, Hangul, CJK punctuation and fullwidth forms. Every production
# tokenizer spends roughly one token per character here, so the Latin
# four-characters-per-token rule under-counts these scripts about fourfold: a
# 200k-character Chinese contract would read as 50k tokens, slip under the 80k
# long-context boundary and pass the capacity filter for models that cannot
# hold it.
_CJK_CHARS = re.compile(
    "[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]"
)


def estimate_tokens(text: str) -> int:
    """Approximate token count: one per CJK character, one per four of anything else."""
    if not text:
        return 0
    cjk = len(_CJK_CHARS.findall(text))
    return cjk + math.ceil((len(text) - cjk) / 4)


def scan_limit(prompt_truncation_chars: int) -> int:
    """The scan window the feature regexes read, bounded so a pasted document cannot stall routing."""
    return max(1, min(8_000, prompt_truncation_chars))


def sample_for_scan(value: str, limit: int) -> str:
    """Sample both ends of a long value for the feature regexes.

    Feature regexes need request shape and intent, not the entire document.
    Sampling both ends keeps the task instruction of a long pasted artifact at
    either boundary, while the full length still drives capacity decisions.
    """
    if len(value) <= limit:
        return value
    prefix_length = math.ceil(limit / 2)
    suffix_length = limit - prefix_length
    # JS ``slice(-0)`` is the whole string, not an empty one.
    suffix = value[-suffix_length:] if suffix_length else value
    return f"{value[:prefix_length]}\n{suffix}"
