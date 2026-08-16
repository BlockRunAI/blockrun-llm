"""
Small JavaScript-semantics helpers used by the Router Core port.

The router is a line-by-line port of ``@blockrun/router-core``. A handful of
JS behaviours differ from their obvious Python equivalents in ways that change
routing output, so they are isolated here instead of being approximated at
each call site:

* ``Number.prototype.toFixed`` rounds half away from zero on the exact binary
  value; Python's format spec rounds half to even.
* ``Date.parse`` accepts a bare ``YYYY-MM-DD`` (UTC midnight) and a trailing
  ``Z``; ``datetime.fromisoformat`` before 3.11 accepts neither combination.
* Template literals stringify booleans as ``true`` / ``false``, and the
  reasoning strings the router emits are asserted on by hosts and tests.

Ported regexes are compiled with ``re.ASCII`` so ``\\b``, ``\\w``, ``\\d`` and
``\\s`` keep JavaScript's ASCII-only meaning. Without it a pattern like
``\\b(?:urgent|fast)\\b`` silently stops matching inside CJK text, because
Python treats the surrounding Han characters as word characters while
JavaScript does not.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal

#: Flag set applied to every ported regex (see module docstring).
JS_FLAGS = re.ASCII
JS_FLAGS_I = re.ASCII | re.IGNORECASE


def js_regex(pattern: str, *, ignorecase: bool = False, multiline: bool = False) -> re.Pattern[str]:
    """Compile ``pattern`` with JavaScript-compatible flag semantics."""
    flags = JS_FLAGS_I if ignorecase else JS_FLAGS
    if multiline:
        flags |= re.MULTILINE
    return re.compile(pattern, flags)


def to_fixed(value: float, digits: int) -> str:
    """Port of ``Number.prototype.toFixed`` (round half away from zero)."""
    if not math.isfinite(value):  # NaN / Infinity, which toFixed passes through
        return str(value)
    quantum = Decimal(1).scaleb(-digits)
    # toFixed resolves a tie to the larger integer, which is away from zero for
    # positives and toward zero for negatives.
    rounding = ROUND_HALF_UP if value >= 0 else ROUND_HALF_DOWN
    return str(Decimal(value).quantize(quantum, rounding=rounding))


def js_bool(value: bool) -> str:
    """Port of JS template-literal boolean stringification."""
    return "true" if value else "false"


def parse_date(value: str) -> datetime | None:
    """Port of ``Date.parse`` for the ISO forms the router config uses.

    Returns an aware UTC datetime, or ``None`` when the value is unparseable
    (``Date.parse`` yields ``NaN``, which the portfolio scorer treats as "no
    observation" rather than propagating a NaN score).
    """
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def as_utc(value: object | None) -> datetime:
    """Normalize a caller-supplied ``now`` to an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)
