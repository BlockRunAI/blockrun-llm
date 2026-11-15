"""Decisions — typed judgments, OpenAI Decisions-compatible (``POST /v1/decisions``).

The caller sends evidence (``input``) and questions with a fixed answer space
(``predicate``, ``choice`` or ``score``) and gets one typed answer per question
back instead of text. Served on every rail: an account API key on
api.blockrun.ai, and x402 ``exact`` on Base, Arc and Solana. Only input tokens
are billed.

The gateway owns the limits (64 questions, 255 choices or levels, 120,000
characters of text, 8 inline ``data:image/*`` images, no hosted image URLs)
and refuses an over-limit request with a 400 before any payment. This module
only checks the shape a caller can still get wrong, so a malformed request
fails before anything is sent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The model OpenAI serves Decisions from; the gateway also accepts
#: ``openai/gpt-6-luna`` and ``openai/gpt-6-luna-decisions``.
DECISIONS_DEFAULT_MODEL = "gpt-6-luna"

_QUESTION_TYPES = ("predicate", "choice", "score")


def build_decisions_body(
    input: str | Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The request body for ``POST /v1/decisions``, after a shape check.

    Raises:
        ValueError: on a malformed request. Nothing has been sent.
    """
    empty = (
        len(input) == 0
        if isinstance(input, str)
        else not isinstance(input, Sequence) or len(input) == 0
    )
    if empty:
        raise ValueError(
            "decisions: input must be a non-empty string or a non-empty list of messages"
        )
    if not isinstance(questions, Sequence) or isinstance(questions, str) or len(questions) == 0:
        raise ValueError("decisions: questions must be a non-empty list")
    names: set[str] = set()
    for i, q in enumerate(questions):
        if not isinstance(q, Mapping):
            raise ValueError(f"decisions: questions[{i}] must be a dict")
        if q.get("type") not in _QUESTION_TYPES:
            raise ValueError(
                f'decisions: questions[{i}].type must be "predicate", "choice" or "score"'
            )
        name = q.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"decisions: questions[{i}].name must be a non-empty string")
        instructions = q.get("instructions")
        if not isinstance(instructions, str) or not instructions:
            raise ValueError(f"decisions: questions[{i}].instructions must be a non-empty string")
        if name in names:
            raise ValueError(
                f'decisions: questions[{i}].name "{name}" is used twice; answers are keyed by name'
            )
        names.add(name)
        if q["type"] == "choice" and not q.get("choices"):
            raise ValueError(f"decisions: questions[{i}] (choice) needs a non-empty choices list")
        if q["type"] == "score" and not q.get("levels"):
            raise ValueError(
                f"decisions: questions[{i}] (score) needs a non-empty levels list, lowest first"
            )
    body: dict[str, Any] = {
        "model": model or DECISIONS_DEFAULT_MODEL,
        "input": input if isinstance(input, str) else [dict(m) for m in input],
        "questions": [dict(q) for q in questions],
    }
    if metadata is not None:
        body["metadata"] = dict(metadata)
    return body
