"""
Cross-language decision-snapshot parity.

``router_core_decisions.snapshot.json`` is a verbatim copy of upstream
``decisions.snapshot.json`` at commit ``5ee7c23`` — 88 complete decisions the
TypeScript engine produced for a frozen corpus (22 prompts x 4 profiles with
rotating tool/vision/structured-output shapes, frozen pricing, frozen clock).
This test recomputes every decision with the Python port and compares field
by field, floats included: same IEEE-754 operations must yield the same
doubles, and reasoning strings must match to the character because hosts
assert on their wording.

When upstream re-syncs, copy the regenerated fixture over and re-run — a
mismatch means the port has drifted, not that the fixture is stale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from blockrun_llm.router_core import (
    DEFAULT_MODEL_CAPABILITIES,
    DEFAULT_ROUTING_CONFIG,
    route,
)

FIXTURE = Path(__file__).with_name("router_core_decisions.snapshot.json")

# Upstream decisions.snapshot.test.ts, transliterated. The pricing hash is the
# JS one (charCodeAt * 31, unsigned 32-bit) so both engines price identically.


def _name_hash(model: str) -> int:
    value = 0
    for char in model:
        value = (value * 31 + ord(char)) & 0xFFFFFFFF
    return value


PRICING = {
    model: {
        "input_price": 0.1 + (_name_hash(model) % 7) * 0.7,
        "output_price": 0.4 + (_name_hash(model) % 5) * 2.1,
    }
    for model in DEFAULT_MODEL_CAPABILITIES
}
PRICING["anthropic/claude-opus-4.7"] = {"input_price": 5.0, "output_price": 25.0}

NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)

PROMPTS = [
    "What is the capital of France?",
    "hi",
    "Explain the difference between TCP and UDP in one paragraph.",
    "Write a Python function that checks if a string is a valid IPv4 address. Include edge cases.",
    "Prove that the sum of two odd integers is even, step by step.",
    "Refactor this React component to use hooks:\n```jsx\nclass Foo extends React.Component { render() { return <div/> } }\n```",
    "Cancel order B-42 and book the 9am flight to SFO.",
    "What's the weather in Tokyo, Paris, and New York?",
    "帮我总结这篇文章的要点，不超过三句话。",
    "设计一个分布式限流器，要求支持滑动窗口和多机房容灾，并给出伪代码。",
    "debug: TypeError: Cannot read properties of undefined (reading 'map') at UserList.render",
    "Summarize the following contract clause and list any obligations: " + "lorem ipsum " * 400,
    "Which of the following is NOT a prime? (a) 17 (b) 21 (c) 23 (d) 29. Answer with the letter only.",
    "Solve for x: 3x^2 - 12x + 9 = 0. Show your work.",
    "Extract all email addresses and phone numbers from this text as JSON: contact bob@x.com or 555-1234",
    "rm -rf the old build directory, then rerun the release pipeline and paste the log tail",
    "Investigate why the checkout page p95 regressed after Tuesday's deploy. Check the CDN config, the API gateway logs, and the database slow query log.",
    "Write a haiku about autumn rain.",
    "Translate 'the quick brown fox jumps over the lazy dog' into German, French, and Japanese.",
    "Plan a 7-day itinerary for Kyoto in November with a daily budget of $150, must include one onsen day and avoid Mondays for museums.",
    "Design the architecture for a multi-tenant SaaS billing system: requirements, data model, service boundaries, failure modes, migration plan from the legacy monolith, and a rollout strategy with feature flags.",
    "Here is our full incident log, produce a postmortem timeline: "
    + "07:14 api-gw 502 spike; 07:16 pod restart loop; " * 1200,
]

SHAPES = [
    {},
    {
        "has_tools": True,
        "requires_tools": True,
        "tool_count": 4,
        "tool_names": ["cancel_order", "book_flight", "search_flights", "get_user"],
    },
    {"has_vision": True},
    {"requires_structured_output": True},
    {
        "has_tools": True,
        "requires_tools": False,
        "tool_count": 12,
        "tool_names": ["read_file", "write_file", "run_shell", "search_code"],
    },
]

PROFILES = [None, "eco", "auto", "premium"]

#: snake_case decision key -> camelCase fixture key, for every pinned field.
KEY_MAP = {
    "model": "model",
    "tier": "tier",
    "confidence": "confidence",
    "method": "method",
    "reasoning": "reasoning",
    "cost_estimate": "costEstimate",
    "baseline_cost": "baselineCost",
    "savings": "savings",
    "agentic_score": "agenticScore",
    "profile": "profile",
    "candidates": "candidates",
    "task_type": "taskType",
    "router_version": "routerVersion",
}


def _candidate_scores(entries):
    return [
        {
            "model": entry["model"],
            "score": entry["score"],
            "quality": entry["quality"],
            "cost": entry["cost"],
            "speed": entry["speed"],
            "reliability": entry["reliability"],
        }
        for entry in entries
    ]


def test_the_python_port_reproduces_every_upstream_decision():
    expected_rows = json.loads(FIXTURE.read_text())
    assert len(expected_rows) == len(PROFILES) * len(PROMPTS)

    mismatches: list[str] = []
    index = 0
    for profile in PROFILES:
        for i, prompt in enumerate(PROMPTS):
            expected = expected_rows[index]
            index += 1
            options = {
                "config": DEFAULT_ROUTING_CONFIG,
                "model_pricing": PRICING,
                "routing_profile": profile,
                "now": NOW,
                **SHAPES[i % len(SHAPES)],
            }
            decision = route(
                prompt,
                "You are a helpful assistant." if i % 3 == 0 else None,
                256 + (i % 4) * 1024,
                options,
            )
            row = f"prompt={i} profile={profile or 'default'}"
            for snake, camel in KEY_MAP.items():
                if decision.get(snake) != expected.get(camel):
                    mismatches.append(
                        f"{row} {camel}: py={decision.get(snake)!r} ts={expected.get(camel)!r}"
                    )
            got_scores = _candidate_scores(decision.get("candidate_scores") or [])
            if got_scores != (expected.get("candidateScores") or []):
                mismatches.append(f"{row} candidateScores differ")

    assert not mismatches, "\n".join(mismatches[:20]) + f"\n({len(mismatches)} total)"
