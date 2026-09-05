import pytest

from blockrun_llm.types import ChatUsage


def test_chat_usage_exposes_reasoning_breakdown_without_changing_totals() -> None:
    usage = ChatUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        completion_tokens_details={"reasoning_tokens": 12},
    )

    assert usage.reasoning_tokens == 12
    assert usage.completion_tokens == 20
    assert usage.model_dump(exclude_none=True)["completion_tokens_details"] == {
        "reasoning_tokens": 12
    }


def test_chat_usage_reasoning_is_optional() -> None:
    usage = ChatUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert usage.reasoning_tokens is None


def test_chat_usage_reads_a_flat_reasoning_extra() -> None:
    """`extra = "allow"` is what lets an upstream shape change reach callers.
    A property of the same name wins over the extra, so the flat payload has
    to be read explicitly or the number silently becomes None."""
    usage = ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, reasoning_tokens=12)

    assert usage.reasoning_tokens == 12
    assert usage.model_dump()["reasoning_tokens"] == 12


def test_nested_detail_wins_over_a_flat_extra() -> None:
    usage = ChatUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        reasoning_tokens=99,
        completion_tokens_details={"reasoning_tokens": 12},
    )

    assert usage.reasoning_tokens == 12


@pytest.mark.parametrize(
    "detail",
    [
        {},
        {"reasoning_tokens": None},
        {"reasoning_tokens": "12"},
        {"reasoning_tokens": 12.5},
        {"reasoning_tokens": -1},
        {"reasoning_tokens": True},
        {"audio_tokens": 0, "accepted_prediction_tokens": 0},
    ],
    ids=["empty", "null", "string", "float", "negative", "bool", "other-keys-only"],
)
def test_unusable_values_read_as_absent_not_as_a_count(detail) -> None:
    """A malformed count must not become one. `True` is an int subclass, so
    without the bool check it would arrive as a token total of 1."""
    usage = ChatUsage(
        prompt_tokens=1, completion_tokens=2, total_tokens=3, completion_tokens_details=detail
    )

    assert usage.reasoning_tokens is None


def test_zero_is_a_real_answer_not_a_missing_one() -> None:
    """The gateway sends reasoning_tokens: 0 on non-reasoning turns — that is a
    measurement, not an absence, and must not collapse to None."""
    usage = ChatUsage(
        prompt_tokens=19,
        completion_tokens=5,
        total_tokens=24,
        prompt_tokens_details={"audio_tokens": 0, "cached_tokens": 0},
        completion_tokens_details={
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    )

    assert usage.reasoning_tokens == 0


def test_reasoning_is_not_added_on_top_of_the_completion_total() -> None:
    """Documented invariant: reasoning tokens are already inside
    completion_tokens. A caller summing both would over-report spend."""
    usage = ChatUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        completion_tokens_details={"reasoning_tokens": 12},
    )

    assert usage.reasoning_tokens <= usage.completion_tokens
    assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens
