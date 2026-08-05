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
