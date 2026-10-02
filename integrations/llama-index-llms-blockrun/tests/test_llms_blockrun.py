"""Unit tests for the BlockRun LlamaIndex LLM.

No network and no wallet: the SDK clients are replaced with fakes that record
what they were asked and answer with real blockrun_llm response types, so the
conversion code is exercised against the shapes the SDK actually returns.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.blockrun import BlockRun

import blockrun_llm
from blockrun_llm.types import ChatCompletionChunk
from blockrun_llm.types import ChatResponse as SdkChatResponse

MODEL = "openai/gpt-5.5"


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


def sdk_response(
    content: str | None = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str | None = None,
    cost_usd: float | None = 0.0021,
    citations: list[str] | None = None,
) -> SdkChatResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return SdkChatResponse.model_validate(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "citations": citations,
            "cost_usd": cost_usd,
        }
    )


def chunk(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str | None = None,
    usage: dict[str, int] | None = None,
    no_choices: bool = False,
) -> ChatCompletionChunk:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": MODEL,
            "choices": [] if no_choices else [{"index": 0, "delta": delta}],
            "usage": usage,
        }
    )


def tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeClient:
    def __init__(
        self,
        response: SdkChatResponse | None = None,
        chunks: list[ChatCompletionChunk] | None = None,
        models: list[dict[str, Any]] | None = None,
        models_error: Exception | None = None,
    ) -> None:
        self.response = response or sdk_response()
        self.chunks = chunks or []
        self.models = models or []
        self.models_error = models_error
        self.calls: list[dict[str, Any]] = []
        self.list_models_calls = 0

    def chat_completion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return self.response

    def chat_completion_stream(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.calls.append({"model": model, "messages": messages, "stream": True, **kwargs})
        yield from self.chunks

    def list_models(self) -> list[dict[str, Any]]:
        self.list_models_calls += 1
        if self.models_error is not None:
            raise self.models_error
        return self.models

    def get_wallet_address(self) -> str:
        return "0xabc"


class FakeAsyncClient(FakeClient):
    async def chat_completion(  # type: ignore[override]
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return self.response

    async def chat_completion_stream(  # type: ignore[override]
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.calls.append({"model": model, "messages": messages, "stream": True, **kwargs})
        for c in self.chunks:
            yield c


def make_llm(client: FakeClient | None = None, **kwargs: Any) -> tuple[BlockRun, FakeClient]:
    llm = BlockRun(model=MODEL, **kwargs)
    client = client or FakeClient()
    llm._client = client
    return llm, client


def make_async_llm(client: FakeAsyncClient, **kwargs: Any) -> BlockRun:
    llm = BlockRun(model=MODEL, **kwargs)
    # _get_aclient builds a new client whenever the running loop changes, so
    # pin the fake to whichever loop the test runs on.
    llm._get_aclient = lambda: client  # type: ignore[method-assign]
    return llm


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestConstruction:
    def test_is_a_function_calling_llm(self) -> None:
        assert issubclass(BlockRun, FunctionCallingLLM)
        assert BlockRun.class_name() == "BlockRun_LLM"

    def test_defaults_to_base(self) -> None:
        assert BlockRun(model=MODEL).chain == "base"

    def test_rejects_unknown_chain(self) -> None:
        with pytest.raises(ValueError, match="'base' or 'solana'"):
            BlockRun(model=MODEL, chain="ethereum")  # type: ignore[arg-type]

    def test_private_key_is_never_serialized(self) -> None:
        secret = "0x" + "ab" * 32
        llm = BlockRun(model=MODEL, private_key=secret)
        assert secret not in json.dumps(llm.to_dict())
        assert secret not in llm.model_dump_json()
        assert secret not in repr(llm)

    def test_solana_refuses_unsupported_additional_kwargs_at_construction(self) -> None:
        with pytest.raises(ValueError, match="presence_penalty"):
            BlockRun(model=MODEL, chain="solana", additional_kwargs={"presence_penalty": 1})


class TestClientSelection:
    @pytest.fixture
    def recorded(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        def recorder(name: str) -> type:
            class Recorder:
                def __init__(self, **kwargs: Any) -> None:
                    seen[name] = kwargs

            return Recorder

        for name in ("LLMClient", "AsyncLLMClient", "SolanaLLMClient", "AsyncSolanaLLMClient"):
            monkeypatch.setattr(blockrun_llm, name, recorder(name))
        return seen

    def test_base_uses_the_evm_client(self, recorded: dict[str, Any]) -> None:
        BlockRun(model=MODEL, private_key="k")._get_client()
        assert recorded == {"LLMClient": {"private_key": "k"}}

    def test_solana_uses_the_solana_client(self, recorded: dict[str, Any]) -> None:
        BlockRun(model=MODEL, chain="solana", private_key="k")._get_client()
        assert recorded == {"SolanaLLMClient": {"private_key": "k"}}

    def test_unset_options_are_not_passed(self, recorded: dict[str, Any]) -> None:
        # SolanaLLMClient's api_url default is a URL string; a None passed
        # through would replace the gateway it talks to.
        BlockRun(model=MODEL, chain="solana")._get_client()
        assert recorded["SolanaLLMClient"] == {"private_key": None}

    def test_set_options_are_passed(self, recorded: dict[str, Any]) -> None:
        BlockRun(
            model=MODEL,
            api_url="https://testnet.blockrun.ai/api",
            timeout=30,
            max_cost_per_call=0.05,
        )._get_client()
        assert recorded["LLMClient"] == {
            "private_key": None,
            "api_url": "https://testnet.blockrun.ai/api",
            "timeout": 30,
            "max_cost_per_call": 0.05,
        }

    def test_client_is_reused(self, recorded: dict[str, Any]) -> None:
        llm = BlockRun(model=MODEL)
        assert llm._get_client() is llm._get_client()

    @pytest.mark.parametrize(
        ("chain", "name"), [("base", "AsyncLLMClient"), ("solana", "AsyncSolanaLLMClient")]
    )
    def test_async_client_per_event_loop(
        self, recorded: dict[str, Any], chain: str, name: str
    ) -> None:
        llm = BlockRun(model=MODEL, chain=chain)  # type: ignore[arg-type]

        async def get_twice() -> tuple[Any, Any]:
            return llm._get_aclient(), llm._get_aclient()

        first_a, first_b = asyncio.run(get_twice())
        second, _ = asyncio.run(get_twice())
        assert name in recorded
        assert first_a is first_b  # reused within a loop
        assert second is not first_a  # rebuilt on a new loop


# ----------------------------------------------------------------------
# Request parameters
# ----------------------------------------------------------------------


class TestRequestParameters:
    def test_unset_parameters_are_not_sent(self) -> None:
        llm, client = make_llm()
        llm.chat([ChatMessage(role="user", content="hi")])
        assert client.calls[0] == {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_set_parameters_and_additional_kwargs_are_sent(self) -> None:
        llm, client = make_llm(temperature=0.2, max_tokens=50, additional_kwargs={"top_p": 0.9})
        llm.chat([ChatMessage(role="user", content="hi")], stop=["\n"])
        call = client.calls[0]
        assert call["temperature"] == 0.2
        assert call["max_tokens"] == 50
        assert call["top_p"] == 0.9
        assert call["stop"] == ["\n"]

    def test_call_kwargs_override_instance_defaults(self) -> None:
        llm, client = make_llm(temperature=0.2)
        llm.chat([ChatMessage(role="user", content="hi")], temperature=0.7)
        assert client.calls[0]["temperature"] == 0.7

    def test_base_forwards_other_parameters(self) -> None:
        llm, client = make_llm()
        llm.chat([ChatMessage(role="user", content="hi")], reasoning_effort="low")
        assert client.calls[0]["reasoning_effort"] == "low"

    def test_solana_refuses_other_parameters_before_sending(self) -> None:
        llm, client = make_llm(chain="solana")
        with pytest.raises(ValueError, match="reasoning_effort"):
            llm.chat([ChatMessage(role="user", content="hi")], reasoning_effort="low")
        assert client.calls == []

    def test_system_prompt_is_sent_as_a_system_message(self) -> None:
        llm, client = make_llm()
        llm.chat(
            [
                ChatMessage(role="system", content="Be terse."),
                ChatMessage(role="user", content="hi"),
            ]
        )
        assert client.calls[0]["messages"][0] == {"role": "system", "content": "Be terse."}


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------


class TestChat:
    def test_text_usage_and_cost(self) -> None:
        llm, _ = make_llm(FakeClient(sdk_response("Paris.", citations=["https://x"])))
        response = llm.chat([ChatMessage(role="user", content="Capital of France?")])
        assert response.message.role == MessageRole.ASSISTANT
        assert response.message.content == "Paris."
        assert response.additional_kwargs == {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cost_usd": 0.0021,
            "citations": ["https://x"],
        }
        assert isinstance(response.raw, SdkChatResponse)

    def test_reasoning_becomes_a_thinking_block(self) -> None:
        llm, _ = make_llm(FakeClient(sdk_response("4", reasoning="2+2 is 4")))
        response = llm.chat([ChatMessage(role="user", content="2+2?")])
        thinking = [b for b in response.message.blocks if isinstance(b, ThinkingBlock)]
        assert [b.content for b in thinking] == ["2+2 is 4"]
        assert response.message.content == "4"

    def test_complete(self) -> None:
        llm, client = make_llm(FakeClient(sdk_response("Paris.")))
        assert llm.complete("Capital of France?").text == "Paris."
        assert client.calls[0]["messages"] == [{"role": "user", "content": "Capital of France?"}]

    def test_payment_errors_propagate(self) -> None:
        class Refusing(FakeClient):
            def chat_completion(self, *args: Any, **kwargs: Any) -> Any:
                raise blockrun_llm.SpendLimitError(
                    "quote $0.40 exceeds max_cost_per_call $0.05",
                    quoted_usd=0.40,
                    limit_usd=0.05,
                    scope="call",
                )

        llm, _ = make_llm(Refusing())
        with pytest.raises(blockrun_llm.SpendLimitError):
            llm.chat([ChatMessage(role="user", content="hi")])


class TestAsync:
    async def test_achat(self) -> None:
        client = FakeAsyncClient(sdk_response("async!"))
        llm = make_async_llm(client, temperature=0.3)
        response = await llm.achat([ChatMessage(role="user", content="hi")])
        assert response.message.content == "async!"
        assert client.calls[0]["temperature"] == 0.3

    async def test_acomplete(self) -> None:
        llm = make_async_llm(FakeAsyncClient(sdk_response("done")))
        assert (await llm.acomplete("go")).text == "done"

    async def test_astream_chat(self) -> None:
        client = FakeAsyncClient(chunks=[chunk("Hel"), chunk("lo")])
        llm = make_async_llm(client)
        stream = await llm.astream_chat([ChatMessage(role="user", content="hi")])
        deltas = [r.delta async for r in stream]
        assert deltas == ["Hel", "lo"]

    async def test_astream_complete(self) -> None:
        llm = make_async_llm(FakeAsyncClient(chunks=[chunk("a"), chunk("b")]))
        stream = await llm.astream_complete("go")
        texts = [r.text async for r in stream]
        assert texts == ["a", "ab"]


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------


class TestStreaming:
    def test_content_accumulates_and_usage_arrives_on_the_last_chunk(self) -> None:
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        llm, client = make_llm(
            FakeClient(chunks=[chunk("Hel"), chunk("lo"), chunk(usage=usage, no_choices=True)])
        )
        responses = list(llm.stream_chat([ChatMessage(role="user", content="hi")]))
        assert [r.delta for r in responses] == ["Hel", "lo", ""]
        assert [r.message.content for r in responses] == ["Hel", "Hello", "Hello"]
        assert responses[-1].additional_kwargs == usage
        assert client.calls[0]["stream"] is True

    def test_charge_attached_to_chunks_is_surfaced(self) -> None:
        paid = chunk("hi")
        paid.cost_usd = 0.0021  # type: ignore[attr-defined]  # as the Base SDK does
        llm, _ = make_llm(FakeClient(chunks=[paid]))
        (response,) = list(llm.stream_chat([ChatMessage(role="user", content="hi")]))
        assert response.additional_kwargs["cost_usd"] == 0.0021

    def test_stream_complete(self) -> None:
        llm, _ = make_llm(FakeClient(chunks=[chunk("a"), chunk("b")]))
        assert [r.text for r in llm.stream_complete("go")] == ["a", "ab"]

    def test_reasoning_streams_into_a_thinking_block(self) -> None:
        llm, _ = make_llm(FakeClient(chunks=[chunk(reasoning="think"), chunk("answer")]))
        responses = list(llm.stream_chat([ChatMessage(role="user", content="hi")]))
        assert responses[0].additional_kwargs["thinking_delta"] == "think"
        thinking = [b for b in responses[-1].message.blocks if isinstance(b, ThinkingBlock)]
        assert [b.content for b in thinking] == ["think"]
        assert responses[-1].message.content == "answer"

    def test_tool_call_arguments_accumulate_across_chunks(self) -> None:
        chunks = [
            chunk(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "add"}}]),
            chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"a": 1, '}}]),
            chunk(tool_calls=[{"index": 0, "function": {"arguments": '"b": 2}'}}]),
        ]
        llm, _ = make_llm(FakeClient(chunks=chunks))
        last = list(llm.stream_chat([ChatMessage(role="user", content="1+2")]))[-1]
        (selection,) = llm.get_tool_calls_from_response(last)
        assert (selection.tool_id, selection.tool_name) == ("call_1", "add")
        assert selection.tool_kwargs == {"a": 1, "b": 2}

    def test_parallel_tool_calls_stay_separate(self) -> None:
        chunks = [
            chunk(
                tool_calls=[
                    {"index": 0, "id": "c0", "function": {"name": "f", "arguments": '{"x":'}},
                    {"index": 1, "id": "c1", "function": {"name": "g", "arguments": '{"y":'}},
                ]
            ),
            chunk(tool_calls=[{"index": 1, "function": {"arguments": "2}"}}]),
            chunk(tool_calls=[{"index": 0, "function": {"arguments": "1}"}}]),
        ]
        llm, _ = make_llm(FakeClient(chunks=chunks))
        last = list(llm.stream_chat([ChatMessage(role="user", content="go")]))[-1]
        selections = llm.get_tool_calls_from_response(last)
        assert [(s.tool_id, s.tool_name, s.tool_kwargs) for s in selections] == [
            ("c0", "f", {"x": 1}),
            ("c1", "g", {"y": 2}),
        ]

    def test_tool_call_deltas_without_an_index(self) -> None:
        # Some upstreams omit `index`. A delta with no new id continues the
        # open call; a new id starts the next one.
        chunks = [
            chunk(tool_calls=[{"id": "a", "function": {"name": "f", "arguments": "{"}}]),
            chunk(tool_calls=[{"function": {"arguments": "}"}}]),
            chunk(tool_calls=[{"id": "b", "function": {"name": "g", "arguments": "{}"}}]),
        ]
        llm, _ = make_llm(FakeClient(chunks=chunks))
        last = list(llm.stream_chat([ChatMessage(role="user", content="go")]))[-1]
        selections = llm.get_tool_calls_from_response(last)
        assert [(s.tool_id, s.tool_name, s.tool_kwargs) for s in selections] == [
            ("a", "f", {}),
            ("b", "g", {}),
        ]

    def test_a_mid_stream_tool_call_reads_as_partial_json(self) -> None:
        chunks = [chunk(tool_calls=[{"index": 0, "id": "c", "function": {"name": "f"}}])]
        chunks.append(chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"q": "x", "n'}}]))
        llm, _ = make_llm(FakeClient(chunks=chunks))
        last = list(llm.stream_chat([ChatMessage(role="user", content="go")]))[-1]
        (selection,) = llm.get_tool_calls_from_response(last)
        assert selection.tool_kwargs == {"q": "x"}


# ----------------------------------------------------------------------
# Tool calling
# ----------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


ADD = FunctionTool.from_defaults(fn=add)
MULTIPLY = FunctionTool.from_defaults(fn=multiply)


class TestToolCalling:
    def test_chat_with_tools_sends_openai_tool_specs(self) -> None:
        llm, client = make_llm(
            FakeClient(sdk_response(None, tool_calls=[tool_call("c1", "add", '{"a":1,"b":2}')]))
        )
        response = llm.chat_with_tools([ADD], user_msg="What is 1 + 2?")
        call = client.calls[0]
        assert call["tool_choice"] == "auto"
        assert [t["function"]["name"] for t in call["tools"]] == ["add"]
        assert call["tools"][0]["type"] == "function"
        assert "parallel_tool_calls" not in call
        (selection,) = llm.get_tool_calls_from_response(response)
        assert (selection.tool_id, selection.tool_name) == ("c1", "add")
        assert selection.tool_kwargs == {"a": 1, "b": 2}

    def test_tool_required_and_named_tool_choice(self) -> None:
        llm, client = make_llm()
        llm.chat_with_tools([ADD], user_msg="hi", tool_required=True)
        llm.chat_with_tools([ADD, MULTIPLY], user_msg="hi", tool_choice="multiply")
        assert client.calls[0]["tool_choice"] == "required"
        assert client.calls[1]["tool_choice"] == {
            "type": "function",
            "function": {"name": "multiply"},
        }

    def test_chat_history_is_not_mutated(self) -> None:
        llm, client = make_llm()
        history = [ChatMessage(role="user", content="earlier")]
        llm.chat_with_tools([ADD], user_msg="now", chat_history=history)
        assert len(history) == 1
        assert [m["content"] for m in client.calls[0]["messages"]] == ["earlier", "now"]

    def test_single_tool_call_unless_parallel_allowed(self) -> None:
        calls = [tool_call("c1", "add", "{}"), tool_call("c2", "multiply", "{}")]
        llm, _ = make_llm(FakeClient(sdk_response(None, tool_calls=calls)))
        single = llm.chat_with_tools([ADD, MULTIPLY], user_msg="go")
        parallel = llm.chat_with_tools(
            [ADD, MULTIPLY], user_msg="go", allow_parallel_tool_calls=True
        )
        assert [s.tool_id for s in llm.get_tool_calls_from_response(single)] == ["c1"]
        assert [s.tool_id for s in llm.get_tool_calls_from_response(parallel)] == ["c1", "c2"]

    def test_no_tool_call(self) -> None:
        llm, _ = make_llm()
        response = llm.chat([ChatMessage(role="user", content="hi")])
        with pytest.raises(ValueError, match="at least one tool call"):
            llm.get_tool_calls_from_response(response)
        assert llm.get_tool_calls_from_response(response, error_on_no_tool_call=False) == []

    def test_tool_turn_round_trips_into_the_next_request(self) -> None:
        # The assistant turn that called a tool, and the tool's result, must be
        # sent back in the OpenAI shape or the upstream rejects the follow-up.
        llm, client = make_llm(
            FakeClient(sdk_response(None, tool_calls=[tool_call("c1", "add", '{"a":1,"b":2}')]))
        )
        first = llm.chat_with_tools([ADD], user_msg="1+2?")
        tool_result = ChatMessage(
            role=MessageRole.TOOL,
            content="3",
            additional_kwargs={"tool_call_id": "c1", "name": "add"},
        )
        llm.chat([ChatMessage(role="user", content="1+2?"), first.message, tool_result])
        _, assistant, tool = client.calls[1]["messages"]
        assert assistant["role"] == "assistant"
        assert assistant["content"] is None
        assert assistant["tool_calls"] == [
            {
                "type": "function",
                "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                "id": "c1",
            }
        ]
        assert tool["role"] == "tool"
        assert tool["tool_call_id"] == "c1"
        assert tool["content"] == "3"

    def test_thinking_is_not_sent_back(self) -> None:
        llm, client = make_llm(FakeClient(sdk_response("4", reasoning="secret chain")))
        first = llm.chat([ChatMessage(role="user", content="2+2?")])
        llm.chat([ChatMessage(role="user", content="2+2?"), first.message])
        assert "secret chain" not in json.dumps(client.calls[1]["messages"])

    def test_tool_call_block_kwargs_are_kept_as_sent(self) -> None:
        llm, _ = make_llm(FakeClient(sdk_response(None, tool_calls=[tool_call("c", "f", "{}")])))
        response = llm.chat([ChatMessage(role="user", content="go")])
        (block,) = [b for b in response.message.blocks if isinstance(b, ToolCallBlock)]
        assert block.tool_kwargs == "{}"

    def test_malformed_arguments_become_empty_kwargs(self) -> None:
        llm, _ = make_llm(
            FakeClient(sdk_response(None, tool_calls=[tool_call("c", "f", "not json")]))
        )
        response = llm.chat([ChatMessage(role="user", content="go")])
        (selection,) = llm.get_tool_calls_from_response(response)
        assert selection.tool_kwargs == {}

    def test_predict_and_call_runs_the_tool(self) -> None:
        llm, _ = make_llm(
            FakeClient(sdk_response(None, tool_calls=[tool_call("c1", "add", '{"a":2,"b":3}')]))
        )
        output = llm.predict_and_call([ADD], user_msg="2+3?")
        assert output.response == "5"


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


class TestMetadata:
    def test_context_window_comes_from_the_catalog_once(self) -> None:
        client = FakeClient(models=[{"id": "other"}, {"id": MODEL, "context_window": 1_050_000}])
        llm, _ = make_llm(client)
        assert llm.metadata.context_window == 1_050_000
        assert llm.metadata.context_window == 1_050_000
        assert client.list_models_calls == 1

    def test_explicit_context_window_skips_the_catalog(self) -> None:
        llm, client = make_llm(context_window=8000)
        assert llm.metadata.context_window == 8000
        assert client.list_models_calls == 0

    def test_unknown_model_falls_back(self) -> None:
        llm, _ = make_llm(FakeClient(models=[{"id": "other", "context_window": 5}]))
        assert llm.metadata.context_window == DEFAULT_CONTEXT_WINDOW

    def test_unreachable_catalog_falls_back_once(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeClient(models_error=RuntimeError("offline"))
        llm, _ = make_llm(client)
        assert llm.metadata.context_window == DEFAULT_CONTEXT_WINDOW
        assert llm.metadata.context_window == DEFAULT_CONTEXT_WINDOW
        assert client.list_models_calls == 1
        assert "context_window" in caplog.text

    def test_other_fields(self) -> None:
        llm, _ = make_llm(context_window=8000, max_tokens=300, is_function_calling_model=False)
        meta = llm.metadata
        assert meta.model_name == MODEL
        assert meta.num_output == 300
        assert meta.is_chat_model is True
        assert meta.is_function_calling_model is False

    def test_num_output_defaults_to_the_sdk_default(self) -> None:
        llm, _ = make_llm(context_window=8000)
        assert llm.metadata.num_output == 1024

    def test_wallet_address(self) -> None:
        llm, _ = make_llm()
        assert llm.wallet_address == "0xabc"
