"""BlockRun LLM for LlamaIndex.

BlockRun is an OpenAI-compatible gateway where every call pays for itself in
USDC over x402. There is no API key: the gateway answers an unpaid request with
HTTP 402 and a price, the client signs a payment with a local wallet, and the
request is retried with the signature attached. That payment is either the
quoted amount (x402 ``exact``) or, on Base when the gateway offers it, a
ceiling the gateway settles at the actual cost (x402 ``upto``).

That signing step is why this is a package rather than an ``OpenAILike``
snippet. ``OpenAILike(api_base=..., api_key="fake")`` reaches the gateway and
then stops at the first 402, because nothing in the OpenAI client knows how to
pay. Payment here is delegated to the ``blockrun-llm`` SDK, which owns the
signing, the per-chain differences, and the retry rules that keep a request
from being paid twice.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    ContentBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.utils import parse_partial_json
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.openai.utils import resolve_tool_choice, to_openai_message_dicts

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

logger = logging.getLogger(__name__)

Chain = Literal["base", "solana"]

# What the SDK sends when max_tokens is left unset, on both chains. Reported
# as num_output so LlamaIndex budgets the prompt against what will be generated.
DEFAULT_MAX_TOKENS = 1024

# The parameters both chains' clients accept by name. The Base client also
# forwards any other keyword into the request body; the Solana client takes
# only these, so an unknown key there would be a TypeError from deep inside
# the SDK. It is refused up front instead, naming the key.
_SHARED_REQUEST_PARAMS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "search",
        "search_parameters",
        "tools",
        "tool_choice",
        "response_format",
        "stop",
        "fallback_models",
    }
)


class BlockRun(FunctionCallingLLM):
    """BlockRun LLM: one OpenAI-compatible gateway, paid per call in USDC over x402.

    Every BlockRun chat model is reachable through this class, addressed by its
    catalog id (``openai/gpt-5.5``, ``anthropic/claude-sonnet-4.6``,
    ``deepseek/deepseek-chat`` ...). The full list is at
    https://blockrun.ai/api/v1/models.

    There is no API key. Requests are paid from a wallet on the chain you pick:

    * ``chain="base"`` (default): USDC on Base. The wallet is an EVM private
      key, read from ``private_key``, then ``BLOCKRUN_WALLET_KEY``, then the
      wallet file ``~/.blockrun/.session``.
    * ``chain="solana"``: USDC on Solana. The wallet is a base58 secret key,
      read from ``private_key``, then ``SOLANA_WALLET_KEY``, then a Solana
      wallet file on disk. Needs ``pip install "llama-index-llms-blockrun[solana]"``.

    The key is used only to sign payments locally; it is never sent anywhere.

    Install:
        ``pip install llama-index-llms-blockrun``

    Examples:
        .. code-block:: python

            from llama_index.llms.blockrun import BlockRun

            # Pays in USDC on Base (the default chain).
            llm = BlockRun(model="openai/gpt-5.5")
            print(llm.complete("Paris is the capital of"))

            # Pays in USDC on Solana.
            llm = BlockRun(model="openai/gpt-5.5", chain="solana")

            # Refuse any single call quoted above $0.05 before signing it.
            llm = BlockRun(model="openai/gpt-5.5", max_cost_per_call=0.05)

    """

    model: str = Field(description="BlockRun model id, e.g. 'openai/gpt-5.5'.")
    chain: Chain = Field(
        default="base",
        description="Chain the wallet pays from: 'base' (USDC on Base) or 'solana' (USDC on Solana).",
    )
    temperature: float | None = Field(
        default=None,
        description=(
            "Sampling temperature. Unset sends none, so each model uses its own "
            "default; several reasoning models accept no other value."
        ),
        ge=0.0,
        le=2.0,
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate. Unset sends the SDK default of 1024.",
        gt=0,
    )
    context_window: int | None = Field(
        default=None,
        description=(
            "Context window in tokens. Unset reads the model's own window from the "
            "gateway's /v1/models catalog the first time it is needed."
        ),
        gt=0,
    )
    is_function_calling_model: bool = Field(
        default=True,
        description="Whether the model supports OpenAI-style tool calling.",
    )
    max_cost_per_call: float | None = Field(
        default=None,
        description=(
            "Refuse to sign any single payment quoted above this many USD. The call "
            "fails before anything is sent or paid. Unset signs every quote."
        ),
        gt=0,
    )
    api_url: str | None = Field(
        default=None,
        description=(
            "Gateway URL override, e.g. https://testnet.blockrun.ai/api for Base "
            "Sepolia. Unset uses the chain's mainnet gateway."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description="HTTP timeout in seconds. Unset uses the SDK default (600).",
        gt=0,
    )
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra request parameters sent with every call, e.g. top_p or stop.",
    )

    # Never a field: fields are serialized by to_dict()/to_json() and by any
    # callback that logs the LLM, and this is a wallet key.
    _private_key: str | None = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)
    _aclient: Any = PrivateAttr(default=None)
    _aclient_loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)
    _catalog_entry: dict[str, Any] | None = PrivateAttr(default=None)
    _catalog_loaded: bool = PrivateAttr(default=False)

    def __init__(
        self,
        model: str,
        chain: Chain = "base",
        private_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
        is_function_calling_model: bool = True,
        max_cost_per_call: float | None = None,
        api_url: str | None = None,
        timeout: float | None = None,
        additional_kwargs: dict[str, Any] | None = None,
        callback_manager: CallbackManager | None = None,
        system_prompt: str | None = None,
        output_parser: BaseOutputParser | None = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        **kwargs: Any,
    ) -> None:
        if chain not in ("base", "solana"):
            raise ValueError(f"chain must be 'base' or 'solana', got {chain!r}")
        super().__init__(
            model=model,
            chain=chain,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            is_function_calling_model=is_function_calling_model,
            max_cost_per_call=max_cost_per_call,
            api_url=api_url,
            timeout=timeout,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            output_parser=output_parser,
            pydantic_program_mode=pydantic_program_mode,
            **kwargs,
        )
        self._private_key = private_key
        if chain == "solana":
            # Checked before the request instead of inside it: on Solana an
            # unknown parameter is a TypeError from the SDK, not a 400.
            _check_solana_params(self.additional_kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "BlockRun_LLM"

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def _client_kwargs(self) -> dict[str, Any]:
        # Only what the caller set. The Solana client's api_url default is a
        # string, not None, so passing None through would replace its gateway.
        kwargs: dict[str, Any] = {"private_key": self._private_key}
        if self.api_url is not None:
            kwargs["api_url"] = self.api_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_cost_per_call is not None:
            kwargs["max_cost_per_call"] = self.max_cost_per_call
        return kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            if self.chain == "solana":
                from blockrun_llm import SolanaLLMClient as client_cls
            else:
                from blockrun_llm import LLMClient as client_cls
            self._client = client_cls(**self._client_kwargs())
        return self._client

    def _get_aclient(self) -> Any:
        # An httpx.AsyncClient is bound to the loop it first ran on. Each
        # asyncio.run() is a new loop, so a client cached across them fails
        # with "Event loop is closed". Keep one per running loop.
        loop = asyncio.get_running_loop()
        if self._aclient is None or self._aclient_loop is not loop:
            if self.chain == "solana":
                from blockrun_llm import AsyncSolanaLLMClient as aclient_cls
            else:
                from blockrun_llm import AsyncLLMClient as aclient_cls
            self._aclient = aclient_cls(**self._client_kwargs())
            self._aclient_loop = loop
        return self._aclient

    @property
    def wallet_address(self) -> str:
        """Address of the wallet paying for calls (EVM on Base, base58 on Solana)."""
        return str(self._get_client().get_wallet_address())

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _lookup_catalog_entry(self) -> dict[str, Any] | None:
        # One free GET /v1/models per instance. A failure is cached too, so an
        # unreachable catalog costs one warning rather than a request every
        # time LlamaIndex reads metadata; set context_window to skip this.
        if not self._catalog_loaded:
            self._catalog_loaded = True
            try:
                models = self._get_client().list_models()
            except Exception as exc:
                logger.warning(
                    "BlockRun: could not read the model catalog (%s); using a "
                    "context window of %d. Pass context_window= to set it.",
                    exc,
                    DEFAULT_CONTEXT_WINDOW,
                )
                models = []
            self._catalog_entry = next((m for m in models if m.get("id") == self.model), None)
        return self._catalog_entry

    @property
    def metadata(self) -> LLMMetadata:
        context_window = self.context_window
        if context_window is None:
            entry = self._lookup_catalog_entry()
            context_window = (entry or {}).get("context_window") or DEFAULT_CONTEXT_WINDOW
        return LLMMetadata(
            context_window=context_window,
            num_output=self.max_tokens or DEFAULT_MAX_TOKENS,
            is_chat_model=True,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _request_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        merged = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        }
        # None means "not set" throughout; the SDK applies its own defaults.
        merged = {k: v for k, v in merged.items() if v is not None}
        if self.chain == "solana":
            _check_solana_params(merged)
        return merged

    def _message_dicts(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        # A reasoning trace is output, never input: the chat schema has no
        # field for it. Dropped here rather than left to the serializer:
        # llama-index-llms-openai 0.8 skips a ThinkingBlock, but 0.6.5, inside
        # the supported range, raises on it.
        cleaned = [
            (
                message.model_copy(
                    update={
                        "blocks": [b for b in message.blocks if not isinstance(b, ThinkingBlock)]
                    }
                )
                if any(isinstance(b, ThinkingBlock) for b in message.blocks)
                else message
            )
            for message in messages
        ]
        return list(to_openai_message_dicts(cleaned))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = self._get_client().chat_completion(
            self.model, self._message_dicts(messages), **self._request_kwargs(**kwargs)
        )
        return _to_chat_response(response)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        chunks = self._get_client().chat_completion_stream(
            self.model, self._message_dicts(messages), **self._request_kwargs(**kwargs)
        )

        def gen() -> ChatResponseGen:
            accumulator = _StreamAccumulator()
            for chunk in chunks:
                yield accumulator.step(chunk)

        return gen()

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = await self._get_aclient().chat_completion(
            self.model, self._message_dicts(messages), **self._request_kwargs(**kwargs)
        )
        return _to_chat_response(response)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        chunks: AsyncIterator[Any] = self._get_aclient().chat_completion_stream(
            self.model, self._message_dicts(messages), **self._request_kwargs(**kwargs)
        )

        async def gen() -> ChatResponseAsyncGen:
            accumulator = _StreamAccumulator()
            async for chunk in chunks:
                yield accumulator.step(chunk)

        return gen()

    # ------------------------------------------------------------------
    # Completion (a single user turn through the chat endpoint)
    # ------------------------------------------------------------------

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return chat_to_completion_decorator(self.chat)(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return stream_chat_to_completion_decorator(self.stream_chat)(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return await achat_to_completion_decorator(self.achat)(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await astream_chat_to_completion_decorator(self.astream_chat)(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tool_specs = [tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools]

        # A copy: appending to the caller's chat_history would leave the user
        # turn in their memory a second time when they record it themselves.
        messages = list(chat_history or [])
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
        if user_msg is not None:
            messages.append(user_msg)

        # parallel_tool_calls is not sent: the Solana client does not accept
        # it, and not every upstream behind the gateway honors it. A single
        # call is enforced on the response instead, in
        # _validate_chat_with_tools_response.
        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": (
                resolve_tool_choice(tool_choice, tool_required) if tool_specs else None
            ),
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: Sequence[BaseTool],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        if not allow_parallel_tool_calls:
            _force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        tool_calls = [b for b in response.message.blocks if isinstance(b, ToolCallBlock)]
        if not tool_calls:
            if error_on_no_tool_call:
                raise ValueError("Expected at least one tool call, but got 0 tool calls.")
            return []

        selections = []
        for call in tool_calls:
            arguments = call.tool_kwargs
            if isinstance(arguments, str):
                # Partial JSON too: a streamed call can be read mid-stream.
                try:
                    arguments = parse_partial_json(arguments) if arguments.strip() else {}
                except (ValueError, TypeError):
                    arguments = {}
            selections.append(
                ToolSelection(
                    tool_id=call.tool_call_id or "",
                    tool_name=call.tool_name,
                    tool_kwargs=arguments if isinstance(arguments, dict) else {},
                )
            )
        return selections


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _check_solana_params(params: Iterable[str]) -> None:
    unsupported = sorted(set(params) - _SHARED_REQUEST_PARAMS)
    if unsupported:
        raise ValueError(
            f"chain='solana' does not support the request parameter(s) {unsupported}. "
            f"Supported: {sorted(_SHARED_REQUEST_PARAMS)}."
        )


def _response_kwargs(
    usage: Any,
    cost_usd: float | None = None,
    citations: list[str] | None = None,
    source: Any = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if usage is not None:
        out["prompt_tokens"] = usage.prompt_tokens
        out["completion_tokens"] = usage.completion_tokens
        out["total_tokens"] = usage.total_tokens
    if cost_usd is not None:
        out["cost_usd"] = cost_usd
        # Under x402 upto, cost_usd can be the signed CEILING rather than the
        # settled charge (always, for streams). Carry the SDK's label so a
        # caller never reports an upper bound as money spent.
        scheme = getattr(source, "payment_scheme", None)
        if scheme is not None:
            out["payment_scheme"] = scheme
            out["cost_is_ceiling"] = bool(getattr(source, "cost_is_ceiling", False))
    if citations:
        out["citations"] = citations
    return out


def _to_chat_response(response: Any) -> ChatResponse:
    message = response.choices[0].message
    blocks: list[ContentBlock] = []
    reasoning = message.reasoning_content or message.thinking
    if reasoning:
        blocks.append(ThinkingBlock(content=reasoning))
    blocks.append(TextBlock(text=message.content or ""))
    for call in message.tool_calls or []:
        blocks.append(
            ToolCallBlock(
                tool_call_id=call.id,
                tool_name=call.function.name,
                tool_kwargs=call.function.arguments or {},
            )
        )
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, blocks=blocks),
        raw=response,
        additional_kwargs=_response_kwargs(
            response.usage,
            getattr(response, "cost_usd", None),
            response.citations,
            source=response,
        ),
    )


def _force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = [b for b in response.message.blocks if isinstance(b, ToolCallBlock)]
    if len(tool_calls) > 1:
        response.message.blocks = [
            b for b in response.message.blocks if not isinstance(b, ToolCallBlock)
        ] + [tool_calls[0]]


class _StreamAccumulator:
    """Folds SSE chunks into the running message each streamed ChatResponse carries."""

    def __init__(self) -> None:
        self.content = ""
        self.reasoning = ""
        # Keyed by the delta's tool-call index; dicts keep first-seen order.
        self.tool_calls: dict[int, dict[str, str]] = {}

    def step(self, chunk: Any) -> ChatResponse:
        delta = chunk.choices[0].delta if chunk.choices else None
        content_delta = (delta.content if delta else None) or ""
        reasoning_delta = ((delta.reasoning_content or delta.thinking) if delta else None) or ""
        self.content += content_delta
        self.reasoning += reasoning_delta
        for call_delta in (delta.tool_calls if delta else None) or []:
            self._merge_tool_call(call_delta)

        blocks: list[ContentBlock] = []
        if self.reasoning:
            blocks.append(ThinkingBlock(content=self.reasoning))
        blocks.append(TextBlock(text=self.content))
        for call in self.tool_calls.values():
            blocks.append(
                ToolCallBlock(
                    tool_call_id=call["id"] or None,
                    tool_name=call["name"],
                    tool_kwargs=call["arguments"] or {},
                )
            )

        # The Base SDK attaches the real x402 charge to every chunk; read it
        # with getattr because it is an extra attribute, not a declared field.
        additional = _response_kwargs(
            chunk.usage, getattr(chunk, "cost_usd", None), chunk.citations, source=chunk
        )
        if reasoning_delta:
            additional["thinking_delta"] = reasoning_delta
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, blocks=blocks),
            delta=content_delta,
            raw=chunk,
            additional_kwargs=additional,
        )

    def _merge_tool_call(self, call_delta: Any) -> None:
        if call_delta.index is not None:
            index = call_delta.index
        elif self.tool_calls and (
            not call_delta.id or call_delta.id == self.tool_calls[self._last_index()]["id"]
        ):
            # No index: a delta without a new id continues the open call.
            index = self._last_index()
        else:
            index = max(self.tool_calls) + 1 if self.tool_calls else 0

        call = self.tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if call_delta.id:
            call["id"] = call_delta.id
        if call_delta.function is not None:
            call["name"] += call_delta.function.name or ""
            call["arguments"] += call_delta.function.arguments or ""

    def _last_index(self) -> int:
        return next(reversed(self.tool_calls))
