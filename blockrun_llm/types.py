"""Type definitions for BlockRun LLM SDK."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# Tool calling types (OpenAI compatible)
class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class Tool(BaseModel):
    """Tool definition for chat completions."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """Function call details within a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call made by the assistant."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# Tool choice can be a string or object specifying which tool to use
ToolChoiceFunction = Dict[str, Any]  # {"type": "function", "function": {"name": "..."}}
ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceFunction]


class ChatMessage(BaseModel):
    """A single chat message.

    Passthrough: the named fields below are conveniences; any other field the
    gateway forwards (e.g. ``annotations``, ``audio``, future OpenAI additions)
    is preserved via ``extra = "allow"`` rather than silently dropped.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None  # For tool messages
    tool_call_id: Optional[str] = None  # For tool result messages
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages with tool calls
    # Extended fields returned by reasoning-capable upstream providers
    # (DeepSeek Reasoner, Grok 4 reasoning, xAI multi-agent, etc.).
    # Backend strips these from inbound requests but may forward them on the
    # response side, so we accept them as optional.
    reasoning_content: Optional[str] = None
    thinking: Optional[str] = None

    class Config:
        extra = "allow"


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None  # OpenAI-compatible; upstreams may add new values

    class Config:
        extra = "allow"


class ChatUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_sources_used: Optional[int] = None  # xAI Live Search sources used
    # Anthropic prompt caching — populated on anthropic/* models when cache
    # headers are sent. Reads are cheaper; writes incur a one-time surcharge.
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    # Provider-native token detail. reasoning_tokens is a subset of
    # completion_tokens and must not be added again when calculating spend.
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None

    @property
    def reasoning_tokens(self) -> Optional[int]:
        """Reasoning tokens the model spent, when upstream reports them.

        Nested under ``completion_tokens_details`` in the OpenAI shape the
        gateway forwards. The flat fallback matters because this class allows
        extras: a payload carrying a top-level ``reasoning_tokens`` used to
        reach callers through ``__getattr__``, and a property of the same name
        takes precedence over that, so without the fallback this would answer
        None for a number the payload demonstrably carried.

        ``bool`` is excluded deliberately — it is an ``int`` subclass, and
        ``True`` is not a token count.
        """
        detail = self.completion_tokens_details or {}
        value = detail.get("reasoning_tokens")
        if value is None:
            value = (self.model_extra or {}).get("reasoning_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    class Config:
        extra = "allow"


class ChatResponse(BaseModel):
    """Response from chat completion.

    Passthrough: unknown top-level fields the gateway returns (e.g.
    ``system_fingerprint``, ``service_tier``, ``prompt_logprobs``) are kept via
    ``extra = "allow"`` so the SDK never strips what the API sends.
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None
    citations: Optional[List[str]] = None  # xAI Live Search citation URLs

    # Real x402 charge for THIS call, in USD — the exact amount debited from the
    # wallet (0.0 for free / cached calls). This is the authoritative number to
    # bill/track against; token-count × list-price estimates do NOT match it
    # because the gateway price carries a per-call floor + margin. Populated by
    # the client on every chat completion. ``settlement`` carries the decoded
    # on-chain receipt (tx hash / micro-USDC / network) when the facilitator
    # returned an X-PAYMENT-RESPONSE header.
    cost_usd: Optional[float] = None
    settlement: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Streaming (SSE) chunk types — OpenAI Chat Completions chunk schema.
#
# Backend emits ``data: <json>\n\n`` lines terminated by ``data: [DONE]\n\n``.
# First chunk's delta has ``role="assistant"``; subsequent chunks fill
# ``content``; final chunk carries ``finish_reason`` and optionally ``usage``.
# ---------------------------------------------------------------------------


class ChatChunkFunctionCall(BaseModel):
    """Streaming function-call delta. The model sends ``name`` on the first
    frame and ``arguments`` in fragments afterwards, so both are optional here —
    unlike the non-stream :class:`FunctionCall` where both are required."""

    name: Optional[str] = None
    arguments: Optional[str] = None

    class Config:
        extra = "allow"


class ChatChunkToolCall(BaseModel):
    """One streaming tool-call delta.

    OpenAI streams tool calls incrementally: the first frame carries
    ``index`` + ``id`` + ``function.name`` (+ empty args), later frames carry
    only ``index`` + ``function.arguments`` fragments. Every field is therefore
    optional. The strict non-stream :class:`ToolCall` (``id`` / ``function.name``
    / ``arguments`` all required) rejected the argument-fragment frames, which
    made ``ChatCompletionChunk(**chunk)`` raise and fall back to
    ``model_construct`` — leaving ``choices`` as raw dicts and crashing the
    archive loop with ``'dict' object has no attribute 'delta'``. Using this
    lenient type keeps streamed tool calls parsing into real objects.
    """

    index: Optional[int] = None
    id: Optional[str] = None
    # Kept as a free-form ``str`` (not ``Literal["function"]``) so an upstream
    # that streams a non-"function" tool type can't fail validation and re-trigger
    # the very ``model_construct`` fallback this lenient type exists to avoid.
    type: Optional[str] = None
    function: Optional[ChatChunkFunctionCall] = None

    class Config:
        extra = "allow"


class ChatChunkDelta(BaseModel):
    """Incremental ``message`` delta sent over SSE.

    Any field may be absent in a given chunk — ``role`` typically only on the
    first, ``content`` on body chunks, ``tool_calls`` when the model decides
    to call a tool. ``reasoning_content`` / ``thinking`` appear on
    reasoning-capable upstreams.
    """

    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChatChunkToolCall]] = None
    reasoning_content: Optional[str] = None
    thinking: Optional[str] = None

    class Config:
        extra = "allow"


class ChatChunkChoice(BaseModel):
    """One choice within a streaming chunk."""

    index: int
    delta: ChatChunkDelta
    finish_reason: Optional[str] = None  # OpenAI-compatible; upstreams may add new values

    class Config:
        extra = "allow"


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk emitted by ``/v1/chat/completions`` when stream=True."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatChunkChoice]
    # Usage is populated only on the final chunk for providers that support it
    # (some upstreams omit it entirely — callers must tolerate ``None``).
    usage: Optional[ChatUsage] = None
    citations: Optional[List[str]] = None  # xAI Live Search citation URLs (final chunk only)

    class Config:
        extra = "allow"


def stream_choice_content(choice: Any) -> Optional[str]:
    """Text delta from a streaming choice, tolerant of a raw ``dict`` choice.

    A chunk that fails strict validation falls back to ``model_construct``,
    which leaves nested ``choices`` as plain dicts. Defensive accessors keep the
    stream-archiving loop from crashing on those (``'dict' object has no
    attribute 'delta'``); a tool-call frame simply has no content and yields
    ``None``.
    """
    if isinstance(choice, dict):
        delta = choice.get("delta")
        return delta.get("content") if isinstance(delta, dict) else None
    delta = getattr(choice, "delta", None)
    return getattr(delta, "content", None) if delta is not None else None


def stream_choice_finish_reason(choice: Any) -> Optional[str]:
    """``finish_reason`` from a streaming choice, tolerant of a raw dict choice."""
    if isinstance(choice, dict):
        return choice.get("finish_reason")
    return getattr(choice, "finish_reason", None)


def chunk_meta(chunk: Any) -> "tuple[Optional[str], Optional[str], Optional[int]]":
    """``(id, model, created)`` of a chunk, tolerant of a ``model_construct``'d
    chunk that omits required fields.

    ``model_construct`` does not populate missing required fields, so a drifted
    frame that lost its top-level ``id`` yields a chunk object with no ``id``
    attribute. Reading ``chunk.id`` directly would then raise ``AttributeError``
    and crash the stream-archiving loop — the same failure class the other
    accessors here guard against. ``getattr`` keeps those reads safe.
    """
    return (
        getattr(chunk, "id", None),
        getattr(chunk, "model", None),
        getattr(chunk, "created", None),
    )


def chunk_usage_dict(chunk: Any) -> Optional[Dict[str, Any]]:
    """``usage`` of a chunk as a dict, tolerant of a model_construct'd chunk
    whose ``usage`` is a raw dict (no ``.model_dump``)."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return {k: v for k, v in usage.items() if v is not None}
    return usage.model_dump(exclude_none=True)


class Model(BaseModel):
    """Available model information."""

    id: str
    name: str
    provider: str
    description: str
    input_price: float  # Per 1M tokens (0 when billing_mode != "paid")
    output_price: float  # Per 1M tokens (0 when billing_mode != "paid")
    context_window: int
    max_output: int
    available: bool = True
    # Extended metadata surfaced by /v1/models. `billing_mode` is one of
    # "paid" (per-token), "flat" (flat_price per request) or "free".
    billing_mode: Optional[Literal["paid", "flat", "free"]] = None
    flat_price: Optional[float] = None
    categories: Optional[List[str]] = None  # e.g. ["chat","reasoning","coding","vision"]
    hidden: Optional[bool] = None  # True for deprecated/superseded models still routable


class PaymentRequirement(BaseModel):
    """x402 payment requirement."""

    scheme: str
    network: str
    asset: str
    amount: str
    pay_to: str
    max_timeout_seconds: int = 300


class PaymentRequired(BaseModel):
    """x402 payment required response."""

    x402_version: int = 1
    accepts: List[PaymentRequirement]


class BlockrunError(Exception):
    """Base exception for BlockRun SDK."""


class RetiredEndpointError(BlockrunError):
    """Raised by a helper whose upstream endpoint no longer exists.

    Kept as a raising method rather than deleted so upgrading does not break
    imports or attribute access — the failure is explicit and immediate instead
    of a paid round trip that returns 410/404.
    """


class PaymentError(BlockrunError):
    """Payment-related error.

    Optionally carries ``status_code`` and ``response`` so callers and
    upstream proxies can surface the gateway's real failure reason
    (e.g. a Solana facilitator ``transaction_simulation_failed``)
    instead of seeing only a generic SDK message.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class SpendLimitError(PaymentError):
    """A quote exceeded a spend limit the caller configured, so it was refused.

    Raised *before* the paid request goes out, so nothing settles: the quote is
    declined locally and no funds move. Subclasses :class:`PaymentError` so
    existing ``except PaymentError`` handlers keep working, and so the model
    fallback chain refuses it — retrying another model after declining on cost
    would defeat the limit.

    ``quoted_usd`` is what the gateway asked for, ``limit_usd`` is the ceiling
    that refused it, and ``scope`` is ``"call"`` or ``"session"``.
    """

    def __init__(
        self,
        message: str,
        *,
        quoted_usd: float,
        limit_usd: float,
        scope: str,
    ) -> None:
        super().__init__(message)
        self.quoted_usd = quoted_usd
        self.limit_usd = limit_usd
        self.scope = scope


class APIError(BlockrunError):
    """API-related error."""

    def __init__(self, message: str, status_code: int, response: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


# Image generation types
class ImageData(BaseModel):
    """A single generated image."""

    url: str
    # When the gateway mirrors the asset to its own storage, `url` is the
    # permanent blockrun-hosted URL and `source_url` is the original upstream.
    # `backed_up` is True iff the mirror step succeeded. For data-URI results
    # (e.g. openai/gpt-image-1) both fields are omitted.
    source_url: Optional[str] = None
    backed_up: Optional[bool] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    """Response from image generation."""

    created: int
    data: List[ImageData]


class ImageModel(BaseModel):
    """Available image model information."""

    id: str
    name: str
    provider: str
    description: str
    price_per_image: float
    available: bool = True


# Music / Audio types


class AudioTrack(BaseModel):
    """A single generated audio track."""

    url: str
    duration_seconds: Optional[float] = None
    lyrics: Optional[str] = None


class MusicResponse(BaseModel):
    """Response from music generation."""

    created: int
    model: str
    data: List[AudioTrack]
    txHash: Optional[str] = None


class AudioModel(BaseModel):
    """Available audio/music model information."""

    id: str
    name: str
    provider: str
    description: str
    price_per_track: float
    max_duration_seconds: int


# Speech (TTS / sound effects) types


class SpeechAudio(BaseModel):
    """A single synthesized audio clip."""

    url: str
    format: Optional[str] = None
    characters: Optional[int] = None
    credits: Optional[float] = None


class SpeechResponse(BaseModel):
    """Response from speech synthesis or sound-effect generation."""

    created: int
    model: str
    data: List[SpeechAudio]
    txHash: Optional[str] = None


# Multi-chain RPC types


class RpcError(BaseModel):
    """A JSON-RPC 2.0 error object."""

    code: Optional[int] = None
    message: Optional[str] = None
    data: Optional[Any] = None


class RpcResponse(BaseModel):
    """Response from a multi-chain JSON-RPC call (/v1/rpc/{network}).

    Standard JSON-RPC 2.0 envelope plus BlockRun gateway metadata pulled
    from response headers (X-Network / X-Cache / X-Payment-Receipt).
    """

    jsonrpc: Optional[str] = None
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[RpcError] = None
    # Gateway metadata (response headers)
    network: Optional[str] = None  # canonical network key, e.g. "ethereum"
    cache_hit: bool = False  # served from the gateway's method-aware cache
    tx_hash: Optional[str] = None  # x402 settlement tx (single calls)


# Video generation types


class VideoClip(BaseModel):
    """A single generated video clip."""

    url: str  # Permanent blockrun-hosted URL (falls back to upstream if backup fails)
    source_url: Optional[str] = None  # Original upstream URL (e.g. vidgen.x.ai)
    duration_seconds: Optional[int] = None
    request_id: Optional[str] = None  # Upstream provider's request id (xAI)
    backed_up: Optional[bool] = None


class VideoResponse(BaseModel):
    """Response from video generation."""

    created: int
    model: str
    data: List[VideoClip]
    txHash: Optional[str] = None


class VideoModel(BaseModel):
    """Available video model information."""

    id: str
    name: str
    provider: str
    description: str
    price_per_second: float
    default_duration_seconds: int
    max_duration_seconds: int
    supports_image_input: bool = False
    supports_lyrics: bool
    supports_instrumental: bool
    available: bool = True


# Live Search types
class WebSearchSource(BaseModel):
    """Web search source configuration."""

    type: Literal["web"] = "web"
    country: Optional[str] = None  # ISO alpha-2 country code
    excluded_websites: Optional[List[str]] = None  # Max 5 websites
    allowed_websites: Optional[List[str]] = (
        None  # Max 5 websites (mutually exclusive with excluded)
    )
    safe_search: bool = True


class XSearchSource(BaseModel):
    """X/Twitter search source configuration."""

    type: Literal["x"] = "x"
    included_x_handles: Optional[List[str]] = None  # Max 10 handles
    excluded_x_handles: Optional[List[str]] = None  # Max 10 handles
    post_favorite_count: Optional[int] = None  # Minimum favorites threshold
    post_view_count: Optional[int] = None  # Minimum views threshold


class NewsSearchSource(BaseModel):
    """News search source configuration."""

    type: Literal["news"] = "news"
    country: Optional[str] = None  # ISO alpha-2 country code
    excluded_websites: Optional[List[str]] = None  # Max 5 websites
    allowed_websites: Optional[List[str]] = None  # Max 5 websites
    safe_search: bool = True


class RssSearchSource(BaseModel):
    """RSS feed search source configuration."""

    type: Literal["rss"] = "rss"
    links: List[str]  # RSS feed URLs (currently supports one)


SearchSource = Union[
    WebSearchSource, XSearchSource, NewsSearchSource, RssSearchSource, Dict[str, Any]
]


class SearchParameters(BaseModel):
    """
    Live Search parameters for search-enabled models.

    Enables real-time web and X/Twitter search in chat completions.
    Cost: $0.025 per source used.

    Example:
        search_params = SearchParameters(
            mode="on",
            sources=[{"type": "x"}],  # Search X/Twitter only
            return_citations=True
        )
    """

    mode: Literal["off", "auto", "on"] = "auto"
    sources: Optional[List[SearchSource]] = None  # Default: web, news, x
    return_citations: bool = True
    from_date: Optional[str] = None  # YYYY-MM-DD format
    to_date: Optional[str] = None  # YYYY-MM-DD format
    max_search_results: int = 10  # Max sources (default 10, ~$0.26 with margin)


class SearchUsage(BaseModel):
    """Search usage information from xAI Live Search."""

    num_sources_used: Optional[int] = None


class CostEstimate(BaseModel):
    """
    Cost estimate from dry-run request.

    Returned when dry_run=True to show expected cost before executing.
    """

    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float

    def __str__(self) -> str:
        return f"💰 Estimated cost: ${self.estimated_cost_usd:.6f} ({self.model})"


class SpendingReport(BaseModel):
    """
    Spending report returned after each paid call.

    Shows what was spent on the current call and cumulative session total.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    session_total_usd: float
    session_calls: int

    def __str__(self) -> str:
        return (
            f"💸 This call: ${self.cost_usd:.6f} | "
            f"Session total: ${self.session_total_usd:.6f} ({self.session_calls} calls)"
        )


class ChatResponseWithCost(BaseModel):
    """
    Chat response with spending report attached.

    The content is in response.choices[0].message.content
    The spending report is in spending_report
    """

    response: ChatResponse
    spending_report: SpendingReport

    @property
    def content(self) -> str:
        """Shortcut to get response content."""
        return self.response.choices[0].message.content

    @property
    def cost(self) -> float:
        """Shortcut to get cost of this call."""
        return self.spending_report.cost_usd


# Smart routing types (Router Core integration)
RoutingProfile = Literal["free", "eco", "auto", "premium"]
RoutingTier = Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]
RoutingMethod = Literal["rules", "llm", "portfolio"]
RoutingTaskType = Literal[
    "chat",
    "extraction",
    "code_edit",
    "code_agent",
    "tool_agent",
    "tool_agent_parallel",
    "debug",
    "reasoning",
    "reasoning_mcq",
    "reasoning_math",
    "long_context",
    "vision",
]


class CandidateScore(BaseModel):
    """Per-candidate portfolio score breakdown, ordered with ``candidates``."""

    model: str
    score: float
    quality: float
    cost: float
    speed: float
    reliability: float


class RoutingDecision(BaseModel):
    """Result of smart routing decision."""

    model: str
    tier: RoutingTier
    confidence: float
    #: "portfolio" for the default V3 strategy, "rules" for the V2 rollback and
    #: the free profile.
    method: RoutingMethod
    reasoning: str
    cost_estimate: float
    baseline_cost: float
    savings: float  # 0-1 percentage
    fallbacks: List[str] = []  # remaining models in tier order, for runtime fallback
    # Router Core metadata — present when the portfolio strategy ran.
    candidates: List[str] = []  # ordered, capability-eligible; candidates[0] == model
    candidate_scores: List[CandidateScore] = []
    task_type: Optional[RoutingTaskType] = None
    router_version: Optional[Literal["v2-rules", "v3-portfolio"]] = None
    profile: Optional[Literal["auto", "eco", "premium", "agentic"]] = None
    agentic_score: Optional[float] = None


class SmartChatCompletionResponse(BaseModel):
    """
    Response from smart_chat_completion — the routed full completion.

    ``response`` is the ordinary ChatResponse (choices, usage, citations), so
    tool calls and structured output work exactly as with chat_completion.

    Example:
        result = client.smart_chat_completion([{"role": "user", "content": "hi"}])
        print(result.model)                     # the model routing picked
        print(result.response.choices[0].message.content)
        print(result.routing.task_type)         # 'chat'
    """

    response: ChatResponse
    model: str
    routing: RoutingDecision


class SmartChatResponse(BaseModel):
    """
    Response from smart_chat with routing information.

    Example:
        result = client.smart_chat("What is 2+2?")
        print(result.response)  # '4'
        print(result.model)     # 'google/gemini-2.5-flash'
        print(f"Saved {result.routing.savings * 100:.0f}%")
    """

    response: str
    model: str
    routing: RoutingDecision


# Standalone search response
class SearchResult(BaseModel):
    """Response from standalone search endpoint."""

    query: str
    summary: str
    citations: Optional[List[Dict[str, str]]] = None
    sources_used: Optional[int] = None
    model: Optional[str] = None


# Pyth-backed market data types (crypto, stocks, fx, commodity)
class PricePoint(BaseModel):
    """A single latest price quote from the Pyth network."""

    symbol: str
    price: float
    publish_time: Optional[int] = None  # Unix seconds
    confidence: Optional[float] = None
    feed_id: Optional[str] = None

    class Config:
        extra = "allow"


class PriceBar(BaseModel):
    """OHLC bar in a historical price series."""

    t: Optional[int] = None  # Bar open time (unix seconds)
    o: Optional[float] = None
    h: Optional[float] = None
    l: Optional[float] = None
    c: Optional[float] = None
    v: Optional[float] = None

    class Config:
        extra = "allow"


class PriceHistoryResponse(BaseModel):
    """Response from a historical price endpoint."""

    symbol: str
    resolution: Optional[str] = None
    bars: List[PriceBar] = []

    class Config:
        extra = "allow"


class SymbolListResponse(BaseModel):
    """Response from a market symbol list endpoint."""

    symbols: List[Dict[str, Any]] = []
    count: Optional[int] = None

    class Config:
        extra = "allow"


# Virtual Portrait enrollment types


class PortraitUsage(BaseModel):
    """How the enrolled portrait can be used."""

    compatible_models: List[str] = []
    how_to_use: Optional[str] = None

    class Config:
        extra = "allow"


class PortraitSettlement(BaseModel):
    """On-chain settlement of the enrollment payment."""

    success: bool
    tx_hash: Optional[str] = None
    network: Optional[str] = None

    class Config:
        extra = "allow"


class PortraitEnrollment(BaseModel):
    """Response from POST /v1/portrait/enroll."""

    object: str = "virtual_portrait"
    asset_id: str  # ta_xxxxxxxx — pass as real_face_asset_id on Seedance
    group_id: Optional[str] = None
    name: str
    image_url: str
    created_at: Optional[str] = None
    usage: Optional[PortraitUsage] = None
    price: Optional[Dict[str, Any]] = None  # {amount, currency}
    settlement: Optional[PortraitSettlement] = None

    class Config:
        extra = "allow"


class PortraitListItem(BaseModel):
    """One row in the wallet portrait list (GET /v1/wallet/<addr>/portraits)."""

    # Upstream uses camelCase here, keep matching for transparent ingestion.
    assetId: str
    groupId: Optional[str] = None
    name: Optional[str] = None
    imageUrl: Optional[str] = None
    createdAt: Optional[str] = None
    enrollmentTxHash: Optional[str] = None

    class Config:
        extra = "allow"


class PortraitList(BaseModel):
    """Response from GET /v1/wallet/<address>/portraits."""

    wallet: str
    portraits: List[PortraitListItem] = []
    count: Optional[int] = None

    class Config:
        extra = "allow"


# RealFace enrollment types
#
# RealFace registers a *real person's* face (vs. Virtual Portrait, which is an
# AI-generated character). Enrollment is a three-step flow: init (free) →
# the person completes a phone liveness check → enroll ($0.01 USDC). The
# resulting ta_xxxxxxxx asset id is interchangeable with a Virtual Portrait's
# on Seedance 2.0 / 2.0-fast, so RealFaceEnrollment reuses PortraitUsage and
# PortraitSettlement (identical shapes) rather than duplicating them.


class RealFaceInit(BaseModel):
    """Response from POST /v1/realface/init (free, rate-limited)."""

    object: str = "realface.init"
    group_id: str  # legacy_rf_xxxx — pass to status()/enroll()
    h5_link: str  # URL the real person scans on their phone for liveness
    status: Optional[str] = None  # pending_validation | active
    expires_in_seconds: Optional[int] = None  # H5 session validity (~120s)
    next_steps: Optional[Dict[str, Any]] = None
    refreshed: Optional[bool] = None  # True when re-issued for an existing group

    class Config:
        extra = "allow"


class RealFaceStatus(BaseModel):
    """Response from GET /v1/realface/status?groupId=… (free, rate-limited)."""

    object: str = "realface.status"
    group_id: str
    status: str  # pending_validation | active | …
    asset_count: Optional[int] = None
    ready_to_finalize: bool = False  # True once status == "active"

    class Config:
        extra = "allow"


class RealFaceEnrollment(BaseModel):
    """Response from POST /v1/realface/enroll ($0.01 USDC)."""

    object: str = "realface"
    asset_id: str  # ta_xxxxxxxx — pass as real_face_asset_id on Seedance
    group_id: Optional[str] = None
    byteplus_asset_id: Optional[str] = None
    name: str
    image_url: str
    created_at: Optional[str] = None
    usage: Optional[PortraitUsage] = None
    price: Optional[Dict[str, Any]] = None  # {amount, currency}
    settlement: Optional[PortraitSettlement] = None

    class Config:
        extra = "allow"


class RealFaceListItem(BaseModel):
    """One row in the wallet RealFace list (GET /v1/wallet/<addr>/realfaces)."""

    # Upstream uses camelCase here, keep matching for transparent ingestion.
    assetId: str
    groupId: Optional[str] = None
    name: Optional[str] = None
    imageUrl: Optional[str] = None
    createdAt: Optional[str] = None
    enrollmentTxHash: Optional[str] = None
    byteplusAssetId: Optional[str] = None

    class Config:
        extra = "allow"


class RealFaceList(BaseModel):
    """Response from GET /v1/wallet/<address>/realfaces."""

    wallet: str
    realfaces: List[RealFaceListItem] = []
    count: Optional[int] = None

    class Config:
        extra = "allow"
