"""Type definitions for BlockRun LLM SDK."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel


# Tool calling types (OpenAI compatible)
class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


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
ToolChoiceFunction = dict[str, Any]  # {"type": "function", "function": {"name": "..."}}
ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceFunction]


class ChatMessage(BaseModel):
    """A single chat message.

    Passthrough: the named fields below are conveniences; any other field the
    gateway forwards (e.g. ``annotations``, ``audio``, future OpenAI additions)
    is preserved via ``extra = "allow"`` rather than silently dropped.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None  # For tool messages
    tool_call_id: str | None = None  # For tool result messages
    tool_calls: list[ToolCall] | None = None  # For assistant messages with tool calls
    # Extended fields returned by reasoning-capable upstream providers
    # (DeepSeek Reasoner, Grok 4 reasoning, xAI multi-agent, etc.).
    # Backend strips these from inbound requests but may forward them on the
    # response side, so we accept them as optional.
    reasoning_content: str | None = None
    thinking: str | None = None

    class Config:
        extra = "allow"


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None  # OpenAI-compatible; upstreams may add new values

    class Config:
        extra = "allow"


class ChatUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_sources_used: int | None = None  # xAI Live Search sources used
    # Anthropic prompt caching — populated on anthropic/* models when cache
    # headers are sent. Reads are cheaper; writes incur a one-time surcharge.
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None

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
    choices: list[ChatChoice]
    usage: ChatUsage | None = None
    citations: list[str] | None = None  # xAI Live Search citation URLs

    # Real x402 charge for THIS call, in USD — the exact amount debited from the
    # wallet (0.0 for free / cached calls). This is the authoritative number to
    # bill/track against; token-count × list-price estimates do NOT match it
    # because the gateway price carries a per-call floor + margin. Populated by
    # the client on every chat completion. ``settlement`` carries the decoded
    # on-chain receipt (tx hash / micro-USDC / network) when the facilitator
    # returned an X-PAYMENT-RESPONSE header.
    cost_usd: float | None = None
    settlement: dict[str, Any] | None = None

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

    name: str | None = None
    arguments: str | None = None

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

    index: int | None = None
    id: str | None = None
    # Kept as a free-form ``str`` (not ``Literal["function"]``) so an upstream
    # that streams a non-"function" tool type can't fail validation and re-trigger
    # the very ``model_construct`` fallback this lenient type exists to avoid.
    type: str | None = None
    function: ChatChunkFunctionCall | None = None

    class Config:
        extra = "allow"


class ChatChunkDelta(BaseModel):
    """Incremental ``message`` delta sent over SSE.

    Any field may be absent in a given chunk — ``role`` typically only on the
    first, ``content`` on body chunks, ``tool_calls`` when the model decides
    to call a tool. ``reasoning_content`` / ``thinking`` appear on
    reasoning-capable upstreams.
    """

    role: Literal["system", "user", "assistant", "tool"] | None = None
    content: str | None = None
    tool_calls: list[ChatChunkToolCall] | None = None
    reasoning_content: str | None = None
    thinking: str | None = None

    class Config:
        extra = "allow"


class ChatChunkChoice(BaseModel):
    """One choice within a streaming chunk."""

    index: int
    delta: ChatChunkDelta
    finish_reason: str | None = None  # OpenAI-compatible; upstreams may add new values

    class Config:
        extra = "allow"


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk emitted by ``/v1/chat/completions`` when stream=True."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
    # Usage is populated only on the final chunk for providers that support it
    # (some upstreams omit it entirely — callers must tolerate ``None``).
    usage: ChatUsage | None = None
    citations: list[str] | None = None  # xAI Live Search citation URLs (final chunk only)

    class Config:
        extra = "allow"


def stream_choice_content(choice: Any) -> str | None:
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


def stream_choice_finish_reason(choice: Any) -> str | None:
    """``finish_reason`` from a streaming choice, tolerant of a raw dict choice."""
    if isinstance(choice, dict):
        return choice.get("finish_reason")
    return getattr(choice, "finish_reason", None)


def chunk_meta(chunk: Any) -> tuple[str | None, str | None, int | None]:
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


def chunk_usage_dict(chunk: Any) -> dict[str, Any] | None:
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
    billing_mode: Literal["paid", "flat", "free"] | None = None
    flat_price: float | None = None
    categories: list[str] | None = None  # e.g. ["chat","reasoning","coding","vision"]
    hidden: bool | None = None  # True for deprecated/superseded models still routable


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
    accepts: list[PaymentRequirement]


class BlockrunError(Exception):
    """Base exception for BlockRun SDK."""


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
        status_code: int | None = None,
        response: dict | None = None,
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

    def __init__(self, message: str, status_code: int, response: dict | None = None):
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
    source_url: str | None = None
    backed_up: bool | None = None
    revised_prompt: str | None = None


class ImageResponse(BaseModel):
    """Response from image generation."""

    created: int
    data: list[ImageData]


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
    duration_seconds: float | None = None
    lyrics: str | None = None


class MusicResponse(BaseModel):
    """Response from music generation."""

    created: int
    model: str
    data: list[AudioTrack]
    txHash: str | None = None


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
    format: str | None = None
    characters: int | None = None
    credits: float | None = None


class SpeechResponse(BaseModel):
    """Response from speech synthesis or sound-effect generation."""

    created: int
    model: str
    data: list[SpeechAudio]
    txHash: str | None = None


# Multi-chain RPC types


class RpcError(BaseModel):
    """A JSON-RPC 2.0 error object."""

    code: int | None = None
    message: str | None = None
    data: Any | None = None


class RpcResponse(BaseModel):
    """Response from a multi-chain JSON-RPC call (/v1/rpc/{network}).

    Standard JSON-RPC 2.0 envelope plus BlockRun gateway metadata pulled
    from response headers (X-Network / X-Cache / X-Payment-Receipt).
    """

    jsonrpc: str | None = None
    id: str | int | None = None
    result: Any | None = None
    error: RpcError | None = None
    # Gateway metadata (response headers)
    network: str | None = None  # canonical network key, e.g. "ethereum"
    cache_hit: bool = False  # served from the gateway's method-aware cache
    tx_hash: str | None = None  # x402 settlement tx (single calls)


# Video generation types


class VideoClip(BaseModel):
    """A single generated video clip."""

    url: str  # Permanent blockrun-hosted URL (falls back to upstream if backup fails)
    source_url: str | None = None  # Original upstream URL (e.g. vidgen.x.ai)
    duration_seconds: int | None = None
    request_id: str | None = None  # Upstream provider's request id (xAI)
    backed_up: bool | None = None


class VideoResponse(BaseModel):
    """Response from video generation."""

    created: int
    model: str
    data: list[VideoClip]
    txHash: str | None = None


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
    country: str | None = None  # ISO alpha-2 country code
    excluded_websites: list[str] | None = None  # Max 5 websites
    allowed_websites: list[str] | None = None  # Max 5 websites (mutually exclusive with excluded)
    safe_search: bool = True


class XSearchSource(BaseModel):
    """X/Twitter search source configuration."""

    type: Literal["x"] = "x"
    included_x_handles: list[str] | None = None  # Max 10 handles
    excluded_x_handles: list[str] | None = None  # Max 10 handles
    post_favorite_count: int | None = None  # Minimum favorites threshold
    post_view_count: int | None = None  # Minimum views threshold


class NewsSearchSource(BaseModel):
    """News search source configuration."""

    type: Literal["news"] = "news"
    country: str | None = None  # ISO alpha-2 country code
    excluded_websites: list[str] | None = None  # Max 5 websites
    allowed_websites: list[str] | None = None  # Max 5 websites
    safe_search: bool = True


class RssSearchSource(BaseModel):
    """RSS feed search source configuration."""

    type: Literal["rss"] = "rss"
    links: list[str]  # RSS feed URLs (currently supports one)


SearchSource = Union[
    WebSearchSource, XSearchSource, NewsSearchSource, RssSearchSource, dict[str, Any]
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
    sources: list[SearchSource] | None = None  # Default: web, news, x
    return_citations: bool = True
    from_date: str | None = None  # YYYY-MM-DD format
    to_date: str | None = None  # YYYY-MM-DD format
    max_search_results: int = 10  # Max sources (default 10, ~$0.26 with margin)


class SearchUsage(BaseModel):
    """Search usage information from xAI Live Search."""

    num_sources_used: int | None = None


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


# Smart routing types (ClawRouter integration)
RoutingProfile = Literal["free", "eco", "auto", "premium"]
RoutingTier = Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]


class RoutingDecision(BaseModel):
    """Result of smart routing decision."""

    model: str
    tier: RoutingTier
    confidence: float
    method: Literal["rules"]
    reasoning: str
    cost_estimate: float
    baseline_cost: float
    savings: float  # 0-1 percentage
    fallbacks: list[str] = []  # remaining models in tier order, for runtime fallback


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
    citations: list[dict[str, str]] | None = None
    sources_used: int | None = None
    model: str | None = None


# Pyth-backed market data types (crypto, stocks, fx, commodity)
class PricePoint(BaseModel):
    """A single latest price quote from the Pyth network."""

    symbol: str
    price: float
    publish_time: int | None = None  # Unix seconds
    confidence: float | None = None
    feed_id: str | None = None

    class Config:
        extra = "allow"


class PriceBar(BaseModel):
    """OHLC bar in a historical price series."""

    t: int | None = None  # Bar open time (unix seconds)
    o: float | None = None
    h: float | None = None
    l: float | None = None
    c: float | None = None
    v: float | None = None

    class Config:
        extra = "allow"


class PriceHistoryResponse(BaseModel):
    """Response from a historical price endpoint."""

    symbol: str
    resolution: str | None = None
    bars: list[PriceBar] = []

    class Config:
        extra = "allow"


class SymbolListResponse(BaseModel):
    """Response from a market symbol list endpoint."""

    symbols: list[dict[str, Any]] = []
    count: int | None = None

    class Config:
        extra = "allow"


# Virtual Portrait enrollment types


class PortraitUsage(BaseModel):
    """How the enrolled portrait can be used."""

    compatible_models: list[str] = []
    how_to_use: str | None = None

    class Config:
        extra = "allow"


class PortraitSettlement(BaseModel):
    """On-chain settlement of the enrollment payment."""

    success: bool
    tx_hash: str | None = None
    network: str | None = None

    class Config:
        extra = "allow"


class PortraitEnrollment(BaseModel):
    """Response from POST /v1/portrait/enroll."""

    object: str = "virtual_portrait"
    asset_id: str  # ta_xxxxxxxx — pass as real_face_asset_id on Seedance
    group_id: str | None = None
    name: str
    image_url: str
    created_at: str | None = None
    usage: PortraitUsage | None = None
    price: dict[str, Any] | None = None  # {amount, currency}
    settlement: PortraitSettlement | None = None

    class Config:
        extra = "allow"


class PortraitListItem(BaseModel):
    """One row in the wallet portrait list (GET /v1/wallet/<addr>/portraits)."""

    # Upstream uses camelCase here, keep matching for transparent ingestion.
    assetId: str
    groupId: str | None = None
    name: str | None = None
    imageUrl: str | None = None
    createdAt: str | None = None
    enrollmentTxHash: str | None = None

    class Config:
        extra = "allow"


class PortraitList(BaseModel):
    """Response from GET /v1/wallet/<address>/portraits."""

    wallet: str
    portraits: list[PortraitListItem] = []
    count: int | None = None

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
    status: str | None = None  # pending_validation | active
    expires_in_seconds: int | None = None  # H5 session validity (~120s)
    next_steps: dict[str, Any] | None = None
    refreshed: bool | None = None  # True when re-issued for an existing group

    class Config:
        extra = "allow"


class RealFaceStatus(BaseModel):
    """Response from GET /v1/realface/status?groupId=… (free, rate-limited)."""

    object: str = "realface.status"
    group_id: str
    status: str  # pending_validation | active | …
    asset_count: int | None = None
    ready_to_finalize: bool = False  # True once status == "active"

    class Config:
        extra = "allow"


class RealFaceEnrollment(BaseModel):
    """Response from POST /v1/realface/enroll ($0.01 USDC)."""

    object: str = "realface"
    asset_id: str  # ta_xxxxxxxx — pass as real_face_asset_id on Seedance
    group_id: str | None = None
    byteplus_asset_id: str | None = None
    name: str
    image_url: str
    created_at: str | None = None
    usage: PortraitUsage | None = None
    price: dict[str, Any] | None = None  # {amount, currency}
    settlement: PortraitSettlement | None = None

    class Config:
        extra = "allow"


class RealFaceListItem(BaseModel):
    """One row in the wallet RealFace list (GET /v1/wallet/<addr>/realfaces)."""

    # Upstream uses camelCase here, keep matching for transparent ingestion.
    assetId: str
    groupId: str | None = None
    name: str | None = None
    imageUrl: str | None = None
    createdAt: str | None = None
    enrollmentTxHash: str | None = None
    byteplusAssetId: str | None = None

    class Config:
        extra = "allow"


class RealFaceList(BaseModel):
    """Response from GET /v1/wallet/<address>/realfaces."""

    wallet: str
    realfaces: list[RealFaceListItem] = []
    count: int | None = None

    class Config:
        extra = "allow"
