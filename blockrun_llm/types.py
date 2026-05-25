"""Type definitions for BlockRun LLM SDK."""

from typing import List, Optional, Literal, Dict, Any, Union
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
    """A single chat message."""

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


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None


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


class ChatResponse(BaseModel):
    """Response from chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None
    citations: Optional[List[str]] = None  # xAI Live Search citation URLs


# ---------------------------------------------------------------------------
# Streaming (SSE) chunk types — OpenAI Chat Completions chunk schema.
#
# Backend emits ``data: <json>\n\n`` lines terminated by ``data: [DONE]\n\n``.
# First chunk's delta has ``role="assistant"``; subsequent chunks fill
# ``content``; final chunk carries ``finish_reason`` and optionally ``usage``.
# ---------------------------------------------------------------------------


class ChatChunkDelta(BaseModel):
    """Incremental ``message`` delta sent over SSE.

    Any field may be absent in a given chunk — ``role`` typically only on the
    first, ``content`` on body chunks, ``tool_calls`` when the model decides
    to call a tool. ``reasoning_content`` / ``thinking`` appear on
    reasoning-capable upstreams.
    """

    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None
    thinking: Optional[str] = None


class ChatChunkChoice(BaseModel):
    """One choice within a streaming chunk."""

    index: int
    delta: ChatChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None


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

    pass


class PaymentError(BlockrunError):
    """Payment-related error."""

    pass


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
    fallbacks: List[str] = []  # remaining models in tier order, for runtime fallback


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


# X/Twitter types
class XUser(BaseModel):
    """X/Twitter user profile."""

    id: str
    userName: str
    name: str
    profilePicture: Optional[str] = None
    description: Optional[str] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    isBlueVerified: Optional[bool] = None
    verifiedType: Optional[str] = None
    location: Optional[str] = None
    joined: Optional[str] = None


class XUserLookupResponse(BaseModel):
    """Response from X/Twitter user lookup."""

    users: List[XUser]
    not_found: Optional[List[str]] = None
    total_requested: Optional[int] = None
    total_found: Optional[int] = None


class XFollower(BaseModel):
    """X/Twitter follower/following profile."""

    id: str
    name: Optional[str] = None
    screen_name: Optional[str] = None
    userName: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    protected: Optional[bool] = None
    verified: Optional[bool] = None
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    favourites_count: Optional[int] = None
    statuses_count: Optional[int] = None
    created_at: Optional[str] = None
    profile_image_url_https: Optional[str] = None
    can_dm: Optional[bool] = None


class XFollowersResponse(BaseModel):
    """Response from X/Twitter followers endpoint."""

    followers: List[XFollower]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None
    username: Optional[str] = None


class XFollowingsResponse(BaseModel):
    """Response from X/Twitter followings endpoint."""

    followings: List[XFollower]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None
    username: Optional[str] = None


class XUserInfoResponse(BaseModel):
    """Response from X/Twitter single user info endpoint."""

    data: Dict[str, Any]
    username: Optional[str] = None


class XVerifiedFollowersResponse(BaseModel):
    """Response from X/Twitter verified followers endpoint."""

    followers: List[XFollower]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None


class XTweet(BaseModel):
    """X/Twitter tweet."""

    id: str
    text: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[Dict[str, Any]] = None
    favorite_count: Optional[int] = None
    retweet_count: Optional[int] = None
    reply_count: Optional[int] = None
    view_count: Optional[int] = None
    lang: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    media: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class XTweetsResponse(BaseModel):
    """Response from X/Twitter user tweets endpoint."""

    tweets: List[XTweet]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None


class XMentionsResponse(BaseModel):
    """Response from X/Twitter user mentions endpoint."""

    tweets: List[XTweet]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None
    username: Optional[str] = None


class XTweetLookupResponse(BaseModel):
    """Response from X/Twitter tweet lookup (batch) endpoint."""

    tweets: List[XTweet]
    not_found: Optional[List[str]] = None
    total_requested: Optional[int] = None
    total_found: Optional[int] = None


class XTweetRepliesResponse(BaseModel):
    """Response from X/Twitter tweet replies endpoint."""

    replies: List[XTweet]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None


class XTweetThreadResponse(BaseModel):
    """Response from X/Twitter tweet thread endpoint."""

    tweets: List[XTweet]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None


class XSearchResponse(BaseModel):
    """Response from X/Twitter search endpoint."""

    tweets: List[XTweet]
    has_next_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    total_returned: Optional[int] = None


class XTrendingResponse(BaseModel):
    """Response from X/Twitter trending topics endpoint."""

    data: Dict[str, Any]


class XArticlesRisingResponse(BaseModel):
    """Response from X/Twitter rising articles endpoint."""

    data: Dict[str, Any]


class XAuthorAnalyticsResponse(BaseModel):
    """Response from X/Twitter author analytics endpoint."""

    data: Dict[str, Any]
    handle: Optional[str] = None


class XCompareAuthorsResponse(BaseModel):
    """Response from X/Twitter compare authors endpoint."""

    data: Dict[str, Any]


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
    l: Optional[float] = None  # noqa: E741 — Pyth bar field name
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
