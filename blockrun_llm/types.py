"""Type definitions for BlockRun LLM SDK."""

from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_sources_used: Optional[int] = None  # xAI Live Search sources used


class ChatResponse(BaseModel):
    """Response from chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None
    citations: Optional[List[str]] = None  # xAI Live Search citation URLs


class Model(BaseModel):
    """Available model information."""

    id: str
    name: str
    provider: str
    description: str
    input_price: float  # Per 1M tokens
    output_price: float  # Per 1M tokens
    context_window: int
    max_output: int
    available: bool = True


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


# xAI Live Search types (for Grok models)
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
    xAI Live Search parameters for Grok models.

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
    breakdown: Dict[str, Any] = Field(default_factory=dict)

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
    breakdown: Dict[str, Any] = Field(default_factory=dict)

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
