"""Type definitions for BlockRun LLM SDK."""

from typing import List, Optional, Literal
from pydantic import BaseModel


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


class ChatResponse(BaseModel):
    """Response from chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None


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
