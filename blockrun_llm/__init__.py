"""
BlockRun LLM SDK - Pay-per-request AI via x402 on Base

Usage:
    from blockrun_llm import LLMClient

    client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY from env
    response = client.chat("gpt-4o", "Hello!")
    print(response)

Async usage:
    from blockrun_llm import AsyncLLMClient

    async with AsyncLLMClient() as client:
        response = await client.chat("gpt-4o", "Hello!")
        print(response)

Image generation:
    from blockrun_llm import ImageClient

    client = ImageClient()
    result = client.generate("A cute cat wearing a space helmet")
    print(result.data[0].url)
"""

from .client import LLMClient, AsyncLLMClient
from .image import ImageClient
from .types import (
    ChatMessage,
    ChatResponse,
    Model,
    APIError,
    PaymentError,
    ImageResponse,
    ImageData,
    ImageModel,
)

__version__ = "0.2.0"
__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "ImageClient",
    "ChatMessage",
    "ChatResponse",
    "Model",
    "APIError",
    "PaymentError",
    "ImageResponse",
    "ImageData",
    "ImageModel",
]
