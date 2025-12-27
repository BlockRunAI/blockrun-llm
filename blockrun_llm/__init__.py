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
"""

from .client import LLMClient, AsyncLLMClient
from .types import ChatMessage, ChatResponse, Model, APIError, PaymentError

__version__ = "0.1.0"
__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "ChatMessage",
    "ChatResponse",
    "Model",
    "APIError",
    "PaymentError",
]
