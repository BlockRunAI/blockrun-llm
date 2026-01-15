"""
BlockRun LLM SDK - Pay-per-request AI via x402 on Base

**BlockRun assumes Claude Code as the agent runtime.**

Usage:
    from blockrun_llm import LLMClient

    client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY from env
    response = client.chat("openai/gpt-4o", "Hello!")
    print(response)

    # Check spending
    spending = client.get_spending()
    print(f"Spent ${spending['total_usd']:.4f} across {spending['calls']} calls")

Async usage:
    from blockrun_llm import AsyncLLMClient

    async with AsyncLLMClient() as client:
        response = await client.chat("openai/gpt-4o", "Hello!")
        print(response)

Image generation:
    from blockrun_llm import ImageClient

    client = ImageClient()
    result = client.generate("A cute cat wearing a space helmet")
    print(result.data[0].url)
"""

from .client import LLMClient, AsyncLLMClient, list_models, list_image_models
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
    # xAI Live Search types
    SearchParameters,
    WebSearchSource,
    XSearchSource,
    NewsSearchSource,
    RssSearchSource,
)
from .wallet import (
    get_or_create_wallet,
    get_wallet_address,
    format_wallet_created_message,
    format_needs_funding_message,
    format_funding_message_compact,
    format_error_message,
    generate_wallet_qr_ascii,
    get_payment_links,
    get_eip681_uri,
    save_wallet_qr,
    open_wallet_qr,
    load_wallet,
    create_wallet as generate_wallet,  # User-friendly alias
    WALLET_FILE,
    WALLET_DIR,
)

__version__ = "0.3.0"
__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    # Standalone functions (no wallet required)
    "list_models",
    "list_image_models",
    "ImageClient",
    "ChatMessage",
    "ChatResponse",
    "Model",
    "APIError",
    "PaymentError",
    "ImageResponse",
    "ImageData",
    "ImageModel",
    # xAI Live Search types
    "SearchParameters",
    "WebSearchSource",
    "XSearchSource",
    "NewsSearchSource",
    "RssSearchSource",
    # Wallet utilities
    "get_or_create_wallet",
    "get_wallet_address",
    "generate_wallet",
    "format_wallet_created_message",
    "format_needs_funding_message",
    "format_funding_message_compact",
    "format_error_message",
    "generate_wallet_qr_ascii",
    "get_payment_links",
    "get_eip681_uri",
    "save_wallet_qr",
    "open_wallet_qr",
    "load_wallet",
    "WALLET_FILE",
    "WALLET_DIR",
]
