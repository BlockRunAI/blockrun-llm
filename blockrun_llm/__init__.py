"""
BlockRun LLM SDK - Pay-per-request AI via x402 on Base (USDC)

For developers (bring your own wallet):
    from blockrun_llm import LLMClient

    client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY from env
    response = client.chat("openai/gpt-5.2", "Hello!")
    print(response)

For agents (Claude Code skills, auto-creates wallet):
    from blockrun_llm import setup_agent_wallet

    client = setup_agent_wallet()  # Auto-creates wallet, shows QR
    response = client.chat("openai/gpt-5.2", "Hello!")
    print(response)

Async usage:
    from blockrun_llm import AsyncLLMClient

    async with AsyncLLMClient() as client:
        response = await client.chat("openai/gpt-5.2", "Hello!")
        print(response)

Image generation:
    from blockrun_llm import ImageClient

    client = ImageClient()
    result = client.generate("A cute cat wearing a space helmet")
    print(result.data[0].url)

Other Chains:
    - XRPL (RLUSD): Use blockrun-llm-xrpl (pip install blockrun-llm-xrpl)
    - Solana (USDC): Use SolanaLLMClient (pip install blockrun-llm[solana])
"""

from .client import (
    LLMClient,
    AsyncLLMClient,
    list_models,
    list_image_models,
    testnet_client,
    async_testnet_client,
)
from .anthropic_client import AnthropicClient
from .solana_client import SolanaLLMClient
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
    # Smart routing types
    RoutingDecision,
    SmartChatResponse,
    # Standalone search
    SearchResult,
    # X/Twitter types
    XUser,
    XUserLookupResponse,
    XFollower,
    XFollowersResponse,
    XFollowingsResponse,
    XUserInfoResponse,
    XVerifiedFollowersResponse,
    XTweet,
    XTweetsResponse,
    XMentionsResponse,
    XTweetLookupResponse,
    XTweetRepliesResponse,
    XTweetThreadResponse,
    XSearchResponse,
    XTrendingResponse,
    XArticlesRisingResponse,
    XAuthorAnalyticsResponse,
    XCompareAuthorsResponse,
)
from .wallet import (
    setup_agent_wallet,  # Entry point for agents (auto-creates wallet)
    status,  # One-command verification
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
from .solana_wallet import (
    setup_agent_solana_wallet,
    get_solana_usdc_balance,
    generate_solana_qr_ascii,
    open_solana_wallet_qr,
    get_or_create_solana_wallet,
    create_solana_wallet,
    load_solana_wallet,
    get_solana_public_key,
)
from .cache import clear_cache, get_cost_log_summary

__version__ = "0.8.1"
__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "AnthropicClient",
    "SolanaLLMClient",
    # Testnet convenience functions
    "testnet_client",
    "async_testnet_client",
    # Entry point for agents (auto-creates wallet)
    "setup_agent_wallet",
    "status",
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
    # Smart routing types
    "RoutingDecision",
    "SmartChatResponse",
    # Standalone search
    "SearchResult",
    # X/Twitter types
    "XUser",
    "XUserLookupResponse",
    "XFollower",
    "XFollowersResponse",
    "XFollowingsResponse",
    "XUserInfoResponse",
    "XVerifiedFollowersResponse",
    "XTweet",
    "XTweetsResponse",
    "XMentionsResponse",
    "XTweetLookupResponse",
    "XTweetRepliesResponse",
    "XTweetThreadResponse",
    "XSearchResponse",
    "XTrendingResponse",
    "XArticlesRisingResponse",
    "XAuthorAnalyticsResponse",
    "XCompareAuthorsResponse",
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
    # Solana wallet utilities
    "setup_agent_solana_wallet",
    "get_solana_usdc_balance",
    "generate_solana_qr_ascii",
    "open_solana_wallet_qr",
    "get_or_create_solana_wallet",
    "create_solana_wallet",
    "load_solana_wallet",
    "get_solana_public_key",
    # Cache utilities
    "clear_cache",
    "get_cost_log_summary",
]
