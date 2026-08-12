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

Video generation:
    from blockrun_llm import VideoClient

    client = VideoClient()
    result = client.generate("a red apple slowly spinning on a wooden table")
    print(result.data[0].url)  # permanent MP4 URL

Text-to-speech (BlockRun Voice / ElevenLabs):
    from blockrun_llm import SpeechClient

    client = SpeechClient()
    result = client.generate("Welcome to BlockRun.", voice="sarah")
    print(result.data[0].url)  # audio URL

Multi-chain RPC (40+ chains, $0.002/call):
    from blockrun_llm import RpcClient

    client = RpcClient()
    block = client.call("ethereum", "eth_blockNumber")
    print(block.result)

Other Chains:
    - Solana (USDC): Use SolanaLLMClient (pip install blockrun-llm[solana])
"""

from __future__ import annotations

from .anthropic_client import AnthropicClient
from .cache import (
    clear_cache,
    export_cost_log_csv,
    export_cost_log_json,
    get_cost_log_summary,
)
from .client import (
    AsyncLLMClient,
    LLMClient,
    async_testnet_client,
    list_image_models,
    list_models,
    testnet_client,
)
from .image import ImageClient
from .music import MusicClient
from .phone import PhoneClient
from .portrait import PortraitClient
from .price import PriceClient
from .realface import RealFaceClient
from .rpc import NETWORK_ALIASES, SUPPORTED_NETWORKS, RpcClient
from .search import SearchClient
from .solana_client import AsyncSolanaLLMClient, SolanaLLMClient
from .solana_wallet import (
    create_solana_wallet,
    format_solana_wallet_migration_notice,
    generate_solana_qr_ascii,
    get_or_create_solana_wallet,
    get_solana_public_key,
    get_solana_usdc_balance,
    import_solana_wallet,
    list_discovered_solana_wallets,
    load_solana_wallet,
    open_solana_wallet_qr,
    scan_solana_wallets,
    setup_agent_solana_wallet,
)
from .speech import SpeechClient
from .surf import SurfClient
from .tx_log import TransactionLogger, decode_settlement_header, format_row
from .types import (
    APIError,
    AudioModel,
    AudioTrack,
    ChatChunkChoice,
    ChatChunkDelta,
    ChatChunkFunctionCall,
    ChatChunkToolCall,
    ChatCompletionChunk,
    ChatMessage,
    ChatResponse,
    ImageData,
    ImageModel,
    ImageResponse,
    Model,
    # Music / Audio types
    MusicResponse,
    NewsSearchSource,
    PaymentError,
    # Virtual Portrait types
    PortraitEnrollment,
    PortraitList,
    PortraitListItem,
    PortraitSettlement,
    PortraitUsage,
    PriceBar,
    PriceHistoryResponse,
    # Pyth market data types
    PricePoint,
    RealFaceEnrollment,
    # RealFace types
    RealFaceInit,
    RealFaceList,
    RealFaceListItem,
    RealFaceStatus,
    RetiredEndpointError,
    # Smart routing types
    RoutingDecision,
    RpcError,
    # Multi-chain RPC types
    RpcResponse,
    RssSearchSource,
    # Live Search types
    SearchParameters,
    # Standalone search
    SearchResult,
    SmartChatResponse,
    SpeechAudio,
    # Speech (TTS / sound effects) types
    SpeechResponse,
    SpendLimitError,
    SymbolListResponse,
    VideoClip,
    VideoModel,
    # Video types
    VideoResponse,
    WebSearchSource,
    XSearchSource,
)
from .video import VideoClient
from .voice import VoiceClient
from .wallet import (
    WALLET_DIR,
    WALLET_FILE,
    format_error_message,
    format_funding_message_compact,
    format_needs_funding_message,
    format_wallet_created_message,
    format_wallet_migration_notice,
    generate_wallet_qr_ascii,
    get_eip681_uri,
    get_or_create_wallet,
    get_payment_links,
    get_wallet_address,
    import_wallet,
    list_discovered_wallets,
    load_wallet,
    open_wallet_qr,
    save_wallet_qr,
    scan_wallets,
    setup_agent_wallet,  # Entry point for agents (auto-creates wallet)
    status,  # One-command verification
)
from .wallet import (
    create_wallet as generate_wallet,  # User-friendly alias
)

__version__ = "1.10.1"
__all__ = [
    "NETWORK_ALIASES",
    "SUPPORTED_NETWORKS",
    "WALLET_DIR",
    "WALLET_FILE",
    "APIError",
    "AnthropicClient",
    "AsyncLLMClient",
    "AsyncSolanaLLMClient",
    "AudioModel",
    "AudioTrack",
    "ChatChunkChoice",
    "ChatChunkDelta",
    "ChatChunkFunctionCall",
    "ChatChunkToolCall",
    "ChatCompletionChunk",
    "ChatMessage",
    "ChatResponse",
    "ImageClient",
    "ImageData",
    "ImageModel",
    "ImageResponse",
    "LLMClient",
    "Model",
    "MusicClient",
    "MusicResponse",
    "NewsSearchSource",
    "PaymentError",
    "PhoneClient",
    "PortraitClient",
    "PortraitEnrollment",
    "PortraitList",
    "PortraitListItem",
    "PortraitSettlement",
    "PortraitUsage",
    "PriceBar",
    "PriceClient",
    "PriceHistoryResponse",
    # Pyth market data types
    "PricePoint",
    "RealFaceClient",
    "RealFaceEnrollment",
    "RealFaceInit",
    "RealFaceList",
    "RealFaceListItem",
    "RealFaceStatus",
    "RetiredEndpointError",
    # Smart routing types
    "RoutingDecision",
    "RpcClient",
    "RpcError",
    # Multi-chain RPC types
    "RpcResponse",
    "RssSearchSource",
    "SearchClient",
    # Live Search types
    "SearchParameters",
    # Standalone search
    "SearchResult",
    "SmartChatResponse",
    "SolanaLLMClient",
    "SpeechAudio",
    "SpeechClient",
    "SpeechResponse",
    "SpendLimitError",
    "SurfClient",
    "SymbolListResponse",
    # Per-transaction log (opt-in, project-local ./log/)
    "TransactionLogger",
    "VideoClient",
    "VideoClip",
    "VideoModel",
    "VideoResponse",
    "VoiceClient",
    "WebSearchSource",
    "XSearchSource",
    "async_testnet_client",
    # Cache + billing utilities
    "clear_cache",
    "create_solana_wallet",
    "decode_settlement_header",
    "export_cost_log_csv",
    "export_cost_log_json",
    "format_error_message",
    "format_funding_message_compact",
    "format_needs_funding_message",
    "format_row",
    "format_solana_wallet_migration_notice",
    "format_wallet_created_message",
    "format_wallet_migration_notice",
    "generate_solana_qr_ascii",
    "generate_wallet",
    "generate_wallet_qr_ascii",
    "get_cost_log_summary",
    "get_eip681_uri",
    "get_or_create_solana_wallet",
    # Wallet utilities
    "get_or_create_wallet",
    "get_payment_links",
    "get_solana_public_key",
    "get_solana_usdc_balance",
    "get_wallet_address",
    "import_solana_wallet",
    "import_wallet",
    "list_discovered_solana_wallets",
    "list_discovered_wallets",
    "list_image_models",
    # Standalone functions (no wallet required)
    "list_models",
    "load_solana_wallet",
    "load_wallet",
    "open_solana_wallet_qr",
    "open_wallet_qr",
    "save_wallet_qr",
    "scan_solana_wallets",
    "scan_wallets",
    # Solana wallet utilities
    "setup_agent_solana_wallet",
    # Entry point for agents (auto-creates wallet)
    "setup_agent_wallet",
    "status",
    # Testnet convenience functions
    "testnet_client",
]
