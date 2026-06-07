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

from .client import (
    LLMClient,
    AsyncLLMClient,
    list_models,
    list_image_models,
    testnet_client,
    async_testnet_client,
)
from .anthropic_client import AnthropicClient
from .solana_client import AsyncSolanaLLMClient, SolanaLLMClient
from .image import ImageClient
from .music import MusicClient
from .speech import SpeechClient
from .video import VideoClient
from .portrait import PortraitClient
from .realface import RealFaceClient
from .voice import VoiceClient
from .phone import PhoneClient
from .surf import SurfClient
from .search import SearchClient
from .price import PriceClient
from .rpc import RpcClient, SUPPORTED_NETWORKS, NETWORK_ALIASES
from .types import (
    ChatMessage,
    ChatResponse,
    ChatCompletionChunk,
    ChatChunkChoice,
    ChatChunkDelta,
    Model,
    APIError,
    PaymentError,
    ImageResponse,
    ImageData,
    ImageModel,
    # Music / Audio types
    MusicResponse,
    AudioTrack,
    AudioModel,
    # Speech (TTS / sound effects) types
    SpeechResponse,
    SpeechAudio,
    # Video types
    VideoResponse,
    VideoClip,
    VideoModel,
    # Virtual Portrait types
    PortraitEnrollment,
    PortraitUsage,
    PortraitSettlement,
    PortraitList,
    PortraitListItem,
    # RealFace types
    RealFaceInit,
    RealFaceStatus,
    RealFaceEnrollment,
    RealFaceList,
    RealFaceListItem,
    # Live Search types
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
    # Pyth market data types
    PricePoint,
    PriceBar,
    PriceHistoryResponse,
    SymbolListResponse,
    # Multi-chain RPC types
    RpcResponse,
    RpcError,
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
from .cache import (
    clear_cache,
    export_cost_log_csv,
    export_cost_log_json,
    get_cost_log_summary,
)
from .tx_log import TransactionLogger, decode_settlement_header, format_row

__version__ = "1.1.0"
__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "AnthropicClient",
    "SolanaLLMClient",
    "AsyncSolanaLLMClient",
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
    "MusicClient",
    "SpeechClient",
    "VideoClient",
    "PortraitClient",
    "RealFaceClient",
    "VoiceClient",
    "PhoneClient",
    "SurfClient",
    "SearchClient",
    "PriceClient",
    "RpcClient",
    "SUPPORTED_NETWORKS",
    "NETWORK_ALIASES",
    "ChatMessage",
    "ChatResponse",
    "ChatCompletionChunk",
    "ChatChunkChoice",
    "ChatChunkDelta",
    "Model",
    "APIError",
    "PaymentError",
    "ImageResponse",
    "ImageData",
    "ImageModel",
    "MusicResponse",
    "AudioTrack",
    "AudioModel",
    "SpeechResponse",
    "SpeechAudio",
    "VideoResponse",
    "VideoClip",
    "VideoModel",
    "PortraitEnrollment",
    "PortraitUsage",
    "PortraitSettlement",
    "PortraitList",
    "PortraitListItem",
    "RealFaceInit",
    "RealFaceStatus",
    "RealFaceEnrollment",
    "RealFaceList",
    "RealFaceListItem",
    # Live Search types
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
    # Pyth market data types
    "PricePoint",
    "PriceBar",
    "PriceHistoryResponse",
    "SymbolListResponse",
    # Multi-chain RPC types
    "RpcResponse",
    "RpcError",
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
    # Cache + billing utilities
    "clear_cache",
    "get_cost_log_summary",
    "export_cost_log_csv",
    "export_cost_log_json",
    # Per-transaction log (opt-in, project-local ./log/)
    "TransactionLogger",
    "decode_settlement_header",
    "format_row",
]
