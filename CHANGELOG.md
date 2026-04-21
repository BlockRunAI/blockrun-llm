# Changelog

All notable changes to blockrun-llm will be documented in this file.

## 0.14.0

- **New `SearchClient`** — wraps `POST /v1/search` (standalone Grok Live Search). $0.025 per source + margin, 1–50 sources per call.
- **New `XClient`** — 13 methods mapping the `/v1/x/*` endpoints (user lookup/info/followers/following/verified-followers/tweets/mentions, tweet lookup/replies/thread, search, trending, articles/rising). Replaces orphaned `X*` types that had no caller.
- **New `PriceClient`** — Pyth-backed market data across crypto/fx/commodity (free) and usstock/stocks (paid) with `.price()`, `.history()`, `.list_symbols()`. Supports 12 global stock markets (us/hk/jp/kr/gb/de/fr/nl/ie/lu/cn/ca).
- `ChatMessage` gains optional `reasoning_content` and `thinking` fields for reasoning-capable models (DeepSeek Reasoner, Grok 4 / 4.20 reasoning).
- `ChatUsage` gains optional `cache_read_input_tokens` / `cache_creation_input_tokens` for Anthropic prompt caching telemetry.
- `Model` gains optional `billing_mode` (`paid`/`flat`/`free`), `flat_price`, `categories`, `hidden` so `list_models()` can surface full backend metadata.
- New market-data types: `PricePoint`, `PriceBar`, `PriceHistoryResponse`, `SymbolListResponse`.
- `VERSION` file synced to match `__init__.py`.

## 0.13.0

- **New `VideoClient`** — generate AI videos via `xai/grok-imagine-video` ($0.05/sec, 8s default).
- `VideoResponse`, `VideoClip`, `VideoModel` types added.
- Text-to-video and image-to-video supported; client blocks until polling completes (~30-120s).
- `ImageData` now exposes `source_url` and `backed_up` for gateway-mirrored assets.
- Grok Imagine image models (`xai/grok-imagine-image`, `-pro`) routable via `ImageClient`.
- Grok 4.20 chat models (`xai/grok-4.20-reasoning`, `-non-reasoning`, `-multi-agent`) routable via the chat API.

## 0.11.0

- 43+ models supported
- Base and Solana chain payments
- x402 v2 protocol
- Image generation support
- Anthropic-compatible client
- Smart model routing
- Response caching
