# Changelog

All notable changes to blockrun-llm will be documented in this file.

## 0.16.1

- **`ImageClient` default timeout 120s → 200s.** The gateway's per-call OpenAI
  timeout for `gpt-image-2` was bumped to 180s server-side (it routinely takes
  ~120-180s at 1536x1024 and larger), so the SDK's old 120s default was cutting
  the request before the server had a chance to return. New default leaves
  ~20s of buffer above the server cap. Existing users passing an explicit
  `timeout=` are unaffected.

## 0.16.0

- **VideoClient switches to async submit+poll**. Upstream `/v1/videos/generations`
  moved from sync to async on 2026-04-23 (submit returns a job id; client polls
  until completion). Public signature of `VideoClient.generate(...)` is unchanged
  — still blocks until the video is ready and returns `VideoResponse` with the
  MP4 URL and tx hash. Internally the client now signs once, submits, and
  replays the same signature on GET polls every 5s until upstream completes.
  Settlement only fires on the first completed poll, so upstream failure or
  budget exhaustion = zero charge.
- Added `budget_seconds` parameter to `generate()` (default 300s) to cap the
  polling window.
- Bumped advertised `max_timeout_seconds` on video requests from 300s to 600s
  so the signed auth stays valid across the full polling window.

## 0.15.0

- **New image model: `openai/gpt-image-2`** (ChatGPT Images 2.0). Reasoning-driven generation with multilingual text rendering + character consistency. Pricing: $0.06 for 1024² / $0.12 for 1536×1024 or 1024×1536. Supports both `client.generate()` and `client.edit()` via the `/v1/images/image2image` endpoint.
- **New video models: 3 ByteDance Seedance variants** on `VideoClient`:
  - `bytedance/seedance-1.5-pro` — $0.03/sec, 720p, 5s default (up to 10s).
  - `bytedance/seedance-2.0-fast` — $0.15/sec, ~60-80s generation, sweet-spot price/quality.
  - `bytedance/seedance-2.0` — $0.30/sec, 720p Pro quality.
  All support text-to-video and image-to-video. Pass the model ID to `VideoClient.generate(..., model=...)`.
- README Image/Video sections list new models; image editing section notes `gpt-image-1` and `gpt-image-2` as supported.
- Also: `pyproject.toml` version was stuck at 0.13.0 despite `__version__` saying 0.14.1 (prevented PyPI publishes from shipping the NVIDIA refresh). Both now aligned at 0.15.0.

## 0.14.1

- **NVIDIA free-tier refresh (backend 2026-04-21).** Router updated to point at the current survivors + the two new models: `nvidia/qwen3-next-80b-a3b-thinking` (reasoning flagship, 116 tok/s) and `nvidia/mistral-small-4-119b` (fastest free chat, 114 tok/s).
- Retired IDs no longer referenced by `router.py`: `nvidia/nemotron-super-49b`, `nvidia/nemotron-ultra-253b`, `nvidia/mistral-large-3-675b`. The backend still redirects them, but offline routing now points at the canonical successors (`nvidia/qwen3-next-80b-a3b-thinking`, `nvidia/mistral-small-4-119b`, `nvidia/llama-4-maverick`, `nvidia/glm-4.7`).
- AUTO / ECO `SIMPLE` primaries switched from `nvidia/kimi-k2.5` (retired) to `moonshot/kimi-k2.5` — backend redirect still works, but the router now references the canonical target.
- README NVIDIA table refreshed (8 visible models + `moonshot/kimi-k2.5`).

## 0.14.0

- **New `SearchClient`** — wraps `POST /v1/search` (standalone Grok Live Search). $0.025 per source + margin, 1–50 sources per call.
- **New `XClient`** — 13 methods mapping the `/v1/x/*` endpoints (user lookup/info/followers/following/verified-followers/tweets/mentions, tweet lookup/replies/thread, search, trending, articles/rising). Replaces orphaned `X*` types that had no caller.
- **New `PriceClient`** — Pyth-backed market data with `.price()`, `.history()`, `.list_symbols()`. Crypto, FX and commodity are fully free (price + history + list); stocks across 12 markets (us/hk/jp/kr/gb/de/fr/nl/ie/lu/cn/ca) and the `usstock` legacy alias charge for price + history, list stays free. The client handles both paths transparently.
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
