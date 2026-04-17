# Changelog

All notable changes to blockrun-llm will be documented in this file.

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
