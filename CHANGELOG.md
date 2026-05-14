# Changelog

All notable changes to blockrun-llm will be documented in this file.

## 0.23.0 — 2026-05-14

### New
- **Custom Solana RPC support via env vars** — Solana clients
  (`SolanaLLMClient` + `AsyncSolanaLLMClient`) now resolve their RPC
  endpoint from explicit args, then these env vars, then the public
  default:
  - ``SOLANA_RPC_URL`` — the JSON-RPC endpoint URL. Use this when
    your provider embeds auth in the URL (Helius style:
    ``https://mainnet.helius-rpc.com/?api-key=...``).
  - ``SOLANA_RPC_API_KEY`` — convenience shortcut for the common
    ``x-api-key: <value>`` header style (Tatum, some Triton tiers).
    Internally becomes ``SOLANA_RPC_HEADERS='{"x-api-key":"..."}'``.
  - ``SOLANA_RPC_HEADERS`` — JSON dict for arbitrary header auth
    (``'{"x-api-key":"...","x-rate-tier":"pro"}'``).

  This unblocks production traffic — the public
  ``api.mainnet-beta.solana.com`` rate-limits aggressively
  (~10-40 RPS) and a partner deploying behind a free-tier Helius
  key was seeing failures at 30-100 concurrent requests.

  Previously the only way to switch RPCs was to edit
  ``_adapter.py`` source; that change is lost on every upgrade.
  Env vars make this idempotent across releases.

- **Header-auth Solana gateways (Tatum, header-only Triton) now
  work** — the upstream x402 SDK's
  ``register_exact_svm_client`` only takes ``rpc_url``, not custom
  headers, so the underlying ``solana.rpc.api.Client`` was always
  built without ``extra_headers``. We now pre-populate the SVM
  scheme's client cache with a properly-configured ``SolanaClient``
  before any payment payload is constructed.

### Configuration example

For Tatum (header-auth):
```bash
export SOLANA_RPC_URL=https://solana-mainnet.gateway.tatum.io
export SOLANA_RPC_API_KEY=t-...
```

For Helius (URL-embedded auth):
```bash
export SOLANA_RPC_URL='https://mainnet.helius-rpc.com/?api-key=...'
```

For arbitrary header schemes:
```bash
export SOLANA_RPC_URL=https://your.gateway/...
export SOLANA_RPC_HEADERS='{"x-api-key":"...","x-rate-tier":"pro"}'
```

### Verified e2e
- Live test against ``solana-mainnet.gateway.tatum.io`` with
  ``x-api-key`` header — the signing pipeline (blockhash fetch +
  TransferChecked tx construction + signature) completed
  end-to-end through Tatum and submitted the payment to BlockRun's
  gateway. (Final on-chain settlement failed for an unrelated
  reason: the test wallet was empty.)

### Notes for partners hitting RPC rate limits
- Helius free tier is 10 RPS — adequate for low QPS, not for
  bursty 50-100 concurrent. Move to Helius Developer ($99/mo,
  25 RPS) or Tatum (200 RPS).
- A separate ``0.24.0`` will add client-side blockhash caching so
  ~10 RPS of paid traffic resolves to <1 RPS of upstream RPC calls
  — at that point Helius free becomes viable for most production
  loads. Tracked separately because the change touches the x402
  scheme cache more invasively.

## 0.22.1 — 2026-05-12

### Fixed
- **Tool calling on Solana.** `SolanaLLMClient.chat_completion`,
  `SolanaLLMClient.chat_completion_stream`,
  `AsyncSolanaLLMClient.chat_completion`, and
  `AsyncSolanaLLMClient.chat_completion_stream` now accept ``tools`` /
  ``tool_choice`` kwargs and forward them to the upstream model.
  Previously the parameters were missing from the Solana SDK methods so
  partners couldn't use function calling on the Solana chain — but the
  BlockRun backend always supported the field uniformly; the SDK was
  the bottleneck.

  Live-verified: ``client.chat_completion("nvidia/deepseek-v4-flash",
  [...], tools=[get_weather], tool_choice="auto")`` returned
  ``tool_call: get_weather('{"city": "Tokyo"}')`` against
  ``sol.blockrun.ai``.

## 0.22.0 — 2026-05-12

### New
- **``AsyncSolanaLLMClient``** — async counterpart of
  ``SolanaLLMClient``. Mirrors the sync API for chat completions (both
  non-streaming and streaming) so ``asyncio`` callers don't need to
  thread-pool around blocking I/O. Built on the async ``x402Client``
  (instead of ``x402ClientSync``) + ``httpx.AsyncClient``. Public
  surface for the first release: ``chat()``, ``chat_completion()``,
  ``chat_completion_stream()``, ``list_models()``, ``close()`` plus
  ``__aenter__`` / ``__aexit__``. Image / Exa / Predexon / Music
  endpoints are still sync-only on Solana (they'll follow if there's
  demand). Same retry policy and ``fallback_models`` semantics as
  every other streaming client.
- **Paid streaming now writes to ``~/.blockrun/cost_log.jsonl`` and
  ``~/.blockrun/data/``** — closing the audit-trail gap that 0.20.x
  introduced. ``LLMClient`` (sync + async) and ``SolanaLLMClient``
  (sync + new async) all accumulate streamed content during the SSE
  iteration, then call ``save_to_cache`` once ``data: [DONE]`` arrives,
  building a synthetic ``chat.completion`` response so the local
  archive matches the non-stream paid path one-for-one. Free models
  skip the archive (``cost_usd == 0``). Failures during the stream do
  not produce a partial archive row.

### Verified e2e
- Async Solana streaming via ``AsyncSolanaLLMClient.chat_completion_stream``
  against ``sol.blockrun.ai`` with the free
  ``nvidia/deepseek-v4-flash`` model: 2 content chunks,
  ``"Hello! How can I"``, on the second attempt (first hit a
  transient NVIDIA NIM upstream timeout that resolved itself).
- 12/12 Base streaming unit tests + 6/6 Solana streaming unit tests
  still pass — the archive-on-completion change is additive and
  doesn't touch the existing assertions.

## 0.21.0 — 2026-05-12

### New
- **Streaming on Solana.** `SolanaLLMClient.chat_completion_stream(...)`
  is now a thing, mirroring the Base `LLMClient` API one-for-one:
  yields `ChatCompletionChunk` per SSE `data:` line, does the 402 →
  sign-locally-with-SVM-x402 → retry-with-PAYMENT-SIGNATURE dance
  before the first chunk, supports the same retry policy (5xx ×3 with
  1s/2s/4s backoff) and `fallback_models` chain walking.
- Constraint: like Base, fallback can only fire **before** the first
  chunk is yielded — once any chunk has reached the caller, switching
  models would concatenate two distinct responses.
- Async is not yet implemented for the Solana client (consistent with
  the rest of `SolanaLLMClient` which is sync-only today).

### Tests
- 6 new mock-based unit tests in `tests/unit/test_streaming_solana.py`:
  free-model direct streaming, paid-model sign-and-retry, recovery
  after 2× 503, raising after exhausted retries, fallback-chain
  walking, and payment-rejected → `PaymentError`.

### Verified e2e
- Live call against `sol.blockrun.ai` with the free
  `nvidia/deepseek-v4-flash` model: 2 content chunks, content
  "Silence.", 0.8s.

## 0.20.1 — 2026-05-12

### Improved
- **Streaming 5xx retry policy.** `_stream_with_payment` now retries
  transient upstream errors (500 / 502 / 503 / 504) up to three times per
  phase with exponential backoff (1s / 2s / 4s), instead of the single
  retry shipped in 0.20.0. Both the unauthenticated probe and the
  paid retry honor the same policy. Tuned for NVIDIA NIM upstream
  flakiness on free models — most transient hiccups now self-heal
  before bubbling up to the caller. Exposed as
  `LLMClient._STREAM_5XX_STATUSES` / `_STREAM_5XX_BACKOFFS` so callers
  can monkey-patch the policy in tests or override at runtime.
- **`fallback_models` parameter on `chat_completion_stream`** (sync +
  async). Walks the chain when the primary upstream produces a retriable
  error (timeouts, network errors, 5xx after exhausting in-band retries).
  **Constraint:** fallback only triggers *before the first chunk is
  yielded* — once any byte has reached the caller, switching upstreams
  would concatenate two distinct responses. After-first-chunk failures
  propagate to the caller as before.

### Tests
- Six new unit tests in `tests/unit/test_streaming.py` covering:
  recovery after two 503s, raising after exhausting retries, retry on
  the paid (post-402) retry leg, fallback to a healthy model after a
  primary 503-storm, no fallback after a chunk has been yielded, and
  no fallback on a non-retriable 4xx.

## 0.20.0 — 2026-05-11

### New

- **Server-Sent Events streaming for chat completions.** New methods
  `LLMClient.chat_completion_stream(...)` and
  `AsyncLLMClient.chat_completion_stream(...)` return an iterator of
  :class:`ChatCompletionChunk` objects, yielding one chunk per SSE event
  until the upstream emits `data: [DONE]`. The 402 → sign-locally →
  retry flow is identical to the non-streaming path; free models
  (e.g. `nvidia/deepseek-v4-flash`) stream directly without a payment
  dance. New types exported: `ChatCompletionChunk`, `ChatChunkChoice`,
  `ChatChunkDelta`. Validated end-to-end against the production
  `blockrun.ai` gateway (sync + async, free model). Caveats:
  `search_parameters` and the Responses-API models (`codex`,
  `gpt-5.4-pro`) reject streaming server-side with 400 — same constraint
  as the gateway. Six new unit tests cover the free path, paid 402-sign-
  retry path, payment rejection, and tolerance for malformed chunks.
- **Local billing / cost-tracking surface.** Every paid call now writes a
  `{ts, endpoint, cost_usd, model, wallet, network, client_kind}` row to
  `~/.blockrun/cost_log.jsonl`. New helpers on top:
  - `get_cost_log_summary(*, from_date, to_date, wallet, network, group_by)`
    — aggregate by `endpoint` / `model` / `wallet` / `network` /
    `client_kind` / `day` / `month`.
  - `export_cost_log_csv(...)` and `export_cost_log_json(...)` — render
    filtered per-call records, optionally to a file.
  - `python -m blockrun_llm.billing summary | export {csv|json}` CLI with
    `--from / --to / --wallet / --network / --group-by / --output` flags.
  Older 3-field cost-log rows remain readable; `by_endpoint` is still
  emitted as a backwards-compat alias when grouping by endpoint.
- **Predexon v2 typed helpers — full coverage across sync, async, Solana.**
  All three clients now expose the same 17 `pm_*` methods:
  - Canonical cross-venue (Tier 1): `pm_markets`, `pm_listings`, `pm_outcome`
  - Polymarket (Tier 1): `pm_polymarket_markets`, `pm_polymarket_events`,
    `pm_polymarket_markets_keyset`, `pm_polymarket_events_keyset`,
    `pm_polymarket_positions`, `pm_polymarket_trades`,
    `pm_polymarket_leaderboard`
  - Kalshi / Limitless (Tier 1): `pm_kalshi_markets`, `pm_limitless_markets`
  - Sports (Tier 1): `pm_sports_categories`, `pm_sports_markets`
  - Wallet identity (Tier 2): `pm_wallet_identity`, `pm_wallet_identities`,
    `pm_wallet_cluster`
- **`exa_*` methods on `LLMClient` (Base USDC).** `exa()`, `exa_search()`,
  `exa_find_similar()`, `exa_contents()`, `exa_answer()` — same surface and
  pricing as the existing `SolanaLLMClient` versions ($0.01/request for
  search / find-similar / answer, $0.002/URL for contents).
- **`fallback_models=[...]` on `chat()` and `chat_completion()`** (sync +
  async). On timeout, network error, or 5xx, the SDK transparently walks
  the list before raising. 4xx and `PaymentError` propagate immediately.
  Each fallback hop logs one line to stderr so the caller can see which
  model actually served the response.
- **`smart_chat()` uses the tier's fallback chain automatically.**
  `RoutingDecision` gained a `fallbacks: List[str]` field populated from
  the chosen tier; `smart_chat()` plumbs it through to `chat()`.
- **`examples/sweep_all_chat_models.py`** — runnable end-to-end sweep over
  every chat model the SDK exposes, with a forward-compat diff against
  `/v1/models`, async smoke, budget guard, and optional JSON output.
- **`examples/sweep_all_media_models.py`** — sister script for image and
  music models. Video is excluded by design (long polling, expensive).
- **New chat models in router / pricing tables:**
  - `anthropic/claude-opus-4.7` ($5/$25 per M, 1M context, 128K output,
    agentic coding + adaptive thinking) — promoted to
    `PREMIUM_TIERS["COMPLEX"]` primary; opus-4.5 retained as fallback.
  - `zai/glm-5.1` (flat $0.001/call, 200K context) — added to
    `ECO_TIERS["COMPLEX"]` fallback chain for long-context work.

### Changed

- **`/v1/images/models` is deprecated; image models live in `/v1/models`
  with `categories: ["image"]`.** `list_image_models()` (module-level,
  sync, async) and `list_all_models()` now read the unified catalog with
  the same return shape, so existing callers keep working without an
  extra request.
- **Pricing reads aligned with the current `/v1/models` schema.**
  `_get_model_pricing()` now reads nested `pricing.input` / `pricing.output`
  for paid models and `pricing.flat` for flat-billed models, falling back
  to the legacy top-level keys. Router cost estimates and savings %
  reflect the right numbers again, and flat-billed models compete in
  routing decisions on the right basis.
- **`FREE_TIERS["MEDIUM"]` primary** moved from `nvidia/deepseek-v4-flash`
  to `nvidia/llama-4-maverick`; v4-flash references in `AUTO_TIERS` /
  `ECO_TIERS` / `FREE_TIERS` fallback chains likewise redirected so the
  safety net hits a working model when the primary is unavailable.
- **ZAI GLM-5 family pricing** corrected from per-token to flat
  $0.001/call across the README pricing tables to match the catalog.
- **OpenAI dated-version responses** (e.g. `gpt-5.5-2026-04-20` for a
  request to `openai/gpt-5.5`) are no longer flagged as redirects — only
  base-id mismatches count.

### Removed

- `black-forest/flux-1.1-pro` — dropped from the README image table and
  from the media-sweep target list. Not in the live catalog.

## 0.19.0

- **Predexon v2 endpoints exposed via typed helpers.** All v2 endpoints went live in production on 2026-05-07 (`blockrun-web-00451-cnw`). The generic `pm()` / `pm_query()` passthrough already handled them, but agents can now discover the new shape from method names + docstrings. Ten new convenience methods on `LLMClient` — each is a thin wrapper, no breaking changes to the existing `pm()` API:
  - **Canonical cross-venue (Tier 1):** `pm_markets(**filters)`, `pm_listings(**filters)`, `pm_outcome(predexon_id)`. Predexon's unified data layer with cross-venue IDs across Polymarket, Kalshi, Limitless, Opinion, Predict.Fun.
  - **Polymarket keyset pagination (Tier 1):** `pm_polymarket_markets_keyset(**filters)`, `pm_polymarket_events_keyset(**filters)` — cursor-based for stable traversal of large result sets.
  - **Sports markets (Tier 1):** `pm_sports_categories()`, `pm_sports_markets(**filters)`.
  - **Wallet identity & clustering (Tier 2):** `pm_wallet_identity(wallet)` (GET), `pm_wallet_identities(addresses)` (POST, up to 200), `pm_wallet_cluster(address)` (GET on-chain relationship graph).
- `pm()` / `pm_query()` docstrings updated to advertise v2 examples and surface the Tier 1 / Tier 2 split inline.

## 0.18.0

- **DeepSeek V4 family in paid catalog.** Backend added `deepseek/deepseek-v4-pro` (1.6T MoE / 49B active, 1M context — strongest open-weight reasoner; MMLU-Pro 87.5, GPQA 90.1, SWE-bench 80.6, LiveCodeBench 93.5; **$0.50 in / $1.00 out per 1M under the 75% promo through 2026-05-31**, list $2.00/$4.00). The legacy `deepseek/deepseek-chat` and `deepseek/deepseek-reasoner` IDs are now V4 Flash non-thinking / thinking modes — repriced to **$0.20 in / $0.40 out per 1M, 1M context** (was $0.28/$0.42, 128K). Same upstream as `nvidia/deepseek-v4-flash` but on the paid endpoint with higher reliability and 5MB request bodies.
- **Smart router: free tier primaries repointed to visible models.** `FREE_TIERS["SIMPLE"]` was pinned to `nvidia/gpt-oss-120b` (now `hidden: true` in catalog — privacy-delisted from `/v1/models` though `available: true` for direct callers) and `FREE_TIERS["MEDIUM"]` to `nvidia/deepseek-v3.2` (hidden — NVIDIA NIM hung, backend redirects to v4-flash). Both are absent from `/v1/models`, so Python's pricing dict (built from that endpoint) could not resolve them and SmartChat silently fell through. Repointed primaries to visible IDs: `SIMPLE` → `nvidia/mistral-small-4-119b`, `MEDIUM` → `nvidia/deepseek-v4-flash`. Direct calls by full ID (`client.chat("nvidia/gpt-oss-120b", ...)`) still work — only auto-routing changed.
- **Smart router: V4 Pro promoted into reasoning fallbacks.** `AUTO_TIERS["REASONING"]` and `ECO_TIERS["REASONING"]` now list `deepseek/deepseek-v4-pro` as the first fallback after `deepseek-reasoner` (V4 Flash thinking stays primary because it's cheaper). `ECO_TIERS["COMPLEX"]` adds V4 Pro to fallbacks for harder reasoning tasks.
- README refresh: DeepSeek pricing table shows V4 Pro / V4 Flash chat / V4 Flash reasoner with correct prices and 1M context. NVIDIA free table notes that `gpt-oss-120b/20b` are hidden from `/v1/models` but still callable by direct ID (re-enabled 2026-04-30 after a brief privacy delisting).
- **`XClient` deprecated.** BlockRun's `/v1/x/*` (AttentionVC-partnered) integration was removed from the backend on 2026-04-30 (commit 80dcf52). The class is kept in the SDK so existing imports do not break, but instantiation now emits a `DeprecationWarning` — all calls return HTTP 404 until a replacement upstream is wired up.
- **DeepSeek V4 thinking + tool-call multi-turn now works.** Backend commit `f8a2d44` (2026-05-03) preserves `reasoning_content` on assistant messages with `tool_calls` for DeepSeek V4 thinking-mode (`deepseek-reasoner` / `deepseek-v4-pro`) — previously the streaming `/v1/messages` path stripped it, causing upstream 400 "reasoning_content in the thinking mode must be passed back" on tool-using multi-turn sessions, which the route then mis-classified as transient 503 → 5 retries with backoff on a deterministic failure. SDK `ChatMessage` already carried `reasoning_content` and `thinking` fields, so the fix is purely server-side; this entry exists so users seeing past failures know they're resolved.

## 0.17.1

- **Smart router: AUTO/ECO `SIMPLE` primaries promoted from `moonshot/kimi-k2.5` → `moonshot/kimi-k2.6`** (Moonshot's flagship — 256K context, vision + `reasoning_content`, $0.95 in / $4.00 out per 1M). The catalog now hides `kimi-k2.5` as superseded, so it no longer appears in `/v1/models` and the SDK could not resolve its pricing — routing was silently falling through to the next fallback. `kimi-k2.5` retained as the first fallback for clients explicitly pinned to its pricing.
- Doc refresh: README Smart Routing example output and SIMPLE tier table now reference `moonshot/kimi-k2.6`.

## 0.17.0

- **New flagship model: `openai/gpt-5.5`** (released 2026-04-23, first fully retrained base since GPT-4.5). 1M context, 128K output, native agent + computer use. Pricing $5.00 / $30.00 per 1M tokens.
- **Smart router: `PREMIUM_TIERS["MEDIUM"]` now points at `openai/gpt-5.5`**; `gpt-5.4` demoted to first fallback. The cost-savings baseline in `estimate_cost` was rebased from GPT-5.4 ($2.50/$15) to GPT-5.5 ($5.00/$30) so reported savings stay meaningful against the current flagship.
- Doc-example refresh: `AnthropicClient` cross-provider example and `examples/arbitrage_analyzer.py` `frontier` tier now reference `openai/gpt-5.5`.
- Reconciles `__version__` and `VERSION` (previously drifted at 0.16.1 vs 0.15.0); both now 0.17.0.

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
