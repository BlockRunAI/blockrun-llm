# Changelog

All notable changes to blockrun-llm will be documented in this file.

## 1.7.0 — 2026-07-15

### Added
- **`input_type` on video generation** (`VideoClient.generate`, `SolanaLLMClient.video`,
  `AsyncSolanaLLMClient.video`). Declares the intended seed mode — `text` /
  `image` / `first_last_frame` / `reference`. The gateway infers the mode from
  the seed fields and rejects with 400 **before charging** when the declared
  value disagrees, turning an expensive silent failure into a loud one: a
  dropped `image_url` otherwise yields a text-to-video clip you still pay for.
  Accepted on both chains.
- **`quality` on Solana image generation + editing** (`SolanaLLMClient.image` /
  `image_edit`, sync and async). `low` / `medium` / `high` / `auto` for
  `openai/gpt-image-*`; `low` meaningfully cuts generation time.

  **Solana only, by design.** The Base gateway defines no `quality` field and
  strips unknown keys, so a value sent there would be silently dropped —
  `ImageClient.generate`/`edit` therefore keep rejecting it, now with a hint
  pointing at the Solana client.

### Notes
- Reference-to-video (`reference_videos` / `reference_audios`) is **not** exposed.
  Both gateways gate it behind `R2V_ENABLED`, which is currently off, so every
  call would return 503. It slots in once that flips.
- Validation covers spelling only. Whether a declared mode matches the seed
  fields, and which models accept `quality`, stay the gateway's call — it
  answers both before billing, so a second copy here would only drift.

## 1.6.1 — 2026-07-15

### Fixed
- Fail fast when the payer has no USDC token account (#23). Below this an
  unfunded wallet burned all 5 payment retries, each costing the gateway 4
  verify retries — 20 facilitator calls per doomed request.

## 1.6.0 — 2026-07-08

### Added
- Attach the BlockRun builder-code service code to Base-chain x402 payments (#21).

## 1.5.1 — 2026-07-08

### Fixed
- Keep Solana video settlement blockhash fresh via proactive re-sign (#22).
  Seedance 2.0 jobs could run long enough to exhaust the older two-retry
  settlement loop and surface `transaction_simulation_failed`.

## 1.5.0 — 2026-07-06

### Added
- Solana media surface: video / music / speech / portrait / realface / price /
  rpc (#16), plus the `rpc_batch` cache fix (#17), a `solana<0.40` pin (#18),
  and media hardening (#19).

## 1.4.7 — 2026-06-26

### Added
- **`ChatCompletionChunk.cost_usd` on streamed calls.** The streaming paths now
  attach the real per-call x402 charge to every chunk (`_iter_and_archive` /
  `_aiter_and_archive`, Base + Solana), the streaming analogue of
  `ChatResponse.cost_usd`. It rides on the per-call chunk object, so it's
  **race-free** under shared-client concurrency (unlike `client._last_call_cost`,
  which goes stale). Downstream consumers (e.g. `blockrun-litellm`) can report
  the actual wallet deduction on streamed calls instead of a token×list-price
  estimate. Free / 200-first streams skip the signer and carry no `cost_usd`.

## 1.4.6 — 2026-06-24

### Added
- **`ChatResponse.cost_usd` and `ChatResponse.settlement`** (#11, #12). Every
  chat completion now carries the **real per-call x402 charge** (and the decoded
  on-chain settlement receipt when present), so downstream consumers (e.g.
  `blockrun-litellm`) can report the actual wallet deduction instead of a
  token×list-price estimate. The cost is attached **race-free** (set on the
  response object itself, not read back off the shared client); the free /
  200-first path reports exactly `0.0` (never a stale prior charge).

## 1.4.4 — 2026-06-18

### Added
- **`zai/glm-5.2` — Z.AI's newest flagship.** 1M-token context, top
  open-source on long-horizon coding, billed per-token at $1.40/$4.40 (same
  as glm-5.1). Added to the README ZAI table (as the new flagship) and to the
  chat-model sweep (including the reasoning set). Available now via direct
  call; SmartChat sees it live in `/v1/models`.

### Changed
- **SmartChat/Eco SIMPLE tier now routes to `moonshot/kimi-k2.7`.** Moonshot's
  current flagship (256K context, image+video input, `reasoning_content`) and
  the only k2 still visible in `/v1/models` — k2.6 and k2.5 are now
  `hidden:true`, so pinning the primary to either would silently degrade the
  tier. k2.6 retained as the documented previous-gen fallback.

## 1.4.3 — 2026-06-16

### Fixed
- **Clear error when a Solana key is passed to the Base (EVM) client.** Feeding
  a base58 Solana secret key into `LLMClient` / `setup_agent_wallet()` (or any
  EVM-chain client) used to fail with the cryptic `Private key must be 66
  characters (0x + 64 hexadecimal characters)`. The SDK now detects the base58
  Solana key shape and raises an actionable error pointing to `SolanaLLMClient`
  / `setup_agent_solana_wallet()` and the `[solana]` extra. Valid 64-hex EVM
  keys (including malformed ones) are unaffected and still get the hex error.

## 1.4.2 — 2026-06-14

### Fixed
- **Solana clients auto-load the on-disk wallet (parity with Base).**
  `SolanaLLMClient` / `AsyncSolanaLLMClient` now resolve the key as
  `private_key` → `SOLANA_WALLET_KEY` → on-disk wallet (newest
  `~/.<provider>/solana-wallet.json`, else `~/.blockrun/.solana-session`),
  so `SOLANA_WALLET_KEY` is no longer required when a wallet session exists —
  matching the Base `LLMClient.load_wallet()` fallback. A malformed key from any
  source now raises a clean `ValueError` (instead of a raw base58/solders
  exception), and an unreadable session file is treated as "no wallet" rather
  than crashing.

## 1.4.1 — 2026-06-14

### Fixed
- **Streamed tool calls no longer crash the SDK** (`'dict' object has no
  attribute 'delta'`). OpenAI streams tool calls incrementally — the first frame
  carries `id` + `function.name`, later frames only `function.arguments`
  fragments — which the strict non-stream `ToolCall` schema rejected, forcing a
  `model_construct` fallback that left `choices` as raw dicts and crashed the
  stream-archiving loop. Added lenient `ChatChunkToolCall` /
  `ChatChunkFunctionCall` types (all fields optional) for the streaming
  `delta.tool_calls`, and hardened the four sync/async archive loops
  (`client.py`, `solana_client.py`) with dict-tolerant accessors so any future
  `model_construct` fallback can't crash the stream. Affects `LLMClient` and
  `SolanaLLMClient`, sync and async.

## 1.4.0 — 2026-06-11

### Added
- **`LLMClient.onramp(address)` — Coinbase Onramp (FREE).** Mints a one-time
  `pay.coinbase.com` link to fund a wallet with fiat (card/bank, 60+ currencies
  → Base USDC). POSTs `{address, network: "base", asset: "USDC"}` to
  `/v1/onramp/token`. The x402 signature only authenticates the wallet, so the
  funding address must equal the signing wallet — pass
  `client.get_wallet_address()`. The returned URL is single-use and expires in
  ~5 min, so mint it at click time and never cache it. Base / USDC only;
  the address is validated against `^0x[0-9a-fA-F]{40}$` and a non-Coinbase URL
  raises `APIError("gateway returned no onramp url")`. Not added to the Solana
  client (Base-only). Adds `validation.validate_eth_address`.

### Docs
- **README payment section rewritten** into an explicit two-phase money flow:
  Phase 1 fund your wallet once (buy via `onramp()`, transfer Base USDC, or skip
  with free NVIDIA models — `get_balance()` to check); Phase 2 every request pays
  itself via automatic x402. Plus per-call pay-as-you-go costs, spend tracking
  (`get_spending()` / `blockrun_llm.billing`), BaseScan settlement verification,
  and the non-custodial key-never-leaves-your-machine guarantee.

## 1.3.0 — 2026-06-11

### Changed
- **Video poll budget default raised 5min → 15min**
  (`DEFAULT_GENERATE_BUDGET_SECONDS = 900`). Generation itself is 1-3min, but
  the upstream pipeline can lag the status read-path several minutes behind
  actual completion (observed 2026-06-11: video done in 100s, status flipped
  ~7.5min later). Jobs stay claimable ~48h, so a patient default beats a
  premature give-up. Override per call with `budget_seconds`.

### Added
- **Automatic mid-poll re-signing.** The x402 authorization window is 600s; on
  budgets longer than that a poll eventually 402s. The client now fetches a
  fresh challenge from the same poll_url and re-signs with the same wallet
  (the gateway enforces wallet binding, not signature equality), capped at 2
  re-signs — a fresh signature that 402s again raises `PaymentError`.
- **Recoverable timeouts.** The budget-exhausted `APIError` now carries
  `poll_url` in its details and explains that the job stays claimable for
  ~48h — re-GET the poll_url with a fresh same-wallet signature to fetch
  (and settle) the finished video. A client timeout is no longer a dead end.

## 1.2.3 — 2026-06-08

### Added
- **`AsyncSolanaLLMClient` now has `image`, `image_edit`, and `get_balance`.**
  This completes async-Solana public-method parity with the sync
  `SolanaLLMClient` and the async EVM client. `image`/`image_edit` are backed by
  a new async `_request_image_with_payment` that handles the gateway's async
  `202 + poll` slow path (gpt-image-2, dall-e-3, nano-banana-pro 4K) — signing
  once and polling until completion, settling only on the completed poll.
  `get_balance` runs the synchronous Solana RPC read in a worker thread
  (`asyncio.to_thread`) so it doesn't block the event loop.

## 1.2.2 — 2026-06-08

### Fixed
- **Video poll: terminal success is keyed on `status == "completed"`, not a
  literal HTTP 200** (parity with the Go 0.16.2 / TS 3.2.3 fixes). A
  completed-but-non-200 poll no longer spins to the budget deadline and raises
  "did not complete / no payment taken" for a job the caller was already
  charged for.

## 1.2.1 — 2026-06-08

### Added
- **`AsyncSolanaLLMClient.search(...)`** — async standalone search (Grok Live
  Search) parity with the sync `SolanaLLMClient` and the async EVM client. Thin
  wrapper over the async raw payment helper; same signature
  (`query`, `sources`, `max_results`, `from_date`, `to_date`, `timeout`).

## 1.2.0 — 2026-06-08

### Added
- **`AsyncSolanaLLMClient` passthrough parity.** The async Solana client now
  mirrors the sync `SolanaLLMClient` (and `AsyncLLMClient`) for the data
  passthroughs it previously lacked: prediction markets (`pm` + all `pm_*`),
  Exa web search (`exa`, `exa_search`, `exa_find_similar`, `exa_contents`,
  `exa_answer`), DefiLlama (`defi` + `defi_*`), 0x DEX (`dex` + `dex_*`), and
  Modal sandboxes (`modal` + `modal_sandbox_*`). Added the async raw request
  helpers (`_request_with_payment_raw` / `_get_with_payment_raw`) these build
  on, with Solana x402 signing, caching, and settlement capture.
- **`VideoClient.generate_from_content(content, …)`** — submits a standard
  Seedance `content[]` body to the gateway's `POST /v1/videos` endpoint
  (validates unsupported inputs before charging, then delegates to the same
  x402 submit+poll pipeline as `generate`). For migrating existing
  `content[]`-shaped payloads unchanged; most callers should still prefer
  `generate(...)` with structured kwargs.

## 1.1.0 — 2026-06-07

### Added
- **DefiLlama passthrough (`/v1/defillama/*`, live since 2026-05-02 — coverage
  backfill).** `defi(path, **params)` plus typed conveniences
  `defi_protocols` / `defi_protocol(slug)` / `defi_chains` / `defi_yields` /
  `defi_prices(coins)` on `LLMClient`, `AsyncLLMClient` and `SolanaLLMClient`.
  $0.005/call ($0.001 for prices).
- **0x DEX passthrough (`/v1/zerox/*`, live since 2026-05-02 — coverage
  backfill).** Free (no x402; BlockRun monetizes via on-chain affiliate fee):
  `dex(path, ...)` + `dex_price` / `dex_quote` / `dex_gasless_price` /
  `dex_gasless_quote` / `dex_gasless_submit` / `dex_gasless_status` /
  `dex_chains` / `dex_gasless_chains` on all three clients.
- **Modal sandbox compute (`/v1/modal/*`, live since 2026-04-09 — coverage
  backfill).** `modal(path, body)` + `modal_sandbox_create` ($0.01 CPU /
  $0.05 GPU) / `modal_sandbox_exec` / `modal_sandbox_status` /
  `modal_sandbox_terminate` ($0.001 each) on all three clients.

## 1.0.0 — 2026-06-07

### Removed (BREAKING)
- **`XClient` and the entire X/Twitter (AttentionVC) surface.** The backend
  removed the AttentionVC integration on 2026-04-30; every `/v1/x/*` endpoint
  has returned HTTP 404 since. Deleted: `x_client.py` (`XClient`), the 15
  `x_*` methods on `LLMClient` / `AsyncLLMClient` / `SolanaLLMClient`, and the
  18 `X*` response types (`XUser`, `XTweet`, `XSearchResponse`, ...).
  `XSearchSource` (Grok Live Search `sources:["x"]`) is unrelated and stays.
  If you need X/Twitter data, use Grok Live Search (`SearchClient` /
  `client.search(...)` with the `x` source) instead.

## 0.39.0 — 2026-06-07

### Added
- **`RpcClient` — Multi-chain JSON-RPC (40+ chains).** Mirrors the new
  backend `POST /v1/rpc/{network}` (Tatum gateway passthrough, launched
  2026-06-07). Flat $0.002 per call; a JSON-RPC batch charges per element.
  - `call(network, method, params)` — single JSON-RPC 2.0 call. EVM chains
    speak `eth_*`; non-EVM (Solana / Bitcoin-family / NEAR / Sui / XRP
    Ledger / Polkadot) speak their native JSON-RPC.
  - `batch(network, requests)` — JSON-RPC batch, priced per element.
  - `SUPPORTED_NETWORKS` (40 curated chains) + `NETWORK_ALIASES` (eth, arb,
    op, matic, bnb, avax, sol, btc, xrp, dot, ...). Unknown well-formed slugs
    fall through server-side to `{slug}-mainnet`, so new Tatum chains work
    without an SDK update.
  - New types: `RpcResponse` (JSON-RPC envelope + `network` / `cache_hit` /
    `tx_hash` gateway metadata), `RpcError`.
- **`VideoClient.generate()` new Seedance parameters** (backend 2026-06-02):
  - `last_frame_url` — first-and-last-frame interpolation: the model tweens
    from `image_url` (first frame) to `last_frame_url` (final frame).
    Requires `image_url` + a Seedance model. Priced as image-to-video.
  - `reference_image_urls` — omni / multi-reference: up to 9 reference images
    for character/style consistency (Seedance 2.0 only); cite them as
    "image 1", "image 2" in the prompt. Mutually exclusive with `image_url` /
    `last_frame_url` / `real_face_asset_id`.
  - token360 passthroughs that were already live upstream: `aspect_ratio`,
    `seed`, `watermark`, `return_last_frame`.
  - Client-side validation mirrors the backend mutual-exclusion rules.

### Changed
- **Free-tier router table rebuilt from a 2026-06-07 live sweep** (every
  visible free model probed):
  - `nvidia/qwen3-next-80b-a3b-thinking` hit NVIDIA end-of-life 2026-05-21
    (HTTP 410) — dropped as COMPLEX/REASONING primary. COMPLEX →
    `nvidia/qwen3-coder-480b` (871ms probe); REASONING →
    `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (681ms, explicit
    reasoning + vision).
  - `nvidia/mistral-small-4-119b` is timing out upstream (3/3 probes >60s) —
    dropped as SIMPLE primary and from all fallback chains.
  - `nvidia/deepseek-v4-flash` RECOVERED from the 05-09 NIM regression
    (896ms probe) — reinstated as SIMPLE primary.
- README free-model tables updated to match (qwen3-next retired,
  mistral-small flagged as timing out); sweep example pruned.

## 0.38.1 — 2026-06-06

### Changed
- **GLM flat-rate pricing fully retired.** Z.AI's remaining launch promos ended
  2026-06-06: `zai/glm-5` now bills per-token at $0.60/$1.92 and
  `zai/glm-5-turbo` at $1.20/$4.00 (no more flat $0.001/call anywhere in the
  family; glm-5.1 stays $1.40/$4.40). README ZAI section rewritten.
- **`zai/glm-5` removed from the ECO COMPLEX router fallback chain** — its slot
  existed only for the flat-rate pricing; at $0.60/$1.92 the existing per-token
  chain (deepseek-v4-pro $0.435/$0.87 first) is both cheaper and stronger.

## 0.38.0 — 2026-06-05

### Added
- **`SpeechClient` — BlockRun Voice (ElevenLabs TTS + sound effects).**
  - `generate()` (alias `speak()`) → `POST /v1/audio/speech` — OpenAI-compatible
    text-to-speech. Models: `elevenlabs/flash-v2.5` (default, $0.05/1k chars),
    `elevenlabs/turbo-v2.5` ($0.05/1k), `elevenlabs/multilingual-v2` ($0.10/1k),
    `elevenlabs/v3` ($0.10/1k). Voice aliases (sarah, george, laura, charlie,
    river, roger, callum, harry) or raw ElevenLabs voice_ids; `response_format`
    mp3/opus/pcm/wav; optional `speed` 0.7–1.2. Price scales with character
    count, minimum $0.001/request.
  - `sound_effect()` → `POST /v1/audio/sound-effects` — cinematic sound effects
    up to 22s, flat $0.05/generation (`elevenlabs/sound-effects`).
  - `list_voices()` → `GET /v1/audio/voices` — free voice discovery
    (rate-limited 60 req/min/IP).
  - New types: `SpeechResponse`, `SpeechAudio`.
- **xAI catalog additions (resold via OpenRouter credit pool, 2026-06-04):**
  `xai/grok-4.3` ($1.50/$4.00, 1M context, reasoning + vision) and
  `xai/grok-build-0.1` ($1.50/$3.00, 256K, fast agentic coding). Added to the
  chat sweep script and README. Older Grok chat SKUs (grok-3/4/4.1-fast
  families) are now hidden from `/v1/models`; direct calls still work.

### Changed
- **`zai/glm-5.1` launch promo ended (2026-06-05)** — now bills per-token at
  $1.40/$4.40 instead of flat $0.001/call. Removed from the ECO COMPLEX router
  fallback chain (it became the most expensive option there); `zai/glm-5`
  (still flat $0.001/call) takes the cheap long-context fallback slot.
- **`deepseek/deepseek-v4-pro` pricing corrected to $0.435/$0.87** — DeepSeek
  made the 75% launch promo the permanent list price after 2026-05-31 (README
  and router comments previously said the promo would expire back to list).

## 0.37.0 — 2026-06-01

### Fixed
- **Concurrent Solana payments now reach ~100% success.** Sharing one
  `SolanaLLMClient` / `AsyncSolanaLLMClient` across concurrent paid requests from
  a single wallet previously hit `invalid_exact_svm_payload_amount_mismatch` and
  `authorization already used` (replay) rejections under load (~3-10% failures),
  because the underlying x402 client is not concurrency-safe and a rejected
  payment couldn't recover. Two fixes:
  - A per-client signing lock (`threading.Lock` for sync, lazy `asyncio.Lock` for
    async) serialises the fast nonce/signature critical section.
  - A **whole-request payment retry**: a non-permanent payment rejection re-runs
    the entire request with a fresh 402 probe + fresh signature (new nonce,
    correct amount, current blockhash), for sync/async and streaming/non-stream
    (streaming only before the first chunk, so output is never replayed). New
    `_is_unrecoverable_payment_error` narrows the no-retry set to genuinely
    terminal cases (no funds / bad key / denylisted).
  - Verified at concurrency 10 on a shared client: opus-4.7, gemini-3.1-pro and
    gpt-5.5 all went from ~69-99% to **100/100**.

## 0.35.0 — 2026-05-31

### Added

- **`response_format` (JSON mode) and `stop` sequences on chat.** The gateway
  now honors both OpenAI params on `/v1/chat/completions` — natively for
  OpenAI/Azure, and emulated for Anthropic/Bedrock (a raw-JSON system
  instruction with code-fence stripping for `{"type": "json_object"}`; `stop`
  mapped to `stop_sequences`). Threaded through `chat`, `chat_completion`, and
  `chat_completion_stream` on both `LLMClient` and `SolanaLLMClient` (sync and
  async). Example: `client.chat("openai/gpt-4o", "...", response_format={"type": "json_object"})`.
- **Genuine `openai/gpt-4o` and `openai/gpt-4o-mini`** documented in the README
  pricing table (gpt-4o $2.50/$10.00 · 128K; gpt-4o-mini $0.15/$0.60 · 128K).
  The gateway no longer substitutes gpt-5.x for these IDs.

## 0.34.0 — 2026-05-29

### Fixed

- **`SolanaLLMClient` no longer truncates long chats and slow images at 60s.**
  The historical flat `DEFAULT_TIMEOUT = 60.0` applied to every method
  on the mega-class — chat, image, music, search, X, exa, pyth — while
  the Base SDK splits the same surface across per-use-case clients
  (`LLMClient=120s`, `ImageClient=200s`, `MusicClient=210s`,
  `VideoClient=360s`). Long chats with high `max_tokens`, slow image
  generations, and deep search queries were silently dying inside the
  SDK at 60s. Raises the flat `DEFAULT_TIMEOUT` to `120.0` (matches
  Base chat) and introduces per-use-case constants
  (`DEFAULT_CHAT_TIMEOUT`, `DEFAULT_IMAGE_TIMEOUT`,
  `DEFAULT_SEARCH_TIMEOUT`, `DEFAULT_FAST_TIMEOUT`). Each request now
  carries the timeout for its *workload* rather than the single client
  default: `image()` / `image_edit()` use `DEFAULT_IMAGE_TIMEOUT` (200s),
  `search()` and the `exa_*` methods use `DEFAULT_SEARCH_TIMEOUT` (300s),
  and chat uses the 120s baseline — sync **and** async. Closes #7.
- **`solana_key_to_bytes()` now wraps every failure in the documented
  `ValueError("Invalid Solana private key: …")`.** A bare
  `except ValueError: raise` used to let modern `base58`'s raw
  "Invalid character" error escape past the wrapper, so callers (and the
  `test_invalid_key_raises` test) matching on the documented message
  broke. All decode failures are now wrapped consistently.
- **`transaction_simulation_failed` no longer wastes 5+ minutes on
  pointless retries.** Adds a `_PERMANENT_PAYMENT_PATTERNS` table
  mirroring the gateway-side `blockrun-sol/src/lib/x402-solana.ts`
  `PERMANENT_ERRORS` classification. `_should_fallback_solana` now
  short-circuits when the exception's reason matches a permanent
  pattern — even when the exception type itself is "transient"
  (`httpx.Timeout`, `httpx.NetworkError`). Worst-case wall-clock for
  a deterministic Solana settlement failure drops from ~5min
  (3 generation attempts) to one attempt's worth. Closes #6.

### Added

- New module-level helpers:
  - `_is_permanent_payment_error(reason: str) -> bool` — case-insensitive
    substring match against the permanent classification, used by both
    the streaming fallback decision and any future retry classifier so
    one policy applies everywhere.
  - `DEFAULT_CHAT_TIMEOUT`, `DEFAULT_IMAGE_TIMEOUT`,
    `DEFAULT_SEARCH_TIMEOUT`, `DEFAULT_FAST_TIMEOUT` constants
    (importable from `blockrun_llm.solana_client`) so callers can use
    the same numbers as the SDK does.

### Added

- **Per-call `timeout=` override on every long-running public method**
  (level 2 of #7) — `chat`, `chat_completion`, `chat_completion_stream`,
  `image`, `image_edit`, `search`, sync and async. The kwarg wins over
  the per-use-case default and the constructor value, so a single
  oversized request can raise (or tighten) its own budget without
  reconfiguring the client:

  ```python
  client.chat_completion(model, messages, max_tokens=8192, timeout=240)
  client.image("...", model="openai/gpt-image-2", timeout=300)
  ```
- **`image_timeout` / `search_timeout` constructor parameters** on both
  `SolanaLLMClient` and `AsyncSolanaLLMClient` (defaulting to
  `DEFAULT_IMAGE_TIMEOUT` / `DEFAULT_SEARCH_TIMEOUT`) — mirrors the
  per-client tuning the Base SDK gets from separate `ImageClient` /
  search-aware `LLMClient` classes.

### Changed

- **`SolanaLLMClient(..., timeout=<float>)` still works**, but the
  default value of the constructor parameter is now
  `DEFAULT_CHAT_TIMEOUT` (120s) instead of the old 60s, and it governs
  the **chat** baseline specifically; image and search read from their
  own constructor parameters / constants. Callers passing an explicit
  value are unaffected.

### Notes

- 18 Base SDK clients still emit the generic
  `PaymentError("Payment was rejected. Check your wallet balance.")` —
  see the v0.32.0 follow-up note. Tracked separately.

## 0.33.0 — 2026-05-29

### Added
- **`anthropic/claude-opus-4.8`** ($5/$25 per M, 1M context, 128K output,
  agentic coding + adaptive thinking) — Anthropic's most capable Claude.
  Promoted to `PREMIUM_TIERS["COMPLEX"]` primary; opus-4.7 and opus-4.5
  retained as fallbacks. Also replaces opus-4.7 in the
  `PREMIUM_TIERS["REASONING"]` fallback chain. Added to the README pricing
  table and `examples/sweep_all_chat_models.py`.

## 0.32.0 — 2026-05-28

### Fixed
- **Image generation 202 + poll slow path** now handled transparently in both
  `ImageClient.generate()` / `.edit()` (Base) and `SolanaLLMClient.image()` /
  `.image_edit()` (Solana). Slow models (`openai/gpt-image-2`,
  `openai/dall-e-3`, `google/nano-banana-pro` at 4K, etc.) routinely exceed the
  gateway's 30s inline window and come back as `202` + `poll_url` instead of
  the finished image. The Solana path used to pass the job stub straight to
  `ImageResponse(**data)` and crash with a Pydantic ValidationError ("missing
  field `data`"); the Base path raised a confusing `APIError 202`. Both now
  poll the same `poll_url` with the same PAYMENT-SIGNATURE on `IMAGE_POLL_INTERVAL_SECONDS`
  (5s default) until `status: completed`, then return the parsed `ImageResponse`.
  Settlement only happens on the completed poll, so timing out the budget
  (`IMAGE_POLL_BUDGET_SECONDS`, 300s default) raises `APIError 504` and **no
  payment is taken**.
- **PaymentError now preserves the gateway's real failure reason.** On a 402
  retry response, the SDK used to raise a generic
  `"Payment rejected. Check your Solana USDC balance."` — losing the
  facilitator's actual reason (`transaction_simulation_failed`,
  `insufficient_funds`, `payment_expired`, etc.). The new
  `PaymentError(message, *, status_code=..., response=...)` keyword args
  carry the gateway body so callers and upstream proxies can surface the
  real reason. All four `SolanaLLMClient` retry paths (sync raw, sync get,
  sync stream, async post, async stream) and the Base `ImageClient` retry
  use the shared `validation.build_payment_rejected_error` helper.

### Changed
- **`PaymentError` constructor is now keyword-extended.** Existing
  `PaymentError("...")` calls are unchanged. The two new optional kwargs are
  `status_code: Optional[int]` and `response: Optional[dict]`.

### Notes for sidecar / proxy authors
- `blockrun-litellm >= 0.3.9` surfaces `PaymentError.response.details` on
  the 402 HTTP body. If you wrap `PaymentError` yourself, pull
  `exc.response.get("details")` for the structured facilitator reason.
- Follow-up: 18 other Base SDK clients (`client.py`, `phone.py`,
  `realface.py`, `surf.py`, `voice.py`, etc.) still inline the legacy
  `raise PaymentError("Payment was rejected. Check your wallet balance.")`
  pattern. They should migrate to `build_payment_rejected_error` in a
  follow-up PR — not blocking, but customers debugging settlement
  failures on those endpoints still lose context until then.

## 0.31.0 — 2026-05-27

### Added
- **`google/gemini-3.5-flash`** — Google's newest-generation Flash with built-in
  thinking mode: frontier-class quality at Flash speed and pricing ($0.50/M in,
  $3.00/M out, 1M context). Now live in production. Added to the README model
  pricing table and wired into the smart router's COMPLEX tier as the leading
  fallback (ahead of `google/gemini-3-flash-preview`, which remains available).

## 0.30.1 — 2026-05-26

### Changed
- **Default image-edit model is now `openai/gpt-image-2`** (was `openai/gpt-image-1`)
  across `ImageClient.edit()`, `LLMClient.image_edit()` (sync + async), and
  `SolanaLLMClient.image_edit()`. Matches the production `/v1/images/image2image`
  schema default and aligns Python, TypeScript, and Go SDKs. Pass `model=` explicitly
  to keep using the cheaper `gpt-image-1`.

## 0.30.0 — 2026-05-26

### Added
- **Multi-image fusion across all edit entry points.** The `image` parameter
  now accepts `Union[str, List[str]]` on `ImageClient.edit()`,
  `LLMClient.image_edit()` (sync + async), and `SolanaLLMClient.image_edit()`
  — pass a single base64 `data:image/...` data URI to edit one image, or a list
  of 2–4 URIs to fuse them (e.g. a subject photo + a brand logo). Matches the
  now-live `/v1/images/image2image` contract, which previously rejected arrays
  with `400 "expected string, received array"`. Single-string calls are
  unchanged and fully backward compatible. Fusion caps mirror the server:
  `openai/*` up to 4 source images, `google/*` (Nano Banana) up to 3; a `mask`
  cannot be combined with multiple source images.

### Fixed
- Documented the full set of edit-capable models (`openai/gpt-image-1`,
  `openai/gpt-image-2`, `google/nano-banana`, `google/nano-banana-pro`) and
  corrected the `edit()`/`image_edit()` docs, which incorrectly claimed a plain
  URL was accepted — the route requires a base64 `data:image/...` data URI.

## 0.29.0 — 2026-05-25

### Added
- **`RealFaceClient` — real-person face enrollment via x402.** RealFace
  registers a *real person's* likeness (vs. `PortraitClient`, which is for
  AI-generated characters). The asset works exactly like a Virtual Portrait
  on Seedance 2.0 / 2.0-fast — both return a `ta_xxxxxxxx` id you pass as
  `real_face_asset_id` on `VideoClient.generate()` — but enrollment proves
  the rights-holder is the person in the photo via a brief on-phone liveness
  check. **No KYC.** Three-step flow:
  - `init(name)` — *free*, rate-limited. Returns a `group_id` + an `h5_link`
    the real person scans on their phone.
  - `status(group_id)` / `wait_for_active(group_id)` — *free*. Poll until the
    person finishes the liveness check.
  - `enroll(name, image_url, group_id)` — **$0.01 USDC**, one-time. Settles
    only after the face matches the live capture, so `425` (group not active),
    `422` (face mismatch), and `502` (upstream failure) return errors with no
    charge.

  Plus `list_realfaces()` over the free `GET /v1/wallet/<address>/realfaces`
  endpoint.

  ```python
  from blockrun_llm import RealFaceClient
  faces = RealFaceClient()
  init = faces.init(name="Jane — spokesperson")  # show init.h5_link as a QR
  faces.wait_for_active(init.group_id)           # they do the phone check
  rf = faces.enroll(name="Jane — spokesperson",
                    image_url="https://example.com/jane.jpg",
                    group_id=init.group_id)
  print(rf.asset_id)  # ta_… → pass as real_face_asset_id on Seedance 2.0
  ```

- **`RealFaceInit`, `RealFaceStatus`, `RealFaceEnrollment`, `RealFaceList`,
  `RealFaceListItem`** exported from the package root.

### Changed
- **Reversed the v0.28.1 "real-person video is unsupported" stance.**
  Real-person likeness is now supported through the no-KYC RealFace liveness
  flow above (KYC is no longer required). The `VideoClient` class/parameter
  docstrings, the `real_face_asset_id` validator message, and the README now
  describe `real_face_asset_id` as accepting **either** a Virtual Portrait
  (`PortraitClient`, $0.01) **or** a RealFace (`RealFaceClient`, $0.01). No
  wire-format change — both still pass the same `ta_` id. `seedance-1.5-pro`
  does not support either asset type.

## 0.28.1 — 2026-05-23

### Added
- **`PortraitClient` — Virtual Portrait enrollment via x402.** Wraps
  `POST /v1/portrait/enroll` ($0.01 USDC, one-time, no KYC) and the
  free `GET /v1/wallet/<address>/portraits` listing endpoint. Enroll an
  AI character image, get back a `ta_xxxxxxxx` asset id, then reuse it
  as `real_face_asset_id` on `VideoClient.generate()` for Seedance 2.0 /
  2.0-fast to keep the same character across multiple videos. Settlement
  is held until upstream registration succeeds, so failed enrollments
  (content filter, image too large) return 502 with no charge.

  ```python
  from blockrun_llm import PortraitClient
  p = PortraitClient().enroll(
      name="My Spokesperson",
      image_url="https://example.com/character.jpg",
  )
  print(p.asset_id)              # ta_abcdef1234567890
  print(p.settlement.tx_hash)    # 0x9f3a…
  ```

- **`PortraitEnrollment`, `PortraitUsage`, `PortraitSettlement`,
  `PortraitList`, `PortraitListItem`** exported from the package root.

### Changed
- **`VideoClient` Seedance docs realigned with the (then-)dropped
  RealFace path.** _(Reversed in 0.29.0 — real-person video is now
  supported via the no-KYC RealFace liveness flow.)_ At the time, the
  `VideoClient` class docstring, the `real_face_asset_id` parameter
  docstring, the validator error message, and the README example were
  changed to describe `real_face_asset_id` exclusively as a Virtual
  Portrait (`POST /v1/portrait/enroll`, $0.01, no KYC). No behavior
  change — the wire format (the `ta_` id) is unchanged.

## 0.28.0 — 2026-05-22

### Added
- **`VideoClient.generate()` — face-reference, resolution, and audio
  controls** to align with the documented `/v1/videos/generations` schema:
  - `real_face_asset_id="ta_xxxxxx"` — condition Seedance 2.0 fast/pro on
    a Virtual Portrait or Token360 RealFace asset. Validates the `ta_`
    prefix and is mutually exclusive with `image_url`.
  - `resolution="360p" | "480p" | "720p" | "1080p" | "4K"` — drop to 480p
    for ~half the per-clip Seedance cost; bump to 1080p / 4K for higher
    fidelity. Grok ignores this field.
  - `generate_audio=True/False` — override Seedance's default (audio on
    for text-to-video, off for image- or face-conditioned). Grok ignores.

### Changed
- Refreshed Seedance pricing in the `VideoClient` docstring and README
  to match the live per-M-token billing (token360 charges by tokens at
  ~20,256 tok/sec at 720p), replacing the old per-second figures:
  - `bytedance/seedance-1.5-pro` — $4.32/M (flat) ≈ $0.46 / 5s 720p
  - `bytedance/seedance-2.0-fast` — $11.20/M text · $6.60/M image
  - `bytedance/seedance-2.0`       — $14.00/M text · $8.60/M image
  - `xai/grok-imagine-video` unchanged at $0.050/sec.

## 0.27.0 — 2026-05-22

### Added
- **Opt-in per-transaction log to a project-local folder.** Pass
  `transaction_log=True` to `LLMClient`, `AsyncLLMClient`, `SolanaLLMClient`,
  or `AsyncSolanaLLMClient` (or set `BLOCKRUN_TX_LOG=1`) and every paid call
  appends one plain-text row to `./log/transactions.log`:

  ```
  2026-05-21 15:44:46  chat  anthropic/claude-sonnet-4.6    in=    3  out=4  $0.034137  0x6513d128…
  ```

  Columns: timestamp, endpoint tag, model (left-padded 30), prompt/completion
  tokens, USD cost (6 decimals), and the first 10 chars of the on-chain
  settlement hash (Base tx hash or Solana signature). The hash is decoded
  from the `X-PAYMENT-RESPONSE` header the facilitator returns after
  settlement, so each row is verifiable against BaseScan / Solscan with one
  click — the row matches what hit the ledger.

  Pass a string/Path instead of `True` to choose a different directory.
  Disabled by default; no impact on the existing `~/.blockrun/cache`,
  `~/.blockrun/data/`, or `~/.blockrun/cost_log.jsonl` layers — this lives
  in its own folder next to your code.

- **`TransactionLogger`, `decode_settlement_header`, `format_row`** are
  exported from the package root for callers who want to build their own
  reconciliation tooling on top of the same primitives.

## 0.26.0 — 2026-05-18

### Added
- **`PhoneClient` — Twilio-backed phone lookup + number provisioning via x402.**
  New module `blockrun_llm/phone.py` wraps the backend's `/v1/phone/*` partner
  endpoints. Methods:
  - `lookup(phone_number)` — carrier + line-type ($0.01)
  - `lookup_fraud(phone_number)` — adds SIM-swap / call-forwarding signals ($0.05)
  - `buy_number(country="US", area_code=None)` — provision a US/CA number with a
    30-day lease bound to your wallet ($5.00). Settlement is held until Twilio
    confirms the purchase, so failed buys never charge your wallet.
  - `renew_number(phone_number)` — extend by 30 days ($5.00)
  - `list_numbers()` — list your active numbers ($0.001)
  - `release_number(phone_number)` — return a number to the pool (free, still
    flows through x402 for wallet-identity verification)
  Use the provisioned number as the `from_` caller ID in `VoiceClient.call()`.

- **`SurfClient` — asksurf.ai crypto-data gateway via x402.** New module
  `blockrun_llm/surf.py` wraps `/v1/surf/*` and exposes ~83 endpoints covering
  exchange data, on-chain SQL, prediction markets (Polymarket + Kalshi),
  wallet/social analytics, and project intelligence. Tiered pricing matches
  the backend: tier 1 / 2 / 3 → $0.001 / $0.005 / $0.020. API:
  - `SurfClient.endpoints()` — full discovery catalog
  - `SurfClient.endpoint_info(path)` / `SurfClient.price(path)` — single-endpoint metadata
  - `client.get(path, params)` / `client.post(path, body)` — direct callers
  - `client.call(path, params=…, body=…)` — auto-routes GET vs POST from the catalog
  Required-param validation runs client-side before the network round trip.

### Changed
- **`VoiceClient.call()` docs reflect new `from` resolution** on the backend:
  if `from_` is omitted and your wallet owns exactly one active number, the
  backend auto-picks it; 0 owned → 403 `no_active_number`; 2+ owned → 400
  `ambiguous_from` with the candidate list in the error body. No code change
  was needed — the SDK already forwarded `from_` correctly — but the docstring
  was stale.

## 0.25.0 — 2026-05-16

### Added
- **`VoiceClient` — AI-powered outbound phone calls via x402.** New module
  `blockrun_llm/voice.py` wraps the backend's `POST /v1/voice/call` (paid,
  $0.54/call) and `GET /v1/voice/call/{call_id}` (free polling). The AI agent
  dials a US/Canada E.164 number and conducts a real-time conversation
  following your `task` instructions; STT + LLM + TTS are handled upstream by
  Bland.ai. Full pass-through for `from`, `voice` (7 presets + custom Bland
  IDs), `max_duration` (1–30 min), `language`, `first_sentence`,
  `wait_for_greeting`, `interruption_threshold`, and `model` tier (base /
  enhanced / turbo). Status polling returns the full Bland call record
  (status, transcript, recording URL, ended_reason). Exported as `VoiceClient`
  from `blockrun_llm`. See README "Voice Calls" section for usage.

## 0.24.0 — 2026-05-14

### Changed
- **Default Solana RPC is now BlockRun's proxy** —
  `SolanaLLMClient` / `AsyncSolanaLLMClient` resolve their RPC
  endpoint to ``https://sol.blockrun.ai/api/v1/solana/rpc`` when no
  ``SOLANA_RPC_URL`` env var or explicit ``rpc_url`` arg is set.
  This is BlockRun's own multi-region, Tatum-backed Solana JSON-RPC
  proxy. It is free for anyone using the SDK — the cost is bundled
  into LLM inference fees you already pay. Method-aware caching on
  the server (``getLatestBlockhash`` at 30s TTL) collapses bursty
  signing traffic to a handful of upstream RPC calls, so partners
  no longer need to register Helius / Tatum / QuickNode for typical
  loads.

  The previous default ``https://api.mainnet-beta.solana.com`` is
  still reachable via ``SOLANA_RPC_URL=...`` but is no longer the
  default — its public rate limit (~10-40 RPS) is too aggressive
  for any real concurrency.

  No code change required to opt in: upgrade and you're using it.
  To stay on a private Helius / Tatum / QuickNode RPC, set
  ``SOLANA_RPC_URL`` (the 0.23.0 env-var mechanism is unchanged).

### Deprecated
- **`XClient` (BlockRun `/v1/x/*` AttentionVC integration)** — the
  backend ``/v1/x/*`` endpoints were removed on 2026-04-30. All
  ``XClient`` method calls now return HTTP 404 until a replacement
  X/Twitter data upstream is reintroduced. The class is kept in the
  SDK so existing imports do not break; instantiation now emits a
  ``DeprecationWarning`` so callers can migrate cleanly when a
  replacement ships.

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
