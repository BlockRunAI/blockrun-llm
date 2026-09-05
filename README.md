# BlockRun LLM SDK (Python)

> **blockrun-llm** is a Python SDK for accessing 80+ large language models (GPT-5.x, Claude 4.x, Gemini 3.x, DeepSeek, Grok 4.x, GLM, MiniMax, Moonshot and more) plus image / video / music generation, Grok Live Search, prediction-market data (Predexon), Exa neural web search, and Pyth-backed market data. Every call is paid per request — no subscription, no seats, no minimum. Built for AI agents that need to operate autonomously.
>
> **Two ways to pay, same SDK, same catalogue.** Sign up at
> **[user.blockrun.ai](https://user.blockrun.ai)** for an API key and prepaid
> credit (top up with a card), or hold USDC in your own wallet and let each
> request settle itself over x402 — on **Solana or Base**. Every client takes
> either credential in the same first argument.
>
> 🆓 **Includes 8 fully-free NVIDIA-hosted models** — DeepSeek V4 Flash (1M context), Nemotron Nano Omni (vision), Qwen3 Next + Coder, Llama 4 Maverick, Mistral Small 4, plus `gpt-oss-120b/20b` (hidden from `/v1/models` but direct calls still work). Zero USDC, no rate-limit gimmicks. Use `routing_profile="free"` or call any `nvidia/*` model directly.

[![PyPI](https://img.shields.io/pypi/v/blockrun-llm.svg)](https://pypi.org/project/blockrun-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BlockRun assumes Claude Code as the agent runtime.**

## Supported Chains

| Rail | Network | Payment | Status |
|------|---------|---------|--------|
| **API key** | none — `api.blockrun.ai` | prepaid credit, topped up with a card | ✅ |
| **Solana** | Solana Mainnet | USDC (SPL), gasless — the facilitator pays the fee | ✅ Recommended for x402 |
| **Base** | Base Mainnet (Chain ID: 8453) | USDC | ✅ |
| **Base Testnet** | Base Sepolia (Chain ID: 84532) | Testnet USDC | ✅ Development |

**Protocol:** x402 v2 on the wallet rails; plain bearer auth on the API-key rail.

## Installation

```bash
pip install blockrun-llm              # Base chain (EVM/USDC) — includes all core deps
pip install blockrun-llm[solana]      # Base + Solana (USDC SPL) payments
pip install blockrun-llm[dev]         # Base + dev tools (pytest, black, ruff, mypy)
pip install blockrun-llm[dev,solana]  # Everything
```

## Quick Start

```python
from blockrun_llm import LLMClient

# Reads BLOCKRUN_API_KEY if set, otherwise BLOCKRUN_WALLET_KEY for x402.
client = LLMClient()
response = client.chat("openai/gpt-5.2", "Hello!")
```

```bash
# API key — sign up at https://user.blockrun.ai, then:
export BLOCKRUN_API_KEY=brk_live_...

# …or a wallet, and every call pays itself in USDC:
export SOLANA_WALLET_KEY=...        # with SolanaLLMClient
export BLOCKRUN_WALLET_KEY=0x...    # with LLMClient (Base)
```

That's it. Either credential can also be passed directly —
`LLMClient("brk_live_…")` or `LLMClient("0x…")` — and `client.payment_mode`
reports which rail you ended up on.

### Try It Free (No Balance Required)

Want to kick the tires before topping up or funding a wallet? Route to
BlockRun's free NVIDIA tier — it settles $0 on both rails, so an unfunded wallet
or a $0 credit account is enough:

```python
from blockrun_llm import LLMClient

client = LLMClient()  # a credential is still needed; a balance is not

# Option 1: call a free model directly
response = client.chat("nvidia/step-3.7-flash", "Explain x402 in 1 sentence")

# Option 2: let the smart router pick the best free model per request
result = client.smart_chat("What is 2+2?", routing_profile="free")
print(result.model)     # e.g. 'nvidia/step-3.7-flash' (cheapest capable for SIMPLE tier)
print(result.response)  # '4'
```

**Available free models** (input + output both $0, all NVIDIA-hosted):

| Model ID | Context | Best For |
|----------|---------|----------|
| `nvidia/step-3.7-flash` | 131K | Fast general-purpose chat + reasoning |
| `nvidia/mistral-nemotron` | 131K | Fast free Mistral (Mistral × NVIDIA) |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | 256K | Only vision-capable free model — text + images + video (≤2 min) + audio (≤1 hr) |
| `nvidia/nemotron-nano-9b-v2` | 131K | Compact fast chat |
| `nvidia/nemotron-nano-12b-v2-vl` | 131K | Compact vision |
| `nvidia/gpt-oss-120b` | 128K | OpenAI open-weight 120B — the free workhorse. Hidden from `/v1/models` (so SmartChat won't auto-pick it) but direct calls still work |
| `nvidia/gpt-oss-20b` | 128K | OpenAI open-weight 20B — 155 tok/s. Hidden from `/v1/models` but direct calls still work |

> Need V4-Pro-class reasoning? Use the paid `deepseek/deepseek-v4-pro` ($0.435/$0.87 — the 75% launch promo became the permanent list price after 2026-05-31) — `nvidia/deepseek-v4-pro` is hidden because NVIDIA's NIM deployment is hung; backend MODEL_REDIRECTS forwards calls to V4 Flash.

> **Privacy note for `gpt-oss-120b/20b`**: NVIDIA's free build.nvidia.com tier reserves the right to use prompts/outputs for service improvement. The models are hidden from `/v1/models` so SmartChat won't auto-route to them, but direct calls still work — use them only when prompts contain no sensitive data.

> **Retired**: NVIDIA has EOL'd (HTTP 410) most of its early free lineup — the free DeepSeek family (last: `nvidia/deepseek-v4-flash`, 2026-08-12), `llama-4-maverick`, the qwen3 SKUs, free Mistral small/large, and more. The gateway auto-redirects pinned callers to a healthy free model, so old model IDs still return 200.

## Solana Support

**Solana is the recommended chain for x402 payments**: settlement is sub-second
and BlockRun's facilitator co-signs as fee payer, so a transfer costs you no SOL
and you hold nothing but USDC. Base works identically and remains what the bare
`LLMClient` uses, so nothing existing changes — but if you are choosing today,
choose Solana.

Pay for AI calls with Solana USDC via [sol.blockrun.ai](https://sol.blockrun.ai):

```python
from blockrun_llm import SolanaLLMClient

# SOLANA_WALLET_KEY env var (bs58-encoded Solana secret key)
client = SolanaLLMClient()

# Or pass key directly
client = SolanaLLMClient(private_key="your-bs58-solana-key")

# A BlockRun API key works here too — on the account rail there is no transfer
# to sign, so the chain stops being a question.
client = SolanaLLMClient(private_key="brk_live_...")

# Same API as LLMClient
response = client.chat("openai/gpt-5.2", "gm Solana")
print(response)

# DeepSeek on Solana
answer = client.chat("deepseek/deepseek-chat", "Explain Solana consensus", temperature=0.5)
```

**Agent setup (auto-loads or creates a wallet):**
```python
from blockrun_llm import setup_agent_solana_wallet

client = setup_agent_solana_wallet()   # uses ~/.blockrun/.solana-session, env, or creates one
client.chat("openai/gpt-5.2", "gm Solana")
```

**Setup:**
```bash
pip install blockrun-llm[solana]
export SOLANA_WALLET_KEY="your-bs58-solana-key"
```

**Endpoint:** `https://sol.blockrun.ai/api`
**Payment:** Solana USDC (SPL Token, mainnet)

> **Base vs Solana keys are not interchangeable.** A Solana key is base58
> (~44 chars for a seed, ~88 for a full keypair); a Base/EVM key is `0x` + 64
> hex chars. Pass a Solana key to `SolanaLLMClient` — **not** `LLMClient` /
> `setup_agent_wallet()`. If you do mix them up, the SDK now tells you exactly
> what to switch to instead of failing with a cryptic "must be 66 characters"
> error.

## Smart Routing (Router Core)

Let the SDK automatically pick the cheapest capable model for each request:

```python
from blockrun_llm import LLMClient

client = LLMClient()

# Auto-routes to the cheapest capable model
result = client.smart_chat("Summarize this changelog entry in one line")
print(result.response)
print(result.model)             # 'google/gemini-2.5-flash'
print(result.routing.task_type) # 'chat'
print(f"Saved {result.routing.savings * 100:.0f}%")  # 'Saved 90%'

# Complex reasoning task -> routes to a reasoning model
result = client.smart_chat("Prove the Riemann hypothesis step by step")
print(result.model)  # 'deepseek/deepseek-v4-pro'
```

Routing works the same on every client — `LLMClient`, `AsyncLLMClient`,
`SolanaLLMClient` and `AsyncSolanaLLMClient` all expose `route()`,
`smart_chat()` and `smart_chat_completion()`. Both chains run the same engine
against the same catalog, so an identical request picks an identical model; only
the x402 minimum in the cost estimate differs.

```python
# Route a full message list — tools and response_format shape the decision,
# not just the request
result = client.smart_chat_completion(
    [{"role": "user", "content": "Cancel order B-42"}],
    tools=[{"type": "function", "function": {"name": "cancel_order", "parameters": {}}}],
    tool_choice="required",
)
print(result.model)               # a tool-capable model
print(result.routing.task_type)   # 'tool_agent'

# Or opt in from OpenAI-compatible code by changing one string
response = client.chat_completion("blockrun/auto", messages)
```

Want to see the decision without paying for a call? `client.route(...)` runs the
same routing locally and returns the decision only:

```python
decision = client.route("Prove the Riemann hypothesis step by step")
print(decision.model)       # 'deepseek/deepseek-v4-pro'
print(decision.tier)        # 'REASONING'
print(decision.task_type)   # 'reasoning'
print(decision.candidates)  # ordered chain; smart_chat walks it on a 5xx/timeout
print(decision.reasoning)   # human-readable explanation of the pick
```

### Routing Profiles

| Profile | Description | Best For |
|---------|-------------|----------|
| `free` | NVIDIA free tier — smart-routes across the <!-- br:models.free -->7<!-- /br:models.free --> $0 models (Step 3.7 Flash, Mistral Nemotron, Nemotron Nano Omni / 9B / 12B VL) | Zero-cost testing, dev, prod |
| `eco` | Cheapest capable model per tier | Cost-sensitive production |
| `auto` | Best balance of cost/quality (default) | General use |
| `premium` | Top-tier models (Anthropic, OpenAI, Moonshot) | Quality-critical tasks |

```python
# Use premium models for complex tasks
result = client.smart_chat(
    "Write production-grade async Python code",
    routing_profile="premium"
)
print(result.model)  # 'openai/gpt-5.3-codex'
```

### How It Works

Routing runs on [Router Core](https://github.com/BlockRunAI/router-core) — the
same product-neutral engine the TypeScript SDK and the BlockRun gateway use, so
an identical request routes identically across all three. It is 100% local and
takes <1ms; no extra model call is made to decide.

Three stages:

1. **Classify** — a <!-- br:clawrouter.dimensions -->15<!-- /br:clawrouter.dimensions -->-dimension weighted
   scorer maps the request onto a capability tier (token count, code presence,
   reasoning markers, technical terms, creative markers, agentic patterns, and
   more), and a task classifier labels the *shape* of the work: `chat`,
   `code_edit`, `code_agent`, `tool_agent`, `reasoning_math`, `long_context`,
   `extraction`, `vision`, …
2. **Filter** — capability constraints are hard filters, not preferences. A
   model that cannot hold the conversation, emit the requested output length,
   call tools, or read images is dropped before scoring, so the router never
   picks a model the request would fail on.
3. **Rank** — surviving candidates are scored on task affinity, cost, speed and
   reliability. The winner serves the request; the rest become the ordered
   fallback chain that `smart_chat` walks on a timeout or 5xx.

The four capability tiers:

| Tier | Example Tasks | Auto Profile Model |
|------|---------------|-------------------|
| SIMPLE | Short questions, definitions | google/gemini-2.5-flash |
| MEDIUM | Code snippets, explanations | moonshot/kimi-k2.7 |
| COMPLEX | Architecture, long documents | google/gemini-3.1-pro |
| REASONING | Proofs, math, multi-step reasoning | deepseek/deepseek-v4-pro |

Every decision is explainable — `result.routing` carries the tier, the task
type, the confidence, the ranked `candidates`, the per-candidate
`candidate_scores` (quality / cost / speed / reliability) and a `reasoning`
string describing why that model won.

## How Payment Works

Two front doors onto the same gateway, the same catalogue and the same response
shapes. You choose one with the credential you hand the client.

| | **API key** — `api.blockrun.ai` | **Wallet (x402)** — `sol.blockrun.ai` / `blockrun.ai` |
|---|---|---|
| Authenticates with | `brk_live_…` from [user.blockrun.ai](https://user.blockrun.ai) | a signature from your own wallet |
| Pays from | prepaid credit on your account | USDC you hold, settled on-chain per call |
| Set up by | signing in with Google, minting a key, topping up with a card | funding a wallet with USDC |
| Chain | none — credit is off-chain | **Solana** or **Base** |
| Custody | BlockRun holds the credit you bought | non-custodial; your key never leaves your machine |
| Best for | teams that cannot run wallets, CI, anyone who wants a card receipt | agents, autonomous spend, no-signup access |

Free models are free on both.

### Option A — API key (user.blockrun.ai)

1. **Sign in** at **[user.blockrun.ai](https://user.blockrun.ai)** with Google.
2. **Mint a key** on the *API Keys* page. It is shown once — copy it then.
3. **Top up** on the *Billing* page with a card. Minimum $5. The processing fee
   (5.5% + $0.30) is charged **once, at purchase** — never on a call — so $10.85
   buys $10.00 of credit and every model then bills at the published list price,
   with no per-call minimum and no per-call fee.
4. **Export it:**

```bash
export BLOCKRUN_API_KEY=brk_live_...
```

```python
from blockrun_llm import LLMClient

client = LLMClient()               # picks up BLOCKRUN_API_KEY
print(client.payment_mode)         # 'apikey'
print(client.chat("openai/gpt-5.2", "What is 2+2?"))
```

Requests go to `https://api.blockrun.ai/v1` with the key as
`Authorization: Bearer …`. There is no 402 round trip and nothing is signed —
the gateway meters the call at exact usage and draws it from your credit.
Spending, per-call activity and remaining balance are on
[user.blockrun.ai/dashboard](https://user.blockrun.ai/dashboard).

**Precedence**, since it decides whether a call spends credit or on-chain USDC:

1. an explicit argument — `LLMClient("brk_live_…")` or `LLMClient("0x…")`;
2. `BLOCKRUN_API_KEY`, which **beats** `BLOCKRUN_WALLET_KEY` /
   `BASE_CHAIN_WALLET_KEY` / `SOLANA_WALLET_KEY`;
3. the wallet variables.

An existing wallet setup is untouched until you set `BLOCKRUN_API_KEY`, and
passing a wallet key explicitly always opts back out. `BLOCKRUN_API_KEY_URL`
overrides the account-rail host; it is deliberately not `BLOCKRUN_API_URL`,
which names an x402 gateway — an API-key client following that would send your
key to a host configured for a different rail.

**What changes.** `get_balance()`, `get_balance_testnet()` and `onramp()` raise
a `ValueError` pointing at the dashboard rather than answering: returning `0`
is indistinguishable from an empty wallet, and an agent gating on it would stop
calling a well-funded account. `get_wallet_address()` returns `""`. Running out
of credit raises a `PaymentError` naming the top-up page, not a wallet error.
`setup_agent_wallet()` mints nothing and hands back an API-key client, so a
skill can call it unconditionally. Everything else is identical.

### Option B — wallet + x402

You hold USDC in your own wallet — on **Solana** or **Base** — and each request
pays for itself with an on-chain micropayment. No signup, nothing custodial.

#### Phase 1 — Fund your wallet once

You only do this when your balance runs low. Three ways:

```python
from blockrun_llm import LLMClient

client = LLMClient()  # auto-detects wallet from BLOCKRUN_WALLET_KEY

# (a) Buy USDC with a card / bank (FREE) — mint a one-time Coinbase Onramp link
link = client.onramp(client.get_wallet_address())
print(link["url"])  # open https://pay.coinbase.com/... to buy USDC on Base
# The link is single-use and expires in ~5 min — mint it at click time, never cache it.

# (b) Transfer existing Base USDC to your wallet address
print(client.get_wallet_address())  # send USDC on Base to this 0x… address

# (c) Skip funding entirely — the free NVIDIA models cost $0
client.chat("nvidia/step-3.7-flash", "Hello!")  # routing_profile="free" also works
```

`$5` of USDC covers thousands of paid requests. Check your balance any time:

```python
print(f"Balance: ${client.get_balance():.2f} USDC")
```

#### Phase 2 — Every request pays itself (automatic x402)

```python
reply = client.chat("anthropic/claude-sonnet-4.6", "Explain x402 in one line")
```

That single call does all of this under the hood:

1. You send the request to BlockRun's gateway.
2. The gateway returns `402 Payment Required` with the price.
3. The SDK signs a USDC payment on Base **locally** (EIP-712) — your private
   key never leaves your machine.
4. The request is retried with the signed payment proof.
5. The gateway settles on-chain and returns the AI response.

One call, no separate pay step.

### What it costs, and how to verify it

- **Pay-as-you-go, per call.** You pay only the gateway price of each request
  (see [Available Models](#available-models)). The free NVIDIA models are `$0`.
- **Track spend.** `client.get_spending()` returns this session's
  `{total_usd, calls}`. On the API-key rail the gateway does not tell the client
  what a call cost, so treat that total as a floor and
  [user.blockrun.ai/dashboard](https://user.blockrun.ai/dashboard) as the
  authority. Every paid call also appends a line to
  `~/.blockrun/cost_log.jsonl`; summarize/export it with
  `blockrun_llm.billing` (`get_cost_log_summary`, `export_cost_log_csv`) — see
  [Billing & Cost Tracking](#billing--cost-tracking).
- **Verify settlements on-chain.** Each settlement returns a tx hash you can
  inspect on BaseScan — `https://basescan.org/tx/<hash>`, or view all activity
  for your wallet at `https://basescan.org/address/<your-address>`.
- **Non-custodial.** Your wallet is yours; the key is only used for local
  signing and **never leaves your machine**. No deposits held by BlockRun.

## Available Models

### OpenAI GPT-5.5 Family
Released 2026-04-23 — first fully retrained base since GPT-4.5. 1M context, 128K output, native agent + computer use.

| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `openai/gpt-5.5` | $5.00/M | $30.00/M | 1M |

### OpenAI GPT-5.4 Family
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `openai/gpt-5.4` | $2.50/M | $15.00/M | 1M |
| `openai/gpt-5.4-pro` | $30.00/M | $180.00/M | 1M |
| `openai/gpt-5.4-mini` | $0.75/M | $4.50/M | 400K |
| `openai/gpt-5.4-nano` | $0.20/M | $1.25/M | 1M |

### OpenAI GPT-5 Family
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `openai/gpt-5.3` | $1.75/M | $14.00/M | 128K |
| `openai/gpt-5.2` | $1.75/M | $14.00/M | 400K |
| `openai/gpt-5-mini` | $0.25/M | $2.00/M | 200K |
| `openai/gpt-5.2-pro` | $21.00/M | $168.00/M | 400K |
| `openai/gpt-5.3-codex` | $1.75/M | $14.00/M | 400K |

### OpenAI GPT-4o Family
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `openai/gpt-4o` | $2.50/M | $10.00/M | 128K |
| `openai/gpt-4o-mini` | $0.15/M | $0.60/M | 128K |

### OpenAI O-Series (Reasoning)
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `openai/o1` | $15.00/M | $60.00/M | 200K |
| `openai/o3` | $2.00/M | $8.00/M | 200K |
| `openai/o3-mini` | $1.10/M | $4.40/M | 128K |

### Anthropic Claude
| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `anthropic/claude-opus-4.8` | $5.00/M | $25.00/M | 1M | Most capable Claude — agentic coding + adaptive thinking, 128K output |
| `anthropic/claude-opus-4.7` | $5.00/M | $25.00/M | 1M | Agentic coding + adaptive thinking, 128K output |
| `anthropic/claude-opus-4.6` | $5.00/M | $25.00/M | 200K | Hidden from `/v1/models` (superseded by 4.7); direct calls still work |
| `anthropic/claude-opus-4.5` | $5.00/M | $25.00/M | 200K | |
| `anthropic/claude-sonnet-4.6` | $3.00/M | $15.00/M | 200K | |
| `anthropic/claude-haiku-4.5` | $1.00/M | $5.00/M | 200K | |

### Google Gemini
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `google/gemini-3.1-pro` | $2.00/M | $12.00/M | 1M |
| `google/gemini-3.5-flash` | $0.50/M | $3.00/M | 1M |
| `google/gemini-3-flash-preview` | $0.50/M | $3.00/M | 1M |
| `google/gemini-2.5-pro` | $1.25/M | $10.00/M | 1M |
| `google/gemini-2.5-flash` | $0.30/M | $2.50/M | 1M |
| `google/gemini-3.1-flash-lite` | $0.25/M | $1.50/M | 1M |
| `google/gemini-2.5-flash-lite` | $0.10/M | $0.40/M | 1M |

### DeepSeek

V4 family launched 2026-04-24. DeepSeek upstream now serves the legacy
`deepseek-chat` / `deepseek-reasoner` aliases as V4 Flash non-thinking /
thinking modes. V4 Pro is the new flagship paid SKU — 1.6T MoE / 49B active,
1M context, MMLU-Pro 87.5, GPQA 90.1, SWE-bench 80.6, LiveCodeBench 93.5.

| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `deepseek/deepseek-v4-pro` | $0.435/M | $0.87/M | 1M | V4 flagship — strongest open-weight reasoner. The 75% launch promo became the permanent list price after 2026-05-31 |
| `deepseek/deepseek-chat` | $0.14/M | $0.28/M | 1M | V4 Flash non-thinking (paid endpoint with 5MB request bodies) |
| `deepseek/deepseek-reasoner` | $0.20/M | $0.40/M | 1M | V4 Flash thinking (same upstream as `deepseek-chat`, thinking enabled by default) |

### MiniMax
| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `minimax/minimax-m3` | $0.30/M | $1.20/M | 1M | M3 flagship — strong reasoning + coding, 1M context |
| `minimax/minimax-m2.7` | $0.30/M | $1.20/M | 200K | |

### xAI Grok

Grok 4.3 and Grok Build are resold through BlockRun's OpenRouter credit pool
(same pattern as `deepseek/deepseek-v4-pro` and `minimax/minimax-m3`). Older
Grok chat SKUs (grok-3/4/4.1-fast families) are hidden from `/v1/models` but
direct calls by full ID still work.

| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `xai/grok-4.3` | $1.25/M | $2.50/M | 1M | Reasoning model, vision-capable, tuned for agentic workflows |
| `xai/grok-build-0.1` | $1.00/M | $2.00/M | 256K | Fast agentic coding model — interactive software-engineering workflows |

### ZAI

The GLM flat-rate launch promos have fully ended (glm-5.1 on 2026-06-05;
glm-5 and glm-5-turbo on 2026-06-06) — the whole family now bills per-token.

| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `zai/glm-5.2` | $1.40/M | $4.40/M | 1M | Z.AI's newest flagship — 1M-token context, top open-source on long-horizon coding |
| `zai/glm-5.1` | $1.40/M | $4.40/M | 200K | #1 open-source on SWE-Bench Pro, 8-hour autonomous execution |
| `zai/glm-5` | $0.60/M | $1.92/M | 200K | |
| `zai/glm-5-turbo` | $1.20/M | $4.00/M | 200K | |

### NVIDIA (Free & Hosted)

Free tier refreshed 2026-08-12. NVIDIA has retired (HTTP 410 end-of-life)
the entire free DeepSeek family — `nvidia/deepseek-v4-flash` was the last to
go — along with `llama-4-maverick`, `qwen3-coder-480b`, the free Mistral
small/large SKUs, and others. Retired models stay callable by ID: the gateway
auto-redirects them to a healthy free model, so pinned callers still get a
200. `nvidia/gpt-oss-120b` and `nvidia/gpt-oss-20b` remain callable by direct
ID but are hidden from `/v1/models` over the NVIDIA free tier's
prompt-retention terms (so SmartChat won't auto-pick them). The live list is
`GET /v1/models` filtered on the free flag.

| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `nvidia/step-3.7-flash` | **FREE** | **FREE** | 131K | Fast general-purpose chat + reasoning |
| `nvidia/mistral-nemotron` | **FREE** | **FREE** | 131K | Fast free Mistral |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | **FREE** | **FREE** | 256K | First vision-capable free model — RGB images, mp4 video |
| `nvidia/nemotron-nano-9b-v2` | **FREE** | **FREE** | 131K | Compact fast chat |
| `nvidia/gpt-oss-120b` | **FREE** | **FREE** | 128K | OpenAI open-weight 120B — 123 tok/s. Hidden from `/v1/models`; direct calls work |
| `nvidia/gpt-oss-20b` | **FREE** | **FREE** | 128K | OpenAI open-weight 20B — 155 tok/s. Hidden from `/v1/models`; direct calls work |
| `moonshot/kimi-k2.5` | $0.60/M | $3.00/M | 262K | Kimi K2.5 direct from Moonshot (replaces `nvidia/kimi-k2.5`) |
| `moonshot/kimi-k2.6` | $0.95/M | $4.00/M | 256K | Moonshot flagship (vision + reasoning_content) |

### Testnet Models (Base Sepolia)
| Model | Price |
|-------|-------|
| `openai/gpt-oss-20b` | $0.001/request |
| `openai/gpt-oss-120b` | $0.002/request |

*Testnet models use flat pricing (no token counting) for simplicity.*

### Verifying Models End-to-End

The SDK ships two runnable sweep scripts under `examples/`:

```bash
# Chat LLMs — every chat model the SDK exposes
python examples/sweep_all_chat_models.py --output-json sweep-results.json

# Image + music models (video excluded — long polling, expensive per clip)
python examples/sweep_all_media_models.py --output-json sweep-media-results.json
```

Each script captures per-model status, latency, token counts, and per-call
cost, prints a grouped report, and exits non-zero if any expected-to-work
model fails. Useful before a release or after router/catalog changes.

`smart_chat()` and `chat()` accept an optional `fallback_models=[...]` list —
on timeout / 5xx / network error the SDK transparently walks the chain
before raising. `smart_chat()` populates this from the tier's fallback list
automatically.

### Image Generation

| Model | Price |
|-------|-------|
| `openai/dall-e-3` | $0.04/image |
| `openai/gpt-image-1` | $0.02/image |
| `openai/gpt-image-2` | $0.06/image (reasoning-driven, multilingual text rendering, character consistency) |
| `google/nano-banana` | $0.05/image |
| `google/nano-banana-pro` | $0.10/image |
| `xai/grok-imagine-image` | $0.02/image |
| `xai/grok-imagine-image-pro` | $0.07/image |
| `zai/cogview-4` | $0.015/image |

Image editing (`client.edit` / `client.image_edit`) hits the `/v1/images/image2image` endpoint and supports `openai/gpt-image-1`, `openai/gpt-image-2`, `google/nano-banana`, and `google/nano-banana-pro`. Pass a list of source images to fuse multiple inputs (openai/* up to 4, google/* up to 3).

**`quality` (Solana only).** On Solana, `image` and `image_edit` accept
`quality="low" | "medium" | "high" | "auto"` for `openai/gpt-image-*` — `low`
meaningfully cuts generation time:

```python
sol = SolanaLLMClient()
result = sol.image("a red apple", model="openai/gpt-image-2", quality="low")
```

This is deliberately absent from the Base `ImageClient`: the Base gateway has
no `quality` field and would silently ignore the value, so passing it there
raises `TypeError` rather than quietly doing nothing.

### Video Generation
| Model | Price | Default 5s 720p |
|-------|-------|-----------------|
| `xai/grok-imagine-video` | $0.050/sec | 8s ≈ $0.40 |
| `bytedance/seedance-1.5-pro` | $4.32 / M tok (flat) | ≈ $0.46 |
| `bytedance/seedance-2.0-fast` | $11.20 / M text · $6.60 / M image | ≈ $1.19 t2v / $0.70 i2v |
| `bytedance/seedance-2.0` | $14.00 / M text · $8.60 / M image | ≈ $1.49 t2v / $0.91 i2v |

Seedance is billed by token360 in tokens (~20,256 tok/sec at 720p). Drop
`resolution="480p"` for ~half the cost, or bump to `1080p` / `4K`.
Seedance defaults to `720p` with synced audio on text-to-video; image- or
face-conditioned paths default audio off. Grok ignores `resolution` and
`generate_audio`.

```python
from blockrun_llm import VideoClient

client = VideoClient()
result = client.generate("a red apple slowly spinning on a wooden table")
print(result.data[0].url)            # permanent MP4 URL
print(result.data[0].duration_seconds)  # 8

# Image-to-video
result = client.generate(
    "the subject turns its head and smiles",
    image_url="https://example.com/portrait.jpg",
)

# Character-consistency video (Seedance 2.0 fast/pro). Pass a ta_xxxxxx
# asset to keep the same face across clips — either a Virtual Portrait
# (AI character, PortraitClient, $0.01) or a RealFace (real person,
# RealFaceClient, $0.01, no KYC). Mutually exclusive with image_url.
result = client.generate(
    "the subject smiles warmly and waves at the camera",
    model="bytedance/seedance-2.0",
    real_face_asset_id="ta_abc123xyz",
    resolution="1080p",
    generate_audio=True,
)

# First-and-last-frame interpolation (Seedance only): the model tweens
# from image_url (first frame) to last_frame_url (final frame).
# Priced identically to image-to-video.
result = client.generate(
    "the flower blooms in golden morning light",
    model="bytedance/seedance-1.5-pro",
    image_url="https://example.com/bud.jpg",
    last_frame_url="https://example.com/bloom.jpg",
)

# Omni / multi-reference (Seedance 2.0 only): up to 9 reference images
# for character/style consistency. Cite them as "image 1", "image 2"
# in the prompt. Mutually exclusive with image_url / last_frame_url /
# real_face_asset_id.
result = client.generate(
    "the character from image 1 walks through the city from image 2",
    model="bytedance/seedance-2.0",
    reference_image_urls=[
        "https://example.com/character.jpg",
        "https://example.com/city.jpg",
    ],
)

# input_type — declare the seed mode you intend, and get an error instead of
# a surprise. The gateway infers the mode from the seed fields above; if your
# declared value disagrees it returns 400 WITHOUT charging.
#
# Worth it when the seed fields are built dynamically: if `image_url` comes
# back empty, the request quietly degrades to text-to-video and you still pay
# for the clip. Declaring input_type="image" turns that into a 400 instead.
result = client.generate(
    "the portrait turns to face the camera",
    model="bytedance/seedance-2.0",
    image_url=maybe_empty_url,   # if this is falsy...
    input_type="image",          # ...you get a 400, not a text-to-video bill
)
```

### Text-to-Speech & Sound Effects (`SpeechClient`)

BlockRun Voice (ElevenLabs) — OpenAI-compatible TTS plus cinematic sound
effects. TTS price scales with character count: `(chars / 1000) × model
rate`, minimum $0.001/request. Synthesis is synchronous (<1s for Flash).

| Model | Price | Max Input | Notes |
|-------|-------|-----------|-------|
| `elevenlabs/flash-v2.5` | $0.05/1k chars | 40k chars | ~75ms latency, 32 languages (default) |
| `elevenlabs/turbo-v2.5` | $0.05/1k chars | 40k chars | ~250ms latency, balanced quality |
| `elevenlabs/multilingual-v2` | $0.10/1k chars | 10k chars | Long-form narration, audiobooks |
| `elevenlabs/v3` | $0.10/1k chars | 5k chars | Max expressiveness, 70+ languages |
| `elevenlabs/sound-effects` | $0.05/generation | 1k chars | Sound effects up to 22s |

```python
from blockrun_llm import SpeechClient

client = SpeechClient()

# Text-to-speech (voice aliases: sarah, george, laura, charlie,
# river, roger, callum, harry — or any raw ElevenLabs voice_id)
result = client.generate("Welcome to BlockRun.", voice="george")
print(result.data[0].url)  # audio URL (mp3 by default)

# Other formats / speed
result = client.generate(
    "Breaking news from the world of micropayments.",
    model="elevenlabs/v3",
    response_format="wav",
    speed=1.1,
)

# Sound effects (flat $0.05/generation)
result = client.sound_effect("rain on a tin roof, distant thunder")

# List voices (free, rate-limited)
voices = client.list_voices()
```

## Virtual Portraits (`PortraitClient`)

`PortraitClient` wraps `POST /v1/portrait/enroll` ($0.01 USDC, one-time,
no KYC) and the free `GET /v1/wallet/<address>/portraits` listing endpoint.
Enroll an AI-generated character image, get back a `ta_xxxxxxxx` asset id,
then reuse it as `real_face_asset_id` on Seedance 2.0 / 2.0-fast to keep
the same character across as many videos as you want.

> Need a **real person's** likeness instead? Use
> [`RealFaceClient`](#real-person-faces-realfaceclient) below — it
> enrolls a real face for **$0.01** via a quick on-phone liveness check,
> **no KYC**. Virtual Portraits are for AI-generated personas, mascots,
> avatars, and virtual spokespeople; RealFace is for real people. Both
> return a `ta_xxxxxx` id usable as `real_face_asset_id` on Seedance
> 2.0 / 2.0-fast.

```python
from blockrun_llm import PortraitClient, VideoClient

portraits = PortraitClient()
portrait = portraits.enroll(
    name="My Spokesperson",
    image_url="https://example.com/character.jpg",
)
print(portrait.asset_id)              # ta_abcdef1234567890
print(portrait.settlement.tx_hash)    # 0x9f3a… (BaseScan-verifiable)

# Reuse the same ta_ id on any Seedance 2.0 / 2.0-fast call
video = VideoClient()
clip = video.generate(
    "the character smiles warmly and waves at the camera",
    model="bytedance/seedance-2.0-fast",
    real_face_asset_id=portrait.asset_id,
)
print(clip.data[0].url)

# Browse this wallet's enrolled portraits (free, rate-limited)
listing = portraits.list_portraits()
for p in listing.portraits:
    print(p.assetId, p.name, p.enrollmentTxHash)
```

Settlement is held until the upstream registration succeeds — if the
image fails the content filter or exceeds 10 MB, the route returns 502
and **no payment is taken**, safe to retry with a different image.

## Real-Person Faces (`RealFaceClient`)

`RealFaceClient` enrolls a **real person's** likeness so you can keep the
same human face across multiple Seedance 2.0 / 2.0-fast videos. Unlike a
Virtual Portrait (an AI-generated character), RealFace proves the enroller
is the person in the photo via a brief **on-phone liveness check** (nod +
blink, ~1 minute) — **no KYC**, no government ID, no account login.

Enrollment is a three-step flow:

1. **`init(name)`** — *free*. Returns a `group_id` and an `h5_link` the
   real person opens on their phone (render it as a QR code).
2. **phone liveness** — the rights-holder opens the link, allows camera
   access, nods + blinks (~60s). Nothing is sent to BlockRun in this step.
3. **`enroll(name, image_url, group_id)`** — **$0.01 USDC**, one-time.
   Uploads the face photo, matches it against the live capture, and
   returns a `ta_xxxxxxxx` asset id.

```python
from blockrun_llm import RealFaceClient, VideoClient

faces = RealFaceClient()

# 1. Start enrollment (free). Show init.h5_link as a QR for the person.
init = faces.init(name="Jane — Q3 spokesperson")
print(init.h5_link)            # they scan + do the liveness check

# 2. Block until they finish the phone liveness check.
faces.wait_for_active(init.group_id)

# 3. Finalize ($0.01) with the person's face photo.
rf = faces.enroll(
    name="Jane — Q3 spokesperson",
    image_url="https://example.com/jane.jpg",
    group_id=init.group_id,
)
print(rf.asset_id)             # ta_abcdef1234567890
print(rf.settlement.tx_hash)   # 0x9f3a… (BaseScan-verifiable)

# Reuse the ta_ id on any Seedance 2.0 / 2.0-fast call
video = VideoClient()
clip = video.generate(
    "she smiles warmly and waves at the camera",
    model="bytedance/seedance-2.0-fast",
    real_face_asset_id=rf.asset_id,
)
print(clip.data[0].url)

# Browse this wallet's enrolled RealFaces (free, rate-limited)
listing = faces.list_realfaces()
for r in listing.realfaces:
    print(r.assetId, r.name, r.enrollmentTxHash)
```

Settlement happens only *after* the face is successfully matched and
registered, so failed enrollments return an error with **no charge**:
`425` = group not active yet (finish the phone check first), `422` = the
photo did not match the live capture (use a clearer front-facing photo),
`502` = upstream upload failure (safe to retry). The H5 session expires
~120s after each `init`; call `init(group_id=…)` to refresh an expired
link.

## Voice Calls (`VoiceClient`)

`VoiceClient` wraps `POST /v1/voice/call` (paid, $0.54/call) and
`GET /v1/voice/call/{call_id}` (free polling) — AI-powered outbound phone
calls powered by Bland.ai. The agent dials the recipient and runs a real-time
conversation based on your `task` instructions. US + Canada destinations.

```python
from blockrun_llm import VoiceClient

client = VoiceClient()

# Initiate (paid $0.54)
result = client.call(
    to="+14155552671",
    task="You are a friendly assistant calling to confirm a 3pm dentist appointment.",
    voice="maya",          # nat / josh / maya / june / paige / derek / florian
    max_duration=5,        # minutes (1–30)
)
print(result["call_id"])

# Poll for transcript + recording (free)
status = client.get_status(result["call_id"])
print(status.get("status"), status.get("recording_url"))
```

Bring your own caller-ID: pass `from_="+14155552671"` (must be a BlockRun
phone number you own; buy via `PhoneClient.buy_number()` or
`/v1/phone/numbers/buy`). If you omit `from_` and your wallet owns exactly one
active number, the backend auto-picks it; with multiple active numbers you'll
get a `400 ambiguous_from` and the error body lists your candidates.

## Phone Numbers (`PhoneClient`)

`PhoneClient` wraps `/v1/phone/*` — Twilio-backed phone lookup and
wallet-bound number provisioning. Buy a number once to use it as caller ID in
`VoiceClient`; the number is leased for 30 days and tied to your wallet.

```python
from blockrun_llm import PhoneClient

client = PhoneClient()

# Carrier + line-type lookup ($0.01)
info = client.lookup("+14155552671")

# Carrier + SIM-swap/forwarding fraud signals ($0.05)
fraud = client.lookup_fraud("+14155552671")

# Buy a number — 30-day lease, wallet-bound ($5.00).
# Payment is held until Twilio confirms the purchase, so failed buys never charge you.
bought = client.buy_number(country="US", area_code="415")
print(bought["phone_number"], bought["expires_at"])

# List, renew, release
print(client.list_numbers())                    # $0.001
client.renew_number(bought["phone_number"])     # $5.00, +30 days
client.release_number(bought["phone_number"])   # free
```

## Surf — Crypto Intelligence (`SurfClient`)

`SurfClient` wraps `/v1/surf/*` — the asksurf.ai partner gateway, ~83 crypto
endpoints across exchanges, on-chain SQL, prediction markets (Polymarket +
Kalshi), wallets, social analytics, and project intelligence. Tiered pricing:
$0.001 / $0.005 / $0.020 per call (tier 1 / 2 / 3).

```python
from blockrun_llm import SurfClient

client = SurfClient()

# Discovery
print(SurfClient.endpoints())                       # full catalog
print(client.price("market/ranking"))               # 0.001
print(client.endpoint_info("onchain/sql"))          # {'method': 'POST', 'tier': 3, ...}

# GET — pass query params (validated against the catalog)
btc_price = client.get("exchange/price", {"pair": "BTC/USDT"})
holders   = client.get("token/holders", {"address": "0x...", "chain": "ethereum"})

# POST — JSON body
rows = client.post("onchain/sql", {"query": "SELECT count() FROM ethereum.blocks"})

# Generic helper — auto-routes GET vs POST from the catalog
result = client.call("token/holders", params={"address": "0x...", "chain": "ethereum"})
```

## Standalone Search (`SearchClient`)

`SearchClient` wraps `POST /v1/search` — standalone Grok Live Search with
automatic x402 payment. Pricing: `$0.025/source + margin`
(10 sources ≈ `$0.26`).

```python
from blockrun_llm import SearchClient

client = SearchClient()
result = client.search(
    "Latest news on x402 adoption",
    sources=["x", "web"],
    max_results=10,
)
print(result.summary)
for url in result.citations or []:
    print(url)
```

## Market Data (`PriceClient`)

Pyth-backed realtime quotes and OHLC history across crypto, FX, commodities
and 12 global equity markets. Crypto / FX / commodity are **fully free**
across price, history and list; stocks (`stocks/{market}` and the `usstock`
legacy alias) charge `$0.001` per price or history call. Pass
`require_wallet=False` when you only need free endpoints.

```python
from blockrun_llm import PriceClient

# Free usage — no wallet
p = PriceClient(require_wallet=False)
btc = p.price("crypto", "BTC-USD")
eur = p.price("fx", "EUR-USD")
symbols = p.list_symbols("crypto", q="sol", limit=20)

# Paid — requires a wallet
p2 = PriceClient()
aapl = p2.price("stocks", "AAPL", market="us")
bars = p2.history(
    "stocks", "AAPL",
    market="us",
    resolution="D",
    from_ts=1_700_000_000,
    to_ts=1_710_000_000,
)
```

Supported stock markets: `us, hk, jp, kr, gb, de, fr, nl, ie, lu, cn, ca`.

## Multi-chain RPC (`RpcClient`)

Standard JSON-RPC 2.0 access to <!-- br:chains.rpc -->40<!-- /br:chains.rpc --> chains through one endpoint — Ethereum,
Base, Solana, Polygon, BSC, Arbitrum, Optimism, Avalanche, Bitcoin, Sui, and
more (powered by Tatum's RPC gateway). No API key, no per-chain endpoints:
flat **$0.002 per call** in USDC; a JSON-RPC batch charges per element.

```python
from blockrun_llm import RpcClient

client = RpcClient()

# EVM chains speak eth_* JSON-RPC
block = client.call("ethereum", "eth_blockNumber")
print(int(block.result, 16))

balance = client.call(
    "base", "eth_getBalance",
    ["0x4200000000000000000000000000000000000006", "latest"],
)

# Non-EVM chains speak their native JSON-RPC
slot = client.call("solana", "getSlot")
utxo_tip = client.call("bitcoin", "getblockcount")

# Batch: one payment, per-element pricing ($0.002 x N)
out = client.batch("polygon", [
    {"method": "eth_blockNumber"},
    {"method": "eth_gasPrice"},
])

print(block.network)    # "ethereum" (canonical key from X-Network)
print(block.cache_hit)  # True if served from the gateway's hot cache
print(block.tx_hash)    # x402 settlement tx
```

40 curated chains are exported as `blockrun_llm.SUPPORTED_NETWORKS`; common
aliases (`eth`, `arb`, `op`, `matic`, `bnb`, `avax`, `sol`, `btc`, `xrp`,
`dot`, ...) resolve server-side (`blockrun_llm.NETWORK_ALIASES`). Unknown but
well-formed slugs fall through to a generic `{slug}-mainnet` gateway attempt,
so new chains work without an SDK update. Hot, low-volatility reads
(`eth_chainId`, mined blocks/receipts, `getTransaction`, ...) are served from
a method-aware gateway cache — same price, lower latency.

## DeFi Data (Powered by DefiLlama)

GET passthrough to DefiLlama — protocols, TVL, yields, token prices.
$0.005/call ($0.001 for price lookups). Methods live on `LLMClient` /
`AsyncLLMClient` / `SolanaLLMClient`:

```python
client = LLMClient()

protocols = client.defi_protocols()           # all protocols + TVL
aave      = client.defi_protocol("aave")      # one protocol + historical TVL
chains    = client.defi_chains()              # TVL by chain
pools     = client.defi_yields()              # yield pools (APY/TVL)
prices    = client.defi_prices(["coingecko:bitcoin", "base:0x833589..."])

# Generic escape hatch
data = client.defi("protocol/uniswap-v3")
```

## DEX Swaps (Powered by 0x)

Free passthrough to the 0x Swap + Gasless APIs — **no x402 payment**
(BlockRun takes an on-chain affiliate fee on executed swaps instead).

```python
# Indicative price, then firm quote (Permit2)
price = client.dex_price(chainId=8453, sellToken="0x...", buyToken="0x...",
                         sellAmount="1000000")
quote = client.dex_quote(chainId=8453, sellToken="0x...", buyToken="0x...",
                         sellAmount="1000000", taker="0xYourWallet")

# Gasless flow: quote -> sign trade.eip712 -> submit -> poll
gq = client.dex_gasless_quote(chainId=8453, sellToken="0x...",
                              buyToken="0x...", sellAmount="1000000",
                              taker="0xYourWallet")
res = client.dex_gasless_submit({"trade": {...signed...}})
status = client.dex_gasless_status(res["tradeHash"])

client.dex_chains()           # supported swap chains
client.dex_gasless_chains()   # supported gasless chains
```

## Cloud Compute (Powered by Modal)

Pay-per-call sandboxed compute — create a sandbox, run commands, tear it
down. $0.01/create (CPU; $0.05 with GPU), $0.001 per exec/status/terminate.

```python
sb = client.modal_sandbox_create(image="python:3.11")
out = client.modal_sandbox_exec(sb["sandbox_id"], ["python", "-c", "print(40+2)"])
print(out["stdout"])  # 42
client.modal_sandbox_terminate(sb["sandbox_id"])
```

## Fund a Wallet with Fiat (Coinbase Onramp)

Mint a one-time `pay.coinbase.com` link to buy Base USDC with a card or bank
(60+ fiat currencies) — **FREE** (no x402 payment). The signature only
authenticates the wallet, so the funding address **must equal the signing
wallet**. Base / USDC only. The returned URL is single-use and expires in
~5 min, so mint it at click time and never cache it.

```python
link = client.onramp(client.get_wallet_address())
print(link["url"])  # https://pay.coinbase.com/... — open to buy USDC on Base
```

## Prediction Markets (Powered by Predexon v2)

Access real-time prediction market data from Polymarket, Kalshi, Limitless, sports, and Binance Futures via [Predexon](https://predexon.com). No API keys needed — pay-per-request via x402. Tier 1 endpoints are $0.001/call, Tier 2 (wallet identity / clustering) are $0.005/call.

Each method below is available on `LLMClient` (Base), `AsyncLLMClient`, and `SolanaLLMClient`.

### Typed helpers

| Method | Endpoint | Tier |
|---|---|---|
| `pm_markets(**filters)` | canonical cross-venue markets | 1 |
| `pm_listings(**filters)` | venue-native executable listings | 1 |
| `pm_outcome(predexon_id)` | resolve a canonical outcome | 1 |
| `pm_polymarket_markets(**filters)` | Polymarket markets (offset pagination) | 1 |
| `pm_polymarket_events(**filters)` | Polymarket events (offset pagination) | 1 |
| `pm_polymarket_markets_keyset(**filters)` | Polymarket markets, cursor pagination | 1 |
| `pm_polymarket_events_keyset(**filters)` | Polymarket events, cursor pagination | 1 |
| `pm_polymarket_positions(**filters)` | per-wallet open positions + PnL | 1 |
| `pm_polymarket_trades(**filters)` | recent trades (token, side, price, tx_hash) | 1 |
| `pm_polymarket_leaderboard(**filters)` | trader leaderboard (window, sort_by) | 1 |
| `pm_kalshi_markets(**filters)` | Kalshi event contracts | 1 |
| `pm_limitless_markets(**filters)` | Limitless binary AMM markets | 1 |
| `pm_sports_categories()` | available sports categories | 1 |
| `pm_sports_markets(**filters)` | sports markets grouped by game | 1 |
| `pm_wallet_identity(wallet)` | identity + profile for one wallet | 2 |
| `pm_wallet_identities(addresses)` | bulk identity for ≤200 wallets (POST) | 2 |
| `pm_wallet_cluster(address)` | on-chain transfer + identity-proof cluster | 2 |

```python
from blockrun_llm import LLMClient

client = LLMClient()

# Canonical cross-venue snapshot
markets = client.pm_markets(status="active", limit=20)
listings = client.pm_listings(venue="polymarket", limit=20)

# Polymarket
events = client.pm_polymarket_events(limit=10)
positions = client.pm_polymarket_positions(user="0xABC123...")
top = client.pm_polymarket_leaderboard(window="7d", sort_by="pnl", limit=10)

# Sports + Kalshi + Limitless
games = client.pm_sports_markets(league="NBA", limit=10)
kalshi = client.pm_kalshi_markets(limit=10)
limitless = client.pm_limitless_markets(limit=10)

# Wallet identity (Tier 2)
profile = client.pm_wallet_identity("0xABC123...")
batch = client.pm_wallet_identities(["0xABC...", "0xDEF..."])
cluster = client.pm_wallet_cluster("0xABC123...")
```

### Generic passthrough

For endpoints without a typed helper, drop down to `pm()` (GET) or `pm_query()`
(POST). Same pricing tiers, same return shape:

```python
candles = client.pm("polymarket/candlesticks/0x1234abcd...")  # OHLCV
btc = client.pm("binance/candles/BTCUSDT")                    # crypto candles
pairs = client.pm("matching-markets/pairs")                   # cross-platform pairs
```

## Exa Web Search (Powered by Exa)

Access [Exa](https://exa.ai)'s neural web search via x402. No API keys needed — pay-per-request in USDC. Available on both `LLMClient` (Base, recommended) and `SolanaLLMClient` (Solana).

| Endpoint | Method | Price |
|---|---|---|
| `exa_search` | Neural/keyword web search | $0.01/request |
| `exa_find_similar` | Find semantically similar pages | $0.01/request |
| `exa_contents` | Extract full text from URLs | $0.002/URL |
| `exa_answer` | AI answer grounded in web search | $0.01/request |

```python
from blockrun_llm import LLMClient

client = LLMClient()  # uses BLOCKRUN_WALLET_KEY (Base USDC)

# Neural web search ($0.01/request)
results = client.exa_search("latest AI safety research", numResults=5)
results = client.exa_search("bitcoin ETF news", category="news", numResults=10)

# Find similar pages ($0.01/request)
similar = client.exa_find_similar("https://openai.com/research/gpt-4", numResults=5)

# Extract content from URLs ($0.002/URL)
content = client.exa_contents(["https://arxiv.org/abs/2303.08774"])
content = client.exa_contents(
    ["https://example.com/page1", "https://example.com/page2"],
    text=True,
    highlights=True,
)

# AI-generated answer from live web ($0.01/request)
answer = client.exa_answer("What is the current state of AI safety research?")

# Generic proxy for any Exa endpoint
result = client.exa("search", {"query": "transformer architecture", "numResults": 5})
```

For Solana payments use `from blockrun_llm import SolanaLLMClient` — same method
names, same call shape; the Solana gateway requires the backend to be configured
with `EXA_API_KEY`, so prefer Base unless you need SOL/SPL settlement.

## Standalone Search

Search web, X/Twitter, and news without using a chat model:

```python
from blockrun_llm import LLMClient

client = LLMClient()

result = client.search("latest AI agent frameworks 2026")
print(result.summary)
for cite in result.citations or []:
    print(f"  - {cite}")

# Filter by source type and date range
result = client.search(
    "BlockRun x402",
    sources=["web", "x"],
    from_date="2026-01-01",
    max_results=5,
)
```

## Image Editing (img2img)

Edit existing images with text prompts. The source `image` must be a
`data:image/...;base64,...` data URI (plain URLs are not accepted):

```python
from blockrun_llm import LLMClient, ImageClient

# Via LLMClient
client = LLMClient()
result = client.image_edit(
    prompt="Make the sky purple and add northern lights",
    image="data:image/png;base64,...",  # base64 data URI
    model="openai/gpt-image-1",
)
print(result.data[0].url)

# Via ImageClient
img_client = ImageClient()
result = img_client.edit("Add a rainbow", image="data:image/png;base64,...")

# Multi-image fusion — pass a list of data URIs (e.g. a reference + a logo).
# openai/* accepts up to 4 source images, google/* up to 3.
result = img_client.edit(
    "Place the logo on the model's t-shirt",
    image=["data:image/png;base64,...", "data:image/png;base64,..."],
    model="google/nano-banana",
)
print(result.data[0].url)
```

## Usage Examples

### Simple Chat

```python
from blockrun_llm import LLMClient

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)

response = client.chat("openai/gpt-5.2", "Explain quantum computing")
print(response)

# With system prompt
response = client.chat(
    "anthropic/claude-sonnet-4.6",
    "Write a haiku",
    system="You are a creative poet."
)
```

### JSON Mode & Stop Sequences

`response_format` and `stop` are OpenAI-compatible and honored across **all** providers by
the gateway — native for OpenAI/Azure, and emulated for Anthropic/Bedrock (a raw-JSON system
instruction with code-fence stripping for JSON mode, `stop` mapped to `stop_sequences`).

```python
import json
from blockrun_llm import LLMClient

client = LLMClient()

# JSON mode — guaranteed parseable JSON, no markdown fences
response = client.chat(
    "openai/gpt-4o",
    "List 3 primary colors as a JSON array under key 'colors'.",
    response_format={"type": "json_object"},
)
print(json.loads(response))  # {'colors': ['red', 'green', 'blue']}

# Stop sequences (str or list, up to 4)
result = client.chat_completion(
    "openai/gpt-5.2",
    [{"role": "user", "content": "Count: Alpha Beta Gamma"}],
    stop=["Beta"],
)
print(result.choices[0].message.content)  # "Count: Alpha "
```

### Real-time Search (Live Search)

**Note:** Live Search can take 30-120+ seconds as it searches multiple sources. The SDK automatically uses a 5-minute timeout for search requests.

```python
from blockrun_llm import LLMClient

client = LLMClient()

# Simple: Enable live search with search=True (default 10 sources, ~$0.26)
response = client.chat(
    "openai/gpt-5.2",
    "What are the latest posts from @blockrunai?",
    search=True
)
print(response)

# Custom: Limit sources to reduce cost (5 sources, ~$0.13)
response = client.chat(
    "openai/gpt-5.2",
    "What's trending on X?",
    search_parameters={"mode": "on", "max_search_results": 5}
)

# Custom timeout (if 5 min isn't enough)
client = LLMClient(search_timeout=600.0)  # 10 minutes
```

### Check Spending

```python
from blockrun_llm import LLMClient

client = LLMClient()

response = client.chat("openai/gpt-5.2", "Explain quantum computing")
print(response)

# Check how much was spent
spending = client.get_spending()
print(f"Spent ${spending['total_usd']:.4f} across {spending['calls']} calls")
```

### Full Chat Completion

```python
from blockrun_llm import LLMClient

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I read a file in Python?"}
]

result = client.chat_completion("openai/gpt-5.2", messages)
print(result.choices[0].message.content)
```

### Async Usage

```python
import asyncio
from blockrun_llm import AsyncLLMClient

async def main():
    async with AsyncLLMClient() as client:
        # Simple chat
        response = await client.chat("openai/gpt-5.2", "Hello!")
        print(response)

        # Multiple requests concurrently
        tasks = [
            client.chat("openai/gpt-5.2", "What is 2+2?"),
            client.chat("anthropic/claude-sonnet-4.6", "What is 3+3?"),
            client.chat("google/gemini-2.5-flash", "What is 4+4?"),
        ]
        responses = await asyncio.gather(*tasks)
        for r in responses:
            print(r)

asyncio.run(main())
```

### List Available Models

```python
from blockrun_llm import LLMClient

client = LLMClient()
models = client.list_models()

for model in models:
    print(f"{model['id']}: ${model['inputPrice']}/M input, ${model['outputPrice']}/M output")
```

## Testnet Usage

For development and testing without real USDC, use the testnet:

```python
from blockrun_llm import testnet_client

# Create testnet client (uses Base Sepolia)
client = testnet_client()  # Uses BLOCKRUN_WALLET_KEY

# Chat with testnet model
response = client.chat("openai/gpt-oss-20b", "Hello!")
print(response)

# Check testnet USDC balance
balance = client.get_balance()
print(f"Testnet USDC: ${balance:.4f}")
```

### Testnet Setup

1. Get testnet ETH from [Alchemy Base Sepolia Faucet](https://www.alchemy.com/faucets/base-sepolia)
2. Get testnet USDC from [Circle USDC Faucet](https://faucet.circle.com/)
3. Set your wallet key: `export BLOCKRUN_WALLET_KEY=0x...`

### Available Testnet Models

- `openai/gpt-oss-20b` - $0.001/request (flat price)
- `openai/gpt-oss-120b` - $0.002/request (flat price)

### Manual Testnet Configuration

```python
from blockrun_llm import LLMClient

# Or configure manually
client = LLMClient(api_url="https://testnet.blockrun.ai/api")
response = client.chat("openai/gpt-oss-20b", "Hello!")
```

## Billing & Cost Tracking

Every paid call appends one line to `~/.blockrun/cost_log.jsonl` capturing
timestamp, endpoint, cost, and (when available) `model`, `wallet`, `network`,
and `client_kind`. The SDK ships a small reader / exporter on top so you can
audit spending without leaving the Python ecosystem.

### CLI

```bash
# Aggregated summary, default grouped by endpoint
python -m blockrun_llm.billing summary

# Group by model / month / wallet / network / client_kind / day
python -m blockrun_llm.billing summary --group-by model
python -m blockrun_llm.billing summary --group-by month --from 2026-04-01

# Filter by wallet (when one machine drives multiple keys)
python -m blockrun_llm.billing summary --wallet 0xCC8c... --network base-mainnet

# Export per-call records
python -m blockrun_llm.billing export csv  --from 2026-05-01 --output may.csv
python -m blockrun_llm.billing export json --to 2026-05-09
```

### Python API

```python
from blockrun_llm import (
    get_cost_log_summary,
    export_cost_log_csv,
    export_cost_log_json,
)

summary = get_cost_log_summary(group_by="model", from_date="2026-04-01")
print(summary["total_usd"], summary["calls"])
for model, slot in summary["groups"].items():
    print(f"  {model:40s}  {slot['calls']:>5}  ${slot['cost_usd']:.4f}")

# Returns CSV / JSON text; pass output_path to also write to disk
csv_text  = export_cost_log_csv("bill.csv", from_date="2026-05-01")
json_text = export_cost_log_json(from_date="2026-05-01")
```

### Example output

Real session — four cheap chat calls across providers, then queried by model:

```
$ python -m blockrun_llm.billing summary --from 2026-05-10 --group-by model
================================================================
BLOCKRUN — LOCAL COST LOG SUMMARY
================================================================
  log file       : /Users/me/.blockrun/cost_log.jsonl
  from           : 2026-05-10
  group_by       : model
  total          : $0.0070 (9 calls)

  KEY                             CALLS        COST
  ----------------------------  -------  ----------
  deepseek/deepseek-chat              2     $0.0020
  google/gemini-2.5-flash-lite        1     $0.0010
  anthropic/claude-haiku-4.5          1     $0.0010
  zai/glm-5-turbo                     1     $0.0010
  unknown                             4     $0.0020
```

The four `unknown` rows are pre-existing entries from before this release —
they had only `{ts, endpoint, cost_usd}` so the model column reads `unknown`.
Calls made after upgrading carry the full metadata (wallet / network /
client_kind / model). CSV export shows it directly:

```
$ python -m blockrun_llm.billing export csv --from 2026-05-10 | head -3
ts_iso,endpoint,model,wallet,network,client_kind,cost_usd
2026-05-10T03:38:28.198937+00:00,/v1/chat/completions,deepseek/deepseek-chat,0xCC8c...5EF8,base-mainnet,LLMClient,0.001
2026-05-10T03:38:31.192060+00:00,/v1/chat/completions,google/gemini-2.5-flash-lite,0xCC8c...5EF8,base-mainnet,LLMClient,0.001
```

### Scope

The cost log is per-machine. It records calls made by this Python SDK only —
calls from other clients (TS SDK, MCP, raw curl) are not included. For
organization-wide billing, query the gateway's authoritative ledger.

## Transaction Log (project-local, on-chain match)

The cost log above lives in `~/.blockrun/` and is hash-keyed JSON. When you'd
rather have an **eyeballable text log next to your code** that matches the
chain row-for-row, opt into the per-transaction log:

```python
from blockrun_llm import LLMClient

# Default: writes ./log/transactions.log
client = LLMClient(transaction_log=True)

# Or pick a path
client = LLMClient(transaction_log="./var/blockrun.log")

# Or via env var: BLOCKRUN_TX_LOG=1   (default dir)
#                  BLOCKRUN_TX_LOG=./var/blockrun.log
```

Works the same on `AsyncLLMClient`, `SolanaLLMClient`, and `AsyncSolanaLLMClient`.

Every paid call appends one row. Example:

```
2026-05-21 15:44:46  chat  anthropic/claude-sonnet-4.6    in=    3  out=4  $0.034137  0x6513d128…
2026-05-20 04:34:17  chat  openai/gpt-5.5                 in=   14  out=18 $0.001000  0x421796a3…
```

Columns: timestamp · endpoint tag (`chat`/`image`/`video`/`search`/…) · model
(padded to 30) · `in=` prompt tokens · `out=` completion tokens · `$cost` to
6 decimals · first 10 chars of the **on-chain settlement hash**.

### Why it matches the chain

The hash comes from the `X-PAYMENT-RESPONSE` header the x402 facilitator
returns after settlement — Base txs use `transaction`, Solana uses
`signature`. Both normalise to the truncated `0x…` / signature shown in
the row, so each line is verifiable in one click:

- Base mainnet → `https://basescan.org/tx/<full hash>`
- Solana mainnet → `https://solscan.io/tx/<full signature>`

Cached / free responses don't hit the chain, so they show `(no-tx)` instead.

### Scope and trade-offs

- **Independent of the cache layer.** Enabling the log does not change
  `~/.blockrun/cache/`, `~/.blockrun/data/`, or `~/.blockrun/cost_log.jsonl`.
- **Best-effort writes.** OSErrors are swallowed; a read-only filesystem can't
  break a paid call.
- **Plain text only.** If you need a structured ledger as well, query
  `~/.blockrun/cost_log.jsonl` via `blockrun_llm.billing`.

### Programmatic access

```python
from blockrun_llm import TransactionLogger, format_row

# Tail the project log
logger = TransactionLogger("./log")
for row in logger.entries()[-5:]:
    print(row)

# Build your own row (e.g. for tests or custom adapters)
print(format_row(
    endpoint="/v1/chat/completions",
    model="openai/gpt-5.5",
    in_tokens=14,
    out_tokens=18,
    cost_usd=0.001,
    tx_hash="0x421796a3deadbeef",
))
```

## Environment Variables

One credential is required — an API key **or** a wallet key. Everything else is
optional.

| Variable | Description | Default |
|----------|-------------|---------|
| `BLOCKRUN_API_KEY` | API key from [user.blockrun.ai](https://user.blockrun.ai) (`brk_live_…`). **Takes precedence over every wallet variable.** | — |
| `SOLANA_WALLET_KEY` | bs58 Solana wallet key, for `SolanaLLMClient` | falls back to `~/.*/solana-wallet.json`, then `~/.blockrun/.solana-session` |
| `BLOCKRUN_WALLET_KEY` | Base chain wallet private key | falls back to `~/.blockrun/.session` |
| `BASE_CHAIN_WALLET_KEY` | Alias for `BLOCKRUN_WALLET_KEY` | — |
| `BLOCKRUN_API_KEY_URL` | Override the API-key gateway | `https://api.blockrun.ai` |
| `BLOCKRUN_API_URL` | Override the Base x402 gateway | `https://blockrun.ai/api` |
| `SOLANA_RPC_URL` / `SOLANA_RPC_HEADERS` / `SOLANA_RPC_API_KEY` | RPC for blockhash + mint info while signing | BlockRun's free proxy |
| `BLOCKRUN_CHAT_TIMEOUT` | Chat HTTP timeout, in seconds | `600` |
| `BLOCKRUN_MAX_COST_PER_CALL` / `BLOCKRUN_MAX_SESSION_COST` | Opt-in spend limits (wallet rail) | unlimited |

`BLOCKRUN_API_KEY_URL` is deliberately not `BLOCKRUN_API_URL`: that one names an
x402 gateway, and an API-key client must never follow it and send your key to a
host you configured for a different rail.

## Setting Up Your Wallet

1. Create a wallet on Base network (Coinbase Wallet, MetaMask, etc.)
2. Get some ETH on Base for gas (small amount, ~$1)
3. Get USDC on Base for API payments
4. Export your private key and set it as `BLOCKRUN_WALLET_KEY`

```bash
# .env file
BLOCKRUN_WALLET_KEY=0x...your_private_key_here
```

## Error Handling

```python
from blockrun_llm import LLMClient, APIError, PaymentError

client = LLMClient()

try:
    response = client.chat("openai/gpt-5.2", "Hello!")
except PaymentError as e:
    print(f"Payment failed: {e}")
    # Check your USDC balance
except APIError as e:
    print(f"API error ({e.status_code}): {e}")
```

## Testing

### Running Unit Tests

Unit tests do not require API access or funded wallets:

```bash
pytest tests/unit                    # Run unit tests only
pytest tests/unit --cov              # Run with coverage report
pytest tests/unit -v                 # Verbose output
```

### Running Integration Tests

Integration tests call the production API and require:
- A funded Base wallet with USDC ($1+ recommended)
- `BLOCKRUN_WALLET_KEY` environment variable set
- Estimated cost: ~$0.05 per test run

```bash
export BLOCKRUN_WALLET_KEY=0x...
pytest tests/integration             # Run integration tests only
pytest                               # Run all tests
```

Integration tests are automatically skipped if `BLOCKRUN_WALLET_KEY` is not set.

## Security

### Private Key Safety

- **Private key stays local**: Your key is only used for signing on your machine
- **No custody**: BlockRun never holds your funds
- **Verify transactions**: All payments are on-chain and verifiable

### Best Practices

**Private Key Management:**
- Use environment variables, never hard-code keys
- Use dedicated wallets for API payments (separate from main holdings)
- Set spending limits by only funding payment wallets with small amounts
- Never commit `.env` files to version control
- Rotate keys periodically

**Input Validation:**
The SDK validates all inputs before API requests:
- Private keys (format, length, valid hex)
- API URLs (HTTPS required for production, HTTP allowed for localhost)
- Model names and parameters (ranges for max\_tokens, temperature, top\_p)

**Error Sanitization:**
API errors are automatically sanitized to prevent sensitive information leaks.

**Monitoring:**
```python
address = client.get_wallet_address()
print(f"View transactions: https://basescan.org/address/{address}")
```

**Keep Updated:**
```bash
pip install --upgrade blockrun-llm  # Get security patches
```

## Agent Wallet Setup

One-line setup for agent runtimes (Claude Code skills, MCP servers, etc.):

```python
from blockrun_llm import setup_agent_wallet

# Auto-creates wallet if none exists, returns ready client
client = setup_agent_wallet()
response = client.chat("openai/gpt-5.4", "Hello!")
```

For Solana:

```python
from blockrun_llm import setup_agent_solana_wallet

client = setup_agent_solana_wallet()
response = client.chat("anthropic/claude-sonnet-4.6", "Hello!")
```

Check wallet status:

```python
from blockrun_llm import status

status()
# Wallet: 0xCC8c...5EF8
# Balance: $5.30 USDC
```

## Wallet Discovery and Migration

The SDK can discover compatible wallets for an explicit, user-confirmed
migration. It never automatically makes a discovered provider wallet active:

```python
from blockrun_llm.wallet import scan_wallets
from blockrun_llm.solana_wallet import scan_solana_wallets

# Scans ~/.<dir>/wallet.json for Base wallets
base_wallets = scan_wallets()

# Scans ~/.<dir>/solana-wallet.json
sol_wallets = scan_solana_wallets()
```

`get_or_create_wallet()` always uses `~/.blockrun/.session` (or an explicit
wallet environment variable, or the legacy `~/.blockrun/wallet.key`). Review
the discovered addresses and import one explicitly if you intend to switch
wallets.

### Upgrading from a provider wallet

Earlier versions adopted the most recently written provider wallet
automatically. If you relied on that, the first run after upgrading creates a
fresh BlockRun wallet and prints the addresses it found, so you can import the
one you actually own:

```
NOTICE: BlockRun created a new wallet, but also found existing wallet(s)
belonging to other applications on this system:

  0x88f9B82462f6C4bf4a0Fb15e5c3971559a316e7f
...
```

Adopt one deliberately:

```python
from blockrun_llm import list_discovered_wallets, import_wallet

for w in list_discovered_wallets():
    print(w["address"], "from", w["source"])

import_wallet("0x88f9B82462f6C4bf4a0Fb15e5c3971559a316e7f")
```

`import_wallet()` writes your current wallet to
`~/.blockrun/.session.backup-<timestamp>` before switching, so adopting a wallet
never strands funds in the old one. Solana: `list_discovered_solana_wallets()`
and `import_solana_wallet()`.

Addresses shown are derived from the discovered key itself, and `import_wallet()`
matches on that derived address — so a wallet file cannot claim an address it
cannot sign for, nor be adopted by one. `list_discovered_wallets()` never returns
private keys.

For a single run without changing anything, use
`export BLOCKRUN_WALLET_KEY=<private-key>`.

## Response Caching

The SDK caches responses to avoid duplicate payments:

```python
from blockrun_llm import clear_cache

# Automatic TTLs by endpoint:
# - Prediction Markets: 30 minutes
# - Search: 15 minutes
# - Models: 24 hours
# - Chat/Image: no cache (every call is unique)

# Manual cache management
removed = clear_cache()  # Remove all cached responses
```

Per-session spending is also available on any client (see also
[Billing & Cost Tracking](#billing--cost-tracking) for the full surface):

```python
from blockrun_llm import LLMClient

client = LLMClient()
response = client.chat("openai/gpt-5.2", "Hello!")

spending = client.get_spending()
print(f"Session: ${spending['total_usd']:.4f} across {spending['calls']} calls")
```

## Anthropic SDK Compatibility

Use the official Anthropic Python SDK with BlockRun's API gateway and automatic x402 payments:

```bash
pip install blockrun-llm[anthropic]
```

```python
from blockrun_llm import AnthropicClient

client = AnthropicClient()  # Auto-detects wallet, auto-pays

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)

# Works with any BlockRun model in Anthropic format
response = client.messages.create(
    model="openai/gpt-5.4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello from GPT!"}]
)
```

The `AnthropicClient` wraps `anthropic.Anthropic` with a custom httpx transport that handles x402 payment signing transparently. Your private key never leaves your machine.

## Links

- [Website](https://blockrun.ai)
- [Documentation](https://github.com/BlockRunAI/awesome-blockrun/tree/main/docs)
- [GitHub](https://github.com/blockrunai/blockrun-llm)
- [Telegram](https://t.me/+mroQv4-4hGgzOGUx)

## Frequently Asked Questions

### What is blockrun-llm?
blockrun-llm is a Python SDK that provides pay-per-request access to 43+ large language models from OpenAI, Anthropic, Google, DeepSeek, NVIDIA, ZAI, and more. It uses the x402 protocol for automatic USDC micropayments — no API keys, no subscriptions, no vendor lock-in.

### How does payment work?
When you make an API call, the SDK automatically handles x402 payment. It signs a USDC transaction locally using your wallet private key (which never leaves your machine), and includes the payment proof in the request header. Settlement is non-custodial and instant on Base or Solana.

### What is smart routing / Router Core?
Router Core is BlockRun's built-in routing engine — shared with the TypeScript SDK and the gateway, so the same request routes the same way everywhere. It scores your request across <!-- br:clawrouter.dimensions -->15<!-- /br:clawrouter.dimensions --> dimensions, drops every model that can't actually handle it (context, output length, tools, vision), then picks the cheapest capable one and keeps the rest as a fallback chain. Routing happens locally in under 1ms and makes no extra model call. It can save up to <!-- br:savings.autoVsBaselinePct -->84<!-- /br:savings.autoVsBaselinePct -->% on LLM costs compared to using premium models for every request.

### How much does it cost?
Pay only for what you use. Prices start at **FREE** (11 NVIDIA-hosted models). Paid models start at $0.10/M tokens. There are no minimums, subscriptions, or monthly fees. $5 in USDC gets you thousands of requests.

### Can I use it with Solana?
Yes. Install with `pip install blockrun-llm[solana]` and use `SolanaLLMClient` instead of `LLMClient`. Same API, different payment chain.

## License

MIT

### Changing payment methods safely

Register at [user.blockrun.ai](https://user.blockrun.ai), add credit in
[Credits](https://user.blockrun.ai/dashboard/credits), and create a key in
[API keys](https://user.blockrun.ai/dashboard/keys). Set `BLOCKRUN_API_KEY`
or pass the key as the client's credential. Check activity and actual charges
in the dashboard; local cost summaries may omit charges without a gateway receipt.

An explicit wallet credential chooses wallet payments even when `BLOCKRUN_API_KEY`
is set. Choose the Solana wallet client for Solana, or the Base wallet client for
Base. An empty or invalid `BLOCKRUN_API_KEY` fails instead of silently selecting
a wallet. Unset it to restore automatic wallet selection. Create a new client
when changing credentials; an existing client keeps its original account.

The optional `AnthropicClient` also accepts a BlockRun API key as its credential
or through `BLOCKRUN_API_KEY` (`pip install 'blockrun-llm[anthropic]'`). In account
mode, automatic retries are disabled by default because a failed response may
follow a billable request; callers can explicitly configure `max_retries`.

The optional integration currently supports Anthropic SDK **0.x**. The extra
pins `<1` because Anthropic 1.x moved to a different HTTP transport; upgrading
that dependency independently would break both account and wallet clients.
