# BlockRun LLM SDK (Python)

> **blockrun-llm** is a Python SDK for accessing 80+ large language models (GPT-5.x, Claude 4.x, Gemini 3.x, DeepSeek, Grok 4.x, NVIDIA free-tier, GLM, MiniMax, Moonshot and more) plus image / video / music generation, Grok Live Search, X/Twitter APIs, and Pyth-backed market data — all with automatic pay-per-request USDC micropayments via the x402 protocol. No API keys required; your wallet signature is your authentication. Built for AI agents that need to operate autonomously.

[![PyPI](https://img.shields.io/pypi/v/blockrun-llm.svg)](https://pypi.org/project/blockrun-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BlockRun assumes Claude Code as the agent runtime.**

## Supported Chains

| Chain | Network | Payment | Status |
|-------|---------|---------|--------|
| **Base** | Base Mainnet (Chain ID: 8453) | USDC | ✅ Primary |
| **Base Testnet** | Base Sepolia (Chain ID: 84532) | Testnet USDC | ✅ Development |
| **Solana** | Solana Mainnet | USDC (SPL) | ✅ New |

> **XRPL (RLUSD):** Use [blockrun-llm-xrpl](https://pypi.org/project/blockrun-llm-xrpl/) for XRPL payments

**Protocol:** x402 v2

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

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)
response = client.chat("openai/gpt-5.2", "Hello!")
```

That's it. The SDK handles x402 payment automatically.

## Solana Support

Pay for AI calls with Solana USDC via [sol.blockrun.ai](https://sol.blockrun.ai):

```python
from blockrun_llm import SolanaLLMClient

# SOLANA_WALLET_KEY env var (bs58-encoded Solana secret key)
client = SolanaLLMClient()

# Or pass key directly
client = SolanaLLMClient(private_key="your-bs58-solana-key")

# Same API as LLMClient
response = client.chat("openai/gpt-5.2", "gm Solana")
print(response)

# DeepSeek on Solana
answer = client.chat("deepseek/deepseek-chat", "Explain Solana consensus", temperature=0.5)
```

**Setup:**
```bash
pip install blockrun-llm[solana]
export SOLANA_WALLET_KEY="your-bs58-solana-key"
```

**Endpoint:** `https://sol.blockrun.ai/api`
**Payment:** Solana USDC (SPL Token, mainnet)

## Smart Routing (ClawRouter)

Let the SDK automatically pick the cheapest capable model for each request:

```python
from blockrun_llm import LLMClient

client = LLMClient()

# Auto-routes to cheapest capable model
result = client.smart_chat("What is 2+2?")
print(result.response)  # '4'
print(result.model)     # 'nvidia/kimi-k2.5' (cheap, fast)
print(f"Saved {result.routing.savings * 100:.0f}%")  # 'Saved 94%'

# Complex reasoning task -> routes to reasoning model
result = client.smart_chat("Prove the Riemann hypothesis step by step")
print(result.model)  # 'deepseek/deepseek-reasoner'
```

### Routing Profiles

| Profile | Description | Best For |
|---------|-------------|----------|
| `free` | nvidia/gpt-oss-120b only (FREE) | Testing, development |
| `eco` | Cheapest models per tier (DeepSeek, NVIDIA) | Cost-sensitive production |
| `auto` | Best balance of cost/quality (default) | General use |
| `premium` | Top-tier models (OpenAI, Anthropic) | Quality-critical tasks |

```python
# Use premium models for complex tasks
result = client.smart_chat(
    "Write production-grade async Python code",
    routing_profile="premium"
)
print(result.model)  # 'openai/gpt-5.4'
```

### How It Works

ClawRouter uses a 14-dimension rule-based classifier to analyze each request:

- **Token count** - Short vs long prompts
- **Code presence** - Programming keywords
- **Reasoning markers** - "prove", "step by step", etc.
- **Technical terms** - Architecture, optimization, etc.
- **Creative markers** - Story, poem, brainstorm, etc.
- **Agentic patterns** - Multi-step, tool use indicators

The classifier runs in <1ms, 100% locally, and routes to one of four tiers:

| Tier | Example Tasks | Auto Profile Model |
|------|---------------|-------------------|
| SIMPLE | "What is 2+2?", definitions | nvidia/kimi-k2.5 |
| MEDIUM | Code snippets, explanations | google/gemini-2.5-flash |
| COMPLEX | Architecture, long documents | google/gemini-3.1-pro |
| REASONING | Proofs, multi-step reasoning | deepseek/deepseek-reasoner |

## How It Works

1. You send a request to BlockRun's API
2. The API returns a 402 Payment Required with the price
3. The SDK automatically signs a USDC payment on Base
4. The request is retried with the payment proof
5. You receive the AI response

**Your private key never leaves your machine** - it's only used for local signing.

## Available Models

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

### OpenAI O-Series (Reasoning)
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `openai/o1` | $15.00/M | $60.00/M | 200K |
| `openai/o1-mini` | $1.10/M | $4.40/M | 128K |
| `openai/o3` | $2.00/M | $8.00/M | 200K |
| `openai/o3-mini` | $1.10/M | $4.40/M | 128K |

### Anthropic Claude
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `anthropic/claude-opus-4.6` | $5.00/M | $25.00/M | 200K |
| `anthropic/claude-opus-4.5` | $5.00/M | $25.00/M | 200K |
| `anthropic/claude-sonnet-4.6` | $3.00/M | $15.00/M | 200K |
| `anthropic/claude-haiku-4.5` | $1.00/M | $5.00/M | 200K |

### Google Gemini
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `google/gemini-3.1-pro` | $2.00/M | $12.00/M | 1M |
| `google/gemini-3-pro-preview` | $2.00/M | $12.00/M | 1M |
| `google/gemini-3-flash-preview` | $0.50/M | $3.00/M | 1M |
| `google/gemini-2.5-pro` | $1.25/M | $10.00/M | 1M |
| `google/gemini-2.5-flash` | $0.30/M | $2.50/M | 1M |
| `google/gemini-3.1-flash-lite` | $0.25/M | $1.50/M | 1M |
| `google/gemini-2.5-flash-lite` | $0.10/M | $0.40/M | 1M |

### DeepSeek
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `deepseek/deepseek-chat` | $0.28/M | $0.42/M | 128K |
| `deepseek/deepseek-reasoner` | $0.28/M | $0.42/M | 128K |

### MiniMax
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `minimax/minimax-m2.7` | $0.30/M | $1.20/M | 200K |

### ZAI
| Model | Input Price | Output Price | Context |
|-------|-------------|--------------|---------|
| `zai/glm-5` | $1.00/M | $3.20/M | 200K |
| `zai/glm-5-turbo` | $1.20/M | $4.00/M | 200K |

### NVIDIA (Free & Hosted)
| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `nvidia/nemotron-ultra-253b` | **FREE** | **FREE** | 131K | NVIDIA's largest reasoning model |
| `nvidia/nemotron-3-super-120b` | **FREE** | **FREE** | 131K | General-purpose 120B |
| `nvidia/nemotron-super-49b` | **FREE** | **FREE** | 131K | Efficient 49B |
| `nvidia/mistral-large-3-675b` | **FREE** | **FREE** | 131K | Mistral Large 675B |
| `nvidia/qwen3-coder-480b` | **FREE** | **FREE** | 131K | Code generation 480B |
| `nvidia/devstral-2-123b` | **FREE** | **FREE** | 131K | Dev-focused 123B |
| `nvidia/deepseek-v3.2` | **FREE** | **FREE** | 131K | DeepSeek V3.2 hosted |
| `nvidia/glm-4.7` | **FREE** | **FREE** | 131K | GLM-4.7 hosted |
| `nvidia/llama-4-maverick` | **FREE** | **FREE** | 131K | Meta Llama 4 Maverick |
| `nvidia/gpt-oss-120b` | **FREE** | **FREE** | 128K | OpenAI open-weight 120B |
| `nvidia/gpt-oss-20b` | **FREE** | **FREE** | 128K | OpenAI open-weight 20B |
| `moonshot/kimi-k2.6` | $0.95/M | $4.00/M | 256K | Moonshot flagship (vision + reasoning_content) |
| `nvidia/kimi-k2.5` | FREE | FREE | 1M | Moonshot 1T MoE hosted by NVIDIA (free tier) |

### Testnet Models (Base Sepolia)
| Model | Price |
|-------|-------|
| `openai/gpt-oss-20b` | $0.001/request |
| `openai/gpt-oss-120b` | $0.002/request |

*Testnet models use flat pricing (no token counting) for simplicity.*

### E2E Verified Models

All models below have been tested end-to-end via the Python SDK (Mar 2026):

| Provider | Model | Status |
|----------|-------|--------|
| OpenAI | `openai/gpt-5.2` | Passed |
| Anthropic | `anthropic/claude-opus-4.6` | Passed |
| Anthropic | `anthropic/claude-sonnet-4.6` | Passed |
| Google | `google/gemini-2.5-flash` | Passed |
| DeepSeek | `deepseek/deepseek-chat` | Passed |
| NVIDIA | `nvidia/gpt-oss-120b` | Passed |

### Image Generation
| Model | Price |
|-------|-------|
| `openai/dall-e-3` | $0.04-0.08/image |
| `openai/gpt-image-1` | $0.02-0.04/image |
| `black-forest/flux-1.1-pro` | $0.04/image |
| `google/nano-banana` | $0.05/image |
| `google/nano-banana-pro` | $0.10-0.15/image |
| `xai/grok-imagine-image` | $0.02/image |
| `xai/grok-imagine-image-pro` | $0.07/image |
| `zai/cogview-4` | $0.015/image |

### Video Generation
| Model | Price |
|-------|-------|
| `xai/grok-imagine-video` | $0.05/sec (8s default → $0.42/clip) |

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
```

## X/Twitter Data (Powered by AttentionVC)

Access X/Twitter user profiles, followers, and followings via [AttentionVC](https://attentionvc.ai) partner API. No API keys needed — pay-per-request via x402.

```python
from blockrun_llm import LLMClient

client = LLMClient()

# Look up user profiles ($0.002/user, min $0.02)
users = client.x_user_lookup(["elonmusk", "blockaborr"])
for user in users.users:
    print(f"@{user.userName}: {user.followers} followers")

# Get followers ($0.05/page, ~200 accounts)
result = client.x_followers("blockaborr")
for f in result.followers:
    print(f"  @{f.screen_name}")

# Paginate through all followers
while result.has_next_page:
    result = client.x_followers("blockaborr", cursor=result.next_cursor)

# Get followings ($0.05/page)
followings = client.x_followings("blockaborr")
```

Works on all clients: `LLMClient` (Base), `AsyncLLMClient`, and `SolanaLLMClient`.

## Prediction Markets (Powered by Predexon)

Access real-time prediction market data from Polymarket, Kalshi, and Binance Futures via [Predexon](https://predexon.com). No API keys needed — pay-per-request via x402.

### Polymarket

```python
from blockrun_llm import LLMClient

client = LLMClient()

# List markets with optional filters ($0.001/request)
markets = client.pm("polymarket/markets")
markets = client.pm("polymarket/markets", status="active", limit=10)
markets = client.pm("polymarket/markets", search="bitcoin")

# List events ($0.001/request)
events = client.pm("polymarket/events")

# Historical trades ($0.001/request)
trades = client.pm("polymarket/trades")

# OHLCV candlestick data for a specific condition ($0.001/request)
candles = client.pm("polymarket/candlesticks/0x1234abcd...")

# Wallet profile ($0.005/request — tier 2)
profile = client.pm("polymarket/wallet/0xABC123...")

# Wallet P&L ($0.005/request — tier 2)
pnl = client.pm("polymarket/wallet/pnl/0xABC123...")

# Global leaderboard ($0.001/request)
leaderboard = client.pm("polymarket/leaderboard")
```

### Kalshi & Binance

```python
# Kalshi markets ($0.001/request)
kalshi_markets = client.pm("kalshi/markets")

# Kalshi trades ($0.001/request)
kalshi_trades = client.pm("kalshi/trades")

# Binance candles for supported pairs ($0.001/request)
btc_candles = client.pm("binance/candles/BTCUSDT")
eth_candles = client.pm("binance/candles/ETHUSDT")
# Also: SOLUSDT, XRPUSDT
```

### Cross-Platform

```python
# Cross-platform matching pairs ($0.001/request)
pairs = client.pm("matching-markets/pairs")
```

All current endpoints are GET. The `pm_query()` method is available for future POST endpoints.

Works on all clients: `LLMClient` (Base), `AsyncLLMClient`, and `SolanaLLMClient`.

## Exa Web Search (Powered by Exa)

Access [Exa](https://exa.ai)'s neural web search via x402. No API keys needed — pay-per-request via Solana USDC. Available on `SolanaLLMClient` only.

| Endpoint | Method | Price |
|---|---|---|
| `exa_search` | Neural/keyword web search | $0.01/request |
| `exa_find_similar` | Find semantically similar pages | $0.01/request |
| `exa_contents` | Extract full text from URLs | $0.002/URL |
| `exa_answer` | AI answer grounded in web search | $0.01/request |

```python
from blockrun_llm import SolanaLLMClient

client = SolanaLLMClient()

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

`SolanaLLMClient` only — Exa endpoints are on `sol.blockrun.ai`.

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

Edit existing images with text prompts:

```python
from blockrun_llm import LLMClient, ImageClient

# Via LLMClient
client = LLMClient()
result = client.image_edit(
    prompt="Make the sky purple and add northern lights",
    image="data:image/png;base64,...",  # base64 or URL
    model="openai/gpt-image-1",
)
print(result.data[0].url)

# Via ImageClient
img_client = ImageClient()
result = img_client.edit("Add a rainbow", image="https://example.com/photo.jpg")
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

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BLOCKRUN_WALLET_KEY` | Your Base chain wallet private key | Yes (or pass to constructor) |
| `BLOCKRUN_API_URL` | API endpoint | No (default: https://blockrun.ai/api) |

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

## Wallet Scanning

The SDK auto-detects wallets from any provider on your system:

```python
from blockrun_llm.wallet import scan_wallets
from blockrun_llm.solana_wallet import scan_solana_wallets

# Scans ~/.<dir>/wallet.json for Base wallets
base_wallets = scan_wallets()

# Scans ~/.<dir>/solana-wallet.json
sol_wallets = scan_solana_wallets()
```

`get_or_create_wallet()` checks scanned wallets first, so if you already have a wallet from another BlockRun tool, it will be reused automatically.

## Response Caching

The SDK caches responses to avoid duplicate payments:

```python
from blockrun_llm import clear_cache

# Automatic TTLs by endpoint:
# - X/Twitter: 1 hour
# - Search: 15 minutes
# - Models: 24 hours
# - Chat/Image: no cache (every call is unique)

# Manual cache management
removed = clear_cache()  # Remove all cached responses
```

## Cost Logging

Track spending across sessions:

```python
from blockrun_llm import get_cost_log_summary

# Costs are logged to ~/.blockrun/cost_log.jsonl
summary = get_cost_log_summary()
print(f"Total: ${summary['total_usd']:.2f}")
print(f"Calls: {summary['calls']}")
print(f"By endpoint: {summary['by_endpoint']}")
```

Per-session spending is also available on any client:

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

### What is smart routing / ClawRouter?
ClawRouter is a built-in smart routing engine that analyzes your request across 14 dimensions and automatically picks the cheapest model capable of handling it. Routing happens locally in under 1ms. It can save up to 92% on LLM costs compared to using premium models for every request.

### How much does it cost?
Pay only for what you use. Prices start at **FREE** (11 NVIDIA-hosted models). Paid models start at $0.10/M tokens. There are no minimums, subscriptions, or monthly fees. $5 in USDC gets you thousands of requests.

### Can I use it with Solana?
Yes. Install with `pip install blockrun-llm[solana]` and use `SolanaLLMClient` instead of `LLMClient`. Same API, different payment chain.

## License

MIT
