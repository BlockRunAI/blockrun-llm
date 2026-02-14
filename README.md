# BlockRun LLM SDK

Pay-per-request access to GPT-5.2, Claude 4, Gemini 2.5, Grok, and more via x402 micropayments.

**BlockRun assumes Claude Code as the agent runtime.**

## Supported Chains

| Chain | Network | Payment | Status |
|-------|---------|---------|--------|
| **Base** | Base Mainnet (Chain ID: 8453) | USDC | ✅ Primary |
| **Base Testnet** | Base Sepolia (Chain ID: 84532) | Testnet USDC | ✅ Development |
| **XRPL** | XRP Ledger Mainnet | RLUSD | ✅ New |

**Protocol:** x402 v2

## Installation

```bash
pip install blockrun-llm
```

## Quick Start

```python
from blockrun_llm import LLMClient

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)
response = client.chat("openai/gpt-5.2", "Hello!")
```

That's it. The SDK handles x402 payment automatically.

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
print(result.model)  # 'xai/grok-4-1-fast-reasoning'
```

### Routing Profiles

| Profile | Description | Best For |
|---------|-------------|----------|
| `free` | nvidia/gpt-oss-120b only (FREE) | Testing, development |
| `eco` | Cheapest models per tier (DeepSeek, xAI) | Cost-sensitive production |
| `auto` | Best balance of cost/quality (default) | General use |
| `premium` | Top-tier models (OpenAI, Anthropic) | Quality-critical tasks |

```python
# Use premium models for complex tasks
result = client.smart_chat(
    "Write production-grade async Python code",
    routing_profile="premium"
)
print(result.model)  # 'anthropic/claude-opus-4.5'
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
| MEDIUM | Code snippets, explanations | xai/grok-code-fast-1 |
| COMPLEX | Architecture, long documents | google/gemini-3-pro-preview |
| REASONING | Proofs, multi-step reasoning | xai/grok-4-1-fast-reasoning |

## How It Works

1. You send a request to BlockRun's API
2. The API returns a 402 Payment Required with the price
3. The SDK automatically signs a USDC payment on Base
4. The request is retried with the payment proof
5. You receive the AI response

**Your private key never leaves your machine** - it's only used for local signing.

## Available Models

### OpenAI GPT-5 Family
| Model | Input Price | Output Price |
|-------|-------------|--------------|
| `openai/gpt-5.2` | $1.75/M | $14.00/M |
| `openai/gpt-5-mini` | $0.25/M | $2.00/M |
| `openai/gpt-5-nano` | $0.05/M | $0.40/M |
| `openai/gpt-5.2-pro` | $21.00/M | $168.00/M |

### OpenAI GPT-4 Family
| Model | Input Price | Output Price |
|-------|-------------|--------------|
| `openai/gpt-4.1` | $2.00/M | $8.00/M |
| `openai/gpt-4.1-mini` | $0.40/M | $1.60/M |
| `openai/gpt-4.1-nano` | $0.10/M | $0.40/M |
| `openai/gpt-4o` | $2.50/M | $10.00/M |
| `openai/gpt-4o-mini` | $0.15/M | $0.60/M |

### OpenAI O-Series (Reasoning)
| Model | Input Price | Output Price |
|-------|-------------|--------------|
| `openai/o1` | $15.00/M | $60.00/M |
| `openai/o1-mini` | $1.10/M | $4.40/M |
| `openai/o3` | $2.00/M | $8.00/M |
| `openai/o3-mini` | $1.10/M | $4.40/M |
| `openai/o4-mini` | $1.10/M | $4.40/M |

### Testnet Models (Base Sepolia)
| Model | Price |
|-------|-------|
| `openai/gpt-oss-20b` | $0.001/request |
| `openai/gpt-oss-120b` | $0.002/request |

*Testnet models use flat pricing (no token counting) for simplicity.*

### Anthropic Claude
| Model | Input Price | Output Price |
|-------|-------------|--------------|
| `anthropic/claude-opus-4.5` | $5.00/M | $25.00/M |
| `anthropic/claude-opus-4` | $15.00/M | $75.00/M |
| `anthropic/claude-sonnet-4` | $3.00/M | $15.00/M |
| `anthropic/claude-haiku-4.5` | $1.00/M | $5.00/M |

### Google Gemini
| Model | Input Price | Output Price |
|-------|-------------|--------------|
| `google/gemini-3-pro-preview` | $2.00/M | $12.00/M |
| `google/gemini-2.5-pro` | $1.25/M | $10.00/M |
| `google/gemini-2.5-flash` | $0.15/M | $0.60/M |

### DeepSeek
| Model | Input Price | Output Price |
|-------|-------------|--------------|
| `deepseek/deepseek-chat` | $0.28/M | $0.42/M |
| `deepseek/deepseek-reasoner` | $0.28/M | $0.42/M |

### xAI Grok
| Model | Input Price | Output Price | Context | Notes |
|-------|-------------|--------------|---------|-------|
| `xai/grok-3` | $3.00/M | $15.00/M | 131K | Flagship |
| `xai/grok-3-fast` | $5.00/M | $25.00/M | 131K | Tool calling optimized |
| `xai/grok-3-mini` | $0.30/M | $0.50/M | 131K | Fast & affordable |
| `xai/grok-4-1-fast-reasoning` | $0.20/M | $0.50/M | **2M** | Latest, chain-of-thought |
| `xai/grok-4-1-fast-non-reasoning` | $0.20/M | $0.50/M | **2M** | Latest, direct response |
| `xai/grok-4-fast-reasoning` | $0.20/M | $0.50/M | **2M** | Step-by-step reasoning |
| `xai/grok-4-fast-non-reasoning` | $0.20/M | $0.50/M | **2M** | Quick responses |
| `xai/grok-code-fast-1` | $0.20/M | $1.50/M | 256K | Code generation |
| `xai/grok-4-0709` | $0.20/M | $1.50/M | 256K | Premium quality |
| `xai/grok-2-vision` | $2.00/M | $10.00/M | 32K | Vision capabilities |

### Moonshot Kimi
| Model | Input Price | Output Price |
|-------|-------------|--------------|
| `moonshot/kimi-k2.5` | $0.50/M | $2.40/M |

### NVIDIA (Free & Hosted)
| Model | Input Price | Output Price | Notes |
|-------|-------------|--------------|-------|
| `nvidia/gpt-oss-120b` | **FREE** | **FREE** | OpenAI open-weight 120B (Apache 2.0) |
| `nvidia/kimi-k2.5` | $0.55/M | $2.50/M | Moonshot 1T MoE with vision |

### E2E Verified Models

All models below have been tested end-to-end via the Python SDK (Feb 2026):

| Provider | Model | Status |
|----------|-------|--------|
| OpenAI | `openai/gpt-4o-mini` | Passed |
| Anthropic | `anthropic/claude-sonnet-4` | Passed |
| Google | `google/gemini-2.5-flash` | Passed |
| DeepSeek | `deepseek/deepseek-chat` | Passed |
| xAI | `xai/grok-3-fast` | Passed |
| Moonshot | `moonshot/kimi-k2.5` | Passed |

### Image Generation
| Model | Price |
|-------|-------|
| `openai/dall-e-3` | $0.04-0.08/image |
| `openai/gpt-image-1` | $0.02-0.04/image |
| `black-forest/flux-1.1-pro` | $0.04/image |
| `google/nano-banana` | $0.05/image |
| `google/nano-banana-pro` | $0.10-0.15/image |

## Usage Examples

### Simple Chat

```python
from blockrun_llm import LLMClient

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)

response = client.chat("openai/gpt-5.2", "Explain quantum computing")
print(response)

# With system prompt
response = client.chat(
    "anthropic/claude-sonnet-4",
    "Write a haiku",
    system="You are a creative poet."
)
```

### Real-time X/Twitter Search (xAI Live Search)

**Note:** Live Search can take 30-120+ seconds as it searches multiple sources. The SDK automatically uses a 5-minute timeout for search requests.

```python
from blockrun_llm import LLMClient

client = LLMClient()

# Simple: Enable live search with search=True (default 10 sources, ~$0.26)
response = client.chat(
    "xai/grok-3",
    "What are the latest posts from @blockrunai?",
    search=True
)
print(response)

# Custom: Limit sources to reduce cost (5 sources, ~$0.13)
response = client.chat(
    "xai/grok-3",
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
            client.chat("anthropic/claude-sonnet-4", "What is 3+3?"),
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

## XRPL Chain (RLUSD Payments)

BlockRun now supports payments with RLUSD on the XRP Ledger. Same models, same API - just a different payment rail.

```python
from blockrun_llm import xrpl_client

# Create XRPL client (pays with RLUSD)
client = xrpl_client()  # Uses BLOCKRUN_WALLET_KEY

# Chat with any model
response = client.chat("openai/gpt-4o", "Hello!")
print(response)

# Check RLUSD balance
balance = client.get_balance()
print(f"RLUSD: ${balance:.4f}")
```

### Async XRPL Usage

```python
import asyncio
from blockrun_llm import async_xrpl_client

async def main():
    async with async_xrpl_client() as client:
        response = await client.chat("openai/gpt-4o", "Hello!")
        print(response)

asyncio.run(main())
```

### Manual XRPL Configuration

```python
from blockrun_llm import LLMClient

# Or configure manually
client = LLMClient(api_url="https://xrpl.blockrun.ai/api")
response = client.chat("openai/gpt-4o", "Hello!")
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

## Links

- [Website](https://blockrun.ai)
- [Documentation](https://docs.blockrun.ai)
- [GitHub](https://github.com/blockrun/blockrun-llm)
- [Discord](https://discord.gg/blockrun)

## License

MIT
