# BlockRun LLM SDK

Pay-per-request access to GPT-4o, Claude 4, Gemini 2.5, and more via x402 micropayments on Base.

**Network:** Base (Chain ID: 8453)
**Payment:** USDC
**Protocol:** x402 v2 (CDP Facilitator)

## Installation

```bash
pip install blockrun-llm
```

## Quick Start

```python
from blockrun_llm import LLMClient

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)
response = client.chat("openai/gpt-4o", "Hello!")
```

That's it. The SDK handles x402 payment automatically.

## How It Works

1. You send a request to BlockRun's API
2. The API returns a 402 Payment Required with the price
3. The SDK automatically signs a USDC payment on Base
4. The request is retried with the payment proof
5. You receive the AI response

**Your private key never leaves your machine** - it's only used for local signing.

## Available Models

| Model | Provider | Input Price | Output Price |
|-------|----------|-------------|--------------|
| `openai/gpt-5.2` | OpenAI | $1.75/M | $14.00/M |
| `openai/gpt-4o` | OpenAI | $2.50/M | $10.00/M |
| `openai/gpt-4o-mini` | OpenAI | $0.15/M | $0.60/M |
| `openai/o1` | OpenAI | $15.00/M | $60.00/M |
| `openai/o1-mini` | OpenAI | $3.00/M | $12.00/M |
| `anthropic/claude-opus-4` | Anthropic | $15.00/M | $75.00/M |
| `anthropic/claude-sonnet-4` | Anthropic | $3.00/M | $15.00/M |
| `anthropic/claude-haiku-4.5` | Anthropic | $1.00/M | $5.00/M |
| `google/gemini-3-pro-preview` | Google | $2.00/M | $12.00/M |
| `google/gemini-2.5-pro` | Google | $1.25/M | $10.00/M |
| `google/gemini-2.5-flash` | Google | $0.15/M | $0.60/M |
| `x-ai/grok-4-fast` | xAI | $0.20/M | $0.50/M |
| `deepseek/deepseek-v3-0324` | DeepSeek | $0.20/M | $0.88/M |
| `deepseek/deepseek-r1` | DeepSeek | $0.55/M | $2.19/M |
| `meta-llama/llama-3.3-70b-instruct` | Meta | $0.12/M | $0.30/M |
| `meta-llama/llama-3.1-405b-instruct` | Meta | $2.00/M | $2.00/M |
| `qwen/qwen-2.5-72b-instruct` | Qwen | $0.07/M | $0.26/M |
| `mistralai/mistral-large` | Mistral | $2.00/M | $6.00/M |

## Usage Examples

### Simple Chat

```python
from blockrun_llm import LLMClient

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)

response = client.chat("openai/gpt-4o", "Explain quantum computing")
print(response)

# With system prompt
response = client.chat(
    "anthropic/claude-sonnet-4",
    "Write a haiku",
    system="You are a creative poet."
)
```

### Full Chat Completion

```python
from blockrun_llm import LLMClient

client = LLMClient()  # Uses BLOCKRUN_WALLET_KEY (never sent to server)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I read a file in Python?"}
]

result = client.chat_completion("openai/gpt-4o", messages)
print(result.choices[0].message.content)
```

### Async Usage

```python
import asyncio
from blockrun_llm import AsyncLLMClient

async def main():
    async with AsyncLLMClient() as client:
        # Simple chat
        response = await client.chat("openai/gpt-4o", "Hello!")
        print(response)

        # Multiple requests concurrently
        tasks = [
            client.chat("openai/gpt-4o", "What is 2+2?"),
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

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BLOCKRUN_WALLET_KEY` | Your EVM wallet private key | Yes (or pass to constructor) |
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
    response = client.chat("openai/gpt-4o", "Hello!")
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
