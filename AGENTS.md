# AGENTS.md

Guidance for AI coding agents working with the BlockRun Python SDK.

## Project Overview

**blockrun-llm** is a Python SDK for pay-per-request access to AI models (GPT, Claude, Gemini, DeepSeek, NVIDIA) via x402 micropayments on Base. **Includes 8 fully-free NVIDIA-hosted models** — DeepSeek V4 Flash (1M ctx), Nemotron Nano Omni (vision), Qwen3 Next + Coder, Llama 4 Maverick, Mistral Small 4, plus `gpt-oss-120b/20b` (hidden from `/v1/models` but direct calls still work). Accessible via `routing_profile="free"` or any `nvidia/*` model id.

**Package:** `blockrun-llm` (PyPI)
**Python:** >=3.9
**Network:** Base (Chain ID: 8453)
**Payment:** USDC via x402 v2 (or $0 for `nvidia/*` free tier)

## Repository Structure

```
blockrun-llm/
├── blockrun_llm/
│   ├── __init__.py          # Package exports
│   ├── client.py            # LLMClient, AsyncLLMClient
│   ├── anthropic_client.py  # AnthropicClient (official SDK wrapper)
│   ├── solana_client.py     # SolanaLLMClient (Solana payments)
│   ├── router.py            # ClawRouter smart routing
│   ├── image.py             # Image generation client
│   ├── types.py             # Pydantic models and type definitions
│   ├── validation.py        # Input validation utilities
│   ├── wallet.py            # Wallet operations (signing, address)
│   ├── solana_wallet.py     # Solana wallet utilities
│   ├── cache.py             # Response caching & cost logging
│   └── x402.py              # x402 payment protocol implementation
├── tests/
│   ├── unit/          # Unit tests (no API calls)
│   └── integration/   # Integration tests (requires funded wallet)
├── examples/          # Usage examples
├── pyproject.toml     # Package configuration (hatchling)
└── README.md
```

## Development Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Testing
pytest tests/unit           # Unit tests only (no API key needed)
pytest tests/unit --cov     # With coverage
pytest                      # All tests (requires BLOCKRUN_WALLET_KEY)

# Code Quality
black blockrun_llm/         # Format code
ruff check blockrun_llm/    # Lint
mypy blockrun_llm/          # Type check
```

## Code Conventions

### Style
- Black formatter (line-length: 100)
- Ruff linter
- Type hints required (mypy strict mode)

### Architecture
- `LLMClient` - Synchronous client
- `AsyncLLMClient` - Async client with context manager
- All API calls go through x402 payment flow

### Error Handling
- `APIError` - General API errors
- `PaymentError` - Payment-specific errors
- Errors are sanitized to prevent key leakage

## Key Files

| File | Purpose |
|------|---------|
| `client.py` | Main client classes with `chat()`, `chat_completion()`, `list_models()` |
| `x402.py` | x402 payment protocol (402 handling, payment signing) |
| `wallet.py` | Private key management, transaction signing |
| `validation.py` | Input validation for keys, URLs, parameters |
| `types.py` | Pydantic models for API requests/responses |

## Testing

### Unit Tests
No API key or funded wallet required:
```bash
pytest tests/unit -v
```

### Integration Tests
Requires `BLOCKRUN_WALLET_KEY` with funded Base wallet (~$1 USDC):
```bash
export BLOCKRUN_WALLET_KEY=0x...
pytest tests/integration -v
```

### End-to-End Model Sweeps
Before a release or after router/catalog changes:
```bash
python examples/sweep_all_chat_models.py --output-json sweep-results.json
python examples/sweep_all_media_models.py --output-json sweep-media-results.json
```
Each script captures per-model status / latency / token counts / per-call
cost and exits non-zero if any expected-to-work model fails. The chat sweep
also runs a forward-compat diff against `/v1/models` to flag new IDs not in
the sweep list. Video is excluded from the media sweep by design.

## Publishing

```bash
# Build
python -m build

# Upload to PyPI
twine upload dist/*
```

## Security Notes

- Private keys never leave the machine (local signing only)
- Validate private key format before use
- HTTPS required for production API URLs
- Never log or expose private keys in errors
