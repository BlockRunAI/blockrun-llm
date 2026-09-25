# BlockRun LLM SDK (Python)

Python SDK for <!-- br:models.chatVisible -->79<!-- /br:models.chatVisible --> LLMs plus image/video/music/speech generation, standalone search, multi-chain RPC, and Pyth-backed market data — paid two ways: a BlockRun API key drawing on prepaid account credit, or USDC micropayments via x402 where the wallet signature is the authentication and the key never leaves your machine.

## Commands

```bash
pip install -e ".[dev]"           # install in dev mode
pip install -e ".[dev,solana]"    # with Solana support
pytest                            # run tests
black blockrun_llm/               # format code
ruff check blockrun_llm/          # lint
mypy blockrun_llm/                # type check
```

## Project structure

```
blockrun_llm/
├── __init__.py              # Package exports
├── client.py                # LLMClient (EVM: Base, Arc — the 402's network picks the chain)
├── solana_client.py         # SolanaLLMClient
├── wallet.py                # EVM wallet management
├── solana_wallet.py         # Solana wallet management
├── x402.py                  # x402 payment protocol
├── router_core/             # Port of @blockrun/router-core (shared with the TS SDK + gateway)
├── router_adapter.py        # Host glue: catalog ids, payment floors, free profile
├── router.py                # Back-compat shim over router_core
├── types.py                 # Type definitions
├── validation.py            # Input validation
├── cache.py                 # Response caching
├── image.py                 # Image generation (+ image-to-image)
├── music.py                 # Music generation
├── speech.py                # Text-to-speech + sound effects (BlockRun Voice / ElevenLabs)
├── video.py                 # Video generation
├── portrait.py              # Virtual Portrait enrollment (AI characters)
├── realface.py              # RealFace enrollment (real-person likeness)
├── search.py                # Standalone Grok Live Search
├── price.py                 # Pyth market data (crypto/fx/commodity/stocks)
├── rpc.py                   # Multi-chain JSON-RPC (Tatum gateway, 40+ chains)
└── anthropic_client.py      # Anthropic-compatible client
```

## Key dependencies

- `httpx` — HTTP client
- `eth-account` — Ethereum wallet
- `pydantic` — Data validation
- `x402[svm]` — Solana x402 payments (optional)

## Supported chains

- Base Mainnet (primary) — USDC
- Arc (Circle, chain 5042) — USDC, via `api_url="https://arc.blockrun.ai/api"`; same `LLMClient` and key
- Base Sepolia (testnet) — Testnet USDC
- Solana Mainnet — USDC SPL

The EVM domain signed follows the 402's `network` through `EVM_NETWORKS` in `blockrun_llm/x402.py`; a 402's `extra` is never trusted for it, an unknown network and a non-USDC `asset` are refused.

## Conventions

- Python >= 3.9
- Format with Black (line-length 100)
- Lint with Ruff (line-length 100)
- Type check with mypy (strict)
- MIT license
- PyPI: `blockrun-llm`
