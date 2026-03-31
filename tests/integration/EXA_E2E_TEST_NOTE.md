# Exa Web Search — E2E Integration Test Note

**Feature:** Exa neural web search via sol.blockrun.ai
**Payment:** Solana USDC (x402)
**Estimated cost per full run:** ~$0.04
**Date added:** 2026-03-31

---

## What's Being Tested

| Test | Endpoint | Expected Cost |
|---|---|---|
| `test_exa_search` | `POST /api/v1/exa/search` | $0.01 |
| `test_exa_find_similar` | `POST /api/v1/exa/find-similar` | $0.01 |
| `test_exa_contents` | `POST /api/v1/exa/contents` | $0.002/URL |
| `test_exa_answer` | `POST /api/v1/exa/answer` | $0.01 |
| `test_exa_generic_proxy` | `POST /api/v1/exa/search` (via `exa()`) | $0.01 |
| `test_exa_spending_tracked` | Session tracking across all calls | — |

---

## Setup

### 1. Install the SDK

```bash
pip install "blockrun-llm[solana]"
```

Or from source:

```bash
git clone https://github.com/BlockRunAI/blockrun-llm
cd blockrun-llm
pip install -e ".[solana,dev]"
```

### 2. Prepare a Solana wallet with USDC

- You need a Solana mainnet wallet with at least **$0.10 USDC** (covers multiple runs)
- The private key must be **bs58-encoded** (64-byte keypair, standard Solana format)
- USDC mint: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`

### 3. Set environment variable

```bash
export SOLANA_WALLET_KEY="your-bs58-private-key-here"
```

---

## Run the Tests

```bash
# Run only Exa E2E tests
pytest tests/integration -k TestSolanaExa -v

# Run all integration tests (Base + Solana)
pytest tests/integration -v
```

Expected output:

```
tests/integration/test_production_api.py::TestSolanaExa::test_exa_search PASSED
   ✓ exa_search: 3 results, cost=$0.0100
tests/integration/test_production_api.py::TestSolanaExa::test_exa_find_similar PASSED
   ✓ exa_find_similar: 3 results
tests/integration/test_production_api.py::TestSolanaExa::test_exa_contents PASSED
   ✓ exa_contents: response received
tests/integration/test_production_api.py::TestSolanaExa::test_exa_answer PASSED
   ✓ exa_answer: response received
tests/integration/test_production_api.py::TestSolanaExa::test_exa_generic_proxy PASSED
   ✓ exa() generic: 2 results
tests/integration/test_production_api.py::TestSolanaExa::test_exa_spending_tracked PASSED
   ✓ Spending: $0.0400 over 5 calls
```

---

## Manual API Smoke Test (no wallet needed)

Verify endpoints are live and pricing is correct:

```bash
# search — expect $0.0100
curl -s -X POST https://sol.blockrun.ai/api/v1/exa/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}' | python3 -m json.tool | grep -E '"amount"|"network"|"endpoint"'

# contents with 2 URLs — expect $0.0040 ($0.002 × 2)
curl -s -X POST https://sol.blockrun.ai/api/v1/exa/contents \
  -H "Content-Type: application/json" \
  -d '{"urls":["https://a.com","https://b.com"]}' | python3 -m json.tool | grep '"amount"'

# discovery
curl -s https://sol.blockrun.ai/api/.well-known/x402 | python3 -m json.tool | grep exa
```

All should return HTTP 402 with correct `price` and `network: solana`.

---

## Pass Criteria

- All 6 `TestSolanaExa` tests pass
- `exa_search` cost is exactly $0.01 (±$0.0001)
- `exa_contents` cost scales correctly with number of URLs
- Session `total_usd` and `calls` are tracked accurately
- No `APIError` or `PaymentError` raised on valid requests

## Fail Criteria

- HTTP 503 → `EXA_API_KEY` not configured in Cloud Run (contact DevOps)
- HTTP 402 after payment → wallet has insufficient USDC balance
- `AssertionError` on result structure → Exa API response format changed

---

## Contact

Questions → @bc1max on Telegram
