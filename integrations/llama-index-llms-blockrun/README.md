# LlamaIndex LLM Integration: BlockRun

[BlockRun](https://blockrun.ai) is one OpenAI-compatible gateway to models from
OpenAI, Anthropic, Google, DeepSeek, xAI, NVIDIA and others, where every call
pays for itself in USDC over [x402](https://x402.org). There is no API key and
no account: you pick the chain the wallet pays from, Base or Solana.

## Why not `OpenAILike`?

`OpenAILike(api_base="https://blockrun.ai/api/v1", api_key="fake")` reaches the
gateway and stops at the first paid call. The gateway answers an unpaid request
with HTTP 402 and a price, and nothing in the OpenAI client knows how to sign a
payment for it. This package hands that step to the
[`blockrun-llm`](https://pypi.org/project/blockrun-llm/) SDK, which signs the
payment locally and retries the request with the signature attached.

## Install

```bash
pip install llama-index-llms-blockrun            # pay on Base
pip install "llama-index-llms-blockrun[solana]"  # pay on Solana
```

## Pick a chain

| `chain=`           | Pays with      | Wallet, in order of precedence                                                           |
| ------------------ | -------------- | ---------------------------------------------------------------------------------------- |
| `"base"` (default) | USDC on Base   | `private_key=` (EVM hex key), `BLOCKRUN_WALLET_KEY`, `~/.blockrun/.session`              |
| `"solana"`         | USDC on Solana | `private_key=` (base58 secret key), `SOLANA_WALLET_KEY`, a Solana wallet file on disk    |

The key never leaves your machine; it only signs the payment for each call.
`llm.wallet_address` prints the address to fund.

```python
from llama_index.llms.blockrun import BlockRun

llm = BlockRun(model="openai/gpt-5.5")  # Base
llm = BlockRun(model="openai/gpt-5.5", chain="solana")  # Solana

print(llm.wallet_address)
```

## Usage

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.blockrun import BlockRun

llm = BlockRun(model="anthropic/claude-sonnet-4.6")

print(llm.complete("Paris is the capital of"))

response = llm.chat(
    [
        ChatMessage(role="system", content="Answer in one sentence."),
        ChatMessage(role="user", content="What is x402?"),
    ]
)
print(response.message.content)
print(response.additional_kwargs.get("cost_usd"))  # USD charged, where the SDK reports it (Base)

for chunk in llm.stream_complete("Write a haiku about USDC."):
    print(chunk.delta, end="", flush=True)
```

Async works the same way: `achat`, `acomplete`, `astream_chat`, `astream_complete`.

### As the default LLM

```python
from llama_index.core import Settings

Settings.llm = BlockRun(model="openai/gpt-5.5")
```

### Tools and agents

`BlockRun` is a `FunctionCallingLLM`, so tool calling, `predict_and_call`,
structured outputs and `FunctionAgent` work as they do with OpenAI:

```python
from llama_index.core.agent.workflow import FunctionAgent


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = FunctionAgent(tools=[multiply], llm=BlockRun(model="openai/gpt-5.5"))
print(await agent.run("What is 1234 * 4567?"))
```

## Spending

Each call is quoted by the gateway before anything is signed. Cap what a single
call may cost, and a quote above it is refused before it is paid:

```python
llm = BlockRun(model="openai/gpt-5.5", max_cost_per_call=0.05)
```

A refused quote raises `blockrun_llm.SpendLimitError`; an empty wallet raises
`blockrun_llm.PaymentError`. Prices per model are listed at
https://blockrun.ai/api/v1/models.

## Options

| Parameter                   | Default          | Notes                                                                                    |
| --------------------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| `model`                     | required         | Any chat model id from https://blockrun.ai/api/v1/models                                 |
| `chain`                     | `"base"`         | `"base"` or `"solana"`                                                                   |
| `private_key`               | from environment | Never serialized with the LLM                                                            |
| `temperature`               | unset            | Unset lets each model use its own default                                                |
| `max_tokens`                | unset            | The SDK sends 1024 when unset                                                            |
| `context_window`            | from catalog     | Read once from `/v1/models`; set it to skip that request                                 |
| `max_cost_per_call`         | unset            | USD ceiling per call                                                                     |
| `api_url`                   | chain's mainnet  | `https://testnet.blockrun.ai/api` for Base Sepolia                                       |
| `timeout`                   | 600 s            | Reasoning models can think for minutes                                                   |
| `additional_kwargs`         | `{}`             | Sent with every request, e.g. `top_p`, `stop`, `response_format`                         |
| `is_function_calling_model` | `True`           | Set `False` for a model without tool support                                             |

## Development

```bash
pip install -e "../..[dev]" -e ".[dev]"
pytest
```
