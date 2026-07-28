#!/usr/bin/env python3
"""
End-to-end performance benchmark for a Claude model through the BlockRun gateway.

Measures the 12 metrics requested:
  1.  单个请求吞吐 (token/s)         per-request output throughput (avg of per-req tokens/s)
  2.  系统级平均token生成速度        system-level aggregate output tokens / wall-clock
  3.  平均 TTFT(s)                   mean time-to-first-token (streaming)
  4.  P50 TTFT(s)
  5.  P95 TTFT(s)
  6.  P99 TTFT(s)
  7.  平均延迟(s)                    mean end-to-end latency (request → last token)
  8.  P50 延迟(s)
  9.  P95 延迟(s)
  10. P99 延迟(s)
  11. 成功率(%)                      successful requests / total
  12. 缓存命中率(%)                  cache_read_input_tokens / prompt_tokens (2nd call,
                                     shared long prefix). NON-streaming — the gateway's
                                     SSE chunks carry no usage. Requires the
                                     fingerprint-passthrough code DEPLOYED and the test
                                     wallet in ANTHROPIC_DIRECT_PAYER_ALLOWLIST_EXTRA,
                                     otherwise the gateway strips cache tokens → N/A.

Each paid request spends USDC via x402. Pick --requests with that in mind.

Wallet: read by the SDK from --private-key, $SOLANA_WALLET_KEY / $BLOCKRUN_WALLET_KEY,
or ~/.blockrun/.session — the key never leaves the host.

Examples
--------
    # Solana gateway, claude-opus-4.7, 30 reqs @ concurrency 5, + cache probe
    python benchmark_claude.py --chain solana --model anthropic/claude-opus-4.7 \
        --requests 30 --concurrency 5 --cache-probe

    # Base gateway, sonnet, quick 10-req smoke
    python benchmark_claude.py --chain base --model anthropic/claude-sonnet-4.6 \
        --requests 10 --concurrency 3
"""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

SOLANA_API_URL = "https://sol.blockrun.ai/api"
BASE_API_URL = "https://blockrun.ai/api"

# Default workload — a deterministic-ish prompt that yields a few hundred tokens.
DEFAULT_PROMPT = (
    "Explain how an x402 micropayment settles on-chain, step by step, "
    "from the 402 challenge to facilitator verification. Be concise."
)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0,100]). Empty → nan."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # nearest-rank: index = ceil(pct/100 * N) - 1
    import math

    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _count_tokens(text: str, model_hint: str = "") -> int:
    """Best-effort output-token count for throughput. Uses tiktoken if present
    (o200k_base — closest public BPE), else a ~4-chars/token estimate. Claude's
    real tokenizer differs slightly; throughput is reported as an estimate."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, round(len(text) / 4))


@dataclass
class ReqResult:
    ok: bool
    ttft: float | None = None  # seconds to first content token
    latency: float | None = None  # seconds request → last token
    out_tokens: int = 0
    error: str = ""


@dataclass
class Bench:
    chain: str
    model: str
    api_url: str
    requests: int
    concurrency: int
    prompt: str
    max_tokens: int
    private_key: str | None = None
    results: list[ReqResult] = field(default_factory=list)

    def _client(self):
        if self.chain == "solana":
            from blockrun_llm import SolanaLLMClient

            key = self.private_key
            if not key:
                # Fall back to the SDK's wallet resolver ($SOLANA_WALLET_KEY →
                # ~/.blockrun/.solana-session) so the existing session "just works".
                from blockrun_llm.solana_wallet import load_solana_wallet

                key = load_solana_wallet()
            return SolanaLLMClient(private_key=key, api_url=self.api_url)
        from blockrun_llm import LLMClient

        return LLMClient(private_key=self.private_key, api_url=self.api_url)

    def _one_streaming(self, client) -> ReqResult:
        messages = [{"role": "user", "content": self.prompt}]
        start = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        try:
            for chunk in client.chat_completion_stream(
                model=self.model, messages=messages, max_tokens=self.max_tokens
            ):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    text_parts.append(content)
            latency = time.perf_counter() - start
            out = _count_tokens("".join(text_parts), self.model)
            return ReqResult(ok=True, ttft=ttft, latency=latency, out_tokens=out)
        except Exception as exc:
            return ReqResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    def run_throughput_phase(self) -> float:
        """Fire `requests` streaming calls at `concurrency`. Returns wall-clock seconds."""
        client = self._client()
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._one_streaming, client) for _ in range(self.requests)]
            for fut in as_completed(futures):
                self.results.append(fut.result())
        return time.perf_counter() - wall_start

    def cache_probe(self) -> float:
        """Two NON-streaming calls sharing a long system prefix. Returns cache hit
        rate (%) on the 2nd call: cached_input_tokens / prompt_tokens.

        Reads BOTH provider conventions the gateway may surface (for allowlisted
        payers): Anthropic ``cache_read_input_tokens`` and OpenAI
        ``prompt_tokens_details.cached_tokens``. Returns 0.0 when nothing cached
        or the field is absent (not deployed / not allowlisted / model doesn't
        cache) — reported as 0, never "N/A"."""
        client = self._client()
        long_prefix = ("You are a meticulous protocol analyst. " * 240).strip()
        # Identical long system prefix (the cacheable part) but DIFFERENT user
        # messages on the two calls — an identical body would trip the gateway's
        # x402 replay guard, so vary it while keeping the prefix cache-eligible.
        warm = [
            {"role": "system", "content": long_prefix},
            {"role": "user", "content": "Reply with the single word: ready."},
        ]
        measure = [
            {"role": "system", "content": long_prefix},
            {"role": "user", "content": "Now reply with the single word: done."},
        ]
        client.chat_completion(model=self.model, messages=warm, max_tokens=8)
        time.sleep(2.0)
        resp = client.chat_completion(model=self.model, messages=measure, max_tokens=8)
        usage = getattr(resp, "usage", None)
        if usage is None:
            return 0.0
        u: dict[str, Any] = (
            usage.model_dump(exclude_none=True) if hasattr(usage, "model_dump") else dict(usage)
        )
        prompt_tokens = u.get("prompt_tokens") or 0
        cache_read = u.get("cache_read_input_tokens") or 0
        cache_creation = u.get("cache_creation_input_tokens") or 0
        # Anthropic style: prompt_tokens (= input_tokens) EXCLUDES cached tokens —
        # the three counts are disjoint, so total input is their sum.
        if cache_read or cache_creation:
            total_input = prompt_tokens + cache_read + cache_creation
            return 100.0 * cache_read / total_input if total_input else 0.0
        # OpenAI style: cached_tokens is a SUBSET of prompt_tokens.
        details = u.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        if cached and prompt_tokens:
            return 100.0 * cached / prompt_tokens
        return 0.0

    def report(self, wall: float, cache_hit: float | None) -> None:
        ok = [r for r in self.results if r.ok]
        ttfts = [r.ttft for r in ok if r.ttft is not None]
        lats = [r.latency for r in ok if r.latency is not None]
        per_req_tps = [
            r.out_tokens / r.latency for r in ok if r.latency and r.latency > 0 and r.out_tokens
        ]
        total_out = sum(r.out_tokens for r in ok)
        succ = 100.0 * len(ok) / self.requests if self.requests else 0.0

        def fmt(x: float) -> str:
            return "nan" if x != x else f"{x:.3f}"  # noqa: PLR0124 — x!=x is the NaN test

        print("\n" + "=" * 56)
        print(f" Claude E2E benchmark — {self.model}  ({self.chain})")
        print(f" {self.api_url}")
        print(
            f" requests={self.requests} concurrency={self.concurrency} "
            f"max_tokens={self.max_tokens}"
        )
        print("=" * 56)
        rows = [
            ("单个请求吞吐 (token/s)", fmt(statistics.mean(per_req_tps)) if per_req_tps else "nan"),
            ("系统级平均token生成速度 (token/s)", fmt(total_out / wall) if wall > 0 else "nan"),
            ("平均TTFT(s)", fmt(statistics.mean(ttfts)) if ttfts else "nan"),
            ("P50 TTFT(s)", fmt(_percentile(ttfts, 50))),
            ("P95 TTFT(s)", fmt(_percentile(ttfts, 95))),
            ("P99 TTFT(s)", fmt(_percentile(ttfts, 99))),
            ("平均延迟(s)", fmt(statistics.mean(lats)) if lats else "nan"),
            ("P50延迟(s)", fmt(_percentile(lats, 50))),
            ("P95延迟(s)", fmt(_percentile(lats, 95))),
            ("P99延迟(s)", fmt(_percentile(lats, 99))),
            ("成功率(%)", fmt(succ)),
            ("缓存命中率(%)", fmt(cache_hit if cache_hit is not None else 0.0)),
        ]
        for name, val in rows:
            print(f"  {name:<34} {val}")
        print("-" * 56)
        print(
            f"  样本: 成功 {len(ok)}/{self.requests}   总输出≈{total_out} tokens   wall={wall:.2f}s"
        )
        fails = [r for r in self.results if not r.ok]
        if fails:
            print(f"  失败 {len(fails)} 例，示例: {fails[0].error}")
        print("=" * 56 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Claude E2E benchmark via BlockRun gateway")
    p.add_argument("--chain", choices=["solana", "base"], default="solana")
    p.add_argument("--model", default="anthropic/claude-opus-4.7")
    p.add_argument("--api-url", default=None, help="override gateway URL")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--private-key", default=None, help="wallet key (else env / ~/.blockrun)")
    p.add_argument(
        "--cache-probe",
        action="store_true",
        help="add 2 non-streaming calls to measure cache hit rate (extra spend)",
    )
    args = p.parse_args()

    api_url = args.api_url or (SOLANA_API_URL if args.chain == "solana" else BASE_API_URL)
    bench = Bench(
        chain=args.chain,
        model=args.model,
        api_url=api_url,
        requests=args.requests,
        concurrency=args.concurrency,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        private_key=args.private_key,
    )
    print(f"[benchmark] {args.requests} paid streaming requests → {api_url} ({args.model}) …")
    wall = bench.run_throughput_phase()
    cache_hit = 0.0
    if args.cache_probe:
        print("[benchmark] cache probe (2 non-streaming calls) …")
        try:
            cache_hit = bench.cache_probe()
        except Exception as exc:
            print(f"[benchmark] cache probe failed (→ 0): {type(exc).__name__}: {exc}")
            cache_hit = 0.0
    bench.report(wall, cache_hit)


if __name__ == "__main__":
    main()
