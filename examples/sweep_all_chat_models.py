"""Sweep test for every chat LLM the BlockRun Python SDK can call on Base.

Sends a minimal probe ("What is 2+2?") to each model in SWEEP_TARGETS, captures
status / latency / token usage / per-call cost, and prints a grouped report at
the end. Designed to be run manually before releases or after router changes.

Usage:
    export BLOCKRUN_WALLET_KEY=0x...   # ≥ $1 USDC on Base mainnet
    python examples/sweep_all_chat_models.py

Optional flags:
    --budget-cap 2.50         abort sweep when cumulative spend reaches this
    --sleep 1.0               seconds between sequential calls
    --skip-async              skip the AsyncLLMClient gather() smoke
    --only openai,nvidia      restrict sweep to specific providers (CSV)
    --output-json FILE        write per-probe results as JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from blockrun_llm import AsyncLLMClient, LLMClient
from blockrun_llm.types import APIError, PaymentError

# ---------------------------------------------------------------------------
# Sweep targets — hardcoded so we also probe hidden / retired model ids that
# the /v1/models endpoint deliberately omits. Mutually-exclusive groups, in
# the order the report displays them.
# ---------------------------------------------------------------------------

SWEEP_TARGETS: list[str] = [
    # OpenAI
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.3",
    "openai/gpt-5.3-codex",
    "openai/gpt-5.2",
    "openai/gpt-5.2-pro",
    "openai/gpt-5-mini",
    "openai/o1",
    "openai/o3",
    "openai/o3-mini",
    # Anthropic
    "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    # Google
    "google/gemini-3.1-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.5-flash-lite",
    # DeepSeek
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-reasoner",
    # xAI — resold via OpenRouter credit pool (added 2026-06-04)
    "xai/grok-4.3",
    "xai/grok-build-0.1",
    # MiniMax
    "minimax/minimax-m3",
    "minimax/minimax-m2.7",
    # ZAI
    "zai/glm-5.2",
    "zai/glm-5.1",
    "zai/glm-5",
    "zai/glm-5-turbo",
    # Moonshot
    "moonshot/kimi-k2.5",
    "moonshot/kimi-k2.6",
    # NVIDIA — free tier
    # (qwen3-next-80b-a3b-thinking removed: NVIDIA EOL 2026-05-21, HTTP 410;
    # gateway redirects pinned callers to llama-4-maverick)
    "nvidia/deepseek-v4-flash",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/mistral-small-4-119b",
    "nvidia/llama-4-maverick",
    "nvidia/qwen3-coder-480b",
    # NVIDIA — hidden from /v1/models but direct calls still work intentionally
    # (re-enabled 2026-04-30). Privacy caveat: NVIDIA's free build.nvidia.com
    # tier may use prompts/outputs for service improvement.
    "nvidia/gpt-oss-120b",
    "nvidia/gpt-oss-20b",
    # NVIDIA — hidden, backend redirects to v4-flash
    "nvidia/deepseek-v4-pro",
    "nvidia/deepseek-v3.2",
    "nvidia/glm-4.7",
]

REASONING_MODELS = {
    "openai/o1",
    "openai/o3",
    "openai/o3-mini",
    "openai/gpt-5.3-codex",
    "deepseek/deepseek-reasoner",
    "deepseek/deepseek-v4-pro",
    "xai/grok-4.3",
    "zai/glm-5.2",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
}

# Hidden from /v1/models (so SmartChat won't auto-pick them) but direct calls
# still work intentionally per the README "Available free models" table notes.
# A 200 OK from these is the expected state, not a privacy violation.
HIDDEN_CALLABLE = {
    "anthropic/claude-opus-4.6",
    "moonshot/kimi-k2.5",
    "nvidia/gpt-oss-120b",
    "nvidia/gpt-oss-20b",
}

# Hidden from /v1/models, backend redirects to a different model id; expect
# response.model != requested model.
HIDDEN_REDIRECTED = {
    "nvidia/deepseek-v4-pro",
    "nvidia/deepseek-v3.2",
    "nvidia/glm-4.7",
}

ASYNC_SMOKE_MODELS = [
    "deepseek/deepseek-chat",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash-lite",
]

PROBE_PROMPT = "Reply with the digit 4 only. What is 2+2?"
PROBE_MAX_TOKENS = 8
PROBE_MAX_TOKENS_REASONING = 512


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    model_id: str
    provider: str
    status: str
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_total: int | None = None
    cost_delta_usd: float = 0.0
    expected_cost_usd: float | None = None
    cost_drift_pct: float | None = None
    redirected_to: str | None = None
    response_preview: str = ""
    contains_4: bool = False
    error_message: str | None = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize(s: str, limit: int = 200) -> str:
    s = s.replace("\n", "; ").replace("\r", " ")
    for prefix in ("/Users/", "/var/", "/private/", "/tmp/"):
        idx = s.find(prefix)
        if idx >= 0:
            s = s[:idx] + "[path]"
    return s[:limit]


def mask_address(addr: str) -> str:
    return f"{addr[:6]}...{addr[-4:]}" if len(addr) > 10 else addr


def fmt_cost(usd: float) -> str:
    return f"${usd:.5f}"


def provider_of(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight() -> LLMClient:
    print("=" * 78)
    print("BLOCKRUN PYTHON SDK — CHAT-LLM SWEEP")
    print("=" * 78)

    found = None
    for name in ("BLOCKRUN_WALLET_KEY", "BASE_CHAIN_WALLET_KEY"):
        if os.environ.get(name):
            found = name
            break
    if not found:
        if os.path.exists(os.path.expanduser("~/.blockrun/.session")):
            found = "~/.blockrun/.session"
        else:
            sys.stderr.write(
                "ERROR: no wallet key found.\n"
                "  set BLOCKRUN_WALLET_KEY or BASE_CHAIN_WALLET_KEY,\n"
                "  or run setup_agent_wallet() to create ~/.blockrun/.session.\n"
            )
            sys.exit(2)
    print(f"key source : {found}")

    client = LLMClient()

    if client.is_testnet():
        sys.stderr.write("ERROR: client resolved to testnet. refusing to run.\n")
        sys.exit(2)

    print(f"wallet     : {mask_address(client.get_wallet_address())}")
    print(f"api url    : {client.api_url}")
    print("network    : Base mainnet")

    try:
        balance = client.get_balance()
        print(f"USDC bal   : ${balance:.4f}")
        if balance < 1.0:
            print("WARN       : balance below $1.00 — sweep may abort mid-run")
    except Exception as e:
        print(f"USDC bal   : (unavailable: {sanitize(str(e), 80)})")

    initial = client.get_spending()
    print(f"spending   : ${initial['total_usd']:.4f} ({initial['calls']} calls)")
    print(f"sweep size : {len(SWEEP_TARGETS)} models")
    print()
    return client


# ---------------------------------------------------------------------------
# Forward-compat diff vs /v1/models
# ---------------------------------------------------------------------------


def forward_compat_check(client: LLMClient) -> dict[str, dict[str, Any]]:
    print(">>> Forward-compat check vs /v1/models")
    try:
        listed_raw = client.list_models()
    except Exception as e:
        print(f"    list_models() failed ({sanitize(str(e), 60)}); skipping")
        print()
        return {}

    listed_chat: dict[str, dict[str, Any]] = {}
    for m in listed_raw:
        cats = m.get("categories")
        if cats is None or "chat" in cats:
            listed_chat[m["id"]] = m

    listed_ids = set(listed_chat.keys())
    hardcoded = set(SWEEP_TARGETS)

    new_in_api = listed_ids - hardcoded
    missing_in_api = hardcoded - listed_ids

    print(f"    listed in API     : {len(listed_ids)} chat models")
    print(f"    in our sweep      : {len(hardcoded)}")
    print(f"    overlap           : {len(listed_ids & hardcoded)}")

    if missing_in_api:
        print(f"    {len(missing_in_api)} sweep targets not listed (hidden/retired expected):")
        for m in sorted(missing_in_api):
            print(f"      - {m}")

    if new_in_api:
        print(f"    {len(new_in_api)} NEW models in API not in sweep list:")
        for m in sorted(new_in_api):
            print(f"      + {m}  (consider adding to SWEEP_TARGETS)")

    print()
    return listed_chat


# ---------------------------------------------------------------------------
# Single probe
# ---------------------------------------------------------------------------


def probe_one(
    client: LLMClient,
    model_id: str,
    pricing: dict[str, dict[str, Any]],
) -> ProbeResult:
    provider = provider_of(model_id)
    max_toks = PROBE_MAX_TOKENS_REASONING if model_id in REASONING_MODELS else PROBE_MAX_TOKENS
    pre = client.get_spending()["total_usd"]
    t0 = time.monotonic()
    try:
        response = client.chat_completion(
            model_id,
            [{"role": "user", "content": PROBE_PROMPT}],
            max_tokens=max_toks,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        post = client.get_spending()["total_usd"]
        cost_delta = post - pre

        text = ""
        try:
            text = response.choices[0].message.content or ""
        except Exception:
            text = ""

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else None
        tokens_out = usage.completion_tokens if usage else None
        tokens_total = usage.total_tokens if usage else None

        responding_model = getattr(response, "model", model_id) or model_id
        # Only treat as "redirected" if the responding model fundamentally differs
        # (different base id). Upstreams often append dated suffixes like
        # "gpt-5.5-2026-04-20" — that's not a redirect.
        requested_tail = model_id.split("/", 1)[-1]
        responding_tail = (
            responding_model.split("/", 1)[-1] if "/" in responding_model else responding_model
        )
        same_family = (
            requested_tail in responding_model
            or responding_tail in model_id
            or model_id in HIDDEN_CALLABLE  # backend reports canonical id, that's fine
        )
        redirected_to = responding_model if responding_model and not same_family else None

        # Cost-drift check vs published pricing. /v1/models returns
        # `pricing.input` / `pricing.output` (USD per 1M tokens) for paid models
        # and `pricing.flat` (USD per call) for flat-priced models.
        expected_cost = None
        cost_drift_pct = None
        meta = pricing.get(model_id, {})
        price_block = meta.get("pricing") or {}
        if tokens_in is not None and tokens_out is not None and price_block:
            if "flat" in price_block:
                expected_cost = float(price_block["flat"])
            else:
                ip = float(
                    price_block.get(
                        "input",
                        meta.get("inputPrice", meta.get("input_price", 0)),
                    )
                )
                op = float(
                    price_block.get(
                        "output",
                        meta.get("outputPrice", meta.get("output_price", 0)),
                    )
                )
                expected_cost = (tokens_in * ip + tokens_out * op) / 1_000_000.0
            if expected_cost > 0:
                cost_drift_pct = (cost_delta - expected_cost) / expected_cost * 100.0

        if not text and (usage and usage.completion_tokens == 0):
            status = "ok_empty"
        elif redirected_to:
            status = "ok_redirected"
        else:
            status = "ok"

        return ProbeResult(
            model_id=model_id,
            provider=provider,
            status=status,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            cost_delta_usd=cost_delta,
            expected_cost_usd=expected_cost,
            cost_drift_pct=cost_drift_pct,
            redirected_to=redirected_to,
            response_preview=text[:80],
            contains_4="4" in text[:80],
            error_message=None,
        )

    except APIError as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        status = "http_error"
        return ProbeResult(
            model_id=model_id,
            provider=provider,
            status=status,
            latency_ms=latency_ms,
            cost_delta_usd=0.0,
            error_message=f"status={e.status_code}: {sanitize(str(e), 200)}",
        )
    except PaymentError as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            provider=provider,
            status="payment_error",
            latency_ms=latency_ms,
            error_message=sanitize(str(e), 200),
        )
    except httpx.TimeoutException:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            provider=provider,
            status="timeout",
            latency_ms=latency_ms,
            error_message=f"timeout after {latency_ms}ms",
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            provider=provider,
            status="unexpected",
            latency_ms=latency_ms,
            error_message=f"{type(e).__name__}: {sanitize(str(e), 200)}",
        )


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------


def run_sweep(
    client: LLMClient,
    targets: list[str],
    args: argparse.Namespace,
    pricing: dict[str, dict[str, Any]],
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    n = len(targets)
    warned = False

    print(f">>> Sweep ({n} models, {args.sleep}s between calls)")
    print()

    for i, model_id in enumerate(targets, start=1):
        spending = client.get_spending()["total_usd"]
        if spending >= args.budget_cap:
            print(f"[BUDGET-ABORT] cumulative spend ${spending:.4f} >= ${args.budget_cap:.2f}")
            for remaining in targets[i - 1 :]:
                results.append(
                    ProbeResult(
                        model_id=remaining,
                        provider=provider_of(remaining),
                        status="skipped_budget",
                        latency_ms=0,
                        error_message=f"budget cap ${args.budget_cap:.2f} reached",
                    )
                )
            return results
        if spending >= args.budget_cap * 0.8 and not warned:
            print(f"[BUDGET-WARN] at ${spending:.4f} of ${args.budget_cap:.2f}")
            warned = True

        result = probe_one(client, model_id, pricing)
        results.append(result)

        token_str = (
            f"{result.tokens_in}/{result.tokens_out}" if result.tokens_in is not None else "-/-"
        )
        preview = (result.response_preview or "").replace("\n", " ")[:30]
        if result.error_message:
            preview = result.error_message[:30]
        print(
            f"[{i:03d}/{n}] {result.model_id:50s} {result.status:18s} "
            f"{fmt_cost(result.cost_delta_usd)} {result.latency_ms:5d}ms "
            f"{token_str:>9s}  {preview}"
        )

        if i < n:
            time.sleep(args.sleep)

    return results


# ---------------------------------------------------------------------------
# Async smoke (verifies AsyncLLMClient + asyncio.gather()).
# ---------------------------------------------------------------------------


async def _async_probe(client: AsyncLLMClient, model_id: str) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        response = await client.chat_completion(
            model_id,
            [{"role": "user", "content": PROBE_PROMPT}],
            max_tokens=PROBE_MAX_TOKENS,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        usage = response.usage
        return {
            "model_id": model_id,
            "ok": True,
            "latency_ms": latency_ms,
            "tokens_in": usage.prompt_tokens if usage else None,
            "tokens_out": usage.completion_tokens if usage else None,
            "preview": (response.choices[0].message.content or "")[:30],
            "error": None,
        }
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "model_id": model_id,
            "ok": False,
            "latency_ms": latency_ms,
            "tokens_in": None,
            "tokens_out": None,
            "preview": "",
            "error": f"{type(e).__name__}: {sanitize(str(e), 120)}",
        }


async def _async_smoke() -> list[dict[str, Any]]:
    async with AsyncLLMClient() as client:
        coros = [_async_probe(client, m) for m in ASYNC_SMOKE_MODELS]
        return await asyncio.gather(*coros)


def run_async_smoke() -> list[dict[str, Any]]:
    print(">>> Async smoke (asyncio.gather over 3 models)")
    t0 = time.monotonic()
    results = asyncio.run(_async_smoke())
    total_ms = int((time.monotonic() - t0) * 1000)
    for r in results:
        flag = "ok  " if r["ok"] else "FAIL"
        token_str = f"{r['tokens_in']}/{r['tokens_out']}" if r["tokens_in"] is not None else "-/-"
        detail = r["error"] if not r["ok"] else r["preview"]
        print(
            f"  [async] {r['model_id']:42s} {flag}  {r['latency_ms']:5d}ms  "
            f"{token_str:>7s}  {detail}"
        )
    print(f"  total wall: {total_ms}ms")
    print()
    return results


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


def report(
    client: LLMClient,
    results: list[ProbeResult],
    async_results: list[dict[str, Any]] | None,
    started_at: float,
    args: argparse.Namespace,
) -> bool:
    failures = [r for r in results if not r.status.startswith("ok")]
    drifts = [r for r in results if r.cost_drift_pct is not None and abs(r.cost_drift_pct) > 5.0]

    if failures:
        print(">>> Failures")
        for r in failures:
            print(f"  {r.model_id:50s} {r.status:18s}")
            if r.error_message:
                print(f"    {r.error_message}")
        print()

    if drifts:
        print(">>> Cost drift > 5% (potential billing inconsistency)")
        for r in drifts:
            print(
                f"  {r.model_id:50s} actual={fmt_cost(r.cost_delta_usd)}  "
                f"expected={fmt_cost(r.expected_cost_usd or 0)}  "
                f"drift={r.cost_drift_pct:+.1f}%"
            )
        print()

    print(">>> Provider summary")
    by_provider: dict[str, list[ProbeResult]] = {}
    for r in results:
        by_provider.setdefault(r.provider, []).append(r)
    for provider in sorted(by_provider):
        rows = by_provider[provider]
        ok = sum(1 for r in rows if r.status.startswith("ok"))
        cost = sum(r.cost_delta_usd for r in rows)
        toks_in = sum(r.tokens_in or 0 for r in rows)
        toks_out = sum(r.tokens_out or 0 for r in rows)
        line = f"  {provider:10s}  {ok}/{len(rows)} ok   cost={fmt_cost(cost)}"
        if toks_in or toks_out:
            line += f"   tokens={toks_in}/{toks_out}"
        print(line)
    print()

    spending = client.get_spending()
    duration = time.monotonic() - started_at
    minutes, seconds = divmod(int(duration), 60)

    # NVIDIA "free path" = models in the README's Available free models table that
    # we expect to actually run inference. Excludes the hidden+redirected ids
    # (deepseek-v4-pro/v3.2/glm-4.7) which forward to v4-flash on the backend.
    nvidia_free = [
        r for r in results if r.provider == "nvidia" and r.model_id not in HIDDEN_REDIRECTED
    ]
    nvidia_free_ok = all(r.status.startswith("ok") for r in nvidia_free)

    successes = [r for r in results if r.status.startswith("ok")]
    success_rate = len(successes) / max(len(results), 1) * 100

    total_in = sum(r.tokens_in or 0 for r in results)
    total_out = sum(r.tokens_out or 0 for r in results)

    async_ok = True
    if async_results is not None:
        async_ok = all(r["ok"] for r in async_results)

    main_threshold = 40 if not args.only else max(int(len(results) * 0.9), 1)
    main_pass = len(successes) >= main_threshold

    overall_pass = main_pass and nvidia_free_ok and async_ok

    print(">>> Summary")
    print(f"  total cost      : {fmt_cost(spending['total_usd'])}")
    print(f"  total tokens    : {total_in} in / {total_out} out / {total_in + total_out} total")
    print(f"  total calls     : {spending['calls']}")
    print(f"  sweep success   : {len(successes)}/{len(results)}  ({success_rate:.0f}%)")
    print(f"  duration        : {minutes}m {seconds}s")
    print(f"  budget used     : {fmt_cost(spending['total_usd'])} of {fmt_cost(args.budget_cap)}")
    if async_results is not None:
        passed = sum(1 for r in async_results if r["ok"])
        print(f"  async smoke     : {passed}/{len(async_results)}")
    print(f"  free tier       : {'all ok' if nvidia_free_ok else 'FAIL'}")
    print(f"  status          : {'PASS' if overall_pass else 'FAIL'}")
    print()

    return overall_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep test every chat LLM in the BlockRun SDK on Base mainnet."
    )
    parser.add_argument("--budget-cap", type=float, default=2.50)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--skip-async", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated provider prefixes (e.g. openai,nvidia)",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    started_at = time.monotonic()

    client = preflight()
    listed = forward_compat_check(client)

    targets = SWEEP_TARGETS
    if args.only:
        keep = {p.strip() for p in args.only.split(",") if p.strip()}
        targets = [m for m in SWEEP_TARGETS if provider_of(m) in keep]
        print(f"--only filter: {sorted(keep)}  →  {len(targets)} models")
        print()

    results = run_sweep(client, targets, args, listed)

    async_results: list[dict[str, Any]] | None = None
    if not args.skip_async:
        async_results = run_async_smoke()

    overall_pass = report(client, results, async_results, started_at, args)

    if args.output_json:
        payload = {
            "started_at": started_at,
            "args": vars(args),
            "results": [asdict(r) for r in results],
            "async_results": async_results,
            "spending": client.get_spending(),
            "pass": overall_pass,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"results JSON written to {args.output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
