"""Sweep test for every image + music model the BlockRun SDK exposes.

Runs each model with a short fixed prompt, captures status / latency / cost,
and prints a grouped report at the end. Mirror of examples/sweep_all_chat_models.py
but for ImageClient and MusicClient. Video is intentionally separate (single
clip can take >2 min and cost up to $0.30 — run that one manually).

Usage:
    export BLOCKRUN_WALLET_KEY=0x...   # ≥ $1 USDC on Base mainnet
    python examples/sweep_all_media_models.py

Optional:
    --budget-cap 1.00         abort sweep when cumulative spend reaches this
    --skip-image              run only music
    --skip-music              run only image
    --output-json FILE        write per-probe results as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from blockrun_llm import ImageClient, LLMClient, MusicClient
from blockrun_llm.types import APIError, PaymentError


IMAGE_TARGETS: List[Dict[str, Any]] = [
    # Each entry: model_id + size override if model has a constrained set.
    {"model": "google/nano-banana", "size": "1024x1024"},
    {"model": "google/nano-banana-pro", "size": "1024x1024"},
    {"model": "openai/dall-e-3", "size": "1024x1024"},
    {"model": "openai/gpt-image-1", "size": "1024x1024"},
    {"model": "openai/gpt-image-2", "size": "1024x1024"},
    {"model": "zai/cogview-4", "size": "1024x1024"},
    {"model": "xai/grok-imagine-image", "size": "1024x1024"},
    {"model": "xai/grok-imagine-image-pro", "size": "1024x1024"},
]

MUSIC_TARGETS: List[str] = [
    "minimax/music-2.5+",
    "minimax/music-2.5",
]

IMAGE_PROMPT = "a single red apple on a plain white background, photographic"
MUSIC_PROMPT = "30-second chill lo-fi beat with mellow piano"


@dataclass
class ProbeResult:
    model_id: str
    modality: str  # "image" | "music"
    status: str  # ok / http_error / timeout / payment_error / unexpected
    latency_ms: int
    cost_delta_usd: float = 0.0
    artifact_url: Optional[str] = None  # first asset URL/data preview
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


def sanitize(s: str, limit: int = 200) -> str:
    s = s.replace("\n", "; ").replace("\r", " ")
    for prefix in ("/Users/", "/var/", "/private/", "/tmp/"):
        idx = s.find(prefix)
        if idx >= 0:
            s = s[:idx] + "[path]"
    return s[:limit]


def fmt_cost(usd: float) -> str:
    return f"${usd:.5f}"


def mask_address(addr: str) -> str:
    return f"{addr[:6]}...{addr[-4:]}" if len(addr) > 10 else addr


def preview_url(url: str, max_len: int = 60) -> str:
    if not url:
        return ""
    if url.startswith("data:"):
        # Data URL — show prefix + length
        comma = url.find(",")
        head = url[:comma] if comma >= 0 else url[:60]
        body_len = len(url) - comma - 1 if comma >= 0 else 0
        return f"{head[:30]}...({body_len} bytes)"
    return url[:max_len] + ("..." if len(url) > max_len else "")


def probe_image(
    client: ImageClient,
    target: Dict[str, Any],
    pricing: Dict[str, float],
) -> ProbeResult:
    model_id = target["model"]
    t0 = time.monotonic()
    try:
        result = client.generate(IMAGE_PROMPT, model=model_id, size=target.get("size"))
        latency_ms = int((time.monotonic() - t0) * 1000)
        url = result.data[0].url if result.data else ""
        return ProbeResult(
            model_id=model_id,
            modality="image",
            status="ok",
            latency_ms=latency_ms,
            cost_delta_usd=pricing.get(model_id, 0.0),
            artifact_url=url,
        )
    except APIError as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="image",
            status="http_error",
            latency_ms=latency_ms,
            error_message=f"status={e.status_code}: {sanitize(str(e))}",
        )
    except PaymentError as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="image",
            status="payment_error",
            latency_ms=latency_ms,
            error_message=sanitize(str(e)),
        )
    except httpx.TimeoutException:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="image",
            status="timeout",
            latency_ms=latency_ms,
            error_message=f"timeout after {latency_ms}ms",
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="image",
            status="unexpected",
            latency_ms=latency_ms,
            error_message=f"{type(e).__name__}: {sanitize(str(e))}",
        )


def probe_music(
    client: MusicClient,
    model_id: str,
    pricing: Dict[str, float],
) -> ProbeResult:
    t0 = time.monotonic()
    try:
        result = client.generate(MUSIC_PROMPT, model=model_id, instrumental=True)
        latency_ms = int((time.monotonic() - t0) * 1000)
        url = result.data[0].url if result.data else ""
        return ProbeResult(
            model_id=model_id,
            modality="music",
            status="ok",
            latency_ms=latency_ms,
            cost_delta_usd=pricing.get(model_id, 0.0),
            artifact_url=url,
        )
    except APIError as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="music",
            status="http_error",
            latency_ms=latency_ms,
            error_message=f"status={e.status_code}: {sanitize(str(e))}",
        )
    except PaymentError as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="music",
            status="payment_error",
            latency_ms=latency_ms,
            error_message=sanitize(str(e)),
        )
    except httpx.TimeoutException:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="music",
            status="timeout",
            latency_ms=latency_ms,
            error_message=f"timeout after {latency_ms}ms",
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            model_id=model_id,
            modality="music",
            status="unexpected",
            latency_ms=latency_ms,
            error_message=f"{type(e).__name__}: {sanitize(str(e))}",
        )


def preflight() -> tuple:
    """Return (ImageClient, image_pricing, music_pricing, initial_balance).

    ImageClient/MusicClient don't expose spending tracking, so we use a
    parallel LLMClient to read wallet balance for the budget guard.
    Per-call costs are looked up from the published pricing tables.
    """
    print("=" * 78)
    print("BLOCKRUN PYTHON SDK — IMAGE + MUSIC SWEEP")
    print("=" * 78)

    image_client = ImageClient()
    print(f"wallet     : {mask_address(image_client.get_wallet_address())}")
    print(f"api url    : {image_client.api_url}")

    # Use LLMClient for balance + pricing — image/music clients don't have
    # those helpers and the wallet is shared.
    llm = LLMClient()
    initial_balance = 0.0
    try:
        initial_balance = llm.get_balance()
        print(f"USDC bal   : ${initial_balance:.4f}")
        if initial_balance < 1.0:
            print("WARN       : balance below $1.00 — sweep may abort")
    except Exception as e:
        print(f"USDC bal   : (unavailable: {sanitize(str(e), 80)})")

    # Image + music pricing both come from /v1/models filtered by category;
    # the legacy /v1/images/models endpoint currently returns 404 server-side
    # (2026-05-09), so don't rely on it.
    image_pricing: Dict[str, float] = {}
    music_pricing: Dict[str, float] = {}
    try:
        for m in llm.list_models():
            mid = m.get("id", "")
            if not mid:
                continue
            cats = m.get("categories") or []
            block = m.get("pricing") or {}
            if "image" in cats:
                price = block.get("per_image") or block.get("flat") or block.get("perImage")
                image_pricing[mid] = float(price or 0)
            elif "music" in cats or "audio" in cats:
                price = block.get("per_track") or block.get("flat") or block.get("perTrack")
                music_pricing[mid] = float(price or 0)
    except Exception as e:
        print(f"WARN       : list_models() failed: {sanitize(str(e), 80)}")

    print(f"image price catalog: {len(image_pricing)} models")
    print(f"music price catalog: {len(music_pricing)} models")
    print()
    return image_client, image_pricing, music_pricing, initial_balance


def run_image_sweep(
    client: ImageClient,
    pricing: Dict[str, float],
    args: argparse.Namespace,
    spent_so_far: float = 0.0,
) -> List[ProbeResult]:
    print(f">>> Image sweep ({len(IMAGE_TARGETS)} models)")
    print()
    results: List[ProbeResult] = []
    n = len(IMAGE_TARGETS)
    warned = False
    spent = spent_so_far
    for i, target in enumerate(IMAGE_TARGETS, start=1):
        if spent >= args.budget_cap:
            print(f"[BUDGET-ABORT] ${spent:.4f} >= ${args.budget_cap:.2f}")
            for remaining in IMAGE_TARGETS[i - 1 :]:
                results.append(
                    ProbeResult(
                        model_id=remaining["model"],
                        modality="image",
                        status="skipped_budget",
                        latency_ms=0,
                        error_message=f"budget cap ${args.budget_cap:.2f} reached",
                    )
                )
            return results
        if spent >= args.budget_cap * 0.8 and not warned:
            print(f"[BUDGET-WARN] at ${spent:.4f} of ${args.budget_cap:.2f}")
            warned = True

        result = probe_image(client, target, pricing)
        results.append(result)
        spent += result.cost_delta_usd
        preview = preview_url(result.artifact_url or "")
        if result.error_message:
            preview = result.error_message[:60]
        print(
            f"[{i:02d}/{n}] {result.model_id:36s} {result.status:14s} "
            f"{fmt_cost(result.cost_delta_usd)} {result.latency_ms:>6d}ms  {preview}"
        )
        if i < n:
            time.sleep(1.0)
    return results


def run_music_sweep(
    pricing: Dict[str, float],
    args: argparse.Namespace,
    spent_so_far: float = 0.0,
) -> List[ProbeResult]:
    print(f">>> Music sweep ({len(MUSIC_TARGETS)} models)")
    print()
    client = MusicClient()
    results: List[ProbeResult] = []
    n = len(MUSIC_TARGETS)
    spent = spent_so_far
    for i, model_id in enumerate(MUSIC_TARGETS, start=1):
        if spent >= args.budget_cap:
            print(f"[BUDGET-ABORT] ${spent:.4f} >= ${args.budget_cap:.2f}")
            for remaining in MUSIC_TARGETS[i - 1 :]:
                results.append(
                    ProbeResult(
                        model_id=remaining,
                        modality="music",
                        status="skipped_budget",
                        latency_ms=0,
                        error_message=f"budget cap ${args.budget_cap:.2f} reached",
                    )
                )
            return results

        result = probe_music(client, model_id, pricing)
        results.append(result)
        spent += result.cost_delta_usd
        preview = preview_url(result.artifact_url or "")
        if result.error_message:
            preview = result.error_message[:60]
        print(
            f"[{i:02d}/{n}] {result.model_id:36s} {result.status:14s} "
            f"{fmt_cost(result.cost_delta_usd)} {result.latency_ms:>6d}ms  {preview}"
        )
        if i < n:
            time.sleep(1.0)
    return results


def report(
    results: List[ProbeResult],
    started_at: float,
    args: argparse.Namespace,
    initial_balance: float,
    final_balance: float,
) -> bool:
    failures = [r for r in results if r.status != "ok"]

    if failures:
        print()
        print(">>> Failures")
        for r in failures:
            print(f"  [{r.modality}] {r.model_id:36s} {r.status:14s}")
            if r.error_message:
                print(f"    {r.error_message}")

    print()
    print(">>> Modality summary")
    by_mod: Dict[str, List[ProbeResult]] = {}
    for r in results:
        by_mod.setdefault(r.modality, []).append(r)
    for modality in sorted(by_mod):
        rows = by_mod[modality]
        ok = sum(1 for r in rows if r.status == "ok")
        cost = sum(r.cost_delta_usd for r in rows)
        avg_ms = sum(r.latency_ms for r in rows) / max(len(rows), 1)
        print(
            f"  {modality:6s}  {ok}/{len(rows)} ok   cost={fmt_cost(cost)}   "
            f"avg_latency={avg_ms / 1000:.1f}s"
        )

    duration = time.monotonic() - started_at
    minutes, seconds = divmod(int(duration), 60)
    successes = [r for r in results if r.status == "ok"]
    success_rate = len(successes) / max(len(results), 1) * 100
    total_cost = sum(r.cost_delta_usd for r in results)

    overall_pass = len(failures) == 0

    actual_charged = max(0.0, initial_balance - final_balance)

    print()
    print(">>> Summary")
    print(f"  est cost from pricing: {fmt_cost(total_cost)}")
    print(
        f"  actual USDC charged  : {fmt_cost(actual_charged)} "
        f"(balance: ${initial_balance:.4f} -> ${final_balance:.4f})"
    )
    print(f"  total calls          : {len(results)}")
    print(f"  success              : {len(successes)}/{len(results)}  ({success_rate:.0f}%)")
    print(f"  duration             : {minutes}m {seconds}s")
    print(f"  budget used          : {fmt_cost(actual_charged)} of {fmt_cost(args.budget_cap)}")
    print(f"  status               : {'PASS' if overall_pass else 'FAIL'}")
    print()
    return overall_pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep test every image + music model in the BlockRun SDK."
    )
    parser.add_argument("--budget-cap", type=float, default=1.00)
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-music", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.skip_image and args.skip_music:
        sys.stderr.write("ERROR: nothing to do — both --skip-image and --skip-music set\n")
        return 2

    started_at = time.monotonic()
    image_client, image_pricing, music_pricing, initial_balance = preflight()

    results: List[ProbeResult] = []
    spent = 0.0
    if not args.skip_image:
        image_results = run_image_sweep(image_client, image_pricing, args, spent)
        results.extend(image_results)
        spent += sum(r.cost_delta_usd for r in image_results)
    if not args.skip_music:
        results.extend(run_music_sweep(music_pricing, args, spent))

    # Re-read balance to reconcile actual charged amount.
    final_balance = initial_balance
    try:
        final_balance = LLMClient().get_balance()
    except Exception:
        pass

    overall_pass = report(results, started_at, args, initial_balance, final_balance)

    if args.output_json:
        payload = {
            "started_at": started_at,
            "args": vars(args),
            "results": [asdict(r) for r in results],
            "pass": overall_pass,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"results JSON written to {args.output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
