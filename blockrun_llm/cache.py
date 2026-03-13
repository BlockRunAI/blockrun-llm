"""
Local response cache and archive for paid BlockRun API calls.

Two storage layers:
1. **Cache** (~/.blockrun/cache/) — hash-keyed, TTL-based dedup to avoid paying twice
2. **Data**  (~/.blockrun/data/)  — human-readable JSON files for every paid call

Cache keys are based on (endpoint, request body).
TTL is configurable per endpoint type.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# Default TTL in seconds per endpoint pattern
DEFAULT_TTL: Dict[str, int] = {
    # X/Twitter data — cache 1 hour (followers/tweets don't change every minute)
    "/v1/x/": 3600,
    "/v1/partner/": 3600,
    # Chat completions — no cache (each call is unique)
    "/v1/chat/": 0,
    # Search — cache 15 minutes
    "/v1/search": 900,
    # Image — no cache
    "/v1/image": 0,
}

CACHE_DIR = Path.home() / ".blockrun" / "cache"
DATA_DIR = Path.home() / ".blockrun" / "data"


def _get_ttl(endpoint: str) -> int:
    """Get TTL for an endpoint based on pattern matching."""
    for pattern, ttl in DEFAULT_TTL.items():
        if pattern in endpoint:
            return ttl
    # Default: cache 1 hour for unknown endpoints
    return 3600


def _cache_key(endpoint: str, body: Dict[str, Any]) -> str:
    """Generate a deterministic cache key from endpoint + request body."""
    # Remove cursor/pagination from cache key — different pages are different requests
    # But keep everything else
    key_data = json.dumps({"endpoint": endpoint, "body": body}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    """Get the file path for a cache entry."""
    return CACHE_DIR / f"{key}.json"


def get_cached(endpoint: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check if a cached response exists and is still fresh.

    Returns the cached response dict if hit, None if miss or expired.
    """
    ttl = _get_ttl(endpoint)
    if ttl <= 0:
        return None

    key = _cache_key(endpoint, body)
    path = _cache_path(key)

    if not path.exists():
        return None

    try:
        entry = json.loads(path.read_text())
        cached_at = entry.get("cached_at", 0)

        if time.time() - cached_at > ttl:
            # Expired
            path.unlink(missing_ok=True)
            return None

        return entry.get("response")
    except (json.JSONDecodeError, OSError):
        return None


def _readable_filename(endpoint: str, body: Dict[str, Any]) -> str:
    """
    Generate a human-readable filename from endpoint + request body.

    Examples:
        x_search_2026-03-13_x402_payment.json
        chat_2026-03-13_gpt-4o.json
        x_followers_2026-03-13_elonmusk.json
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Extract a short label from the endpoint
    ep = endpoint.rstrip("/").rsplit("/", 1)[-1]  # e.g. "completions", "followers", "search"
    if "/v1/chat/" in endpoint:
        ep = "chat"
    elif "/v1/x/" in endpoint:
        ep = "x_" + ep
    elif "/v1/search" in endpoint:
        ep = "search"
    elif "/v1/image" in endpoint:
        ep = "image"

    # Extract a short identifier from the body
    label = (
        body.get("query")
        or body.get("username")
        or body.get("handle")
        or body.get("model")
        or body.get("prompt", "")[:40]
        or ""
    )
    # Sanitize for filesystem
    label = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(label))[:40].strip("_")

    return f"{ep}_{ts}_{label}.json" if label else f"{ep}_{ts}.json"


def save_to_cache(
    endpoint: str,
    body: Dict[str, Any],
    response: Dict[str, Any],
    cost_usd: float = 0.0,
) -> None:
    """
    Save a paid API response locally.

    1. Hash-keyed cache file (for TTL-based dedup)
    2. Human-readable data file (browsable archive of every paid call)
    3. Cost log entry
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = _cache_key(endpoint, body)
    entry = {
        "cached_at": time.time(),
        "endpoint": endpoint,
        "body": body,
        "response": response,
        "cost_usd": cost_usd,
    }

    try:
        _cache_path(key).write_text(json.dumps(entry, default=str))
    except OSError:
        pass  # Don't fail the request if cache write fails

    # Save human-readable copy to ~/.blockrun/data/
    _save_readable(endpoint, body, response, cost_usd)

    # Also append to the cost log (never overwritten)
    _append_cost_log(endpoint, cost_usd)


def _save_readable(
    endpoint: str,
    body: Dict[str, Any],
    response: Dict[str, Any],
    cost_usd: float,
) -> None:
    """Save a human-readable JSON file to ~/.blockrun/data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = _readable_filename(endpoint, body)
    entry = {
        "saved_at": datetime.now().isoformat(),
        "endpoint": endpoint,
        "cost_usd": cost_usd,
        "request": body,
        "response": response,
    }
    try:
        (DATA_DIR / filename).write_text(json.dumps(entry, indent=2, default=str))
    except OSError:
        pass


def _append_cost_log(endpoint: str, cost_usd: float) -> None:
    """Append to a running cost log at ~/.blockrun/cost_log.jsonl"""
    if cost_usd <= 0:
        return

    log_path = Path.home() / ".blockrun" / "cost_log.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            entry = {
                "ts": time.time(),
                "endpoint": endpoint,
                "cost_usd": cost_usd,
            }
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def clear_cache() -> int:
    """Clear all cached responses. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
        count += 1
    return count


def get_cost_log_summary() -> Dict[str, Any]:
    """Read the cost log and return a summary."""
    log_path = Path.home() / ".blockrun" / "cost_log.jsonl"
    if not log_path.exists():
        return {"total_usd": 0.0, "calls": 0, "by_endpoint": {}}

    total = 0.0
    calls = 0
    by_endpoint: Dict[str, float] = {}

    try:
        for line in log_path.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            cost = entry.get("cost_usd", 0.0)
            ep = entry.get("endpoint", "unknown")
            total += cost
            calls += 1
            by_endpoint[ep] = by_endpoint.get(ep, 0.0) + cost
    except (json.JSONDecodeError, OSError):
        pass

    return {"total_usd": total, "calls": calls, "by_endpoint": by_endpoint}
