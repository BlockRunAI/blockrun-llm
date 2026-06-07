"""
Local response cache, archive, and cost log for paid BlockRun API calls.

Three storage layers:
1. **Cache**     (~/.blockrun/cache/)        — hash-keyed, TTL-based dedup to avoid paying twice
2. **Data**      (~/.blockrun/data/)         — human-readable JSON files for every paid call
3. **Cost log**  (~/.blockrun/cost_log.jsonl) — append-only ledger for billing / audit

Cost log entries (one per line) include endpoint, cost, plus model / wallet /
network / client_kind metadata when the caller provides it. Older entries with
only `{ts, endpoint, cost_usd}` are still readable — missing fields surface as
``None`` in the summary / export views.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


# Default TTL in seconds per endpoint pattern
DEFAULT_TTL: Dict[str, int] = {
    # X/Twitter data — cache 1 hour (followers/tweets don't change every minute)
    "/v1/partner/": 3600,
    # Prediction markets — cache 30 minutes
    "/v1/pm/": 1800,
    # Chat completions — no cache (each call is unique)
    "/v1/chat/": 0,
    # Search — cache 15 minutes
    "/v1/search": 900,
    # Image — no cache
    "/v1/image": 0,
}

CACHE_DIR = Path.home() / ".blockrun" / "cache"
DATA_DIR = Path.home() / ".blockrun" / "data"
COST_LOG_PATH = Path.home() / ".blockrun" / "cost_log.jsonl"


def _get_ttl(endpoint: str) -> int:
    """Get TTL for an endpoint based on pattern matching."""
    for pattern, ttl in DEFAULT_TTL.items():
        if pattern in endpoint:
            return ttl
    # Default: cache 1 hour for unknown endpoints
    return 3600


def _cache_key(endpoint: str, body: Dict[str, Any]) -> str:
    """Generate a deterministic cache key from endpoint + request body."""
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
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    ep = endpoint.rstrip("/").rsplit("/", 1)[-1]
    if "/v1/chat/" in endpoint:
        ep = "chat"
    elif "/v1/search" in endpoint:
        ep = "search"
    elif "/v1/image" in endpoint:
        ep = "image"

    label = (
        body.get("query")
        or body.get("username")
        or body.get("handle")
        or body.get("model")
        or body.get("prompt", "")[:40]
        or ""
    )
    label = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(label))[:40].strip("_")

    return f"{ep}_{ts}_{label}.json" if label else f"{ep}_{ts}.json"


def save_to_cache(
    endpoint: str,
    body: Dict[str, Any],
    response: Dict[str, Any],
    cost_usd: float = 0.0,
    *,
    model: Optional[str] = None,
    wallet: Optional[str] = None,
    network: Optional[str] = None,
    client_kind: Optional[str] = None,
) -> None:
    """
    Save a paid API response locally.

    1. Hash-keyed cache file (for TTL-based dedup)
    2. Human-readable data file (browsable archive of every paid call)
    3. Cost log entry (with billing metadata when supplied)
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
        pass

    # Save human-readable copy to ~/.blockrun/data/
    _save_readable(endpoint, body, response, cost_usd)

    # Append to the cost log (never overwritten). Pull model from the body
    # if the caller didn't pass one explicitly.
    _append_cost_log(
        endpoint,
        cost_usd,
        model=model or body.get("model"),
        wallet=wallet,
        network=network,
        client_kind=client_kind,
    )


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


def _append_cost_log(
    endpoint: str,
    cost_usd: float,
    *,
    model: Optional[str] = None,
    wallet: Optional[str] = None,
    network: Optional[str] = None,
    client_kind: Optional[str] = None,
) -> None:
    """Append one JSONL row to ``~/.blockrun/cost_log.jsonl``.

    The full schema is::

        {ts, endpoint, cost_usd, model, wallet, network, client_kind}

    Older rows that only carry ``{ts, endpoint, cost_usd}`` are still readable
    — missing fields surface as ``None`` in summary / export views.
    """
    if cost_usd <= 0:
        return

    try:
        COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(COST_LOG_PATH, "a") as f:
            entry: Dict[str, Any] = {
                "ts": time.time(),
                "endpoint": endpoint,
                "cost_usd": cost_usd,
            }
            if model is not None:
                entry["model"] = model
            if wallet is not None:
                entry["wallet"] = wallet
            if network is not None:
                entry["network"] = network
            if client_kind is not None:
                entry["client_kind"] = client_kind
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


# ---------------------------------------------------------------------------
# Cost-log readers
# ---------------------------------------------------------------------------


def _parse_date(value: Optional[str]) -> Optional[float]:
    """Parse a YYYY-MM-DD or ISO 8601 date into a unix timestamp.

    Bare dates (YYYY-MM-DD) anchor to UTC midnight. Returns ``None`` when
    ``value`` is ``None``; raises ``ValueError`` on malformed input.
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = s + "T00:00:00+00:00"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"could not parse date {value!r}: {exc}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iter_cost_log(
    *,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    wallet: Optional[str] = None,
    network: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield cost-log entries that match the optional filters."""
    if not COST_LOG_PATH.exists():
        return
    try:
        text = COST_LOG_PATH.read_text()
    except OSError:
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        if from_ts is not None and ts < from_ts:
            continue
        if to_ts is not None and ts > to_ts:
            continue
        if wallet is not None and entry.get("wallet") != wallet:
            continue
        if network is not None and entry.get("network") != network:
            continue
        yield entry


def _group_key(entry: Dict[str, Any], group_by: str) -> str:
    """Compute the bucket key for a given grouping field."""
    if group_by == "day":
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)):
            return "unknown"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    if group_by == "month":
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)):
            return "unknown"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
    return str(entry.get(group_by) or "unknown")


_VALID_GROUP_BY = {"endpoint", "model", "wallet", "network", "client_kind", "day", "month"}


def get_cost_log_summary(
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    wallet: Optional[str] = None,
    network: Optional[str] = None,
    group_by: str = "endpoint",
) -> Dict[str, Any]:
    """Read the cost log and return an aggregated summary.

    Args:
        from_date: ISO date / datetime — entries strictly older are skipped.
        to_date: ISO date / datetime — entries strictly newer are skipped.
        wallet: Filter to a single wallet address.
        network: Filter to a single network (``base-mainnet`` etc.).
        group_by: One of ``endpoint`` (default), ``model``, ``wallet``,
            ``network``, ``client_kind``, ``day``, ``month``.

    Returns:
        ``{"from_date", "to_date", "total_usd", "calls", "group_by",
        "groups"}`` where ``groups`` maps each bucket key to
        ``{"calls": int, "cost_usd": float}``. When ``group_by == "endpoint"``
        the response also includes a ``by_endpoint`` alias mapping endpoint
        path to total cost (a float) for backwards compatibility with the
        original 3-key shape.
    """
    if group_by not in _VALID_GROUP_BY:
        raise ValueError(f"group_by must be one of {sorted(_VALID_GROUP_BY)}; got {group_by!r}")

    from_ts = _parse_date(from_date)
    to_ts = _parse_date(to_date)

    total = 0.0
    calls = 0
    groups: Dict[str, Dict[str, Any]] = {}

    for entry in _iter_cost_log(from_ts=from_ts, to_ts=to_ts, wallet=wallet, network=network):
        cost = float(entry.get("cost_usd") or 0.0)
        bucket = _group_key(entry, group_by)
        total += cost
        calls += 1
        slot = groups.setdefault(bucket, {"calls": 0, "cost_usd": 0.0})
        slot["calls"] += 1
        slot["cost_usd"] += cost

    result: Dict[str, Any] = {
        "from_date": from_date,
        "to_date": to_date,
        "total_usd": total,
        "calls": calls,
        "group_by": group_by,
        "groups": groups,
    }
    # Backwards-compat: callers that depended on the historical
    # ``by_endpoint`` mapping (cost only, no calls) keep it when the user
    # is grouping by endpoint.
    if group_by == "endpoint":
        result["by_endpoint"] = {k: v["cost_usd"] for k, v in groups.items()}
    return result


# ---------------------------------------------------------------------------
# Cost-log exporters
# ---------------------------------------------------------------------------


_EXPORT_COLUMNS = (
    "ts_iso",
    "endpoint",
    "model",
    "wallet",
    "network",
    "client_kind",
    "cost_usd",
)


def _entry_to_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one raw cost-log entry into the export record shape."""
    ts = entry.get("ts")
    ts_iso = (
        datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if isinstance(ts, (int, float))
        else None
    )
    return {
        "ts_iso": ts_iso,
        "endpoint": entry.get("endpoint"),
        "model": entry.get("model"),
        "wallet": entry.get("wallet"),
        "network": entry.get("network"),
        "client_kind": entry.get("client_kind"),
        "cost_usd": float(entry.get("cost_usd") or 0.0),
    }


def _filtered_records(
    *,
    from_date: Optional[str],
    to_date: Optional[str],
    wallet: Optional[str],
    network: Optional[str],
) -> List[Dict[str, Any]]:
    from_ts = _parse_date(from_date)
    to_ts = _parse_date(to_date)
    return [
        _entry_to_record(e)
        for e in _iter_cost_log(from_ts=from_ts, to_ts=to_ts, wallet=wallet, network=network)
    ]


def export_cost_log_csv(
    output_path: Optional[Union[str, Path]] = None,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    wallet: Optional[str] = None,
    network: Optional[str] = None,
) -> str:
    """Render filtered cost-log entries as CSV.

    Columns: ``ts_iso, endpoint, model, wallet, network, client_kind, cost_usd``.

    When ``output_path`` is supplied the CSV is also written to that file (the
    parent directory is created if missing). The CSV text is always returned
    so callers can pipe it into other tools.
    """
    records = _filtered_records(
        from_date=from_date, to_date=to_date, wallet=wallet, network=network
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(_EXPORT_COLUMNS))
    writer.writeheader()
    for record in records:
        writer.writerow({k: ("" if record.get(k) is None else record[k]) for k in _EXPORT_COLUMNS})
    text = buffer.getvalue()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def export_cost_log_json(
    output_path: Optional[Union[str, Path]] = None,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    wallet: Optional[str] = None,
    network: Optional[str] = None,
) -> str:
    """Render filtered cost-log entries as a JSON array of records.

    Same fields as ``export_cost_log_csv``. Pretty-printed with 2-space
    indentation. Returns the JSON text; writes to ``output_path`` when given.
    """
    records = _filtered_records(
        from_date=from_date, to_date=to_date, wallet=wallet, network=network
    )
    text = json.dumps(records, indent=2, default=str)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text
