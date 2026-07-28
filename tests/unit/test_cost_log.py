"""Unit tests for the cost-log reader / exporter.

Uses monkeypatch to redirect ``COST_LOG_PATH`` to a temp file so tests don't
touch the real ``~/.blockrun/cost_log.jsonl``.
"""

from __future__ import annotations

import json
import time

import pytest

from blockrun_llm import cache


def _write_log(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


@pytest.fixture
def temp_log(tmp_path, monkeypatch):
    log = tmp_path / "cost_log.jsonl"
    monkeypatch.setattr(cache, "COST_LOG_PATH", log)
    return log


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_legacy_by_endpoint_alias_still_exposed(temp_log):
    """When grouping by endpoint, ``by_endpoint`` is still emitted as a
    backwards-compat alias mapping endpoint -> total cost (float)."""
    _write_log(
        temp_log,
        [
            {"ts": time.time(), "endpoint": "/v1/chat/completions", "cost_usd": 0.001},
            {"ts": time.time(), "endpoint": "/v1/chat/completions", "cost_usd": 0.002},
            {"ts": time.time(), "endpoint": "/v1/search", "cost_usd": 0.01},
        ],
    )
    summary = cache.get_cost_log_summary()
    assert summary["calls"] == 3
    assert summary["total_usd"] == pytest.approx(0.013)
    # New shape always includes total_usd / calls / groups + by_endpoint alias
    assert "groups" in summary
    assert summary["by_endpoint"]["/v1/chat/completions"] == pytest.approx(0.003)
    assert summary["by_endpoint"]["/v1/search"] == pytest.approx(0.01)


def test_legacy_3_field_rows_still_readable(temp_log):
    """Older entries with only ``{ts, endpoint, cost_usd}`` must aggregate
    cleanly alongside new entries that carry the full metadata."""
    now = time.time()
    _write_log(
        temp_log,
        [
            {"ts": now, "endpoint": "/v1/chat/completions", "cost_usd": 0.001},  # old
            {
                "ts": now,
                "endpoint": "/v1/chat/completions",
                "cost_usd": 0.002,
                "model": "openai/gpt-5.2",
                "wallet": "0xabc",
                "network": "base-mainnet",
                "client_kind": "LLMClient",
            },
        ],
    )
    summary = cache.get_cost_log_summary(group_by="model")
    # New schema: legacy row groups under "unknown" model, new row under id.
    assert summary["total_usd"] == pytest.approx(0.003)
    assert summary["calls"] == 2
    assert summary["groups"]["openai/gpt-5.2"]["cost_usd"] == pytest.approx(0.002)
    assert summary["groups"]["unknown"]["cost_usd"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Filters + grouping
# ---------------------------------------------------------------------------


def test_group_by_model_aggregates_correctly(temp_log):
    now = time.time()
    _write_log(
        temp_log,
        [
            {"ts": now, "endpoint": "/v1/chat/completions", "cost_usd": 0.001, "model": "a"},
            {"ts": now, "endpoint": "/v1/chat/completions", "cost_usd": 0.002, "model": "a"},
            {"ts": now, "endpoint": "/v1/chat/completions", "cost_usd": 0.005, "model": "b"},
        ],
    )
    summary = cache.get_cost_log_summary(group_by="model")
    assert summary["groups"]["a"] == {"calls": 2, "cost_usd": pytest.approx(0.003)}
    assert summary["groups"]["b"] == {"calls": 1, "cost_usd": pytest.approx(0.005)}


def test_wallet_filter_isolates_to_one_wallet(temp_log):
    now = time.time()
    _write_log(
        temp_log,
        [
            {"ts": now, "endpoint": "/v1/x", "cost_usd": 0.01, "wallet": "0xa"},
            {"ts": now, "endpoint": "/v1/x", "cost_usd": 0.02, "wallet": "0xb"},
            {"ts": now, "endpoint": "/v1/x", "cost_usd": 0.04, "wallet": "0xa"},
        ],
    )
    summary = cache.get_cost_log_summary(wallet="0xa")
    assert summary["calls"] == 2
    assert summary["total_usd"] == pytest.approx(0.05)


def test_date_range_filters_correctly(temp_log):
    """``YYYY-MM-DD`` strings anchor to UTC midnight; pass distinct
    from/to dates to bracket a window."""
    from datetime import datetime, timezone

    # Pick a base timestamp at UTC noon on a known date so the entries are
    # clearly inside / outside the window.
    base = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    _write_log(
        temp_log,
        [
            {"ts": base - 86_400, "endpoint": "/x", "cost_usd": 0.01},  # 2026-05-08
            {"ts": base, "endpoint": "/x", "cost_usd": 0.02},  # 2026-05-09 12:00
            {"ts": base + 86_400, "endpoint": "/x", "cost_usd": 0.04},  # 2026-05-10
        ],
    )
    summary = cache.get_cost_log_summary(from_date="2026-05-09", to_date="2026-05-10")
    # Window is [2026-05-09 00:00 UTC, 2026-05-10 00:00 UTC] — only the
    # 2026-05-09 noon entry should be inside.
    assert summary["calls"] == 1
    assert summary["total_usd"] == pytest.approx(0.02)


def test_invalid_group_by_raises(temp_log):
    _write_log(temp_log, [])
    with pytest.raises(ValueError):
        cache.get_cost_log_summary(group_by="bogus")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_export_csv_has_header_and_rows(temp_log):
    now = time.time()
    _write_log(
        temp_log,
        [
            {
                "ts": now,
                "endpoint": "/v1/chat/completions",
                "cost_usd": 0.001,
                "model": "openai/gpt-5.2",
                "wallet": "0xabc",
                "network": "base-mainnet",
                "client_kind": "LLMClient",
            },
        ],
    )
    csv_text = cache.export_cost_log_csv()
    lines = csv_text.strip().split("\n")
    assert lines[0].startswith("ts_iso,endpoint,model,wallet,network,client_kind,cost_usd")
    assert "openai/gpt-5.2" in lines[1]
    assert "base-mainnet" in lines[1]


def test_export_json_returns_list_of_dicts(temp_log):
    now = time.time()
    _write_log(
        temp_log,
        [
            {"ts": now, "endpoint": "/x", "cost_usd": 0.01, "model": "m"},
        ],
    )
    payload = json.loads(cache.export_cost_log_json())
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["model"] == "m"
    assert payload[0]["cost_usd"] == pytest.approx(0.01)
    assert "ts_iso" in payload[0]


def test_export_csv_writes_to_path(temp_log, tmp_path):
    now = time.time()
    _write_log(temp_log, [{"ts": now, "endpoint": "/x", "cost_usd": 0.001}])
    out = tmp_path / "out.csv"
    cache.export_cost_log_csv(out)
    assert out.exists()
    assert "ts_iso" in out.read_text()
