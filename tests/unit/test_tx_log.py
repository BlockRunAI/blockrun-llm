"""Unit tests for the opt-in project-local transaction log.

Covered surface:
* ``TransactionLogger.log`` formats one plain-text row per call.
* The on-chain ``tx_hash`` is pulled from a decoded ``X-PAYMENT-RESPONSE``
  payload (both EVM ``transaction`` and Solana ``signature`` field names).
* ``format_row`` matches the column layout shown in the README so future
  reformat regressions get caught immediately.
* ``_resolve_log_dir`` honors the constructor argument + ``BLOCKRUN_TX_LOG``
  env var fallback.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


from blockrun_llm.tx_log import (
    DEFAULT_LOG_DIR,
    TransactionLogger,
    _resolve_log_dir,
    decode_settlement_header,
    format_row,
)


# ---------------------------------------------------------------------------
# Logger writes
# ---------------------------------------------------------------------------


def test_log_writes_one_row(tmp_path):
    logger = TransactionLogger(tmp_path)
    logger.log(
        endpoint="/v1/chat/completions",
        request={"model": "openai/gpt-5.5", "messages": []},
        response={"usage": {"prompt_tokens": 14, "completion_tokens": 18}},
        cost_usd=0.001,
        settlement={"tx_hash": "0x421796a3deadbeef"},
    )
    rows = logger.entries()
    assert len(rows) == 1
    row = rows[0]
    assert "chat" in row
    assert "openai/gpt-5.5" in row
    assert "in=   14" in row
    assert "out=18" in row
    assert "$0.001000" in row
    assert "0x421796a3" in row  # truncated to 10 chars + ellipsis


def test_log_appends(tmp_path):
    """Two calls → two lines, oldest first."""
    logger = TransactionLogger(tmp_path)
    for i in range(2):
        logger.log(
            endpoint="/v1/chat/completions",
            request={"model": "openai/gpt-5.5"},
            response={"usage": {"prompt_tokens": i, "completion_tokens": i}},
            cost_usd=0.001,
            settlement={"tx_hash": f"0x{i:064x}"},
        )
    rows = logger.entries()
    assert len(rows) == 2
    # Second row should have the second tx hash
    assert "0x00000000" in rows[0]
    assert rows[0] != rows[1]


def test_log_without_settlement_emits_placeholder(tmp_path):
    """A free / cached call → no tx hash → ``(no-tx)``."""
    logger = TransactionLogger(tmp_path)
    logger.log(
        endpoint="/v1/chat/completions",
        request={"model": "free/model"},
        response={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        cost_usd=0.0,
    )
    assert "(no-tx)" in logger.entries()[0]


# ---------------------------------------------------------------------------
# Settlement header decoding
# ---------------------------------------------------------------------------


def _b64(obj):
    return base64.b64encode(json.dumps(obj).encode()).decode()


def test_decode_evm_settlement():
    settlement = decode_settlement_header(
        _b64(
            {
                "success": True,
                "transaction": "0xdeadbeef",
                "network": "eip155:8453",
                "payer": "0xabc",
                "payee": "0xdef",
                "amount": "1000",
            }
        )
    )
    assert settlement["tx_hash"] == "0xdeadbeef"
    assert settlement["amount_micro_usdc"] == "1000"
    assert settlement["network"] == "eip155:8453"


def test_decode_solana_settlement_uses_signature():
    settlement = decode_settlement_header(
        _b64({"signature": "5h7Kabc…", "network": "solana:mainnet", "amount": 500})
    )
    assert settlement["tx_hash"] == "5h7Kabc…"
    assert settlement["amount_micro_usdc"] == "500"


def test_decode_returns_none_for_missing_or_garbage():
    assert decode_settlement_header(None) is None
    assert decode_settlement_header("not-base64") is None
    # Valid base64 but not JSON
    assert decode_settlement_header(base64.b64encode(b"not-json").decode()) is None


# ---------------------------------------------------------------------------
# Row formatting (regression guard for the README example)
# ---------------------------------------------------------------------------


def test_format_row_matches_readme_layout():
    row = format_row(
        ts=1747842286.0,  # arbitrary
        endpoint="/v1/chat/completions",
        model="anthropic/claude-sonnet-4.6",
        in_tokens=3,
        out_tokens=4,
        cost_usd=0.034137,
        tx_hash="0x6513d12812345",
    )
    assert "chat" in row
    assert "anthropic/claude-sonnet-4.6" in row
    assert "in=    3" in row
    assert "out=4" in row
    assert "$0.034137" in row
    assert "0x6513d128" in row
    assert row.endswith("0x6513d128…")


# ---------------------------------------------------------------------------
# Path resolution / env-var fallback
# ---------------------------------------------------------------------------


def test_resolve_log_dir_truthy_returns_default():
    assert _resolve_log_dir(True) == DEFAULT_LOG_DIR


def test_resolve_log_dir_path_passes_through(tmp_path):
    assert _resolve_log_dir(str(tmp_path)) == Path(str(tmp_path))


def test_resolve_log_dir_false_is_disabled():
    assert _resolve_log_dir(False) is None


def test_resolve_log_dir_env_enables_default(monkeypatch):
    monkeypatch.setenv("BLOCKRUN_TX_LOG", "1")
    assert _resolve_log_dir(None) == DEFAULT_LOG_DIR


def test_resolve_log_dir_env_path(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOCKRUN_TX_LOG", str(tmp_path))
    assert _resolve_log_dir(None) == Path(str(tmp_path))


def test_resolve_log_dir_env_missing_is_disabled(monkeypatch):
    monkeypatch.delenv("BLOCKRUN_TX_LOG", raising=False)
    assert _resolve_log_dir(None) is None


# ---------------------------------------------------------------------------
# Best-effort behaviour
# ---------------------------------------------------------------------------


def test_log_into_unwritable_dir_returns_none(tmp_path):
    """A read-only parent must not crash a paid call."""
    parent = tmp_path / "ro"
    parent.mkdir()
    parent.chmod(0o500)  # read+execute, no write
    try:
        logger = TransactionLogger(parent / "log")
        result = logger.log(
            endpoint="/v1/chat/completions",
            request={"model": "x"},
            response={},
            cost_usd=0.0,
        )
        assert result is None
    finally:
        parent.chmod(0o700)  # restore so pytest can clean up


def test_pydantic_usage_object_is_handled():
    """Pydantic response objects with usage attrs should not crash the row.

    Mirrors how ``LLMClient`` passes a ``ChatResponse`` in for chat calls."""

    class Usage:
        prompt_tokens = 7
        completion_tokens = 3

    class Resp:
        usage = Usage()

    row = format_row(
        endpoint="/v1/chat/completions",
        model="openai/gpt-5.5",
        in_tokens=7,
        out_tokens=3,
        cost_usd=0.001,
        tx_hash="0xabc",
    )
    assert "in=    7" in row and "out=3" in row
    # _extract_tokens path: feed via TransactionLogger
    from blockrun_llm.tx_log import _extract_tokens

    assert _extract_tokens(Resp()) == (7, 3)
