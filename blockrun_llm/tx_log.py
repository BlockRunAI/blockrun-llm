"""
Opt-in per-transaction log for paid BlockRun API calls.

When a client is constructed with ``transaction_log=True`` (or with the
``BLOCKRUN_TX_LOG`` env var set), every paid call appends ONE plain-text
line to a project-local file — default ``./log/transactions.log``. The
format is designed to be eyeballable in a terminal and ``grep``-friendly::

    2026-05-21 15:44:46  chat  anthropic/claude-sonnet-4.6    in=    3  out=4  $0.034137  0x6513d128...

Columns (single space between blocks, two spaces between fields):

* ``ts``        local timestamp ``YYYY-MM-DD HH:MM:SS``
* ``endpoint``  short tag (``chat``, ``image``, ``video``, ``search``, …)
* ``model``     left-padded to 30 chars
* ``in=N``      prompt tokens (right-aligned width 5)
* ``out=N``     completion tokens
* ``$cost``     six-decimal USD ``$0.034137``
* ``tx…``       first 10 chars of the on-chain settlement hash + ``…``

This log is **independent of the ``~/.blockrun/cache`` layer**: enabling
``transaction_log`` does not change the cache, the response archive, or
``cost_log.jsonl``. It just adds a clean, human-readable ledger next to
your code, with the on-chain tx hash so each row is verifiable against
the chain explorer.

All writes are best-effort and swallow OSErrors so a read-only filesystem
can never break a paid call.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


DEFAULT_LOG_DIR = Path("./log")
LOG_NAME = "transactions.log"


# ---------------------------------------------------------------------------
# Settlement header decoding (X-PAYMENT-RESPONSE → on-chain dict)
# ---------------------------------------------------------------------------


def decode_settlement_header(header_value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Decode an ``X-PAYMENT-RESPONSE`` header into a settlement dict.

    The x402 facilitator returns a base64-encoded JSON describing what
    landed on chain. Field names vary by chain — EVM uses ``transaction``,
    Solana uses ``signature`` — so both are normalised to ``tx_hash``.

    Returns ``None`` when the header is missing or unparseable; settlement
    is informational, never load-bearing.
    """
    if not header_value:
        return None
    try:
        data = json.loads(base64.b64decode(header_value))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    tx_hash = (
        data.get("transaction")
        or data.get("txHash")
        or data.get("transactionHash")
        or data.get("signature")
    )
    amount = data.get("amount") or data.get("value")
    return {
        "tx_hash": tx_hash,
        "amount_micro_usdc": str(amount) if amount is not None else None,
        "network": data.get("network"),
        "payer": data.get("payer") or data.get("from"),
        "payee": data.get("payee") or data.get("to") or data.get("recipient"),
        "success": data.get("success"),
        "raw": data,
    }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_log_dir(option: Union[bool, str, "os.PathLike[str]", Path, None]) -> Optional[Path]:
    """Translate the ``transaction_log=...`` constructor argument into a Path.

    ``True``       → default ``./log``
    string / Path  → that path (``~`` expanded)
    ``None``       → honor ``BLOCKRUN_TX_LOG`` env var (``1``/``true`` → default,
                     anything else → that path); env unset → disabled
    ``False``      → disabled
    """
    if option is None:
        env = os.environ.get("BLOCKRUN_TX_LOG")
        if not env:
            return None
        if env.strip().lower() in {"1", "true", "yes", "on"}:
            return DEFAULT_LOG_DIR
        return Path(env).expanduser()
    if option is False:
        return None
    if option is True:
        return DEFAULT_LOG_DIR
    return Path(option).expanduser()


# ---------------------------------------------------------------------------
# Endpoint → short tag mapping (matches the example in the README)
# ---------------------------------------------------------------------------


def _endpoint_tag(endpoint: str) -> str:
    """Compress an API path into the 4–6 char tag used in the log."""
    if "/v1/chat/" in endpoint:
        return "chat"
    if "/v1/image" in endpoint:
        return "image"
    if "/v1/video" in endpoint:
        return "video"
    if "/v1/music" in endpoint or "/v1/audio" in endpoint:
        return "music"
    if "/v1/search" in endpoint:
        return "search"
    if "/v1/voice" in endpoint:
        return "voice"
    if "/v1/phone" in endpoint:
        return "phone"
    if "/v1/surf" in endpoint:
        return "surf"
    if "/v1/pm/" in endpoint:
        return "pm"
    if "/v1/price" in endpoint:
        return "price"
    # Fallback: last path segment
    tail = endpoint.rstrip("/").rsplit("/", 1)[-1]
    return tail[:6] or "call"


# ---------------------------------------------------------------------------
# Token-count extraction
# ---------------------------------------------------------------------------


def _extract_tokens(response: Any) -> tuple[int, int]:
    """Best-effort ``(prompt_tokens, completion_tokens)`` from a chat response.

    Handles both the OpenAI-shaped dict (``usage.prompt_tokens`` /
    ``usage.completion_tokens``) and the pydantic ``ChatResponse`` model
    used by the SDK. Returns ``(0, 0)`` when no usage is reported, which
    is the right answer for image / video / search calls.
    """
    if response is None:
        return 0, 0
    usage: Any = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    else:
        prompt = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
        completion = (
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        )
    try:
        return int(prompt), int(completion)
    except (TypeError, ValueError):
        return 0, 0


# ---------------------------------------------------------------------------
# Row formatter
# ---------------------------------------------------------------------------


def format_row(
    *,
    ts: Optional[float] = None,
    endpoint: str,
    model: Optional[str],
    in_tokens: int,
    out_tokens: int,
    cost_usd: float,
    tx_hash: Optional[str],
) -> str:
    """Format one log row exactly like the example in the module docstring."""
    if ts is None:
        ts = time.time()
    when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    tag = _endpoint_tag(endpoint)
    model_str = (model or "-")[:30].ljust(30)
    tx_str = f"{tx_hash[:10]}…" if tx_hash else "(no-tx)"
    return (
        f"{when}  {tag:<5}  {model_str}  "
        f"in={in_tokens:>5}  out={out_tokens:<3}  ${cost_usd:.6f}  {tx_str}"
    )


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TransactionLogger:
    """Appends one plain-text row per paid call to a project-local file.

    Construct directly to bypass the client wiring::

        logger = TransactionLogger("./log")
        logger.log(
            endpoint="/v1/chat/completions",
            request=body,
            response=chat_response,
            cost_usd=0.034137,
            model="anthropic/claude-sonnet-4.6",
            settlement={"tx_hash": "0x6513d128…"},
        )

    Most callers will let ``LLMClient`` / ``SolanaLLMClient`` build one
    automatically via the ``transaction_log=`` constructor argument.
    """

    def __init__(self, directory: Union[str, "os.PathLike[str]", Path] = DEFAULT_LOG_DIR):
        self.directory = Path(directory).expanduser()
        self.path = self.directory / LOG_NAME

    def log(
        self,
        *,
        endpoint: str,
        request: Dict[str, Any],
        response: Any,
        cost_usd: float,
        model: Optional[str] = None,
        wallet: Optional[str] = None,
        network: Optional[str] = None,
        client_kind: Optional[str] = None,
        settlement: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Append one formatted row to ``./log/transactions.log``.

        Returns the log path on success, or ``None`` if the file could not
        be created — logging is best-effort and never raises.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None

        in_tokens, out_tokens = _extract_tokens(response)
        tx_hash = (settlement or {}).get("tx_hash") if settlement else None
        row = format_row(
            ts=time.time(),
            endpoint=endpoint,
            model=model or (request.get("model") if isinstance(request, dict) else None),
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            cost_usd=float(cost_usd or 0.0),
            tx_hash=tx_hash,
        )

        # Silence unused-arg warnings without changing the public API — the
        # extra metadata is intentionally accepted for future structured
        # outputs (e.g. a `.jsonl` companion behind a flag) but the text
        # log keeps just what fits on one line.
        del wallet, network, client_kind

        try:
            with open(self.path, "a") as f:
                f.write(row + "\n")
        except OSError:
            return None
        return self.path

    def entries(self) -> list[str]:
        """Return every log line as a list of strings (oldest first).

        Useful for tests and for users who want to reconcile the log
        against an on-chain explorer programmatically.
        """
        if not self.path.exists():
            return []
        try:
            return [
                line.rstrip("\n") for line in self.path.read_text().splitlines() if line.strip()
            ]
        except OSError:
            return []


__all__ = [
    "TransactionLogger",
    "decode_settlement_header",
    "format_row",
    "DEFAULT_LOG_DIR",
]
