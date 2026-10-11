"""Proof of job ownership for Solana media polls, without a payment.

A Solana x402 payment is a transaction pinned to a recent blockhash that dies
in ~60-90s, so polling a multi-minute video by re-signing a payment meant a
fresh signature on almost every poll. sol.blockrun.ai also accepts a short
ed25519 signature over the job id from the wallet that submitted it:

    message  = "blockrun-poll:v1:<kind>:<job id>:<unix seconds>"
    headers  x-poll-wallet:    base58 public key of the submitting wallet
             x-poll-timestamp: the <unix seconds> in the message
             x-poll-signature: base58 ed25519 signature over the UTF-8 message

``<kind>`` is ``video``, ``audio`` (music) or ``image``; ``<job id>`` is the
last path segment of the ``poll_url`` the POST returned, URL-decoded, which is
the value the gateway verifies against. The gateway accepts a timestamp within
300s of its clock, so the headers are signed fresh for every poll.

Server contract: blockrun-sol ``src/lib/poll-ownership.ts``. Base never sees
these headers: an EIP-3009 authorization lives for hours, so the Base clients
keep polling with their payment signature.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

PollKind = Literal["video", "audio", "image"]

POLL_WALLET_HEADER = "x-poll-wallet"
POLL_TIMESTAMP_HEADER = "x-poll-timestamp"
POLL_SIGNATURE_HEADER = "x-poll-signature"

# The gateway's poll routes, by the <kind> its signature must name. Matched on
# the path the POST returned, so an unknown route never gets a proof.
_KIND_BY_ROUTE: dict[str, PollKind] = {
    "videos": "video",
    "audio": "audio",
    "images": "image",
}

# What each poll route answers a refused proof with: the same bytes as a job id
# that does not exist (wrong wallet, bad signature, stale timestamp, a job bound
# to no wallet).
_REFUSAL_BY_KIND: dict[PollKind, tuple[int, str]] = {
    "video": (400, "Invalid job id"),
    "audio": (404, "Job not found"),
    "image": (404, "Job not found"),
}


def poll_kind_for_url(poll_url: str) -> PollKind | None:
    """The ``<kind>`` for a ``.../v1/<route>/generations/<id>`` poll URL, or
    ``None`` for any other path (no proof is sent there)."""
    segments = [s for s in urlsplit(poll_url).path.split("/") if s]
    if len(segments) < 4 or segments[-2] != "generations" or segments[-4] != "v1":
        return None
    return _KIND_BY_ROUTE.get(segments[-3])


def poll_job_id(poll_url: str) -> str:
    """The job id the gateway verifies: the poll URL's last path segment,
    URL-decoded (the route reads ``decodeURIComponent(params.id)``)."""
    path = urlsplit(poll_url).path.rstrip("/")
    return unquote(path.rsplit("/", 1)[-1])


def poll_auth_message(kind: PollKind, job_id: str, timestamp_s: int) -> str:
    """The exact text the wallet signs."""
    return f"blockrun-poll:v1:{kind}:{job_id}:{timestamp_s}"


def poll_auth_headers(
    keypair: Any, kind: PollKind, job_id: str, *, now_s: int | None = None
) -> dict[str, str]:
    """Sign a fresh ownership proof for one poll.

    ``keypair`` is the wallet's ``solders.keypair.Keypair`` — the same key that
    signed the submit payment, which is the payer the gateway bound the job to.
    """
    timestamp_s = int(time.time()) if now_s is None else int(now_s)
    message = poll_auth_message(kind, job_id, timestamp_s).encode("utf-8")
    return {
        POLL_WALLET_HEADER: str(keypair.pubkey()),
        POLL_TIMESTAMP_HEADER: str(timestamp_s),
        POLL_SIGNATURE_HEADER: str(keypair.sign_message(message)),
    }


def is_poll_ownership_refusal(kind: PollKind, status_code: int, body: Any) -> bool:
    """True when a poll carrying x-poll-* got the route's "no such job" answer."""
    expected_status, expected_error = _REFUSAL_BY_KIND[kind]
    return (
        status_code == expected_status
        and isinstance(body, Mapping)
        and body.get("error") == expected_error
    )


def poll_keypair_of(signer: Any) -> Any:
    """The ``solders`` Keypair behind an x402 ``KeypairSigner``, or ``None``
    when the signer does not expose one (then polls keep the payment path)."""
    try:
        from solders.keypair import Keypair
    except ImportError:  # pragma: no cover - solders ships with x402[svm]
        return None
    keypair = getattr(signer, "keypair", None)
    return keypair if isinstance(keypair, Keypair) else None


__all__ = [
    "POLL_SIGNATURE_HEADER",
    "POLL_TIMESTAMP_HEADER",
    "POLL_WALLET_HEADER",
    "PollKind",
    "is_poll_ownership_refusal",
    "poll_auth_headers",
    "poll_auth_message",
    "poll_job_id",
    "poll_keypair_of",
    "poll_kind_for_url",
]
