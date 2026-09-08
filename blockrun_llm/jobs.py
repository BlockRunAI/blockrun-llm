"""Polling for the gateway's async media jobs.

A slow generation answers ``202`` with a ``poll_url`` and settles on the first
poll that observes ``completed``, so a poll that times out has cost nothing.
Images and music share this loop: the same statuses, the same settlement rule,
the same two rails. One copy, so the two clients cannot drift apart on how a
job ends.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .apikey import raise_for_api_key_402, resolve_poll_url
from .types import APIError, retry_after_of
from .validation import build_payment_rejected_error, sanitize_error_response


def absolute_poll_url(url: str, api_url: str, api_key: str | None) -> str:
    """Resolve a relative ``poll_url`` against the configured API host.

    Server-returned poll URLs look like ``/api/v1/images/generations/<id>``;
    ``api_url`` already ends with ``/api`` on the wallet rail, and the account
    rail serves the same route without that prefix.
    """
    if url.startswith(("http://", "https://")):
        return url
    return resolve_poll_url(url, api_url, api_key)


def poll_until_completed(
    client: httpx.Client,
    submit_resp: httpx.Response,
    payment_payload: str | None,
    *,
    api_url: str,
    api_key: str | None,
    interval_seconds: float,
    budget_seconds: float,
    label: str,
) -> dict[str, Any]:
    """Poll ``poll_url`` until the job completes; return the completed body.

    ``payment_payload`` is the create's PAYMENT-SIGNATURE on the wallet rail
    (the job is bound to that wallet and settles against it) and ``None`` on
    the account rail, where the key rides on the client's default headers.
    ``label`` names the product in errors ("Image", "Music").
    """
    try:
        submit_data = submit_resp.json()
    except Exception:
        submit_data = {}

    poll_url_rel = submit_data.get("poll_url")
    job_id = submit_data.get("id")
    if not poll_url_rel:
        raise APIError("Slow-path 202 missing poll_url", 202, {"response": submit_data})

    poll_url = absolute_poll_url(poll_url_rel, api_url, api_key)
    poll_headers = {"PAYMENT-SIGNATURE": payment_payload} if payment_payload else {}
    deadline = time.monotonic() + budget_seconds
    last_status = submit_data.get("status", "queued")

    while time.monotonic() < deadline:
        time.sleep(interval_seconds)

        poll_resp = client.get(poll_url, headers=poll_headers)
        try:
            poll_data = poll_resp.json()
        except Exception:
            poll_data = {}
        last_status = poll_data.get("status", last_status)

        if poll_resp.status_code == 402:
            # Account rail: a 402 is the account being out of credit, not a
            # challenge to sign. Nothing here can sign, so say so plainly.
            raise_for_api_key_402(poll_resp, api_key)
            # Settlement failed on this poll — surface the gateway reason.
            raise build_payment_rejected_error(poll_resp)

        if last_status == "failed":
            raise APIError(
                f"{label} generation failed upstream: {poll_data.get('error', 'unknown')}",
                poll_resp.status_code,
                sanitize_error_response(poll_data if isinstance(poll_data, dict) else {}),
                retry_after=retry_after_of(poll_resp),
            )

        if poll_resp.status_code == 200 and last_status == "completed":
            tx_hash = poll_resp.headers.get("x-payment-receipt")
            if tx_hash and "txHash" not in poll_data:
                poll_data["txHash"] = tx_hash
            return poll_data

        if poll_resp.status_code in (202, 504):
            # 202 = still queued/in_progress; 504 = transient upstream
            # hiccup. Both retriable inside the budget.
            continue

        if poll_resp.status_code != 200:
            try:
                error_body = poll_resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"{label} poll failed: HTTP {poll_resp.status_code}",
                poll_resp.status_code,
                sanitize_error_response(error_body),
                retry_after=retry_after_of(poll_resp),
            )

    raise APIError(
        (
            f"{label} generation did not complete within {budget_seconds:.0f}s "
            f"(last status: {last_status}). Settlement only happens on "
            "completion, so no payment was taken."
        ),
        504,
        {"id": job_id, "last_status": last_status},
    )
