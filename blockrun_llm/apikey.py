"""The API-key rail.

BlockRun sells the same catalogue through two front doors. The x402 rail
(``blockrun.ai`` / ``sol.blockrun.ai``) authenticates a caller by wallet
signature and settles USDC on-chain per request. The account rail
(``api.blockrun.ai``) authenticates a caller by API key and draws down prepaid
credit held against the account at `user.blockrun.ai <https://user.blockrun.ai>`_.

The two are the same backend and the same response shapes, which is what makes
one client able to serve both: the only differences are the host, the header
that authenticates, and the fact that a 402 on the account rail means "out of
credit" rather than "sign this".

A key is never a wallet. In API-key mode ``self.account`` is ``None``, there is
no address, and nothing is signed locally — so the wallet-only helpers report
that plainly instead of returning a zero that looks like an answer.

Every client class in this package wires itself up through
:func:`configure_credential`, so the rail is one decision made in one place
rather than fourteen copies that can drift.
"""

from __future__ import annotations

import os
from typing import Any

from .types import APIError, PaymentError

#: Prefix every BlockRun API key carries. It is what lets one credential
#: parameter accept either kind: a hex private key can never start with ``brk_``.
API_KEY_PREFIX = "brk_"

#: The account-rail gateway. Unlike the x402 default it carries no ``/api``
#: suffix: api.blockrun.ai serves ``/v1/...`` at the root and answers
#: ``/api/v1/...`` with a ``wrong_host`` error.
DEFAULT_API_KEY_URL = "https://api.blockrun.ai"

#: Holds a BlockRun API key. Setting it puts every client in this process on the
#: account rail, the Solana clients included — the key is the payment method, so
#: the chain stops being a question.
ENV_API_KEY = "BLOCKRUN_API_KEY"

#: Overrides the account-rail host. Deliberately not ``BLOCKRUN_API_URL``: that
#: one names the x402 gateway, and a developer who has it pointed at a private
#: x402 deployment must not have an API-key client silently follow it there and
#: hand over the key.
ENV_API_KEY_URL = "BLOCKRUN_API_KEY_URL"

#: Which rail a client pays on, as reported by ``client.payment_mode``.
PAYMENT_MODE_WALLET = "wallet"
PAYMENT_MODE_API_KEY = "apikey"


def is_api_key(credential: str | None) -> bool:
    """Is this credential a BlockRun API key rather than a wallet private key?"""
    return bool(credential) and str(credential).strip().startswith(API_KEY_PREFIX)


def resolve_api_key(credential: str | None) -> str | None:
    """Decide whether a constructor call is an API-key call.

    Precedence, and the reason for it: an explicit argument beats everything,
    because the caller wrote it at the call site. Then ``BLOCKRUN_API_KEY``
    beats the wallet variables, because it is the new variable — a developer
    who has not set it keeps the wallet behaviour they already had, and one who
    has set it meant to, even if an old ``BLOCKRUN_WALLET_KEY`` is still sitting
    in their profile. ``client.payment_mode`` exists so that decision is never
    invisible.
    """
    if is_api_key(credential):
        return str(credential).strip()
    # An explicit non-key credential is a deliberate choice of the x402 rail and
    # must not be overridden by the environment.
    if credential and str(credential).strip():
        return None
    env = os.environ.get(ENV_API_KEY, "").strip()
    return env if is_api_key(env) else None


def api_key_base_url(api_url: str | None = None) -> str:
    """Resolve the account-rail host: explicit argument, env override, default."""
    if api_url and api_url.strip():
        return api_url.strip().rstrip("/")
    env = os.environ.get(ENV_API_KEY_URL, "").strip()
    if env:
        return env.rstrip("/")
    return DEFAULT_API_KEY_URL


def auth_headers(api_key: str | None) -> dict[str, str]:
    """The header that authenticates on the account rail, empty on the x402 one.

    ``Authorization: Bearer`` is the OpenAI-SDK shape; the gateway also accepts
    ``x-api-key`` for Anthropic-shaped clients. One is sent, not both, so a
    proxy that logs headers records the key once.
    """
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def configure_credential(
    obj: Any,
    private_key: str | None,
    api_url: str | None,
) -> bool:
    """Put ``obj`` on the account rail if the credential says so.

    Sets ``obj.api_key``, ``obj.api_url`` and ``obj.account`` and returns True
    when an API key was found, so a caller's ``__init__`` can skip every
    wallet-loading step. Returns False and touches nothing otherwise, leaving
    the existing wallet path exactly as it was.
    """
    api_key = resolve_api_key(private_key)
    if not api_key:
        obj.api_key = None
        return False
    obj.api_key = api_key
    obj.account = None
    obj.api_url = api_key_base_url(api_url)
    return True


def payment_mode(obj: Any) -> str:
    """Which rail ``obj`` pays on. Worth checking once at startup when both a
    key and a wallet are configured: it is the difference between spending
    credit and spending USDC."""
    return PAYMENT_MODE_API_KEY if getattr(obj, "api_key", None) else PAYMENT_MODE_WALLET


def resolve_poll_url(poll_url: str, api_url: str, api_key: str | None) -> str:
    """Resolve a server-supplied relative ``poll_url`` against the API host.

    ``poll_url`` is minted by the x402 gateway and is relative to *its* host, so
    it arrives as ``/api/v1/...``. api.blockrun.ai serves the same route at
    ``/v1/...`` and answers ``/api/v1/...`` with ``wrong_host``, so on the
    account rail the prefix has to come off here — the alternative is every
    async job (video, slow images) polling a 404 until its budget runs out.
    """
    if poll_url.startswith(("http://", "https://")):
        return poll_url
    if api_key:
        return f"{api_url}{poll_url.removeprefix('/api')}"
    return f"{api_url.removesuffix('/api')}{poll_url}"


def api_key_payment_error(body: Any = None) -> PaymentError:
    """Explain a 402 that arrived on the account rail.

    On the x402 rail a 402 is the normal opening move of a conversation. On this
    one it is a refusal: the account is out of credit, suspended, or past its
    limit. Signing is not the answer and there is nothing to sign with, so the
    error says what to do instead rather than letting the caller fall into the
    wallet path and get a wallet error for a problem that has nothing to do with
    wallets.
    """
    detail = ""
    if body is not None:
        detail = str(body).strip()
        if len(detail) > 400:
            detail = detail[:400] + "…"
    message = (
        "402 from api.blockrun.ai: this account has no credit left for that call. "
        "Top up at https://user.blockrun.ai/dashboard/credits, or call one of the "
        "free models, which need no credit."
    )
    if detail:
        message = f"{message} Gateway said: {detail}"
    return PaymentError(message)


def wallet_only(helper: str) -> ValueError:
    """The error every wallet-only helper raises on the account rail.

    Naming the helper matters: "no wallet" alone leaves the caller guessing
    which of ``get_balance`` / ``onramp`` / ``get_wallet_address`` they should
    not have called.
    """
    return ValueError(
        f"{helper}() is wallet-only and this client authenticates with a BlockRun "
        f"API key. Credit balance, usage and top-ups live at "
        f"https://user.blockrun.ai/dashboard. Construct the client with a wallet "
        f"private key (or unset {ENV_API_KEY}) to use {helper}()."
    )


def missing_credential_error(*, extra: str = "") -> ValueError:
    """The 'nothing configured' error, now that a key is one of the options.

    Every client raised its own wording listing only wallet routes, which
    stopped being the whole truth the moment a key became a credential. One
    message, listing both.
    """
    lines = [
        "No credential configured. Either:",
        f"  1. Set {ENV_API_KEY} to an API key from https://user.blockrun.ai",
        "  2. Set BLOCKRUN_WALLET_KEY to a wallet private key",
        "  3. Pass either one as the first constructor argument",
    ]
    if extra:
        lines.append(f"  4. {extra}")
    lines.append("NOTE: a wallet key never leaves your machine — only signatures are sent.")
    return ValueError("\n".join(lines))


def raise_for_api_key_402(response: Any, api_key: str | None) -> None:
    """Turn a 402 on the account rail into a credit refusal, before any signing.

    A no-op on the x402 rail, so every request site can call it unconditionally.
    """
    if not api_key or response.status_code != 402:
        return
    try:
        body = response.json()
    except Exception:
        body = response.text
    raise api_key_payment_error(body)


__all__ = [
    "API_KEY_PREFIX",
    "DEFAULT_API_KEY_URL",
    "ENV_API_KEY",
    "ENV_API_KEY_URL",
    "PAYMENT_MODE_API_KEY",
    "PAYMENT_MODE_WALLET",
    "APIError",
    "api_key_base_url",
    "api_key_payment_error",
    "auth_headers",
    "configure_credential",
    "is_api_key",
    "missing_credential_error",
    "payment_mode",
    "raise_for_api_key_402",
    "resolve_api_key",
    "resolve_poll_url",
    "wallet_only",
]
