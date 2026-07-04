"""
BlockRun Solana LLM Client.

Usage:
    from blockrun_llm import SolanaLLMClient

    # SOLANA_WALLET_KEY env var (bs58-encoded Solana secret key)
    client = SolanaLLMClient()

    # Or pass key directly
    client = SolanaLLMClient(private_key="your-bs58-key")

    # Same API as LLMClient
    response = client.chat("openai/gpt-5.2", "gm Solana")
    print(response)
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import re
import sys
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import httpx

from .types import (
    ChatCompletionChunk,
    ChatResponse,
    ImageResponse,
    VideoResponse,
    MusicResponse,
    SpeechResponse,
    PortraitEnrollment,
    PortraitList,
    RealFaceInit,
    RealFaceStatus,
    RealFaceEnrollment,
    RealFaceList,
    PricePoint,
    PriceHistoryResponse,
    SymbolListResponse,
    RpcResponse,
    APIError,
    PaymentError,
    SearchResult,
    stream_choice_content,
    stream_choice_finish_reason,
    chunk_meta,
    chunk_usage_dict,
)
from .solana_wallet import get_solana_public_key
from .tx_log import TransactionLogger, decode_settlement_header, _resolve_log_dir
from .price import Category, Market, Resolution, Session
from .realface import _GROUP_ID_RE
from .validation import (
    build_payment_rejected_error,
    sanitize_error_response,
    validate_api_url,
)

try:
    from x402 import x402ClientSync
    from x402.mechanisms.svm import KeypairSigner
    from x402.mechanisms.svm.exact.register import register_exact_svm_client
    from x402.http.utils import decode_payment_required_header, encode_payment_signature_header

    _HAS_X402 = True
except ImportError:
    _HAS_X402 = False

SOLANA_API_URL = "https://sol.blockrun.ai/api"


def _create_signer(private_key: str) -> KeypairSigner:
    """Create a KeypairSigner, handling both full keypair and seed-only formats."""
    try:
        return KeypairSigner.from_base58(private_key)
    except (ValueError, Exception):
        # Fallback: might be a 32-byte seed (agentcash, etc.)
        import base58 as b58
        from solders.keypair import Keypair

        decoded = b58.b58decode(private_key)
        if len(decoded) == 32:
            kp = Keypair.from_seed(decoded)
            full_key = b58.b58encode(bytes(kp)).decode()
            return KeypairSigner.from_base58(full_key)
        raise


DEFAULT_MAX_TOKENS = 1024

# Per-use-case HTTP timeouts (seconds). The Base SDK splits these across
# multiple clients (LLMClient=120s, ImageClient=200s, MusicClient=210s,
# VideoClient=360s, ...); the Solana mega-class needs the same separation
# so a long chat or slow image generation does not silently die at 60s
# (the historical single-value default).
#
# Public callers can also override per-call via ``timeout=`` on
# ``chat_completion`` / ``image`` / ``image_edit`` / ``search``.
DEFAULT_CHAT_TIMEOUT = float(
    os.environ.get("BLOCKRUN_CHAT_TIMEOUT", "600")
)  # was 120; reasoning models need 200–300s+
DEFAULT_IMAGE_TIMEOUT = 200.0
DEFAULT_SEARCH_TIMEOUT = 300.0
DEFAULT_FAST_TIMEOUT = 30.0  # pyth / x_user_info / quick lookups

# Kept as a single fallback so old callers that pass a flat ``timeout=``
# to ``SolanaLLMClient(...)`` continue to work — but the value now matches
# the chat client's budget so a 120s chat doesn't die under the old 60s.
DEFAULT_TIMEOUT = DEFAULT_CHAT_TIMEOUT


# ---------------------------------------------------------------------------
# Permanent payment errors — don't retry, don't fall back
# ---------------------------------------------------------------------------
#
# Mirrors the gateway-side classification at
# blockrun-sol/src/lib/x402-solana.ts PERMANENT_ERRORS. These reasons are
# deterministic on the SIGNED AUTHORIZATION level — re-signing without
# fixing the root cause produces the same failure within seconds. Surfacing
# the first failure immediately drops worst-case wall-clock from ~5min
# (3 generation attempts) to one attempt's worth.
_PERMANENT_PAYMENT_PATTERNS = (
    "insufficient",  # insufficient_funds, "insufficient balance"
    "invalid signature",  # bad signing key / malformed payload
    "invalid_payload",  # gateway rejected payload shape
    "expired",  # payment_expired
    "authorization is used",  # replay-nonce hit
    "transaction_simulation_failed",  # CDP svm sim rejected (often blockhash window)
    "blockhash not found",  # blockhash already aged out — same class
    "block height exceeded",  # past slot lifetime — same class
)


def _is_permanent_payment_error(reason: str) -> bool:
    """True iff the payment reason matches a permanent-failure pattern.

    Used by both the streaming fallback decision and the raw retry
    classifier so the same policy applies to every Solana code path.
    Case-insensitive substring match — patterns above are the BlockRun
    gateway's own enums, and CDP returns the long form
    (``invalid_exact_svm_payload_transaction_simulation_failed``) which
    still contains the short form as a substring.
    """
    if not reason:
        return False
    low = reason.lower()
    return any(p in low for p in _PERMANENT_PAYMENT_PATTERNS)


# Payment failures a FRESH payment genuinely can't fix — re-running the whole
# request (new nonce, new 402 probe, new blockhash) won't help, so fail fast.
# Deliberately NARROWER than _PERMANENT_PAYMENT_PATTERNS: replay
# ("authorization is used"), amount mismatch, expired, blockhash/simulation
# errors ARE recoverable with a fresh signature and so are OMITTED here — they
# are exactly the concurrent-load failures the whole-request retry exists to fix.
_UNRECOVERABLE_PAYMENT_PATTERNS = (
    "insufficient",  # wallet has no USDC
    "invalid signature",  # bad signing key
    "invalid_payload",  # structurally malformed payload
    "denied",  # payer denylisted
)


def _is_unrecoverable_payment_error(reason: str) -> bool:
    """True iff retrying with a brand-new payment cannot possibly succeed.

    Used by the whole-request payment retry to decide fail-fast vs retry. Unlike
    :func:`_is_permanent_payment_error` (which classifies re-signing the SAME
    authorization), a fresh nonce/probe/blockhash recovers replay, amount-
    mismatch, expiry and blockhash-window failures, so only truly terminal
    conditions (no funds, bad key, denylisted) short-circuit the retry.
    """
    if not reason:
        return False
    low = reason.lower()
    return any(p in low for p in _UNRECOVERABLE_PAYMENT_PATTERNS)


def _get_user_agent() -> str:
    from . import __version__

    return f"blockrun-python/{__version__}"


DEFAULT_SOLANA_RPC_URL = "https://sol.blockrun.ai/api/v1/solana/rpc"


def _resolve_rpc_config(
    rpc_url: Optional[str],
    rpc_headers: Optional[Dict[str, str]],
) -> Tuple[str, Optional[Dict[str, str]]]:
    """Resolve the effective RPC URL + headers from explicit args, env vars,
    or defaults — in that priority order.

    Since 0.24.0 the default is ``https://sol.blockrun.ai/api/v1/solana/rpc``
    — BlockRun's own multi-region Tatum-backed proxy. Free for anyone
    using the BlockRun SDK; the cost is bundled into LLM inference fees
    you already pay. Method-aware caching on the server side
    (``getLatestBlockhash`` at 30s TTL) collapses bursty signing traffic
    to a handful of upstream RPC calls, so partners no longer need to
    register Helius / Tatum / QuickNode for typical loads. The public
    ``api.mainnet-beta.solana.com`` is still reachable via explicit
    config but is no longer the default — too aggressive a rate limit
    for any real concurrency.

    Env vars (since 0.23.0):
      * ``SOLANA_RPC_URL`` — full RPC URL. Override to point at your
        own Helius / Tatum / QuickNode account, or to bypass the
        BlockRun proxy entirely.
      * ``SOLANA_RPC_API_KEY`` — convenience shortcut for the common
        ``x-api-key: <value>`` header style (Tatum, some QuickNode
        setups). Not needed when using the BlockRun default (the
        proxy handles its own upstream auth server-side).
      * ``SOLANA_RPC_HEADERS`` — JSON dict for arbitrary headers
        (``'{"x-api-key":"...", "x-rate-limit-tier":"pro"}'``).

    Helius style (key embedded in URL) needs ``SOLANA_RPC_URL`` only.
    Tatum style (header auth) needs ``SOLANA_RPC_URL`` + one of
    ``SOLANA_RPC_API_KEY`` / ``SOLANA_RPC_HEADERS``.
    """
    import json as _json

    resolved_url = rpc_url or os.environ.get("SOLANA_RPC_URL") or DEFAULT_SOLANA_RPC_URL

    resolved_headers: Optional[Dict[str, str]] = None
    if rpc_headers is not None:
        resolved_headers = dict(rpc_headers)
    else:
        env_headers_json = os.environ.get("SOLANA_RPC_HEADERS")
        env_api_key = os.environ.get("SOLANA_RPC_API_KEY")
        if env_headers_json:
            try:
                parsed = _json.loads(env_headers_json)
                if isinstance(parsed, dict):
                    resolved_headers = {str(k): str(v) for k, v in parsed.items()}
            except Exception:
                pass
        elif env_api_key:
            resolved_headers = {"x-api-key": env_api_key}

    return resolved_url, resolved_headers


def _register_svm_with_headers(
    x402_client: Any,
    signer: Any,
    rpc_url: str,
    rpc_headers: Optional[Dict[str, str]],
) -> None:
    """Register the SVM exact scheme on an x402 client, with optional
    extra HTTP headers for the underlying Solana RPC.

    The x402 SDK's :func:`register_exact_svm_client` doesn't pass headers
    through to ``solana.rpc.api.Client``, so when the user picks a gateway
    that authenticates by header (Tatum, some Triton setups) we need a
    pre-populated client cache. This function wires that up.

    If ``rpc_headers`` is ``None`` we delegate to the upstream helper so
    behavior is unchanged for users on Helius-URL-style auth.
    """
    if not rpc_headers:
        register_exact_svm_client(x402_client, signer, rpc_url=rpc_url)
        return

    # Header-auth path — pre-build SolanaClients with extra_headers and
    # populate the scheme's _clients cache so it never falls back to the
    # header-less default.
    from solana.rpc.api import Client as SolanaClient
    from x402.mechanisms.svm.exact.client import ExactSvmScheme
    from x402.mechanisms.svm.exact.v1.client import ExactSvmSchemeV1
    from x402.mechanisms.svm.exact.register import V1_NETWORKS

    pre_client = SolanaClient(rpc_url, extra_headers=rpc_headers)

    def _populated(scheme):
        # Hit several common network keys so the lazy _get_client never
        # constructs a header-less SolanaClient.
        for net in ("solana", "solana:mainnet", "solana-mainnet", "solana:devnet"):
            scheme._clients[net] = pre_client
        return scheme

    v2 = _populated(ExactSvmScheme(signer, rpc_url))
    v1 = _populated(ExactSvmSchemeV1(signer, rpc_url))

    x402_client.register("solana:*", v2)
    for network in V1_NETWORKS:
        x402_client.register_v1(network, v1)


def _should_fallback_solana(exc: Exception) -> bool:
    """Whether an exception during Solana streaming is retriable enough to
    warrant trying the next ``fallback_models`` entry. Matches the Base
    :func:`blockrun_llm.client._should_fallback` semantics:

    - Timeouts and network errors → fall back
    - APIError with 5xx-ish status → fall back
    - 4xx and PaymentError → propagate

    Defensive guard for issue #6: even when the exception is a transient
    type (Timeout/Network), if the underlying reason is a permanent
    payment classification (``transaction_simulation_failed``, etc.) we
    do NOT fall back — re-signing a fresh request hits the same wall in
    seconds. The first failure surfaces immediately.
    """
    # PaymentError always carries the gateway reason now (v0.32.0+).
    if isinstance(exc, PaymentError):
        return False
    # Even for "transient" types, sniff the message for a permanent reason.
    if _is_permanent_payment_error(str(exc)):
        return False
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.NetworkError):
        return True
    if isinstance(exc, APIError) and exc.status_code in (502, 503, 504, 522, 524):
        return True
    return False


# Characters safe to interpolate into a single URL path segment. network /
# symbol / market / wallet address all get f-string'd into a paid endpoint
# path; a '/', '..', '?' or '#' would silently re-target the payment-signing
# request. These values often come from LLM output in agent use, so validate
# before building the URL.
_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_path_segment(value: str, field: str) -> str:
    """Return ``value`` if it is a single safe URL path segment, else raise."""
    if not value or not _SAFE_PATH_SEGMENT_RE.match(value):
        raise ValueError(
            f"{field} must contain only letters, digits, '.', '_' or '-' " f"(got {value!r})"
        )
    return value


def _receipt_from_headers(headers: Any) -> Optional[str]:
    """Pull the x402 settlement tx hash from a paid response's headers."""
    if headers is None:
        return None
    return headers.get("x-payment-receipt") or headers.get("X-Payment-Receipt")


def _assert_same_payment_terms(signed_payload: Any, orig_amount: Any, orig_pay_to: Any) -> None:
    """Guard a mid-poll re-sign: the fresh 402 challenge must charge the same
    amount to the same recipient as the payment originally authorized for this
    job. A gateway (buggy or hostile) that reprices or redirects the re-challenge
    would otherwise extract an unbounded, unrelated payment from the wallet.
    Raises :class:`PaymentError` on any mismatch so no signature is submitted."""
    accepted = signed_payload.accepted
    if str(accepted.amount) != str(orig_amount) or accepted.pay_to != orig_pay_to:
        raise PaymentError(
            "Mid-poll re-sign challenge changed the payment terms "
            f"(amount {orig_amount!r} -> {accepted.amount!r}, "
            f"pay_to {orig_pay_to!r} -> {accepted.pay_to!r}); refusing to "
            "authorize a different payment for the same job."
        )


class SolanaLLMClient:
    """
    BlockRun LLM Client for Solana — pays via Solana USDC x402.

    Connects to sol.blockrun.ai by default.
    """

    SOLANA_API_URL = SOLANA_API_URL

    # Image generation slow-path polling. Models like ``openai/gpt-image-2``
    # or ``openai/dall-e-3`` routinely exceed the gateway's 30s inline window
    # and come back as 202 + ``poll_url`` instead of the finished image. The
    # SDK replays the same PAYMENT-SIGNATURE on every poll; settlement only
    # happens on the first completed poll, so a poll-loop timeout = zero
    # spend. Budget is conservative — most upstreams finish in 1-3 min.
    IMAGE_POLL_INTERVAL_SECONDS = 5.0
    IMAGE_POLL_BUDGET_SECONDS = 300.0

    # Video generation slow-path polling. Video always comes back as
    # 202 + ``poll_url`` and can run far past the 600s x402 authorization
    # window, so the poll loop re-signs a fresh PAYMENT-SIGNATURE (same
    # wallet) when a poll 402s mid-flight. Settlement still only happens on
    # the first completed poll, so a poll-loop timeout = zero spend.
    VIDEO_DEFAULT_MODEL = "xai/grok-imagine-video"
    VIDEO_POLL_INTERVAL_SECONDS = 5.0
    VIDEO_POLL_BUDGET_SECONDS = 900.0
    # Matches Base VideoClient.MAX_POLL_RESIGNS (2) — each re-sign is only used
    # to refresh an expired blockhash, and every fresh signature is validated
    # against the original payment terms before use.
    MEDIA_POLL_MAX_RESIGNS = 2

    # Media generation defaults (mirror the Base MusicClient/SpeechClient).
    MUSIC_DEFAULT_MODEL = "minimax/music-2.5+"
    SPEECH_DEFAULT_MODEL = "elevenlabs/flash-v2.5"
    SOUNDFX_DEFAULT_MODEL = "elevenlabs/sound-effects"

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: str = SOLANA_API_URL,
        rpc_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        image_timeout: float = DEFAULT_IMAGE_TIMEOUT,
        search_timeout: float = DEFAULT_SEARCH_TIMEOUT,
        rpc_headers: Optional[Dict[str, str]] = None,
        transaction_log: Union[bool, str, "os.PathLike[str]", None] = None,
    ) -> None:
        """Initialise the Solana client.

        ``timeout`` is the baseline (chat) HTTP timeout; ``image_timeout``
        and ``search_timeout`` tune the slower workloads independently —
        mirroring the per-client tuning the Base SDK gets from having
        separate ``LLMClient`` / ``ImageClient`` classes. Every public
        method also takes a per-call ``timeout=`` override that wins over
        all three. The historical single ``timeout=`` keyword still works
        and now governs chat.

        ``rpc_url`` / ``rpc_headers`` fall back to the env vars
        ``SOLANA_RPC_URL`` / ``SOLANA_RPC_HEADERS`` / ``SOLANA_RPC_API_KEY``
        when not passed explicitly (see :func:`_resolve_rpc_config`).
        Default is the public mainnet-beta RPC if nothing is configured —
        fine for low QPS, will 429 under burst load (~10-40 RPS).

        For production traffic point this at Helius / Tatum / QuickNode /
        Triton. Tatum uses header-auth (``x-api-key``), which the upstream
        x402 SDK doesn't pass through — we handle it here via
        :func:`_register_svm_with_headers`.

        ``transaction_log`` mirrors :class:`LLMClient` — opt-in per-call
        log written to a project folder (default ``./log/``) containing
        the request, response, USD cost, and the on-chain settlement
        signature returned by the facilitator.
        """
        if not _HAS_X402:
            raise ImportError(
                "Solana payment requires the x402 SDK. "
                "Install with: pip install blockrun-llm[solana]"
            )
        from .solana_wallet import load_solana_wallet

        key = (
            private_key
            or os.environ.get("SOLANA_WALLET_KEY")
            or load_solana_wallet()  # disk: newest ~/.*/solana-wallet.json, else ~/.blockrun/.solana-session
        )
        if not key:
            raise ValueError(
                "Private key required. Pass private_key, set SOLANA_WALLET_KEY, "
                "or have a Solana wallet on disk "
                "(~/.<provider>/solana-wallet.json or ~/.blockrun/.solana-session)."
            )
        self._private_key = key
        validate_api_url(api_url)
        self._api_url = api_url.rstrip("/")

        # Resolve effective RPC URL + headers (explicit args > env vars > default).
        resolved_url, resolved_headers = _resolve_rpc_config(rpc_url, rpc_headers)
        self._rpc_url = resolved_url
        self._rpc_headers = resolved_headers

        self._timeout = timeout
        self._image_timeout = image_timeout
        self._search_timeout = search_timeout
        # httpx.Client carries the chat baseline as its default; image /
        # search / per-call overrides are applied per request below.
        self._client = httpx.Client(timeout=timeout)
        self._session_total_usd = 0.0
        self._session_calls = 0
        self._last_call_cost: float = 0.0
        self._address: Optional[str] = None

        log_dir = _resolve_log_dir(transaction_log)
        self._tx_logger: Optional[TransactionLogger] = (
            TransactionLogger(log_dir) if log_dir is not None else None
        )
        self._last_settlement: Optional[Dict[str, Any]] = None
        # Response headers from the most recent raw paid POST — consumed by
        # rpc()/music()/speech() to surface the settlement receipt + gateway
        # metadata the shared JSON-only helper would otherwise drop. Read it
        # immediately after the helper returns (no intervening await).
        self._last_raw_headers: Optional[httpx.Headers] = None

        # Initialize x402 SDK client for Solana payment signing.
        self._x402_client = x402ClientSync()
        try:
            signer = _create_signer(self._private_key)
        except Exception as e:
            # Parity with the Base client, which validates the resolved key up
            # front: turn a malformed key (incl. one auto-loaded from disk) into
            # a clean error instead of a raw base58/solders exception.
            raise ValueError(
                "Invalid Solana private key (expected a base58-encoded keypair " "or 32-byte seed)."
            ) from e
        _register_svm_with_headers(self._x402_client, signer, resolved_url, resolved_headers)
        # x402ClientSync is NOT thread-safe: concurrent payment signing on one
        # shared client races on nonce/authorization state. This lock serializes
        # just the (fast) signing step so a single client can be shared across
        # threads — see _sign_payment.
        self._payment_lock = threading.Lock()

    def _sign_payment(self, payment_required: Any) -> Any:
        """Thread-safe wrapper around ``x402_client.create_payment_payload``.

        Without this, sharing one ``SolanaLLMClient`` across threads (e.g. a
        ThreadPoolExecutor issuing many concurrent paid requests from one wallet)
        produces gateway rejections under load — ``authorization already used``
        (duplicate replay nonce) and ``invalid_exact_svm_payload_amount_mismatch``
        — because the underlying x402 client's nonce/auth state is mutated
        concurrently. Serializing only this brief signing critical section fixes
        it while the upstream streaming continues to run fully concurrently.
        """
        with self._payment_lock:
            return self._x402_client.create_payment_payload(payment_required)

    def _capture_settlement(self, response: httpx.Response) -> Optional[Dict[str, Any]]:
        """Decode the x402 settlement header on a Solana paid response.

        Solana facilitators put the on-chain transaction signature in the
        same ``X-PAYMENT-RESPONSE`` header EVM does — different chain id,
        same wire format. ``None`` when no header is returned.
        """
        header = response.headers.get("x-payment-response") or response.headers.get(
            "X-PAYMENT-RESPONSE"
        )
        settlement = decode_settlement_header(header)
        self._last_settlement = settlement
        return settlement

    def _attach_receipt(self, data: Any) -> None:
        """Inject the settlement tx hash from the most recent paid POST into a
        raw response dict under ``txHash`` (mirrors the Base Music/Speech
        clients). No-op on free responses (no receipt header)."""
        tx_hash = _receipt_from_headers(self._last_raw_headers)
        if tx_hash and isinstance(data, dict) and not data.get("txHash"):
            data["txHash"] = tx_hash

    def get_wallet_address(self) -> str:
        if not self._address:
            self._address = get_solana_public_key(self._private_key)
        return self._address

    def is_solana(self) -> bool:
        return "sol.blockrun.ai" in self._api_url

    def get_balance(self) -> float:
        """Get USDC balance on Solana (matches LLMClient.get_balance() API)."""
        from .solana_wallet import get_solana_usdc_balance

        return get_solana_usdc_balance(self.get_wallet_address(), rpc_url=self._rpc_url)

    def get_spending(self) -> Dict[str, Any]:
        return {"total_usd": self._session_total_usd, "calls": self._session_calls}

    def _billing_meta(self) -> Dict[str, Optional[str]]:
        """Billing metadata for cost-log entries."""
        return {
            "wallet": self.get_wallet_address(),
            "network": "solana-mainnet" if self.is_solana() else "solana-other",
            "client_kind": type(self).__name__,
        }

    def _log_transaction(
        self,
        endpoint: str,
        body: Dict[str, Any],
        response: Any,
        cost_usd: float,
    ) -> None:
        """Append one row to the project-local transaction log when the
        sync Solana client is constructed with ``transaction_log=…``.

        Consumes ``self._last_settlement`` so the on-chain Solana signature
        captured from the paid retry is written exactly once per call."""
        logger = self._tx_logger
        if logger is None:
            return
        settlement = self._last_settlement
        self._last_settlement = None
        try:
            logger.log(
                endpoint=endpoint,
                request=body,
                response=response,
                cost_usd=cost_usd,
                model=(body.get("model") if isinstance(body, dict) else None),
                wallet=self.get_wallet_address(),
                network="solana-mainnet" if self.is_solana() else "solana-other",
                client_kind=type(self).__name__,
                settlement=settlement,
            )
        except Exception:
            pass

    def chat(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        search: bool = False,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Simple 1-line chat."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat_completion(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            search=search,
            timeout=timeout,
            response_format=response_format,
            stop=stop,
        )
        return result.choices[0].message.content or ""

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> ChatResponse:
        """Full chat completion (OpenAI-compatible).

        Supports OpenAI-style function calling via ``tools`` /
        ``tool_choice`` — the BlockRun gateway forwards them to the
        upstream model unchanged (Base and Solana use the same backend
        schema; the only chain difference is the payment leg).

        ``timeout`` overrides the per-call HTTP timeout (defaults to the
        client's chat baseline, ``DEFAULT_CHAT_TIMEOUT``). Raise it for
        large ``max_tokens`` runs against slow models.
        """
        body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop
        return self._request_with_payment("/v1/chat/completions", body, timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def list_models(self) -> List[Dict[str, Any]]:
        resp = self._client.get(f"{self._api_url}/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    @staticmethod
    def _extract_payment_header(response: httpx.Response) -> Optional[str]:
        """Extract x402 payment header from a 402 response (header or body)."""
        payment_header = response.headers.get("payment-required")
        if not payment_header:
            try:
                import base64
                import json

                resp_body = response.json()
                if resp_body.get("accepts") or resp_body.get("x402Version"):
                    payment_header = base64.b64encode(json.dumps(resp_body).encode()).decode()
            except Exception:
                pass
        return payment_header

    # ------------------------------------------------------------------
    # Streaming (SSE) chat completions
    # ------------------------------------------------------------------

    # Retry policy mirrors LLMClient. ``1 + len(_STREAM_5XX_BACKOFFS)`` tries
    # per phase (probe / paid-retry), exponential backoff in seconds.
    _STREAM_5XX_STATUSES = (500, 502, 503, 504)
    _STREAM_5XX_BACKOFFS = (1.0, 2.0, 4.0)

    # Whole-request payment retry: on a NON-permanent payment rejection (concurrent
    # single-wallet replay-nonce / amount mismatch, transient facilitator flake),
    # re-run the ENTIRE paid request — fresh 402 probe + fresh signature (new nonce,
    # correct amount) — but only before the first chunk is yielded. This is what
    # gets concurrent load to ~100% success; the per-call signing lock alone can't
    # recover a transient/amount failure once it has happened.
    _MAX_PAYMENT_RETRIES = 4
    _PAYMENT_RETRY_BACKOFFS = (0.25, 0.5, 1.0, 2.0)

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """
        Stream a chat completion via Server-Sent Events, paid in Solana USDC
        via x402. Mirrors :meth:`LLMClient.chat_completion_stream` semantics:

        - Yields one :class:`ChatCompletionChunk` per ``data:`` line until
          the upstream emits ``data: [DONE]``.
        - Free models stream on the first request; paid models do the
          402 → sign locally with the SVM signer → retry with
          ``PAYMENT-SIGNATURE`` dance before the first chunk.
        - 5xx upstream errors are retried in-band with exponential
          backoff (1s / 2s / 4s).
        - ``fallback_models`` walks the chain on retriable errors, but
          only **before** the first chunk has been yielded (mid-stream
          fallback would concatenate two distinct responses).
        - ``tools`` / ``tool_choice`` work the same as on Base — the
          gateway forwards them to the upstream model regardless of
          chain.

        Note: ``search_parameters`` is rejected by the BlockRun gateway in
        stream mode (HTTP 400). Codex / GPT-5.4-Pro also can't stream.
        """
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop

        attempts = [model, *(fallback_models or [])]
        last_exc: Optional[Exception] = None

        for i, attempt_model in enumerate(attempts):
            body["model"] = attempt_model
            inner = self._stream_with_payment("/v1/chat/completions", body, timeout=timeout)
            chunks_yielded = 0
            try:
                for chunk in inner:
                    chunks_yielded += 1
                    yield chunk
                return  # finished cleanly
            except Exception as exc:
                if chunks_yielded > 0:
                    raise  # mid-stream — can't fall back
                if not _should_fallback_solana(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] solana stream {attempt_model} -> "
                        f"{next_model} ({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        assert last_exc is not None
        raise last_exc

    def _stream_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """Whole-request payment-retry wrapper around :meth:`_stream_once`.

        Re-runs the entire paid request (fresh 402 probe + fresh signature) on a
        non-permanent payment rejection, but only before the first chunk is
        yielded — once the 200 stream starts, :meth:`_stream_once` returns
        without raising, so output is never replayed. See _MAX_PAYMENT_RETRIES.
        """
        import time

        for payment_attempt in range(self._MAX_PAYMENT_RETRIES + 1):
            yielded = 0
            try:
                for chunk in self._stream_once(endpoint, body, timeout=timeout):
                    yielded += 1
                    yield chunk
                return
            except PaymentError as exc:
                if (
                    yielded > 0
                    or _is_unrecoverable_payment_error(str(exc))
                    or payment_attempt >= self._MAX_PAYMENT_RETRIES
                ):
                    raise
                time.sleep(
                    self._PAYMENT_RETRY_BACKOFFS[
                        min(payment_attempt, len(self._PAYMENT_RETRY_BACKOFFS) - 1)
                    ]
                )

    def _stream_once(
        self,
        endpoint: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Iterator[ChatCompletionChunk]:
        """402 → sign (SVM) → retry → SSE iter. Same shape as the Base
        :meth:`LLMClient._stream_with_payment`; differs only in the
        signing path (we go through the x402 SDK's SVM client)."""
        url = f"{self._api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout

        backoffs = self._STREAM_5XX_BACKOFFS

        # ----- Phase 1: probe (no payment header) -----
        payment_headers: Optional[Dict[str, str]] = None
        cost_usd = 0.0

        for attempt in range(len(backoffs) + 1):
            with self._client.stream(
                "POST", url, json=body, headers=req_headers, timeout=eff_timeout
            ) as resp1:
                if resp1.status_code == 200:
                    # Free model — stream directly.
                    yield from self._iter_sse_chunks(resp1)
                    return
                resp1.read()
                if resp1.status_code == 402:
                    payment_headers, cost_usd = self._sign_payment_from_response(resp1)
                    break
                if resp1.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import time

                    time.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp1, after_payment=False)
        else:
            raise APIError("solana stream probe exhausted retries", 0, None)

        # ----- Phase 2: stream with PAYMENT-SIGNATURE -----
        assert payment_headers is not None
        for attempt in range(len(backoffs) + 1):
            with self._client.stream(
                "POST", url, json=body, headers=payment_headers, timeout=eff_timeout
            ) as resp2:
                if resp2.status_code == 200:
                    if cost_usd > 0:
                        self._session_calls += 1
                        self._session_total_usd += cost_usd
                        self._last_call_cost = cost_usd
                        self._capture_settlement(resp2)
                    yield from self._iter_and_archive(resp2, body, cost_usd)
                    return
                resp2.read()
                if resp2.status_code == 402:
                    raise build_payment_rejected_error(resp2)
                if resp2.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import time

                    time.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp2, after_payment=True)

    def _iter_and_archive(
        self,
        response: httpx.Response,
        body: Dict[str, Any],
        cost_usd: float,
    ) -> Iterator[ChatCompletionChunk]:
        """Yield SSE chunks; on stream completion, archive the assembled
        response to ``~/.blockrun/data/`` and append a row to
        ``~/.blockrun/cost_log.jsonl``. Paid streaming calls now show up
        in the same audit trail as non-stream paid calls.

        ``cost_usd == 0`` skips the archive (free models / unauth probe)."""
        assembled_id: Optional[str] = None
        assembled_model: Optional[str] = None
        assembled_created: int = 0
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None

        for chunk in self._iter_sse_chunks(response):
            if chunk.choices:
                choice = chunk.choices[0]
                content = stream_choice_content(choice)
                if content:
                    content_parts.append(content)
                fr = stream_choice_finish_reason(choice)
                if fr:
                    finish_reason = fr
            if assembled_id is None:
                _id, _model, _created = chunk_meta(chunk)
                if _id:
                    assembled_id = _id
                    assembled_model = _model
                    assembled_created = _created
            _usage = chunk_usage_dict(chunk)
            if _usage is not None:
                usage_dict = _usage
            # Race-free per-call x402 charge — see LLMClient._iter_and_archive.
            chunk.cost_usd = cost_usd
            yield chunk

        if cost_usd > 0:
            from .cache import save_to_cache

            response_data: Dict[str, Any] = {
                "id": assembled_id or "stream",
                "object": "chat.completion",
                "created": assembled_created or int(__import__("time").time()),
                "model": assembled_model or body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "".join(content_parts),
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "stream": True,
            }
            if usage_dict:
                response_data["usage"] = usage_dict
            try:
                save_to_cache(
                    "/v1/chat/completions",
                    body,
                    response_data,
                    cost_usd=cost_usd,
                    **self._billing_meta(),
                )
            except Exception:
                pass
            self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)

    @staticmethod
    def _iter_sse_chunks(response: httpx.Response) -> Iterator[ChatCompletionChunk]:
        """OpenAI-format SSE parser. ``data: <json>\\n\\n`` lines, terminated
        by ``data: [DONE]``. Malformed chunks are skipped, not raised."""
        for raw_line in response.iter_lines():
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[6:].strip()
            if payload == "[DONE]":
                return
            try:
                chunk_dict = _json.loads(payload)
            except Exception:
                continue
            try:
                yield ChatCompletionChunk(**chunk_dict)
            except Exception:
                yield ChatCompletionChunk.model_construct(**chunk_dict)

    def _sign_payment_from_response(
        self,
        response: httpx.Response,
    ) -> Tuple[Dict[str, str], float]:
        """Extract a 402 response's payment requirements, sign locally with
        the SVM x402 client, return ``(headers_with_PAYMENT_SIGNATURE,
        cost_usd)``. Mirrors the inline logic in
        :meth:`_handle_payment_and_retry` but returns headers instead of
        making the retry POST itself — lets the streaming path open an
        SSE connection for the retry."""
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._sign_payment(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        cost_usd = float(payment_payload.accepted.amount) / 1e6

        return (
            {
                "Content-Type": "application/json",
                "User-Agent": _get_user_agent(),
                "PAYMENT-SIGNATURE": encoded_payment,
            },
            cost_usd,
        )

    @staticmethod
    def _raise_stream_error(response: httpx.Response, *, after_payment: bool) -> None:
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": "Stream request failed"}
        prefix = "API error after payment" if after_payment else "API error"
        raise APIError(
            f"{prefix}: {response.status_code}",
            response.status_code,
            sanitize_error_response(error_body),
        )

    def _request_with_payment(
        self, endpoint: str, body: Dict[str, Any], timeout: Optional[float] = None
    ) -> ChatResponse:
        """Whole-request payment-retry wrapper around :meth:`_request_once`.

        Re-runs the entire paid request (fresh 402 probe + fresh signature) on a
        recoverable payment rejection — concurrent replay-nonce / amount mismatch
        / transient facilitator flake — so a shared client under concurrent load
        reaches ~100%. See _MAX_PAYMENT_RETRIES.
        """
        import time

        for payment_attempt in range(self._MAX_PAYMENT_RETRIES + 1):
            try:
                return self._request_once(endpoint, body, timeout=timeout)
            except PaymentError as exc:
                if (
                    _is_unrecoverable_payment_error(str(exc))
                    or payment_attempt >= self._MAX_PAYMENT_RETRIES
                ):
                    raise
                time.sleep(
                    self._PAYMENT_RETRY_BACKOFFS[
                        min(payment_attempt, len(self._PAYMENT_RETRY_BACKOFFS) - 1)
                    ]
                )

    def _request_once(
        self, endpoint: str, body: Dict[str, Any], timeout: Optional[float] = None
    ) -> ChatResponse:
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout

        response = self._client.post(url, json=body, headers=headers, timeout=eff_timeout)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.post(url, json=body, headers=headers, timeout=eff_timeout)

        if response.status_code == 402:
            return self._handle_payment_and_retry(url, body, response, timeout=eff_timeout)

        if not response.is_success:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return ChatResponse(**response.json())

    def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
        timeout: Optional[float] = None,
    ) -> ChatResponse:
        eff_timeout = timeout if timeout is not None else self._timeout
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        # Use x402 SDK to decode 402 response and create signed payment
        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._sign_payment(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = self._client.post(
            url, json=body, headers=payment_headers, timeout=eff_timeout
        )
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.post(
                url, json=body, headers=payment_headers, timeout=eff_timeout
            )

        if retry_response.status_code == 402:
            raise build_payment_rejected_error(retry_response)

        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        cost_usd = float(payment_payload.accepted.amount) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd
        self._capture_settlement(retry_response)

        # Save full response locally
        response_data = retry_response.json()
        from .cache import save_to_cache

        save_to_cache(
            "/v1/chat/completions",
            body,
            response_data,
            cost_usd=cost_usd,
            **self._billing_meta(),
        )
        self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)

        return ChatResponse(**response_data)

    def _request_with_payment_raw(
        self, endpoint: str, body: Dict[str, Any], timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make a request with Solana x402 payment, returning raw JSON."""
        from .cache import get_cached, save_to_cache

        # Check cache first — don't pay twice for same data
        cached = get_cached(endpoint, body)
        if cached is not None:
            return cached

        # Reset per-call receipt headers; only a paid retry repopulates them, so
        # a free/cached model can't inherit a prior call's settlement receipt.
        self._last_raw_headers = None

        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout

        response = self._client.post(url, json=body, headers=headers, timeout=eff_timeout)

        # Auto-retry on transient server errors
        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.post(url, json=body, headers=headers, timeout=eff_timeout)

        if response.status_code == 402:
            result = self._handle_payment_and_retry_raw(url, body, response, timeout=eff_timeout)
            save_to_cache(
                endpoint,
                body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            self._log_transaction(endpoint, body, result, self._last_call_cost)
            return result

        if not response.is_success:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return response.json()

    def _handle_payment_and_retry_raw(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Handle 402 for raw endpoints with Solana payment."""
        eff_timeout = timeout if timeout is not None else self._timeout
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        # Use x402 SDK to decode 402 response and create signed payment
        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._sign_payment(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        payment_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        # Retry with payment, with one automatic retry on 502/503
        retry_response = self._client.post(
            url, json=body, headers=payment_headers, timeout=eff_timeout
        )
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.post(
                url, json=body, headers=payment_headers, timeout=eff_timeout
            )

        if retry_response.status_code == 402:
            raise build_payment_rejected_error(retry_response)

        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        cost_usd = float(payment_payload.accepted.amount) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd
        self._capture_settlement(retry_response)
        self._last_raw_headers = retry_response.headers

        return retry_response.json()

    def _get_with_payment_raw(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """GET with Solana x402 payment, returning raw JSON."""
        from .cache import get_cached, save_to_cache

        cache_key_body = params or {}
        cached = get_cached(endpoint, cache_key_body)
        if cached is not None:
            return cached

        url = f"{self._api_url}{endpoint}"
        headers = {"User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout

        response = self._client.get(url, params=params, headers=headers, timeout=eff_timeout)

        if response.status_code in (502, 503):
            import time

            time.sleep(1)
            response = self._client.get(url, params=params, headers=headers, timeout=eff_timeout)

        if response.status_code == 402:
            result = self._handle_get_payment_and_retry(url, params, response, timeout=eff_timeout)
            save_to_cache(
                endpoint,
                cache_key_body,
                result,
                cost_usd=self._last_call_cost,
                **self._billing_meta(),
            )
            self._log_transaction(endpoint, cache_key_body, result, self._last_call_cost)
            return result

        if not response.is_success:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )

        return response.json()

    def _handle_get_payment_and_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]],
        response: httpx.Response,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Handle 402 for GET endpoints with Solana payment."""
        eff_timeout = timeout if timeout is not None else self._timeout
        payment_header = self._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = decode_payment_required_header(payment_header)
        payment_payload = self._sign_payment(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)

        payment_headers = {
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        retry_response = self._client.get(
            url, params=params, headers=payment_headers, timeout=eff_timeout
        )
        if retry_response.status_code in (502, 503):
            import time

            time.sleep(1)
            retry_response = self._client.get(
                url, params=params, headers=payment_headers, timeout=eff_timeout
            )

        if retry_response.status_code == 402:
            raise build_payment_rejected_error(retry_response)

        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        cost_usd = float(payment_payload.accepted.amount) / 1e6
        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd
        self._capture_settlement(retry_response)
        self._last_raw_headers = retry_response.headers

        return retry_response.json()

    def _absolute_url(self, url: str) -> str:
        """Resolve a server-supplied relative ``poll_url`` against the API host.

        Poll URLs come back as ``/api/v1/images/generations/<id>``; our
        configured ``api_url`` already includes the trailing ``/api`` so
        we strip it once to avoid ``/api/api/...``.
        """
        if url.startswith("http://") or url.startswith("https://"):
            return url
        base = self._api_url[: -len("/api")] if self._api_url.endswith("/api") else self._api_url
        return f"{base}{url}"

    def _request_image_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
        *,
        poll_budget_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
        max_resigns: int = 0,
        label: str = "Image",
    ) -> Dict[str, Any]:
        """Sign + submit + poll wrapper for async media generation.

        Shared by :meth:`image` (5-min budget, no mid-poll re-signing needed)
        and :meth:`video` (15-min budget, ``max_resigns`` re-signs to survive
        the 600s x402 authorization window). ``poll_budget_seconds`` /
        ``poll_interval_seconds`` default to the image constants; ``label``
        only tunes error text.

        Why this exists instead of reusing ``_request_with_payment_raw``:
        the gateway falls back to an async ``202 + poll_url`` flow when a
        model exceeds the 30s inline window (gpt-image-2, dall-e-3, slow
        nano-banana-pro 4K, etc.). The raw helper treats 202 as success and
        feeds the job-stub JSON to ``ImageResponse(**data)``, which then
        raises a Pydantic validation error because the ``data`` field
        isn't populated until the upstream finishes.

        Flow:

        1. Probe POST → expect 402 (payment required) from the gateway.
        2. Sign the x402 SVM payload locally; resubmit with PAYMENT-SIGNATURE.
        3. Fast path: 200 with the finished image → settle inline.
        4. Slow path: 202 with ``{id, poll_url, status: queued}`` → loop
           GET poll_url with the PAYMENT-SIGNATURE until status = ``completed``.
           If a poll 402s (settlement failed, e.g. stale blockhash), re-GET
           poll_url for a fresh challenge and re-sign (up to ``max_resigns``).
           Settlement happens on the first completed poll; giving up before
           then costs the caller nothing.

        Returns the raw response JSON from the final completed response.
        """
        import time as _time

        from .cache import get_cached, save_to_cache

        cached = get_cached(endpoint, body)
        if cached is not None:
            return cached

        url = f"{self._api_url}{endpoint}"
        probe_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._image_timeout

        # Step 1: probe — expect 402 unless the model is free or cached upstream.
        probe = self._client.post(url, json=body, headers=probe_headers, timeout=eff_timeout)
        if probe.status_code in (502, 503):
            _time.sleep(1)
            probe = self._client.post(url, json=body, headers=probe_headers, timeout=eff_timeout)

        if probe.status_code != 402:
            if not probe.is_success:
                try:
                    error_body = probe.json()
                except Exception:
                    error_body = {"error": "Request failed"}
                raise APIError(
                    f"Image request: HTTP {probe.status_code}",
                    probe.status_code,
                    sanitize_error_response(error_body),
                )
            # Free / cached upstream — return whatever the gateway gave us.
            return probe.json()

        # Step 2: sign x402 SVM payload.
        payment_header_str = self._extract_payment_header(probe)
        if not payment_header_str:
            raise PaymentError("402 response but no payment requirements found")

        payment_required = decode_payment_required_header(payment_header_str)
        payment_payload_obj = self._sign_payment(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload_obj)
        cost_usd = float(payment_payload_obj.accepted.amount) / 1e6
        # Terms this job is authorized to pay — any mid-poll re-sign must match.
        orig_amount = payment_payload_obj.accepted.amount
        orig_pay_to = payment_payload_obj.accepted.pay_to

        paid_headers = {
            "Content-Type": "application/json",
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        # Step 3: submit with signature.
        submit_resp = self._client.post(url, json=body, headers=paid_headers, timeout=eff_timeout)
        if submit_resp.status_code in (502, 503):
            _time.sleep(1)
            submit_resp = self._client.post(
                url, json=body, headers=paid_headers, timeout=eff_timeout
            )

        if submit_resp.status_code == 402:
            raise build_payment_rejected_error(submit_resp)

        if submit_resp.status_code == 200:
            # Fast path — image was produced inline.
            self._session_calls += 1
            self._session_total_usd += cost_usd
            self._last_call_cost = cost_usd
            self._capture_settlement(submit_resp)
            data = submit_resp.json()
            save_to_cache(endpoint, body, data, cost_usd=cost_usd, **self._billing_meta())
            self._log_transaction(endpoint, body, data, cost_usd)
            return data

        if submit_resp.status_code != 202:
            try:
                error_body = submit_resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"Image request after payment: HTTP {submit_resp.status_code}",
                submit_resp.status_code,
                sanitize_error_response(error_body),
            )

        # Step 4: slow path — poll until completed (or budget exhausted).
        try:
            submit_data = submit_resp.json()
        except Exception:
            submit_data = {}

        poll_url_rel = submit_data.get("poll_url")
        job_id = submit_data.get("id")
        if not poll_url_rel:
            raise APIError(
                "Slow-path 202 missing poll_url",
                202,
                {"response": submit_data},
            )
        poll_url = self._absolute_url(poll_url_rel)
        poll_headers = {
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        budget = (
            poll_budget_seconds
            if poll_budget_seconds is not None
            else self.IMAGE_POLL_BUDGET_SECONDS
        )
        interval = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self.IMAGE_POLL_INTERVAL_SECONDS
        )
        deadline = _time.monotonic() + budget
        last_status = submit_data.get("status", "queued")
        resigns_left = max_resigns

        while _time.monotonic() < deadline:
            _time.sleep(interval)

            poll_resp = self._client.get(poll_url, headers=poll_headers, timeout=eff_timeout)
            try:
                poll_data = poll_resp.json()
            except Exception:
                poll_data = {}
            last_status = poll_data.get("status", last_status)

            if poll_resp.status_code == 402:
                # Mid-poll 402 = settlement of the signed payment failed. For
                # long jobs this is almost always a stale blockhash: the payment
                # was signed at submit time, but the on-chain settlement only
                # runs once the job completes, and by then the signed
                # transaction's recent-blockhash can be expired — the facilitator
                # reports ``transaction_simulation_failed``. The failing poll
                # response carries NO fresh challenge, so re-GET poll_url WITHOUT
                # the stale signature to solicit a fresh 402 (new blockhash),
                # re-sign, and keep polling. Mirrors the Base VideoClient. A
                # fresh signature that 402s again is a genuine payment problem.
                if resigns_left > 0:
                    resigns_left -= 1
                    resign_payload = None
                    try:
                        challenge = self._client.get(
                            poll_url,
                            headers={"User-Agent": _get_user_agent()},
                            timeout=eff_timeout,
                        )
                        resign_header = self._extract_payment_header(challenge)
                        if challenge.status_code == 402 and resign_header:
                            resign_required = decode_payment_required_header(resign_header)
                            resign_payload = self._sign_payment(resign_required)
                    except (PaymentError, httpx.HTTPError):
                        # Challenge GET failed, or signing was rejected — fall
                        # through to surface the gateway's real 402 reason rather
                        # than masking it with a network/signing error. Nothing
                        # settled here.
                        resign_payload = None
                    if resign_payload is not None:
                        # Refuse a re-challenge that reprices or redirects the
                        # payment vs. what this job originally authorized. This
                        # PaymentError must propagate (NOT fall through to the
                        # generic 402). The guard also pins the amount, so the
                        # submit-time cost_usd stays correct for the ledger.
                        _assert_same_payment_terms(resign_payload, orig_amount, orig_pay_to)
                        poll_headers["PAYMENT-SIGNATURE"] = encode_payment_signature_header(
                            resign_payload
                        )
                        continue
                raise build_payment_rejected_error(poll_resp)

            if last_status == "failed":
                raise APIError(
                    f"{label} failed upstream: {poll_data.get('error', 'unknown')}",
                    poll_resp.status_code,
                    sanitize_error_response(poll_data if isinstance(poll_data, dict) else {}),
                )

            # Terminal success is keyed on status, NOT the HTTP code — the
            # gateway settles the moment a poll reports completed, so a
            # completed-but-non-200 poll (which the caller was already charged
            # for) must still be treated as success.
            if last_status == "completed":
                tx_hash = poll_resp.headers.get("x-payment-receipt") or poll_resp.headers.get(
                    "X-Payment-Receipt"
                )
                if tx_hash and isinstance(poll_data, dict) and not poll_data.get("txHash"):
                    poll_data["txHash"] = tx_hash
                self._session_calls += 1
                self._session_total_usd += cost_usd
                self._last_call_cost = cost_usd
                self._capture_settlement(poll_resp)
                save_to_cache(endpoint, body, poll_data, cost_usd=cost_usd, **self._billing_meta())
                self._log_transaction(endpoint, body, poll_data, cost_usd)
                return poll_data

            if poll_resp.status_code in (202, 504):
                # 202 = still queued/in_progress; 504 = transient upstream
                # hiccup. Both are retriable inside the budget.
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
                )

        raise APIError(
            (
                f"{label} did not complete within {budget:.0f}s "
                f"(last status: {last_status}). Settlement only happens on "
                "completion, so no payment was taken. The job stays claimable "
                "for ~48h — re-poll poll_url with a fresh signature from the "
                "same wallet to fetch (and settle) the finished result."
            ),
            504,
            {"id": job_id, "last_status": last_status, "poll_url": poll_url},
        )

    def image(
        self,
        prompt: str,
        *,
        model: str = "google/nano-banana",
        size: str = "1024x1024",
        n: int = 1,
        timeout: Optional[float] = None,
    ) -> ImageResponse:
        """Generate an image from a text prompt (Solana payment).

        Supports the same model catalog as ``ImageClient.generate`` on Base:
        ``google/nano-banana``, ``google/nano-banana-pro``,
        ``openai/dall-e-3``, ``openai/gpt-image-1``, ``openai/gpt-image-2``,
        ``zai/cogview-4``, ``xai/grok-imagine-image``,
        ``xai/grok-imagine-image-pro``, ``black-forest/flux-1.1-pro``.

        Slow models (gpt-image-2, dall-e-3) trigger the gateway's async
        202 + poll flow; the client polls transparently until completion
        and only settles on the final completed poll. If the poll budget
        (``IMAGE_POLL_BUDGET_SECONDS``, 5 min) is exhausted, an
        :class:`APIError` 504 is raised and **no payment is taken**.
        """
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }
        data = self._request_image_with_payment("/v1/images/generations", body, timeout=timeout)
        return ImageResponse(**data)

    def image_edit(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        model: str = "openai/gpt-image-2",
        mask: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
        timeout: Optional[float] = None,
    ) -> ImageResponse:
        """Edit an image using img2img (Solana payment). ``image`` may be a
        single data URI or a list of 1-4 data URIs for multi-image fusion
        (openai/* up to 4, google/* up to 3).

        Like :meth:`image`, this handles the gateway's async 202 + poll
        slow path transparently — settlement only happens on completion.
        """
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image": image,
            "size": size,
            "n": n,
        }
        if mask is not None:
            body["mask"] = mask

        data = self._request_image_with_payment("/v1/images/image2image", body, timeout=timeout)
        return ImageResponse(**data)

    # ------------------------------------------------------------------
    # Video generation (Solana payment) — async 202 + poll, mid-poll re-sign
    # ------------------------------------------------------------------

    def video(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        real_face_asset_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        generate_audio: Optional[bool] = None,
        seed: Optional[int] = None,
        watermark: Optional[bool] = None,
        return_last_frame: Optional[bool] = None,
        budget_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> VideoResponse:
        """Generate a video clip from a text prompt (Solana payment).

        Mirrors ``VideoClient.generate`` on Base: submits an async job and
        polls until the clip is ready (typical 60-180s). Settlement only
        happens on the first completed poll, so a poll-budget timeout takes
        **no payment** and leaves the job claimable ~48h. Default model is
        ``xai/grok-imagine-video``.
        """
        body = self._build_video_body(
            prompt,
            model=model,
            image_url=image_url,
            last_frame_url=last_frame_url,
            reference_image_urls=reference_image_urls,
            real_face_asset_id=real_face_asset_id,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            generate_audio=generate_audio,
            seed=seed,
            watermark=watermark,
            return_last_frame=return_last_frame,
        )

        data = self._request_image_with_payment(
            "/v1/videos/generations",
            body,
            timeout=timeout,
            poll_budget_seconds=(
                budget_seconds if budget_seconds is not None else self.VIDEO_POLL_BUDGET_SECONDS
            ),
            poll_interval_seconds=self.VIDEO_POLL_INTERVAL_SECONDS,
            max_resigns=self.MEDIA_POLL_MAX_RESIGNS,
            label="Video generation",
        )
        return VideoResponse(**data)

    def video_from_content(
        self,
        content: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        budget_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
        **options: Any,
    ) -> VideoResponse:
        """Generate a video from a Seedance ``content[]`` body (Solana payment).

        Targets ``POST /v1/videos`` (the multimodal ``content`` array shape).
        Prefer :meth:`video` for structured kwargs; this exists for migrating
        existing ``content[]`` payloads unchanged.
        """
        if not content:
            raise ValueError("content must be a non-empty list of Seedance content items.")
        body: Dict[str, Any] = {"content": content, **options}
        if model is not None:
            body["model"] = model
        data = self._request_image_with_payment(
            "/v1/videos",
            body,
            timeout=timeout,
            poll_budget_seconds=(
                budget_seconds if budget_seconds is not None else self.VIDEO_POLL_BUDGET_SECONDS
            ),
            poll_interval_seconds=self.VIDEO_POLL_INTERVAL_SECONDS,
            max_resigns=self.MEDIA_POLL_MAX_RESIGNS,
            label="Video generation",
        )
        return VideoResponse(**data)

    # ------------------------------------------------------------------
    # Music generation (Solana payment)
    # ------------------------------------------------------------------

    def music(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        instrumental: bool = True,
        lyrics: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> MusicResponse:
        """Generate a music track from a text prompt (Solana payment).

        Mirrors ``MusicClient.generate`` on Base. Takes 1-3 minutes; the
        returned CDN URL is valid ~24h. Default model ``minimax/music-2.5+``.
        """
        if instrumental and lyrics and lyrics.strip():
            raise ValueError("Cannot specify lyrics when instrumental is True")
        body: Dict[str, Any] = {
            "model": model or self.MUSIC_DEFAULT_MODEL,
            "prompt": prompt,
            "instrumental": instrumental,
        }
        if lyrics and lyrics.strip():
            body["lyrics"] = lyrics.strip()
        data = self._request_with_payment_raw("/v1/audio/generations", body, timeout=timeout)
        self._attach_receipt(data)
        return MusicResponse(**data)

    # ------------------------------------------------------------------
    # Speech / TTS + sound effects (Solana payment)
    # ------------------------------------------------------------------

    def speech(
        self,
        input: str,
        *,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> SpeechResponse:
        """Synthesize speech from text (Solana payment).

        Mirrors ``SpeechClient.generate`` on Base. Synchronous; price scales
        with character count. Default model ``elevenlabs/flash-v2.5``, default
        voice ``sarah``.
        """
        body: Dict[str, Any] = {
            "model": model or self.SPEECH_DEFAULT_MODEL,
            "input": input,
        }
        if voice:
            body["voice"] = voice
        if response_format:
            body["response_format"] = response_format
        if speed is not None:
            body["speed"] = speed
        data = self._request_with_payment_raw("/v1/audio/speech", body, timeout=timeout)
        self._attach_receipt(data)
        return SpeechResponse(**data)

    def sound_effect(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        prompt_influence: Optional[float] = None,
        response_format: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SpeechResponse:
        """Generate a cinematic sound effect from a text prompt (Solana
        payment). Mirrors ``SpeechClient.sound_effect``. Flat $0.05, <=22s."""
        body: Dict[str, Any] = {
            "model": model or self.SOUNDFX_DEFAULT_MODEL,
            "text": text,
        }
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        if prompt_influence is not None:
            body["prompt_influence"] = prompt_influence
        if response_format:
            body["response_format"] = response_format
        data = self._request_with_payment_raw("/v1/audio/sound-effects", body, timeout=timeout)
        self._attach_receipt(data)
        return SpeechResponse(**data)

    def list_voices(self) -> List[Dict[str, Any]]:
        """List available speech voices (free)."""
        url = f"{self._api_url}/v1/audio/voices"
        resp = self._client.get(
            url, headers={"User-Agent": _get_user_agent()}, timeout=DEFAULT_FAST_TIMEOUT
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"List voices failed: HTTP {resp.status_code}",
                resp.status_code,
                sanitize_error_response(error_body),
            )
        data = resp.json()
        # Gateway wraps the voice list under "data" (mirrors SpeechClient.list_voices).
        return data.get("data", []) if isinstance(data, dict) else data

    # ------------------------------------------------------------------
    # Virtual Portrait enrollment (Solana payment)
    # ------------------------------------------------------------------

    def portrait_enroll(self, name: str, image_url: str) -> PortraitEnrollment:
        """Enroll a Virtual Portrait ($0.01 USDC, one-time). Returns the
        ``ta_xxxxxxxx`` asset id usable as ``real_face_asset_id`` in
        :meth:`video`. Mirrors ``PortraitClient.enroll``."""
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > 64:
            raise ValueError(f"name must be 64 chars or fewer (got {len(name)})")
        if not image_url or not image_url.lower().startswith(("https://", "http://")):
            raise ValueError("image_url must be an http(s) URL")
        body: Dict[str, Any] = {"name": name, "image_url": image_url}
        data = self._request_with_payment_raw("/v1/portrait/enroll", body)
        return PortraitEnrollment(**data)

    def list_portraits(self, wallet_address: Optional[str] = None) -> PortraitList:
        """List Virtual Portraits enrolled by a wallet (free, rate-limited)."""
        addr = _safe_path_segment(wallet_address or self.get_wallet_address(), "wallet_address")
        url = f"{self._api_url}/v1/wallet/{addr}/portraits"
        resp = self._client.get(
            url, headers={"User-Agent": _get_user_agent()}, timeout=DEFAULT_FAST_TIMEOUT
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "Portrait listing failed",
                resp.status_code,
                sanitize_error_response(error_body),
            )
        return PortraitList(**resp.json())

    # ------------------------------------------------------------------
    # RealFace enrollment (Solana payment)
    # ------------------------------------------------------------------

    def realface_init(self, name: str, group_id: Optional[str] = None) -> RealFaceInit:
        """Start/refresh a RealFace enrollment (free, rate-limited). Returns
        the ``group_id`` and an ``h5_link`` (render as a QR for the real
        person's phone liveness check)."""
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > 64:
            raise ValueError(f"name must be 64 chars or fewer (got {len(name)})")
        if group_id is not None and not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")
        body: Dict[str, Any] = {"name": name}
        if group_id:
            body["groupId"] = group_id
        url = f"{self._api_url}/v1/realface/init"
        resp = self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", "User-Agent": _get_user_agent()},
            timeout=DEFAULT_FAST_TIMEOUT,
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "RealFace init failed", resp.status_code, sanitize_error_response(error_body)
            )
        return RealFaceInit(**resp.json())

    def realface_status(self, group_id: str) -> RealFaceStatus:
        """Poll a RealFace group's state (free, rate-limited)."""
        if not group_id or not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")
        url = f"{self._api_url}/v1/realface/status"
        resp = self._client.get(
            url,
            params={"groupId": group_id},
            headers={"User-Agent": _get_user_agent()},
            timeout=DEFAULT_FAST_TIMEOUT,
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "RealFace status check failed",
                resp.status_code,
                sanitize_error_response(error_body),
            )
        return RealFaceStatus(**resp.json())

    def realface_wait_for_active(
        self,
        group_id: str,
        timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 4.0,
    ) -> RealFaceStatus:
        """Block until the RealFace group is active (person finished the phone
        liveness check). Convenience wrapper around :meth:`realface_status`."""
        import time as _time

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        deadline = _time.monotonic() + timeout_seconds
        while True:
            state = self.realface_status(group_id)
            if state.ready_to_finalize:
                return state
            if _time.monotonic() + poll_interval_seconds >= deadline:
                raise TimeoutError(
                    f"RealFace group {group_id} not active after {timeout_seconds:.0f}s "
                    f"(last status: {state.status!r})."
                )
            _time.sleep(poll_interval_seconds)

    def realface_enroll(self, name: str, image_url: str, group_id: str) -> RealFaceEnrollment:
        """Finalize a RealFace enrollment ($0.01 USDC). Requires the group to
        be active (see :meth:`realface_wait_for_active`)."""
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > 64:
            raise ValueError(f"name must be 64 chars or fewer (got {len(name)})")
        if not image_url or not image_url.lower().startswith(("https://", "http://")):
            raise ValueError("image_url must be an http(s) URL")
        if not group_id or not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")
        body: Dict[str, Any] = {"name": name, "image_url": image_url, "group_id": group_id}
        data = self._request_with_payment_raw("/v1/realface/enroll", body)
        return RealFaceEnrollment(**data)

    def list_realfaces(self, wallet_address: Optional[str] = None) -> RealFaceList:
        """List RealFace assets enrolled by a wallet (free, rate-limited)."""
        addr = _safe_path_segment(wallet_address or self.get_wallet_address(), "wallet_address")
        url = f"{self._api_url}/v1/wallet/{addr}/realfaces"
        resp = self._client.get(
            url, headers={"User-Agent": _get_user_agent()}, timeout=DEFAULT_FAST_TIMEOUT
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "RealFace listing failed",
                resp.status_code,
                sanitize_error_response(error_body),
            )
        return RealFaceList(**resp.json())

    # ------------------------------------------------------------------
    # Pyth market data (Solana payment for paid categories)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_video_body(
        prompt: str,
        *,
        model: Optional[str],
        image_url: Optional[str],
        last_frame_url: Optional[str],
        reference_image_urls: Optional[List[str]],
        real_face_asset_id: Optional[str],
        duration_seconds: Optional[int],
        aspect_ratio: Optional[str],
        resolution: Optional[str],
        generate_audio: Optional[bool],
        seed: Optional[int],
        watermark: Optional[bool],
        return_last_frame: Optional[bool],
    ) -> Dict[str, Any]:
        """Validate video kwargs and build the request body. Shared by the sync
        and async ``video()`` so their validation and payload never drift."""
        if image_url and real_face_asset_id:
            raise ValueError(
                "image_url and real_face_asset_id are mutually exclusive; pass at most one."
            )
        if last_frame_url and not image_url:
            raise ValueError(
                "last_frame_url requires image_url: image_url seeds the FIRST frame and "
                "last_frame_url the FINAL frame — send both."
            )
        if last_frame_url and real_face_asset_id:
            raise ValueError(
                "last_frame_url and real_face_asset_id are mutually exclusive; "
                "first-and-last-frame uses image_url + last_frame_url."
            )
        if reference_image_urls:
            if image_url or last_frame_url or real_face_asset_id:
                raise ValueError(
                    "reference_image_urls is mutually exclusive with image_url, "
                    "last_frame_url, and real_face_asset_id."
                )
            if len(reference_image_urls) > 9:
                raise ValueError("reference_image_urls accepts at most 9 images.")
        if real_face_asset_id is not None and not real_face_asset_id.startswith("ta_"):
            raise ValueError(
                "real_face_asset_id must start with 'ta_' "
                "(a Virtual Portrait or RealFace asset id, e.g. 'ta_abc123xyz')"
            )

        body: Dict[str, Any] = {
            "model": model or SolanaLLMClient.VIDEO_DEFAULT_MODEL,
            "prompt": prompt,
        }
        if image_url:
            body["image_url"] = image_url
        if last_frame_url:
            body["last_frame_url"] = last_frame_url
        if reference_image_urls:
            body["reference_image_urls"] = reference_image_urls
        if real_face_asset_id:
            body["real_face_asset_id"] = real_face_asset_id
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        if aspect_ratio is not None:
            body["aspect_ratio"] = aspect_ratio
        if resolution is not None:
            body["resolution"] = resolution
        if generate_audio is not None:
            body["generate_audio"] = generate_audio
        if seed is not None:
            body["seed"] = seed
        if watermark is not None:
            body["watermark"] = watermark
        if return_last_frame is not None:
            body["return_last_frame"] = return_last_frame
        return body

    @staticmethod
    def _rpc_response(
        data: Any, headers: Optional[httpx.Headers], fallback_network: str
    ) -> RpcResponse:
        """Build an RpcResponse, surfacing gateway metadata from the paid
        response headers (canonical network, cache hit, settlement tx) exactly
        like the Base RPCClient. Strips body keys that would collide with those
        metadata kwargs."""
        if not isinstance(data, dict):
            data = {"result": data}
        else:
            data = {k: v for k, v in data.items() if k not in ("network", "cache_hit", "tx_hash")}
        hdrs = headers if headers is not None else httpx.Headers()
        return RpcResponse(
            **data,
            network=hdrs.get("x-network") or fallback_network,
            cache_hit=(hdrs.get("x-cache", "") or "").upper() == "HIT",
            tx_hash=_receipt_from_headers(hdrs),
        )

    @staticmethod
    def _price_category_path(
        category: str, market: Optional[str], kind: str, symbol: Optional[str]
    ) -> str:
        if category == "stocks":
            if not market:
                raise ValueError("market is required for category='stocks' (e.g. market='us')")
            base = f"/v1/stocks/{_safe_path_segment(market, 'market')}"
        elif category in ("crypto", "fx", "commodity", "usstock"):
            base = f"/v1/{category}"
        else:
            raise ValueError(f"Unknown category: {category}")
        if symbol is None:
            return f"{base}/{kind}"
        return f"{base}/{kind}/{_safe_path_segment(symbol.upper(), 'symbol')}"

    def price(
        self,
        category: Category,
        symbol: str,
        *,
        market: Optional[Market] = None,
        session: Optional[Session] = None,
    ) -> PricePoint:
        """Fetch a realtime Pyth price quote (Solana payment for paid
        categories). ``market`` is required for ``category='stocks'``."""
        endpoint = self._price_category_path(category, market, "price", symbol)
        params: Dict[str, Any] = {}
        if session is not None:
            params["session"] = session
        data = self._get_with_payment_raw(
            endpoint, params=params or None, timeout=DEFAULT_FAST_TIMEOUT
        )
        return PricePoint(
            symbol=data.get("symbol", symbol.upper()),
            price=data.get("price"),
            publish_time=data.get("publishTime"),
            confidence=data.get("confidence"),
            feed_id=data.get("feedId"),
            **{
                k: v
                for k, v in data.items()
                if k not in {"symbol", "price", "publishTime", "confidence", "feedId"}
            },
        )

    def price_history(
        self,
        category: Category,
        symbol: str,
        *,
        resolution: Resolution = "D",
        from_ts: int,
        to_ts: int,
        market: Optional[Market] = None,
        session: Optional[Session] = None,
    ) -> PriceHistoryResponse:
        """Fetch OHLC bars between two Unix timestamps (seconds)."""
        endpoint = self._price_category_path(category, market, "history", symbol)
        params: Dict[str, Any] = {"resolution": resolution, "from": from_ts, "to": to_ts}
        if session is not None:
            params["session"] = session
        data = self._get_with_payment_raw(endpoint, params=params, timeout=DEFAULT_FAST_TIMEOUT)
        return PriceHistoryResponse(
            symbol=data.get("symbol", symbol.upper()),
            resolution=data.get("resolution", resolution),
            bars=data.get("bars", []),
            **{k: v for k, v in data.items() if k not in {"symbol", "resolution", "bars"}},
        )

    def list_symbols(
        self,
        category: Category,
        *,
        q: Optional[str] = None,
        limit: int = 100,
        market: Optional[Market] = None,
    ) -> SymbolListResponse:
        """List available symbols in a Pyth category (free discovery)."""
        endpoint = self._price_category_path(category, market, "list", None)
        params: Dict[str, Any] = {"limit": limit}
        if q:
            params["q"] = q
        data = self._get_with_payment_raw(endpoint, params=params, timeout=DEFAULT_FAST_TIMEOUT)
        if isinstance(data, list):
            return SymbolListResponse(symbols=data, count=len(data))
        return SymbolListResponse(
            symbols=data.get("symbols", data.get("feeds", [])),
            count=data.get("count"),
            **{k: v for k, v in data.items() if k not in {"symbols", "feeds", "count"}},
        )

    # ------------------------------------------------------------------
    # Multi-chain JSON-RPC (Solana payment)
    # ------------------------------------------------------------------

    def rpc(
        self,
        network: str,
        method: str,
        params: Optional[List[Any]] = None,
        *,
        id: Union[str, int] = 1,
    ) -> RpcResponse:
        """Make a single JSON-RPC 2.0 call (Solana payment, flat $0.002).

        Mirrors ``RPCClient.call``. ``network`` may be a chain name or alias
        (``eth``, ``sol``, ``base`` …); the gateway resolves it.
        """
        _safe_path_segment(network, "network")
        body: Dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            body["params"] = params
        data = self._request_with_payment_raw(f"/v1/rpc/{network}", body)
        return self._rpc_response(data, self._last_raw_headers, network)

    def rpc_batch(self, network: str, requests: List[Dict[str, Any]]) -> List[RpcResponse]:
        """Make a JSON-RPC 2.0 batch call (Solana payment, $0.002 x N)."""
        if not requests:
            raise ValueError("batch requires at least one request")
        _safe_path_segment(network, "network")
        body: List[Dict[str, Any]] = []
        for i, req in enumerate(requests):
            if "method" not in req:
                raise ValueError(f"batch request {i} is missing 'method'")
            body.append({"jsonrpc": "2.0", "id": i + 1, **req})
        data = self._request_with_payment_raw(f"/v1/rpc/{network}", body)  # type: ignore[arg-type]
        headers = self._last_raw_headers
        if not isinstance(data, list):
            data = [data]
        return [self._rpc_response(item, headers, network) for item in data]

    def search(
        self,
        query: str,
        *,
        sources: Optional[List[str]] = None,
        max_results: int = 10,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SearchResult:
        """Standalone search (Solana payment).

        ``timeout`` overrides the per-call HTTP timeout (defaults to
        ``DEFAULT_SEARCH_TIMEOUT`` — deep web/X tool-use can run minutes).
        """
        body: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }
        if sources is not None:
            body["sources"] = sources
        if from_date is not None:
            body["from_date"] = from_date
        if to_date is not None:
            body["to_date"] = to_date

        eff_timeout = timeout if timeout is not None else self._search_timeout
        data = self._request_with_payment_raw("/v1/search", body, timeout=eff_timeout)
        return SearchResult(**data)

    # ── Prediction Markets (Powered by Predexon) ────────────────────────────

    def pm(self, path: str, **params: Any) -> Dict[str, Any]:
        """Query Predexon prediction market data (GET, Solana payment). Powered by Predexon."""
        return self._get_with_payment_raw(f"/v1/pm/{path}", params or None)

    def pm_query(self, path: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Structured query for Predexon data (POST, Solana payment). Powered by Predexon."""
        return self._request_with_payment_raw(f"/v1/pm/{path}", query)

    def pm_markets(self, **params: Any) -> Dict[str, Any]:
        """List canonical cross-venue markets (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("markets", **params)

    def pm_listings(self, **params: Any) -> Dict[str, Any]:
        """List venue-native executable listings (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("markets/listings", **params)

    def pm_outcome(self, predexon_id: str) -> Dict[str, Any]:
        """Resolve a canonical Predexon outcome ID (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm(f"outcomes/{predexon_id}")

    def pm_polymarket_markets(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket markets (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/markets", **params)

    def pm_polymarket_events(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket events (Predexon v2). Tier 1 ($0.001/call)."""
        return self.pm("polymarket/events", **params)

    def pm_polymarket_markets_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket markets with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/markets/keyset", **params)

    def pm_polymarket_events_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket events with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/events/keyset", **params)

    def pm_polymarket_positions(self, **params: Any) -> Dict[str, Any]:
        """Polymarket open positions (per-wallet, market-level PnL).
        Tier 1 ($0.001/call)."""
        return self.pm("polymarket/positions", **params)

    def pm_polymarket_trades(self, **params: Any) -> Dict[str, Any]:
        """Recent Polymarket trades. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/trades", **params)

    def pm_polymarket_leaderboard(self, **params: Any) -> Dict[str, Any]:
        """Polymarket trader leaderboard. Tier 1 ($0.001/call)."""
        return self.pm("polymarket/leaderboard", **params)

    def pm_kalshi_markets(self, **params: Any) -> Dict[str, Any]:
        """List Kalshi markets. Tier 1 ($0.001/call)."""
        return self.pm("kalshi/markets", **params)

    def pm_limitless_markets(self, **params: Any) -> Dict[str, Any]:
        """List Limitless markets. Tier 1 ($0.001/call)."""
        return self.pm("limitless/markets", **params)

    def pm_sports_categories(self) -> Dict[str, Any]:
        """List available sports categories. Tier 1 ($0.001/call)."""
        return self.pm("sports/categories")

    def pm_sports_markets(self, **params: Any) -> Dict[str, Any]:
        """List sports markets grouped by game. Tier 1 ($0.001/call)."""
        return self.pm("sports/markets", **params)

    def pm_wallet_identity(self, wallet: str) -> Dict[str, Any]:
        """Identity + profile for one wallet. Tier 2 ($0.005/call)."""
        return self.pm(f"polymarket/wallet/identity/{wallet}")

    def pm_wallet_identities(self, addresses: List[str]) -> Dict[str, Any]:
        """Bulk identity for up to 200 wallet addresses. Tier 2 ($0.005/call)."""
        return self.pm_query("polymarket/wallet/identities", {"addresses": addresses})

    def pm_wallet_cluster(self, address: str) -> Dict[str, Any]:
        """Wallet-cluster discovery (on-chain transfers + identity proofs).
        Tier 2 ($0.005/call)."""
        return self.pm(f"polymarket/wallet/{address}/cluster")

    # ── Exa Web Search (Powered by Exa) ─────────────────────────────────────

    def exa(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Generic Exa endpoint proxy (POST, Solana payment). Powered by Exa.

        Args:
            path: Exa endpoint — one of: "search", "find-similar", "contents", "answer"
            body: Request body (see Exa API docs)

        Example::

            result = client.exa("search", {"query": "latest AI research", "numResults": 5})
        """
        return self._request_with_payment_raw(f"/v1/exa/{path}", body, timeout=self._search_timeout)

    def exa_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Neural and keyword web search via Exa (Solana payment, $0.01/request).

        Args:
            query: Search query string
            **kwargs: Additional Exa parameters (numResults, category, useAutoprompt, etc.)

        Example::

            results = client.exa_search("latest AI papers", numResults=5)
        """
        return self._request_with_payment_raw(
            "/v1/exa/search", {"query": query, **kwargs}, timeout=self._search_timeout
        )

    def exa_find_similar(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Find pages semantically similar to a given URL via Exa (Solana payment, $0.01/request).

        Args:
            url: URL to find similar pages for
            **kwargs: Additional Exa parameters (numResults, etc.)

        Example::

            results = client.exa_find_similar("https://openai.com/research/gpt-4", numResults=5)
        """
        return self._request_with_payment_raw(
            "/v1/exa/find-similar", {"url": url, **kwargs}, timeout=self._search_timeout
        )

    def exa_contents(self, urls: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Extract full text content from URLs via Exa (Solana payment, $0.002/URL).

        Args:
            urls: List of URLs to extract content from
            **kwargs: Additional Exa parameters (text, highlights, summary, etc.)

        Example::

            data = client.exa_contents(["https://arxiv.org/abs/2303.08774"])
        """
        return self._request_with_payment_raw(
            "/v1/exa/contents", {"urls": urls, **kwargs}, timeout=self._search_timeout
        )

    def exa_answer(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """AI-generated answer grounded in live web search via Exa (Solana payment, $0.01/request).

        Args:
            query: Question to answer
            **kwargs: Additional Exa parameters

        Example::

            answer = client.exa_answer("What is the current state of AI safety research?")
        """
        return self._request_with_payment_raw(
            "/v1/exa/answer", {"query": query, **kwargs}, timeout=self._search_timeout
        )

    # ── DefiLlama (DeFi protocols / TVL / yields / prices) ──────────────────

    def defi(self, path: str, **params: Any) -> Dict[str, Any]:
        """Query DefiLlama DeFi data (GET, Solana payment). $0.005/call
        ($0.001 for prices/{coins})."""
        return self._get_with_payment_raw(f"/v1/defillama/{path}", params or None)

    def defi_protocols(self) -> Dict[str, Any]:
        """All DeFi protocols with TVL ($0.005/call)."""
        return self.defi("protocols")

    def defi_protocol(self, slug: str) -> Dict[str, Any]:
        """Single protocol details + historical TVL ($0.005/call)."""
        return self.defi(f"protocol/{slug}")

    def defi_chains(self) -> Dict[str, Any]:
        """Current TVL of every chain ($0.005/call)."""
        return self.defi("chains")

    def defi_yields(self, **params: Any) -> Dict[str, Any]:
        """Yield pools with APY/TVL ($0.005/call)."""
        return self.defi("yields", **params)

    def defi_prices(self, coins: Union[List[str], str]) -> Dict[str, Any]:
        """Token price lookup ($0.001/call)."""
        joined = ",".join(coins) if isinstance(coins, list) else coins
        return self.defi(f"prices/{joined}")

    # ── 0x DEX (swap quotes + gasless) — free passthrough ───────────────────

    def dex(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """Query the 0x Swap / Gasless APIs (free — no x402 payment)."""
        endpoint = f"/v1/zerox/{path}"
        if method.upper() == "POST":
            return self._request_with_payment_raw(endpoint, body or {})
        return self._get_with_payment_raw(endpoint, params or None)

    def dex_price(self, **params: Any) -> Dict[str, Any]:
        """Indicative Permit2 swap price — no commitment (free)."""
        return self.dex("price", **params)

    def dex_quote(self, **params: Any) -> Dict[str, Any]:
        """Firm Permit2 swap quote with permit2.eip712 + tx data (free)."""
        return self.dex("quote", **params)

    def dex_gasless_price(self, **params: Any) -> Dict[str, Any]:
        """Gasless indicative price quote (free)."""
        return self.dex("gasless/price", **params)

    def dex_gasless_quote(self, **params: Any) -> Dict[str, Any]:
        """Gasless firm quote — returns trade.eip712 to sign (free)."""
        return self.dex("gasless/quote", **params)

    def dex_gasless_submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a signed gasless trade; the 0x relayer pays gas (free)."""
        return self.dex("gasless/submit", method="POST", body=body)

    def dex_gasless_status(self, trade_hash: str) -> Dict[str, Any]:
        """Poll a gasless trade's status by tradeHash (free)."""
        return self.dex(f"gasless/status/{trade_hash}")

    def dex_chains(self) -> Dict[str, Any]:
        """Chains where the Swap API is supported (free)."""
        return self.dex("swap/chains")

    def dex_gasless_chains(self) -> Dict[str, Any]:
        """Chains where the Gasless API is supported (free)."""
        return self.dex("gasless/chains")

    # ── Modal Sandbox (pay-per-call cloud compute) ───────────────────────────

    def modal(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call the Modal sandbox compute API (POST, Solana payment)."""
        return self._request_with_payment_raw(f"/v1/modal/{path}", body or {})

    def modal_sandbox_create(self, **body: Any) -> Dict[str, Any]:
        """Create a sandboxed compute environment ($0.01 CPU / $0.05 GPU)."""
        return self.modal("sandbox/create", body)

    def modal_sandbox_exec(
        self, sandbox_id: str, command: List[str], **body: Any
    ) -> Dict[str, Any]:
        """Execute a command in a sandbox; returns stdout/stderr ($0.001)."""
        return self.modal("sandbox/exec", {"sandbox_id": sandbox_id, "command": command, **body})

    def modal_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Check a sandbox's status ($0.001)."""
        return self.modal("sandbox/status", {"sandbox_id": sandbox_id})

    def modal_sandbox_terminate(self, sandbox_id: str) -> Dict[str, Any]:
        """Terminate a sandbox ($0.001)."""
        return self.modal("sandbox/terminate", {"sandbox_id": sandbox_id})


# ===========================================================================
# AsyncSolanaLLMClient — async mirror of SolanaLLMClient (chat only, v0.22.0)
# ===========================================================================
#
# Scope for the first release: chat completions, sync **and** streaming. Image,
# music, video, exa, predexon are sync-only on Solana for now — same as the
# Solana sync class shipped initially. They can be added in follow-up releases.


class AsyncSolanaLLMClient:
    """
    Async BlockRun Solana LLM Client — pays via Solana USDC x402.

    Mirrors :class:`SolanaLLMClient` but exposes ``await``-able methods so
    Python ``asyncio`` callers (FastAPI handlers, LiteLLM Proxy, etc.) don't
    have to thread-pool around blocking I/O.

    Usage::

        client = AsyncSolanaLLMClient()                  # SOLANA_WALLET_KEY env
        resp = await client.chat_completion(
            "openai/gpt-5.5",
            [{"role": "user", "content": "gm Solana"}],
        )
        await client.close()
    """

    SOLANA_API_URL = SOLANA_API_URL
    _STREAM_5XX_STATUSES = SolanaLLMClient._STREAM_5XX_STATUSES
    _STREAM_5XX_BACKOFFS = SolanaLLMClient._STREAM_5XX_BACKOFFS
    _MAX_PAYMENT_RETRIES = SolanaLLMClient._MAX_PAYMENT_RETRIES
    _PAYMENT_RETRY_BACKOFFS = SolanaLLMClient._PAYMENT_RETRY_BACKOFFS

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_url: str = SOLANA_API_URL,
        rpc_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        image_timeout: float = DEFAULT_IMAGE_TIMEOUT,
        search_timeout: float = DEFAULT_SEARCH_TIMEOUT,
        rpc_headers: Optional[Dict[str, str]] = None,
        transaction_log: Union[bool, str, "os.PathLike[str]", None] = None,
    ) -> None:
        """Async mirror of :class:`SolanaLLMClient.__init__`. Same env-var
        fallback for ``rpc_url`` / ``rpc_headers`` — see
        :func:`_resolve_rpc_config`. ``transaction_log`` works the same way
        — opt-in per-call log to a project folder (default ``./log/``)."""
        if not _HAS_X402:
            raise ImportError(
                "Solana payment requires the x402 SDK. "
                "Install with: pip install blockrun-llm[solana]"
            )
        from .solana_wallet import load_solana_wallet

        key = (
            private_key
            or os.environ.get("SOLANA_WALLET_KEY")
            or load_solana_wallet()  # disk: newest ~/.*/solana-wallet.json, else ~/.blockrun/.solana-session
        )
        if not key:
            raise ValueError(
                "Private key required. Pass private_key, set SOLANA_WALLET_KEY, "
                "or have a Solana wallet on disk "
                "(~/.<provider>/solana-wallet.json or ~/.blockrun/.solana-session)."
            )
        self._private_key = key
        validate_api_url(api_url)
        self._api_url = api_url.rstrip("/")

        resolved_url, resolved_headers = _resolve_rpc_config(rpc_url, rpc_headers)
        self._rpc_url = resolved_url
        self._rpc_headers = resolved_headers

        self._timeout = timeout
        self._image_timeout = image_timeout
        self._search_timeout = search_timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._session_total_usd = 0.0
        self._session_calls = 0
        self._last_call_cost: float = 0.0
        self._address: Optional[str] = None

        log_dir = _resolve_log_dir(transaction_log)
        self._tx_logger: Optional[TransactionLogger] = (
            TransactionLogger(log_dir) if log_dir is not None else None
        )
        self._last_settlement: Optional[Dict[str, Any]] = None
        # Response headers from the most recent raw paid POST — consumed by
        # rpc()/music()/speech() to surface the settlement receipt + gateway
        # metadata the shared JSON-only helper would otherwise drop. Read it
        # immediately after the helper returns (no intervening await).
        self._last_raw_headers: Optional[httpx.Headers] = None

        # Async x402 client + same SVM signer the sync class uses.
        from x402 import x402Client  # local import to keep optional dep clean

        self._x402_client = x402Client()
        try:
            signer = _create_signer(self._private_key)
        except Exception as e:
            # Parity with the Base client, which validates the resolved key up
            # front: turn a malformed key (incl. one auto-loaded from disk) into
            # a clean error instead of a raw base58/solders exception.
            raise ValueError(
                "Invalid Solana private key (expected a base58-encoded keypair " "or 32-byte seed)."
            ) from e
        _register_svm_with_headers(self._x402_client, signer, resolved_url, resolved_headers)
        # Lazily created on first sign (avoids binding asyncio.Lock to a loop at
        # construction time). Serializes the async signing critical section so a
        # shared client is safe across concurrent coroutines — see _sign_payment.
        self._payment_lock: Optional[asyncio.Lock] = None

    async def _sign_payment(self, payment_required: Any) -> Any:
        """Task-safe async wrapper around ``x402_client.create_payment_payload``.

        Mirrors the sync :meth:`SolanaLLMClient._sign_payment`: concurrent
        coroutines sharing one client would otherwise race on the x402 client's
        nonce/auth state and trip replay / amount-mismatch rejections under load.
        """
        if self._payment_lock is None:
            self._payment_lock = asyncio.Lock()
        async with self._payment_lock:
            return await self._x402_client.create_payment_payload(payment_required)

    def _capture_settlement(self, response: httpx.Response) -> Optional[Dict[str, Any]]:
        """Async-Solana twin of :meth:`SolanaLLMClient._capture_settlement`."""
        header = response.headers.get("x-payment-response") or response.headers.get(
            "X-PAYMENT-RESPONSE"
        )
        settlement = decode_settlement_header(header)
        self._last_settlement = settlement
        return settlement

    def _attach_receipt(self, data: Any) -> None:
        """Inject the settlement tx hash from the most recent paid POST into a
        raw response dict under ``txHash`` (mirrors the Base Music/Speech
        clients). No-op on free responses (no receipt header)."""
        tx_hash = _receipt_from_headers(self._last_raw_headers)
        if tx_hash and isinstance(data, dict) and not data.get("txHash"):
            data["txHash"] = tx_hash

    def _log_transaction(
        self,
        endpoint: str,
        body: Dict[str, Any],
        response: Any,
        cost_usd: float,
    ) -> None:
        """Async-Solana twin of :meth:`SolanaLLMClient._log_transaction`."""
        logger = self._tx_logger
        if logger is None:
            return
        settlement = self._last_settlement
        self._last_settlement = None
        try:
            logger.log(
                endpoint=endpoint,
                request=body,
                response=response,
                cost_usd=cost_usd,
                model=(body.get("model") if isinstance(body, dict) else None),
                wallet=self.get_wallet_address(),
                network="solana-mainnet" if self.is_solana() else "solana-other",
                client_kind=type(self).__name__,
                settlement=settlement,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncSolanaLLMClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Identity / state
    # ------------------------------------------------------------------

    def get_wallet_address(self) -> str:
        if not self._address:
            self._address = get_solana_public_key(self._private_key)
        return self._address

    def is_solana(self) -> bool:
        return "sol.blockrun.ai" in self._api_url

    def get_spending(self) -> Dict[str, Any]:
        return {"total_usd": self._session_total_usd, "calls": self._session_calls}

    def _billing_meta(self) -> Dict[str, Optional[str]]:
        return {
            "wallet": self.get_wallet_address(),
            "network": "solana-mainnet" if self.is_solana() else "solana-other",
            "client_kind": type(self).__name__,
        }

    # ------------------------------------------------------------------
    # Non-streaming chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        search: bool = False,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = await self.chat_completion(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            search=search,
            timeout=timeout,
            response_format=response_format,
            stop=stop,
        )
        return result.choices[0].message.content or ""

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> ChatResponse:
        body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop
        return await self._request_with_payment("/v1/chat/completions", body, timeout=timeout)

    async def list_models(self) -> List[Dict[str, Any]]:
        resp = await self._client.get(f"{self._api_url}/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        search: bool = False,
        search_parameters: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> "AsyncSolanaIterator":
        """Async streaming. Same protocol semantics as the sync
        :meth:`SolanaLLMClient.chat_completion_stream`; only the iteration
        protocol differs (``async for``)."""
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if search_parameters:
            body["search_parameters"] = search_parameters
        elif search:
            body["search_parameters"] = {"mode": "on"}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if stop is not None:
            body["stop"] = stop

        attempts = [model, *(fallback_models or [])]
        last_exc: Optional[Exception] = None

        for i, attempt_model in enumerate(attempts):
            body["model"] = attempt_model
            inner = self._stream_with_payment("/v1/chat/completions", body, timeout=timeout)
            chunks_yielded = 0
            try:
                async for chunk in inner:
                    chunks_yielded += 1
                    yield chunk
                return
            except Exception as exc:
                if chunks_yielded > 0:
                    raise
                if not _should_fallback_solana(exc):
                    raise
                last_exc = exc
                if i + 1 < len(attempts):
                    next_model = attempts[i + 1]
                    sys.stderr.write(
                        f"[blockrun_llm] async solana stream {attempt_model} -> "
                        f"{next_model} ({type(exc).__name__}: {str(exc)[:80]})\n"
                    )
        assert last_exc is not None
        raise last_exc

    async def _stream_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
    ):
        """Whole-request payment-retry wrapper around :meth:`_stream_once`
        (async). Re-runs the paid request on a recoverable payment rejection,
        only before the first chunk is yielded. See _MAX_PAYMENT_RETRIES."""
        for payment_attempt in range(self._MAX_PAYMENT_RETRIES + 1):
            yielded = 0
            try:
                async for chunk in self._stream_once(endpoint, body, timeout=timeout):
                    yielded += 1
                    yield chunk
                return
            except PaymentError as exc:
                if (
                    yielded > 0
                    or _is_unrecoverable_payment_error(str(exc))
                    or payment_attempt >= self._MAX_PAYMENT_RETRIES
                ):
                    raise
                await asyncio.sleep(
                    self._PAYMENT_RETRY_BACKOFFS[
                        min(payment_attempt, len(self._PAYMENT_RETRY_BACKOFFS) - 1)
                    ]
                )

    async def _stream_once(
        self,
        endpoint: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
    ):
        """Async version of :meth:`SolanaLLMClient._stream_once`."""
        url = f"{self._api_url}{endpoint}"
        req_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout
        backoffs = self._STREAM_5XX_BACKOFFS

        # ----- Phase 1: probe (no payment header) -----
        payment_headers: Optional[Dict[str, str]] = None
        cost_usd = 0.0

        for attempt in range(len(backoffs) + 1):
            async with self._client.stream(
                "POST", url, json=body, headers=req_headers, timeout=eff_timeout
            ) as resp1:
                if resp1.status_code == 200:
                    async for chunk in self._aiter_sse_chunks(resp1):
                        yield chunk
                    return
                await resp1.aread()
                if resp1.status_code == 402:
                    payment_headers, cost_usd = await self._sign_payment_from_response(resp1)
                    break
                if resp1.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import asyncio

                    await asyncio.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp1, after_payment=False)
        else:
            raise APIError("solana stream probe exhausted retries", 0, None)

        # ----- Phase 2: stream with PAYMENT-SIGNATURE -----
        assert payment_headers is not None
        for attempt in range(len(backoffs) + 1):
            async with self._client.stream(
                "POST", url, json=body, headers=payment_headers, timeout=eff_timeout
            ) as resp2:
                if resp2.status_code == 200:
                    if cost_usd > 0:
                        self._session_calls += 1
                        self._session_total_usd += cost_usd
                        self._last_call_cost = cost_usd
                        self._capture_settlement(resp2)
                    async for chunk in self._aiter_and_archive(resp2, body, cost_usd):
                        yield chunk
                    return
                await resp2.aread()
                if resp2.status_code == 402:
                    raise build_payment_rejected_error(resp2)
                if resp2.status_code in self._STREAM_5XX_STATUSES and attempt < len(backoffs):
                    import asyncio

                    await asyncio.sleep(backoffs[attempt])
                    continue
                self._raise_stream_error(resp2, after_payment=True)

    @staticmethod
    async def _aiter_sse_chunks(response: httpx.Response):
        async for raw_line in response.aiter_lines():
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[6:].strip()
            if payload == "[DONE]":
                return
            try:
                chunk_dict = _json.loads(payload)
            except Exception:
                continue
            try:
                yield ChatCompletionChunk(**chunk_dict)
            except Exception:
                yield ChatCompletionChunk.model_construct(**chunk_dict)

    async def _aiter_and_archive(
        self,
        response: httpx.Response,
        body: Dict[str, Any],
        cost_usd: float,
    ):
        """Async version of :meth:`SolanaLLMClient._iter_and_archive`."""
        assembled_id: Optional[str] = None
        assembled_model: Optional[str] = None
        assembled_created: int = 0
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None

        async for chunk in self._aiter_sse_chunks(response):
            if chunk.choices:
                choice = chunk.choices[0]
                content = stream_choice_content(choice)
                if content:
                    content_parts.append(content)
                fr = stream_choice_finish_reason(choice)
                if fr:
                    finish_reason = fr
            if assembled_id is None:
                _id, _model, _created = chunk_meta(chunk)
                if _id:
                    assembled_id = _id
                    assembled_model = _model
                    assembled_created = _created
            _usage = chunk_usage_dict(chunk)
            if _usage is not None:
                usage_dict = _usage
            # Race-free per-call x402 charge — see LLMClient._iter_and_archive.
            chunk.cost_usd = cost_usd
            yield chunk

        if cost_usd > 0:
            from .cache import save_to_cache

            response_data: Dict[str, Any] = {
                "id": assembled_id or "stream",
                "object": "chat.completion",
                "created": assembled_created or int(__import__("time").time()),
                "model": assembled_model or body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "".join(content_parts),
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "stream": True,
            }
            if usage_dict:
                response_data["usage"] = usage_dict
            try:
                save_to_cache(
                    "/v1/chat/completions",
                    body,
                    response_data,
                    cost_usd=cost_usd,
                    **self._billing_meta(),
                )
            except Exception:
                pass
            self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)

    # ------------------------------------------------------------------
    # Payment + transport helpers
    # ------------------------------------------------------------------

    async def _sign_payment_from_response(
        self,
        response: httpx.Response,
    ) -> Tuple[Dict[str, str], float]:
        payment_header = SolanaLLMClient._extract_payment_header(response)
        if not payment_header:
            raise PaymentError("402 response but no payment requirements found")
        payment_required = decode_payment_required_header(payment_header)
        payment_payload = await self._sign_payment(payment_required)
        encoded_payment = encode_payment_signature_header(payment_payload)
        cost_usd = float(payment_payload.accepted.amount) / 1e6
        return (
            {
                "Content-Type": "application/json",
                "User-Agent": _get_user_agent(),
                "PAYMENT-SIGNATURE": encoded_payment,
            },
            cost_usd,
        )

    # Reuse the sync class's pure helper — it doesn't touch async state.
    _raise_stream_error = SolanaLLMClient._raise_stream_error

    async def _request_with_payment(
        self, endpoint: str, body: Dict[str, Any], timeout: Optional[float] = None
    ) -> ChatResponse:
        """Whole-request payment-retry wrapper around :meth:`_request_once`
        (async). Same policy as the sync path — recoverable payment rejections
        re-run the entire request with a fresh signature. See _MAX_PAYMENT_RETRIES."""
        for payment_attempt in range(self._MAX_PAYMENT_RETRIES + 1):
            try:
                return await self._request_once(endpoint, body, timeout=timeout)
            except PaymentError as exc:
                if (
                    _is_unrecoverable_payment_error(str(exc))
                    or payment_attempt >= self._MAX_PAYMENT_RETRIES
                ):
                    raise
                await asyncio.sleep(
                    self._PAYMENT_RETRY_BACKOFFS[
                        min(payment_attempt, len(self._PAYMENT_RETRY_BACKOFFS) - 1)
                    ]
                )

    async def _request_once(
        self, endpoint: str, body: Dict[str, Any], timeout: Optional[float] = None
    ) -> ChatResponse:
        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout

        response = await self._client.post(url, json=body, headers=headers, timeout=eff_timeout)
        if response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            response = await self._client.post(url, json=body, headers=headers, timeout=eff_timeout)

        if response.status_code == 402:
            return await self._handle_payment_and_retry(url, body, response, timeout=eff_timeout)

        if not response.is_success:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )
        return ChatResponse(**response.json())

    async def _handle_payment_and_retry(
        self,
        url: str,
        body: Dict[str, Any],
        response: httpx.Response,
        timeout: Optional[float] = None,
    ) -> ChatResponse:
        eff_timeout = timeout if timeout is not None else self._timeout
        payment_headers, cost_usd = await self._sign_payment_from_response(response)

        retry_response = await self._client.post(
            url, json=body, headers=payment_headers, timeout=eff_timeout
        )
        if retry_response.status_code in (502, 503):
            import asyncio

            await asyncio.sleep(1)
            retry_response = await self._client.post(
                url, json=body, headers=payment_headers, timeout=eff_timeout
            )

        if retry_response.status_code == 402:
            raise build_payment_rejected_error(retry_response)
        if not retry_response.is_success:
            try:
                error_body = retry_response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error after payment: {retry_response.status_code}",
                retry_response.status_code,
                sanitize_error_response(error_body),
            )

        self._session_calls += 1
        self._session_total_usd += cost_usd
        self._last_call_cost = cost_usd
        self._capture_settlement(retry_response)

        response_data = retry_response.json()
        from .cache import save_to_cache

        save_to_cache(
            "/v1/chat/completions",
            body,
            response_data,
            cost_usd=cost_usd,
            **self._billing_meta(),
        )
        self._log_transaction("/v1/chat/completions", body, response_data, cost_usd)
        return ChatResponse(**response_data)

    # ── Raw passthrough request helpers (async, Solana payment) ─────────────

    async def _request_with_payment_raw(
        self, endpoint: str, body: Dict[str, Any], timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """POST with Solana x402 payment, returning raw JSON (async mirror of
        the sync :class:`SolanaLLMClient` helper)."""
        from .cache import get_cached, save_to_cache

        cached = get_cached(endpoint, body)
        if cached is not None:
            return cached

        # Reset per-call receipt headers; only a paid retry repopulates them, so
        # a free/cached model can't inherit a prior call's settlement receipt.
        self._last_raw_headers = None

        url = f"{self._api_url}{endpoint}"
        headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout

        response = await self._client.post(url, json=body, headers=headers, timeout=eff_timeout)
        if response.status_code in (502, 503):
            await asyncio.sleep(1)
            response = await self._client.post(url, json=body, headers=headers, timeout=eff_timeout)

        if response.status_code == 402:
            payment_headers, cost_usd = await self._sign_payment_from_response(response)
            retry_response = await self._client.post(
                url, json=body, headers=payment_headers, timeout=eff_timeout
            )
            if retry_response.status_code in (502, 503):
                await asyncio.sleep(1)
                retry_response = await self._client.post(
                    url, json=body, headers=payment_headers, timeout=eff_timeout
                )
            if retry_response.status_code == 402:
                raise build_payment_rejected_error(retry_response)
            if not retry_response.is_success:
                try:
                    error_body = retry_response.json()
                except Exception:
                    error_body = {"error": "Request failed"}
                raise APIError(
                    f"API error after payment: {retry_response.status_code}",
                    retry_response.status_code,
                    sanitize_error_response(error_body),
                )
            self._session_calls += 1
            self._session_total_usd += cost_usd
            self._last_call_cost = cost_usd
            self._capture_settlement(retry_response)
            self._last_raw_headers = retry_response.headers
            result = retry_response.json()
            save_to_cache(endpoint, body, result, cost_usd=cost_usd, **self._billing_meta())
            self._log_transaction(endpoint, body, result, cost_usd)
            return result

        if not response.is_success:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )
        return response.json()

    async def _get_with_payment_raw(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """GET with Solana x402 payment, returning raw JSON (async)."""
        from .cache import get_cached, save_to_cache

        cache_key_body = params or {}
        cached = get_cached(endpoint, cache_key_body)
        if cached is not None:
            return cached

        url = f"{self._api_url}{endpoint}"
        headers = {"User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._timeout

        response = await self._client.get(url, params=params, headers=headers, timeout=eff_timeout)
        if response.status_code in (502, 503):
            await asyncio.sleep(1)
            response = await self._client.get(
                url, params=params, headers=headers, timeout=eff_timeout
            )

        if response.status_code == 402:
            payment_headers, cost_usd = await self._sign_payment_from_response(response)
            retry_response = await self._client.get(
                url, params=params, headers=payment_headers, timeout=eff_timeout
            )
            if retry_response.status_code in (502, 503):
                await asyncio.sleep(1)
                retry_response = await self._client.get(
                    url, params=params, headers=payment_headers, timeout=eff_timeout
                )
            if retry_response.status_code == 402:
                raise build_payment_rejected_error(retry_response)
            if not retry_response.is_success:
                try:
                    error_body = retry_response.json()
                except Exception:
                    error_body = {"error": "Request failed"}
                raise APIError(
                    f"API error after payment: {retry_response.status_code}",
                    retry_response.status_code,
                    sanitize_error_response(error_body),
                )
            self._session_calls += 1
            self._session_total_usd += cost_usd
            self._last_call_cost = cost_usd
            self._capture_settlement(retry_response)
            result = retry_response.json()
            save_to_cache(
                endpoint, cache_key_body, result, cost_usd=cost_usd, **self._billing_meta()
            )
            self._log_transaction(endpoint, cache_key_body, result, cost_usd)
            return result

        if not response.is_success:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"API error: {response.status_code}",
                response.status_code,
                sanitize_error_response(error_body),
            )
        return response.json()

    # ── Standalone search (Grok Live Search) ────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        sources: Optional[List[str]] = None,
        max_results: int = 10,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SearchResult:
        """Standalone search (Solana payment).

        ``timeout`` overrides the per-call HTTP timeout (defaults to
        ``DEFAULT_SEARCH_TIMEOUT`` — deep web/X tool-use can run minutes).
        """
        body: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }
        if sources is not None:
            body["sources"] = sources
        if from_date is not None:
            body["from_date"] = from_date
        if to_date is not None:
            body["to_date"] = to_date

        eff_timeout = timeout if timeout is not None else self._search_timeout
        data = await self._request_with_payment_raw("/v1/search", body, timeout=eff_timeout)
        return SearchResult(**data)

    # ── Balance ─────────────────────────────────────────────────────────────

    async def get_balance(self) -> float:
        """Get USDC balance on Solana (async; matches the sync client API).

        The underlying RPC read is synchronous, so it runs in a worker thread
        to avoid blocking the event loop.
        """
        from .solana_wallet import get_solana_usdc_balance

        return await asyncio.to_thread(
            get_solana_usdc_balance, self.get_wallet_address(), rpc_url=self._rpc_url
        )

    # ── Image generation + editing ──────────────────────────────────────────

    async def image(
        self,
        prompt: str,
        *,
        model: str = "google/nano-banana",
        size: str = "1024x1024",
        n: int = 1,
        timeout: Optional[float] = None,
    ) -> ImageResponse:
        """Generate an image from a text prompt (Solana payment).

        Slow models (gpt-image-2, dall-e-3, nano-banana-pro 4K) trigger the
        gateway's async 202 + poll flow; this polls transparently until
        completion and only settles on the final completed poll. If the poll
        budget is exhausted an :class:`APIError` 504 is raised and **no payment
        is taken**.
        """
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }
        data = await self._request_image_with_payment(
            "/v1/images/generations", body, timeout=timeout
        )
        return ImageResponse(**data)

    async def image_edit(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        model: str = "openai/gpt-image-2",
        mask: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
        timeout: Optional[float] = None,
    ) -> ImageResponse:
        """Edit an image using img2img (Solana payment). ``image`` may be a
        single data URI or a list of 1-4 data URIs for multi-image fusion
        (openai/* up to 4, google/* up to 3). Handles the async 202 + poll
        slow path transparently — settlement only happens on completion.
        """
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image": image,
            "size": size,
            "n": n,
        }
        if mask is not None:
            body["mask"] = mask

        data = await self._request_image_with_payment(
            "/v1/images/image2image", body, timeout=timeout
        )
        return ImageResponse(**data)

    def _absolute_url(self, url: str) -> str:
        """Resolve a server-supplied relative ``poll_url`` against the API host
        (``api_url`` already includes the trailing ``/api`` — strip it once)."""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        base = self._api_url[: -len("/api")] if self._api_url.endswith("/api") else self._api_url
        return f"{base}{url}"

    # ── Video / music / speech / enrollment / market data (async) ──────────

    async def video(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        real_face_asset_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        generate_audio: Optional[bool] = None,
        seed: Optional[int] = None,
        watermark: Optional[bool] = None,
        return_last_frame: Optional[bool] = None,
        budget_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> VideoResponse:
        """Generate a video clip (Solana payment). Async mirror of
        :meth:`SolanaLLMClient.video`."""
        body = SolanaLLMClient._build_video_body(
            prompt,
            model=model,
            image_url=image_url,
            last_frame_url=last_frame_url,
            reference_image_urls=reference_image_urls,
            real_face_asset_id=real_face_asset_id,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            generate_audio=generate_audio,
            seed=seed,
            watermark=watermark,
            return_last_frame=return_last_frame,
        )

        data = await self._request_image_with_payment(
            "/v1/videos/generations",
            body,
            timeout=timeout,
            poll_budget_seconds=(
                budget_seconds
                if budget_seconds is not None
                else SolanaLLMClient.VIDEO_POLL_BUDGET_SECONDS
            ),
            poll_interval_seconds=SolanaLLMClient.VIDEO_POLL_INTERVAL_SECONDS,
            max_resigns=SolanaLLMClient.MEDIA_POLL_MAX_RESIGNS,
            label="Video generation",
        )
        return VideoResponse(**data)

    async def video_from_content(
        self,
        content: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        budget_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
        **options: Any,
    ) -> VideoResponse:
        """Generate a video from a Seedance ``content[]`` body (Solana payment)."""
        if not content:
            raise ValueError("content must be a non-empty list of Seedance content items.")
        body: Dict[str, Any] = {"content": content, **options}
        if model is not None:
            body["model"] = model
        data = await self._request_image_with_payment(
            "/v1/videos",
            body,
            timeout=timeout,
            poll_budget_seconds=(
                budget_seconds
                if budget_seconds is not None
                else SolanaLLMClient.VIDEO_POLL_BUDGET_SECONDS
            ),
            poll_interval_seconds=SolanaLLMClient.VIDEO_POLL_INTERVAL_SECONDS,
            max_resigns=SolanaLLMClient.MEDIA_POLL_MAX_RESIGNS,
            label="Video generation",
        )
        return VideoResponse(**data)

    async def music(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        instrumental: bool = True,
        lyrics: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> MusicResponse:
        """Generate a music track (Solana payment)."""
        if instrumental and lyrics and lyrics.strip():
            raise ValueError("Cannot specify lyrics when instrumental is True")
        body: Dict[str, Any] = {
            "model": model or SolanaLLMClient.MUSIC_DEFAULT_MODEL,
            "prompt": prompt,
            "instrumental": instrumental,
        }
        if lyrics and lyrics.strip():
            body["lyrics"] = lyrics.strip()
        data = await self._request_with_payment_raw("/v1/audio/generations", body, timeout=timeout)
        self._attach_receipt(data)
        return MusicResponse(**data)

    async def speech(
        self,
        input: str,
        *,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> SpeechResponse:
        """Synthesize speech from text (Solana payment)."""
        body: Dict[str, Any] = {
            "model": model or SolanaLLMClient.SPEECH_DEFAULT_MODEL,
            "input": input,
        }
        if voice:
            body["voice"] = voice
        if response_format:
            body["response_format"] = response_format
        if speed is not None:
            body["speed"] = speed
        data = await self._request_with_payment_raw("/v1/audio/speech", body, timeout=timeout)
        self._attach_receipt(data)
        return SpeechResponse(**data)

    async def sound_effect(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        prompt_influence: Optional[float] = None,
        response_format: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SpeechResponse:
        """Generate a cinematic sound effect (Solana payment)."""
        body: Dict[str, Any] = {
            "model": model or SolanaLLMClient.SOUNDFX_DEFAULT_MODEL,
            "text": text,
        }
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        if prompt_influence is not None:
            body["prompt_influence"] = prompt_influence
        if response_format:
            body["response_format"] = response_format
        data = await self._request_with_payment_raw(
            "/v1/audio/sound-effects", body, timeout=timeout
        )
        self._attach_receipt(data)
        return SpeechResponse(**data)

    async def list_voices(self) -> List[Dict[str, Any]]:
        """List available speech voices (free)."""
        url = f"{self._api_url}/v1/audio/voices"
        resp = await self._client.get(
            url, headers={"User-Agent": _get_user_agent()}, timeout=DEFAULT_FAST_TIMEOUT
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"List voices failed: HTTP {resp.status_code}",
                resp.status_code,
                sanitize_error_response(error_body),
            )
        data = resp.json()
        # Gateway wraps the voice list under "data" (mirrors SpeechClient.list_voices).
        return data.get("data", []) if isinstance(data, dict) else data

    async def portrait_enroll(self, name: str, image_url: str) -> PortraitEnrollment:
        """Enroll a Virtual Portrait ($0.01 USDC). Returns a ``ta_`` asset id."""
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > 64:
            raise ValueError(f"name must be 64 chars or fewer (got {len(name)})")
        if not image_url or not image_url.lower().startswith(("https://", "http://")):
            raise ValueError("image_url must be an http(s) URL")
        data = await self._request_with_payment_raw(
            "/v1/portrait/enroll", {"name": name, "image_url": image_url}
        )
        return PortraitEnrollment(**data)

    async def list_portraits(self, wallet_address: Optional[str] = None) -> PortraitList:
        """List Virtual Portraits enrolled by a wallet (free, rate-limited)."""
        addr = _safe_path_segment(wallet_address or self.get_wallet_address(), "wallet_address")
        url = f"{self._api_url}/v1/wallet/{addr}/portraits"
        resp = await self._client.get(
            url, headers={"User-Agent": _get_user_agent()}, timeout=DEFAULT_FAST_TIMEOUT
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "Portrait listing failed", resp.status_code, sanitize_error_response(error_body)
            )
        return PortraitList(**resp.json())

    async def realface_init(self, name: str, group_id: Optional[str] = None) -> RealFaceInit:
        """Start/refresh a RealFace enrollment (free, rate-limited)."""
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > 64:
            raise ValueError(f"name must be 64 chars or fewer (got {len(name)})")
        if group_id is not None and not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")
        body: Dict[str, Any] = {"name": name}
        if group_id:
            body["groupId"] = group_id
        url = f"{self._api_url}/v1/realface/init"
        resp = await self._client.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", "User-Agent": _get_user_agent()},
            timeout=DEFAULT_FAST_TIMEOUT,
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "RealFace init failed", resp.status_code, sanitize_error_response(error_body)
            )
        return RealFaceInit(**resp.json())

    async def realface_status(self, group_id: str) -> RealFaceStatus:
        """Poll a RealFace group's state (free, rate-limited)."""
        if not group_id or not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")
        url = f"{self._api_url}/v1/realface/status"
        resp = await self._client.get(
            url,
            params={"groupId": group_id},
            headers={"User-Agent": _get_user_agent()},
            timeout=DEFAULT_FAST_TIMEOUT,
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "RealFace status check failed",
                resp.status_code,
                sanitize_error_response(error_body),
            )
        return RealFaceStatus(**resp.json())

    async def realface_wait_for_active(
        self,
        group_id: str,
        timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 4.0,
    ) -> RealFaceStatus:
        """Block until the RealFace group is active (person finished the phone
        liveness check)."""
        import time as _time

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        deadline = _time.monotonic() + timeout_seconds
        while True:
            state = await self.realface_status(group_id)
            if state.ready_to_finalize:
                return state
            if _time.monotonic() + poll_interval_seconds >= deadline:
                raise TimeoutError(
                    f"RealFace group {group_id} not active after {timeout_seconds:.0f}s "
                    f"(last status: {state.status!r})."
                )
            await asyncio.sleep(poll_interval_seconds)

    async def realface_enroll(self, name: str, image_url: str, group_id: str) -> RealFaceEnrollment:
        """Finalize a RealFace enrollment ($0.01 USDC)."""
        if not name or not name.strip():
            raise ValueError("name is required (1-64 chars)")
        if len(name) > 64:
            raise ValueError(f"name must be 64 chars or fewer (got {len(name)})")
        if not image_url or not image_url.lower().startswith(("https://", "http://")):
            raise ValueError("image_url must be an http(s) URL")
        if not group_id or not _GROUP_ID_RE.match(group_id):
            raise ValueError("group_id must look like 'legacy_rf_<digits>'")
        data = await self._request_with_payment_raw(
            "/v1/realface/enroll", {"name": name, "image_url": image_url, "group_id": group_id}
        )
        return RealFaceEnrollment(**data)

    async def list_realfaces(self, wallet_address: Optional[str] = None) -> RealFaceList:
        """List RealFace assets enrolled by a wallet (free, rate-limited)."""
        addr = _safe_path_segment(wallet_address or self.get_wallet_address(), "wallet_address")
        url = f"{self._api_url}/v1/wallet/{addr}/realfaces"
        resp = await self._client.get(
            url, headers={"User-Agent": _get_user_agent()}, timeout=DEFAULT_FAST_TIMEOUT
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                "RealFace listing failed", resp.status_code, sanitize_error_response(error_body)
            )
        return RealFaceList(**resp.json())

    async def price(
        self,
        category: Category,
        symbol: str,
        *,
        market: Optional[Market] = None,
        session: Optional[Session] = None,
    ) -> PricePoint:
        """Fetch a realtime Pyth price quote (Solana payment for paid categories)."""
        endpoint = SolanaLLMClient._price_category_path(category, market, "price", symbol)
        params: Dict[str, Any] = {}
        if session is not None:
            params["session"] = session
        data = await self._get_with_payment_raw(
            endpoint, params=params or None, timeout=DEFAULT_FAST_TIMEOUT
        )
        return PricePoint(
            symbol=data.get("symbol", symbol.upper()),
            price=data.get("price"),
            publish_time=data.get("publishTime"),
            confidence=data.get("confidence"),
            feed_id=data.get("feedId"),
            **{
                k: v
                for k, v in data.items()
                if k not in {"symbol", "price", "publishTime", "confidence", "feedId"}
            },
        )

    async def price_history(
        self,
        category: Category,
        symbol: str,
        *,
        resolution: Resolution = "D",
        from_ts: int,
        to_ts: int,
        market: Optional[Market] = None,
        session: Optional[Session] = None,
    ) -> PriceHistoryResponse:
        """Fetch OHLC bars between two Unix timestamps (seconds)."""
        endpoint = SolanaLLMClient._price_category_path(category, market, "history", symbol)
        params: Dict[str, Any] = {"resolution": resolution, "from": from_ts, "to": to_ts}
        if session is not None:
            params["session"] = session
        data = await self._get_with_payment_raw(
            endpoint, params=params, timeout=DEFAULT_FAST_TIMEOUT
        )
        return PriceHistoryResponse(
            symbol=data.get("symbol", symbol.upper()),
            resolution=data.get("resolution", resolution),
            bars=data.get("bars", []),
            **{k: v for k, v in data.items() if k not in {"symbol", "resolution", "bars"}},
        )

    async def list_symbols(
        self,
        category: Category,
        *,
        q: Optional[str] = None,
        limit: int = 100,
        market: Optional[Market] = None,
    ) -> SymbolListResponse:
        """List available symbols in a Pyth category (free discovery)."""
        endpoint = SolanaLLMClient._price_category_path(category, market, "list", None)
        params: Dict[str, Any] = {"limit": limit}
        if q:
            params["q"] = q
        data = await self._get_with_payment_raw(
            endpoint, params=params, timeout=DEFAULT_FAST_TIMEOUT
        )
        if isinstance(data, list):
            return SymbolListResponse(symbols=data, count=len(data))
        return SymbolListResponse(
            symbols=data.get("symbols", data.get("feeds", [])),
            count=data.get("count"),
            **{k: v for k, v in data.items() if k not in {"symbols", "feeds", "count"}},
        )

    async def rpc(
        self,
        network: str,
        method: str,
        params: Optional[List[Any]] = None,
        *,
        id: Union[str, int] = 1,
    ) -> RpcResponse:
        """Make a single JSON-RPC 2.0 call (Solana payment, flat $0.002)."""
        _safe_path_segment(network, "network")
        body: Dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            body["params"] = params
        data = await self._request_with_payment_raw(f"/v1/rpc/{network}", body)
        return SolanaLLMClient._rpc_response(data, self._last_raw_headers, network)

    async def rpc_batch(self, network: str, requests: List[Dict[str, Any]]) -> List[RpcResponse]:
        """Make a JSON-RPC 2.0 batch call (Solana payment, $0.002 x N)."""
        if not requests:
            raise ValueError("batch requires at least one request")
        _safe_path_segment(network, "network")
        body: List[Dict[str, Any]] = []
        for i, req in enumerate(requests):
            if "method" not in req:
                raise ValueError(f"batch request {i} is missing 'method'")
            body.append({"jsonrpc": "2.0", "id": i + 1, **req})
        data = await self._request_with_payment_raw(f"/v1/rpc/{network}", body)  # type: ignore[arg-type]
        headers = self._last_raw_headers
        if not isinstance(data, list):
            data = [data]
        return [SolanaLLMClient._rpc_response(item, headers, network) for item in data]

    async def _request_image_with_payment(
        self,
        endpoint: str,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
        *,
        poll_budget_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
        max_resigns: int = 0,
        label: str = "Image",
    ) -> Dict[str, Any]:
        """Async sign + submit + poll wrapper for async media generation — the
        async mirror of the sync :class:`SolanaLLMClient` helper. Shared by
        :meth:`image` and :meth:`video` (``max_resigns`` re-signs to survive
        the 600s x402 authorization window on long video polls).
        """
        import time as _time

        from .cache import get_cached, save_to_cache

        cached = get_cached(endpoint, body)
        if cached is not None:
            return cached

        url = f"{self._api_url}{endpoint}"
        probe_headers = {"Content-Type": "application/json", "User-Agent": _get_user_agent()}
        eff_timeout = timeout if timeout is not None else self._image_timeout

        # Step 1: probe — expect 402 unless the model is free or cached upstream.
        probe = await self._client.post(url, json=body, headers=probe_headers, timeout=eff_timeout)
        if probe.status_code in (502, 503):
            await asyncio.sleep(1)
            probe = await self._client.post(
                url, json=body, headers=probe_headers, timeout=eff_timeout
            )

        if probe.status_code != 402:
            if not probe.is_success:
                try:
                    error_body = probe.json()
                except Exception:
                    error_body = {"error": "Request failed"}
                raise APIError(
                    f"Image request: HTTP {probe.status_code}",
                    probe.status_code,
                    sanitize_error_response(error_body),
                )
            return probe.json()

        # Step 2: sign x402 SVM payload (reuse the encoded signature on polls).
        payment_headers, cost_usd = await self._sign_payment_from_response(probe)
        encoded_payment = payment_headers["PAYMENT-SIGNATURE"]

        # Step 3: submit with signature.
        submit_resp = await self._client.post(
            url, json=body, headers=payment_headers, timeout=eff_timeout
        )
        if submit_resp.status_code in (502, 503):
            await asyncio.sleep(1)
            submit_resp = await self._client.post(
                url, json=body, headers=payment_headers, timeout=eff_timeout
            )

        if submit_resp.status_code == 402:
            raise build_payment_rejected_error(submit_resp)

        if submit_resp.status_code == 200:
            # Fast path — image produced inline.
            self._session_calls += 1
            self._session_total_usd += cost_usd
            self._last_call_cost = cost_usd
            self._capture_settlement(submit_resp)
            data = submit_resp.json()
            save_to_cache(endpoint, body, data, cost_usd=cost_usd, **self._billing_meta())
            self._log_transaction(endpoint, body, data, cost_usd)
            return data

        if submit_resp.status_code != 202:
            try:
                error_body = submit_resp.json()
            except Exception:
                error_body = {"error": "Request failed"}
            raise APIError(
                f"Image request after payment: HTTP {submit_resp.status_code}",
                submit_resp.status_code,
                sanitize_error_response(error_body),
            )

        # Step 4: slow path — poll until completed (or budget exhausted).
        try:
            submit_data = submit_resp.json()
        except Exception:
            submit_data = {}

        poll_url_rel = submit_data.get("poll_url")
        job_id = submit_data.get("id")
        if not poll_url_rel:
            raise APIError("Slow-path 202 missing poll_url", 202, {"response": submit_data})
        poll_url = self._absolute_url(poll_url_rel)
        poll_headers = {
            "User-Agent": _get_user_agent(),
            "PAYMENT-SIGNATURE": encoded_payment,
        }

        budget = (
            poll_budget_seconds
            if poll_budget_seconds is not None
            else SolanaLLMClient.IMAGE_POLL_BUDGET_SECONDS
        )
        interval = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else SolanaLLMClient.IMAGE_POLL_INTERVAL_SECONDS
        )
        deadline = _time.monotonic() + budget
        last_status = submit_data.get("status", "queued")
        resigns_left = max_resigns

        while _time.monotonic() < deadline:
            await asyncio.sleep(interval)

            poll_resp = await self._client.get(poll_url, headers=poll_headers, timeout=eff_timeout)
            try:
                poll_data = poll_resp.json()
            except Exception:
                poll_data = {}
            last_status = poll_data.get("status", last_status)

            if poll_resp.status_code == 402:
                # Mid-poll 402 = settlement failed, almost always a stale
                # blockhash (the payment was signed at submit time but only
                # settles when the job completes; by then the signed tx's
                # recent-blockhash can be expired -> transaction_simulation_failed).
                # The failing poll carries NO fresh challenge, so re-GET poll_url
                # WITHOUT the stale signature to solicit a fresh 402 (new
                # blockhash), re-sign, and keep polling. Mirrors the sync helper /
                # Base VideoClient.
                if resigns_left > 0:
                    resigns_left -= 1
                    try:
                        challenge = await self._client.get(
                            poll_url,
                            headers={"User-Agent": _get_user_agent()},
                            timeout=eff_timeout,
                        )
                        if challenge.status_code == 402:
                            resign_headers, _ = await self._sign_payment_from_response(challenge)
                            poll_headers["PAYMENT-SIGNATURE"] = resign_headers["PAYMENT-SIGNATURE"]
                            continue
                    except (PaymentError, httpx.HTTPError):
                        # Challenge GET or re-sign failed — surface the gateway's
                        # real 402 reason, not a network/signing error.
                        pass
                raise build_payment_rejected_error(poll_resp)

            if last_status == "failed":
                raise APIError(
                    f"{label} failed upstream: {poll_data.get('error', 'unknown')}",
                    poll_resp.status_code,
                    sanitize_error_response(poll_data if isinstance(poll_data, dict) else {}),
                )

            # Terminal success is keyed on status, NOT the HTTP code (see the
            # sync helper) — a completed-but-non-200 poll is still success.
            if last_status == "completed":
                tx_hash = poll_resp.headers.get("x-payment-receipt") or poll_resp.headers.get(
                    "X-Payment-Receipt"
                )
                if tx_hash and isinstance(poll_data, dict) and not poll_data.get("txHash"):
                    poll_data["txHash"] = tx_hash
                self._session_calls += 1
                self._session_total_usd += cost_usd
                self._last_call_cost = cost_usd
                self._capture_settlement(poll_resp)
                save_to_cache(endpoint, body, poll_data, cost_usd=cost_usd, **self._billing_meta())
                self._log_transaction(endpoint, body, poll_data, cost_usd)
                return poll_data

            if poll_resp.status_code in (202, 504):
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
                )

        raise APIError(
            (
                f"{label} did not complete within {budget:.0f}s "
                f"(last status: {last_status}). Settlement only happens on "
                "completion, so no payment was taken. The job stays claimable "
                "for ~48h — re-poll poll_url with a fresh signature from the "
                "same wallet to fetch (and settle) the finished result."
            ),
            504,
            {"id": job_id, "last_status": last_status, "poll_url": poll_url},
        )

    # ── Prediction Markets (Powered by Predexon) ────────────────────────────

    async def pm(self, path: str, **params: Any) -> Dict[str, Any]:
        """Query Predexon prediction market data (GET, Solana payment). Powered by Predexon."""
        return await self._get_with_payment_raw(f"/v1/pm/{path}", params or None)

    async def pm_query(self, path: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Structured query for Predexon data (POST, Solana payment). Powered by Predexon."""
        return await self._request_with_payment_raw(f"/v1/pm/{path}", query)

    async def pm_markets(self, **params: Any) -> Dict[str, Any]:
        """List canonical cross-venue markets (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("markets", **params)

    async def pm_listings(self, **params: Any) -> Dict[str, Any]:
        """List venue-native executable listings (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("markets/listings", **params)

    async def pm_outcome(self, predexon_id: str) -> Dict[str, Any]:
        """Resolve a canonical Predexon outcome ID (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm(f"outcomes/{predexon_id}")

    async def pm_polymarket_markets(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket markets (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/markets", **params)

    async def pm_polymarket_events(self, **params: Any) -> Dict[str, Any]:
        """List Polymarket events (Predexon v2). Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/events", **params)

    async def pm_polymarket_markets_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket markets with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/markets/keyset", **params)

    async def pm_polymarket_events_keyset(self, **params: Any) -> Dict[str, Any]:
        """Polymarket events with cursor-based keyset pagination. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/events/keyset", **params)

    async def pm_polymarket_positions(self, **params: Any) -> Dict[str, Any]:
        """Polymarket open positions (per-wallet, market-level PnL). Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/positions", **params)

    async def pm_polymarket_trades(self, **params: Any) -> Dict[str, Any]:
        """Recent Polymarket trades. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/trades", **params)

    async def pm_polymarket_leaderboard(self, **params: Any) -> Dict[str, Any]:
        """Polymarket trader leaderboard. Tier 1 ($0.001/call)."""
        return await self.pm("polymarket/leaderboard", **params)

    async def pm_kalshi_markets(self, **params: Any) -> Dict[str, Any]:
        """List Kalshi markets. Tier 1 ($0.001/call)."""
        return await self.pm("kalshi/markets", **params)

    async def pm_limitless_markets(self, **params: Any) -> Dict[str, Any]:
        """List Limitless markets. Tier 1 ($0.001/call)."""
        return await self.pm("limitless/markets", **params)

    async def pm_sports_categories(self) -> Dict[str, Any]:
        """List available sports categories. Tier 1 ($0.001/call)."""
        return await self.pm("sports/categories")

    async def pm_sports_markets(self, **params: Any) -> Dict[str, Any]:
        """List sports markets grouped by game. Tier 1 ($0.001/call)."""
        return await self.pm("sports/markets", **params)

    async def pm_wallet_identity(self, wallet: str) -> Dict[str, Any]:
        """Identity + profile for one wallet. Tier 2 ($0.005/call)."""
        return await self.pm(f"polymarket/wallet/identity/{wallet}")

    async def pm_wallet_identities(self, addresses: List[str]) -> Dict[str, Any]:
        """Bulk identity for up to 200 wallet addresses. Tier 2 ($0.005/call)."""
        return await self.pm_query("polymarket/wallet/identities", {"addresses": addresses})

    async def pm_wallet_cluster(self, address: str) -> Dict[str, Any]:
        """Wallet-cluster discovery (on-chain transfers + identity proofs). Tier 2 ($0.005/call)."""
        return await self.pm(f"polymarket/wallet/{address}/cluster")

    # ── Exa Web Search (Powered by Exa) ─────────────────────────────────────

    async def exa(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Generic Exa endpoint proxy (POST, Solana payment). Powered by Exa.

        Args:
            path: Exa endpoint — one of: "search", "find-similar", "contents", "answer"
            body: Request body (see Exa API docs)
        """
        return await self._request_with_payment_raw(
            f"/v1/exa/{path}", body, timeout=self._search_timeout
        )

    async def exa_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Neural and keyword web search via Exa (Solana payment, $0.01/request)."""
        return await self._request_with_payment_raw(
            "/v1/exa/search", {"query": query, **kwargs}, timeout=self._search_timeout
        )

    async def exa_find_similar(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Find pages semantically similar to a given URL via Exa (Solana payment, $0.01/request)."""
        return await self._request_with_payment_raw(
            "/v1/exa/find-similar", {"url": url, **kwargs}, timeout=self._search_timeout
        )

    async def exa_contents(self, urls: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Extract full text content from URLs via Exa (Solana payment, $0.002/URL)."""
        return await self._request_with_payment_raw(
            "/v1/exa/contents", {"urls": urls, **kwargs}, timeout=self._search_timeout
        )

    async def exa_answer(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """AI-generated answer grounded in live web search via Exa (Solana payment, $0.01/request)."""
        return await self._request_with_payment_raw(
            "/v1/exa/answer", {"query": query, **kwargs}, timeout=self._search_timeout
        )

    # ── DefiLlama (DeFi protocols / TVL / yields / prices) ──────────────────

    async def defi(self, path: str, **params: Any) -> Dict[str, Any]:
        """Query DefiLlama DeFi data (GET, Solana payment). $0.005/call
        ($0.001 for prices/{coins})."""
        return await self._get_with_payment_raw(f"/v1/defillama/{path}", params or None)

    async def defi_protocols(self) -> Dict[str, Any]:
        """All DeFi protocols with TVL ($0.005/call)."""
        return await self.defi("protocols")

    async def defi_protocol(self, slug: str) -> Dict[str, Any]:
        """Single protocol details + historical TVL ($0.005/call)."""
        return await self.defi(f"protocol/{slug}")

    async def defi_chains(self) -> Dict[str, Any]:
        """Current TVL of every chain ($0.005/call)."""
        return await self.defi("chains")

    async def defi_yields(self, **params: Any) -> Dict[str, Any]:
        """Yield pools with APY/TVL ($0.005/call)."""
        return await self.defi("yields", **params)

    async def defi_prices(self, coins: Union[List[str], str]) -> Dict[str, Any]:
        """Token price lookup ($0.001/call)."""
        joined = ",".join(coins) if isinstance(coins, list) else coins
        return await self.defi(f"prices/{joined}")

    # ── 0x DEX (swap quotes + gasless) — free passthrough ───────────────────

    async def dex(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """Query the 0x Swap / Gasless APIs (free — no x402 payment)."""
        endpoint = f"/v1/zerox/{path}"
        if method.upper() == "POST":
            return await self._request_with_payment_raw(endpoint, body or {})
        return await self._get_with_payment_raw(endpoint, params or None)

    async def dex_price(self, **params: Any) -> Dict[str, Any]:
        """Indicative Permit2 swap price — no commitment (free)."""
        return await self.dex("price", **params)

    async def dex_quote(self, **params: Any) -> Dict[str, Any]:
        """Firm Permit2 swap quote with permit2.eip712 + tx data (free)."""
        return await self.dex("quote", **params)

    async def dex_gasless_price(self, **params: Any) -> Dict[str, Any]:
        """Gasless indicative price quote (free)."""
        return await self.dex("gasless/price", **params)

    async def dex_gasless_quote(self, **params: Any) -> Dict[str, Any]:
        """Gasless firm quote — returns trade.eip712 to sign (free)."""
        return await self.dex("gasless/quote", **params)

    async def dex_gasless_submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a signed gasless trade; the 0x relayer pays gas (free)."""
        return await self.dex("gasless/submit", method="POST", body=body)

    async def dex_gasless_status(self, trade_hash: str) -> Dict[str, Any]:
        """Poll a gasless trade's status by tradeHash (free)."""
        return await self.dex(f"gasless/status/{trade_hash}")

    async def dex_chains(self) -> Dict[str, Any]:
        """Chains where the Swap API is supported (free)."""
        return await self.dex("swap/chains")

    async def dex_gasless_chains(self) -> Dict[str, Any]:
        """Chains where the Gasless API is supported (free)."""
        return await self.dex("gasless/chains")

    # ── Modal Sandbox (pay-per-call cloud compute) ───────────────────────────

    async def modal(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call the Modal sandbox compute API (POST, Solana payment)."""
        return await self._request_with_payment_raw(f"/v1/modal/{path}", body or {})

    async def modal_sandbox_create(self, **body: Any) -> Dict[str, Any]:
        """Create a sandboxed compute environment ($0.01 CPU / $0.05 GPU)."""
        return await self.modal("sandbox/create", body)

    async def modal_sandbox_exec(
        self, sandbox_id: str, command: List[str], **body: Any
    ) -> Dict[str, Any]:
        """Execute a command in a sandbox; returns stdout/stderr ($0.001)."""
        return await self.modal(
            "sandbox/exec", {"sandbox_id": sandbox_id, "command": command, **body}
        )

    async def modal_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Check a sandbox's status ($0.001)."""
        return await self.modal("sandbox/status", {"sandbox_id": sandbox_id})

    async def modal_sandbox_terminate(self, sandbox_id: str) -> Dict[str, Any]:
        """Terminate a sandbox ($0.001)."""
        return await self.modal("sandbox/terminate", {"sandbox_id": sandbox_id})


# A typing placeholder so the chat_completion_stream return type docs above
# don't reference a name pyright can't resolve.
AsyncSolanaIterator = Any
