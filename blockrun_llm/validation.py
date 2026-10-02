"""
Input validation and security utilities for BlockRun LLM SDK.

This module provides validation functions to ensure:
- Private keys are properly formatted
- API URLs use HTTPS
- Parameters are within valid ranges
- Server responses don't leak sensitive information
- Resource URLs match expected domains
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .types import PaymentError


# Localhost domains that are allowed to use HTTP
LOCALHOST_DOMAINS = {"localhost", "127.0.0.1"}

# Known LLM providers (for optional validation)
KNOWN_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "mistralai",
    "meta-llama",
    "together",
    "xai",
    "moonshot",
    "nvidia",
    "minimax",
    "zai",
}

# Seed modes a caller may assert via `input_type` on /v1/videos/generations.
# Mirrors the gateway enum; the gateway stays the authority on whether the
# declared mode matches the seed fields actually sent.
VIDEO_INPUT_TYPES = ("text", "image", "first_last_frame", "reference")

# Latency/fidelity levels for `quality` on Solana image generation + editing.
# Mirrors the gateway enum, which accepts the field for openai/gpt-image-* only.
IMAGE_QUALITY_LEVELS = ("low", "medium", "high", "auto")


# Base58 alphabet characters that never appear in a hex string. Their presence
# is a strong signal that a key is a base58-encoded Solana key, not an EVM key.
_BASE58_ONLY_CHARS = frozenset("GHJKLMNPQRSTUVWXYZghijkmnopqrstuvwxyz")


def _looks_like_solana_key(key: str) -> bool:
    """
    Heuristically detect a base58-encoded Solana secret key.

    Solana secret keys are base58, not hex: a 32-byte seed is ~43-44 chars and a
    64-byte keypair is ~87-88 chars. An EVM key is exactly 64 hex chars (sans the
    ``0x`` prefix). We treat a key as Solana when it contains a base58-only
    character (one absent from the hex alphabet) and its length is outside the
    EVM 64-char range — so a malformed 64-char hex key still routes to the
    regular hex error rather than the Solana hint.
    """
    candidate = key.removeprefix("0x")
    if len(candidate) == 64 or not (40 <= len(candidate) <= 90):
        return False
    return any(c in _BASE58_ONLY_CHARS for c in candidate)


# bool is a subclass of int in Python, so `isinstance(True, int)` is True and a
# bare numeric type check lets booleans straight through. Before this was fixed,
# validate_max_tokens(True), validate_temperature(True) and validate_top_p(False)
# all passed — the value then serialized as JSON `true`/`false` and went to the
# gateway as a request parameter. `max_tokens=False` was caught, but by the
# positivity check, so a type error was reported as a range error.
#
# Every numeric validator below excludes bool explicitly. Keep it that way when
# adding one.


def validate_private_key(key: str) -> None:
    """
    Validate that a private key is properly formatted.

    Args:
        key: The private key to validate

    Raises:
        ValueError: If the key format is invalid

    Example:
        >>> validate_private_key("0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
    """
    if not isinstance(key, str):
        raise ValueError("Private key must be a string")

    # Detect a base58 Solana key fed into the EVM (Base) client and point the
    # user at the right entry point instead of the cryptic "66 characters" error.
    if _looks_like_solana_key(key):
        raise ValueError(
            "This looks like a Solana (base58) private key, but this client uses "
            "the Base (EVM) chain. Use the Solana client instead:\n"
            "    from blockrun_llm import SolanaLLMClient\n"
            '    client = SolanaLLMClient(private_key="<your base58 key>")\n'
            "Or for agent use:\n"
            "    from blockrun_llm import setup_agent_solana_wallet\n"
            "    client = setup_agent_solana_wallet()\n"
            'Install Solana support first: pip install "blockrun-llm[solana]"'
        )

    # Must start with 0x
    if not key.startswith("0x"):
        raise ValueError("Private key must start with 0x")

    # Must be exactly 66 characters (0x + 64 hex chars)
    if len(key) != 66:
        raise ValueError("Private key must be 66 characters (0x + 64 hexadecimal characters)")

    # Must contain only valid hexadecimal characters
    if not re.match(r"^0x[0-9a-fA-F]{64}$", key):
        raise ValueError("Private key must contain only hexadecimal characters (0-9, a-f, A-F)")


def validate_eth_address(address: str) -> None:
    """
    Validate that a value is a well-formed Ethereum / Base address.

    Args:
        address: The 0x-prefixed 20-byte address to validate

    Raises:
        ValueError: If the address format is invalid

    Example:
        >>> validate_eth_address("0x036CbD53842c5426634e7929541eC2318f3dCF7e")
    """
    if not isinstance(address, str):
        raise ValueError("Address must be a string")

    # Must be a 0x-prefixed 40-character hexadecimal string
    if not re.match(r"^0x[0-9a-fA-F]{40}$", address):
        raise ValueError("Address must be a 0x-prefixed 40-character hexadecimal string")


def validate_model(model: str) -> None:
    """
    Validate model ID format.

    Args:
        model: The model ID (e.g., "openai/gpt-5.2", "anthropic/claude-sonnet-4.5")

    Raises:
        ValueError: If model is invalid

    Example:
        >>> validate_model("openai/gpt-5.2")
    """
    if not model or not isinstance(model, str):
        raise ValueError("Model must be a non-empty string")

    # Optionally validate provider (just a warning, don't fail)
    if "/" in model:
        provider = model.split("/", 1)[0]
        if provider not in KNOWN_PROVIDERS:
            # Just log, don't fail (allows new providers)
            pass


def validate_video_input_type(input_type: str | None) -> None:
    """
    Validate the optional `input_type` seed-mode assertion on video generation.

    Only the spelling is checked. Whether the declared mode agrees with the
    seed fields actually sent is the gateway's call — it infers the mode and
    rejects with 400 *before* charging, so re-deriving that inference here
    would add a second copy to keep in sync for no benefit.

    Args:
        input_type: One of VIDEO_INPUT_TYPES, or None to leave it unset.

    Raises:
        ValueError: If input_type is not one of the accepted values.

    Example:
        >>> validate_video_input_type("first_last_frame")
    """
    if input_type is None:
        return
    if input_type not in VIDEO_INPUT_TYPES:
        raise ValueError(
            f"input_type must be one of {', '.join(VIDEO_INPUT_TYPES)}; got {input_type!r}."
        )


# Per-model Seedance capabilities. Mirrors the gateway registry and the MCP's
# table (blockrun-mcp src/tools/video.ts), named rather than pattern-matched so
# a guard and its message can never drift apart.
SEEDANCE_25 = "bytedance/seedance-2.5"
SEEDANCE_15_PRO = "bytedance/seedance-1.5-pro"
# Reference IMAGES and the per-model count ceiling the gateway enforces.
SEEDANCE_REFERENCE_IMAGE_LIMIT: dict[str, int] = {
    "bytedance/seedance-2.0": 9,
    "bytedance/seedance-2.0-fast": 9,
    "bytedance/seedance-2.0-mini": 9,
    SEEDANCE_25: 30,
}
# Reference VIDEO/AUDIO clips. 2.5 is absent on purpose: it takes reference
# images (supportsReferenceImages) but not clips (supportsReferenceMedia:false).
SEEDANCE_REFERENCE_MEDIA_MODELS = frozenset(
    {"bytedance/seedance-2.0", "bytedance/seedance-2.0-fast", "bytedance/seedance-2.0-mini"}
)
SEEDANCE_BITRATE_MODE_MODELS = SEEDANCE_REFERENCE_MEDIA_MODELS | {SEEDANCE_25}
SEEDANCE_MAX_REFERENCE_CLIPS = 3
SEEDANCE_BITRATE_MODES = ("standard", "high")
SEEDANCE_OUTPUT_FORMATS = ("mp4", "mov")
_REFERENCE_CLIP_KEYS = frozenset({"url", "role"})


def _is_http_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("https://", "http://"))


def validate_video_request(
    model: str,
    *,
    api_key_mode: bool,
    solana_wallet: bool = False,
    image_url: str | None = None,
    last_frame_url: str | None = None,
    reference_image_urls: list[str] | None = None,
    reference_videos: list[dict[str, str]] | None = None,
    reference_audios: list[dict[str, str]] | None = None,
    real_face_asset_id: str | None = None,
    bitrate_mode: str | None = None,
    output_format: str | None = None,
    camera_fixed: bool | None = None,
    safety_identifier: str | None = None,
) -> None:
    """
    Refuse a video request the gateway would refuse, before anything is sent.

    Shared by ``VideoClient.generate`` (Base) and ``SolanaLLMClient.video``
    (sync and async), so the rails can never disagree about what they accept.
    On the wallet rails most of these would come back as a 400 before quoting;
    on the account rail there is no quote step at all (the key is billed at
    submit), so a refusal here is the only gate that runs before money moves.

    Args:
        model: The resolved model id (defaults already applied).
        api_key_mode: True when the client pays with a ``brk_`` API key.
        solana_wallet: True for the Solana wallet rail (sol.blockrun.ai).
        Remaining args mirror the video kwargs; None (or an empty list) omits.

    Raises:
        ValueError: On any combination the gateway would refuse.
    """
    has_refs = bool(reference_image_urls or reference_videos or reference_audios)
    has_clips = bool(reference_videos or reference_audios)

    # The rail fact dominates every other reference guard: both wallet
    # gateways answer any reference_* field with a 400 before quoting
    # (blockrun#728, blockrun-sol#374), so a per-model message would point the
    # caller at a fix that still cannot work.
    if has_refs and not api_key_mode:
        raise ValueError(
            "Reference media (reference_image_urls / reference_videos / "
            "reference_audios) is served only by the BlockRun account rail "
            "(api.blockrun.ai); the Base and Solana wallet gateways refuse it. "
            "Set BLOCKRUN_API_KEY to use the account rail, or use image_url / "
            "last_frame_url frame seeding, which works on the wallet rails."
        )
    if has_refs and (image_url or last_frame_url or real_face_asset_id):
        raise ValueError(
            "Reference inputs are mutually exclusive with image_url, last_frame_url, "
            "and real_face_asset_id; pass character or style images as "
            "reference_image_urls instead."
        )

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
    # sol.blockrun.ai is a separate deploy that has not taken 2.5 into its
    # first-and-last-frame list: it 400s where Base quotes the same body
    # (probed 2026-09-29). Same refusal as blockrun-mcp 0.53.1.
    if last_frame_url and solana_wallet and model == SEEDANCE_25:
        raise ValueError(
            f"{SEEDANCE_25} first-and-last-frame (last_frame_url) is not served by the "
            "Solana gateway yet. Use it on Base (VideoClient) or the account rail "
            "(BLOCKRUN_API_KEY), or pick seedance-1.5-pro / 2.0 / 2.0-fast / 2.0-mini."
        )

    if reference_image_urls:
        limit = SEEDANCE_REFERENCE_IMAGE_LIMIT.get(model)
        if limit is None:
            raise ValueError(
                f"Model {model} does not accept reference images (reference_image_urls). "
                f"Supported: {', '.join(SEEDANCE_REFERENCE_IMAGE_LIMIT)}."
            )
        if len(reference_image_urls) > limit:
            raise ValueError(
                f"reference_image_urls accepts at most {limit} images on {model}; "
                f"got {len(reference_image_urls)}."
            )
        if not all(_is_http_url(u) for u in reference_image_urls):
            raise ValueError("reference_image_urls must all be http(s) URLs.")

    if has_clips and model not in SEEDANCE_REFERENCE_MEDIA_MODELS:
        raise ValueError(
            f"Model {model} does not accept reference video or audio clips. "
            f"Supported: {', '.join(sorted(SEEDANCE_REFERENCE_MEDIA_MODELS))}."
            + (
                " 2.5 takes reference IMAGES (up to 30) but no clips."
                if model == SEEDANCE_25
                else ""
            )
        )
    for field, clips in (
        ("reference_videos", reference_videos),
        ("reference_audios", reference_audios),
    ):
        if not clips:
            continue
        if len(clips) > SEEDANCE_MAX_REFERENCE_CLIPS:
            raise ValueError(
                f"{field} accepts at most {SEEDANCE_MAX_REFERENCE_CLIPS} clips; got {len(clips)}."
            )
        for clip in clips:
            if (
                not isinstance(clip, dict)
                or not _is_http_url(clip.get("url"))
                or not set(clip) <= _REFERENCE_CLIP_KEYS
                or clip.get("role", "reference") != "reference"
            ):
                raise ValueError(
                    f"{field} entries must be {{'url': 'https://…'}} with an optional "
                    "'role': 'reference' and no other keys."
                )
    if reference_audios and not (reference_image_urls or reference_videos):
        raise ValueError(
            "reference_audios requires a reference image or video — combine it with "
            "reference_image_urls or reference_videos."
        )

    if bitrate_mode is not None:
        if bitrate_mode not in SEEDANCE_BITRATE_MODES:
            raise ValueError(
                f"bitrate_mode must be one of {', '.join(SEEDANCE_BITRATE_MODES)}; "
                f"got {bitrate_mode!r}."
            )
        if model not in SEEDANCE_BITRATE_MODE_MODELS:
            raise ValueError(
                f"bitrate_mode requires a Seedance 2.x model; got {model}. "
                f"Supported: {', '.join(sorted(SEEDANCE_BITRATE_MODE_MODELS))}."
            )
    if output_format is not None:
        if output_format not in SEEDANCE_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {', '.join(SEEDANCE_OUTPUT_FORMATS)}; "
                f"got {output_format!r}."
            )
        if model != SEEDANCE_25:
            raise ValueError(
                f"output_format requires {SEEDANCE_25}; got {model}. Every other model returns MP4."
            )
    if camera_fixed is not None and model != SEEDANCE_15_PRO:
        raise ValueError(f"camera_fixed requires {SEEDANCE_15_PRO}; got {model}.")
    if safety_identifier is not None and not model.startswith("bytedance/seedance-"):
        raise ValueError(f"safety_identifier requires a Seedance model; got {model}.")

    if real_face_asset_id is not None and not real_face_asset_id.startswith("ta_"):
        raise ValueError(
            "real_face_asset_id must start with 'ta_' "
            "(a Virtual Portrait or RealFace asset id, e.g. 'ta_abc123xyz' — "
            "enroll via PortraitClient / POST /v1/portrait/enroll or "
            "RealFaceClient / POST /v1/realface/enroll)"
        )


def validate_image_quality(quality: str | None) -> None:
    """
    Validate the optional `quality` knob on Solana image generation/editing.

    Model compatibility is left to the gateway, which accepts `quality` only
    for openai/gpt-image-* and returns a clear error otherwise — encoding that
    model list here would go stale every time the catalog changes.

    Args:
        quality: One of IMAGE_QUALITY_LEVELS, or None to leave it unset.

    Raises:
        ValueError: If quality is not one of the accepted values.

    Example:
        >>> validate_image_quality("low")
    """
    if quality is None:
        return
    if quality not in IMAGE_QUALITY_LEVELS:
        raise ValueError(
            f"quality must be one of {', '.join(IMAGE_QUALITY_LEVELS)}; got {quality!r}."
        )


# Client-side typo guard, NOT a model limit.
#
# This was 100000, which sat below what models actually serve: zai/glm-5.2
# serves 262144 and the common ceiling is 128000, so the SDK — not the model —
# was the binding constraint, and callers got a ValueError naming a limit no
# provider had set.
#
# The gateway does NOT reject an over-ceiling max_tokens. It silently clamps to
# the model's ceiling and quotes payment for the clamped value (probed against
# the live 402 leg 2026-07-21: opus-4.8 sent 262144 and 1000000 both quote the
# 128000 price; gpt-5.2 sent 1e12 returns a quote, not a 400). So there is no
# server-side rejection to fall back on — whatever passes here gets priced, and
# anything above the model's ceiling is money spent on tokens you won't get.
# ``LLMClient`` warns when it sees the gateway clamp; see ``_warn_if_clamped``.
#
# Keep a bound so an obvious mistake (1e9, a byte count, a timestamp) fails
# fast locally instead of becoming a payment quote. Set it far above any real
# model so it can never be the binding constraint again.
MAX_TOKENS_SANITY_LIMIT = 1_000_000


def validate_max_tokens(max_tokens: int | None) -> None:
    """
    Validate max_tokens parameter.

    Rejects only values no request could have meant. The gateway does not
    reject an over-ceiling ``max_tokens`` — it clamps to the model's own
    ceiling and charges for the clamped value — so this guard exists to stop a
    typo locally, not to enforce any model's limit.

    Args:
        max_tokens: Maximum number of tokens to generate

    Raises:
        ValueError: If max_tokens is invalid

    Example:
        >>> validate_max_tokens(1000)
    """
    if max_tokens is None:
        return

    # bool is an int subclass, so `isinstance(True, int)` is True and a stray
    # flag threaded into the wrong keyword would sail through and reach the
    # wire as `"max_tokens": true`. Say so explicitly — "must be an integer"
    # reads as wrong to anyone who knows bool is one.
    if isinstance(max_tokens, bool):
        raise ValueError("max_tokens must be an integer, got a bool")

    if not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive (minimum: 1)")

    if max_tokens > MAX_TOKENS_SANITY_LIMIT:
        raise ValueError(
            f"max_tokens implausibly large (client-side sanity limit: "
            f"{MAX_TOKENS_SANITY_LIMIT}). This is not a model limit — no "
            f"provider set it. Anything under it is sent to the gateway, "
            f"which clamps to the model's own ceiling and charges for the "
            f"clamped value rather than rejecting."
        )


def validate_temperature(temperature: float | None) -> None:
    """
    Validate temperature parameter.

    Args:
        temperature: Sampling temperature (0-2)

    Raises:
        ValueError: If temperature is invalid

    Example:
        >>> validate_temperature(0.7)
    """
    if temperature is None:
        return

    if isinstance(temperature, bool):
        raise ValueError("temperature must be a number, got a bool")

    if not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")

    if temperature < 0 or temperature > 2:
        raise ValueError("temperature must be between 0 and 2")


def validate_top_p(top_p: float | None) -> None:
    """
    Validate top_p parameter (nucleus sampling).

    Args:
        top_p: Top-p sampling parameter (0-1)

    Raises:
        ValueError: If top_p is invalid

    Example:
        >>> validate_top_p(0.9)
    """
    if top_p is None:
        return

    if isinstance(top_p, bool):
        raise ValueError("top_p must be a number, got a bool")

    if not isinstance(top_p, (int, float)):
        raise ValueError("top_p must be a number")

    if top_p < 0 or top_p > 1:
        raise ValueError("top_p must be between 0 and 1")


def validate_api_url(url: str) -> None:
    """
    Validate that an API URL is secure and properly formatted.

    Args:
        url: The API URL to validate

    Raises:
        ValueError: If the URL is invalid or insecure

    Example:
        >>> validate_api_url("https://blockrun.ai/api")
        >>> validate_api_url("http://localhost:3000")  # OK for development
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid API URL: {e}")

    if not parsed.scheme:
        raise ValueError("API URL must include scheme (http:// or https://)")

    if not parsed.netloc:
        raise ValueError("API URL must include domain")

    # Require HTTPS for non-localhost URLs
    is_localhost = parsed.netloc.split(":")[0] in LOCALHOST_DOMAINS

    if parsed.scheme != "https" and not is_localhost:
        raise ValueError(
            "API URL must use HTTPS for non-localhost endpoints. "
            f"Use https:// instead of {parsed.scheme}://"
        )


def build_payment_rejected_error(response: Any) -> PaymentError:
    """Translate a 402 retry response into a :class:`PaymentError` that
    preserves the gateway's original failure reason.

    Without this helper, clients used to throw a generic
    ``"Payment rejected. Check your wallet balance."`` and the real
    facilitator reason (e.g. ``transaction_simulation_failed``,
    ``insufficient_funds``) was lost.

    The gateway's ``details`` field on a 402 settlement-failed response
    is the x402 facilitator's well-defined error enum — safe to surface
    verbatim. We bound the length defensively in case a future server
    bug widens the field.

    Args:
        response: An ``httpx.Response`` with status 402 from a paid
            retry. Anything with a ``.json()`` method works for tests.

    Returns:
        A :class:`PaymentError` carrying ``status_code=402`` and a
        ``response`` dict that includes the gateway's ``details``.
    """
    # Local import to avoid a circular module dependency at import time.
    from .types import PaymentError

    try:
        body = response.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    sanitized = dict(sanitize_error_response(body))
    raw_details = body.get("details")
    if isinstance(raw_details, str) and 0 < len(raw_details) < 256:
        sanitized["details"] = raw_details
    # The x402 facilitator's `invalidMessage` — the simulation-level cause that
    # the coarse `invalidReason` enum collapses away (an unfunded wallet and a
    # stale blockhash both arrive as transaction_simulation_failed). Same
    # provenance and safety rationale as `details` above: a facilitator error
    # string, not upstream text, so it's safe to surface verbatim — bounded
    # defensively all the same. Still folded into the message for human-readable
    # errors and for `str(exc)` consumers; `_is_safe_resign_error` reads the
    # structured `exc.response` keys directly.
    raw_invalid_message = body.get("invalidMessage")
    if isinstance(raw_invalid_message, str) and 0 < len(raw_invalid_message) < 256:
        sanitized["invalidMessage"] = raw_invalid_message
    # Machine-readable payment classification used by the Solana client's
    # re-sign phase gate (:func:`solana_client._is_safe_resign_error`). These are
    # gateway-owned enums, not raw upstream text. Preserve only short strings;
    # debug remains intentionally filtered.
    raw_reason = body.get("reason")
    if isinstance(raw_reason, str) and 0 < len(raw_reason) < 128:
        sanitized["reason"] = raw_reason
    raw_code = body.get("code")
    if isinstance(raw_code, str) and 0 < len(raw_code) < 128:
        sanitized["code"] = raw_code
    detail_part = sanitized.get("details") or sanitized.get("message") or ""
    invalid_message = sanitized.get("invalidMessage")
    if invalid_message:
        detail_part = f"{detail_part} ({invalid_message})" if detail_part else invalid_message
    msg = (
        f"Payment rejected by gateway: {detail_part}"
        if detail_part
        else "Payment rejected by gateway"
    )
    return PaymentError(msg, status_code=402, response=sanitized)


def sanitize_error_response(error_body: Any) -> dict[str, Any]:
    """
    Sanitize API error responses to prevent information leakage.

    Only exposes safe error fields to the caller, filtering out:
    - Internal stack traces
    - Server-side paths
    - API keys or tokens
    - Debugging information

    Args:
        error_body: The raw error response from the API

    Returns:
        Sanitized error dict with only safe fields

    Example:
        >>> sanitize_error_response({
        ...     "error": "Invalid model",
        ...     "internal_stack": "/var/app/handler.py:123",
        ...     "api_key": "secret"
        ... })
        {'message': 'Invalid model', 'code': None}
    """
    if not isinstance(error_body, dict):
        return {"message": "API request failed", "code": None}

    # The gateway returns OpenAI-compatible *nested* errors:
    #   {"error": {"message", "type", "code", "param"}, "message", "code", "debug"}
    # while older endpoints (and the SDK's own fallbacks) still use the *flat* shape:
    #   {"error": "Request failed", "code": "..."}
    # Pass the real message/code through for either shape. Never surface `debug`
    # (raw upstream error text — may leak internal paths/keys).
    nested = error_body.get("error")

    if isinstance(nested, dict):
        message = nested.get("message")
        code = nested.get("code") or error_body.get("code")
        result: dict[str, Any] = {
            "message": message if isinstance(message, str) else "API request failed",
            "code": code if isinstance(code, str) else None,
        }
        # Pass through OpenAI error metadata when present.
        if isinstance(nested.get("type"), str):
            result["type"] = nested["type"]
        if isinstance(nested.get("param"), str):
            result["param"] = nested["param"]
        return result

    # Flat shape: `error` is the human-readable title; fall back to top-level `message`.
    if isinstance(nested, str):
        message = nested
    elif isinstance(error_body.get("message"), str):
        message = error_body["message"]
    else:
        message = "API request failed"

    return {
        "message": message,
        "code": (error_body.get("code") if isinstance(error_body.get("code"), str) else None),
    }


def validate_resource_url(url: str, base_url: str) -> str:
    """
    Validate a resource URL from the server to prevent redirection attacks.

    Ensures that the resource URL's hostname matches the API's hostname.
    If domains don't match, returns a safe default URL instead.

    Args:
        url: The resource URL provided by the server
        base_url: The base API URL (trusted)

    Returns:
        The validated URL or a safe default

    Example:
        >>> validate_resource_url(
        ...     "https://blockrun.ai/api/v1/chat",
        ...     "https://blockrun.ai/api"
        ... )
        'https://blockrun.ai/api/v1/chat'

        >>> validate_resource_url(
        ...     "https://malicious.com/steal",
        ...     "https://blockrun.ai/api"
        ... )
        'https://blockrun.ai/api/v1/chat/completions'
    """
    try:
        parsed = urlparse(url)
        base_parsed = urlparse(base_url)

        # Resource URL hostname must match API hostname
        if parsed.netloc != base_parsed.netloc:
            # Return safe default
            return f"{base_url}/v1/chat/completions"

        # Ensure resource uses same protocol as base
        if parsed.scheme != base_parsed.scheme:
            return f"{base_url}/v1/chat/completions"

        return url

    except Exception:
        # Invalid URL format, return safe default
        return f"{base_url}/v1/chat/completions"


def resolve_spend_limit(explicit: float | None, env_var: str) -> float | None:
    """Resolve a spend limit from the constructor argument or its env var.

    ``None`` means unlimited, which is the default and the pre-1.9.0 behavior.
    An unparseable or non-positive env value is ignored rather than raising:
    a malformed env var must not brick every client in a deployment, and the
    explicit argument always wins.
    """
    if explicit is not None:
        limit = float(explicit)
        if limit <= 0:
            raise ValueError(f"spend limit must be positive; got {explicit!r}")
        return limit

    import os

    raw = os.environ.get(env_var)
    if not raw:
        return None
    try:
        limit = float(raw)
    except ValueError:
        return None
    return limit if limit > 0 else None


def check_spend_limits(
    cost_usd: float,
    *,
    max_cost_per_call: float | None,
    max_session_cost: float | None,
    session_spent_usd: float,
    model: str | None = None,
) -> None:
    """Refuse a quote that would breach a caller-configured spend limit.

    Call this after the gateway's price is known and BEFORE the paid request is
    sent. Signing alone moves no money — the gateway submitting the signed
    authorization does — so declining here means nothing settles.

    Both limits are opt-in. With neither set this is a no-op, which is why
    adding it changes no existing behavior.

    Raises:
        SpendLimitError: If the quote exceeds the per-call limit, or if it would
            push the session past its total.
    """
    from .types import SpendLimitError

    where = f" for {model}" if model else ""

    if max_cost_per_call is not None and cost_usd > max_cost_per_call:
        raise SpendLimitError(
            f"Refused a ${cost_usd:.6f} quote{where}: it exceeds the per-call "
            f"limit of ${max_cost_per_call:.6f}. Nothing was sent and nothing "
            f"was charged. Raise max_cost_per_call to allow it.",
            quoted_usd=cost_usd,
            limit_usd=max_cost_per_call,
            scope="call",
        )

    if max_session_cost is not None and session_spent_usd + cost_usd > max_session_cost:
        remaining = max_session_cost - session_spent_usd
        raise SpendLimitError(
            f"Refused a ${cost_usd:.6f} quote{where}: this client has spent "
            f"${session_spent_usd:.6f} of its ${max_session_cost:.6f} session "
            f"limit, leaving ${remaining:.6f}. Nothing was sent and nothing was "
            f"charged.",
            quoted_usd=cost_usd,
            limit_usd=max_session_cost,
            scope="session",
        )


def raise_api_error(resp: Any, prefix: str) -> NoReturn:
    """Turn a failed HTTP response into an ``APIError``, keeping ``Retry-After``.

    Three clients carried byte-identical private copies of this. One copy means
    a change to how failures are reported — such as keeping the rate-limit
    header — lands everywhere at once instead of in two places out of three.
    """
    # Local, like build_payment_rejected_error below: types.py is imported by
    # every client, and importing it at module scope here would close a cycle.
    from .types import APIError

    try:
        error_body = resp.json()
    except Exception:
        error_body = {"error": "Request failed"}
    raise APIError.from_response(
        resp,
        f"{prefix}: HTTP {resp.status_code}",
        sanitize_error_response(error_body),
    )
