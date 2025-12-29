"""
Input validation and security utilities for BlockRun LLM SDK.

This module provides validation functions to ensure:
- Private keys are properly formatted
- API URLs use HTTPS
- Parameters are within valid ranges
- Server responses don't leak sensitive information
- Resource URLs match expected domains
"""

import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse


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
}


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

    # Must start with 0x
    if not key.startswith("0x"):
        raise ValueError("Private key must start with 0x")

    # Must be exactly 66 characters (0x + 64 hex chars)
    if len(key) != 66:
        raise ValueError(
            "Private key must be 66 characters (0x + 64 hexadecimal characters)"
        )

    # Must contain only valid hexadecimal characters
    if not re.match(r"^0x[0-9a-fA-F]{64}$", key):
        raise ValueError(
            "Private key must contain only hexadecimal characters (0-9, a-f, A-F)"
        )


def validate_model(model: str) -> None:
    """
    Validate model ID format.

    Args:
        model: The model ID (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4.5")

    Raises:
        ValueError: If model is invalid

    Example:
        >>> validate_model("openai/gpt-4o")
    """
    if not model or not isinstance(model, str):
        raise ValueError("Model must be a non-empty string")

    # Optionally validate provider (just a warning, don't fail)
    if "/" in model:
        provider = model.split("/", 1)[0]
        if provider not in KNOWN_PROVIDERS:
            # Just log, don't fail (allows new providers)
            pass


def validate_max_tokens(max_tokens: Optional[int]) -> None:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Maximum number of tokens to generate

    Raises:
        ValueError: If max_tokens is invalid

    Example:
        >>> validate_max_tokens(1000)
    """
    if max_tokens is None:
        return

    if not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive (minimum: 1)")

    if max_tokens > 100000:
        raise ValueError("max_tokens too large (maximum: 100000)")


def validate_temperature(temperature: Optional[float]) -> None:
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

    if not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")

    if temperature < 0 or temperature > 2:
        raise ValueError("temperature must be between 0 and 2")


def validate_top_p(top_p: Optional[float]) -> None:
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


def sanitize_error_response(error_body: Any) -> Dict[str, Any]:
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

    # Only expose safe fields
    return {
        "message": (
            error_body.get("error")
            if isinstance(error_body.get("error"), str)
            else "API request failed"
        ),
        "code": (
            error_body.get("code")
            if isinstance(error_body.get("code"), str)
            else None
        ),
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
