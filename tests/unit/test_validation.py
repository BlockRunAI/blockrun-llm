"""Unit tests for validation module."""

import pytest
from blockrun_llm.validation import (
    validate_private_key,
    validate_api_url,
    validate_model,
    validate_max_tokens,
    validate_temperature,
    validate_top_p,
    sanitize_error_response,
    validate_resource_url,
)


class TestValidatePrivateKey:
    def test_valid_private_key(self):
        """Should accept valid private key."""
        key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        validate_private_key(key)  # Should not raise

    def test_reject_non_string(self):
        """Should reject non-string input."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_private_key(123)  # type: ignore

    def test_reject_no_prefix(self):
        """Should reject key without 0x prefix."""
        with pytest.raises(ValueError, match="must start with 0x"):
            validate_private_key(
                "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            )

    def test_reject_short_key(self):
        """Should reject short key."""
        with pytest.raises(ValueError, match="66 characters"):
            validate_private_key("0x123")

    def test_reject_long_key(self):
        """Should reject long key."""
        with pytest.raises(ValueError, match="66 characters"):
            validate_private_key(
                "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80123"
            )

    def test_reject_non_hex(self):
        """Should reject non-hexadecimal characters."""
        with pytest.raises(ValueError, match="hexadecimal"):
            validate_private_key(
                "0xGGGG74bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            )

    def test_accept_uppercase(self):
        """Should accept uppercase hex."""
        key = "0xAC0974BEC39A17E36BA4A6B4D238FF944BACB478CBED5EFCAE784D7BF4F2FF80"
        validate_private_key(key)  # Should not raise

    def test_accept_mixed_case(self):
        """Should accept mixed case hex."""
        key = "0xAc0974Bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        validate_private_key(key)  # Should not raise


class TestValidateApiUrl:
    def test_accept_https(self):
        """Should accept HTTPS URLs."""
        validate_api_url("https://api.blockrun.ai")
        validate_api_url("https://example.com:8443")

    def test_accept_localhost_http(self):
        """Should accept localhost HTTP."""
        validate_api_url("http://localhost")
        validate_api_url("http://localhost:3000")
        validate_api_url("http://127.0.0.1")
        validate_api_url("http://127.0.0.1:8080")

    def test_reject_http_production(self):
        """Should reject HTTP for non-localhost."""
        with pytest.raises(ValueError, match="HTTPS"):
            validate_api_url("http://api.example.com")
        with pytest.raises(ValueError, match="HTTPS"):
            validate_api_url("http://192.168.1.1")

    def test_reject_invalid_url(self):
        """Should reject invalid URL format."""
        with pytest.raises(ValueError, match="Invalid"):
            validate_api_url("not-a-url")
        with pytest.raises(ValueError, match="Invalid"):
            validate_api_url("")


class TestValidateModel:
    def test_accept_valid_model(self):
        """Should accept valid model IDs."""
        validate_model("openai/gpt-4o")
        validate_model("anthropic/claude-sonnet-4.5")
        validate_model("google/gemini-2.5-flash")

    def test_reject_empty_string(self):
        """Should reject empty string."""
        with pytest.raises(ValueError, match="non-empty string"):
            validate_model("")

    def test_reject_non_string(self):
        """Should reject non-string."""
        with pytest.raises(ValueError, match="non-empty string"):
            validate_model(None)  # type: ignore


class TestValidateMaxTokens:
    def test_accept_valid_values(self):
        """Should accept valid max_tokens."""
        validate_max_tokens(1)
        validate_max_tokens(100)
        validate_max_tokens(1000)
        validate_max_tokens(100000)

    def test_accept_none(self):
        """Should accept None."""
        validate_max_tokens(None)

    def test_reject_negative(self):
        """Should reject negative values."""
        with pytest.raises(ValueError, match="positive"):
            validate_max_tokens(-1)

    def test_reject_zero(self):
        """Should reject zero."""
        with pytest.raises(ValueError, match="positive"):
            validate_max_tokens(0)

    def test_reject_too_large(self):
        """Should reject values too large."""
        with pytest.raises(ValueError, match="too large"):
            validate_max_tokens(200000)

    def test_reject_non_integer(self):
        """Should reject non-integer."""
        with pytest.raises(ValueError, match="integer"):
            validate_max_tokens(100.5)  # type: ignore


class TestValidateTemperature:
    def test_accept_valid_values(self):
        """Should accept valid temperature."""
        validate_temperature(0.0)
        validate_temperature(0.7)
        validate_temperature(1.0)
        validate_temperature(2.0)

    def test_accept_none(self):
        """Should accept None."""
        validate_temperature(None)

    def test_reject_negative(self):
        """Should reject negative values."""
        with pytest.raises(ValueError, match="between 0 and 2"):
            validate_temperature(-0.1)

    def test_reject_too_large(self):
        """Should reject values > 2."""
        with pytest.raises(ValueError, match="between 0 and 2"):
            validate_temperature(2.1)

    def test_reject_non_number(self):
        """Should reject non-numeric."""
        with pytest.raises(ValueError, match="number"):
            validate_temperature("0.7")  # type: ignore


class TestValidateTopP:
    def test_accept_valid_values(self):
        """Should accept valid top_p."""
        validate_top_p(0.0)
        validate_top_p(0.5)
        validate_top_p(0.9)
        validate_top_p(1.0)

    def test_accept_none(self):
        """Should accept None."""
        validate_top_p(None)

    def test_reject_negative(self):
        """Should reject negative values."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_top_p(-0.1)

    def test_reject_too_large(self):
        """Should reject values > 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_top_p(1.1)

    def test_reject_non_number(self):
        """Should reject non-numeric."""
        with pytest.raises(ValueError, match="number"):
            validate_top_p("0.9")  # type: ignore


class TestSanitizeErrorResponse:
    def test_extract_safe_fields(self):
        """Should extract only safe error fields."""
        result = sanitize_error_response(
            {
                "error": "User-facing error",
                "internal_stack": "/var/app/sensitive.py:123",
                "api_key": "sk-secret",
                "database_url": "postgres://user:pass@host/db",
            }
        )
        assert result == {"message": "User-facing error", "code": None}

    def test_include_code_if_present(self):
        """Should include code if present."""
        result = sanitize_error_response(
            {"error": "Invalid request", "code": "invalid_request_error"}
        )
        assert result == {"message": "Invalid request", "code": "invalid_request_error"}

    def test_handle_non_dict(self):
        """Should handle non-dict input."""
        assert sanitize_error_response("error") == {
            "message": "API request failed",
            "code": None,
        }
        assert sanitize_error_response(None) == {
            "message": "API request failed",
            "code": None,
        }
        assert sanitize_error_response(123) == {
            "message": "API request failed",
            "code": None,
        }

    def test_handle_missing_error_field(self):
        """Should handle missing error field."""
        result = sanitize_error_response({"something": "else"})
        assert result == {"message": "API request failed", "code": None}


class TestValidateResourceUrl:
    def test_allow_matching_domain(self):
        """Should allow matching domain."""
        result = validate_resource_url(
            "https://api.blockrun.ai/v1/chat", "https://api.blockrun.ai"
        )
        assert result == "https://api.blockrun.ai/v1/chat"

    def test_allow_different_path(self):
        """Should allow different path on same domain."""
        result = validate_resource_url(
            "https://api.blockrun.ai/v2/models", "https://api.blockrun.ai"
        )
        assert result == "https://api.blockrun.ai/v2/models"

    def test_reject_different_domain(self):
        """Should reject different domain."""
        result = validate_resource_url(
            "https://malicious.com/steal", "https://api.blockrun.ai"
        )
        assert result == "https://api.blockrun.ai/v1/chat/completions"

    def test_reject_different_protocol(self):
        """Should reject different protocol."""
        result = validate_resource_url(
            "http://api.blockrun.ai/v1/chat", "https://api.blockrun.ai"
        )
        assert result == "https://api.blockrun.ai/v1/chat/completions"

    def test_handle_invalid_url(self):
        """Should handle invalid URL format."""
        result = validate_resource_url("not-a-url", "https://api.blockrun.ai")
        assert result == "https://api.blockrun.ai/v1/chat/completions"

    def test_reject_subdomain_difference(self):
        """Should reject subdomain differences."""
        result = validate_resource_url(
            "https://evil.api.blockrun.ai/v1/chat", "https://api.blockrun.ai"
        )
        assert result == "https://api.blockrun.ai/v1/chat/completions"
