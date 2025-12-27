"""Unit tests for LLMClient."""

import pytest
from unittest.mock import Mock, patch
from blockrun_llm import LLMClient, APIError, PaymentError
from ..helpers import (
    TEST_PRIVATE_KEY,
    build_chat_response,
    build_error_response,
    build_models_response,
    MockResponse,
)


class TestLLMClientInit:
    def test_init_with_valid_key(self):
        """Should create client with valid private key."""
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        assert client is not None
        assert client.get_wallet_address().startswith("0x")

    def test_init_missing_key(self):
        """Should raise ValueError when private key is missing."""
        with pytest.raises(ValueError, match="Private key required"):
            LLMClient(private_key=None)

    def test_init_invalid_key_format(self):
        """Should raise ValueError for invalid key format."""
        with pytest.raises(ValueError, match="must start with 0x"):
            LLMClient(private_key="invalid")

    def test_init_short_key(self):
        """Should raise ValueError for short key."""
        with pytest.raises(ValueError, match="66 characters"):
            LLMClient(private_key="0x123")

    def test_init_non_hex_key(self):
        """Should raise ValueError for non-hex key."""
        with pytest.raises(ValueError, match="hexadecimal"):
            LLMClient(
                private_key="0xGGGG74bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            )

    def test_default_api_url(self):
        """Should use default API URL."""
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        assert client.api_url == "https://api.blockrun.ai"

    def test_custom_api_url(self):
        """Should accept custom API URL."""
        client = LLMClient(
            private_key=TEST_PRIVATE_KEY, api_url="https://custom.example.com"
        )
        assert client.api_url == "https://custom.example.com"

    def test_invalid_api_url_http(self):
        """Should reject HTTP for non-localhost."""
        with pytest.raises(ValueError, match="HTTPS"):
            LLMClient(private_key=TEST_PRIVATE_KEY, api_url="http://insecure.com")

    def test_allow_localhost_http(self):
        """Should allow HTTP for localhost."""
        client = LLMClient(
            private_key=TEST_PRIVATE_KEY, api_url="http://localhost:3000"
        )
        assert client.api_url == "http://localhost:3000"


class TestLLMClientMethods:
    def test_get_wallet_address(self):
        """Should return valid Ethereum address."""
        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        address = client.get_wallet_address()

        assert address == "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        assert address.startswith("0x")
        assert len(address) == 42

    @patch("blockrun_llm.client.httpx.Client")
    def test_list_models(self, mock_client_class):
        """Should list available models."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = MockResponse(200, build_models_response())
        mock_client.get.return_value = mock_response

        client = LLMClient(private_key=TEST_PRIVATE_KEY)
        models = client.list_models()

        assert len(models) == 3
        assert models[0]["id"] == "openai/gpt-4o"
        assert models[0]["provider"] == "openai"

    @patch("blockrun_llm.client.httpx.Client")
    def test_list_models_error(self, mock_client_class):
        """Should raise APIError on failure."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = MockResponse(500)
        mock_client.get.return_value = mock_response

        client = LLMClient(private_key=TEST_PRIVATE_KEY)

        with pytest.raises(APIError):
            client.list_models()


class TestErrorSanitization:
    @patch("blockrun_llm.client.httpx.Client")
    def test_sanitize_error_responses(self, mock_client_class):
        """Should sanitize error responses."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        raw_error = build_error_response(
            error="Invalid model", include_sensitive=True
        )
        mock_response = MockResponse(400, raw_error)
        mock_client.get.return_value = mock_response

        client = LLMClient(private_key=TEST_PRIVATE_KEY)

        try:
            client.list_models()
            pytest.fail("Should have raised APIError")
        except APIError as e:
            # Should only contain safe fields
            assert e.response == {"message": "Invalid model", "code": "test_error"}

            # Should NOT contain sensitive fields
            assert "internal_stack" not in e.response
            assert "api_key" not in e.response
            assert "database_url" not in e.response


class TestInputValidation:
    @patch("blockrun_llm.client.httpx.Client")
    def test_validate_model_parameter(self, mock_client_class):
        """Should validate model parameter."""
        client = LLMClient(private_key=TEST_PRIVATE_KEY)

        with pytest.raises(ValueError, match="non-empty string"):
            client.chat_completion("", [{"role": "user", "content": "test"}])

    @patch("blockrun_llm.client.httpx.Client")
    def test_validate_max_tokens(self, mock_client_class):
        """Should validate max_tokens parameter."""
        client = LLMClient(private_key=TEST_PRIVATE_KEY)

        with pytest.raises(ValueError, match="positive"):
            client.chat_completion(
                "gpt-4o", [{"role": "user", "content": "test"}], max_tokens=-1
            )

        with pytest.raises(ValueError, match="too large"):
            client.chat_completion(
                "gpt-4o", [{"role": "user", "content": "test"}], max_tokens=200000
            )

    @patch("blockrun_llm.client.httpx.Client")
    def test_validate_temperature(self, mock_client_class):
        """Should validate temperature parameter."""
        client = LLMClient(private_key=TEST_PRIVATE_KEY)

        with pytest.raises(ValueError, match="between 0 and 2"):
            client.chat_completion(
                "gpt-4o", [{"role": "user", "content": "test"}], temperature=3.0
            )

    @patch("blockrun_llm.client.httpx.Client")
    def test_validate_top_p(self, mock_client_class):
        """Should validate top_p parameter."""
        client = LLMClient(private_key=TEST_PRIVATE_KEY)

        with pytest.raises(ValueError, match="between 0 and 1"):
            client.chat_completion(
                "gpt-4o", [{"role": "user", "content": "test"}], top_p=1.5
            )
