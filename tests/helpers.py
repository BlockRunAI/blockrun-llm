"""
Test utilities and mock builders for BlockRun LLM SDK tests.
"""

import json
import base64
from typing import Dict, Any, Optional
from eth_account import Account

# Test private key (DO NOT use in production)
# This is a well-known test key from Hardhat/Foundry
TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# Test account derived from TEST_PRIVATE_KEY
# Address: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
TEST_ACCOUNT = Account.from_key(TEST_PRIVATE_KEY)

# Test recipient address for payment mocks
TEST_RECIPIENT = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"


def build_payment_required_response(
    amount: str = "1000000",
    recipient: str = TEST_RECIPIENT,
    network: str = "eip155:8453",
    resource: Optional[Dict[str, str]] = None,
) -> str:
    """Build a mock 402 Payment Required response."""
    payment_required = {
        "x402Version": 2,
        "accepts": [
            {
                "scheme": "exact",
                "network": network,
                "amount": amount,
                "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "payTo": recipient,
                "maxTimeoutSeconds": 300,
                "extra": {"name": "USD Coin", "version": "2"},
            }
        ],
        "resource": resource
        or {
            "url": "https://api.blockrun.ai/v1/chat/completions",
            "description": "BlockRun AI API call",
        },
    }

    return base64.b64encode(json.dumps(payment_required).encode()).decode()


def build_chat_response(
    content: str = "This is a test response.",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> Dict[str, Any]:
    """Build a mock successful chat response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_error_response(
    error: str = "Test error message",
    code: str = "test_error",
    include_sensitive: bool = True,
) -> Dict[str, Any]:
    """Build a mock error response."""
    response = {"error": error, "code": code}

    if include_sensitive:
        # These should be filtered out by sanitization
        response.update(
            {
                "internal_stack": "/var/app/handler.py:123",
                "api_key": "secret_key_should_be_filtered",
                "database_url": "postgres://user:pass@host/db",
            }
        )

    return response


def build_models_response() -> Dict[str, Any]:
    """Build a mock models list response."""
    return {
        "data": [
            {
                "id": "openai/gpt-4o",
                "provider": "openai",
                "name": "GPT-4o",
                "inputPrice": 2.5,
                "outputPrice": 10.0,
            },
            {
                "id": "anthropic/claude-sonnet-4.5",
                "provider": "anthropic",
                "name": "Claude Sonnet 4.5",
                "inputPrice": 3.0,
                "outputPrice": 15.0,
            },
            {
                "id": "google/gemini-2.5-flash",
                "provider": "google",
                "name": "Gemini 2.5 Flash",
                "inputPrice": 0.15,
                "outputPrice": 0.6,
            },
        ]
    }


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(
        self,
        status_code: int,
        json_data: Optional[Dict[str, Any]] = None,
        text_data: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self._text_data = text_data or (json.dumps(json_data) if json_data else "")
        self.headers = headers or {}

    def json(self) -> Dict[str, Any]:
        if self._json_data is None:
            raise ValueError("No JSON data available")
        return self._json_data

    @property
    def text(self) -> str:
        return self._text_data
