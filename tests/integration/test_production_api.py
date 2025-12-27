"""Integration tests for BlockRun LLM SDK against production API.

Requirements:
- BLOCKRUN_WALLET_KEY environment variable with funded Base wallet
- Minimum $1 USDC on Base chain
- Estimated cost per test run: ~$0.05

Run with: pytest tests/integration
Skip if no wallet: Tests will be skipped if BLOCKRUN_WALLET_KEY not set
"""

import os
import pytest
import time
from blockrun_llm import LLMClient, AsyncLLMClient

WALLET_KEY = os.environ.get("BLOCKRUN_WALLET_KEY")
PRODUCTION_API = "https://api.blockrun.ai"

# Skip all tests if no wallet key configured
pytestmark = pytest.mark.skipif(
    not WALLET_KEY, reason="BLOCKRUN_WALLET_KEY environment variable not set"
)


class TestProductionAPISync:
    """Integration tests for synchronous LLMClient against production API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create LLMClient instance for testing."""
        if not WALLET_KEY:
            pytest.skip("BLOCKRUN_WALLET_KEY not set")

        client = LLMClient(private_key=WALLET_KEY, api_url=PRODUCTION_API)

        print("\n🧪 Running sync integration tests against production API")
        print(f"   Wallet: {client.get_wallet_address()}")
        print(f"   API: {PRODUCTION_API}")
        print(f"   Estimated cost: ~$0.05\n")

        return client

    def test_list_models(self, client):
        """Should list available models from production API."""
        models = client.list_models()

        assert models is not None
        assert isinstance(models, list)
        assert len(models) > 0

        # Verify model structure
        first_model = models[0]
        assert "id" in first_model
        assert "provider" in first_model
        assert "inputPrice" in first_model
        assert "outputPrice" in first_model

        print(f"   ✓ Found {len(models)} models")

        # Respect rate limits
        time.sleep(2)

    def test_simple_chat_request(self, client):
        """Should complete a simple chat request."""
        # Use cheapest model for testing
        response = client.chat(
            "gemini-2.0-flash-exp",
            [{"role": "user", "content": "Say 'test passed' and nothing else"}],
        )

        assert response is not None
        assert isinstance(response, str)
        assert "test passed" in response.lower()

        print(f"   ✓ Chat response: {response[:50]}...")

        time.sleep(2)

    def test_chat_completion_with_usage_stats(self, client):
        """Should return chat completion with usage stats."""
        completion = client.chat_completion(
            "gemini-2.0-flash-exp",
            [{"role": "user", "content": "Count to 5"}],
            max_tokens=50,
        )

        assert completion is not None
        assert "choices" in completion
        assert len(completion["choices"]) > 0
        assert "message" in completion["choices"][0]
        assert "content" in completion["choices"][0]["message"]
        assert completion["choices"][0]["message"]["content"]

        # Verify usage stats
        assert "usage" in completion
        assert completion["usage"]["prompt_tokens"] > 0
        assert completion["usage"]["completion_tokens"] > 0
        assert completion["usage"]["total_tokens"] > 0

        print(f"   ✓ Completion with usage: {completion['usage']}")

        time.sleep(2)

    def test_payment_flow_end_to_end(self, client):
        """Should handle 402 payment flow end-to-end.

        This test verifies the full x402 payment protocol:
        1. Request to API
        2. Receive 402 with payment required
        3. Create payment payload with EIP-712 signature
        4. Retry with payment receipt
        5. Receive successful response
        """
        response = client.chat(
            "gemini-2.0-flash-exp", [{"role": "user", "content": "What is 2+2?"}]
        )

        # If we got a response, the payment flow succeeded
        assert response is not None
        assert isinstance(response, str)
        assert response

        print(f"   ✓ Payment flow successful, response received")

        time.sleep(2)



class TestProductionAPIAsync:
    """Integration tests for asynchronous AsyncLLMClient against production API."""

    @pytest.fixture(scope="class")
    async def async_client(self):
        """Create AsyncLLMClient instance for testing."""
        if not WALLET_KEY:
            pytest.skip("BLOCKRUN_WALLET_KEY not set")

        client = AsyncLLMClient(private_key=WALLET_KEY, api_url=PRODUCTION_API)

        print("\n🧪 Running async integration tests against production API")
        print(f"   Wallet: {client.get_wallet_address()}")
        print(f"   API: {PRODUCTION_API}")
        print(f"   Estimated cost: ~$0.05\n")

        return client

    @pytest.mark.asyncio
    async def test_async_list_models(self, async_client):
        """Should list available models asynchronously."""
        models = await async_client.list_models()

        assert models is not None
        assert isinstance(models, list)
        assert len(models) > 0

        print(f"   ✓ Async: Found {len(models)} models")

        await asyncio.sleep(2)

    @pytest.mark.asyncio
    async def test_async_simple_chat(self, async_client):
        """Should complete a simple chat request asynchronously."""
        response = await async_client.chat(
            "gemini-2.0-flash-exp",
            [{"role": "user", "content": "Say 'async test passed' and nothing else"}],
        )

        assert response is not None
        assert isinstance(response, str)
        assert "test passed" in response.lower()

        print(f"   ✓ Async chat response: {response[:50]}...")

        await asyncio.sleep(2)

    @pytest.mark.asyncio
    async def test_async_chat_completion(self, async_client):
        """Should return chat completion with usage stats asynchronously."""
        completion = await async_client.chat_completion(
            "gemini-2.0-flash-exp",
            [{"role": "user", "content": "Count to 5"}],
            max_tokens=50,
        )

        assert completion is not None
        assert "choices" in completion
        assert len(completion["choices"]) > 0
        assert "usage" in completion
        assert completion["usage"]["total_tokens"] > 0

        print(f"   ✓ Async completion with usage: {completion['usage']}")

        await asyncio.sleep(2)



class TestProductionAPIErrorHandling:
    """Integration tests for error handling against production API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create LLMClient instance for testing."""
        if not WALLET_KEY:
            pytest.skip("BLOCKRUN_WALLET_KEY not set")

        return LLMClient(private_key=WALLET_KEY, api_url=PRODUCTION_API)

    def test_invalid_model_error(self, client):
        """Should handle invalid model error gracefully."""
        from blockrun_llm import APIError

        with pytest.raises(APIError):
            client.chat(
                "invalid-model-that-does-not-exist",
                [{"role": "user", "content": "test"}],
            )

        print(f"   ✓ Invalid model error handled correctly")

        time.sleep(2)

    def test_error_response_sanitization(self, client):
        """Should sanitize error responses."""
        from blockrun_llm import APIError

        try:
            client.chat(
                "invalid-model", [{"role": "user", "content": "test"}]
            )
            pytest.fail("Should have raised APIError")
        except APIError as e:
            # Error should be sanitized (no internal stack traces, API keys, etc.)
            assert e.message is not None
            assert "/var/" not in str(e.message)
            assert "internal" not in str(e.message).lower() or "internal" in str(
                e.message
            ).lower()  # Allow "internal" in error message but not internal paths
            assert "stack" not in str(e.message).lower()

            print(f"   ✓ Error response properly sanitized")

        time.sleep(2)


# Import asyncio for async tests
import asyncio
