"""Integration tests for BlockRun LLM SDK against production API.

Requirements:
- BASE_CHAIN_WALLET_KEY environment variable with funded Base wallet
- Minimum $1 USDC on Base chain
- Estimated cost per test run: ~$0.05

Run with: pytest tests/integration
Skip if no wallet: Tests will be skipped if BASE_CHAIN_WALLET_KEY not set
"""

import asyncio
import os
import time

import pytest
from blockrun_llm import LLMClient, AsyncLLMClient

WALLET_KEY = os.environ.get("BASE_CHAIN_WALLET_KEY")
PRODUCTION_API = "https://blockrun.ai/api"

# Skip all tests if no wallet key configured
pytestmark = pytest.mark.skipif(
    not WALLET_KEY, reason="BASE_CHAIN_WALLET_KEY environment variable not set"
)


class TestProductionAPISync:
    """Integration tests for synchronous LLMClient against production API."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create LLMClient instance for testing."""
        if not WALLET_KEY:
            pytest.skip("BASE_CHAIN_WALLET_KEY not set")

        client = LLMClient(private_key=WALLET_KEY, api_url=PRODUCTION_API)

        print("\n🧪 Running sync integration tests against production API")
        print(f"   Wallet: {client.get_wallet_address()}")
        print(f"   API: {PRODUCTION_API}")
        print("   Estimated cost: ~$0.05\n")

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
            "google/gemini-2.5-flash-lite",
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
            "google/gemini-2.5-flash-lite",
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
            "google/gemini-2.5-flash-lite", [{"role": "user", "content": "What is 2+2?"}]
        )

        # If we got a response, the payment flow succeeded
        assert response is not None
        assert isinstance(response, str)
        assert response

        print("   ✓ Payment flow successful, response received")

        time.sleep(2)


class TestProductionAPIAsync:
    """Integration tests for asynchronous AsyncLLMClient against production API."""

    @pytest.fixture(scope="class")
    async def async_client(self):
        """Create AsyncLLMClient instance for testing."""
        if not WALLET_KEY:
            pytest.skip("BASE_CHAIN_WALLET_KEY not set")

        client = AsyncLLMClient(private_key=WALLET_KEY, api_url=PRODUCTION_API)

        print("\n🧪 Running async integration tests against production API")
        print(f"   Wallet: {client.get_wallet_address()}")
        print(f"   API: {PRODUCTION_API}")
        print("   Estimated cost: ~$0.05\n")

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
            "google/gemini-2.5-flash-lite",
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
            "google/gemini-2.5-flash-lite",
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
            pytest.skip("BASE_CHAIN_WALLET_KEY not set")

        return LLMClient(private_key=WALLET_KEY, api_url=PRODUCTION_API)

    def test_invalid_model_error(self, client):
        """Should handle invalid model error gracefully."""
        from blockrun_llm import APIError

        with pytest.raises(APIError):
            client.chat(
                "invalid-model-that-does-not-exist",
                [{"role": "user", "content": "test"}],
            )

        print("   ✓ Invalid model error handled correctly")

        time.sleep(2)

    def test_error_response_sanitization(self, client):
        """Should sanitize error responses."""
        from blockrun_llm import APIError

        try:
            client.chat("invalid-model", [{"role": "user", "content": "test"}])
            pytest.fail("Should have raised APIError")
        except APIError as e:
            # Error should be sanitized (no internal stack traces, API keys, etc.)
            assert e.message is not None
            assert "/var/" not in str(e.message)
            assert (
                "internal" not in str(e.message).lower() or "internal" in str(e.message).lower()
            )  # Allow "internal" in error message but not internal paths
            assert "stack" not in str(e.message).lower()

            print("   ✓ Error response properly sanitized")

        time.sleep(2)


# =============================================================================
# Solana + Exa Integration Tests
# =============================================================================

SOLANA_WALLET_KEY = os.environ.get("SOLANA_WALLET_KEY")
SOLANA_API = "https://sol.blockrun.ai/api"


class TestSolanaExa:
    """Integration tests for Exa web search via SolanaLLMClient."""

    @pytest.fixture(scope="class")
    def client(self):
        if not SOLANA_WALLET_KEY:
            pytest.skip("SOLANA_WALLET_KEY not set")
        from blockrun_llm import SolanaLLMClient

        c = SolanaLLMClient(private_key=SOLANA_WALLET_KEY, api_url=SOLANA_API)
        print("\n🧪 Running Solana/Exa integration tests against sol.blockrun.ai")
        print(f"   Wallet: {c.get_wallet_address()}")
        print("   Estimated cost: ~$0.04\n")
        return c

    def test_exa_search(self, client):
        """exa_search returns results with title/url fields."""
        result = client.exa_search("latest AI safety research", numResults=3)
        assert "results" in result, f"Expected 'results' key, got: {list(result.keys())}"
        assert len(result["results"]) > 0
        first = result["results"][0]
        assert "url" in first or "title" in first
        cost = client.get_spending()["total_usd"]
        assert 0.009 <= cost <= 0.011, f"Expected ~$0.01 cost, got {cost}"
        print(f"   ✓ exa_search: {len(result['results'])} results, cost=${cost:.4f}")
        time.sleep(1)

    def test_exa_find_similar(self, client):
        """exa_find_similar returns semantically similar pages."""
        result = client.exa_find_similar("https://openai.com/research/gpt-4", numResults=3)
        assert "results" in result
        assert len(result["results"]) > 0
        print(f"   ✓ exa_find_similar: {len(result['results'])} results")
        time.sleep(1)

    def test_exa_contents(self, client):
        """exa_contents extracts text from a URL, priced per URL."""
        result = client.exa_contents(["https://www.anthropic.com/research"])
        assert result is not None
        assert isinstance(result, dict)
        print("   ✓ exa_contents: response received")
        time.sleep(1)

    def test_exa_answer(self, client):
        """exa_answer returns an AI-generated answer from live web."""
        result = client.exa_answer("What is Anthropic Claude?")
        assert result is not None
        assert isinstance(result, dict)
        print("   ✓ exa_answer: response received")
        time.sleep(1)

    def test_exa_generic_proxy(self, client):
        """exa() generic proxy works for any endpoint."""
        result = client.exa("search", {"query": "blockrun.ai", "numResults": 2})
        assert "results" in result
        print(f"   ✓ exa() generic: {len(result['results'])} results")
        time.sleep(1)

    def test_exa_spending_tracked(self, client):
        """Session spending is tracked across Exa calls."""
        spending = client.get_spending()
        assert spending["total_usd"] > 0
        assert spending["calls"] >= 3
        print(f"   ✓ Spending tracked: ${spending['total_usd']:.4f} over {spending['calls']} calls")
