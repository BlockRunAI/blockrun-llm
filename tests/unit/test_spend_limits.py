"""Client-side spend limits.

Before 1.9.0 there was no ceiling anywhere: `client.py` computed `cost_usd` and
signed the quote in the next statement, with nothing compared against anything.
`chat_completion` even documented a `PaymentError: If budget is set and would be
exceeded` for a `budget` parameter that did not exist.

The rule these tests encode: when a limit refuses a quote, **no paid request is
ever sent**. Signing alone moves no money — the gateway submitting the signed
authorization does — so a refusal before the send costs the caller nothing.
"""

import httpx
import pytest

from blockrun_llm import LLMClient
from blockrun_llm.types import PaymentError, SpendLimitError
from blockrun_llm.validation import check_spend_limits, resolve_spend_limit

from ..helpers import (
    TEST_PRIVATE_KEY,
    build_chat_response,
    build_payment_required_response,
)

MESSAGES = [{"role": "user", "content": "hi"}]

# build_payment_required_response defaults to amount "1000000" = 1 USDC.
QUOTED_USD = 1.0


def _client(**kwargs):
    """A client whose paid leg fails the test if it is ever reached."""
    signed = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "PAYMENT-SIGNATURE" in request.headers:
            signed.append(request)
            return httpx.Response(200, json=build_chat_response())
        return httpx.Response(
            402,
            json={"error": "Payment Required"},
            headers={"payment-required": build_payment_required_response()},
        )

    client = LLMClient(private_key=TEST_PRIVATE_KEY, **kwargs)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, signed


class TestNoLimitsIsUnchanged:
    def test_default_client_still_pays(self):
        """Limits are opt-in. Omitting them must behave exactly as before."""
        client, signed = _client()
        client.chat_completion("a/b", MESSAGES)
        assert len(signed) == 1

    def test_helper_is_a_noop_without_limits(self):
        check_spend_limits(
            999.0, max_cost_per_call=None, max_session_cost=None, session_spent_usd=0.0
        )


class TestPerCallLimit:
    def test_refuses_over_limit_quote_without_sending(self):
        client, signed = _client(max_cost_per_call=0.10)
        with pytest.raises(SpendLimitError) as exc:
            client.chat_completion("a/b", MESSAGES)
        assert signed == [], "a refused quote must never be sent"
        assert exc.value.scope == "call"
        assert exc.value.quoted_usd == pytest.approx(QUOTED_USD)
        assert exc.value.limit_usd == pytest.approx(0.10)

    def test_allows_quote_at_or_under_limit(self):
        client, signed = _client(max_cost_per_call=QUOTED_USD)
        client.chat_completion("a/b", MESSAGES)
        assert len(signed) == 1, "the limit is inclusive"

    def test_message_names_both_numbers_and_the_model(self):
        client, _ = _client(max_cost_per_call=0.10)
        with pytest.raises(SpendLimitError) as exc:
            client.chat_completion("anthropic/claude-opus-4.8", MESSAGES)
        msg = str(exc.value)
        assert "1.000000" in msg and "0.100000" in msg
        assert "claude-opus-4.8" in msg
        assert "nothing was charged" in msg.lower()

    def test_nothing_is_recorded_as_spent(self):
        client, _ = _client(max_cost_per_call=0.10)
        with pytest.raises(SpendLimitError):
            client.chat_completion("a/b", MESSAGES)
        assert client.get_spending()["total_usd"] == 0.0
        assert client.get_spending()["calls"] == 0


class TestSessionLimit:
    def test_refuses_the_call_that_would_breach_the_total(self):
        client, signed = _client(max_session_cost=1.5)
        client.chat_completion("a/b", MESSAGES)  # 1.0 spent, 0.5 left
        assert len(signed) == 1
        with pytest.raises(SpendLimitError) as exc:
            client.chat_completion("a/b", MESSAGES)  # would reach 2.0
        assert len(signed) == 1, "the second quote must not be sent"
        assert exc.value.scope == "session"

    def test_message_reports_what_is_left(self):
        client, _ = _client(max_session_cost=1.5)
        client.chat_completion("a/b", MESSAGES)
        with pytest.raises(SpendLimitError) as exc:
            client.chat_completion("a/b", MESSAGES)
        assert "0.500000" in str(exc.value)


class TestAsyncClient:
    """The async handler computes its cost_usd only after the paid POST returns,
    so the guard has to read the quote off the 402 instead. Without a test the
    limit silently ran too late to refuse anything."""

    def _run(self, **kwargs):
        from blockrun_llm import AsyncLLMClient

        signed = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "PAYMENT-SIGNATURE" in request.headers:
                signed.append(request)
                return httpx.Response(200, json=build_chat_response())
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={"payment-required": build_payment_required_response()},
            )

        async def go():
            client = AsyncLLMClient(private_key=TEST_PRIVATE_KEY, **kwargs)
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            return await client.chat_completion("a/b", MESSAGES)

        return go, signed

    def test_refuses_over_limit_without_sending(self):
        import asyncio

        go, signed = self._run(max_cost_per_call=0.10)
        with pytest.raises(SpendLimitError):
            asyncio.run(go())
        assert signed == [], "the async path must refuse before the paid POST"

    def test_allows_quote_under_limit(self):
        import asyncio

        go, signed = self._run(max_cost_per_call=QUOTED_USD)
        asyncio.run(go())
        assert len(signed) == 1


class TestErrorContract:
    def test_is_a_payment_error(self):
        """Existing `except PaymentError` handlers must keep working."""
        assert issubclass(SpendLimitError, PaymentError)

    def test_does_not_trigger_model_fallback(self):
        """Falling back to another model after refusing on cost would defeat
        the limit, and would sign a second quote."""
        from blockrun_llm.client import _should_fallback

        exc = SpendLimitError("x", quoted_usd=1.0, limit_usd=0.1, scope="call")
        assert _should_fallback(exc) is False

    def test_fallback_chain_refuses_rather_than_shopping_for_a_cheaper_model(self):
        client, signed = _client(max_cost_per_call=0.10)
        with pytest.raises(SpendLimitError):
            client.chat_completion("a/b", MESSAGES, fallback_models=["c/d", "e/f"])
        assert signed == [], "must not try the next model looking for a cheaper quote"


class TestLimitResolution:
    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_MAX_COST_PER_CALL", "5.0")
        assert resolve_spend_limit(0.25, "BLOCKRUN_MAX_COST_PER_CALL") == 0.25

    def test_env_var_applies_when_no_argument(self, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_MAX_COST_PER_CALL", "0.25")
        assert resolve_spend_limit(None, "BLOCKRUN_MAX_COST_PER_CALL") == 0.25

    def test_env_var_reaches_a_real_client(self, monkeypatch):
        monkeypatch.setenv("BLOCKRUN_MAX_COST_PER_CALL", "0.10")
        client, signed = _client()
        with pytest.raises(SpendLimitError):
            client.chat_completion("a/b", MESSAGES)
        assert signed == []

    def test_malformed_env_is_ignored_not_fatal(self, monkeypatch):
        """A bad env var must not brick every client in a deployment."""
        for bad in ("abc", "", "-1", "0"):
            monkeypatch.setenv("BLOCKRUN_MAX_COST_PER_CALL", bad)
            assert resolve_spend_limit(None, "BLOCKRUN_MAX_COST_PER_CALL") is None

    def test_explicit_non_positive_is_a_programming_error(self):
        with pytest.raises(ValueError, match="positive"):
            resolve_spend_limit(0, "BLOCKRUN_MAX_COST_PER_CALL")
        with pytest.raises(ValueError, match="positive"):
            resolve_spend_limit(-1.0, "BLOCKRUN_MAX_COST_PER_CALL")
