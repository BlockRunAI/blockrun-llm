"""max_tokens validation on the Solana chain (issue #31).

The guard was Base-only: `validate_max_tokens` was called from client.py and
nowhere else, while solana_client.py put the caller's value straight into paid
request bodies. `max_tokens=2_000_000` raised on Base and was signed and sent on
Solana. Since the gateway clamps rather than rejects, there was no server-side
backstop behind the missing client-side one.

Guarded with importorskip: the 3.9 CI job installs without the solana extra, and
an unguarded Solana test file turns that job red (see #19/#20).
"""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

pytest.importorskip("x402")
pytest.importorskip("solders")

from blockrun_llm import SolanaLLMClient
from blockrun_llm.validation import MAX_TOKENS_SANITY_LIMIT

MESSAGES = [{"role": "user", "content": "hi"}]


def _client() -> SolanaLLMClient:
    """A client whose transport fails loudly. Validation must reject before any
    request is built, so a passing test proves nothing reached the network."""

    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"validation let a bad max_tokens reach {request.url}")

    with (
        mock.patch("blockrun_llm.solana_client.register_exact_svm_client"),
        mock.patch("blockrun_llm.solana_client._create_signer"),
    ):
        client = SolanaLLMClient(
            private_key="bogus_signer_is_patched",
            api_url="https://sol.blockrun.ai/api",
            rpc_url="http://test",
        )
    client._x402_client = mock.MagicMock()
    client._client = httpx.Client(transport=httpx.MockTransport(explode))
    client._address = "11111111111111111111111111111111"
    return client


class TestSolanaMaxTokensValidation:
    """Every Solana chat entry point that puts max_tokens in a paid body."""

    def test_chat_completion_rejects_implausible(self):
        with pytest.raises(ValueError, match="implausibly large"):
            _client().chat_completion("a/b", MESSAGES, max_tokens=2_000_000)

    def test_chat_completion_stream_rejects_implausible(self):
        with pytest.raises(ValueError, match="implausibly large"):
            list(_client().chat_completion_stream("a/b", MESSAGES, max_tokens=2_000_000))

    def test_rejects_zero_and_negative(self):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="positive"):
                _client().chat_completion("a/b", MESSAGES, max_tokens=bad)

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="bool"):
            _client().chat_completion("a/b", MESSAGES, max_tokens=True)

    def test_real_ceilings_are_not_capped(self):
        """The bound must never be the binding constraint on either chain.
        These reach the transport, which is what the AssertionError proves."""
        for real_ceiling in (128_000, 262_144, MAX_TOKENS_SANITY_LIMIT):
            with pytest.raises(AssertionError, match="reach"):
                _client().chat_completion("a/b", MESSAGES, max_tokens=real_ceiling)

    def test_both_chains_share_one_bound(self):
        """Base and Solana must agree, or the SDK's guard is not an invariant."""
        from blockrun_llm import LLMClient

        base = LLMClient(private_key="0x" + "11" * 32)
        with pytest.raises(ValueError, match="implausibly large"):
            base.chat_completion("a/b", MESSAGES, max_tokens=2_000_000)
        with pytest.raises(ValueError, match="implausibly large"):
            _client().chat_completion("a/b", MESSAGES, max_tokens=2_000_000)
