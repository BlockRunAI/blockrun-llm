"""Tests for the settled-payment boundary and the gateway clamp warning.

Both mechanisms exist to protect money, and both were shipped without coverage.
The rule they encode: signing is settlement, so exactly one PAYMENT-SIGNATURE
leaves the process per user-initiated call, no matter how the paid leg fails.
"""

import httpx
import pytest

from blockrun_llm import LLMClient
from blockrun_llm.client import (
    _SETTLED_ATTR,
    _mark_settled,
    _should_fallback,
    _warn_if_clamped,
)
from blockrun_llm.types import APIError, PaymentError

from ..helpers import (
    TEST_PRIVATE_KEY,
    build_chat_response,
    build_payment_required_response,
)


def _client(handler) -> LLMClient:
    client = LLMClient(private_key=TEST_PRIVATE_KEY)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


class TestSettledTagClassification:
    def test_untagged_timeout_still_falls_back(self):
        """An unpaid timeout is a genuine transient failure; keep retrying."""
        assert _should_fallback(httpx.ReadTimeout("boom")) is True

    def test_untagged_503_still_falls_back(self):
        assert _should_fallback(APIError("upstream", 503, None)) is True

    def test_settled_timeout_does_not_fall_back(self):
        assert _should_fallback(_mark_settled(httpx.ReadTimeout("boom"))) is False

    def test_settled_network_error_does_not_fall_back(self):
        assert _should_fallback(_mark_settled(httpx.ConnectError("boom"))) is False

    def test_settled_5xx_does_not_fall_back(self):
        """The dominant post-settlement failure. Tagging only timeouts left
        this open, so the six-settlement path survived the first fix."""
        assert _should_fallback(_mark_settled(APIError("upstream", 503, None))) is False

    def test_mark_settled_preserves_identity_and_type(self):
        exc = httpx.ReadTimeout("boom")
        assert _mark_settled(exc) is exc
        with pytest.raises(httpx.TimeoutException):
            raise exc


class TestTagScope:
    """What must NOT be tagged, driven through the real client. The handlers
    were once `except Exception`, which labeled rejected payments and SDK bugs
    as settled payments. These fail if the handlers widen again."""

    def test_payment_rejection_is_not_tagged_as_settled(self):
        """A paid-leg 402 means the facilitator refused; the funds did not
        move. Calling that 'settled' is exactly backwards."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={"payment-required": build_payment_required_response()},
            )

        client = _client(handler)
        with pytest.raises(PaymentError) as exc:
            client.chat_completion("a/b", [{"role": "user", "content": "hi"}])
        assert getattr(exc.value, _SETTLED_ATTR, False) is False

    def test_programming_error_in_paid_leg_is_not_tagged(self, monkeypatch):
        """An AttributeError raised after the paid response is an SDK bug. It
        must propagate as itself, not as a settled-payment failure."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "PAYMENT-SIGNATURE" in request.headers:
                return httpx.Response(200, json=build_chat_response())
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={"payment-required": build_payment_required_response()},
            )

        client = _client(handler)

        def boom(_response):
            raise AttributeError("SDK bug in settlement capture")

        monkeypatch.setattr(client, "_capture_settlement", boom)

        with pytest.raises(AttributeError) as exc:
            client.chat_completion("a/b", [{"role": "user", "content": "hi"}])
        assert getattr(exc.value, _SETTLED_ATTR, False) is False

    def test_tagged_types_are_a_superset_of_fallback_eligible(self):
        """The handlers catch `(httpx.HTTPError, APIError)`. That must cover
        everything _should_fallback says yes to, or a settled failure escapes
        untagged and the chain pays again."""
        assert issubclass(httpx.TimeoutException, httpx.HTTPError)
        assert issubclass(httpx.NetworkError, httpx.HTTPError)
        assert not issubclass(PaymentError, APIError)

    def test_tagging_preserves_traceback_and_context(self):
        """Handlers re-raise bare rather than `from None`, so an opaque wrapper
        error keeps the cause that explains it."""
        try:
            try:
                raise ValueError("underlying base64 failure")
            except ValueError:
                raise APIError("invalid format", 500, None)
        except APIError as outer:
            _mark_settled(outer)
            assert isinstance(outer.__context__, ValueError)
            assert outer.__suppress_context__ is False


class TestNoSecondSettlement:
    """One user-initiated call must never settle more than once."""

    def _paid_leg_fails(self, failure):
        # Count distinct signatures, not signed requests. The paid leg retries
        # 502/503 once with the SAME PAYMENT-SIGNATURE, which is one settlement
        # replayed, not a second charge. A new signature is a new settlement.
        signed = set()

        def handler(request: httpx.Request) -> httpx.Response:
            if "PAYMENT-SIGNATURE" in request.headers:
                signed.add(request.headers["PAYMENT-SIGNATURE"])
                return failure()
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={"payment-required": build_payment_required_response()},
            )

        return signed, handler

    def test_timeout_after_payment_does_not_pay_the_next_model(self):
        def fail():
            raise httpx.ReadTimeout("upstream hung after settlement")

        signed, handler = self._paid_leg_fails(fail)
        client = _client(handler)

        with pytest.raises(httpx.TimeoutException):
            client.chat_completion(
                "primary/slow",
                [{"role": "user", "content": "hi"}],
                fallback_models=["fallback/good", "fallback/other"],
            )
        assert len(signed) == 1, f"settled {len(signed)} times for one call"

    def test_paid_5xx_does_not_pay_the_next_model(self, monkeypatch):
        """Regression: a 402-then-503 chain across 3 models signed six payments
        and returned nothing."""
        monkeypatch.setattr("time.sleep", lambda _s: None)
        signed, handler = self._paid_leg_fails(
            lambda: httpx.Response(503, json={"error": "upstream down"})
        )
        client = _client(handler)

        with pytest.raises(APIError):
            client.chat_completion(
                "a/b",
                [{"role": "user", "content": "hi"}],
                fallback_models=["c/d", "e/f"],
            )
        assert len(signed) == 1, f"settled {len(signed)} times for one call"

    def test_unpaid_failure_still_walks_the_chain(self):
        """The guard must not disable legitimate free retries: if nothing was
        signed, falling back costs the caller nothing."""
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            model = _json.loads(request.read())["model"]
            seen.append(model)
            if model == "primary/bad":
                return httpx.Response(503, json={"error": "down"})
            return httpx.Response(200, json=build_chat_response())

        client = _client(handler)
        client.chat_completion(
            "primary/bad",
            [{"role": "user", "content": "hi"}],
            fallback_models=["fallback/good"],
        )
        # primary appears twice: the unpaid leg retries 502/503 once before the
        # chain advances. What matters is that it advanced at all.
        assert "fallback/good" in seen


class TestPaidStreamCleanup:
    """Extracting the paid phase into its own generator changed who is
    responsible for closing it."""

    def test_abandoned_async_paid_stream_closes_its_inner_generator(self):
        """Drives the real AsyncLLMClient. `async for` does not aclose the inner
        generator when the outer is closed, so the paid
        `async with self._client.stream(...)` would stay suspended and hold the
        connection until GC finalization. Fails if the explicit
        `finally: await paid.aclose()` is removed.
        """
        import asyncio

        from blockrun_llm import AsyncLLMClient

        closed = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={"payment-required": build_payment_required_response()},
            )

        async def run():
            client = AsyncLLMClient(private_key=TEST_PRIVATE_KEY)
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

            async def fake_paid_phase(*_a, **_kw):
                try:
                    yield {"chunk": 1}
                    yield {"chunk": 2}
                finally:
                    closed.append(True)

            client._astream_paid_phase = fake_paid_phase

            stream = client.chat_completion_stream("a/b", [{"role": "user", "content": "hi"}])
            assert await stream.__anext__() == {"chunk": 1}
            # Abandon mid-stream, exactly like `break` in a caller's loop.
            await stream.aclose()
            # Assert HERE, not after asyncio.run(). Without the explicit
            # aclose, CPython's asyncgen finalizer still closes the inner
            # generator eventually during loop teardown — which is precisely
            # the "connection held until GC" behavior being fixed. Only a
            # synchronous check at the close point tells the two apart.
            return list(closed)

        assert asyncio.run(run()) == [True], "inner paid generator was not closed at aclose()"

    def test_sync_delegation_closes_via_yield_from(self):
        """The sync path gets this for free, which is why only the async path
        needed the explicit close. Recorded so nobody 'fixes' it symmetrically."""
        closed = []

        def inner():
            try:
                yield 1
                yield 2
            finally:
                closed.append(True)

        def outer():
            yield from inner()

        gen = outer()
        assert next(gen) == 1
        gen.close()
        assert closed == [True]


class TestWarnIfClamped:
    def test_warns_when_quoted_below_requested(self, capsys):
        _warn_if_clamped(
            {"model": "claude-opus-4.8", "max_tokens": 262144},
            "claude-opus-4.8 chat completion, 128000 max output tokens",
        )
        err = capsys.readouterr().err
        assert "clamped" in err and "262144" in err and "128000" in err

    def test_parses_comma_grouped_ceiling(self, capsys):
        _warn_if_clamped({"max_tokens": 200000}, "gpt-5.5, 128,000 max output tokens")
        assert "128000" in capsys.readouterr().err

    def test_silent_when_quoted_meets_the_request(self, capsys):
        _warn_if_clamped({"max_tokens": 128000}, "128000 max output tokens")
        _warn_if_clamped({"max_tokens": 1000}, "128000 max output tokens")
        assert capsys.readouterr().err == ""

    def test_silent_when_no_ceiling_in_description(self, capsys):
        _warn_if_clamped({"max_tokens": 999999}, "BlockRun AI API call")
        _warn_if_clamped({"max_tokens": 999999}, None)
        _warn_if_clamped({"max_tokens": 999999}, "")
        assert capsys.readouterr().err == ""

    def test_bool_is_not_a_token_count(self, capsys):
        _warn_if_clamped({"max_tokens": True}, "1 max output tokens")
        assert capsys.readouterr().err == ""

    def test_non_string_description_does_not_raise(self, capsys):
        """The field is server-controlled and JSON allows anything. A warning
        must never be the reason a paid request fails."""
        _warn_if_clamped({"max_tokens": 100}, {"nested": "128 max output tokens"})
        _warn_if_clamped({"max_tokens": 100}, 12345)
        assert capsys.readouterr().err == ""

    def test_ambiguous_description_stays_silent(self, capsys):
        """A per-unit rate is not a ceiling. Two candidates means the format is
        not what we think it is, so say nothing."""
        _warn_if_clamped(
            {"max_tokens": 4096, "model": "m"},
            "$0.002 per 1000 max output tokens, 128000 max output tokens",
        )
        assert capsys.readouterr().err == ""

    def test_long_description_does_not_hang(self, capsys):
        """Outer defense: the scan limit caps what reaches the regex at all."""
        import time

        start = time.perf_counter()
        _warn_if_clamped({"max_tokens": 100}, "9" * 100_000)
        assert time.perf_counter() - start < 1.0
        assert capsys.readouterr().err == ""

    def test_pattern_itself_is_not_backtracking(self):
        """Inner defense, pinned separately so removing the scan limit cannot
        silently reintroduce the ReDoS. The old `(\\d[\\d,]*)` pattern took
        ~1.95s on this input; the bounded one takes ~0.0002s.
        """
        import time

        from blockrun_llm.client import _QUOTED_MAX_TOKENS_RE

        start = time.perf_counter()
        _QUOTED_MAX_TOKENS_RE.search("9" * 16_000)
        assert time.perf_counter() - start < 0.05

    def test_pattern_still_matches_the_real_shapes(self):
        from blockrun_llm.client import _QUOTED_MAX_TOKENS_RE

        for text, expected in (
            ("claude-opus-4.8 - 128000 max output tokens", "128000"),
            ("gpt-5.5, 128,000 max output tokens", "128,000"),
            ("262144 MAX OUTPUT TOKENS", "262144"),
        ):
            assert _QUOTED_MAX_TOKENS_RE.findall(text) == [expected], text

    def test_warning_fires_on_the_real_402_leg(self, capsys):
        def handler(request: httpx.Request) -> httpx.Response:
            if "PAYMENT-SIGNATURE" in request.headers:
                return httpx.Response(200, json=build_chat_response())
            return httpx.Response(
                402,
                json={"error": "Payment Required"},
                headers={
                    "payment-required": build_payment_required_response(
                        resource={
                            "url": "https://blockrun.ai/api/v1/chat/completions",
                            "description": "claude-opus-4.8 - 128000 max output tokens",
                        }
                    )
                },
            )

        _client(handler).chat_completion(
            "claude-opus-4.8",
            [{"role": "user", "content": "hi"}],
            max_tokens=262144,
        )
        assert "max_tokens clamped" in capsys.readouterr().err
