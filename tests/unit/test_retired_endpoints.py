"""Helpers for endpoints Predexon retired must fail fast, not silently 410.

Probed upstream 2026-08-04 (3 runs each): /v1/pm/markets, /v1/pm/markets/listings
and /v1/pm/outcomes/{id} all return
    410 "This endpoint has been sunset as of 2026-07-20. Market matching is
        discontinued."
The helpers are kept rather than deleted so upgrading does not break imports;
this pins that they raise instead of quietly costing a round trip.
"""

import pytest

from blockrun_llm import LLMClient, RetiredEndpointError
from blockrun_llm.client import AsyncLLMClient


def _bare(cls):
    """Instance without running __init__ — no wallet or network needed."""
    return cls.__new__(cls)


@pytest.mark.parametrize(
    "method,args",
    [
        ("pm_markets", ()),
        ("pm_listings", ()),
        ("pm_outcome", ("PXM-12345",)),
    ],
)
def test_sync_helpers_raise(method, args):
    with pytest.raises(RetiredEndpointError, match="2026-07-20"):
        getattr(_bare(LLMClient), method)(*args)


@pytest.mark.parametrize(
    "method,args",
    [
        ("pm_markets", ()),
        ("pm_listings", ()),
        ("pm_outcome", ("PXM-12345",)),
    ],
)
@pytest.mark.asyncio
async def test_async_helpers_raise(method, args):
    # An async def raises on await, not on call — but still before any network
    # I/O, which is the point: no paid round trip to learn it is gone.
    with pytest.raises(RetiredEndpointError, match="2026-07-20"):
        await getattr(_bare(AsyncLLMClient), method)(*args)


def test_message_points_at_the_replacement():
    with pytest.raises(RetiredEndpointError) as exc:
        _bare(LLMClient).pm_markets()
    assert "markets/search" in str(exc.value)


def test_surviving_helper_is_untouched():
    # markets/search survived the sunset (422 on a missing q, i.e. alive).
    assert not (getattr(LLMClient, "pm_wallet_identity", None) is None)
