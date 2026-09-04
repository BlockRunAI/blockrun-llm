"""Select account authentication or the preferred wallet chain."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .api_key import resolve_api_auth

if TYPE_CHECKING:
    from .client import LLMClient
    from .solana_client import SolanaLLMClient


def setup_agent_client(
    *, api_key: str | None = None, chain: str | None = None, silent: bool = False
) -> LLMClient | SolanaLLMClient:
    """Use an API key when configured; otherwise prefer Solana for new users.

    Preserve saved chain selections and existing Base-only wallets. Named
    setup_agent_wallet/setup_agent_solana_wallet helpers remain chain-specific.
    """
    from .client import LLMClient
    from .solana_wallet import load_solana_wallet, setup_agent_solana_wallet
    from .wallet import load_wallet, setup_agent_wallet

    if resolve_api_auth(api_key, None, None):
        return LLMClient(api_key=api_key)
    if chain is not None and chain not in ("solana", "base"):
        raise ValueError("chain must be solana or base")
    if chain is None:
        for filename in ("payment-chain", ".chain"):
            try:
                saved = (Path.home() / ".blockrun" / filename).read_text().strip()
            except OSError:
                continue
            if saved in ("solana", "base"):
                chain = saved
                break
    if chain is None:
        chain = "base" if not load_solana_wallet() and load_wallet() else "solana"
    return (
        setup_agent_solana_wallet(silent=silent)
        if chain == "solana"
        else setup_agent_wallet(silent=silent)
    )
