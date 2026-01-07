"""
BlockRun Integration Example: Crypto Arbitrage Analysis

This example shows how to integrate BlockRun's AI capabilities into
a cryptocurrency arbitrage bot (like Polymarket-Kalshi BTC arbitrage).

Setup:
    pip install blockrun-llm
    export BASE_CHAIN_WALLET_KEY=0x...  # Your Base wallet private key

Usage:
    python arbitrage_analyzer.py
"""

from dataclasses import dataclass
from blockrun_llm import LLMClient, AsyncLLMClient, PaymentError, APIError


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""

    platform_a: str
    platform_b: str
    price_a: float  # e.g., 0.52 (52% probability)
    price_b: float  # e.g., 0.47 (47% probability)
    spread: float  # Combined cost below $1.00
    expiry: str
    market: str  # e.g., "BTC > $100,000"


class ArbitrageAnalyzer:
    """
    AI-powered arbitrage opportunity analyzer using BlockRun.

    Provides risk assessment, market sentiment, and execution recommendations
    for detected arbitrage opportunities.
    """

    # Model recommendations by use case
    MODELS = {
        "fast": "openai/gpt-4o-mini",  # $0.15/M input - quick analysis
        "balanced": "anthropic/claude-haiku-4.5",  # $1.00/M input - good reasoning
        "deep": "anthropic/claude-sonnet-4",  # $3.00/M input - thorough analysis
        "frontier": "openai/gpt-5.2",  # $1.75/M input - latest capabilities
    }

    def __init__(self, model_tier: str = "fast"):
        """
        Initialize the analyzer.

        Args:
            model_tier: One of "fast", "balanced", "deep", "frontier"
        """
        self.client = LLMClient()
        self.model = self.MODELS.get(model_tier, self.MODELS["fast"])

    def analyze_opportunity(self, opp: ArbitrageOpportunity) -> dict:
        """
        Analyze an arbitrage opportunity for risk and execution.

        Args:
            opp: The detected arbitrage opportunity

        Returns:
            Analysis dict with risk_score, recommendation, and reasoning
        """
        prompt = f"""Analyze this prediction market arbitrage opportunity:

Market: {opp.market}
Platform A ({opp.platform_a}): {opp.price_a:.2%} probability
Platform B ({opp.platform_b}): {opp.price_b:.2%} probability
Combined cost: ${opp.spread:.4f} (potential profit: ${1 - opp.spread:.4f})
Expiry: {opp.expiry}

Evaluate:
1. Is this spread large enough to be worth executing after fees?
2. What are the execution risks (slippage, timing, liquidity)?
3. Any concerns about the market or timing?

Provide a risk score (1-10, 10=highest risk) and clear recommendation."""

        try:
            response = self.client.chat(
                self.model,
                prompt,
                system="You are a quantitative trading analyst specializing in prediction market arbitrage. Be concise and actionable.",
            )

            return {
                "success": True,
                "analysis": response,
                "model": self.model,
                "cost_estimate": "~$0.001-0.01",
            }

        except PaymentError as e:
            return {"success": False, "error": f"Payment failed - check USDC balance: {e}"}
        except APIError as e:
            return {"success": False, "error": f"API error: {e}"}

    def get_market_sentiment(self, asset: str = "BTC") -> dict:
        """
        Get AI-powered market sentiment analysis.

        Args:
            asset: The asset to analyze (default: BTC)

        Returns:
            Sentiment analysis dict
        """
        prompt = f"""What is the current market sentiment for {asset}?

Consider:
- Recent price action and trends
- Market structure (support/resistance levels)
- Macro factors affecting crypto
- Any upcoming events that could impact prices

Provide a sentiment score (-100 to +100) and brief reasoning."""

        try:
            response = self.client.chat(
                self.model,
                prompt,
                system="You are a crypto market analyst. Provide objective, data-driven analysis.",
            )

            return {"success": True, "asset": asset, "sentiment": response, "model": self.model}

        except (PaymentError, APIError) as e:
            return {"success": False, "error": str(e)}

    def compare_opportunities(self, opportunities: list[ArbitrageOpportunity]) -> dict:
        """
        Rank multiple opportunities by risk-adjusted return.

        Args:
            opportunities: List of detected opportunities

        Returns:
            Ranked list with recommendations
        """
        opp_descriptions = "\n".join(
            [
                f"{i+1}. {o.market}: {o.platform_a} @ {o.price_a:.2%} vs {o.platform_b} @ {o.price_b:.2%}, "
                f"spread: ${o.spread:.4f}, expires: {o.expiry}"
                for i, o in enumerate(opportunities)
            ]
        )

        prompt = f"""Rank these arbitrage opportunities by risk-adjusted return:

{opp_descriptions}

Consider:
- Profit potential vs execution risk
- Time to expiry
- Liquidity concerns
- Market volatility

Return a ranked list with brief reasoning for each."""

        try:
            response = self.client.chat(
                self.model,
                prompt,
                system="You are a quantitative trading analyst. Rank opportunities objectively.",
            )

            return {
                "success": True,
                "ranking": response,
                "count": len(opportunities),
                "model": self.model,
            }

        except (PaymentError, APIError) as e:
            return {"success": False, "error": str(e)}


class AsyncArbitrageAnalyzer:
    """
    Async version for high-throughput analysis.

    Use this when analyzing multiple opportunities concurrently.
    """

    MODELS = ArbitrageAnalyzer.MODELS

    def __init__(self, model_tier: str = "fast"):
        self.model = self.MODELS.get(model_tier, self.MODELS["fast"])

    async def analyze_batch(self, opportunities: list[ArbitrageOpportunity]) -> list[dict]:
        """
        Analyze multiple opportunities concurrently.

        Args:
            opportunities: List of opportunities to analyze

        Returns:
            List of analysis results
        """
        import asyncio

        async with AsyncLLMClient() as client:
            tasks = []
            for opp in opportunities:
                prompt = f"Quick analysis: {opp.market}, spread ${opp.spread:.4f}, expires {opp.expiry}. Worth it? (Yes/No + 1 sentence)"
                tasks.append(
                    client.chat(
                        self.model,
                        prompt,
                        system="Be extremely concise. Yes/No + one sentence max.",
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            return [
                (
                    {"opportunity": opp, "analysis": r}
                    if isinstance(r, str)
                    else {"opportunity": opp, "error": str(r)}
                )
                for opp, r in zip(opportunities, results)
            ]


# Example usage
if __name__ == "__main__":
    # Create sample opportunity
    opportunity = ArbitrageOpportunity(
        platform_a="Polymarket",
        platform_b="Kalshi",
        price_a=0.52,
        price_b=0.47,
        spread=0.99,  # $0.99 combined cost
        expiry="2024-01-15 17:00 UTC",
        market="BTC > $100,000 by Jan 15",
    )

    # Initialize analyzer (uses BASE_CHAIN_WALLET_KEY from env)
    analyzer = ArbitrageAnalyzer(model_tier="fast")

    print("=" * 60)
    print("BlockRun Arbitrage Analyzer")
    print("=" * 60)
    print(f"Wallet: {analyzer.client.get_wallet_address()}")
    print(f"Model: {analyzer.model}")
    print("=" * 60)

    # Analyze the opportunity
    print("\n1. Analyzing opportunity...")
    result = analyzer.analyze_opportunity(opportunity)

    if result["success"]:
        print(f"\nAnalysis ({result['model']}):")
        print("-" * 40)
        print(result["analysis"])
    else:
        print(f"Error: {result['error']}")

    # Get market sentiment
    print("\n2. Getting BTC sentiment...")
    sentiment = analyzer.get_market_sentiment("BTC")

    if sentiment["success"]:
        print(f"\nSentiment ({sentiment['model']}):")
        print("-" * 40)
        print(sentiment["sentiment"])
    else:
        print(f"Error: {sentiment['error']}")

    print("\n" + "=" * 60)
    print("Cost: ~$0.002-0.02 total (pay-per-request)")
    print("=" * 60)
