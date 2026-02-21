"""
crypto_analyzer.py — Crypto market analysis using CoinGecko (free, no API key).

Covers:
- Price & market data
- Technical analysis (RSI, MACD, Bollinger Bands)
- Market dominance & ranking
- On-chain metrics proxy
- Sentiment from market context
- NVT ratio approximation
- Monte Carlo simulation for crypto
"""

import numpy as np
import pandas as pd
from data_fetcher import DataFetcher
from technical_analysis import TechnicalAnalyzer
from simulations import SimulationEngine


class CryptoAnalyzer:

    def __init__(self):
        self.fetcher = DataFetcher()

    # ─── PUBLIC ────────────────────────────────────────────────────────────────

    def full_analysis(self, coin_id: str) -> dict:
        """
        Full crypto analysis.
        coin_id should be CoinGecko ID, e.g. 'bitcoin', 'ethereum', 'solana'
        """
        print(f"  Fetching CoinGecko data for {coin_id}...")
        data = self.fetcher.get_crypto_data(coin_id, days=365)

        if not data:
            return {"error": f"Could not fetch data for {coin_id}"}

        results = {
            "coin_id": coin_id,
        }

        # Market info
        info = data.get("info", {})
        market_data = info.get("market_data", {})
        results["market"] = self._parse_market_data(info, market_data)

        # Technical analysis from price history
        if "prices" in data:
            results["technical"] = self._technical_on_prices(data["prices"])
            results["simulations"] = self._simulate(data["prices"])
        else:
            results["technical"] = {}
            results["simulations"] = {}

        # Market context
        if "market_context" in data:
            results["market_context"] = self._market_context(coin_id, data["market_context"])

        # Fundamental metrics
        results["fundamentals"] = self._fundamentals(info, market_data)

        # Overall score
        results["score"] = self._score(results)
        results["signal"] = self._signal(results["score"])

        return results

    # ─── MARKET DATA ───────────────────────────────────────────────────────────

    def _parse_market_data(self, info: dict, md: dict) -> dict:
        m = {}
        m["name"]            = info.get("name", "Unknown")
        m["symbol"]          = info.get("symbol", "").upper()
        m["rank"]            = info.get("market_cap_rank")
        m["current_price"]   = md.get("current_price", {}).get("usd")
        m["market_cap"]      = md.get("market_cap", {}).get("usd")
        m["total_volume"]    = md.get("total_volume", {}).get("usd")
        m["high_24h"]        = md.get("high_24h", {}).get("usd")
        m["low_24h"]         = md.get("low_24h", {}).get("usd")
        m["price_change_24h"] = md.get("price_change_percentage_24h")
        m["price_change_7d"]  = md.get("price_change_percentage_7d")
        m["price_change_30d"] = md.get("price_change_percentage_30d")
        m["ath"]              = md.get("ath", {}).get("usd")
        m["ath_change_pct"]   = md.get("ath_change_percentage", {}).get("usd")
        m["atl"]              = md.get("atl", {}).get("usd")
        m["circulating_supply"] = md.get("circulating_supply")
        m["total_supply"]     = md.get("total_supply")
        m["max_supply"]       = md.get("max_supply")

        # Market cap tier
        cap = m["market_cap"] or 0
        if cap > 100e9:      m["tier"] = "Mega Cap (>$100B)"
        elif cap > 10e9:     m["tier"] = "Large Cap ($10B-$100B)"
        elif cap > 1e9:      m["tier"] = "Mid Cap ($1B-$10B)"
        elif cap > 100e6:    m["tier"] = "Small Cap ($100M-$1B)"
        else:                m["tier"] = "Micro Cap (<$100M)"

        return m

    # ─── TECHNICAL ─────────────────────────────────────────────────────────────

    def _technical_on_prices(self, prices_df: pd.DataFrame) -> dict:
        """Run technical analysis on the crypto price history."""
        # Build OHLCV-style DataFrame from CoinGecko price data
        df = prices_df[["price"]].copy()
        df.columns = ["Close"]
        df["Open"]   = df["Close"].shift(1)
        df["High"]   = df["Close"].rolling(2).max()
        df["Low"]    = df["Close"].rolling(2).min()
        df["Volume"] = 1e6  # volume not available from simple price endpoint

        df = df.dropna()
        if df.empty:
            return {}

        analyzer = TechnicalAnalyzer(df)
        return analyzer.full_analysis()

    # ─── SIMULATIONS ───────────────────────────────────────────────────────────

    def _simulate(self, prices_df: pd.DataFrame) -> dict:
        """Monte Carlo simulation for crypto."""
        df = prices_df[["price"]].copy()
        df.columns = ["Close"]
        df["High"]   = df["Close"]
        df["Low"]    = df["Close"]
        df["Open"]   = df["Close"].shift(1)
        df["Volume"] = 1e6
        df = df.dropna()

        sim = SimulationEngine(df, {})
        mc  = sim.monte_carlo_simulation()
        cagr = sim.cagr_projection()
        return {"monte_carlo": mc, "cagr_projection": cagr}

    # ─── MARKET CONTEXT ────────────────────────────────────────────────────────

    def _market_context(self, coin_id: str, market_list: list) -> dict:
        """Assess coin's position relative to top 20 cryptos."""
        total_market_cap = sum(c.get("market_cap", 0) or 0 for c in market_list)
        coin_data = next((c for c in market_list if c.get("id") == coin_id), None)

        if not coin_data or total_market_cap == 0:
            return {}

        coin_cap = coin_data.get("market_cap", 0) or 0
        dominance = coin_cap / total_market_cap * 100

        return {
            "dominance_in_top20": round(dominance, 2),
            "rank_in_top20": coin_data.get("market_cap_rank"),
            "24h_volume_to_mcap": round(
                (coin_data.get("total_volume", 0) or 0) / max(coin_cap, 1), 4
            ),
            "total_top20_market_cap": round(total_market_cap / 1e9, 2),
        }

    # ─── FUNDAMENTALS ──────────────────────────────────────────────────────────

    def _fundamentals(self, info: dict, md: dict) -> dict:
        f = {}

        # Developer activity (from CoinGecko)
        dev = info.get("developer_data", {})
        f["github_stars"]     = dev.get("stars")
        f["github_forks"]     = dev.get("forks")
        f["commit_activity"]  = dev.get("commit_count_4_weeks")
        f["pull_requests"]    = dev.get("pull_requests_merged")

        # Community
        comm = info.get("community_data", {})
        f["twitter_followers"] = comm.get("twitter_followers")
        f["reddit_subscribers"] = comm.get("reddit_subscribers")

        # Supply inflation proxy
        circ = md.get("circulating_supply") or 0
        total = md.get("total_supply") or 0
        if total > 0 and circ > 0:
            f["supply_issued_pct"] = round(circ / total * 100, 2)

        # Sentiment score from CoinGecko
        f["sentiment_up_pct"] = info.get("sentiment_votes_up_percentage")
        f["sentiment_dn_pct"] = info.get("sentiment_votes_down_percentage")

        return f

    # ─── SCORING ───────────────────────────────────────────────────────────────

    def _score(self, results: dict) -> float:
        score = 50.0

        market = results.get("market", {})
        tech   = results.get("technical", {})

        # Rank bonus
        rank = market.get("rank") or 999
        if rank <= 5:    score += 15
        elif rank <= 20: score += 10
        elif rank <= 50: score += 5
        elif rank > 200: score -= 5

        # Price momentum
        ch30 = market.get("price_change_30d") or 0
        if ch30 > 20:      score += 10
        elif ch30 > 5:     score += 5
        elif ch30 < -30:   score -= 10
        elif ch30 < -10:   score -= 5

        # Technical score
        tech_s = tech.get("score", 50)
        score = score * 0.6 + tech_s * 0.4

        # ATH distance (avoid extreme FOMO)
        ath_chg = market.get("ath_change_pct") or 0
        if -20 < ath_chg < -5:  score += 5   # slight discount from ATH = opportunity
        elif ath_chg > -5:      score -= 5   # near ATH = risk
        elif ath_chg < -80:     score -= 5   # very far from ATH = fallen knife risk

        # Developer activity
        commits = results.get("fundamentals", {}).get("commit_activity") or 0
        if commits > 50:  score += 5
        elif commits < 5: score -= 5

        return round(min(max(score, 0), 100), 2)

    def _signal(self, score: float) -> str:
        if score >= 72:   return "STRONG BUY"
        elif score >= 60: return "BUY"
        elif score >= 45: return "HOLD"
        elif score >= 35: return "WEAK"
        else:             return "AVOID"

    # ─── PRINT ─────────────────────────────────────────────────────────────────

    def print_summary(self, results: dict):
        market = results.get("market", {})
        tech   = results.get("technical", {})
        fund   = results.get("fundamentals", {})
        mc     = results.get("simulations", {}).get("monte_carlo", {})

        print(f"\n  --- CRYPTO MARKET DATA ---")
        print(f"  Name:            {market.get('name')} ({market.get('symbol')})")
        print(f"  Rank:            #{market.get('rank')}")
        print(f"  Tier:            {market.get('tier')}")
        cap = market.get("market_cap") or 0
        print(f"  Market Cap:      ${cap/1e9:.2f}B" if cap else "  Market Cap:      N/A")
        price = market.get("current_price")
        print(f"  Price:           ${price:,.2f}" if price else "  Price:           N/A")
        print(f"  24h Change:      {market.get('price_change_24h', 0):.2f}%")
        print(f"  7d Change:       {market.get('price_change_7d', 0):.2f}%")
        print(f"  30d Change:      {market.get('price_change_30d', 0):.2f}%")
        ath = market.get("ath")
        ath_chg = market.get("ath_change_pct")
        if ath:
            print(f"  ATH:             ${ath:,.2f} ({ath_chg:.1f}% from ATH)")

        print(f"\n  --- TECHNICAL SIGNALS ---")
        for k, v in tech.get("signals", {}).items():
            print(f"  {k}: {v}")

        print(f"\n  --- DEVELOPER ACTIVITY ---")
        print(f"  GitHub Stars:    {fund.get('github_stars', 'N/A')}")
        print(f"  Commits (4wk):   {fund.get('commit_activity', 'N/A')}")
        print(f"  Twitter:         {fund.get('twitter_followers', 'N/A')}")

        if mc:
            print(f"\n  --- MONTE CARLO (1-Year) ---")
            print(f"  Median Return:   {mc.get('median_return_1y', 0)*100:.1f}%")
            print(f"  Bull Case:       {mc.get('bull_case', 0)*100:.1f}%")
            print(f"  Bear Case:       {mc.get('bear_case', 0)*100:.1f}%")
            print(f"  Prob Profitable: {mc.get('probability_profit', 0)*100:.0f}%")

        print(f"\n  ═══════════════════════════════")
        print(f"  SCORE:  {results.get('score', 0):.1f}/100")
        print(f"  SIGNAL: {results.get('signal')}")
        print(f"  ═══════════════════════════════\n")
