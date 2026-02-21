"""
advanced_tools.py — Options Flow Scanner + Portfolio Risk Attribution + NL Screener

1. Options Flow Scanner
   - Scans all tickers for unusual options activity daily
   - Flags large block trades, sweep orders, OTM call buying
   - Smart money signal detector

2. Portfolio Risk Attribution
   - What % of risk comes from each position
   - Correlation matrix (hidden concentration)
   - Factor decomposition of portfolio risk
   - Marginal contribution to risk (MCTR)

3. Natural Language Screener
   - Type what you want in plain English
   - Converts to screening criteria automatically
   - Uses Claude API (free with claude.ai access)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
import json


# ─── OPTIONS FLOW SCANNER ─────────────────────────────────────────────────────

class OptionsFlowScanner:
    """Scan entire watchlist for unusual options activity."""

    def scan(self, tickers: list, min_volume_ratio: float = 3.0) -> list:
        """
        Scan list of tickers for unusual options activity.
        Flags where volume >> open interest (new positions being opened).
        """
        print(f"\n  Scanning options flow for {len(tickers)} tickers...")
        alerts = []

        for ticker in tickers:
            try:
                t    = yf.Ticker(ticker)
                info = t.info or {}
                exps = t.options
                if not exps:
                    continue

                # Use nearest 2 expirations
                for exp in exps[:2]:
                    chain = t.option_chain(exp)
                    price = info.get("regularMarketPrice") or info.get("currentPrice") or 0

                    for opt_type, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                        if df is None or df.empty:
                            continue

                        df = df.copy()
                        avg_vol = df["volume"].mean() if "volume" in df.columns else 0
                        if avg_vol == 0:
                            continue

                        for _, row in df.iterrows():
                            vol = row.get("volume", 0) or 0
                            oi  = row.get("openInterest", 0) or 0
                            strike = row.get("strike", 0)
                            iv   = row.get("impliedVolatility", 0) or 0
                            last = row.get("lastPrice", 0) or 0

                            # Flag unusual activity
                            if vol > avg_vol * min_volume_ratio and vol > 500:
                                dist = (strike - price) / price * 100 if price > 0 else 0
                                notional = vol * last * 100

                                alert_type = "SWEEP" if vol > oi * 2 else "BLOCK"
                                direction  = "BULLISH" if opt_type == "CALL" else "BEARISH"

                                alerts.append({
                                    "ticker":      ticker,
                                    "type":        opt_type,
                                    "expiration":  exp,
                                    "strike":      strike,
                                    "volume":      int(vol),
                                    "open_interest": int(oi),
                                    "vol_oi_ratio": round(vol/max(oi,1), 2),
                                    "iv_pct":      round(iv*100, 1),
                                    "premium":     last,
                                    "notional":    round(notional, 0),
                                    "distance_pct": round(dist, 1),
                                    "alert_type":  alert_type,
                                    "direction":   direction,
                                    "signal":      f"{'🐂' if direction=='BULLISH' else '🐻'} {alert_type}: {ticker} {opt_type} ${strike:.0f} exp {exp} | {vol:,} vol vs {avg_vol:.0f} avg | ${notional:,.0f} notional",
                                })

                time.sleep(0.1)

            except Exception:
                continue

        # Sort by notional (biggest money first)
        alerts.sort(key=lambda x: x.get("notional", 0), reverse=True)
        return alerts[:20]

    def print_flow(self, alerts: list):
        print(f"\n{'='*70}")
        print(f"  OPTIONS FLOW SCANNER — Top {len(alerts)} Unusual Positions")
        print(f"{'='*70}\n")
        if not alerts:
            print("  No unusual activity detected in this universe.")
            return
        for a in alerts:
            print(f"  {a['signal']}")
            print(f"    IV: {a['iv_pct']:.1f}% | Strike {a['distance_pct']:+.1f}% from price | Vol/OI: {a['vol_oi_ratio']:.1f}x\n")


# ─── PORTFOLIO RISK ATTRIBUTION ───────────────────────────────────────────────

class PortfolioRiskAnalyzer:
    """Decompose portfolio risk by position, factor, and correlation."""

    def analyze(self, holdings: dict, period: str = "2y") -> dict:
        """
        holdings = {"AAPL": 0.30, "MSFT": 0.25, "NVDA": 0.20, "BND": 0.25}
        (ticker → weight)
        """
        tickers = list(holdings.keys())
        weights = np.array(list(holdings.values()))
        weights = weights / weights.sum()  # Normalize

        print(f"  Downloading price data for {len(tickers)} holdings...")
        try:
            data = yf.download(tickers + ["SPY"], period=period,
                               progress=False, auto_adjust=True)["Close"]
            data = data.dropna()
        except Exception as e:
            return {"error": str(e)}

        returns = data.pct_change().dropna()

        # Covariance matrix
        cov     = returns[tickers].cov() * 252  # Annualized
        corr    = returns[tickers].corr()

        # Portfolio variance and vol
        w       = weights
        port_var = float(w @ cov.values @ w)
        port_vol = np.sqrt(port_var)

        # Marginal Contribution to Risk (MCTR)
        mctr    = (cov.values @ w) / port_vol  # Per unit weight
        pct_contribution = (mctr * w) / port_vol  # % of total vol

        # Individual vols
        ind_vols = {t: float(returns[t].std() * np.sqrt(252)) for t in tickers if t in returns}

        # Beta to SPY
        betas = {}
        if "SPY" in returns.columns:
            spy_ret = returns["SPY"]
            for t in tickers:
                if t in returns.columns:
                    cov_spy = float(returns[t].cov(spy_ret) * 252)
                    var_spy = float(spy_ret.var() * 252)
                    betas[t] = round(cov_spy / var_spy, 3) if var_spy > 0 else 1.0
        port_beta = sum(w[i] * betas.get(t, 1.0) for i, t in enumerate(tickers))

        # Correlation clusters (hidden concentration)
        high_corr = []
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                c = corr.iloc[i,j]
                if abs(c) > 0.70:
                    high_corr.append({
                        "pair":        f"{tickers[i]}/{tickers[j]}",
                        "correlation": round(float(c), 3),
                        "warning":     "⚠️  High correlation — less diversification than you think",
                    })

        # Concentration risk
        hhi = float(np.sum(weights**2))  # Herfindahl index

        # Return attribution
        position_risk = {}
        for i, t in enumerate(tickers):
            position_risk[t] = {
                "weight":              round(float(w[i]) * 100, 1),
                "individual_vol":      round(ind_vols.get(t, 0) * 100, 1),
                "mctr":               round(float(mctr[i]) * 100, 2),
                "pct_of_portfolio_risk": round(float(pct_contribution[i]) * 100, 1),
                "beta":               betas.get(t, "N/A"),
            }

        # Diversification ratio
        weighted_vol = sum(w[i] * ind_vols.get(t, 0) for i, t in enumerate(tickers))
        div_ratio    = weighted_vol / port_vol if port_vol > 0 else 1.0

        return {
            "portfolio_vol_annual":  round(port_vol * 100, 2),
            "portfolio_beta":        round(port_beta, 3),
            "diversification_ratio": round(div_ratio, 3),
            "herfindahl_index":      round(hhi, 4),
            "concentration_risk":    "High" if hhi > 0.25 else "Moderate" if hhi > 0.15 else "Well Diversified",
            "position_risk":         position_risk,
            "correlation_matrix":    corr.round(3).to_dict(),
            "high_correlations":     high_corr,
            "largest_risk_driver":   max(position_risk.items(), key=lambda x: x[1]["pct_of_portfolio_risk"])[0],
            "summary":               self._summarize(port_vol, port_beta, div_ratio, hhi, high_corr, position_risk),
        }

    def _summarize(self, vol, beta, div_ratio, hhi, high_corr, pos_risk) -> str:
        largest = max(pos_risk.items(), key=lambda x: x[1]["pct_of_portfolio_risk"])
        parts   = [
            f"Portfolio vol: {vol*100:.1f}%/yr",
            f"Beta: {beta:.2f}",
            f"Largest risk driver: {largest[0]} ({largest[1]['pct_of_portfolio_risk']:.0f}% of risk)",
        ]
        if high_corr:
            parts.append(f"{len(high_corr)} high-correlation pairs detected")
        return " | ".join(parts)

    def print_attribution(self, result: dict):
        print(f"\n{'='*62}")
        print(f"  PORTFOLIO RISK ATTRIBUTION")
        print(f"{'='*62}\n")
        print(f"  Annual Volatility:   {result['portfolio_vol_annual']:.1f}%")
        print(f"  Portfolio Beta:      {result['portfolio_beta']:.2f}")
        print(f"  Diversification:     {result['diversification_ratio']:.2f}x  ({result['concentration_risk']})")
        print(f"\n  {'Position':<10} {'Weight':>8} {'Ind.Vol':>8} {'MCTR':>8} {'% of Risk':>10} {'Beta':>8}")
        print("  " + "─"*55)
        for ticker, d in sorted(result["position_risk"].items(),
                                 key=lambda x: x[1]["pct_of_portfolio_risk"], reverse=True):
            bar = "█" * int(d["pct_of_portfolio_risk"] / 5)
            print(f"  {ticker:<10} {d['weight']:>7.1f}% {d['individual_vol']:>7.1f}% "
                  f"{d['mctr']:>7.2f}% {d['pct_of_portfolio_risk']:>9.1f}%  {str(d['beta']):>6}  {bar}")

        if result.get("high_correlations"):
            print(f"\n  ⚠️  High Correlation Pairs:")
            for hc in result["high_correlations"]:
                print(f"    {hc['pair']}: {hc['correlation']:.2f}")


# ─── NATURAL LANGUAGE SCREENER ────────────────────────────────────────────────

class NaturalLanguageScreener:
    """
    Convert plain English queries into screening criteria.
    Uses Claude API (free via Anthropic).
    Fallback: keyword-based parsing if API unavailable.
    """

    PROFILE_KEYWORDS = {
        "aggressive_growth": ["aggressive","high growth","fast growing","explosive","hypergrowth","momentum"],
        "garp":              ["reasonable","moderate","garp","quality growth","solid"],
        "deep_value":        ["cheap","undervalued","value","discount","low pe","bargain"],
        "quality_compounder":["quality","moat","compounder","wide moat","durable","compounding"],
        "dividend_growth":   ["dividend","income","yield","passive income","dividends"],
        "momentum":          ["momentum","trending","hot","breakout","52 week high"],
        "small_cap_growth":  ["small cap","small-cap","micro cap","under the radar","hidden gem"],
    }

    SECTOR_KEYWORDS = {
        "Technology":    ["tech","technology","software","ai","semiconductor","cloud","saas"],
        "Healthcare":    ["health","pharma","biotech","medical","drug","clinical"],
        "Financials":    ["bank","finance","financial","insurance","fintech"],
        "Energy":        ["energy","oil","gas","renewable","solar","wind","ev"],
        "Consumer":      ["consumer","retail","restaurant","brand","e-commerce"],
        "Industrials":   ["industrial","manufacturing","defense","aerospace"],
    }

    def parse(self, query: str) -> dict:
        """Parse natural language into screening criteria."""
        q = query.lower()

        # Try Claude API first
        result = self._try_claude_api(query)
        if result:
            return result

        # Fallback: keyword matching
        return self._keyword_parse(q)

    def _try_claude_api(self, query: str) -> dict:
        """Use Claude API to parse the query."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            prompt = f"""Convert this stock screening request into JSON screening criteria.

User request: "{query}"

Return ONLY a JSON object with these optional fields:
{{
  "screen_profile": "one of: aggressive_growth|garp|deep_value|quality_compounder|dividend_growth|momentum|small_cap_growth",
  "sector_focus": ["list of sectors if mentioned"],
  "min_revenue_growth": 0.XX (decimal),
  "max_pe": number,
  "min_dividend_yield": 0.XX,
  "min_roe": 0.XX,
  "max_debt_equity": number,
  "min_gross_margin": 0.XX,
  "description": "one sentence describing what was requested"
}}

Only include fields that are clearly implied by the request. Return valid JSON only."""

            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            text = msg.content[0].text.strip()
            # Extract JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())

        except Exception:
            pass
        return None

    def _keyword_parse(self, q: str) -> dict:
        """Rule-based fallback parser."""
        result = {}

        # Profile detection
        for profile, keywords in self.PROFILE_KEYWORDS.items():
            if any(k in q for k in keywords):
                result["screen_profile"] = profile
                break

        if "screen_profile" not in result:
            result["screen_profile"] = "garp"

        # Sector detection
        sectors = []
        for sector, keywords in self.SECTOR_KEYWORDS.items():
            if any(k in q for k in keywords):
                sectors.append(sector)
        if sectors:
            result["sector_focus"] = sectors

        # Numeric extraction
        pe_match  = re.search(r'p/?e\s*(?:under|below|<)?\s*(\d+)', q)
        div_match = re.search(r'(?:yield|dividend)\s*(?:over|above|>)?\s*(\d+\.?\d*)%?', q)
        grw_match = re.search(r'(\d+)%?\s*(?:revenue\s*)?growth', q)

        if pe_match:   result["max_pe"]             = int(pe_match.group(1))
        if div_match:  result["min_dividend_yield"]  = float(div_match.group(1)) / 100
        if grw_match:  result["min_revenue_growth"]  = float(grw_match.group(1)) / 100

        # Special cases
        if "insider buying" in q:    result["min_insider_own"] = 0.03
        if "profitable" in q:        result["min_net_margin"]  = 0.05
        if "no debt" in q:           result["max_debt_equity"]  = 0.3
        if "large cap" in q:         result["min_market_cap"]   = 10e9
        if "small cap" in q:         result["max_market_cap"]   = 2e9

        result["description"] = f"Screening for {result.get('screen_profile','garp').replace('_',' ')} stocks"
        if sectors: result["description"] += f" in {', '.join(sectors)}"

        return result

    def run(self, query: str, universe: list = None) -> dict:
        """Full pipeline: parse → screen → return results."""
        from stock_screener import StockScreener

        print(f"\n  🔍 Natural Language Query: \"{query}\"")
        criteria = self.parse(query)
        print(f"  Interpreted as: {criteria.get('description','')}")
        print(f"  Profile: {criteria.get('screen_profile','garp')}")

        screener = StockScreener()
        if universe is None:
            universe = screener.get_sp500_tickers()

        df = screener.fetch_bulk_metrics(universe)

        # Apply sector filter
        sectors = criteria.pop("sector_focus", [])
        profile = criteria.pop("screen_profile", "garp")
        desc    = criteria.pop("description", "")

        if sectors:
            df = df[df["sector"].str.lower().apply(
                lambda s: any(sec.lower() in str(s).lower() for sec in sectors)
            )]

        # Apply custom filters from criteria
        custom_filters = {k: v for k, v in criteria.items()
                         if k not in ["description"]}

        if custom_filters:
            results = screener.custom_screen(df, custom_filters)
        else:
            results = screener.screen(profile, df)

        screener.print_screen_results(results, f'Results for: "{query}"')

        return {
            "query":       query,
            "interpreted": desc,
            "criteria":    criteria,
            "results":     results.to_dict("records") if not results.empty else [],
        }
