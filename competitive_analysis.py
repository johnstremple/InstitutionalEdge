"""
competitive_analysis.py — Industry positioning and competitive intelligence.

Covers:
- Peer comparison across all key metrics
- Relative valuation (is this cheap vs peers?)
- Market share proxy analysis
- Moat scoring (switching costs, network effects, cost advantage, intangibles)
- Industry growth tailwinds/headwinds
- Competitive positioning score
"""

import yfinance as yf
import pandas as pd
import numpy as np


# Sector peer maps with competitive context
SECTOR_PEERS = {
    "Technology": {
        "Semiconductors": {
            "peers": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TSM", "MRVL"],
            "moat_factors": ["IP/Patents", "Fab capacity", "R&D spend", "Node leadership"],
            "key_driver": "AI compute demand, data center capex",
        },
        "Computer Hardware": {
            "peers": ["SMCI", "HPE", "DELL", "HPQ", "NTAP"],
            "moat_factors": ["Supply chain", "Customization", "Partnerships"],
            "key_driver": "AI server demand, enterprise IT refresh",
        },
        "Software—Application": {
            "peers": ["MSFT", "ORCL", "SAP", "CRM", "NOW", "ADBE"],
            "moat_factors": ["Switching costs", "Ecosystem lock-in", "Data network effects"],
            "key_driver": "Cloud migration, AI integration, SaaS penetration",
        },
        "Software—Infrastructure": {
            "peers": ["MSFT", "AMZN", "GOOGL", "SNOW", "DDOG", "MDB"],
            "moat_factors": ["Scale", "API ecosystem", "Developer adoption"],
            "key_driver": "Cloud spend, AI workloads, digital transformation",
        },
        "Internet Content & Information": {
            "peers": ["GOOGL", "META", "SNAP", "PINS", "RDDT"],
            "moat_factors": ["Network effects", "Ad targeting data", "User habit"],
            "key_driver": "Digital ad spend, AI search, social commerce",
        },
    },
    "Consumer Cyclical": {
        "Auto Manufacturers": {
            "peers": ["TSLA", "GM", "F", "RIVN", "NIO", "LCID"],
            "moat_factors": ["Brand", "Charging network", "Software OTA", "Manufacturing scale"],
            "key_driver": "EV adoption rate, interest rates, China demand",
        },
        "Specialty Retail": {
            "peers": ["AMZN", "WMT", "TGT", "COST", "HD"],
            "moat_factors": ["Scale", "Logistics", "Private label", "Membership"],
            "key_driver": "Consumer spending, e-commerce share shift",
        },
    },
    "Healthcare": {
        "Drug Manufacturers": {
            "peers": ["LLY", "NVO", "PFE", "MRK", "ABBV", "BMY", "AMGN"],
            "moat_factors": ["Patent pipeline", "FDA approvals", "R&D productivity"],
            "key_driver": "GLP-1 demand, oncology pipeline, biosimilar pressure",
        },
        "Healthcare Plans": {
            "peers": ["UNH", "CVS", "CI", "HUM", "ELV"],
            "moat_factors": ["Scale", "Vertical integration", "Medicare contracts"],
            "key_driver": "ACA enrollment, Medicare Advantage margins",
        },
    },
    "Financial Services": {
        "Banks—Diversified": {
            "peers": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
            "moat_factors": ["Scale", "Capital", "Relationships", "Brand trust"],
            "key_driver": "Interest rates, loan growth, credit quality",
        },
    },
    "Communication Services": {
        "Internet Content & Information": {
            "peers": ["GOOGL", "META", "NFLX", "SPOT", "DIS"],
            "moat_factors": ["Content", "Algorithm", "Ad data", "Subscription base"],
            "key_driver": "Ad market, streaming subscribers, AI integration",
        },
    },
    "Energy": {
        "Oil & Gas E&P": {
            "peers": ["XOM", "CVX", "COP", "EOG", "PXD", "DVN"],
            "moat_factors": ["Reserve quality", "Cost per barrel", "Balance sheet"],
            "key_driver": "Oil price, OPEC policy, renewable transition timeline",
        },
    },
}


class CompetitiveAnalyzer:

    def __init__(self, ticker: str, info: dict):
        self.ticker = ticker.upper()
        self.info = info or {}
        self.sector = info.get("sector", "")
        self.industry = info.get("industry", "")

    def full_analysis(self) -> dict:
        peers = self._get_peers()
        if not peers:
            return {
                "score": 50,
                "peer_comparison": {},
                "moat_score": self._moat_score(),
                "competitive_position": "Insufficient peer data",
                "industry_context": self._industry_context(),
            }

        peer_data = self._fetch_peer_data(peers)
        comparison = self._compare_to_peers(peer_data)
        moat = self._moat_score()
        position = self._competitive_position(comparison, moat)
        score = self._competitive_score(comparison, moat)

        return {
            "score": score,
            "peers_analyzed": peers,
            "peer_comparison": comparison,
            "relative_valuation": self._relative_valuation(peer_data),
            "moat_score": moat,
            "competitive_position": position,
            "industry_context": self._industry_context(),
            "key_advantages": self._key_advantages(comparison),
            "key_risks": self._competitive_risks(comparison),
        }

    def _get_peers(self) -> list:
        """Find peer tickers for this stock's industry."""
        # Try exact industry match first
        sector_data = SECTOR_PEERS.get(self.sector, {})
        for industry_key, data in sector_data.items():
            if industry_key.lower() in self.industry.lower() or self.industry.lower() in industry_key.lower():
                peers = [p for p in data["peers"] if p != self.ticker]
                return peers[:5]

        # Fallback: use first industry in sector
        if sector_data:
            first = list(sector_data.values())[0]
            return [p for p in first["peers"] if p != self.ticker][:5]

        # Generic fallback
        return []

    def _fetch_peer_data(self, peers: list) -> dict:
        """Fetch key metrics for all peers + the target stock."""
        all_tickers = [self.ticker] + peers
        data = {}

        for ticker in all_tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                data[ticker] = {
                    "name": info.get("shortName", ticker),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                    "forward_pe": info.get("forwardPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "gross_margin": info.get("grossMargins"),
                    "net_margin": info.get("profitMargins"),
                    "roe": info.get("returnOnEquity"),
                    "debt_equity": (info.get("debtToEquity") or 0) / 100,
                    "current_ratio": info.get("currentRatio"),
                    "beta": info.get("beta"),
                    "analyst_target": info.get("targetMeanPrice"),
                    "analyst_high": info.get("targetHighPrice"),
                    "analyst_low": info.get("targetLowPrice"),
                    "recommendation": info.get("recommendationKey", "").upper(),
                    "num_analysts": info.get("numberOfAnalystOpinions"),
                    "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                }
            except Exception:
                pass

        return data

    def _compare_to_peers(self, peer_data: dict) -> dict:
        """Rank target stock vs peers on each metric."""
        if self.ticker not in peer_data:
            return {}

        metrics = ["pe_ratio", "pb_ratio", "ev_ebitda", "revenue_growth",
                   "earnings_growth", "gross_margin", "net_margin", "roe"]

        comparison = {}
        target = peer_data[self.ticker]

        for metric in metrics:
            values = {}
            for ticker, data in peer_data.items():
                v = data.get(metric)
                if v is not None:
                    values[ticker] = v

            if len(values) < 2 or self.ticker not in values:
                continue

            target_val = values[self.ticker]
            all_vals = list(values.values())
            sorted_tickers = sorted(values.keys(),
                                    key=lambda t: values[t],
                                    reverse=metric in ["revenue_growth", "earnings_growth",
                                                       "gross_margin", "net_margin", "roe"])

            rank = sorted_tickers.index(self.ticker) + 1
            total = len(sorted_tickers)
            percentile = (total - rank) / (total - 1) * 100 if total > 1 else 50

            peer_avg = np.mean([v for t, v in values.items() if t != self.ticker])
            peer_median = np.median([v for t, v in values.items() if t != self.ticker])

            comparison[metric] = {
                "target_value": round(target_val, 4),
                "peer_average": round(peer_avg, 4),
                "peer_median": round(peer_median, 4),
                "rank": rank,
                "total_peers": total,
                "percentile": round(percentile, 1),
                "vs_peer_avg": round((target_val - peer_avg) / abs(peer_avg) * 100, 1) if peer_avg != 0 else 0,
                "assessment": self._assess_metric(metric, target_val, peer_avg),
            }

        return comparison

    def _assess_metric(self, metric: str, value: float, peer_avg: float) -> str:
        """Qualitative assessment of how target compares to peers."""
        ratio = value / peer_avg if peer_avg != 0 else 1

        # For growth and profitability: higher is better
        if metric in ["revenue_growth", "earnings_growth", "gross_margin", "net_margin", "roe"]:
            if ratio > 1.5:   return "Best-in-class"
            elif ratio > 1.2: return "Above average"
            elif ratio > 0.8: return "In line with peers"
            elif ratio > 0.5: return "Below average"
            else:             return "Lagging peers"

        # For valuation: lower is generally cheaper (better)
        elif metric in ["pe_ratio", "pb_ratio", "ev_ebitda", "ps_ratio"]:
            if ratio < 0.7:   return "Cheap vs peers"
            elif ratio < 0.9: return "Slight discount to peers"
            elif ratio < 1.1: return "In line with peers"
            elif ratio < 1.4: return "Premium to peers"
            else:             return "Significant premium"

        return "N/A"

    def _relative_valuation(self, peer_data: dict) -> dict:
        """Is this stock cheap or expensive relative to peers?"""
        if self.ticker not in peer_data:
            return {}

        target = peer_data[self.ticker]
        peers_only = {k: v for k, v in peer_data.items() if k != self.ticker}

        result = {}

        for metric in ["pe_ratio", "ev_ebitda", "ps_ratio"]:
            target_val = target.get(metric)
            peer_vals = [v.get(metric) for v in peers_only.values() if v.get(metric)]
            if not target_val or not peer_vals:
                continue
            peer_med = np.median(peer_vals)
            discount = (target_val - peer_med) / peer_med * 100
            result[metric] = {
                "target": round(target_val, 2),
                "peer_median": round(peer_med, 2),
                "premium_discount_pct": round(discount, 1),
                "verdict": f"{abs(discount):.0f}% {'premium' if discount > 0 else 'discount'} to peers",
            }

        # Overall relative valuation verdict
        premiums = [v["premium_discount_pct"] for v in result.values()]
        if premiums:
            avg_premium = np.mean(premiums)
            if avg_premium > 30:
                result["overall"] = "Significantly Overvalued vs Peers"
            elif avg_premium > 10:
                result["overall"] = "Moderately Overvalued vs Peers"
            elif avg_premium > -10:
                result["overall"] = "Fairly Valued vs Peers"
            elif avg_premium > -30:
                result["overall"] = "Moderately Undervalued vs Peers"
            else:
                result["overall"] = "Significantly Undervalued vs Peers"

        return result

    def _moat_score(self) -> dict:
        """Score economic moat based on available financial indicators."""
        info = self.info
        moat_score = 0
        moat_signals = []

        # 1. Pricing power (gross margins)
        gm = info.get("grossMargins") or 0
        if gm > 0.60:
            moat_score += 20
            moat_signals.append(f"Very high gross margins ({gm*100:.0f}%) — strong pricing power")
        elif gm > 0.40:
            moat_score += 12
            moat_signals.append(f"Healthy gross margins ({gm*100:.0f}%)")
        elif gm > 0.25:
            moat_score += 6

        # 2. Capital efficiency (ROIC proxy via ROE)
        roe = info.get("returnOnEquity") or 0
        if roe > 0.30:
            moat_score += 20
            moat_signals.append(f"Exceptional ROE ({roe*100:.0f}%) — high-quality compounding")
        elif roe > 0.15:
            moat_score += 12
            moat_signals.append(f"Strong ROE ({roe*100:.0f}%)")
        elif roe > 0.08:
            moat_score += 6

        # 3. Revenue consistency (growth + stability)
        rev_g = info.get("revenueGrowth") or 0
        if rev_g > 0.20:
            moat_score += 15
            moat_signals.append(f"Rapid revenue growth ({rev_g*100:.0f}%) — market share gains")
        elif rev_g > 0.08:
            moat_score += 10
        elif rev_g > 0:
            moat_score += 5

        # 4. Low debt (financial durability)
        de = (info.get("debtToEquity") or 0) / 100
        if de < 0.3:
            moat_score += 15
            moat_signals.append("Very low leverage — durable balance sheet")
        elif de < 0.8:
            moat_score += 8
        elif de > 2.0:
            moat_score -= 5

        # 5. Scale / market cap
        mktcap = info.get("marketCap") or 0
        if mktcap > 500e9:
            moat_score += 15
            moat_signals.append("Mega-cap scale — network effects and cost advantages")
        elif mktcap > 50e9:
            moat_score += 8
        elif mktcap > 10e9:
            moat_score += 4

        # 6. Intangibles proxy — R&D intensity
        rev = info.get("totalRevenue") or 1
        rd = info.get("researchDevelopment") or 0
        rd_intensity = rd / rev
        if rd_intensity > 0.15:
            moat_score += 15
            moat_signals.append(f"High R&D intensity ({rd_intensity*100:.0f}% of revenue) — IP moat")
        elif rd_intensity > 0.05:
            moat_score += 8

        moat_score = min(moat_score, 100)

        if moat_score >= 75:   moat_width = "Wide Moat"
        elif moat_score >= 50: moat_width = "Narrow Moat"
        elif moat_score >= 30: moat_width = "Uncertain Moat"
        else:                  moat_width = "No Moat"

        return {
            "score": moat_score,
            "width": moat_width,
            "signals": moat_signals,
        }

    def _competitive_score(self, comparison: dict, moat: dict) -> float:
        score = 50.0

        # Score from peer comparisons
        for metric, data in comparison.items():
            assessment = data.get("assessment", "")
            if assessment == "Best-in-class":     score += 6
            elif assessment == "Above average":   score += 3
            elif assessment == "Below average":   score -= 3
            elif assessment == "Lagging peers":   score -= 6
            elif assessment == "Cheap vs peers":  score += 4
            elif assessment == "Significant premium": score -= 3

        # Moat bonus
        score += moat["score"] * 0.15

        return round(min(max(score, 0), 100), 2)

    def _competitive_position(self, comparison: dict, moat: dict) -> str:
        above = sum(1 for m in comparison.values()
                    if m.get("assessment") in ["Best-in-class", "Above average"])
        below = sum(1 for m in comparison.values()
                    if m.get("assessment") in ["Below average", "Lagging peers"])

        moat_w = moat.get("width", "Uncertain Moat")

        if above >= 4 and moat_w in ["Wide Moat", "Narrow Moat"]:
            return "Industry Leader"
        elif above >= 3:
            return "Above Average Competitor"
        elif below >= 4:
            return "Weak Competitive Position"
        elif below >= 2:
            return "Average / Challenged"
        else:
            return "Competitive — Mixed Profile"

    def _industry_context(self) -> dict:
        """Return key industry drivers and context."""
        sector_data = SECTOR_PEERS.get(self.sector, {})
        for industry_key, data in sector_data.items():
            if industry_key.lower() in self.industry.lower() or self.industry.lower() in industry_key.lower():
                return {
                    "key_driver": data.get("key_driver", "N/A"),
                    "moat_factors": data.get("moat_factors", []),
                    "industry": industry_key,
                }
        return {"key_driver": "N/A", "moat_factors": [], "industry": self.industry}

    def _key_advantages(self, comparison: dict) -> list:
        return [
            f"{metric.replace('_', ' ').title()}: {data['assessment']} (vs peer avg {data['peer_average']:.2f})"
            for metric, data in comparison.items()
            if data.get("assessment") in ["Best-in-class", "Above average", "Cheap vs peers"]
        ]

    def _competitive_risks(self, comparison: dict) -> list:
        return [
            f"{metric.replace('_', ' ').title()}: {data['assessment']} (vs peer avg {data['peer_average']:.2f})"
            for metric, data in comparison.items()
            if data.get("assessment") in ["Below average", "Lagging peers", "Significant premium"]
        ]
