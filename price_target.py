"""
price_target.py — 12-Month Price Target Engine

Methodology (institutional multi-model approach):
1. Analyst Consensus Target (from yfinance)
2. DCF-based target
3. Monte Carlo median (momentum-adjusted)
4. Relative valuation target (peer P/E re-rating)
5. Technical price target (measured move / trend projection)

Final target = weighted blend of all available models.
Output includes: price target, upside/downside, confidence interval,
conviction level, and a plain-English investment thesis.
"""

import numpy as np
import pandas as pd
from typing import Optional


class PriceTargetEngine:

    # Model weights (must sum to 1.0)
    MODEL_WEIGHTS = {
        "analyst_consensus": 0.30,   # Wall Street consensus
        "dcf":               0.25,   # Fundamental intrinsic value
        "monte_carlo":       0.20,   # Statistical momentum
        "relative_val":      0.15,   # Peer-based re-rating
        "technical":         0.10,   # Chart-based target
    }

    def __init__(self, ticker: str, current_price: float, info: dict,
                 simulations: dict, technical: dict, competitive: dict):
        self.ticker         = ticker
        self.current_price  = current_price
        self.info           = info or {}
        self.simulations    = simulations or {}
        self.technical      = technical or {}
        self.competitive    = competitive or {}

    def generate(self) -> dict:
        models = {}

        # 1. Analyst consensus
        models["analyst_consensus"] = self._analyst_target()

        # 2. DCF target
        models["dcf"] = self._dcf_target()

        # 3. Monte Carlo target
        models["monte_carlo"] = self._mc_target()

        # 4. Relative valuation target
        models["relative_val"] = self._relative_val_target()

        # 5. Technical target
        models["technical"] = self._technical_target()

        # Weighted blend
        final = self._blend(models)

        # Confidence interval
        ci = self._confidence_interval(models, final)

        # Conviction
        conviction = self._conviction(models, final)

        # Investment thesis
        thesis = self._build_thesis(final, models, conviction)

        # Catalysts
        catalysts = self._identify_catalysts()

        # Risks
        risks = self._identify_risks()

        return {
            "current_price":      round(self.current_price, 2),
            "price_target_12m":   round(final, 2),
            "upside_downside":    round((final - self.current_price) / self.current_price * 100, 1),
            "confidence_low":     round(ci["low"], 2),
            "confidence_high":    round(ci["high"], 2),
            "conviction":         conviction,
            "models":             models,
            "thesis":             thesis,
            "catalysts":          catalysts,
            "risks":              risks,
        }

    # ─── INDIVIDUAL MODELS ─────────────────────────────────────────────────────

    def _analyst_target(self) -> dict:
        """Wall Street analyst consensus."""
        target     = self.info.get("targetMeanPrice")
        high       = self.info.get("targetHighPrice")
        low        = self.info.get("targetLowPrice")
        n          = self.info.get("numberOfAnalystOpinions") or 0
        rec        = (self.info.get("recommendationKey") or "").upper()

        if not target:
            return {"price": None, "confidence": 0, "source": "Analyst Consensus", "note": "No analyst data"}

        upside = (target - self.current_price) / self.current_price * 100

        return {
            "price":      round(target, 2),
            "high":       round(high, 2) if high else None,
            "low":        round(low, 2)  if low  else None,
            "upside_pct": round(upside, 1),
            "n_analysts": n,
            "recommendation": rec,
            "confidence": min(n * 4, 100),   # More analysts = higher confidence
            "source": "Wall Street Analyst Consensus",
        }

    def _dcf_target(self) -> dict:
        """DCF-derived 12-month target (adjusted for growth momentum)."""
        dcf = self.simulations.get("dcf", {})
        intrinsic = dcf.get("intrinsic_value")
        if not intrinsic:
            return {"price": None, "confidence": 0, "source": "DCF", "note": "DCF not available"}

        # Blend current price and intrinsic value — market won't fully re-rate in 1 year
        # 40% weight to intrinsic, 60% to price momentum toward intrinsic
        rev_growth = self.info.get("revenueGrowth") or 0
        growth_premium = 1 + min(rev_growth * 0.5, 0.30)   # High growth gets partial benefit of doubt

        adjusted_intrinsic = intrinsic * growth_premium
        # 1-year convergence: assume 30% of gap closes in 12 months
        gap = adjusted_intrinsic - self.current_price
        target = self.current_price + gap * 0.30

        upside = (target - self.current_price) / self.current_price * 100

        return {
            "price":      round(target, 2),
            "upside_pct": round(upside, 1),
            "intrinsic_value": round(adjusted_intrinsic, 2),
            "wacc": dcf.get("wacc"),
            "status": dcf.get("valuation_status"),
            "confidence": 60,
            "source": "DCF Valuation (1-yr convergence)",
        }

    def _mc_target(self) -> dict:
        """Monte Carlo median 12-month price."""
        mc = self.simulations.get("monte_carlo", {})
        median_price = mc.get("median_price_1y")
        median_ret   = mc.get("median_return_1y", 0)
        prob_profit  = mc.get("probability_profit", 0.5)
        ann_vol      = mc.get("annualized_vol", 0.3)

        if not median_price:
            return {"price": None, "confidence": 0, "source": "Monte Carlo"}

        # Confidence based on probability of profit and volatility
        conf = int(prob_profit * 80 * (1 - min(ann_vol, 0.8)))

        return {
            "price":      round(median_price, 2),
            "upside_pct": round(median_ret * 100, 1),
            "prob_profit": round(prob_profit * 100, 0),
            "annual_vol":  round(ann_vol * 100, 1),
            "confidence":  conf,
            "source": "Monte Carlo (GBM, 10k paths)",
        }

    def _relative_val_target(self) -> dict:
        """Peer-based price target using median sector P/E re-rating."""
        comp = self.competitive
        rel_val = comp.get("relative_valuation", {})
        peer_comp = comp.get("peer_comparison", {})

        if not peer_comp:
            return {"price": None, "confidence": 0, "source": "Relative Valuation"}

        # If stock is at a discount to peers on P/E, target = peer median P/E * EPS
        pe_data = peer_comp.get("pe_ratio", {})
        target_pe = pe_data.get("target_value")
        peer_med_pe = pe_data.get("peer_median")

        if not target_pe or not peer_med_pe:
            return {"price": None, "confidence": 0, "source": "Relative Valuation"}

        # Price if re-rated to peer median P/E
        pe_ratio = peer_med_pe / target_pe
        target_price = self.current_price * pe_ratio
        upside = (target_price - self.current_price) / self.current_price * 100

        assessment = rel_val.get("overall", "Fairly Valued vs Peers")

        return {
            "price":       round(target_price, 2),
            "upside_pct":  round(upside, 1),
            "current_pe":  round(target_pe, 1),
            "peer_median_pe": round(peer_med_pe, 1),
            "assessment":  assessment,
            "confidence":  55,
            "source": "Peer P/E Re-rating",
        }

    def _technical_target(self) -> dict:
        """Technical price target based on trend and momentum."""
        tech = self.technical
        current = self.current_price

        sma200 = tech.get("sma_200")
        sma50  = tech.get("sma_50")
        rsi    = tech.get("rsi", 50)
        resistance = tech.get("resistance")
        support    = tech.get("support")
        ann_vol    = tech.get("atr")   # ATR as volatility proxy

        # Base: project trend
        if sma50 and sma200:
            if sma50 > sma200:
                # Uptrend: target = current + (current - sma200) * trend factor
                trend_target = current + (current - sma200) * 0.5
            else:
                # Downtrend: more conservative
                trend_target = current + (current - sma200) * 0.2
        else:
            trend_target = current * 1.08   # default 8% base

        # RSI adjustment
        if rsi < 35:
            trend_target *= 1.05   # Oversold bounce potential
        elif rsi > 70:
            trend_target *= 0.97   # Overbought, slight pullback expected

        # Resistance cap
        if resistance and trend_target > resistance * 1.1:
            trend_target = resistance * 1.05

        upside = (trend_target - current) / current * 100

        return {
            "price":       round(trend_target, 2),
            "upside_pct":  round(upside, 1),
            "support":     round(support, 2) if support else None,
            "resistance":  round(resistance, 2) if resistance else None,
            "rsi":         round(rsi, 1) if rsi else None,
            "confidence":  45,
            "source": "Technical Trend Projection",
        }

    # ─── BLEND ─────────────────────────────────────────────────────────────────

    def _blend(self, models: dict) -> float:
        """Weighted average of all valid model targets."""
        total_weight = 0.0
        weighted_sum = 0.0

        for model_name, weight in self.MODEL_WEIGHTS.items():
            model = models.get(model_name, {})
            price = model.get("price")
            if price and price > 0:
                conf = model.get("confidence", 50) / 100
                effective_weight = weight * conf
                weighted_sum  += price * effective_weight
                total_weight  += effective_weight

        if total_weight == 0:
            return self.current_price * 1.10

        return weighted_sum / total_weight

    # ─── CONFIDENCE INTERVAL ───────────────────────────────────────────────────

    def _confidence_interval(self, models: dict, target: float) -> dict:
        """68% confidence interval around the price target."""
        mc = self.simulations.get("monte_carlo", {})
        ann_vol = mc.get("annualized_vol", 0.30)

        # Use 1-sigma band
        low  = target * (1 - ann_vol * 0.7)
        high = target * (1 + ann_vol * 0.7)

        # Anchor to analyst range if available
        analyst = models.get("analyst_consensus", {})
        if analyst.get("low") and analyst.get("high"):
            low  = min(low,  analyst["low"]  * 0.95)
            high = max(high, analyst["high"] * 1.05)

        return {"low": low, "high": high}

    # ─── CONVICTION ────────────────────────────────────────────────────────────

    def _conviction(self, models: dict, target: float) -> str:
        """How aligned are the models?"""
        prices = [m["price"] for m in models.values()
                  if m.get("price") and m["price"] > 0]
        if len(prices) < 2:
            return "Low Conviction"

        std = np.std(prices)
        mean = np.mean(prices)
        cv = std / mean  # coefficient of variation

        upside = (target - self.current_price) / self.current_price

        if cv < 0.10 and upside > 0.10:
            return "High Conviction BUY"
        elif cv < 0.10 and upside < -0.10:
            return "High Conviction SELL"
        elif cv < 0.15 and upside > 0.05:
            return "Moderate Conviction BUY"
        elif cv < 0.20:
            return "Moderate Conviction"
        else:
            return "Low Conviction — Models Diverge"

    # ─── THESIS ────────────────────────────────────────────────────────────────

    def _build_thesis(self, target: float, models: dict, conviction: str) -> str:
        """Plain-English investment thesis."""
        upside_pct = (target - self.current_price) / self.current_price * 100
        name = self.info.get("shortName") or self.ticker
        sector = self.info.get("sector", "")
        rev_growth = (self.info.get("revenueGrowth") or 0) * 100
        net_margin = (self.info.get("profitMargins") or 0) * 100
        roe = (self.info.get("returnOnEquity") or 0) * 100

        analyst = models.get("analyst_consensus", {})
        analyst_rec = analyst.get("recommendation", "")
        n_analysts = analyst.get("n_analysts", 0)

        dcf = models.get("dcf", {})
        dcf_status = dcf.get("status", "")

        parts = []

        # Opening
        direction = "upside" if upside_pct > 0 else "downside"
        parts.append(
            f"{name} ({self.ticker}) has a 12-month price target of ${target:.2f}, "
            f"implying {abs(upside_pct):.1f}% {direction} from the current price of ${self.current_price:.2f}."
        )

        # Business quality
        if rev_growth > 20:
            parts.append(
                f"The company is growing revenue at {rev_growth:.0f}% year-over-year — "
                f"a standout growth rate that reflects strong demand and market share gains."
            )
        elif rev_growth > 8:
            parts.append(f"Revenue is growing at a healthy {rev_growth:.0f}% annually.")
        elif rev_growth < 0:
            parts.append(f"Revenue declined {abs(rev_growth):.0f}% — a key concern that weighs on the outlook.")

        if net_margin > 20:
            parts.append(f"With a {net_margin:.0f}% net margin, the business is highly profitable and cash generative.")
        elif net_margin < 5:
            parts.append(f"Net margins of {net_margin:.0f}% are thin — execution and scale are critical to upside.")

        # Valuation
        if dcf_status:
            parts.append(f"On a DCF basis, the stock appears {dcf_status.lower()}.")

        # Analyst view
        if n_analysts > 0 and analyst.get("price"):
            parts.append(
                f"Wall Street's consensus ({n_analysts} analysts) has a price target of ${analyst['price']:.2f} "
                f"with a {analyst_rec.replace('_', ' ')} recommendation."
            )

        # Conviction
        parts.append(f"Model conviction: {conviction}.")

        return " ".join(parts)

    # ─── CATALYSTS & RISKS ─────────────────────────────────────────────────────

    def _identify_catalysts(self) -> list:
        """Identify potential positive catalysts."""
        catalysts = []
        info = self.info

        # Earnings beat potential
        eg = info.get("earningsGrowth") or 0
        if eg > 0.20:
            catalysts.append("Earnings acceleration — strong growth could trigger upward estimate revisions")

        # Analyst upgrades potential
        rec = (info.get("recommendationKey") or "").lower()
        if rec in ["underperform", "sell", "hold"]:
            target = info.get("targetMeanPrice") or 0
            if target > self.current_price * 1.1:
                catalysts.append("Analyst upgrade potential — consensus target significantly above current price")

        # DCF upside
        dcf = self.simulations.get("dcf", {})
        if (dcf.get("upside_downside") or 0) > 0.15:
            catalysts.append("Fundamental re-rating — stock trading below intrinsic value estimate")

        # Revenue growth
        rg = info.get("revenueGrowth") or 0
        if rg > 0.30:
            catalysts.append("Hyper-growth revenue trajectory — potential for multiple expansion")

        # Share buybacks
        fcf = info.get("freeCashflow") or 0
        mktcap = info.get("marketCap") or 1
        if fcf > 0 and fcf / mktcap > 0.04:
            catalysts.append(f"Strong FCF yield ({fcf/mktcap*100:.1f}%) supports buybacks or dividend growth")

        # Sector tailwinds
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        if "semiconductor" in industry.lower() or "artificial" in industry.lower():
            catalysts.append("AI infrastructure buildout — secular demand tailwind for multiple years")
        if "software" in industry.lower():
            catalysts.append("Cloud/AI integration driving SaaS expansion and pricing power")
        if "drug" in industry.lower() or "biotech" in industry.lower():
            catalysts.append("Pipeline catalysts — FDA approvals or positive trial data could re-rate the stock")

        if not catalysts:
            catalysts.append("Continued execution on existing growth strategy")

        return catalysts[:5]

    def _identify_risks(self) -> list:
        """Identify key downside risks."""
        risks = []
        info = self.info

        # Valuation risk
        pe = info.get("trailingPE") or 0
        if pe > 40:
            risks.append(f"Elevated valuation (P/E {pe:.0f}x) — multiple compression risk if growth disappoints")

        # High beta
        beta = info.get("beta") or 1
        if beta > 1.8:
            risks.append(f"High beta ({beta:.1f}x) — amplified downside in market corrections")

        # Debt
        de = (info.get("debtToEquity") or 0) / 100
        if de > 1.5:
            risks.append(f"High leverage (D/E {de:.1f}x) — vulnerable to rising interest rates")

        # Margin compression
        gm = info.get("grossMargins") or 1
        nm = info.get("profitMargins") or 0
        if gm > 0 and nm / gm < 0.20:
            risks.append("Operating leverage risk — high fixed costs mean margin compression in a slowdown")

        # Concentration risk
        mc = self.simulations.get("monte_carlo", {})
        vol = mc.get("annualized_vol", 0)
        if vol > 0.50:
            risks.append(f"Very high volatility ({vol*100:.0f}% annualized) — wide outcome range")

        # Competitive
        comp_pos = self.competitive.get("competitive_position", "")
        if "Weak" in comp_pos or "Challenged" in comp_pos:
            risks.append("Weak competitive position vs peers — risk of market share loss")

        # Regulatory
        sector = info.get("sector", "")
        if sector in ["Financial Services", "Healthcare"]:
            risks.append("Regulatory risk — policy changes could impact business model")

        if not risks:
            risks.append("General market risk and macroeconomic uncertainty")

        return risks[:5]
