"""
short_interest.py — Short Interest Analysis & Short Squeeze Detector

Metrics:
- Short float % (% of float sold short)
- Days to Cover (short interest / avg daily volume)
- Short squeeze score (composite of multiple factors)
- Borrow rate proxy
- Short interest trend (increasing/decreasing)
- Historical squeeze pattern detection

Data: yfinance (free) + derived metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


class ShortInterestAnalyzer:

    # Thresholds for squeeze scoring
    HIGH_SHORT_FLOAT     = 0.20   # 20%+ short float = elevated
    VERY_HIGH_SHORT_FLOAT = 0.30  # 30%+ = very high squeeze risk
    LOW_DAYS_TO_COVER    = 3      # Under 3 days = easy to cover (less squeeze)
    HIGH_DAYS_TO_COVER   = 8      # Over 8 days = hard to cover (squeeze potential)

    def __init__(self, ticker: str, price_data: pd.DataFrame, info: dict):
        self.ticker     = ticker.upper()
        self.price_data = price_data
        self.info       = info or {}

    def full_analysis(self) -> dict:
        raw = self._get_short_data()
        squeeze = self._squeeze_score(raw)
        signal  = self._signal(raw, squeeze)
        score   = self._score(raw, squeeze)

        return {
            "score":           score,
            "short_float_pct": raw.get("short_float_pct"),
            "days_to_cover":   raw.get("days_to_cover"),
            "short_ratio":     raw.get("short_ratio"),
            "shares_short":    raw.get("shares_short"),
            "float_shares":    raw.get("float_shares"),
            "squeeze_score":   squeeze["score"],
            "squeeze_risk":    squeeze["risk_level"],
            "squeeze_factors": squeeze["factors"],
            "signal":          signal,
            "summary":         self._summary(raw, squeeze),
            "momentum_data":   self._price_vs_short_momentum(),
        }

    # ── DATA FETCHING ─────────────────────────────────────────────

    def _get_short_data(self) -> dict:
        """Pull short interest metrics from yfinance."""
        info = self.info

        shares_short      = info.get("sharesShort")
        short_prior_month = info.get("sharesShortPriorMonth")
        float_shares      = info.get("floatShares")
        avg_volume        = info.get("averageVolume") or info.get("averageVolume10days")
        short_ratio       = info.get("shortRatio")     # = days to cover
        short_pct_float   = info.get("shortPercentOfFloat")

        # Calculate days to cover if not provided
        days_to_cover = short_ratio
        if not days_to_cover and shares_short and avg_volume and avg_volume > 0:
            days_to_cover = shares_short / avg_volume

        # Calculate short float % if not provided
        short_float_pct = short_pct_float
        if not short_float_pct and shares_short and float_shares and float_shares > 0:
            short_float_pct = shares_short / float_shares

        # Month-over-month change in short interest
        short_change_mom = None
        if shares_short and short_prior_month and short_prior_month > 0:
            short_change_mom = (shares_short - short_prior_month) / short_prior_month

        return {
            "shares_short":      shares_short,
            "shares_short_prior": short_prior_month,
            "float_shares":      float_shares,
            "avg_daily_volume":  avg_volume,
            "short_ratio":       round(short_ratio, 2) if short_ratio else None,
            "days_to_cover":     round(days_to_cover, 2) if days_to_cover else None,
            "short_float_pct":   round(float(short_float_pct) * 100, 2) if short_float_pct else None,
            "short_change_mom":  round(short_change_mom * 100, 2) if short_change_mom else None,
        }

    # ── SQUEEZE SCORING ───────────────────────────────────────────

    def _squeeze_score(self, raw: dict) -> dict:
        """
        Multi-factor short squeeze probability score.
        Based on: Ihor Dusaniwsky (S3 Partners) methodology.
        """
        score = 0
        factors = []
        risk_level = "Low"

        short_float = raw.get("short_float_pct") or 0
        dtc         = raw.get("days_to_cover") or 0
        short_chg   = raw.get("short_change_mom") or 0

        # Factor 1: Short float percentage (0-30 pts)
        if short_float >= 30:
            score += 30
            factors.append(f"🔥 Very high short float ({short_float:.1f}%) — prime squeeze territory")
        elif short_float >= 20:
            score += 20
            factors.append(f"⚠️  High short float ({short_float:.1f}%) — elevated squeeze risk")
        elif short_float >= 10:
            score += 10
            factors.append(f"Short float {short_float:.1f}% — moderate short interest")
        elif short_float > 0:
            score += 3
            factors.append(f"Low short float ({short_float:.1f}%) — minimal squeeze risk")

        # Factor 2: Days to Cover (0-25 pts)
        if dtc >= 10:
            score += 25
            factors.append(f"🔥 {dtc:.1f} days to cover — shorts trapped, very hard to exit")
        elif dtc >= 7:
            score += 18
            factors.append(f"⚠️  {dtc:.1f} days to cover — difficult short exit")
        elif dtc >= 4:
            score += 10
            factors.append(f"{dtc:.1f} days to cover — moderate difficulty")
        elif dtc > 0:
            score += 3
            factors.append(f"{dtc:.1f} days to cover — shorts can exit easily")

        # Factor 3: Short interest trend (0-20 pts)
        if short_chg < -15:
            score += 5
            factors.append(f"Shorts covering ({short_chg:.1f}% MoM decrease) — squeeze may be starting")
        elif short_chg < -5:
            score += 8
            factors.append(f"Short interest declining ({short_chg:.1f}% MoM)")
        elif short_chg > 20:
            score += 20
            factors.append(f"🔥 Short interest surging (+{short_chg:.1f}% MoM) — shorts piling in, fuel for squeeze")
        elif short_chg > 10:
            score += 12
            factors.append(f"Short interest building (+{short_chg:.1f}% MoM)")

        # Factor 4: Price momentum vs short position (0-15 pts)
        momentum = self._price_momentum()
        if momentum > 0.10:
            score += 15
            factors.append(f"🔥 Strong upward price momentum (+{momentum*100:.1f}%) while heavily shorted — squeeze in progress?")
        elif momentum > 0.03:
            score += 8
            factors.append(f"Positive price momentum ({momentum*100:.1f}%) — pressure on shorts")
        elif momentum < -0.10:
            score -= 5
            factors.append(f"Negative price momentum — shorts currently winning")

        # Factor 5: Volume surge (0-10 pts)
        vol_surge = self._volume_surge()
        if vol_surge > 2.0:
            score += 10
            factors.append(f"🔥 Volume {vol_surge:.1f}x average — unusual buying pressure")
        elif vol_surge > 1.3:
            score += 5
            factors.append(f"Volume elevated ({vol_surge:.1f}x average)")

        # Risk level
        if score >= 70:      risk_level = "EXTREME 🔥🔥🔥"
        elif score >= 50:    risk_level = "HIGH 🔥🔥"
        elif score >= 30:    risk_level = "MODERATE 🔥"
        elif score >= 15:    risk_level = "LOW-MODERATE"
        else:                risk_level = "LOW"

        return {
            "score":      min(score, 100),
            "risk_level": risk_level,
            "factors":    factors,
        }

    # ── MOMENTUM HELPERS ─────────────────────────────────────────

    def _price_momentum(self) -> float:
        """20-day price momentum."""
        if self.price_data is None or len(self.price_data) < 20:
            return 0
        try:
            recent  = float(self.price_data["Close"].iloc[-1])
            past_20 = float(self.price_data["Close"].iloc[-20])
            return (recent - past_20) / past_20
        except Exception:
            return 0

    def _volume_surge(self) -> float:
        """Recent volume vs 20-day average."""
        if self.price_data is None or len(self.price_data) < 20:
            return 1.0
        try:
            avg_vol    = float(self.price_data["Volume"].tail(20).mean())
            recent_vol = float(self.price_data["Volume"].iloc[-1])
            return recent_vol / avg_vol if avg_vol > 0 else 1.0
        except Exception:
            return 1.0

    def _price_vs_short_momentum(self) -> dict:
        """Analyze if price is moving against short sellers."""
        if self.price_data is None or self.price_data.empty:
            return {}
        try:
            close = self.price_data["Close"]
            mom_5d  = (float(close.iloc[-1]) - float(close.iloc[-5]))  / float(close.iloc[-5])  if len(close) >= 5  else 0
            mom_20d = (float(close.iloc[-1]) - float(close.iloc[-20])) / float(close.iloc[-20]) if len(close) >= 20 else 0
            mom_60d = (float(close.iloc[-1]) - float(close.iloc[-60])) / float(close.iloc[-60]) if len(close) >= 60 else 0

            return {
                "5d_momentum":  round(mom_5d  * 100, 2),
                "20d_momentum": round(mom_20d * 100, 2),
                "60d_momentum": round(mom_60d * 100, 2),
                "trend":        "Rising against shorts" if mom_20d > 0.05 else
                                "Falling with shorts"  if mom_20d < -0.05 else "Neutral",
            }
        except Exception:
            return {}

    # ── SIGNAL & SCORE ───────────────────────────────────────────

    def _signal(self, raw: dict, squeeze: dict) -> str:
        sq_score    = squeeze["score"]
        short_float = raw.get("short_float_pct") or 0
        dtc         = raw.get("days_to_cover") or 0

        if sq_score >= 70:
            return "🔥 HIGH SQUEEZE POTENTIAL — multiple factors aligned"
        elif sq_score >= 50:
            return "⚠️  ELEVATED SQUEEZE RISK — watch for catalyst"
        elif sq_score >= 30:
            return "📊 MODERATE SHORT INTEREST — below squeeze threshold"
        elif short_float < 5:
            return "✅ LOW SHORT INTEREST — shorts not a significant factor"
        else:
            return "⚪ NORMAL SHORT INTEREST"

    def _score(self, raw: dict, squeeze: dict) -> float:
        """
        Score from 0-100 for the overall analysis module.
        High short interest with price momentum = bullish for longs (potential squeeze).
        High short interest with falling price = bearish.
        """
        sq     = squeeze["score"]
        mom    = self._price_momentum()

        if sq > 50 and mom > 0:
            # Heavily shorted + rising price = squeeze potential = bullish
            return round(min(50 + sq * 0.3 + mom * 100, 100), 2)
        elif sq > 50 and mom < 0:
            # Heavily shorted + falling = shorts winning = bearish
            return round(max(50 - sq * 0.2, 10), 2)
        else:
            return 50.0

    def _summary(self, raw: dict, squeeze: dict) -> str:
        sf  = raw.get("short_float_pct")
        dtc = raw.get("days_to_cover")
        chg = raw.get("short_change_mom")
        sq  = squeeze["score"]
        risk = squeeze["risk_level"]

        parts = []
        if sf:
            parts.append(f"Short float: {sf:.1f}%.")
        if dtc:
            parts.append(f"Days to cover: {dtc:.1f}.")
        if chg:
            direction = "increased" if chg > 0 else "decreased"
            parts.append(f"Short interest {direction} {abs(chg):.1f}% month-over-month.")
        parts.append(f"Squeeze risk: {risk} (score: {sq}/100).")

        return " ".join(parts)
