"""
options_signals.py — Options Market Intelligence

Metrics:
- Put/Call Ratio (bearish/bullish sentiment gauge)
- Implied Volatility (IV) vs Historical Volatility (HV)
- IV Rank & IV Percentile (is options expensive or cheap?)
- Unusual options activity detection
- Options-implied move (what market expects for earnings)
- Max Pain price (where options market makers profit most)
- Gamma exposure (GEX) approximation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


class OptionsSignalAnalyzer:

    def __init__(self, ticker: str, price_data: pd.DataFrame, info: dict):
        self.ticker     = ticker.upper()
        self.price_data = price_data
        self.info       = info or {}
        self._yf        = yf.Ticker(ticker)

    def full_analysis(self) -> dict:
        print(f"    Fetching options chain...")
        chain_data  = self._get_options_chain()
        iv_data     = self._iv_analysis()
        pcr_data    = self._put_call_ratio(chain_data)
        max_pain    = self._max_pain(chain_data)
        unusual     = self._unusual_activity(chain_data)
        iv_rank     = self._iv_rank()
        implied_move = self._implied_move(chain_data)
        score       = self._score(pcr_data, iv_data, iv_rank, unusual)

        return {
            "score":            score,
            "put_call_ratio":   pcr_data,
            "iv_analysis":      iv_data,
            "iv_rank":          iv_rank,
            "max_pain":         max_pain,
            "implied_move":     implied_move,
            "unusual_activity": unusual,
            "signal":           self._signal(pcr_data, iv_data, iv_rank, unusual),
            "summary":          self._summary(pcr_data, iv_data, iv_rank, max_pain),
        }

    # ── OPTIONS CHAIN ─────────────────────────────────────────────

    def _get_options_chain(self) -> dict:
        """Fetch nearest-dated options chain."""
        try:
            expirations = self._yf.options
            if not expirations:
                return {}

            # Use nearest expiration (most liquid)
            nearest = expirations[0]
            chain   = self._yf.option_chain(nearest)

            return {
                "expiration": nearest,
                "calls": chain.calls,
                "puts":  chain.puts,
            }
        except Exception:
            return {}

    # ── PUT/CALL RATIO ────────────────────────────────────────────

    def _put_call_ratio(self, chain: dict) -> dict:
        """
        Put/Call ratio by volume and open interest.
        PCR > 1.0 = bearish sentiment (more puts than calls)
        PCR < 0.7 = bullish sentiment (calls dominate)
        PCR 0.7-1.0 = neutral
        """
        if not chain or "calls" not in chain or "puts" not in chain:
            return {"pcr_volume": None, "pcr_oi": None, "sentiment": "No Data"}

        calls = chain["calls"]
        puts  = chain["puts"]

        call_vol = float(calls["volume"].sum()) if "volume" in calls else 0
        put_vol  = float(puts["volume"].sum())  if "volume" in puts  else 0
        call_oi  = float(calls["openInterest"].sum()) if "openInterest" in calls else 0
        put_oi   = float(puts["openInterest"].sum())  if "openInterest" in puts  else 0

        pcr_vol = put_vol  / call_vol  if call_vol  > 0 else None
        pcr_oi  = put_oi   / call_oi   if call_oi   > 0 else None

        if pcr_vol is None:
            sentiment = "No Data"
        elif pcr_vol > 1.5:
            sentiment = "🔴 Very Bearish (heavy put buying)"
        elif pcr_vol > 1.0:
            sentiment = "🔴 Bearish (more puts than calls)"
        elif pcr_vol > 0.7:
            sentiment = "🟡 Neutral"
        elif pcr_vol > 0.5:
            sentiment = "🟢 Bullish (calls dominating)"
        else:
            sentiment = "🟢🟢 Very Bullish (extreme call buying)"

        return {
            "pcr_volume":  round(pcr_vol, 3) if pcr_vol else None,
            "pcr_oi":      round(pcr_oi,  3) if pcr_oi  else None,
            "call_volume": int(call_vol),
            "put_volume":  int(put_vol),
            "call_oi":     int(call_oi),
            "put_oi":      int(put_oi),
            "sentiment":   sentiment,
            "expiration":  chain.get("expiration"),
        }

    # ── IMPLIED VOLATILITY ────────────────────────────────────────

    def _iv_analysis(self) -> dict:
        """Compare IV to historical volatility."""
        try:
            chain_data = self._get_options_chain()
            if not chain_data or "calls" not in chain_data:
                return self._iv_from_info()

            calls = chain_data["calls"]
            puts  = chain_data["puts"]

            # ATM IV (nearest to current price)
            current_price = float(self.price_data["Close"].iloc[-1]) if self.price_data is not None else 0
            if current_price == 0:
                return self._iv_from_info()

            # Get ATM call IV
            atm_calls = calls.copy()
            atm_calls["dist"] = (atm_calls["strike"] - current_price).abs()
            atm_calls = atm_calls.sort_values("dist")
            atm_iv_call = float(atm_calls.iloc[0]["impliedVolatility"]) if len(atm_calls) > 0 else 0

            # Get ATM put IV
            atm_puts = puts.copy()
            atm_puts["dist"] = (atm_puts["strike"] - current_price).abs()
            atm_puts = atm_puts.sort_values("dist")
            atm_iv_put = float(atm_puts.iloc[0]["impliedVolatility"]) if len(atm_puts) > 0 else 0

            atm_iv = (atm_iv_call + atm_iv_put) / 2 if atm_iv_call and atm_iv_put else (atm_iv_call or atm_iv_put)

            # Historical volatility (30-day realized)
            hv_30 = 0
            if self.price_data is not None and len(self.price_data) >= 30:
                returns = self.price_data["Close"].pct_change().dropna()
                hv_30 = float(returns.tail(30).std() * np.sqrt(252))

            iv_premium = atm_iv - hv_30 if hv_30 > 0 else 0

            return {
                "atm_iv":       round(atm_iv * 100, 2),
                "hv_30":        round(hv_30 * 100, 2),
                "iv_premium":   round(iv_premium * 100, 2),
                "iv_hv_ratio":  round(atm_iv / hv_30, 2) if hv_30 > 0 else None,
                "assessment":   self._iv_assessment(atm_iv, hv_30),
            }

        except Exception:
            return self._iv_from_info()

    def _iv_from_info(self) -> dict:
        """Fallback IV estimate from yfinance info fields."""
        try:
            hv = None
            if self.price_data is not None and len(self.price_data) >= 30:
                returns = self.price_data["Close"].pct_change().dropna()
                hv = float(returns.tail(30).std() * np.sqrt(252))
            return {
                "atm_iv":  None,
                "hv_30":   round(hv * 100, 2) if hv else None,
                "iv_premium": None,
                "iv_hv_ratio": None,
                "assessment": "IV data unavailable — options chain empty",
            }
        except Exception:
            return {"atm_iv": None, "hv_30": None, "assessment": "No data"}

    def _iv_assessment(self, iv: float, hv: float) -> str:
        if hv == 0:
            return "Cannot assess — no historical vol data"
        ratio = iv / hv
        if ratio > 1.5:
            return "🔴 IV very elevated vs HV — options expensive, sell premium strategies"
        elif ratio > 1.2:
            return "🟠 IV elevated — options somewhat expensive"
        elif ratio > 0.9:
            return "🟡 IV near historical vol — fair options pricing"
        elif ratio > 0.7:
            return "🟢 IV compressed — options cheap, buy premium strategies"
        else:
            return "🟢🟢 IV very compressed — unusually cheap options"

    # ── IV RANK ───────────────────────────────────────────────────

    def _iv_rank(self) -> dict:
        """
        IV Rank: where is current IV relative to past 52 weeks?
        IVR > 50 = options expensive
        IVR < 20 = options cheap
        """
        try:
            iv_data = self._iv_analysis()
            atm_iv  = iv_data.get("atm_iv")
            if not atm_iv:
                return {"iv_rank": None, "iv_percentile": None, "assessment": "No data"}

            # Approximate 52-week IV range using HV from different windows
            if self.price_data is None or len(self.price_data) < 30:
                return {"iv_rank": None, "assessment": "Insufficient data"}

            returns = self.price_data["Close"].pct_change().dropna()
            # Rolling 30-day vol as HV proxy for each window
            rolling_hvs = []
            for i in range(30, min(len(returns), 252), 5):
                window = returns.iloc[i-30:i]
                rolling_hvs.append(float(window.std() * np.sqrt(252) * 100))

            if not rolling_hvs:
                return {"iv_rank": None, "assessment": "Insufficient data"}

            iv_52w_low  = min(rolling_hvs)
            iv_52w_high = max(rolling_hvs)

            if iv_52w_high == iv_52w_low:
                return {"iv_rank": 50, "assessment": "No IV range available"}

            iv_rank = (atm_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100

            if iv_rank > 80:
                assessment = "🔴 Very high IV rank — options are expensive (sell premium)"
            elif iv_rank > 50:
                assessment = "🟠 Elevated IV rank — options somewhat expensive"
            elif iv_rank > 30:
                assessment = "🟡 Average IV rank"
            else:
                assessment = "🟢 Low IV rank — options cheap (buy premium)"

            return {
                "iv_rank":       round(iv_rank, 1),
                "iv_52w_low":    round(iv_52w_low, 2),
                "iv_52w_high":   round(iv_52w_high, 2),
                "current_iv":    atm_iv,
                "assessment":    assessment,
            }
        except Exception:
            return {"iv_rank": None, "assessment": "Error calculating IV rank"}

    # ── MAX PAIN ──────────────────────────────────────────────────

    def _max_pain(self, chain: dict) -> dict:
        """
        Max pain = strike price where most options expire worthless.
        Stock price tends to gravitate toward max pain near expiration.
        Theory: options market makers hedge to minimize payouts.
        """
        if not chain or "calls" not in chain or "puts" not in chain:
            return {"max_pain_price": None}

        try:
            calls = chain["calls"][["strike", "openInterest"]].copy()
            puts  = chain["puts"][["strike", "openInterest"]].copy()

            all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))

            pain = {}
            for strike in all_strikes:
                # Loss to call writers if price = this strike
                call_pain = calls[calls["strike"] < strike].apply(
                    lambda r: (strike - r["strike"]) * r["openInterest"] * 100, axis=1
                ).sum()
                # Loss to put writers
                put_pain = puts[puts["strike"] > strike].apply(
                    lambda r: (r["strike"] - strike) * r["openInterest"] * 100, axis=1
                ).sum()
                pain[strike] = call_pain + put_pain

            max_pain_price = min(pain, key=pain.get)
            current_price  = float(self.price_data["Close"].iloc[-1]) if self.price_data is not None else 0

            distance = (max_pain_price - current_price) / current_price * 100 if current_price > 0 else 0

            return {
                "max_pain_price":   round(max_pain_price, 2),
                "current_price":    round(current_price, 2),
                "distance_pct":     round(distance, 2),
                "interpretation":   f"Max pain at ${max_pain_price:.2f} — "
                                    f"{'stock may drift UP' if distance > 2 else 'stock may drift DOWN' if distance < -2 else 'near current price'} "
                                    f"toward expiration ({chain.get('expiration', 'N/A')})",
            }
        except Exception:
            return {"max_pain_price": None}

    # ── IMPLIED MOVE ──────────────────────────────────────────────

    def _implied_move(self, chain: dict) -> dict:
        """
        Implied move for upcoming expiration.
        Approximation: ATM straddle price / current price.
        """
        if not chain or "calls" not in chain:
            return {"implied_move_pct": None}

        try:
            current_price = float(self.price_data["Close"].iloc[-1]) if self.price_data is not None else 0
            if current_price == 0:
                return {"implied_move_pct": None}

            calls = chain["calls"].copy()
            puts  = chain["puts"].copy()

            # Find ATM strike
            calls["dist"] = (calls["strike"] - current_price).abs()
            atm_call = calls.sort_values("dist").iloc[0]

            puts["dist"] = (puts["strike"] - current_price).abs()
            atm_put = puts.sort_values("dist").iloc[0]

            straddle_price = (float(atm_call.get("lastPrice", 0)) +
                              float(atm_put.get("lastPrice", 0)))
            implied_move   = straddle_price / current_price * 100

            return {
                "implied_move_pct":  round(implied_move, 2),
                "straddle_price":    round(straddle_price, 2),
                "expiration":        chain.get("expiration"),
                "interpretation":    f"Market implying ±{implied_move:.1f}% move by {chain.get('expiration', 'expiration')}",
            }
        except Exception:
            return {"implied_move_pct": None}

    # ── UNUSUAL ACTIVITY ──────────────────────────────────────────

    def _unusual_activity(self, chain: dict) -> list:
        """Detect unusually large options positions (smart money signals)."""
        unusual = []
        if not chain or "calls" not in chain:
            return unusual

        try:
            current_price = float(self.price_data["Close"].iloc[-1]) if self.price_data is not None else 0

            for opt_type, df in [("CALL", chain["calls"]), ("PUT", chain["puts"])]:
                if df is None or df.empty:
                    continue

                df = df.copy()
                avg_oi  = df["openInterest"].mean() if "openInterest" in df else 0
                avg_vol = df["volume"].mean() if "volume" in df else 0

                for _, row in df.iterrows():
                    oi  = row.get("openInterest", 0) or 0
                    vol = row.get("volume", 0) or 0
                    strike = row.get("strike", 0)

                    # Flag if volume is 5x+ average and OI is significant
                    if avg_vol > 0 and vol > avg_vol * 5 and oi > 1000:
                        distance_pct = (strike - current_price) / current_price * 100
                        unusual.append({
                            "type":         opt_type,
                            "strike":       strike,
                            "volume":       int(vol),
                            "open_interest": int(oi),
                            "distance_pct": round(distance_pct, 1),
                            "signal":       f"{'🐂' if opt_type == 'CALL' else '🐻'} Unusual {opt_type} activity at ${strike:.0f} "
                                            f"({vol:,} vol vs {avg_vol:.0f} avg) — "
                                            f"{'OTM' if abs(distance_pct) > 5 else 'ATM/ITM'} "
                                            f"{distance_pct:+.1f}% from current",
                        })

            # Sort by volume descending
            unusual.sort(key=lambda x: x["volume"], reverse=True)
            return unusual[:5]

        except Exception:
            return []

    # ── SCORE & SIGNAL ────────────────────────────────────────────

    def _score(self, pcr, iv, iv_rank, unusual) -> float:
        score = 50.0

        # PCR
        pcr_vol = pcr.get("pcr_volume")
        if pcr_vol:
            if pcr_vol < 0.5:    score += 15
            elif pcr_vol < 0.7:  score += 8
            elif pcr_vol > 1.5:  score -= 15
            elif pcr_vol > 1.0:  score -= 8

        # Unusual activity
        call_unusual = sum(1 for u in unusual if u["type"] == "CALL")
        put_unusual  = sum(1 for u in unusual if u["type"] == "PUT")
        if call_unusual > put_unusual:    score += 10
        elif put_unusual > call_unusual:  score -= 10

        # IV — low IV = cheaper to buy calls = bullish setup
        iv_r = iv_rank.get("iv_rank")
        if iv_r is not None:
            if iv_r < 20:    score += 8
            elif iv_r > 70:  score -= 5

        return round(min(max(score, 0), 100), 2)

    def _signal(self, pcr, iv, iv_rank, unusual) -> str:
        pcr_vol = pcr.get("pcr_volume")
        iv_r    = iv_rank.get("iv_rank")

        signals = []
        if pcr_vol:
            if pcr_vol < 0.6:
                signals.append("Heavy call buying (bullish)")
            elif pcr_vol > 1.2:
                signals.append("Heavy put buying (bearish)")

        call_unusual = sum(1 for u in unusual if u["type"] == "CALL")
        if call_unusual >= 2:
            signals.append(f"{call_unusual} unusual call positions detected")

        if iv_r is not None:
            if iv_r > 70:
                signals.append("IV elevated — event risk priced in")
            elif iv_r < 25:
                signals.append("IV compressed — cheap options")

        return " | ".join(signals) if signals else "No strong options signal"

    def _summary(self, pcr, iv, iv_rank, max_pain) -> str:
        parts = []

        pcr_vol = pcr.get("pcr_volume")
        if pcr_vol:
            parts.append(f"P/C ratio: {pcr_vol:.2f} ({pcr.get('sentiment', '')})")

        atm_iv = iv.get("atm_iv")
        hv = iv.get("hv_30")
        if atm_iv and hv:
            parts.append(f"IV {atm_iv:.1f}% vs HV {hv:.1f}%")

        iv_r = iv_rank.get("iv_rank")
        if iv_r is not None:
            parts.append(f"IV Rank: {iv_r:.0f}/100")

        mp = max_pain.get("max_pain_price")
        if mp:
            parts.append(f"Max pain: ${mp:.2f}")

        return " | ".join(parts) if parts else "Options data unavailable"
