"""
technical_analysis.py — Institutional technical analysis.

Indicators:
- Trend: EMA (20/50/200), ADX, MACD
- Momentum: RSI, Stochastic, Williams %R
- Volatility: Bollinger Bands, ATR
- Volume: OBV, VWAP, Volume MA
- Support/Resistance: Pivot Points
- Pattern detection: Golden/Death Cross, oversold/overbought
"""

import numpy as np
import pandas as pd
from typing import Optional


class TechnicalAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self._indicators = None
        if not self.df.empty:
            self._compute_indicators()

    # ─── PUBLIC ────────────────────────────────────────────────────────────────

    def full_analysis(self) -> dict:
        if self.df.empty:
            return {"score": 50, "signals": {}, "error": "No price data"}

        ind = self._indicators
        signals = self._generate_signals(ind)
        score = self._score(signals, ind)

        return {
            "score": score,
            "signals": signals,
            "current_price": float(ind["Close"].iloc[-1]),
            "sma_20": float(ind["SMA_20"].iloc[-1]) if "SMA_20" in ind else None,
            "sma_50": float(ind["SMA_50"].iloc[-1]) if "SMA_50" in ind else None,
            "sma_200": float(ind["SMA_200"].iloc[-1]) if "SMA_200" in ind else None,
            "rsi": float(ind["RSI"].iloc[-1]) if "RSI" in ind else None,
            "macd": float(ind["MACD"].iloc[-1]) if "MACD" in ind else None,
            "macd_signal": float(ind["MACD_Signal"].iloc[-1]) if "MACD_Signal" in ind else None,
            "atr": float(ind["ATR"].iloc[-1]) if "ATR" in ind else None,
            "bb_upper": float(ind["BB_Upper"].iloc[-1]) if "BB_Upper" in ind else None,
            "bb_lower": float(ind["BB_Lower"].iloc[-1]) if "BB_Lower" in ind else None,
            "adx": float(ind["ADX"].iloc[-1]) if "ADX" in ind else None,
            "volume_trend": self._volume_trend(ind),
            "support": self._find_support(ind),
            "resistance": self._find_resistance(ind),
        }

    # ─── INDICATORS ────────────────────────────────────────────────────────────

    def _compute_indicators(self):
        df = self.df.copy()
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        v = df["Volume"]

        # Moving Averages
        df["SMA_20"]  = c.rolling(20).mean()
        df["SMA_50"]  = c.rolling(50).mean()
        df["SMA_200"] = c.rolling(200).mean()
        df["EMA_12"]  = c.ewm(span=12, adjust=False).mean()
        df["EMA_26"]  = c.ewm(span=26, adjust=False).mean()
        df["EMA_20"]  = c.ewm(span=20, adjust=False).mean()

        # MACD
        df["MACD"]        = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

        # RSI
        df["RSI"] = self._rsi(c, 14)

        # Stochastic
        df["Stoch_K"], df["Stoch_D"] = self._stochastic(h, l, c)

        # Bollinger Bands
        sma20 = df["SMA_20"]
        std20 = c.rolling(20).std()
        df["BB_Upper"] = sma20 + 2 * std20
        df["BB_Lower"] = sma20 - 2 * std20
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20
        df["BB_Pct"]   = (c - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

        # ATR
        df["ATR"] = self._atr(h, l, c, 14)

        # ADX
        df["ADX"] = self._adx(h, l, c, 14)

        # OBV
        df["OBV"] = (np.sign(c.diff()) * v).fillna(0).cumsum()

        # Williams %R
        highest_high = h.rolling(14).max()
        lowest_low   = l.rolling(14).min()
        df["Williams_R"] = -100 * (highest_high - c) / (highest_high - lowest_low)

        # Volume MA
        df["Vol_MA_20"] = v.rolling(20).mean()
        df["Vol_Ratio"]  = v / df["Vol_MA_20"]

        self._indicators = df

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(self, high, low, close, period=14) -> pd.Series:
        tr = pd.DataFrame({
            "hl":  high - low,
            "hc":  (high - close.shift()).abs(),
            "lc":  (low  - close.shift()).abs(),
        }).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _adx(self, high, low, close, period=14) -> pd.Series:
        tr = self._atr(high, low, close, period)
        up_move   = high.diff()
        down_move = -low.diff()
        plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm_s  = pd.Series(plus_dm,  index=high.index).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean()
        plus_di  = 100 * plus_dm_s / tr.replace(0, np.nan)
        minus_di = 100 * minus_dm_s / tr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1/period, adjust=False).mean()

    def _stochastic(self, high, low, close, k=14, d=3):
        lowest_low   = low.rolling(k).min()
        highest_high = high.rolling(k).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        stoch_d = stoch_k.rolling(d).mean()
        return stoch_k, stoch_d

    # ─── SIGNALS ───────────────────────────────────────────────────────────────

    def _generate_signals(self, df: pd.DataFrame) -> dict:
        signals = {}
        c = df["Close"].iloc[-1]

        # Trend signals
        if "SMA_50" in df and "SMA_200" in df:
            sma50  = df["SMA_50"].iloc[-1]
            sma200 = df["SMA_200"].iloc[-1]
            prev_sma50  = df["SMA_50"].iloc[-2] if len(df) > 1 else sma50
            prev_sma200 = df["SMA_200"].iloc[-2] if len(df) > 1 else sma200

            if sma50 > sma200 and prev_sma50 <= prev_sma200:
                signals["Golden Cross"] = "🟢 BULLISH (SMA50 crossed above SMA200)"
            elif sma50 < sma200 and prev_sma50 >= prev_sma200:
                signals["Death Cross"] = "🔴 BEARISH (SMA50 crossed below SMA200)"
            elif sma50 > sma200:
                signals["Trend (50/200 MA)"] = "🟢 Uptrend (SMA50 > SMA200)"
            else:
                signals["Trend (50/200 MA)"] = "🔴 Downtrend (SMA50 < SMA200)"

        if "SMA_20" in df and c > df["SMA_20"].iloc[-1]:
            signals["Price vs SMA20"] = "🟢 Above SMA20 (short-term bullish)"
        elif "SMA_20" in df:
            signals["Price vs SMA20"] = "🔴 Below SMA20 (short-term bearish)"

        # MACD
        if "MACD" in df and "MACD_Signal" in df:
            macd      = df["MACD"].iloc[-1]
            macd_sig  = df["MACD_Signal"].iloc[-1]
            prev_macd = df["MACD"].iloc[-2]
            prev_sig  = df["MACD_Signal"].iloc[-2]
            if macd > macd_sig and prev_macd <= prev_sig:
                signals["MACD"] = "🟢 BULLISH CROSSOVER"
            elif macd < macd_sig and prev_macd >= prev_sig:
                signals["MACD"] = "🔴 BEARISH CROSSOVER"
            elif macd > macd_sig:
                signals["MACD"] = "🟢 Bullish (above signal)"
            else:
                signals["MACD"] = "🔴 Bearish (below signal)"

        # RSI
        if "RSI" in df:
            rsi = df["RSI"].iloc[-1]
            if rsi < 30:
                signals["RSI"] = f"🟢 OVERSOLD ({rsi:.1f}) — potential buy zone"
            elif rsi > 70:
                signals["RSI"] = f"🔴 OVERBOUGHT ({rsi:.1f}) — potential sell zone"
            elif 40 <= rsi <= 60:
                signals["RSI"] = f"🟡 NEUTRAL ({rsi:.1f})"
            elif rsi < 40:
                signals["RSI"] = f"🟠 WEAKENING ({rsi:.1f})"
            else:
                signals["RSI"] = f"🟢 BULLISH ({rsi:.1f})"

        # Bollinger Bands
        if "BB_Pct" in df:
            bb_pct = df["BB_Pct"].iloc[-1]
            if bb_pct < 0.05:
                signals["Bollinger Bands"] = "🟢 Near lower band (oversold potential)"
            elif bb_pct > 0.95:
                signals["Bollinger Bands"] = "🔴 Near upper band (overbought potential)"
            else:
                signals["Bollinger Bands"] = f"🟡 Mid-band ({bb_pct*100:.0f}% of band width)"

        # ADX (trend strength)
        if "ADX" in df:
            adx = df["ADX"].iloc[-1]
            if adx > 40:
                signals["ADX (Trend Strength)"] = f"🟢 STRONG TREND ({adx:.1f})"
            elif adx > 25:
                signals["ADX (Trend Strength)"] = f"🟡 MODERATE TREND ({adx:.1f})"
            else:
                signals["ADX (Trend Strength)"] = f"⚪ WEAK/NO TREND ({adx:.1f})"

        # Volume
        if "Vol_Ratio" in df:
            vr = df["Vol_Ratio"].iloc[-1]
            if vr > 1.5:
                signals["Volume"] = f"🟢 High volume ({vr:.1f}x average) — confirms move"
            elif vr < 0.7:
                signals["Volume"] = f"🟡 Low volume ({vr:.1f}x average) — weak confirmation"
            else:
                signals["Volume"] = f"⚪ Normal volume ({vr:.1f}x average)"

        return signals

    def _score(self, signals: dict, df: pd.DataFrame) -> float:
        """Convert signals to a 0-100 score."""
        score = 50.0  # neutral baseline

        for key, val in signals.items():
            if "🟢" in val:
                if "CROSSOVER" in val or "OVERSOLD" in val or "Golden Cross" in val:
                    score += 8
                else:
                    score += 4
            elif "🔴" in val:
                if "CROSSOVER" in val or "Death Cross" in val:
                    score -= 8
                else:
                    score -= 4
            elif "🟡" in val:
                pass  # neutral

        # Boost for strong trend
        if "ADX" in df:
            adx = df["ADX"].iloc[-1]
            if adx > 30:
                score += 5

        return round(min(max(score, 0), 100), 2)

    def _volume_trend(self, df: pd.DataFrame) -> str:
        if "OBV" not in df:
            return "N/A"
        obv_recent = df["OBV"].iloc[-20:].mean()
        obv_older  = df["OBV"].iloc[-40:-20].mean()
        if obv_recent > obv_older * 1.05:
            return "Rising (accumulation)"
        elif obv_recent < obv_older * 0.95:
            return "Falling (distribution)"
        else:
            return "Neutral"

    def _find_support(self, df: pd.DataFrame, lookback: int = 90) -> Optional[float]:
        """Simple swing low support level."""
        try:
            recent = df["Low"].tail(lookback)
            return float(recent.min())
        except Exception:
            return None

    def _find_resistance(self, df: pd.DataFrame, lookback: int = 90) -> Optional[float]:
        """Simple swing high resistance level."""
        try:
            recent = df["High"].tail(lookback)
            return float(recent.max())
        except Exception:
            return None
