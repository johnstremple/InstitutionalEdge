"""
macro_regime.py — Macro Economic Regime Detection & Sector Rotation

Identifies the current macro environment and maps it to:
- Which sectors historically outperform
- Risk-on vs risk-off positioning
- Interest rate sensitivity
- Inflation impact on sector

Data sources (all free):
- FRED API (free, no key needed for basic data via yfinance proxies)
- Market-based indicators (yield curve, credit spreads via ETF prices)
- Sector ETF momentum (XLK, XLF, XLE, etc.)

Regimes:
1. Expansion (rising growth, low inflation)
2. Overheating (rising growth, rising inflation)
3. Stagflation (falling growth, rising inflation)
4. Recession (falling growth, falling inflation)
5. Recovery (bottoming growth, falling inflation)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests


# Sector ETFs for rotation analysis
SECTOR_ETFS = {
    "Technology":            "XLK",
    "Financials":            "XLF",
    "Healthcare":            "XLV",
    "Energy":                "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":      "XLP",
    "Industrials":           "XLI",
    "Utilities":             "XLU",
    "Real Estate":           "XLRE",
    "Materials":             "XLB",
    "Communication":         "XLC",
}

# Macro regime playbook (what outperforms in each regime)
REGIME_PLAYBOOK = {
    "Expansion": {
        "favored_sectors":    ["Technology", "Consumer Discretionary", "Industrials", "Financials"],
        "avoid_sectors":      ["Utilities", "Consumer Staples"],
        "asset_bias":         "Risk-On",
        "rate_environment":   "Rising rates OK initially",
        "description":        "GDP growing, inflation moderate. Best time to own growth stocks.",
        "historical_leaders": "Tech, discretionary, and small-caps outperform.",
    },
    "Overheating": {
        "favored_sectors":    ["Energy", "Materials", "Industrials", "Real Estate"],
        "avoid_sectors":      ["Technology", "Consumer Discretionary"],
        "asset_bias":         "Inflation Hedge",
        "rate_environment":   "Rising rates pressure growth stocks",
        "description":        "GDP strong but inflation building. Rotate to real assets.",
        "historical_leaders": "Commodities, energy, and inflation hedges dominate.",
    },
    "Stagflation": {
        "favored_sectors":    ["Energy", "Consumer Staples", "Healthcare", "Utilities"],
        "avoid_sectors":      ["Technology", "Financials", "Consumer Discretionary"],
        "asset_bias":         "Defensive",
        "rate_environment":   "Worst environment for equities",
        "description":        "Worst regime. Inflation high, growth slowing. Preserve capital.",
        "historical_leaders": "Cash, commodities, defensive sectors, gold.",
    },
    "Recession": {
        "favored_sectors":    ["Consumer Staples", "Healthcare", "Utilities", "Communication"],
        "avoid_sectors":      ["Industrials", "Energy", "Financials"],
        "asset_bias":         "Defensive / Bonds",
        "rate_environment":   "Falling rates — bonds rally",
        "description":        "GDP contracting. Focus on defensive, dividend-paying stocks.",
        "historical_leaders": "Staples, healthcare, utilities outperform. Bonds rally.",
    },
    "Recovery": {
        "favored_sectors":    ["Financials", "Industrials", "Technology", "Consumer Discretionary"],
        "avoid_sectors":      ["Utilities", "Consumer Staples"],
        "asset_bias":         "Risk-On (early)",
        "rate_environment":   "Low rates, beginning to rise",
        "description":        "GDP bottoming and starting to recover. Best time to add risk.",
        "historical_leaders": "Cyclicals, small-caps, and beaten-down growth stocks lead.",
    },
}


class MacroRegimeModel:

    def __init__(self):
        self._cache = {}

    def full_analysis(self, target_sector: str = None) -> dict:
        """Run complete macro regime analysis."""
        print(f"    Fetching macro indicators...")
        indicators  = self._fetch_macro_indicators()
        regime      = self._detect_regime(indicators)
        playbook    = REGIME_PLAYBOOK.get(regime["regime"], {})
        sector_rot  = self._sector_rotation_signals()
        sector_fit  = self._sector_fit(target_sector, regime["regime"], playbook)

        return {
            "regime":           regime["regime"],
            "regime_score":     regime["score"],
            "regime_signals":   regime["signals"],
            "description":      playbook.get("description", ""),
            "asset_bias":       playbook.get("asset_bias", ""),
            "favored_sectors":  playbook.get("favored_sectors", []),
            "avoid_sectors":    playbook.get("avoid_sectors", []),
            "historical_leaders": playbook.get("historical_leaders", ""),
            "sector_rotation":  sector_rot,
            "sector_fit":       sector_fit,
            "indicators":       indicators,
        }

    # ── MACRO INDICATORS ─────────────────────────────────────────

    def _fetch_macro_indicators(self) -> dict:
        """
        Fetch key macro indicators using market-based proxies (all free via yfinance).

        Yield curve:  10Y-2Y spread (^TNX - US2Y proxy)
        Credit spread: HYG vs LQD ratio
        Inflation:     TIPS ETF (TIP) momentum
        Growth:        S&P 500 trend + industrial ETF
        Volatility:    VIX level
        Dollar:        DXY (UUP ETF)
        Commodities:   DJP or GSG
        """
        indicators = {}

        try:
            # Fetch all at once
            symbols = ["^TNX", "^IRX", "^VIX", "SPY", "XLI", "TIP", "HYG", "LQD",
                       "UUP", "GSG", "GLD", "TLT", "DIA"]
            data = yf.download(symbols, period="1y", progress=False, auto_adjust=True)["Close"]

            # ── Yield Curve (10Y - 3M spread) ──────────────────
            if "^TNX" in data and "^IRX" in data:
                tnx_now   = float(data["^TNX"].dropna().iloc[-1])
                irx_now   = float(data["^IRX"].dropna().iloc[-1])
                spread    = tnx_now - irx_now
                spread_3m_ago = float((data["^TNX"] - data["^IRX"]).dropna().iloc[-63]) if len(data) >= 63 else spread

                indicators["yield_curve"] = {
                    "spread_10y_3m":     round(spread, 3),
                    "spread_3m_ago":     round(spread_3m_ago, 3),
                    "trend":             "Steepening" if spread > spread_3m_ago else "Flattening/Inverting",
                    "inverted":          spread < 0,
                    "signal":            "🔴 Inverted (recession risk)" if spread < 0
                                         else "🟡 Flat (caution)" if spread < 0.5
                                         else "🟢 Normal (expansion likely)",
                    "10y_rate":          round(tnx_now, 3),
                    "3m_rate":           round(irx_now, 3),
                }

            # ── VIX (Fear gauge) ────────────────────────────────
            if "^VIX" in data:
                vix_now = float(data["^VIX"].dropna().iloc[-1])
                vix_avg = float(data["^VIX"].dropna().tail(252).mean())
                indicators["vix"] = {
                    "current":     round(vix_now, 2),
                    "1y_average":  round(vix_avg, 2),
                    "vs_average":  round(vix_now - vix_avg, 2),
                    "regime":      "Crisis" if vix_now > 35 else
                                   "High Fear" if vix_now > 25 else
                                   "Elevated" if vix_now > 20 else
                                   "Normal" if vix_now > 15 else "Complacent",
                    "signal":      "🔴 Extreme fear" if vix_now > 35 else
                                   "🟠 Elevated fear" if vix_now > 20 else
                                   "🟢 Calm markets",
                }

            # ── S&P 500 Trend (Growth proxy) ────────────────────
            if "SPY" in data:
                spy = data["SPY"].dropna()
                spy_now    = float(spy.iloc[-1])
                spy_200d   = float(spy.tail(200).mean())
                spy_50d    = float(spy.tail(50).mean())
                spy_3m_ret = (spy_now - float(spy.iloc[-63])) / float(spy.iloc[-63]) if len(spy) >= 63 else 0
                spy_1y_ret = (spy_now - float(spy.iloc[-252])) / float(spy.iloc[-252]) if len(spy) >= 252 else 0

                indicators["market_trend"] = {
                    "spy_price":     round(spy_now, 2),
                    "vs_200d_ma":    round((spy_now - spy_200d) / spy_200d * 100, 2),
                    "vs_50d_ma":     round((spy_now - spy_50d) / spy_50d * 100, 2),
                    "3m_return":     round(spy_3m_ret * 100, 2),
                    "1y_return":     round(spy_1y_ret * 100, 2),
                    "trend":         "🟢 Bull Market" if spy_now > spy_200d * 1.02 else
                                     "🟡 Consolidating" if spy_now > spy_200d * 0.98 else
                                     "🔴 Bear Market",
                }

            # ── Credit Spreads (HYG vs LQD) ─────────────────────
            if "HYG" in data and "LQD" in data:
                hyg = data["HYG"].dropna()
                lqd = data["LQD"].dropna()
                ratio_now = float(hyg.iloc[-1]) / float(lqd.iloc[-1])
                ratio_3m  = float(hyg.iloc[-63]) / float(lqd.iloc[-63]) if len(hyg) >= 63 else ratio_now
                spread_chg = (ratio_now - ratio_3m) / ratio_3m

                indicators["credit"] = {
                    "hyg_lqd_ratio":  round(ratio_now, 4),
                    "3m_change":      round(spread_chg * 100, 2),
                    "signal":         "🟢 Credit healthy" if spread_chg > -0.02 else
                                      "🟠 Credit stress building" if spread_chg > -0.05 else
                                      "🔴 Credit tightening — risk-off signal",
                }

            # ── Inflation Proxy (TIP ETF) ────────────────────────
            if "TIP" in data and "TLT" in data:
                tip = data["TIP"].dropna()
                tlt = data["TLT"].dropna()
                tip_ret = (float(tip.iloc[-1]) - float(tip.iloc[-63])) / float(tip.iloc[-63]) if len(tip) >= 63 else 0
                tlt_ret = (float(tlt.iloc[-1]) - float(tlt.iloc[-63])) / float(tlt.iloc[-63]) if len(tlt) >= 63 else 0
                inflation_rising = tip_ret > tlt_ret

                indicators["inflation"] = {
                    "tip_3m_return":   round(tip_ret * 100, 2),
                    "tlt_3m_return":   round(tlt_ret * 100, 2),
                    "inflation_trend": "Rising" if inflation_rising else "Falling/Stable",
                    "signal":          "🔴 Inflation expectations rising" if inflation_rising else
                                       "🟢 Inflation expectations falling",
                }

            # ── Dollar Strength ──────────────────────────────────
            if "UUP" in data:
                uup = data["UUP"].dropna()
                uup_1m = (float(uup.iloc[-1]) - float(uup.iloc[-21])) / float(uup.iloc[-21]) if len(uup) >= 21 else 0
                indicators["dollar"] = {
                    "uup_1m_return": round(uup_1m * 100, 2),
                    "trend":         "Strengthening" if uup_1m > 0.01 else
                                     "Weakening" if uup_1m < -0.01 else "Neutral",
                    "signal":        "Strong dollar = headwind for commodities/EM" if uup_1m > 0.02 else
                                     "Weak dollar = tailwind for commodities/international" if uup_1m < -0.02 else
                                     "Dollar neutral",
                }

        except Exception as e:
            pass

        return indicators

    # ── REGIME DETECTION ─────────────────────────────────────────

    def _detect_regime(self, indicators: dict) -> dict:
        """Classify current macro regime based on growth + inflation signals."""
        growth_score = 0
        inflation_score = 0
        signals = []

        # Growth signals
        mkt = indicators.get("market_trend", {})
        if mkt:
            ret_3m = mkt.get("3m_return", 0)
            if ret_3m > 5:
                growth_score += 2; signals.append(f"SPY +{ret_3m:.1f}% (3M) — growth positive")
            elif ret_3m > 0:
                growth_score += 1
            elif ret_3m < -5:
                growth_score -= 2; signals.append(f"SPY {ret_3m:.1f}% (3M) — growth negative")
            else:
                growth_score -= 1

        yc = indicators.get("yield_curve", {})
        if yc:
            if yc.get("inverted"):
                growth_score -= 2; signals.append("Inverted yield curve — recession risk")
            elif yc.get("spread_10y_3m", 0) > 1.0:
                growth_score += 1; signals.append("Healthy yield curve spread")

        credit = indicators.get("credit", {})
        if credit:
            chg = credit.get("3m_change", 0)
            if chg > 0:
                growth_score += 1  # HY outperforming = risk on
            elif chg < -3:
                growth_score -= 1

        # Inflation signals
        infl = indicators.get("inflation", {})
        if infl:
            if infl.get("inflation_trend") == "Rising":
                inflation_score += 2; signals.append("Inflation expectations rising (TIPS outperforming)")
            else:
                inflation_score -= 1; signals.append("Inflation expectations falling/stable")

        vix = indicators.get("vix", {})
        if vix:
            vix_val = vix.get("current", 20)
            if vix_val > 25:
                growth_score -= 1; signals.append(f"VIX elevated at {vix_val:.0f}")

        # Classify
        if growth_score >= 1 and inflation_score <= 0:
            regime = "Expansion"
            score  = 75
        elif growth_score >= 1 and inflation_score >= 2:
            regime = "Overheating"
            score  = 60
        elif growth_score <= -1 and inflation_score >= 1:
            regime = "Stagflation"
            score  = 30
        elif growth_score <= -1 and inflation_score <= 0:
            regime = "Recession"
            score  = 25
        elif growth_score == 0 and inflation_score <= 0:
            regime = "Recovery"
            score  = 65
        else:
            regime = "Expansion"
            score  = 55

        return {"regime": regime, "score": score, "signals": signals,
                "growth_score": growth_score, "inflation_score": inflation_score}

    # ── SECTOR ROTATION ──────────────────────────────────────────

    def _sector_rotation_signals(self) -> dict:
        """Rank sectors by 1-month, 3-month, and 6-month momentum."""
        try:
            etf_list = list(SECTOR_ETFS.values())
            data = yf.download(etf_list, period="1y", progress=False, auto_adjust=True)["Close"]

            results = {}
            for sector, etf in SECTOR_ETFS.items():
                if etf not in data:
                    continue
                prices = data[etf].dropna()
                if len(prices) < 21:
                    continue

                ret_1m  = (float(prices.iloc[-1]) - float(prices.iloc[-21])) / float(prices.iloc[-21]) if len(prices) >= 21 else 0
                ret_3m  = (float(prices.iloc[-1]) - float(prices.iloc[-63])) / float(prices.iloc[-63]) if len(prices) >= 63 else 0
                ret_6m  = (float(prices.iloc[-1]) - float(prices.iloc[-126])) / float(prices.iloc[-126]) if len(prices) >= 126 else 0
                ret_1y  = (float(prices.iloc[-1]) - float(prices.iloc[-252])) / float(prices.iloc[-252]) if len(prices) >= 252 else 0

                # Relative strength vs SPY
                results[sector] = {
                    "etf":       etf,
                    "1m_return": round(ret_1m * 100, 2),
                    "3m_return": round(ret_3m * 100, 2),
                    "6m_return": round(ret_6m * 100, 2),
                    "1y_return": round(ret_1y * 100, 2),
                    "momentum_score": round((ret_1m * 0.4 + ret_3m * 0.35 + ret_6m * 0.25) * 100, 2),
                }

            # Rank by momentum
            sorted_sectors = sorted(results.items(),
                                    key=lambda x: x[1]["momentum_score"], reverse=True)

            ranked = {}
            for i, (sector, data_s) in enumerate(sorted_sectors, 1):
                data_s["rank"] = i
                data_s["signal"] = ("🟢 HOT" if i <= 3 else
                                    "🟡 WARM" if i <= 6 else
                                    "🔴 COLD")
                ranked[sector] = data_s

            return ranked

        except Exception:
            return {}

    # ── SECTOR FIT ───────────────────────────────────────────────

    def _sector_fit(self, target_sector: str, regime: str, playbook: dict) -> dict:
        """How well does the target stock's sector fit the current macro regime?"""
        if not target_sector:
            return {"assessment": "No sector provided"}

        favored = playbook.get("favored_sectors", [])
        avoid   = playbook.get("avoid_sectors", [])

        # Check for partial matches (e.g. "Technology" matches "Technology Hardware")
        is_favored = any(f.lower() in target_sector.lower() or target_sector.lower() in f.lower()
                         for f in favored)
        is_avoid   = any(a.lower() in target_sector.lower() or target_sector.lower() in a.lower()
                         for a in avoid)

        if is_favored:
            return {
                "assessment": f"🟢 TAILWIND — {target_sector} is a favored sector in {regime} regime",
                "macro_score_adjustment": +10,
            }
        elif is_avoid:
            return {
                "assessment": f"🔴 HEADWIND — {target_sector} is an avoid sector in {regime} regime",
                "macro_score_adjustment": -10,
            }
        else:
            return {
                "assessment": f"🟡 NEUTRAL — {target_sector} is neither favored nor avoided in {regime} regime",
                "macro_score_adjustment": 0,
            }
