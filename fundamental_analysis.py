"""
fundamental_analysis.py — Institutional-grade fundamental analysis.

Covers:
- Valuation ratios (P/E, P/B, EV/EBITDA, P/S, P/FCF)
- Growth metrics (revenue, earnings, FCF growth)
- Financial health (debt, coverage, liquidity)
- Profitability (margins, ROE, ROIC, ROA)
- Quality score (Piotroski F-Score style)
- Moat indicators
"""

import numpy as np
import pandas as pd
from typing import Optional


class FundamentalAnalyzer:

    # Industry average benchmarks for scoring
    BENCHMARKS = {
        "pe_ratio": {"excellent": 15, "good": 25, "fair": 35, "poor": 50},
        "pb_ratio": {"excellent": 1.5, "good": 3, "fair": 5, "poor": 10},
        "ev_ebitda": {"excellent": 8, "good": 14, "fair": 20, "poor": 30},
        "debt_equity": {"excellent": 0.3, "good": 0.8, "fair": 1.5, "poor": 3.0},
        "roe": {"excellent": 0.20, "good": 0.15, "fair": 0.10, "poor": 0.05},
        "gross_margin": {"excellent": 0.50, "good": 0.35, "fair": 0.20, "poor": 0.10},
        "fcf_yield": {"excellent": 0.07, "good": 0.04, "fair": 0.02, "poor": 0.0},
        "current_ratio": {"excellent": 2.0, "good": 1.5, "fair": 1.0, "poor": 0.8},
    }

    def __init__(self, info: dict, financials: dict):
        self.info = info or {}
        self.financials = financials or {}
        self.income = financials.get("income_stmt", pd.DataFrame())
        self.balance = financials.get("balance_sheet", pd.DataFrame())
        self.cashflow = financials.get("cash_flow", pd.DataFrame())

    # ─── PUBLIC ────────────────────────────────────────────────────────────────

    def full_analysis(self) -> dict:
        valuation = self._valuation()
        growth = self._growth_metrics()
        health = self._financial_health()
        profitability = self._profitability()
        quality = self._quality_score(valuation, growth, health, profitability)
        score = self._composite_fundamental_score(valuation, growth, health, profitability, quality)

        highlights = self._build_highlights(valuation, growth, profitability, health)

        return {
            "score": score,
            "valuation": valuation,
            "growth": growth,
            "health": health,
            "profitability": profitability,
            "quality_score": quality,
            "highlights": highlights,
        }

    # ─── VALUATION ─────────────────────────────────────────────────────────────

    def _valuation(self) -> dict:
        info = self.info
        v = {}

        v["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
        v["forward_pe"] = info.get("forwardPE")
        v["pb_ratio"] = info.get("priceToBook")
        v["ps_ratio"] = info.get("priceToSalesTrailing12Months")
        v["ev_ebitda"] = info.get("enterpriseToEbitda")
        v["ev_revenue"] = info.get("enterpriseToRevenue")

        # Price-to-FCF
        market_cap = info.get("marketCap", 0)
        fcf = self._safe_get(self.cashflow, "Free Cash Flow", 0)
        if market_cap and fcf and fcf > 0:
            v["p_fcf"] = market_cap / fcf
        else:
            v["p_fcf"] = None

        # FCF Yield
        if v["p_fcf"] and v["p_fcf"] > 0:
            v["fcf_yield"] = 1 / v["p_fcf"]
        else:
            v["fcf_yield"] = None

        # PEG Ratio
        v["peg_ratio"] = info.get("pegRatio")

        # Dividend yield
        v["dividend_yield"] = info.get("dividendYield")

        return v

    # ─── GROWTH ────────────────────────────────────────────────────────────────

    def _growth_metrics(self) -> dict:
        g = {}

        # Use yfinance provided growth rates
        g["revenue_growth_yoy"] = self.info.get("revenueGrowth")
        g["earnings_growth_yoy"] = self.info.get("earningsGrowth")
        g["eps_growth"] = self.info.get("earningsQuarterlyGrowth")

        # Calculate historical revenue CAGR from income stmt
        if not self.income.empty and "Total Revenue" in self.income.index:
            rev = self.income.loc["Total Revenue"].dropna()
            if len(rev) >= 2:
                try:
                    rev_vals = rev.values
                    # Most recent vs oldest (yfinance cols are dates, newest first)
                    oldest = float(rev_vals[-1])
                    newest = float(rev_vals[0])
                    years = len(rev_vals) - 1
                    if oldest > 0 and newest > 0 and years > 0:
                        g["revenue_cagr_3y"] = (newest / oldest) ** (1 / years) - 1
                except Exception:
                    pass

        # FCF growth
        if not self.cashflow.empty:
            fcf_row = self._find_row(self.cashflow, ["Free Cash Flow", "FreeCashFlow"])
            if fcf_row is not None:
                fcf_vals = fcf_row.dropna().values
                if len(fcf_vals) >= 2:
                    try:
                        if float(fcf_vals[-1]) > 0 and float(fcf_vals[0]) > 0:
                            g["fcf_cagr_3y"] = (float(fcf_vals[0]) / float(fcf_vals[-1])) ** (1 / max(len(fcf_vals) - 1, 1)) - 1
                    except Exception:
                        pass

        return g

    # ─── FINANCIAL HEALTH ──────────────────────────────────────────────────────

    def _financial_health(self) -> dict:
        h = {}
        info = self.info

        h["current_ratio"] = info.get("currentRatio")
        h["quick_ratio"] = info.get("quickRatio")
        h["debt_equity"] = info.get("debtToEquity")
        if h["debt_equity"]:
            h["debt_equity"] = h["debt_equity"] / 100  # yfinance returns as percentage

        # Interest coverage = EBIT / Interest Expense
        if not self.income.empty:
            ebit = self._safe_get(self.income, "EBIT", None) or self._safe_get(self.income, "Operating Income", None)
            interest = self._safe_get(self.income, "Interest Expense", None)
            if ebit and interest and interest != 0:
                h["interest_coverage"] = abs(ebit / interest)

        # Total debt / total assets
        if not self.balance.empty:
            total_debt = self._safe_get(self.balance, "Total Debt", None)
            total_assets = self._safe_get(self.balance, "Total Assets", None)
            if total_debt and total_assets and total_assets > 0:
                h["debt_assets_ratio"] = total_debt / total_assets

        h["total_cash"] = info.get("totalCash")
        h["total_debt"] = info.get("totalDebt")
        h["net_cash"] = (h["total_cash"] or 0) - (h["total_debt"] or 0)

        return h

    # ─── PROFITABILITY ─────────────────────────────────────────────────────────

    def _profitability(self) -> dict:
        p = {}
        info = self.info

        p["gross_margin"] = info.get("grossMargins")
        p["operating_margin"] = info.get("operatingMargins")
        p["net_margin"] = info.get("profitMargins")
        p["roe"] = info.get("returnOnEquity")
        p["roa"] = info.get("returnOnAssets")

        # ROIC approximation = Net Income / (Total Equity + Total Debt)
        if not self.balance.empty and not self.income.empty:
            equity = self._safe_get(self.balance, "Stockholders Equity", None) or \
                     self._safe_get(self.balance, "Total Stockholders Equity", None)
            debt = self._safe_get(self.balance, "Total Debt", None)
            net_income = self._safe_get(self.income, "Net Income", None)
            if equity and debt and net_income:
                invested_capital = equity + debt
                if invested_capital > 0:
                    p["roic"] = net_income / invested_capital

        return p

    # ─── QUALITY SCORE (Piotroski F-Score inspired) ────────────────────────────

    def _quality_score(self, valuation, growth, health, profitability) -> dict:
        """Score each dimension 0-9 (Piotroski-style)."""
        score = 0
        signals = []

        # Profitability signals
        if (profitability.get("roe") or 0) > 0.15:
            score += 1; signals.append("Strong ROE (>15%)")
        if (profitability.get("net_margin") or 0) > 0.10:
            score += 1; signals.append("Healthy net margin (>10%)")
        if (profitability.get("roa") or 0) > 0.05:
            score += 1; signals.append("Positive ROA (>5%)")

        # Leverage / health signals
        de = health.get("debt_equity")
        if de is not None and de < 1.0:
            score += 1; signals.append("Conservative debt/equity (<1.0)")
        cr = health.get("current_ratio")
        if cr is not None and cr > 1.5:
            score += 1; signals.append("Strong liquidity (current ratio >1.5)")
        ic = health.get("interest_coverage")
        if ic is not None and ic > 5:
            score += 1; signals.append("High interest coverage (>5x)")

        # Growth signals
        if (growth.get("revenue_growth_yoy") or 0) > 0.10:
            score += 1; signals.append("Revenue growth >10% YoY")
        if (growth.get("earnings_growth_yoy") or 0) > 0.10:
            score += 1; signals.append("Earnings growth >10% YoY")

        # Valuation signal (not overpriced)
        pe = valuation.get("pe_ratio")
        if pe and 5 < pe < 25:
            score += 1; signals.append("Reasonable P/E (5-25)")

        return {
            "f_score": score,
            "f_score_max": 9,
            "interpretation": self._interpret_f_score(score),
            "signals": signals,
        }

    def _interpret_f_score(self, score: int) -> str:
        if score >= 8:   return "Very Strong (High Quality)"
        elif score >= 6: return "Strong"
        elif score >= 4: return "Average"
        elif score >= 2: return "Weak"
        else:            return "Very Weak (Poor Quality)"

    # ─── COMPOSITE SCORE ───────────────────────────────────────────────────────

    def _composite_fundamental_score(self, valuation, growth, health, profitability, quality) -> float:
        """Produce a 0-100 fundamental score."""
        score = 0.0

        # Valuation component (25 pts)
        pe = valuation.get("pe_ratio")
        if pe:
            if pe < 15:      score += 25
            elif pe < 25:    score += 20
            elif pe < 35:    score += 12
            elif pe < 50:    score += 5

        # Growth component (30 pts)
        rev_g = growth.get("revenue_growth_yoy") or 0
        earn_g = growth.get("earnings_growth_yoy") or 0
        avg_growth = (rev_g + earn_g) / 2
        if avg_growth > 0.25:    score += 30
        elif avg_growth > 0.15:  score += 24
        elif avg_growth > 0.08:  score += 16
        elif avg_growth > 0.0:   score += 8

        # Profitability (20 pts)
        roe = profitability.get("roe") or 0
        margin = profitability.get("net_margin") or 0
        if roe > 0.20 and margin > 0.10: score += 20
        elif roe > 0.12 and margin > 0.05: score += 14
        elif roe > 0.05: score += 8

        # Financial health (15 pts)
        de = health.get("debt_equity")
        cr = health.get("current_ratio")
        if (de is not None and de < 0.5) and (cr is not None and cr > 2):
            score += 15
        elif (de is not None and de < 1.0) and (cr is not None and cr > 1.2):
            score += 10
        elif cr is not None and cr > 1.0:
            score += 5

        # Quality F-Score (10 pts)
        f = quality.get("f_score", 0)
        score += (f / 9) * 10

        return round(min(score, 100), 2)

    # ─── HIGHLIGHTS ────────────────────────────────────────────────────────────

    def _build_highlights(self, valuation, growth, profitability, health) -> dict:
        h = {}
        pe = valuation.get("pe_ratio")
        if pe:   h["P/E Ratio"] = f"{pe:.1f}x"
        pb = valuation.get("pb_ratio")
        if pb:   h["P/B Ratio"] = f"{pb:.1f}x"
        ev = valuation.get("ev_ebitda")
        if ev:   h["EV/EBITDA"] = f"{ev:.1f}x"
        rg = growth.get("revenue_growth_yoy")
        if rg:   h["Revenue Growth YoY"] = f"{rg*100:.1f}%"
        eg = growth.get("earnings_growth_yoy")
        if eg:   h["Earnings Growth YoY"] = f"{eg*100:.1f}%"
        roe = profitability.get("roe")
        if roe:  h["ROE"] = f"{roe*100:.1f}%"
        nm = profitability.get("net_margin")
        if nm:   h["Net Margin"] = f"{nm*100:.1f}%"
        de = health.get("debt_equity")
        if de:   h["Debt/Equity"] = f"{de:.2f}"
        cr = health.get("current_ratio")
        if cr:   h["Current Ratio"] = f"{cr:.2f}"
        return h

    # ─── HELPERS ───────────────────────────────────────────────────────────────

    def _safe_get(self, df: pd.DataFrame, key: str, default):
        """Safely get the most recent value from a financial statement row."""
        try:
            if key in df.index:
                val = df.loc[key].dropna()
                if len(val) > 0:
                    return float(val.iloc[0])
        except Exception:
            pass
        return default

    def _find_row(self, df: pd.DataFrame, possible_keys: list):
        """Find a row by any of several possible key names."""
        for key in possible_keys:
            if key in df.index:
                return df.loc[key]
        return None
