"""
growth_quality_scorer.py — Buffett-Meets-Growth Conviction Scoring Engine

Six-pillar investment framework combining growth stock criteria with
Buffett-style quality and valuation analysis.

Pillars:
  1. Revenue & Earnings Growth  — Consistent scaling, margin expansion
  2. Total Addressable Market    — Room to grow, secular trend strength
  3. Competitive Moat (Buffett)  — 5 moat types scored: brand, network, switching, patents, cost
  4. Management Quality          — Insider ownership, founder-led, capital allocation
  5. Return on Capital           — ROIC, ROE, operating margin trends
  6. Valuation (Fair Price)      — Intrinsic value vs price, margin of safety, PEG

Each pillar scores 0-100. Weighted composite = Conviction Score.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple


# ─── TAM DATABASE ──────────────────────────────────────────────────────────────
# Industry-level TAM estimates (in $B), growth rate, and maturity stage.
# Source: aggregated from public market research reports.

TAM_DATABASE = {
    # Technology
    "Semiconductors": {"tam_2024": 600, "tam_2030": 1200, "cagr": 0.12, "stage": "expanding"},
    "Software—Application": {"tam_2024": 350, "tam_2030": 700, "cagr": 0.12, "stage": "expanding"},
    "Software—Infrastructure": {"tam_2024": 250, "tam_2030": 580, "cagr": 0.15, "stage": "expanding"},
    "Internet Content & Information": {"tam_2024": 600, "tam_2030": 1100, "cagr": 0.11, "stage": "expanding"},
    "Consumer Electronics": {"tam_2024": 800, "tam_2030": 1050, "cagr": 0.05, "stage": "mature"},
    "Computer Hardware": {"tam_2024": 180, "tam_2030": 400, "cagr": 0.14, "stage": "expanding"},
    "Information Technology Services": {"tam_2024": 1200, "tam_2030": 1900, "cagr": 0.08, "stage": "growing"},
    "Electronic Components": {"tam_2024": 200, "tam_2030": 350, "cagr": 0.10, "stage": "growing"},

    # AI & Cloud (sub-sectors, matched via keywords)
    "Artificial Intelligence": {"tam_2024": 200, "tam_2030": 1500, "cagr": 0.40, "stage": "early_expansion"},
    "Cloud Computing": {"tam_2024": 600, "tam_2030": 1600, "cagr": 0.17, "stage": "expanding"},
    "Cybersecurity": {"tam_2024": 180, "tam_2030": 400, "cagr": 0.14, "stage": "expanding"},

    # Healthcare
    "Drug Manufacturers—General": {"tam_2024": 1400, "tam_2030": 2100, "cagr": 0.07, "stage": "growing"},
    "Drug Manufacturers—Specialty & Generic": {"tam_2024": 400, "tam_2030": 550, "cagr": 0.05, "stage": "mature"},
    "Biotechnology": {"tam_2024": 500, "tam_2030": 1100, "cagr": 0.14, "stage": "expanding"},
    "Medical Devices": {"tam_2024": 500, "tam_2030": 750, "cagr": 0.07, "stage": "growing"},
    "Health Information Services": {"tam_2024": 100, "tam_2030": 230, "cagr": 0.15, "stage": "expanding"},
    "Healthcare Plans": {"tam_2024": 1200, "tam_2030": 1600, "cagr": 0.05, "stage": "mature"},

    # Financial Services
    "Banks—Diversified": {"tam_2024": 7000, "tam_2030": 8500, "cagr": 0.03, "stage": "mature"},
    "Capital Markets": {"tam_2024": 600, "tam_2030": 900, "cagr": 0.07, "stage": "growing"},
    "Insurance—Diversified": {"tam_2024": 6000, "tam_2030": 7500, "cagr": 0.04, "stage": "mature"},
    "Financial Data & Stock Exchanges": {"tam_2024": 120, "tam_2030": 220, "cagr": 0.11, "stage": "expanding"},
    "Credit Services": {"tam_2024": 500, "tam_2030": 800, "cagr": 0.08, "stage": "growing"},
    "Fintech": {"tam_2024": 300, "tam_2030": 900, "cagr": 0.20, "stage": "early_expansion"},

    # Energy
    "Oil & Gas Integrated": {"tam_2024": 3000, "tam_2030": 2800, "cagr": -0.01, "stage": "declining"},
    "Oil & Gas E&P": {"tam_2024": 500, "tam_2030": 480, "cagr": -0.01, "stage": "declining"},
    "Solar": {"tam_2024": 200, "tam_2030": 600, "cagr": 0.20, "stage": "early_expansion"},
    "Utilities—Renewable": {"tam_2024": 300, "tam_2030": 700, "cagr": 0.15, "stage": "expanding"},

    # Consumer
    "Auto Manufacturers": {"tam_2024": 2500, "tam_2030": 3500, "cagr": 0.06, "stage": "growing"},
    "Internet Retail": {"tam_2024": 5500, "tam_2030": 9000, "cagr": 0.09, "stage": "growing"},
    "Specialty Retail": {"tam_2024": 3000, "tam_2030": 3600, "cagr": 0.03, "stage": "mature"},
    "Restaurants": {"tam_2024": 900, "tam_2030": 1200, "cagr": 0.05, "stage": "mature"},
    "Entertainment": {"tam_2024": 300, "tam_2030": 500, "cagr": 0.09, "stage": "growing"},

    # Industrials
    "Aerospace & Defense": {"tam_2024": 800, "tam_2030": 1100, "cagr": 0.05, "stage": "growing"},
    "Railroads": {"tam_2024": 250, "tam_2030": 290, "cagr": 0.03, "stage": "mature"},

    # Communication
    "Telecom Services": {"tam_2024": 1700, "tam_2030": 2000, "cagr": 0.03, "stage": "mature"},
}

# Keywords that map business descriptions to high-growth sub-TAMs
TAM_KEYWORD_BOOSTS = {
    "artificial intelligence": "Artificial Intelligence",
    "machine learning": "Artificial Intelligence",
    "generative ai": "Artificial Intelligence",
    "cloud": "Cloud Computing",
    "cloud computing": "Cloud Computing",
    "cybersecurity": "Cybersecurity",
    "fintech": "Fintech",
    "digital payments": "Fintech",
    "blockchain": "Fintech",
    "solar energy": "Solar",
    "electric vehicle": "Auto Manufacturers",
}


SEC_HEADERS = {
    "User-Agent": "InstitutionalEdge research@institutionaledge.com",
    "Accept-Encoding": "gzip, deflate",
}


class GrowthQualityScorer:
    """
    Six-pillar Buffett-meets-Growth conviction scoring engine.

    Usage:
        scorer = GrowthQualityScorer(ticker, info, financials, stock_data)
        result = scorer.full_analysis()
    """

    # Weight each pillar in the final conviction score
    PILLAR_WEIGHTS = {
        "growth":       0.22,
        "tam":          0.12,
        "moat":         0.22,
        "management":   0.12,
        "returns":      0.17,
        "valuation":    0.15,
    }

    def __init__(self, ticker: str, info: dict, financials: dict,
                 stock_data: pd.DataFrame):
        self.ticker = ticker.upper()
        self.info = info or {}
        self.financials = financials or {}
        self.stock_data = stock_data
        self.income = financials.get("income_stmt", pd.DataFrame())
        self.balance = financials.get("balance_sheet", pd.DataFrame())
        self.cashflow = financials.get("cash_flow", pd.DataFrame())

    # ═══════════════════════════════════════════════════════════════════════════
    #   PUBLIC
    # ═══════════════════════════════════════════════════════════════════════════

    def full_analysis(self) -> dict:
        """Run all six pillars and produce conviction score."""
        growth      = self._pillar_growth()
        tam         = self._pillar_tam()
        moat        = self._pillar_moat()
        management  = self._pillar_management()
        returns     = self._pillar_returns()
        valuation   = self._pillar_valuation()

        pillars = {
            "growth":      growth,
            "tam":         tam,
            "moat":        moat,
            "management":  management,
            "returns":     returns,
            "valuation":   valuation,
        }

        # Weighted conviction score
        conviction = sum(
            pillars[p]["score"] * self.PILLAR_WEIGHTS[p]
            for p in self.PILLAR_WEIGHTS
        )
        conviction = round(min(max(conviction, 0), 100), 1)

        # Generate investment thesis
        thesis = self._generate_thesis(pillars, conviction)

        return {
            "score": conviction,
            "conviction_label": self._conviction_label(conviction),
            "pillars": pillars,
            "thesis": thesis,
            "strengths": self._get_strengths(pillars),
            "weaknesses": self._get_weaknesses(pillars),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   PILLAR 1: REVENUE & EARNINGS GROWTH
    # ═══════════════════════════════════════════════════════════════════════════

    def _pillar_growth(self) -> dict:
        """Score revenue growth, EPS growth, cash flow growth, margin expansion."""
        score = 0
        signals = []
        metrics = {}

        # ── Revenue Growth ────────────────────────────────────────────
        rev_growth = self.info.get("revenueGrowth")
        metrics["revenue_growth_yoy"] = rev_growth

        # Multi-year revenue CAGR
        rev_cagr = self._calc_cagr("Total Revenue")
        metrics["revenue_cagr"] = rev_cagr

        if rev_growth is not None:
            if rev_growth > 0.30:
                score += 20
                signals.append(f"Exceptional revenue growth ({rev_growth*100:.0f}% YoY)")
            elif rev_growth > 0.15:
                score += 15
                signals.append(f"Strong revenue growth ({rev_growth*100:.0f}% YoY)")
            elif rev_growth > 0.08:
                score += 10
                signals.append(f"Moderate revenue growth ({rev_growth*100:.0f}% YoY)")
            elif rev_growth > 0:
                score += 5
                signals.append(f"Slow revenue growth ({rev_growth*100:.0f}% YoY)")
            else:
                signals.append(f"Revenue declining ({rev_growth*100:.0f}% YoY)")

        if rev_cagr and rev_cagr > 0.15:
            score += 5
            signals.append(f"Revenue CAGR {rev_cagr*100:.0f}% over multi-year period")

        # ── EPS Growth ────────────────────────────────────────────────
        eps_growth = self.info.get("earningsGrowth")
        eps_qtr = self.info.get("earningsQuarterlyGrowth")
        metrics["eps_growth_annual"] = eps_growth
        metrics["eps_growth_quarterly"] = eps_qtr

        if eps_growth is not None:
            if eps_growth > 0.25:
                score += 15
                signals.append(f"Exceptional EPS growth ({eps_growth*100:.0f}%)")
            elif eps_growth > 0.10:
                score += 10
                signals.append(f"Solid EPS growth ({eps_growth*100:.0f}%)")
            elif eps_growth > 0:
                score += 5

        # ── FCF Growth ────────────────────────────────────────────────
        fcf_cagr = self._calc_cagr("Free Cash Flow")
        metrics["fcf_cagr"] = fcf_cagr

        if fcf_cagr is not None:
            if fcf_cagr > 0.20:
                score += 15
                signals.append(f"Strong FCF compounding ({fcf_cagr*100:.0f}% CAGR)")
            elif fcf_cagr > 0.10:
                score += 10
                signals.append(f"Healthy FCF growth ({fcf_cagr*100:.0f}% CAGR)")
            elif fcf_cagr > 0:
                score += 5

        # ── Margin Expansion ─────────────────────────────────────────
        margin_trend = self._margin_expansion()
        metrics["margin_trend"] = margin_trend

        if margin_trend.get("expanding"):
            score += 15
            signals.append("Expanding profit margins — operating leverage")
        elif margin_trend.get("stable"):
            score += 8
            signals.append("Stable margins")
        else:
            signals.append("Margins compressing — watch for cost pressure")

        # ── Growth Consistency ────────────────────────────────────────
        consistency = self._revenue_consistency()
        metrics["growth_consistency"] = consistency

        if consistency.get("consecutive_growth_years", 0) >= 4:
            score += 10
            signals.append(f"{consistency['consecutive_growth_years']} consecutive years of revenue growth")
        elif consistency.get("consecutive_growth_years", 0) >= 2:
            score += 5

        score = min(score, 100)

        return {
            "score": score,
            "label": "Revenue & Earnings Growth",
            "emoji": "",
            "signals": signals,
            "metrics": metrics,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   PILLAR 2: TOTAL ADDRESSABLE MARKET
    # ═══════════════════════════════════════════════════════════════════════════

    def _pillar_tam(self) -> dict:
        """Score the TAM opportunity: industry size, growth rate, company penetration."""
        score = 0
        signals = []
        metrics = {}

        industry = self.info.get("industry", "")
        sector = self.info.get("sector", "")
        description = self.info.get("longBusinessSummary", "").lower()
        market_cap = self.info.get("marketCap", 0)
        revenue = self.info.get("totalRevenue", 0)

        # Match to TAM database
        tam_info = self._match_tam(industry, description)
        metrics["matched_industry"] = tam_info.get("matched", industry)

        if tam_info.get("tam_2024"):
            tam_b = tam_info["tam_2024"]
            tam_2030 = tam_info.get("tam_2030", tam_b)
            tam_cagr = tam_info.get("cagr", 0)
            stage = tam_info.get("stage", "unknown")
            metrics["tam_2024_billions"] = tam_b
            metrics["tam_2030_billions"] = tam_2030
            metrics["tam_cagr"] = tam_cagr
            metrics["market_stage"] = stage

            # Market penetration (revenue / TAM)
            if revenue and tam_b:
                penetration = (revenue / 1e9) / tam_b
                metrics["market_penetration_pct"] = round(penetration * 100, 2)

                # Low penetration = more room to grow
                if penetration < 0.01:
                    score += 20
                    signals.append(f"Tiny market share ({penetration*100:.2f}%) — massive runway")
                elif penetration < 0.05:
                    score += 15
                    signals.append(f"Low market share ({penetration*100:.1f}%) — significant growth room")
                elif penetration < 0.15:
                    score += 10
                    signals.append(f"Moderate market share ({penetration*100:.1f}%)")
                else:
                    score += 5
                    signals.append(f"High market share ({penetration*100:.1f}%) — growth capped by TAM")

            # TAM growth rate
            if tam_cagr > 0.15:
                score += 25
                signals.append(f"TAM growing at {tam_cagr*100:.0f}% CAGR — strong secular tailwind")
            elif tam_cagr > 0.08:
                score += 18
                signals.append(f"TAM growing at {tam_cagr*100:.0f}% CAGR — healthy tailwind")
            elif tam_cagr > 0.03:
                score += 10
                signals.append(f"TAM growing at {tam_cagr*100:.0f}% CAGR — moderate")
            elif tam_cagr > 0:
                score += 5
                signals.append("Slow TAM growth — limited industry expansion")
            else:
                signals.append("TAM shrinking — structural headwinds")

            # Market stage scoring
            stage_scores = {
                "early_expansion": 25,
                "expanding": 20,
                "growing": 12,
                "mature": 5,
                "declining": 0,
            }
            stage_score = stage_scores.get(stage, 10)
            score += stage_score
            signals.append(f"Market stage: {stage.replace('_', ' ').title()}")

            # TAM size — bigger markets can produce bigger winners
            if tam_2030 > 1000:
                score += 10
                signals.append(f"${tam_2030:,.0f}B addressable market by 2030 — mega opportunity")
            elif tam_2030 > 500:
                score += 7
            elif tam_2030 > 100:
                score += 4

        else:
            # No TAM match — use sector heuristics
            score += 30  # neutral default
            signals.append("Industry TAM not in database — using sector heuristics")

        score = min(score, 100)

        return {
            "score": score,
            "label": "Total Addressable Market",
            "emoji": "",
            "signals": signals,
            "metrics": metrics,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   PILLAR 3: COMPETITIVE MOAT (BUFFETT-STYLE)
    # ═══════════════════════════════════════════════════════════════════════════

    def _pillar_moat(self) -> dict:
        """
        Score five moat types:
          1. Brand Power — premium pricing, gross margins, brand value
          2. Network Effects — user base growth, platform dynamics
          3. Switching Costs — customer retention, recurring revenue
          4. Patents / IP — R&D intensity, technology moat
          5. Cost Advantage — scale, operating efficiency, low-cost producer
        """
        score = 0
        signals = []
        moat_types = {}

        info = self.info
        industry = info.get("industry", "")

        # ── 1. Brand Power (20 pts max) ──────────────────────────────
        brand_score = 0
        gm = info.get("grossMargins") or 0
        if gm > 0.65:
            brand_score = 20
            signals.append(f"Elite gross margins ({gm*100:.0f}%) — exceptional brand/pricing power")
        elif gm > 0.50:
            brand_score = 15
            signals.append(f"Strong gross margins ({gm*100:.0f}%) — good pricing power")
        elif gm > 0.35:
            brand_score = 10
        elif gm > 0.20:
            brand_score = 5
        moat_types["brand_power"] = brand_score

        # ── 2. Network Effects (20 pts max) ──────────────────────────
        network_score = 0
        # Proxy: companies in social, payments, marketplaces likely have network effects
        network_industries = [
            "Internet Content", "Software—Application", "Credit Services",
            "Financial Data", "Internet Retail", "Entertainment",
        ]
        has_network_proxy = any(ni in industry for ni in network_industries)

        if has_network_proxy:
            # High revenue per employee = platform efficiency (network effects)
            employees = info.get("fullTimeEmployees")
            revenue = info.get("totalRevenue")
            if employees and revenue and employees > 0:
                rev_per_emp = revenue / employees
                if rev_per_emp > 1_000_000:
                    network_score = 20
                    signals.append(f"${rev_per_emp/1e6:.1f}M rev/employee — strong platform leverage")
                elif rev_per_emp > 500_000:
                    network_score = 15
                elif rev_per_emp > 200_000:
                    network_score = 10
                else:
                    network_score = 5
            else:
                network_score = 10  # Industry suggests network effects
        moat_types["network_effects"] = network_score

        # ── 3. Switching Costs (20 pts max) ──────────────────────────
        switching_score = 0
        # High switching costs proxied by: recurring revenue, high retention,
        # enterprise software, healthcare, financial infrastructure
        switching_industries = [
            "Software", "Healthcare Plans", "Banks", "Insurance",
            "Information Technology", "Financial Data",
        ]
        has_switching = any(si in industry for si in switching_industries)

        if has_switching:
            # Stable/growing revenue + high margins = sticky customers
            rev_growth = info.get("revenueGrowth") or 0
            op_margin = info.get("operatingMargins") or 0
            if rev_growth > 0.05 and op_margin > 0.20:
                switching_score = 20
                signals.append("High switching costs — sticky recurring revenue")
            elif rev_growth > 0 and op_margin > 0.10:
                switching_score = 14
                signals.append("Moderate switching costs")
            else:
                switching_score = 8
        moat_types["switching_costs"] = switching_score

        # ── 4. Patents / IP / Technology (20 pts max) ────────────────
        ip_score = 0
        # R&D intensity as proxy for tech moat
        rd_revenue_ratio = self._rd_intensity()
        if rd_revenue_ratio is not None:
            if rd_revenue_ratio > 0.20:
                ip_score = 20
                signals.append(f"Heavy R&D investment ({rd_revenue_ratio*100:.0f}% of revenue) — building tech moat")
            elif rd_revenue_ratio > 0.10:
                ip_score = 15
                signals.append(f"Strong R&D spend ({rd_revenue_ratio*100:.0f}% of revenue)")
            elif rd_revenue_ratio > 0.05:
                ip_score = 10
            else:
                ip_score = 5

        # Sector-specific IP bonus
        ip_industries = ["Semiconductor", "Biotechnology", "Drug Manufacturers", "Aerospace"]
        if any(ipi in industry for ipi in ip_industries):
            ip_score = min(ip_score + 5, 20)
        moat_types["patents_ip"] = ip_score

        # ── 5. Cost Advantage / Scale (20 pts max) ───────────────────
        cost_score = 0
        market_cap = info.get("marketCap") or 0
        op_margin = info.get("operatingMargins") or 0

        # Mega-cap + high margins = scale advantage
        if market_cap > 500e9:
            cost_score += 10
        elif market_cap > 100e9:
            cost_score += 7
        elif market_cap > 20e9:
            cost_score += 4

        if op_margin > 0.30:
            cost_score += 10
            signals.append(f"Exceptional operating margins ({op_margin*100:.0f}%) — cost/scale advantage")
        elif op_margin > 0.20:
            cost_score += 7
        elif op_margin > 0.10:
            cost_score += 4

        cost_score = min(cost_score, 20)
        moat_types["cost_advantage"] = cost_score

        # ── Aggregate ────────────────────────────────────────────────
        score = sum(moat_types.values())
        score = min(score, 100)

        # Identify dominant moat type
        dominant = max(moat_types, key=moat_types.get)
        moat_label_map = {
            "brand_power": "Brand Power",
            "network_effects": "Network Effects",
            "switching_costs": "Switching Costs",
            "patents_ip": "Patents & IP",
            "cost_advantage": "Scale & Cost Advantage",
        }

        # Moat width classification
        if score >= 70:
            width = "Wide Moat"
        elif score >= 50:
            width = "Narrow Moat"
        elif score >= 30:
            width = "Thin Moat"
        else:
            width = "No Moat"

        signals.insert(0, f"{width} — primary: {moat_label_map.get(dominant, dominant)}")

        return {
            "score": score,
            "label": "Competitive Moat",
            "emoji": "",
            "width": width,
            "dominant_moat": moat_label_map.get(dominant, dominant),
            "moat_breakdown": moat_types,
            "signals": signals,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   PILLAR 4: MANAGEMENT QUALITY
    # ═══════════════════════════════════════════════════════════════════════════

    def _pillar_management(self) -> dict:
        """
        Score management quality:
          - Insider ownership %
          - Founder-led status
          - Capital allocation (buybacks, dividends, reinvestment efficiency)
          - Insider buying signals (from SEC data already available)
        """
        score = 0
        signals = []
        metrics = {}

        info = self.info

        # ── Insider Ownership ────────────────────────────────────────
        insider_pct = info.get("heldPercentInsiders")
        metrics["insider_ownership_pct"] = insider_pct

        if insider_pct is not None:
            if insider_pct > 0.15:
                score += 25
                signals.append(f"High insider ownership ({insider_pct*100:.1f}%) — strong alignment")
            elif insider_pct > 0.05:
                score += 18
                signals.append(f"Meaningful insider ownership ({insider_pct*100:.1f}%)")
            elif insider_pct > 0.01:
                score += 10
                signals.append(f"Low insider ownership ({insider_pct*100:.1f}%)")
            else:
                score += 3
                signals.append("Minimal insider ownership — watch for agency risk")

        # ── Institutional Ownership (smart money vote) ───────────────
        inst_pct = info.get("heldPercentInstitutions")
        metrics["institutional_ownership_pct"] = inst_pct

        if inst_pct is not None:
            if inst_pct > 0.70:
                score += 10
                signals.append(f"Strong institutional backing ({inst_pct*100:.0f}%)")
            elif inst_pct > 0.40:
                score += 7
            elif inst_pct > 0.20:
                score += 4

        # ── Capital Allocation: Buybacks ─────────────────────────────
        buyback_score = self._score_buybacks()
        score += buyback_score["score"]
        if buyback_score.get("signal"):
            signals.append(buyback_score["signal"])
        metrics["buyback_yield"] = buyback_score.get("buyback_yield")

        # ── Capital Allocation: Dividend Discipline ──────────────────
        div_yield = info.get("dividendYield") or 0
        payout_ratio = info.get("payoutRatio") or 0
        metrics["dividend_yield"] = div_yield
        metrics["payout_ratio"] = payout_ratio

        if div_yield > 0:
            if payout_ratio < 0.60:
                score += 10
                signals.append(f"Sustainable dividend ({div_yield*100:.1f}% yield, {payout_ratio*100:.0f}% payout)")
            elif payout_ratio < 0.80:
                score += 5
                signals.append(f"Dividend yield {div_yield*100:.1f}% but payout ratio elevated")
            else:
                signals.append("High payout ratio — dividend sustainability at risk")
        else:
            # Growth companies reinvesting — not necessarily bad
            rev_growth = info.get("revenueGrowth") or 0
            if rev_growth > 0.15:
                score += 10
                signals.append("No dividend — reinvesting in high-growth business (appropriate)")
            elif rev_growth > 0:
                score += 5

        # ── ROE Consistency (capital allocation quality) ─────────────
        roe = info.get("returnOnEquity") or 0
        roic = self._calc_roic()
        if roe > 0.20 and (roic or 0) > 0.15:
            score += 10
            signals.append("Excellent returns on deployed capital — management creating value")
        elif roe > 0.12:
            score += 5

        score = min(score, 100)

        return {
            "score": score,
            "label": "Management Quality",
            "emoji": "",
            "signals": signals,
            "metrics": metrics,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   PILLAR 5: RETURN ON CAPITAL
    # ═══════════════════════════════════════════════════════════════════════════

    def _pillar_returns(self) -> dict:
        """
        Score capital efficiency: ROIC, ROE, operating margin trends.
        Great companies turn every dollar invested into more profit.
        """
        score = 0
        signals = []
        metrics = {}

        info = self.info

        # ── ROIC ────────────────────────────────────────────────────
        roic = self._calc_roic()
        metrics["roic"] = roic

        if roic is not None:
            if roic > 0.25:
                score += 25
                signals.append(f"Exceptional ROIC ({roic*100:.0f}%) — elite capital efficiency")
            elif roic > 0.15:
                score += 20
                signals.append(f"Strong ROIC ({roic*100:.0f}%) — high-quality compounder")
            elif roic > 0.10:
                score += 14
                signals.append(f"Decent ROIC ({roic*100:.0f}%)")
            elif roic > 0.05:
                score += 8
            elif roic > 0:
                score += 3
            else:
                signals.append("Negative ROIC — destroying shareholder value")

        # ── ROE ─────────────────────────────────────────────────────
        roe = info.get("returnOnEquity")
        metrics["roe"] = roe

        if roe is not None:
            if roe > 0.30:
                score += 20
                signals.append(f"Exceptional ROE ({roe*100:.0f}%)")
            elif roe > 0.20:
                score += 15
            elif roe > 0.12:
                score += 10
            elif roe > 0.05:
                score += 5

        # ── Operating Margin ────────────────────────────────────────
        op_margin = info.get("operatingMargins")
        metrics["operating_margin"] = op_margin

        if op_margin is not None:
            if op_margin > 0.30:
                score += 20
                signals.append(f"Elite operating margin ({op_margin*100:.0f}%) — pricing power + efficiency")
            elif op_margin > 0.20:
                score += 15
            elif op_margin > 0.10:
                score += 10
            elif op_margin > 0:
                score += 5
            else:
                signals.append("Negative operating margin — not yet profitable")

        # ── Margin Trend ────────────────────────────────────────────
        margin_trend = self._margin_expansion()
        if margin_trend.get("expanding"):
            score += 15
            signals.append("Operating leverage — margins expanding over time")
        elif margin_trend.get("stable"):
            score += 8
        else:
            score -= 5
            signals.append("Margin compression — cost structure deteriorating")

        # ── FCF Conversion ──────────────────────────────────────────
        fcf_conv = self._fcf_conversion()
        metrics["fcf_conversion"] = fcf_conv

        if fcf_conv is not None:
            if fcf_conv > 0.90:
                score += 10
                signals.append(f"Strong FCF conversion ({fcf_conv*100:.0f}%) — earnings are real cash")
            elif fcf_conv > 0.70:
                score += 7
            elif fcf_conv > 0.50:
                score += 4

        score = min(max(score, 0), 100)

        return {
            "score": score,
            "label": "Return on Capital",
            "emoji": "",
            "signals": signals,
            "metrics": metrics,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   PILLAR 6: VALUATION — "FAIR PRICE" FRAMEWORK
    # ═══════════════════════════════════════════════════════════════════════════

    def _pillar_valuation(self) -> dict:
        """
        Buffett's valuation framework:
          - Is the stock trading below intrinsic value?
          - What's the margin of safety?
          - PEG ratio (growth-adjusted value)
          - Quality-adjusted P/E
        """
        score = 0
        signals = []
        metrics = {}

        info = self.info
        current_price = self.info.get("currentPrice") or self.info.get("regularMarketPrice") or 0

        # ── PEG Ratio (growth-adjusted valuation) ────────────────────
        pe = info.get("trailingPE") or info.get("forwardPE")
        eps_growth = info.get("earningsGrowth")
        metrics["pe_ratio"] = pe
        metrics["eps_growth"] = eps_growth

        peg = None
        if pe and eps_growth and eps_growth > 0:
            peg = pe / (eps_growth * 100)
            metrics["peg_ratio"] = round(peg, 2)

            if peg < 0.8:
                score += 25
                signals.append(f"PEG {peg:.2f} — undervalued relative to growth (Buffett would like this)")
            elif peg < 1.0:
                score += 20
                signals.append(f"PEG {peg:.2f} — fairly valued for growth")
            elif peg < 1.5:
                score += 14
                signals.append(f"PEG {peg:.2f} — slight premium for growth")
            elif peg < 2.5:
                score += 8
                signals.append(f"PEG {peg:.2f} — expensive relative to growth")
            else:
                score += 3
                signals.append(f"PEG {peg:.2f} — significantly overvalued")

        # ── P/E vs Quality (quality-adjusted valuation) ──────────────
        roe = info.get("returnOnEquity") or 0
        gm = info.get("grossMargins") or 0

        if pe and pe > 0:
            # High-quality companies deserve higher P/E
            quality_adjusted_pe = pe
            if roe > 0.25 and gm > 0.50:
                quality_adjusted_pe = pe * 0.70  # 30% quality discount
                signals.append("High quality justifies premium valuation")
            elif roe > 0.15 and gm > 0.35:
                quality_adjusted_pe = pe * 0.85

            metrics["quality_adjusted_pe"] = round(quality_adjusted_pe, 1)

            if quality_adjusted_pe < 15:
                score += 20
                signals.append("Quality-adjusted P/E is attractive")
            elif quality_adjusted_pe < 25:
                score += 14
            elif quality_adjusted_pe < 35:
                score += 8
            else:
                score += 3

        # ── FCF Yield (Buffett loves this) ───────────────────────
        fcf_yield = None
        market_cap = info.get("marketCap", 0)
        try:
            fcf = None
            if not self.cashflow.empty:
                for label in ["Free Cash Flow", "FreeCashFlow"]:
                    if label in self.cashflow.index:
                        vals = self.cashflow.loc[label].dropna()
                        if len(vals) > 0:
                            fcf = float(vals.iloc[0])
                        break
            if fcf and market_cap and market_cap > 0:
                fcf_yield = fcf / market_cap
                metrics["fcf_yield"] = round(fcf_yield * 100, 2)

                if fcf_yield > 0.06:
                    score += 18
                    signals.append(f"FCF yield {fcf_yield*100:.1f}% — excellent cash generation relative to price")
                elif fcf_yield > 0.04:
                    score += 12
                    signals.append(f"Healthy FCF yield ({fcf_yield*100:.1f}%)")
                elif fcf_yield > 0.02:
                    score += 7
                elif fcf_yield > 0:
                    score += 3
                else:
                    signals.append("Negative FCF — burning cash")
        except Exception:
            pass

        # ── Margin of Safety ─────────────────────────────────────────
        # Compare current price to analyst targets and historical valuation
        target_price = info.get("targetMeanPrice")
        if current_price and target_price and current_price > 0:
            upside = (target_price - current_price) / current_price
            metrics["analyst_upside"] = round(upside * 100, 1)

            if upside > 0.30:
                score += 12
                signals.append(f"Large margin of safety ({upside*100:.0f}% upside to analyst target)")
            elif upside > 0.15:
                score += 8
                signals.append(f"Decent margin of safety ({upside*100:.0f}% upside)")
            elif upside > 0:
                score += 4
            else:
                signals.append(f"Trading above analyst target ({upside*100:.0f}%)")

        score = min(max(score, 0), 100)

        return {
            "score": score,
            "label": "Valuation (Fair Price)",
            "emoji": "",
            "signals": signals,
            "metrics": metrics,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   THESIS GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_thesis(self, pillars: dict, conviction: float) -> dict:
        """Generate a structured investment thesis from all six pillars."""
        strengths = self._get_strengths(pillars)
        weaknesses = self._get_weaknesses(pillars)

        # Bull case: strongest pillars
        bull_points = []
        for pillar_name, data in sorted(pillars.items(),
                                         key=lambda x: x[1]["score"], reverse=True)[:3]:
            if data["signals"]:
                bull_points.append(data["signals"][0])

        # Bear case: weakest pillars
        bear_points = []
        for pillar_name, data in sorted(pillars.items(),
                                         key=lambda x: x[1]["score"])[:2]:
            if data["signals"]:
                bear_points.append(data["signals"][-1])

        # Overall thesis
        company = self.info.get("longName", self.ticker)
        sector = self.info.get("sector", "N/A")

        if conviction >= 75:
            verdict = f"{company} is a high-conviction opportunity with strong fundamentals across multiple dimensions."
        elif conviction >= 60:
            verdict = f"{company} shows solid quality characteristics with some areas to monitor."
        elif conviction >= 45:
            verdict = f"{company} presents a mixed picture — strengths exist but meaningful risks remain."
        elif conviction >= 30:
            verdict = f"{company} has significant weaknesses that outweigh its strengths at current valuation."
        else:
            verdict = f"{company} does not meet quality thresholds for a growth or value investment."

        return {
            "verdict": verdict,
            "bull_case": bull_points,
            "bear_case": bear_points,
            "sector": sector,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #   HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _calc_cagr(self, line_item: str) -> Optional[float]:
        """Calculate CAGR for a financial statement line item."""
        stmt = self.income if line_item in ("Total Revenue",) else self.cashflow
        if stmt.empty:
            return None
        try:
            if line_item not in stmt.index:
                # Try alternate names
                for alt in [line_item, line_item.replace(" ", "")]:
                    if alt in stmt.index:
                        line_item = alt
                        break
                else:
                    return None

            vals = stmt.loc[line_item].dropna()
            if len(vals) < 2:
                return None

            oldest = float(vals.iloc[-1])
            newest = float(vals.iloc[0])
            years = len(vals) - 1

            if oldest > 0 and newest > 0 and years > 0:
                return (newest / oldest) ** (1 / years) - 1
        except Exception:
            pass
        return None

    def _margin_expansion(self) -> dict:
        """Check if operating margins are expanding over time."""
        result = {"expanding": False, "stable": False, "compressing": False}

        if self.income.empty:
            result["stable"] = True
            return result

        try:
            rev_row = None
            op_row = None
            if "Total Revenue" in self.income.index:
                rev_row = self.income.loc["Total Revenue"].dropna()
            for label in ["Operating Income", "EBIT"]:
                if label in self.income.index:
                    op_row = self.income.loc[label].dropna()
                    break

            if rev_row is None or op_row is None or len(rev_row) < 2 or len(op_row) < 2:
                result["stable"] = True
                return result

            margins = []
            for i in range(min(len(rev_row), len(op_row))):
                rev_val = float(rev_row.iloc[i])
                op_val = float(op_row.iloc[i])
                if rev_val > 0:
                    margins.append(op_val / rev_val)

            if len(margins) >= 2:
                # newest is index 0
                diff = margins[0] - margins[-1]
                if diff > 0.02:
                    result["expanding"] = True
                elif diff > -0.02:
                    result["stable"] = True
                else:
                    result["compressing"] = True
                result["margin_change"] = diff
            else:
                result["stable"] = True

        except Exception:
            result["stable"] = True

        return result

    def _revenue_consistency(self) -> dict:
        """Count consecutive years of revenue growth."""
        result = {"consecutive_growth_years": 0}

        if self.income.empty or "Total Revenue" not in self.income.index:
            return result

        try:
            rev = self.income.loc["Total Revenue"].dropna()
            if len(rev) < 2:
                return result

            consecutive = 0
            for i in range(len(rev) - 1):
                # yfinance: newest is index 0
                if float(rev.iloc[i]) > float(rev.iloc[i + 1]):
                    consecutive += 1
                else:
                    break

            result["consecutive_growth_years"] = consecutive
        except Exception:
            pass

        return result

    def _rd_intensity(self) -> Optional[float]:
        """R&D spending as % of revenue."""
        if self.income.empty:
            return None
        try:
            rd = None
            for label in ["Research Development", "Research And Development", "ResearchAndDevelopment"]:
                if label in self.income.index:
                    vals = self.income.loc[label].dropna()
                    if len(vals) > 0:
                        rd = float(vals.iloc[0])
                    break

            rev = None
            if "Total Revenue" in self.income.index:
                vals = self.income.loc["Total Revenue"].dropna()
                if len(vals) > 0:
                    rev = float(vals.iloc[0])

            if rd and rev and rev > 0:
                return rd / rev
        except Exception:
            pass
        return None

    def _score_buybacks(self) -> dict:
        """Score share buyback activity."""
        result = {"score": 0, "signal": None, "buyback_yield": None}

        if self.cashflow.empty:
            return result

        try:
            buyback = None
            for label in ["Repurchase Of Capital Stock", "Common Stock Repurchased"]:
                if label in self.cashflow.index:
                    vals = self.cashflow.loc[label].dropna()
                    if len(vals) > 0:
                        buyback = float(vals.iloc[0])
                    break

            if buyback and buyback < 0:  # Buybacks are negative in cash flow
                mkt_cap = self.info.get("marketCap", 0)
                if mkt_cap > 0:
                    buyback_yield = abs(buyback) / mkt_cap
                    result["buyback_yield"] = round(buyback_yield * 100, 2)

                    if buyback_yield > 0.03:
                        result["score"] = 15
                        result["signal"] = f"Aggressive buybacks ({buyback_yield*100:.1f}% yield) — returning capital to shareholders"
                    elif buyback_yield > 0.01:
                        result["score"] = 10
                        result["signal"] = f"Active buyback program ({buyback_yield*100:.1f}% yield)"
                    else:
                        result["score"] = 5
        except Exception:
            pass

        return result

    def _calc_roic(self) -> Optional[float]:
        """Calculate Return on Invested Capital."""
        try:
            if self.balance.empty or self.income.empty:
                return None

            equity = None
            for label in ["Stockholders Equity", "Total Stockholders Equity"]:
                if label in self.balance.index:
                    vals = self.balance.loc[label].dropna()
                    if len(vals) > 0:
                        equity = float(vals.iloc[0])
                    break

            debt = None
            if "Total Debt" in self.balance.index:
                vals = self.balance.loc["Total Debt"].dropna()
                if len(vals) > 0:
                    debt = float(vals.iloc[0])

            net_income = None
            if "Net Income" in self.income.index:
                vals = self.income.loc["Net Income"].dropna()
                if len(vals) > 0:
                    net_income = float(vals.iloc[0])

            if equity and debt and net_income:
                invested = equity + debt
                if invested > 0:
                    return net_income / invested
        except Exception:
            pass
        return None

    def _fcf_conversion(self) -> Optional[float]:
        """FCF / Net Income — how much of earnings are real cash."""
        try:
            if self.cashflow.empty or self.income.empty:
                return None

            fcf = None
            for label in ["Free Cash Flow", "FreeCashFlow"]:
                if label in self.cashflow.index:
                    vals = self.cashflow.loc[label].dropna()
                    if len(vals) > 0:
                        fcf = float(vals.iloc[0])
                    break

            ni = None
            if "Net Income" in self.income.index:
                vals = self.income.loc["Net Income"].dropna()
                if len(vals) > 0:
                    ni = float(vals.iloc[0])

            if fcf and ni and ni > 0:
                return fcf / ni
        except Exception:
            pass
        return None

    def _match_tam(self, industry: str, description: str) -> dict:
        """Match a stock's industry to the TAM database."""
        result = {"matched": industry}

        # First: check keyword boosts (catches AI, cloud, fintech, etc.)
        for keyword, tam_key in TAM_KEYWORD_BOOSTS.items():
            if keyword in description:
                if tam_key in TAM_DATABASE:
                    tam = TAM_DATABASE[tam_key]
                    result.update(tam)
                    result["matched"] = tam_key
                    return result

        # Second: direct industry match
        if industry in TAM_DATABASE:
            result.update(TAM_DATABASE[industry])
            return result

        # Third: partial match
        for tam_industry, tam_data in TAM_DATABASE.items():
            if tam_industry.lower() in industry.lower() or industry.lower() in tam_industry.lower():
                result.update(tam_data)
                result["matched"] = tam_industry
                return result

        return result

    @staticmethod
    def _conviction_label(score: float) -> str:
        if score >= 80:
            return "STRONG BUY — High Conviction"
        elif score >= 65:
            return "BUY — Buffett Quality"
        elif score >= 50:
            return "HOLD — Decent Quality"
        elif score >= 35:
            return "WEAK — Below Average"
        else:
            return "AVOID — Poor Quality"

    @staticmethod
    def _get_strengths(pillars: dict) -> list:
        """Top 3 pillars by score."""
        sorted_p = sorted(pillars.items(), key=lambda x: x[1]["score"], reverse=True)
        return [
            {"pillar": data["label"], "score": data["score"], "emoji": data["emoji"]}
            for _, data in sorted_p[:3]
            if data["score"] >= 50
        ]

    @staticmethod
    def _get_weaknesses(pillars: dict) -> list:
        """Bottom 2 pillars by score."""
        sorted_p = sorted(pillars.items(), key=lambda x: x[1]["score"])
        return [
            {"pillar": data["label"], "score": data["score"], "emoji": data["emoji"]}
            for _, data in sorted_p[:2]
            if data["score"] < 50
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#   STANDALONE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yfinance as yf

    ticker = "NVDA"
    print(f"\n  Testing GrowthQualityScorer on {ticker}...")

    t = yf.Ticker(ticker)
    info = t.info or {}
    financials = {
        "income_stmt": t.income_stmt,
        "balance_sheet": t.balance_sheet,
        "cash_flow": t.cashflow,
    }
    stock_data = t.history(period="5y", auto_adjust=True)

    scorer = GrowthQualityScorer(ticker, info, financials, stock_data)
    result = scorer.full_analysis()

    print(f"\n  CONVICTION SCORE: {result['score']}/100 — {result['conviction_label']}")
    print(f"\n  Pillar Breakdown:")
    for name, data in result["pillars"].items():
        print(f"    {data['label']:34s} {data['score']:5.0f}/100")
        for sig in data["signals"][:2]:
            print(f"       → {sig}")

    print(f"\n  Thesis: {result['thesis']['verdict']}")
    print(f"\n  Bull Case:")
    for point in result["thesis"]["bull_case"]:
        print(f"  [+]  {point}")
    print(f"\n  Bear Case:")
    for point in result["thesis"]["bear_case"]:
        print(f"  [-]  {point}")