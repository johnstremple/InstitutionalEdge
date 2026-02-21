"""
fraud_detection.py — Accounting Fraud & Bankruptcy Risk

1. Beneish M-Score (1999) — detects earnings manipulation
   M > -1.78 = likely manipulator (flagged Enron, WorldCom before collapse)
   M < -2.22 = unlikely manipulator

2. Altman Z-Score (1968, updated) — bankruptcy probability
   Z > 2.99 = Safe zone
   Z 1.81-2.99 = Grey zone (watch)
   Z < 1.81 = Distress zone (high bankruptcy risk)

3. Earnings Quality Score — are profits real or accounting tricks?
   - Accruals ratio (cash vs reported earnings)
   - Cash conversion ratio
   - Revenue quality check

All math uses yfinance financial data — completely free.
"""

import yfinance as yf
import numpy as np
from typing import Optional


class FraudDetector:

    def __init__(self, ticker: str, info: dict, financials: dict):
        self.ticker     = ticker.upper()
        self.info       = info or {}
        self.financials = financials or {}

    def full_analysis(self) -> dict:
        beneish  = self._beneish_mscore()
        altman   = self._altman_zscore()
        eq_score = self._earnings_quality()
        risk     = self._overall_risk(beneish, altman, eq_score)

        return {
            "score":           risk["composite_score"],
            "beneish":         beneish,
            "altman":          altman,
            "earnings_quality": eq_score,
            "overall_risk":    risk,
            "signal":          risk["signal"],
            "summary":         risk["summary"],
        }

    # ── BENEISH M-SCORE ──────────────────────────────────────────

    def _beneish_mscore(self) -> dict:
        """
        Beneish M-Score: 8-variable model detecting earnings manipulation.
        Uses year-over-year changes in financial ratios.

        Variables:
        DSRI  = Days Sales Receivable Index
        GMI   = Gross Margin Index
        AQI   = Asset Quality Index
        SGI   = Sales Growth Index
        DEPI  = Depreciation Index
        SGAI  = SG&A Index
        LVGI  = Leverage Index
        TATA  = Total Accruals to Total Assets
        """
        try:
            t = yf.Ticker(self.ticker)

            # Get 2 years of financials for YoY comparison
            income  = t.income_stmt
            balance = t.balance_sheet
            cash    = t.cashflow

            if income is None or income.empty or balance is None or balance.empty:
                return self._beneish_from_info()

            # Helper to safely get value
            def get(df, *keys):
                for key in keys:
                    if key in df.index:
                        vals = df.loc[key].dropna()
                        if len(vals) >= 2:
                            return float(vals.iloc[0]), float(vals.iloc[1])
                        elif len(vals) == 1:
                            return float(vals.iloc[0]), float(vals.iloc[0])
                return None, None

            # Current and prior year values
            rev_t,  rev_t1  = get(income,  "Total Revenue", "Revenue")
            cogs_t, cogs_t1 = get(income,  "Cost Of Revenue", "Cost of Revenue", "Cost Of Goods Sold")
            rec_t,  rec_t1  = get(balance, "Net Receivables", "Accounts Receivable")
            assets_t, assets_t1 = get(balance, "Total Assets")
            ppe_t,  ppe_t1  = get(balance, "Net PPE", "Property Plant Equipment Net")
            dep_t,  dep_t1  = get(income,  "Reconciled Depreciation", "Depreciation")
            sga_t,  sga_t1  = get(income,  "Selling General Administrative", "SGA")
            ltd_t,  ltd_t1  = get(balance, "Long Term Debt")
            ca_t,   ca_t1   = get(balance, "Current Assets", "Total Current Assets")
            cl_t,   cl_t1   = get(balance, "Current Liabilities", "Total Current Liabilities")
            ni_t            = float(income.loc["Net Income"].iloc[0]) if "Net Income" in income.index else 0
            cfo_t           = None
            if cash is not None and "Operating Cash Flow" in cash.index:
                cfo_t = float(cash.loc["Operating Cash Flow"].iloc[0])

            # Guard against None/zero
            def safe_div(a, b, default=1.0):
                if a is None or b is None or b == 0:
                    return default
                return a / b

            # 1. DSRI — receivables growing faster than revenue = revenue manipulation
            dsri = safe_div(
                safe_div(rec_t, rev_t),
                safe_div(rec_t1, rev_t1)
            )

            # 2. GMI — gross margin deteriorating = pressure to manipulate
            gm_t  = safe_div((rev_t  - (cogs_t  or 0)), rev_t,  0.3)
            gm_t1 = safe_div((rev_t1 - (cogs_t1 or 0)), rev_t1, 0.3)
            gmi   = safe_div(gm_t1, gm_t)

            # 3. AQI — asset quality (non-current assets relative to total)
            nca_t  = (assets_t  or 0) - (ca_t  or 0) - (ppe_t  or 0)
            nca_t1 = (assets_t1 or 0) - (ca_t1 or 0) - (ppe_t1 or 0)
            aqi    = safe_div(
                safe_div(nca_t,  assets_t),
                safe_div(nca_t1, assets_t1)
            )

            # 4. SGI — sales growth index (high growth = incentive to manipulate)
            sgi = safe_div(rev_t, rev_t1)

            # 5. DEPI — depreciation slowing = asset inflation
            depr_rate_t  = safe_div(dep_t,  (dep_t  or 0) + (ppe_t  or 1))
            depr_rate_t1 = safe_div(dep_t1, (dep_t1 or 0) + (ppe_t1 or 1))
            depi = safe_div(depr_rate_t1, depr_rate_t)

            # 6. SGAI — SGA growing faster than sales = inefficiency / manipulation
            sgai = safe_div(
                safe_div(sga_t,  rev_t),
                safe_div(sga_t1, rev_t1)
            )

            # 7. LVGI — leverage increasing = financial pressure
            lev_t  = safe_div((ltd_t  or 0) + (cl_t  or 0), assets_t  or 1)
            lev_t1 = safe_div((ltd_t1 or 0) + (cl_t1 or 0), assets_t1 or 1)
            lvgi   = safe_div(lev_t, lev_t1)

            # 8. TATA — total accruals / total assets (cash vs reported earnings)
            if cfo_t is not None and assets_t:
                tata = (ni_t - cfo_t) / assets_t
            else:
                tata = 0.0

            # M-Score formula (Beneish 1999)
            m_score = (
                -4.84
                + 0.920  * dsri
                + 0.528  * gmi
                + 0.404  * aqi
                + 0.892  * sgi
                + 0.115  * depi
                - 0.172  * sgai
                + 4.679  * tata
                - 0.327  * lvgi
            )

            # Clip to reasonable range
            m_score = max(min(m_score, 5.0), -5.0)

            if m_score > -1.78:
                verdict    = "🔴 MANIPULATOR — High probability of earnings manipulation"
                risk_level = "HIGH"
            elif m_score > -2.22:
                verdict    = "🟠 GREY ZONE — Some manipulation signals present"
                risk_level = "MODERATE"
            else:
                verdict    = "🟢 CLEAN — Low probability of earnings manipulation"
                risk_level = "LOW"

            components = {
                "DSRI (Receivables)":   round(dsri,  3),
                "GMI (Gross Margin)":   round(gmi,   3),
                "AQI (Asset Quality)":  round(aqi,   3),
                "SGI (Sales Growth)":   round(sgi,   3),
                "DEPI (Depreciation)":  round(depi,  3),
                "SGAI (SGA)":           round(sgai,  3),
                "LVGI (Leverage)":      round(lvgi,  3),
                "TATA (Accruals)":      round(tata,  4),
            }

            # Red flags
            flags = []
            if dsri > 1.31:  flags.append(f"DSRI {dsri:.2f} — receivables growing faster than revenue")
            if gmi  > 1.14:  flags.append(f"GMI {gmi:.2f} — gross margins deteriorating")
            if aqi  > 1.25:  flags.append(f"AQI {aqi:.2f} — asset quality declining")
            if sgi  > 1.32:  flags.append(f"SGI {sgi:.2f} — high growth creates manipulation incentive")
            if tata > 0.031: flags.append(f"TATA {tata:.3f} — high accruals vs cash earnings")
            if lvgi > 1.11:  flags.append(f"LVGI {lvgi:.2f} — leverage increasing")

            return {
                "m_score":    round(m_score, 4),
                "verdict":    verdict,
                "risk_level": risk_level,
                "components": components,
                "red_flags":  flags,
                "threshold":  "-1.78 (manipulator) / -2.22 (clean)",
            }

        except Exception as e:
            return self._beneish_from_info()

    def _beneish_from_info(self) -> dict:
        """Simplified Beneish approximation from available info."""
        info = self.info
        flags = []

        # Use available signals as proxies
        rev_g  = info.get("revenueGrowth") or 0
        nm     = info.get("profitMargins") or 0
        de     = (info.get("debtToEquity") or 0) / 100
        roe    = info.get("returnOnEquity") or 0

        # Simple heuristic score
        risk = 0
        if rev_g > 0.40:  risk += 1; flags.append("Very high revenue growth — manipulation incentive")
        if nm < 0.02:     risk += 1; flags.append("Very thin margins — pressure to report better numbers")
        if de > 2.0:      risk += 1; flags.append("High leverage — financial pressure")
        if roe > 0.50 and nm < 0.10: risk += 1; flags.append("High ROE with low margins — possible accounting tricks")

        if risk >= 3:
            verdict = "🟠 GREY ZONE — Some risk indicators present (simplified)"
            risk_level = "MODERATE"
            m_score = -2.0
        else:
            verdict = "🟢 LOW RISK — No major red flags (simplified)"
            risk_level = "LOW"
            m_score = -2.5

        return {
            "m_score":    m_score,
            "verdict":    verdict,
            "risk_level": risk_level,
            "components": {},
            "red_flags":  flags,
            "threshold":  "-1.78 (manipulator) / -2.22 (clean)",
            "note":       "Simplified — insufficient financial statement data",
        }

    # ── ALTMAN Z-SCORE ───────────────────────────────────────────

    def _altman_zscore(self) -> dict:
        """
        Altman Z-Score (1968, revised).
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Cap / Total Liabilities
        X5 = Revenue / Total Assets
        """
        try:
            t       = yf.Ticker(self.ticker)
            balance = t.balance_sheet
            income  = t.income_stmt

            if balance is None or balance.empty:
                return self._altman_from_info()

            def val(df, *keys):
                for k in keys:
                    if k in df.index:
                        v = df.loc[k].dropna()
                        if len(v) > 0:
                            return float(v.iloc[0])
                return None

            total_assets = val(balance, "Total Assets")
            if not total_assets or total_assets == 0:
                return self._altman_from_info()

            ca    = val(balance, "Current Assets", "Total Current Assets") or 0
            cl    = val(balance, "Current Liabilities", "Total Current Liabilities") or 0
            re    = val(balance, "Retained Earnings") or 0
            ebit  = val(income,  "EBIT", "Operating Income") or 0
            rev   = val(income,  "Total Revenue", "Revenue") or 0
            tl    = val(balance, "Total Liabilities Net Minority Interest", "Total Liabilities") or (total_assets * 0.5)
            mktcap = self.info.get("marketCap") or (total_assets * 1.2)

            x1 = (ca - cl) / total_assets          # Working capital ratio
            x2 = re / total_assets                  # Retained earnings ratio
            x3 = ebit / total_assets                # Profitability ratio
            x4 = mktcap / tl if tl > 0 else 1.0    # Market leverage
            x5 = rev / total_assets                 # Asset turnover

            z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

            if z > 2.99:
                zone    = "🟢 SAFE ZONE"
                risk    = "LOW"
                prob    = "<5% bankruptcy probability"
            elif z > 1.81:
                zone    = "🟡 GREY ZONE"
                risk    = "MODERATE"
                prob    = "15-20% bankruptcy probability"
            else:
                zone    = "🔴 DISTRESS ZONE"
                risk    = "HIGH"
                prob    = ">30% bankruptcy probability (watch closely)"

            return {
                "z_score":    round(z, 3),
                "zone":       zone,
                "risk_level": risk,
                "probability": prob,
                "components": {
                    "X1 Working Capital/Assets": round(x1, 4),
                    "X2 Retained Earnings/Assets": round(x2, 4),
                    "X3 EBIT/Assets":             round(x3, 4),
                    "X4 MktCap/Total Liabilities": round(x4, 4),
                    "X5 Revenue/Assets":           round(x5, 4),
                },
                "thresholds": "Safe >2.99 | Grey 1.81-2.99 | Distress <1.81",
            }

        except Exception:
            return self._altman_from_info()

    def _altman_from_info(self) -> dict:
        info   = self.info
        mktcap = info.get("marketCap") or 1
        de     = (info.get("debtToEquity") or 50) / 100
        cr     = info.get("currentRatio") or 1.5
        roe    = info.get("returnOnEquity") or 0.1

        # Rough Z-score approximation
        x1 = min(max((cr - 1) / cr, -0.3), 0.4)
        x2 = min(max(roe * 2, -0.5), 1.0)
        x3 = max(roe * 0.5, 0)
        x4 = 1 / max(de, 0.1)
        x5 = info.get("revenueGrowth") or 0.1

        z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

        if z > 2.99:   zone="🟢 SAFE ZONE";     risk="LOW";      prob="<5% bankruptcy"
        elif z > 1.81: zone="🟡 GREY ZONE";     risk="MODERATE"; prob="15-20% bankruptcy"
        else:          zone="🔴 DISTRESS ZONE";  risk="HIGH";     prob=">30% bankruptcy"

        return {
            "z_score":    round(z, 3),
            "zone":       zone,
            "risk_level": risk,
            "probability": prob,
            "components": {},
            "thresholds": "Safe >2.99 | Grey 1.81-2.99 | Distress <1.81",
            "note":       "Approximated from available metrics",
        }

    # ── EARNINGS QUALITY ─────────────────────────────────────────

    def _earnings_quality(self) -> dict:
        """Are reported earnings backed by real cash flow?"""
        info   = self.info
        score  = 70  # start neutral
        flags  = []
        green  = []

        try:
            t     = yf.Ticker(self.ticker)
            cf    = t.cashflow
            inc   = t.income_stmt

            cfo   = None
            ni    = None
            rev   = None
            ta    = None

            if cf is not None and "Operating Cash Flow" in cf.index:
                cfo = float(cf.loc["Operating Cash Flow"].iloc[0])
            if inc is not None and "Net Income" in inc.index:
                ni  = float(inc.loc["Net Income"].iloc[0])
            if inc is not None:
                for k in ["Total Revenue","Revenue"]:
                    if k in inc.index:
                        rev = float(inc.loc[k].iloc[0]); break

            b = t.balance_sheet
            if b is not None and "Total Assets" in b.index:
                ta = float(b.loc["Total Assets"].iloc[0])

        except Exception:
            cfo = ni = rev = ta = None

        # 1. Cash vs Reported Earnings
        if cfo is not None and ni is not None and ni != 0:
            ratio = cfo / ni
            if ratio > 1.2:
                score += 15
                green.append(f"Cash flow {ratio:.1f}x net income — earnings are high quality")
            elif ratio > 0.8:
                score += 5
                green.append(f"Cash flow {ratio:.1f}x net income — reasonable quality")
            elif ratio < 0.5:
                score -= 20
                flags.append(f"Cash flow only {ratio:.1f}x net income — earnings may be inflated")
            elif ratio < 0:
                score -= 30
                flags.append("Negative cash flow with positive earnings — serious red flag")

        # 2. Accruals ratio
        if cfo is not None and ni is not None and ta and ta > 0:
            accruals = (ni - cfo) / ta
            if accruals > 0.10:
                score -= 15
                flags.append(f"High accruals ratio {accruals:.2f} — earnings rely on non-cash items")
            elif accruals < 0.03:
                score += 10
                green.append(f"Low accruals {accruals:.2f} — clean, cash-backed earnings")

        # 3. From info fields
        fcf   = info.get("freeCashflow") or 0
        mktcap = info.get("marketCap") or 1
        nm    = info.get("profitMargins") or 0
        gm    = info.get("grossMargins") or 0

        # FCF yield as quality check
        fcf_yield = fcf / mktcap if mktcap > 0 else 0
        if fcf_yield > 0.05:
            score += 10
            green.append(f"FCF yield {fcf_yield*100:.1f}% — strong cash generation")
        elif fcf_yield < 0:
            score -= 10
            flags.append("Negative free cash flow — not self-funding operations")

        # Margin sustainability
        if gm > 0 and nm > 0:
            op_ratio = nm / gm
            if op_ratio < 0.15:
                flags.append("Very high SGA/operating costs relative to gross profit")

        score = round(min(max(score, 0), 100), 1)

        if score >= 70:   verdict = "🟢 HIGH QUALITY — Earnings are real and cash-backed"
        elif score >= 50: verdict = "🟡 ACCEPTABLE — Some concerns but not alarming"
        else:             verdict = "🔴 LOW QUALITY — Earnings may not reflect true cash generation"

        return {
            "score":    score,
            "verdict":  verdict,
            "green_flags": green,
            "red_flags":   flags,
            "cfo_to_ni":   round(cfo/ni, 3) if (cfo and ni and ni != 0) else None,
            "fcf_yield":   round(fcf_yield * 100, 2),
        }

    # ── OVERALL RISK ─────────────────────────────────────────────

    def _overall_risk(self, beneish: dict, altman: dict, eq: dict) -> dict:
        b_risk = {"LOW": 0, "MODERATE": 1, "HIGH": 2}.get(beneish.get("risk_level","LOW"), 0)
        a_risk = {"LOW": 0, "MODERATE": 1, "HIGH": 2}.get(altman.get("risk_level","LOW"), 0)
        eq_score = eq.get("score", 70)

        total_risk = b_risk + a_risk
        eq_risk    = 0 if eq_score >= 70 else (1 if eq_score >= 50 else 2)
        total_risk += eq_risk

        if total_risk == 0:
            composite = 85
            signal    = "🟢 CLEAN — No significant accounting or financial risk detected"
        elif total_risk == 1:
            composite = 65
            signal    = "🟡 WATCH — Minor risk signals, worth monitoring"
        elif total_risk == 2:
            composite = 45
            signal    = "🟠 CAUTION — Multiple risk factors present, do deeper diligence"
        else:
            composite = 25
            signal    = "🔴 HIGH RISK — Serious accounting or financial distress signals"

        flags = (beneish.get("red_flags",[]) +
                 eq.get("red_flags",[]))

        summary = f"M-Score {beneish.get('m_score','N/A')} ({beneish.get('risk_level','')}) | " \
                  f"Z-Score {altman.get('z_score','N/A')} ({altman.get('zone','').split()[0] if altman.get('zone') else ''}) | " \
                  f"Earnings Quality {eq_score}/100"

        return {
            "composite_score": composite,
            "signal":          signal,
            "summary":         summary,
            "all_flags":       flags[:6],
        }
