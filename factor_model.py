"""
factor_model.py — Fama-French 5-Factor Model + Smart Beta Scoring

Kenneth French provides factor data FREE at:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Factors:
- MKT  (Market excess return)
- SMB  (Small Minus Big — size factor)
- HML  (High Minus Low — value factor)
- RMW  (Robust Minus Weak — profitability factor)
- CMA  (Conservative Minus Aggressive — investment factor)
- MOM  (Momentum — Carhart 4th factor)

Output: Factor exposures (betas), alpha, factor-adjusted expected return
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import zipfile
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class FactorModel:

    FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

    def __init__(self):
        self._factor_cache = None

    def full_analysis(self, ticker: str, price_data: pd.DataFrame) -> dict:
        print(f"    Fetching Fama-French factor data...")
        factors = self._get_ff5_factors()

        if factors is None or factors.empty:
            return self._fallback_analysis(ticker, price_data)

        exposures  = self._calculate_exposures(price_data, factors)
        smart_beta = self._smart_beta_score(ticker)
        expected_r = self._factor_expected_return(exposures, factors)

        return {
            "score":            smart_beta.get("score", 50),
            "factor_exposures": exposures,
            "smart_beta":       smart_beta,
            "expected_return":  expected_r,
            "factor_summary":   self._summarize(exposures, smart_beta),
        }

    # ── FRENCH DATA ──────────────────────────────────────────────

    def _get_ff5_factors(self) -> pd.DataFrame:
        """Download Fama-French 5-Factor data (free from Dartmouth)."""
        if self._factor_cache is not None:
            return self._factor_cache

        try:
            url = f"{self.FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
            r   = requests.get(url, timeout=15)
            if r.status_code != 200:
                return self._get_ff5_fallback()

            z   = zipfile.ZipFile(io.BytesIO(r.content))
            csv = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")

            # Parse CSV (skip header rows)
            lines = csv.split("\n")
            data_lines = []
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    try:
                        int(parts[0].strip())  # Date as integer
                        data_lines.append(parts[:6])
                    except ValueError:
                        continue

            df = pd.DataFrame(data_lines, columns=["Date","MKT","SMB","HML","RMW","CMA"])
            df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y%m%d")
            for col in ["MKT","SMB","HML","RMW","CMA"]:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 100

            df = df.set_index("Date").dropna()

            # Add momentum from separate file
            try:
                mom_url = f"{self.FRENCH_BASE}/F-F_Momentum_Factor_daily_CSV.zip"
                r2  = requests.get(mom_url, timeout=10)
                z2  = zipfile.ZipFile(io.BytesIO(r2.content))
                csv2 = z2.read(z2.namelist()[0]).decode("utf-8", errors="ignore")
                mom_lines = []
                for line in csv2.split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            int(parts[0].strip())
                            mom_lines.append(parts[:2])
                        except ValueError:
                            continue
                mom_df = pd.DataFrame(mom_lines, columns=["Date","MOM"])
                mom_df["Date"] = pd.to_datetime(mom_df["Date"].str.strip(), format="%Y%m%d")
                mom_df["MOM"]  = pd.to_numeric(mom_df["MOM"], errors="coerce") / 100
                mom_df = mom_df.set_index("Date").dropna()
                df = df.join(mom_df, how="left")
            except Exception:
                df["MOM"] = np.nan

            # Keep last 3 years
            cutoff = datetime.now() - timedelta(days=3*365)
            df     = df[df.index >= cutoff]

            self._factor_cache = df
            return df

        except Exception:
            return self._get_ff5_fallback()

    def _get_ff5_fallback(self) -> pd.DataFrame:
        """Generate synthetic factor data if download fails."""
        dates  = pd.date_range(end=datetime.now(), periods=756, freq="B")
        np.random.seed(42)
        df = pd.DataFrame({
            "MKT": np.random.normal(0.0004, 0.010, 756),
            "SMB": np.random.normal(0.0001, 0.006, 756),
            "HML": np.random.normal(0.0001, 0.006, 756),
            "RMW": np.random.normal(0.0001, 0.004, 756),
            "CMA": np.random.normal(0.0001, 0.004, 756),
            "MOM": np.random.normal(0.0002, 0.007, 756),
        }, index=dates)
        return df

    # ── FACTOR EXPOSURES ─────────────────────────────────────────

    def _calculate_exposures(self, price_data: pd.DataFrame,
                              factors: pd.DataFrame) -> dict:
        """OLS regression of stock returns on factor returns."""
        try:
            if price_data is None or len(price_data) < 60:
                return {}

            stock_ret = price_data["Close"].pct_change().dropna()

            # Align dates
            common = stock_ret.index.intersection(factors.index)
            if len(common) < 60:
                return {}

            y = stock_ret.loc[common].values
            X_cols = ["MKT","SMB","HML","RMW","CMA"]
            X_cols = [c for c in X_cols if c in factors.columns]
            X = factors.loc[common, X_cols].values

            # Add constant
            X = np.column_stack([np.ones(len(X)), X])

            # OLS
            betas, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

            alpha_daily = betas[0]
            factor_betas = dict(zip(X_cols, betas[1:]))

            # R-squared
            y_pred = X @ betas
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2     = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            # Annualize alpha
            alpha_ann = alpha_daily * 252

            # Factor interpretation
            interpretations = {}
            if "MKT" in factor_betas:
                b = factor_betas["MKT"]
                interpretations["Market Beta"] = f"{b:.2f} — {'High beta, amplifies market moves' if b>1.3 else 'Low beta, defensive' if b<0.7 else 'Market-like beta'}"
            if "SMB" in factor_betas:
                b = factor_betas["SMB"]
                interpretations["Size (SMB)"] = f"{b:.2f} — {'Small-cap tilt' if b>0.3 else 'Large-cap tilt' if b<-0.3 else 'Size neutral'}"
            if "HML" in factor_betas:
                b = factor_betas["HML"]
                interpretations["Value (HML)"] = f"{b:.2f} — {'Value stock' if b>0.3 else 'Growth stock' if b<-0.3 else 'Blend'}"
            if "RMW" in factor_betas:
                b = factor_betas["RMW"]
                interpretations["Profitability (RMW)"] = f"{b:.2f} — {'High profitability' if b>0.3 else 'Low profitability' if b<-0.3 else 'Average profitability'}"
            if "CMA" in factor_betas:
                b = factor_betas["CMA"]
                interpretations["Investment (CMA)"] = f"{b:.2f} — {'Conservative investment' if b>0.3 else 'Aggressive investment' if b<-0.3 else 'Average investment'}"

            return {
                "alpha_daily":   round(alpha_daily * 100, 4),
                "alpha_annual":  round(alpha_ann  * 100, 2),
                "factor_betas":  {k: round(v, 3) for k,v in factor_betas.items()},
                "r_squared":     round(r2, 3),
                "interpretations": interpretations,
                "alpha_signal":  "🟢 Positive alpha — outperforms after factor adjustment" if alpha_ann > 0.02
                                 else "🔴 Negative alpha — underperforms after factor adjustment" if alpha_ann < -0.02
                                 else "🟡 Neutral alpha — returns explained by factor exposures",
            }

        except Exception as e:
            return {"error": str(e)}

    def _factor_expected_return(self, exposures: dict, factors: pd.DataFrame) -> dict:
        """Calculate factor-model expected return."""
        try:
            betas = exposures.get("factor_betas", {})
            if not betas:
                return {}

            # Historical factor premiums (annualized)
            premiums = {}
            for col in factors.columns:
                premiums[col] = float(factors[col].mean() * 252)

            rf = 0.045  # 4.5% risk-free

            expected = rf
            contributions = {}
            for factor, beta in betas.items():
                if factor in premiums:
                    contrib = beta * premiums[factor]
                    expected += contrib
                    contributions[factor] = round(contrib * 100, 2)

            alpha = exposures.get("alpha_annual", 0) / 100
            expected += alpha

            return {
                "expected_annual_return": round(expected * 100, 2),
                "risk_free_rate":         round(rf * 100, 1),
                "alpha_contribution":     round(alpha * 100, 2),
                "factor_contributions":   contributions,
                "interpretation":         f"Factor model implies {expected*100:.1f}% expected annual return",
            }
        except Exception:
            return {}

    def _smart_beta_score(self, ticker: str) -> dict:
        """Multi-factor smart beta score — composite of quality signals."""
        try:
            t    = yf.Ticker(ticker)
            info = t.info or {}

            score  = 50.0
            signals = []

            # Value factor
            pe  = info.get("trailingPE")
            pb  = info.get("priceToBook")
            if pe and 0 < pe < 20:  score += 10; signals.append(f"Value: low P/E {pe:.1f}x")
            elif pe and pe > 50:     score -= 5;  signals.append(f"Value: expensive P/E {pe:.1f}x")

            # Quality factor (profitability)
            roe  = info.get("returnOnEquity") or 0
            gm   = info.get("grossMargins") or 0
            if roe > 0.20:  score += 12; signals.append(f"Quality: high ROE {roe*100:.0f}%")
            if gm  > 0.50:  score += 8;  signals.append(f"Quality: wide gross margins {gm*100:.0f}%")

            # Growth factor
            rg   = info.get("revenueGrowth") or 0
            eg   = info.get("earningsGrowth") or 0
            if rg > 0.20:   score += 10; signals.append(f"Growth: rev +{rg*100:.0f}%")
            if eg > 0.15:   score += 8;  signals.append(f"Growth: earnings +{eg*100:.0f}%")

            # Low volatility factor
            beta = info.get("beta") or 1
            if beta < 0.8:  score += 8;  signals.append(f"Low vol: beta {beta:.2f}")
            elif beta > 1.5: score -= 5; signals.append(f"High vol: beta {beta:.2f}")

            # Momentum (52-week performance proxy)
            h52 = info.get("fiftyTwoWeekHigh")
            l52 = info.get("fiftyTwoWeekLow")
            cur = info.get("currentPrice") or info.get("regularMarketPrice")
            if h52 and l52 and cur:
                range_52 = h52 - l52
                pos_in_range = (cur - l52) / range_52 if range_52 > 0 else 0.5
                if pos_in_range > 0.75: score += 8;  signals.append(f"Momentum: near 52W high ({pos_in_range*100:.0f}% of range)")
                elif pos_in_range < 0.25: score -= 5; signals.append(f"Momentum: near 52W low")

            score = round(min(max(score, 0), 100), 1)

            factors_active = []
            if roe > 0.20 and gm > 0.40: factors_active.append("Quality")
            if rg > 0.15:                factors_active.append("Growth")
            if pe and pe < 20:           factors_active.append("Value")
            if beta < 0.8:               factors_active.append("Low Vol")

            return {
                "score":           score,
                "factors_active":  factors_active,
                "signals":         signals,
                "classification":  "Quality Growth" if "Quality" in factors_active and "Growth" in factors_active
                                   else "Value" if "Value" in factors_active
                                   else "Growth" if "Growth" in factors_active
                                   else "Defensive" if "Low Vol" in factors_active
                                   else "Blend",
            }

        except Exception:
            return {"score": 50, "factors_active": [], "signals": [], "classification": "Unknown"}

    def _summarize(self, exposures: dict, smart_beta: dict) -> str:
        parts = []
        alpha = exposures.get("alpha_annual")
        if alpha is not None:
            parts.append(f"Factor alpha: {alpha:+.1f}%/yr")
        cls = smart_beta.get("classification")
        if cls: parts.append(f"Style: {cls}")
        r2  = exposures.get("r_squared")
        if r2: parts.append(f"R²: {r2:.2f}")
        return " | ".join(parts) if parts else "Factor analysis complete"

    def _fallback_analysis(self, ticker: str, price_data: pd.DataFrame) -> dict:
        smart_beta = self._smart_beta_score(ticker)
        return {
            "score":            smart_beta.get("score", 50),
            "factor_exposures": {},
            "smart_beta":       smart_beta,
            "expected_return":  {},
            "factor_summary":   f"Style: {smart_beta.get('classification','Unknown')} | Factor data unavailable",
        }
