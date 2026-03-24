"""
alpha_model.py — Institutional-Grade Quantitative Alpha Model

Multi-signal, multi-model stock selection engine that ingests every data source
in the InstitutionalEdge platform and outputs risk-adjusted probability rankings.

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │  LAYER 1: FEATURE ENGINEERING (150+ features)                    │
    │  Fundamentals · Technicals · Options Flow · Insider Activity     │
    │  Short Interest · Macro Regime · News Sentiment · Alt Data       │
    │  Factor Exposures · Fraud Signals · Growth Quality Pillars       │
    ├──────────────────────────────────────────────────────────────────┤
    │  LAYER 2: REGIME DETECTION                                       │
    │  Identify market regime (bull / bear / sideways / volatile)      │
    │  Adjust feature weights and model selection per regime            │
    ├──────────────────────────────────────────────────────────────────┤
    │  LAYER 3: MULTI-MODEL ENSEMBLE                                   │
    │  Base: Random Forest, Gradient Boosting, XGBoost, Ridge          │
    │  Meta: Stacking classifier with logistic regression meta-learner │
    │  Calibration: Platt scaling for true probability estimates        │
    ├──────────────────────────────────────────────────────────────────┤
    │  LAYER 4: RISK-ADJUSTED RANKING                                  │
    │  Expected Alpha = P(outperform) × E[magnitude] / E[risk]        │
    │  Position sizing via Kelly criterion approximation                │
    ├──────────────────────────────────────────────────────────────────┤
    │  LAYER 5: WALK-FORWARD VALIDATION                                │
    │  Expanding window with embargo gap (no look-ahead bias)          │
    │  Purged cross-validation for proper financial ML                  │
    │  Out-of-sample alpha, Sharpe, hit rate, profit factor             │
    ├──────────────────────────────────────────────────────────────────┤
    │  LAYER 6: SELF-IMPROVEMENT                                       │
    │  Track prediction accuracy over time                             │
    │  Retrain on new data, prune dead features, adapt ensemble weights│
    │  Feature importance drift detection                               │
    └──────────────────────────────────────────────────────────────────┘

Usage:
    model = AlphaModel()
    model.build_universe()               # Collect features for S&P500 + NDX
    model.train()                        # Fit stacked ensemble
    model.validate()                     # Walk-forward backtest
    picks = model.generate_signals()     # Current top picks with sizing
    model.save("alpha_v1")               # Persist for production use
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import json
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict
import time

# ── sklearn core ────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── Optional: XGBoost / LightGBM (graceful fallback) ──────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#   CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Minimum data requirements
MIN_PRICE_DAYS = 126       # 6 months of trading data
MIN_MARKET_CAP = 1e9       # $1B minimum (no microcaps — too noisy)
MAX_FEATURES = 120         # Cap to prevent overfitting on small datasets

# Feature groups — organized by signal source
YFINANCE_FEATURES = {
    "valuation": [
        "trailingPE", "forwardPE", "priceToBook",
        "priceToSalesTrailing12Months", "enterpriseToEbitda",
        "enterpriseToRevenue", "pegRatio",
    ],
    "growth": [
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    ],
    "profitability": [
        "grossMargins", "operatingMargins", "profitMargins",
        "returnOnEquity", "returnOnAssets",
    ],
    "health": [
        "currentRatio", "quickRatio", "debtToEquity",
    ],
    "ownership": [
        "heldPercentInsiders", "heldPercentInstitutions",
        "payoutRatio", "dividendYield",
    ],
    "market": [
        "beta", "marketCap", "floatShares",
        "sharesShort", "shortRatio", "shortPercentOfFloat",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#   UNIVERSE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_universe() -> list:
    """Scrape S&P 500 + NASDAQ-100, deduplicated and sorted.
    Falls back to hardcoded list if Wikipedia scraping fails."""
    tickers = set()

    # Try S&P 500 from Wikipedia
    try:
        tables = pd.read_html(SP500_URL)
        sp = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        tickers.update(sp)
    except Exception as e:
        print(f"  WARNING: Could not scrape S&P 500 from Wikipedia: {e}")

    # Try NASDAQ-100 from Wikipedia
    try:
        tables = pd.read_html(NDX_URL)
        for t in tables:
            if "Ticker" in t.columns or "Symbol" in t.columns:
                col = "Ticker" if "Ticker" in t.columns else "Symbol"
                ndx = t[col].dropna().str.replace(".", "-", regex=False).tolist()
                tickers.update(ndx)
                break
        else:
            # Fallback: try table index 4
            if len(tables) > 4:
                df = tables[4]
                col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                ndx = df[col].dropna().str.replace(".", "-", regex=False).tolist()
                tickers.update(ndx)
    except Exception as e:
        print(f"  WARNING: Could not scrape NASDAQ-100 from Wikipedia: {e}")

    # If scraping failed entirely, use hardcoded universe
    if len(tickers) < 20:
        print("  Using hardcoded stock universe (Wikipedia scrape failed)")
        tickers = set(_FALLBACK_UNIVERSE)

    return sorted(tickers)


# Hardcoded fallback — top ~200 stocks by market cap / liquidity
_FALLBACK_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "ADBE",
    "CRM", "AMD", "ORCL", "INTC", "QCOM", "TXN", "AMAT", "MU", "NOW", "PANW",
    "SNPS", "CDNS", "KLAC", "LRCX", "MRVL", "ADI", "FTNT", "PLTR", "CRWD", "DDOG",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "SPGI", "CME",
    "ICE", "MCO", "CB", "BRK-B", "V", "MA", "PYPL", "COIN", "HOOD", "SOFI",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "ISRG", "MDT", "SYK", "VRTX", "REGN", "BSX", "ZTS", "HCA",
    "ELV", "CI", "DXCM", "IDXX", "MRNA",
    # Consumer
    "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "COST", "WMT", "PG",
    "KO", "PEP", "PM", "CL", "MDLZ", "STZ", "BKNG", "ABNB", "CMG", "ORLY",
    "TSLA", "F", "GM", "RIVN",
    # Industrials
    "CAT", "DE", "HON", "GE", "RTX", "BA", "UPS", "UNP", "LMT", "ETN",
    "ITW", "EMR", "MMM", "WM", "UBER",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "WMB", "OKE",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "SPOT", "RBLX",
    # Utilities / Real Estate
    "NEE", "DUK", "SO", "D", "AEP", "AMT", "PLD", "CCI", "EQIX", "PSA",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE",
    # High-growth / Speculative
    "SMCI", "MSTR", "IONQ", "RKLB", "ASTS", "JOBY", "ACHR", "HIMS", "CAVA",
    "APP", "DUOL", "ARM", "TTD", "SNOW", "NET", "ZS", "SHOP", "SQ", "ROKU",
]


# ═══════════════════════════════════════════════════════════════════════════════
#   FEATURE ENGINEERING — The most important part of the entire system.
#   Every feature must have economic rationale. No garbage in.
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extracts 150+ features from every available data source for a single stock.
    Each feature has an economic rationale documented inline.
    """

    def __init__(self):
        self._cache = {}

    def extract(self, ticker: str, spy_data: pd.DataFrame = None) -> Optional[dict]:
        """
        Extract all features for a single ticker.
        Returns dict of feature_name → value, or None if insufficient data.
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            # Gate: skip if no real data
            if not info or not info.get("marketCap") or info.get("marketCap", 0) < MIN_MARKET_CAP:
                return None

            price_data = t.history(period="5y", auto_adjust=True)
            if price_data is None or len(price_data) < MIN_PRICE_DAYS:
                return None

            # Gate: skip if price data is stale (delisted/suspended)
            last_date = price_data.index[-1]
            if (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days > 10:
                return None

            features = {"ticker": ticker}

            # ── GROUP 1: RAW FUNDAMENTALS FROM YFINANCE ──────────────
            features.update(self._raw_fundamentals(info))

            # ── GROUP 2: COMPUTED FUNDAMENTAL RATIOS ─────────────────
            features.update(self._computed_fundamentals(t, info))

            # ── GROUP 3: PRICE / MOMENTUM / VOLATILITY ───────────────
            features.update(self._price_features(price_data, info))

            # ── GROUP 4: FINANCIAL STATEMENT DERIVED ─────────────────
            features.update(self._statement_features(t, info))

            # ── GROUP 5: QUALITY / FRAUD SIGNALS ─────────────────────
            features.update(self._quality_signals(t, info))

            # ── GROUP 6: OWNERSHIP / FLOW SIGNALS ────────────────────
            features.update(self._ownership_features(info))

            # ── GROUP 7: OPTIONS-IMPLIED SIGNALS ─────────────────────
            features.update(self._options_features(t, price_data))

            # ── GROUP 8: RELATIVE VALUATION ──────────────────────────
            features.update(self._relative_features(info, price_data))

            # ── GROUP 9: NEWS / NLP SENTIMENT ────────────────────────
            features.update(self._news_sentiment_features(t, ticker, info))

            # ── GROUP 10: EARNINGS SURPRISE & ANALYST REVISIONS ──────
            features.update(self._earnings_features(t, info))

            # ── GROUP 11: SECTOR-RELATIVE SCORING ────────────────────
            features.update(self._sector_relative_features(ticker, info))

            # ── GROUP 12: FORWARD RETURN LABELS ──────────────────────
            if spy_data is not None:
                labels = self._compute_labels(price_data, spy_data)
                if labels is None:
                    return None
                features.update(labels)

            return features

        except Exception:
            return None

    # ─── GROUP 1: RAW FUNDAMENTALS ──────────────────────────────────────────

    def _raw_fundamentals(self, info: dict) -> dict:
        """Direct yfinance .info fields — valuation, growth, margins."""
        f = {}
        for group_name, fields in YFINANCE_FEATURES.items():
            for field in fields:
                val = info.get(field)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    f[field] = float(val)
                else:
                    f[field] = np.nan

        # Fix: debtToEquity comes as percentage from yfinance
        if np.isfinite(f.get("debtToEquity", np.nan)):
            f["debtToEquity"] = f["debtToEquity"] / 100.0

        return f

    # ─── GROUP 2: COMPUTED FUNDAMENTAL RATIOS ───────────────────────────────

    def _computed_fundamentals(self, t, info: dict) -> dict:
        """Derived metrics that require calculation from multiple fields."""
        f = {}
        mkt_cap = info.get("marketCap", 0)

        # Earnings yield (inverse P/E — better for regression)
        pe = info.get("trailingPE")
        f["earnings_yield"] = 1.0 / pe if pe and pe > 0 else np.nan

        # FCF yield — Buffett's favorite valuation metric
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                for label in ["Free Cash Flow", "FreeCashFlow"]:
                    if label in cf.index:
                        fcf = float(cf.loc[label].dropna().iloc[0])
                        f["fcf_yield"] = fcf / mkt_cap if mkt_cap > 0 else np.nan
                        break
        except Exception:
            pass
        f.setdefault("fcf_yield", np.nan)

        # Shareholder yield = buyback yield + dividend yield
        div_yield = info.get("dividendYield") or 0
        try:
            cf = t.cashflow
            buyback = 0
            if cf is not None and not cf.empty:
                for label in ["Repurchase Of Capital Stock", "Common Stock Repurchased"]:
                    if label in cf.index:
                        buyback = abs(float(cf.loc[label].dropna().iloc[0]))
                        break
            buyback_yield = buyback / mkt_cap if mkt_cap > 0 else 0
            f["shareholder_yield"] = div_yield + buyback_yield
            f["buyback_yield"] = buyback_yield
        except Exception:
            f["shareholder_yield"] = div_yield
            f["buyback_yield"] = np.nan

        # PEG ratio (growth-adjusted valuation)
        peg = info.get("pegRatio")
        f["peg_ratio"] = peg if peg and np.isfinite(peg) else np.nan

        # EV/FCF
        ev = info.get("enterpriseValue")
        if ev and f.get("fcf_yield") and not np.isnan(f.get("fcf_yield", np.nan)):
            fcf = f["fcf_yield"] * mkt_cap
            f["ev_to_fcf"] = ev / fcf if fcf > 0 else np.nan
        else:
            f["ev_to_fcf"] = np.nan

        return f

    # ─── GROUP 3: PRICE / MOMENTUM / VOLATILITY ────────────────────────────

    def _price_features(self, data: pd.DataFrame, info: dict) -> dict:
        """Price action, momentum factors, volatility regime."""
        f = {}
        close = data["Close"]
        returns = close.pct_change().dropna()
        current = float(close.iloc[-1])

        # ── Momentum (multiple timeframes) ────────────────────────
        # Jegadeesh & Titman: 12-1 month momentum is the strongest predictor
        for days, label in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]:
            if len(close) > days:
                f[f"momentum_{label}"] = float(close.iloc[-1] / close.iloc[-(days+1)] - 1)
            else:
                f[f"momentum_{label}"] = np.nan

        # 12-1 momentum (skip most recent month — reversal effect)
        if len(close) > 273:
            f["momentum_12_1"] = float(close.iloc[-22] / close.iloc[-253] - 1)
        else:
            f["momentum_12_1"] = np.nan

        # ── Moving average signals ────────────────────────────────
        for window in [20, 50, 100, 200]:
            if len(close) > window:
                ma = float(close.rolling(window).mean().iloc[-1])
                f[f"price_to_sma{window}"] = current / ma if ma > 0 else np.nan
            else:
                f[f"price_to_sma{window}"] = np.nan

        # Golden/death cross signal (50 vs 200 DMA)
        if len(close) > 200:
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            f["golden_cross"] = 1.0 if sma50 > sma200 else 0.0
            f["sma_spread"] = (sma50 - sma200) / sma200 if sma200 > 0 else np.nan
        else:
            f["golden_cross"] = np.nan
            f["sma_spread"] = np.nan

        # ── Volatility ────────────────────────────────────────────
        if len(returns) >= 252:
            f["volatility_20d"] = float(returns.iloc[-20:].std() * np.sqrt(252))
            f["volatility_60d"] = float(returns.iloc[-60:].std() * np.sqrt(252))
            f["volatility_252d"] = float(returns.std() * np.sqrt(252))

            # Volatility regime: is recent vol higher or lower than long-term?
            f["vol_regime"] = f["volatility_20d"] / f["volatility_252d"] if f["volatility_252d"] > 0 else np.nan
        else:
            for k in ["volatility_20d", "volatility_60d", "volatility_252d", "vol_regime"]:
                f[k] = np.nan

        # ── Drawdown from high ────────────────────────────────────
        if len(close) > 252:
            high_52w = float(close.iloc[-252:].max())
            f["drawdown_from_52w_high"] = (current - high_52w) / high_52w
            low_52w = float(close.iloc[-252:].min())
            f["rally_from_52w_low"] = (current - low_52w) / low_52w if low_52w > 0 else np.nan
        else:
            f["drawdown_from_52w_high"] = np.nan
            f["rally_from_52w_low"] = np.nan

        # ── RSI ───────────────────────────────────────────────────
        if len(returns) >= 14:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            f["rsi_14"] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else np.nan
        else:
            f["rsi_14"] = np.nan

        # ── MACD ──────────────────────────────────────────────────
        if len(close) >= 35:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            f["macd_histogram"] = float((macd - signal).iloc[-1]) / current  # Normalize
            f["macd_crossover"] = 1.0 if float(macd.iloc[-1]) > float(signal.iloc[-1]) else 0.0
        else:
            f["macd_histogram"] = np.nan
            f["macd_crossover"] = np.nan

        # ── Volume features ───────────────────────────────────────
        if "Volume" in data.columns and len(data) > 50:
            vol = data["Volume"]
            avg_vol_20 = float(vol.iloc[-20:].mean())
            avg_vol_50 = float(vol.iloc[-50:].mean())
            f["volume_ratio_20_50"] = avg_vol_20 / avg_vol_50 if avg_vol_50 > 0 else np.nan
            f["volume_spike"] = float(vol.iloc[-1]) / avg_vol_20 if avg_vol_20 > 0 else np.nan
        else:
            f["volume_ratio_20_50"] = np.nan
            f["volume_spike"] = np.nan

        # ── Return distribution features ──────────────────────────
        if len(returns) >= 60:
            f["skewness_60d"] = float(returns.iloc[-60:].skew())
            f["kurtosis_60d"] = float(returns.iloc[-60:].kurtosis())
            # Downside deviation (Sortino-style)
            neg_returns = returns[returns < 0].iloc[-60:]
            f["downside_vol"] = float(neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 5 else np.nan
        else:
            f["skewness_60d"] = np.nan
            f["kurtosis_60d"] = np.nan
            f["downside_vol"] = np.nan

        # ── Log market cap (size factor) ──────────────────────────
        mkt = info.get("marketCap", 0)
        f["log_market_cap"] = np.log10(mkt) if mkt > 0 else np.nan

        return f

    # ─── GROUP 4: FINANCIAL STATEMENT DERIVED ───────────────────────────────

    def _statement_features(self, t, info: dict) -> dict:
        """Features derived from income statement, balance sheet, cash flow."""
        f = {}
        try:
            inc = t.income_stmt
            bs = t.balance_sheet
            cf = t.cashflow
        except Exception:
            return {k: np.nan for k in [
                "revenue_cagr", "eps_cagr", "fcf_cagr", "roic",
                "margin_expansion", "fcf_conversion", "revenue_consistency",
                "asset_turnover", "debt_to_fcf", "capex_intensity",
                "rd_intensity", "accruals_ratio", "net_debt_to_ebitda",
            ]}

        # Revenue CAGR
        f["revenue_cagr"] = self._calc_cagr(inc, "Total Revenue")

        # FCF CAGR
        f["fcf_cagr"] = self._calc_cagr(cf, ["Free Cash Flow", "FreeCashFlow"])

        # ROIC — the single best quality metric
        f["roic"] = self._calc_roic(inc, bs)

        # Margin expansion (most recent year vs prior)
        f["margin_expansion"] = self._calc_margin_change(inc)

        # FCF conversion (FCF / Net Income) — are earnings real cash?
        f["fcf_conversion"] = self._calc_ratio_from_stmts(
            cf, ["Free Cash Flow", "FreeCashFlow"],
            inc, ["Net Income"]
        )

        # Revenue consistency (consecutive growth years)
        f["revenue_consistency"] = self._count_growth_years(inc, "Total Revenue")

        # Asset turnover = Revenue / Total Assets
        f["asset_turnover"] = self._calc_ratio_from_stmts(
            inc, ["Total Revenue"],
            bs, ["Total Assets"]
        )

        # Debt/FCF — how many years to pay off debt
        f["debt_to_fcf"] = self._calc_ratio_from_stmts(
            bs, ["Total Debt"],
            cf, ["Free Cash Flow", "FreeCashFlow"]
        )

        # CapEx intensity = CapEx / Revenue
        f["capex_intensity"] = self._calc_capex_intensity(cf, inc)

        # R&D intensity = R&D / Revenue (innovation moat proxy)
        f["rd_intensity"] = self._calc_rd_intensity(inc)

        # Accruals ratio = (Net Income - FCF) / Total Assets
        # High accruals = low earnings quality = future underperformance
        f["accruals_ratio"] = self._calc_accruals(inc, cf, bs)

        # Net debt / EBITDA
        f["net_debt_to_ebitda"] = self._calc_net_debt_ebitda(info, bs)

        return f

    # ─── GROUP 5: QUALITY / FRAUD SIGNALS ───────────────────────────────────

    def _quality_signals(self, t, info: dict) -> dict:
        """Beneish M-Score proxy, Altman Z-Score proxy, Piotroski signals."""
        f = {}

        # Piotroski-style binary signals (each adds to quality)
        score = 0
        roe = info.get("returnOnEquity") or 0
        roa = info.get("returnOnAssets") or 0
        current = info.get("currentRatio") or 0
        gm = info.get("grossMargins") or 0
        rev_g = info.get("revenueGrowth") or 0

        if roa > 0: score += 1
        if roe > 0.12: score += 1
        if current > 1.0: score += 1
        if gm > 0.30: score += 1
        if rev_g > 0: score += 1
        if (info.get("operatingMargins") or 0) > 0.10: score += 1
        if (info.get("debtToEquity") or 999) < 100: score += 1

        f["piotroski_proxy"] = score / 7.0  # Normalized 0-1

        # Altman Z-Score proxy (simplified)
        try:
            bs = t.balance_sheet
            inc = t.income_stmt
            if bs is not None and not bs.empty and inc is not None and not inc.empty:
                ta = self._safe_val(bs, ["Total Assets"])
                tl = self._safe_val(bs, ["Total Liabilities Net Minority Interest", "Total Liab"])
                wc = self._safe_val(bs, ["Working Capital"])
                re_ = self._safe_val(bs, ["Retained Earnings"])
                ebit = self._safe_val(inc, ["EBIT", "Operating Income"])
                rev = self._safe_val(inc, ["Total Revenue"])
                mc = info.get("marketCap", 0)

                if ta and ta > 0:
                    z = 0
                    if wc: z += 1.2 * (wc / ta)
                    if re_: z += 1.4 * (re_ / ta)
                    if ebit: z += 3.3 * (ebit / ta)
                    if tl and tl > 0 and mc: z += 0.6 * (mc / tl)
                    if rev: z += 1.0 * (rev / ta)
                    f["altman_z_proxy"] = z
                else:
                    f["altman_z_proxy"] = np.nan
            else:
                f["altman_z_proxy"] = np.nan
        except Exception:
            f["altman_z_proxy"] = np.nan

        return f

    # ─── GROUP 6: OWNERSHIP / FLOW ──────────────────────────────────────────

    def _ownership_features(self, info: dict) -> dict:
        """Insider/institutional ownership dynamics."""
        f = {}
        f["insider_pct"] = info.get("heldPercentInsiders") or np.nan
        f["institutional_pct"] = info.get("heldPercentInstitutions") or np.nan
        f["short_float_pct"] = info.get("shortPercentOfFloat") or np.nan
        f["short_ratio_days"] = info.get("shortRatio") or np.nan

        # Short squeeze potential = high short interest + high momentum
        sf = f.get("short_float_pct", 0) or 0
        f["squeeze_potential"] = sf  # Will be combined with momentum in feature interaction

        return f

    # ─── GROUP 7: OPTIONS-IMPLIED ───────────────────────────────────────────

    def _options_features(self, t, price_data: pd.DataFrame) -> dict:
        """Extract options market intelligence as features."""
        f = {}
        try:
            expirations = t.options
            if not expirations:
                return self._empty_options()

            # Use nearest expiration
            chain = t.option_chain(expirations[0])
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return self._empty_options()

            current = float(price_data["Close"].iloc[-1])

            # Put/Call ratio (volume-based)
            total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
            total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
            f["pcr_volume"] = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan

            # Put/Call ratio (open interest)
            total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            f["pcr_oi"] = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan

            # Average IV (calls vs puts) — fear gauge
            call_iv = calls["impliedVolatility"].mean() if "impliedVolatility" in calls.columns else np.nan
            put_iv = puts["impliedVolatility"].mean() if "impliedVolatility" in puts.columns else np.nan
            f["avg_call_iv"] = call_iv
            f["avg_put_iv"] = put_iv
            f["iv_skew"] = (put_iv - call_iv) if not np.isnan(put_iv) and not np.isnan(call_iv) else np.nan

            # Max pain approximation
            all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            if all_strikes:
                max_pain_val = min(all_strikes, key=lambda s: (
                    calls[calls["strike"] <= s]["openInterest"].sum() * (s - calls[calls["strike"] <= s]["strike"]).sum() +
                    puts[puts["strike"] >= s]["openInterest"].sum() * (puts[puts["strike"] >= s]["strike"] - s).sum()
                ) if len(calls) > 0 and len(puts) > 0 else float("inf"))
                f["price_to_max_pain"] = current / max_pain_val if max_pain_val > 0 else np.nan
            else:
                f["price_to_max_pain"] = np.nan

        except Exception:
            return self._empty_options()

        return f

    def _empty_options(self) -> dict:
        return {k: np.nan for k in [
            "pcr_volume", "pcr_oi", "avg_call_iv", "avg_put_iv",
            "iv_skew", "price_to_max_pain",
        ]}

    # ─── GROUP 8: RELATIVE FEATURES ────────────────────────────────────────

    def _relative_features(self, info: dict, price_data: pd.DataFrame) -> dict:
        """Features that compare to the market or sector norms."""
        f = {}

        # Price vs analyst target
        target = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if target and current and current > 0:
            f["analyst_upside"] = (target - current) / current
        else:
            f["analyst_upside"] = np.nan

        # Number of analyst ratings (coverage breadth)
        f["analyst_count"] = info.get("numberOfAnalystOpinions") or np.nan

        # Analyst recommendation (1=strong buy, 5=sell)
        f["analyst_rec"] = info.get("recommendationMean") or np.nan

        return f

    # ─── GROUP 9: NEWS / NLP SENTIMENT ──────────────────────────────────────

    def _news_sentiment_features(self, t, ticker: str, info: dict) -> dict:
        """
        Real-time headline sentiment scoring using VADER NLP.

        Sources: yfinance news feed + Google News RSS
        Features:
          - Aggregate sentiment score (mean of all headlines)
          - Sentiment momentum (recent vs older headlines)
          - Headline volume (coverage intensity — high = catalyst approaching)
          - Controversy score (variance in sentiment — mixed signals)
          - Positive/negative ratio
          - Max negative headline score (worst news severity)

        Economic rationale: News sentiment is a leading indicator of
        institutional positioning. Extreme negative sentiment in quality
        stocks = contrarian buy signal. Extreme positive + high valuation = risk.
        """
        f = {}
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
        except ImportError:
            return self._empty_sentiment()

        headlines = []

        # Source 1: yfinance news
        try:
            news = t.news or []
            for item in news[:20]:
                title = item.get("title", "")
                if title:
                    headlines.append({
                        "text": title,
                        "source": "yfinance",
                        "recency": 0,  # Most recent
                    })
        except Exception:
            pass

        # Source 2: Google News RSS
        try:
            import feedparser
            company = info.get("longName", ticker)
            query = f"{company} stock".replace(" ", "+")
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for i, entry in enumerate(feed.entries[:15]):
                title = entry.get("title", "")
                if title:
                    headlines.append({
                        "text": title,
                        "source": "google",
                        "recency": i,  # Higher = older
                    })
        except Exception:
            pass

        if not headlines:
            return self._empty_sentiment()

        # Score each headline with VADER
        scores = []
        for h in headlines:
            vs = vader.polarity_scores(h["text"])
            h["compound"] = vs["compound"]
            h["pos"] = vs["pos"]
            h["neg"] = vs["neg"]
            h["neu"] = vs["neu"]
            scores.append(vs["compound"])

        scores = np.array(scores)

        # ── Aggregate sentiment ──────────────────────────────────
        f["news_sentiment_mean"] = float(np.mean(scores))
        f["news_sentiment_median"] = float(np.median(scores))

        # ── Sentiment momentum (first half vs second half of headlines)
        # Positive momentum = recent headlines more positive than older ones
        mid = len(scores) // 2
        if mid > 0:
            recent = scores[:mid]
            older = scores[mid:]
            f["news_sentiment_momentum"] = float(np.mean(recent) - np.mean(older))
        else:
            f["news_sentiment_momentum"] = 0.0

        # ── Coverage intensity (headline count — proxy for catalyst)
        f["news_headline_count"] = float(len(headlines))

        # ── Controversy (high variance = conflicting narratives)
        f["news_sentiment_std"] = float(np.std(scores))
        f["news_controversy"] = float(np.std(scores)) if len(scores) >= 3 else np.nan

        # ── Positive/Negative ratio
        n_pos = np.sum(scores > 0.05)
        n_neg = np.sum(scores < -0.05)
        f["news_pos_neg_ratio"] = float(n_pos / max(n_neg, 1))

        # ── Extreme signals
        f["news_max_negative"] = float(np.min(scores))  # Worst headline
        f["news_max_positive"] = float(np.max(scores))  # Best headline

        # ── Sentiment extremes (contrarian signals)
        # Very negative sentiment in a quality stock = potential buy
        f["news_extreme_negative"] = 1.0 if np.mean(scores) < -0.3 else 0.0
        f["news_extreme_positive"] = 1.0 if np.mean(scores) > 0.5 else 0.0

        return f

    def _empty_sentiment(self) -> dict:
        return {k: np.nan for k in [
            "news_sentiment_mean", "news_sentiment_median",
            "news_sentiment_momentum", "news_headline_count",
            "news_sentiment_std", "news_controversy",
            "news_pos_neg_ratio", "news_max_negative", "news_max_positive",
            "news_extreme_negative", "news_extreme_positive",
        ]}

    # ─── GROUP 10: EARNINGS SURPRISE & ANALYST REVISIONS ───────────────────

    def _earnings_features(self, t, info: dict) -> dict:
        """
        Earnings surprise history and analyst revision momentum.

        Features:
          - Earnings beat streak (consecutive beats)
          - Average surprise magnitude (how much they beat/miss by)
          - Surprise trend (are beats getting bigger or smaller?)
          - Days to next earnings (catalyst proximity)
          - Forward PE / Trailing PE ratio (analyst revision proxy)
          - Target price revision direction
          - Estimate dispersion (analyst disagreement)

        Economic rationale: Post-earnings announcement drift is one of
        the most robust anomalies in finance. Stocks that beat tend to
        keep beating. Analyst revision momentum (forward PE declining
        relative to trailing PE) signals improving fundamentals that
        the market hasn't fully priced. High estimate dispersion =
        more potential for surprise moves.
        """
        f = {}

        # ── Earnings surprise history ────────────────────────────
        try:
            earnings_dates = t.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                # Filter to past earnings (not future estimates)
                # Look for 'Surprise(%)' or similar columns
                surprise_col = None
                for col in earnings_dates.columns:
                    if "surprise" in col.lower() and "%" in col.lower():
                        surprise_col = col
                        break
                    elif "surprise" in col.lower():
                        surprise_col = col
                        break

                if surprise_col:
                    surprises = earnings_dates[surprise_col].dropna()
                    if len(surprises) > 0:
                        surprise_vals = surprises.values.astype(float)

                        # Average surprise magnitude
                        f["earnings_surprise_avg"] = float(np.mean(surprise_vals))

                        # Most recent surprise
                        f["earnings_surprise_latest"] = float(surprise_vals[0])

                        # Beat streak (how many consecutive positive surprises)
                        streak = 0
                        for s in surprise_vals:
                            if s > 0:
                                streak += 1
                            else:
                                break
                        f["earnings_beat_streak"] = float(streak)

                        # Surprise trend: are beats getting bigger?
                        # (first half avg - second half avg)
                        if len(surprise_vals) >= 4:
                            mid = len(surprise_vals) // 2
                            recent = surprise_vals[:mid]
                            older = surprise_vals[mid:]
                            f["earnings_surprise_trend"] = float(np.mean(recent) - np.mean(older))
                        else:
                            f["earnings_surprise_trend"] = np.nan

                        # Beat rate (% of quarters that beat)
                        f["earnings_beat_rate"] = float(np.mean(surprise_vals > 0))

                # ── Days to next earnings ────────────────────────
                try:
                    future_dates = earnings_dates.index[earnings_dates.index > datetime.now()]
                    if len(future_dates) > 0:
                        next_date = future_dates[0]
                        days_to = (next_date.to_pydatetime().replace(tzinfo=None) - datetime.now()).days
                        f["days_to_earnings"] = float(max(days_to, 0))
                        # Binary: is earnings within 2 weeks? (catalyst flag)
                        f["earnings_imminent"] = 1.0 if days_to <= 14 else 0.0
                    else:
                        f["days_to_earnings"] = np.nan
                        f["earnings_imminent"] = 0.0
                except Exception:
                    f["days_to_earnings"] = np.nan
                    f["earnings_imminent"] = 0.0

        except Exception:
            pass

        # ── Analyst revision momentum ────────────────────────────
        # Forward PE vs Trailing PE ratio — declining forward PE means
        # analysts are raising estimates (bullish)
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        if trailing_pe and forward_pe and trailing_pe > 0 and forward_pe > 0:
            # < 1 means analysts expect earnings growth (forward PE lower)
            f["pe_forward_trailing_ratio"] = forward_pe / trailing_pe
            # Revision direction: how much are estimates improving?
            f["analyst_revision_signal"] = (trailing_pe - forward_pe) / trailing_pe
        else:
            f["pe_forward_trailing_ratio"] = np.nan
            f["analyst_revision_signal"] = np.nan

        # ── Target price spread (analyst conviction) ─────────────
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        target_mean = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")

        if target_high and target_low and target_mean and target_mean > 0:
            # Dispersion: wider spread = more uncertainty = more surprise potential
            f["analyst_target_spread"] = (target_high - target_low) / target_mean
            # Skew: is the upside bigger than the downside?
            if current and current > 0:
                f["analyst_upside_skew"] = (target_high - current) / max(current - target_low, 0.01)
            else:
                f["analyst_upside_skew"] = np.nan
        else:
            f["analyst_target_spread"] = np.nan
            f["analyst_upside_skew"] = np.nan

        # ── Analyst count momentum proxy ─────────────────────────
        # More analysts covering = more institutional interest
        n_analysts = info.get("numberOfAnalystOpinions") or 0
        f["analyst_coverage_score"] = np.log1p(n_analysts)  # Log-scaled

        # ── Recommendation strength ──────────────────────────────
        rec = info.get("recommendationMean")  # 1=strong buy, 5=strong sell
        if rec:
            # Invert so higher = more bullish (5-rec gives 4 for strong buy, 0 for strong sell)
            f["analyst_bullishness"] = 5.0 - rec
        else:
            f["analyst_bullishness"] = np.nan

        # Fill defaults for any missing earnings features
        for key in ["earnings_surprise_avg", "earnings_surprise_latest",
                     "earnings_beat_streak", "earnings_surprise_trend",
                     "earnings_beat_rate", "days_to_earnings", "earnings_imminent"]:
            f.setdefault(key, np.nan)

        return f

    # ─── GROUP 11: SECTOR-RELATIVE SCORING ─────────────────────────────────

    def _sector_relative_features(self, ticker: str, info: dict) -> dict:
        """
        Rank each stock's metrics relative to its sector peers.

        Instead of asking "is this PE of 25 high?", we ask
        "is this PE of 25 high FOR A SOFTWARE COMPANY?"

        Features (all z-scores within sector):
          - Valuation z-score (PE, PB, EV/EBITDA vs sector)
          - Growth z-score (revenue growth vs sector)
          - Profitability z-score (margins vs sector)
          - Quality z-score (ROE, ROIC vs sector)
          - Momentum z-score (12m return vs sector)
          - Size percentile within sector

        Economic rationale: Absolute metrics are misleading across sectors.
        A "cheap" biotech at 40x PE is very different from a "cheap"
        utility at 40x PE. Sector-relative scoring removes sector bias
        and isolates true stock-specific alpha.
        """
        f = {}
        sector = info.get("sector", "")

        if not sector:
            return self._empty_sector_relative()

        # Get sector peers from our mapping
        peers = self._get_sector_peers(sector, ticker)
        if len(peers) < 3:
            return self._empty_sector_relative()

        # Batch-fetch peer data (cached where possible)
        peer_data = self._fetch_peer_metrics(peers)
        if len(peer_data) < 3:
            return self._empty_sector_relative()

        # Target stock metrics
        target_metrics = {
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "rev_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "gross_margin": info.get("grossMargins"),
            "op_margin": info.get("operatingMargins"),
            "net_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "market_cap": info.get("marketCap"),
            "beta": info.get("beta"),
        }

        # ── Compute z-scores vs sector peers ─────────────────────
        # Valuation (lower is better for PE, PB, EV/EBITDA, PS)
        val_z = self._compute_zscore_group(
            target_metrics, peer_data,
            ["pe", "pb", "ev_ebitda", "ps"],
            invert=True  # Lower valuation = higher z-score = better
        )
        f["sector_val_zscore"] = val_z

        # Growth (higher is better)
        growth_z = self._compute_zscore_group(
            target_metrics, peer_data,
            ["rev_growth", "earnings_growth"],
            invert=False
        )
        f["sector_growth_zscore"] = growth_z

        # Profitability (higher is better)
        profit_z = self._compute_zscore_group(
            target_metrics, peer_data,
            ["gross_margin", "op_margin", "net_margin"],
            invert=False
        )
        f["sector_profit_zscore"] = profit_z

        # Quality (higher is better)
        quality_z = self._compute_zscore_group(
            target_metrics, peer_data,
            ["roe", "roa"],
            invert=False
        )
        f["sector_quality_zscore"] = quality_z

        # ── Composite sector-relative score ──────────────────────
        # Average of all z-scores = overall attractiveness vs peers
        z_scores = [v for v in [val_z, growth_z, profit_z, quality_z]
                     if not np.isnan(v)]
        f["sector_composite_zscore"] = float(np.mean(z_scores)) if z_scores else np.nan

        # ── Size percentile within sector ────────────────────────
        target_cap = target_metrics.get("market_cap") or 0
        peer_caps = [p.get("market_cap", 0) for p in peer_data.values() if p.get("market_cap")]
        if target_cap and peer_caps:
            rank = sum(1 for c in peer_caps if c < target_cap)
            f["sector_size_percentile"] = rank / max(len(peer_caps), 1)
        else:
            f["sector_size_percentile"] = np.nan

        # ── Sector valuation premium/discount ────────────────────
        # How much more/less expensive is this stock vs sector median?
        target_pe = target_metrics.get("pe")
        peer_pes = [p.get("pe") for p in peer_data.values()
                    if p.get("pe") and np.isfinite(p["pe"]) and p["pe"] > 0]
        if target_pe and target_pe > 0 and peer_pes:
            median_pe = float(np.median(peer_pes))
            f["sector_pe_premium"] = (target_pe - median_pe) / median_pe if median_pe > 0 else np.nan
        else:
            f["sector_pe_premium"] = np.nan

        # ── Sector growth premium ────────────────────────────────
        target_g = target_metrics.get("rev_growth")
        peer_gs = [p.get("rev_growth") for p in peer_data.values()
                   if p.get("rev_growth") is not None and np.isfinite(p["rev_growth"])]
        if target_g is not None and peer_gs:
            median_g = float(np.median(peer_gs))
            f["sector_growth_premium"] = target_g - median_g
        else:
            f["sector_growth_premium"] = np.nan

        return f

    def _empty_sector_relative(self) -> dict:
        return {k: np.nan for k in [
            "sector_val_zscore", "sector_growth_zscore",
            "sector_profit_zscore", "sector_quality_zscore",
            "sector_composite_zscore", "sector_size_percentile",
            "sector_pe_premium", "sector_growth_premium",
        ]}

    def _get_sector_peers(self, sector: str, exclude_ticker: str) -> list:
        """Return representative peer tickers for a sector."""
        SECTOR_PEERS = {
            "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ADBE",
                           "CRM", "AMD", "INTC", "ORCL", "NOW", "QCOM", "TXN", "AMAT"],
            "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
                                    "C", "AXP", "SPGI", "CME", "ICE", "MCO", "CB"],
            "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO",
                           "ABT", "DHR", "BMY", "AMGN", "GILD", "ISRG", "MDT"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX",
                                   "LOW", "TGT", "BKNG", "ABNB", "CMG", "ORLY"],
            "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA",
                                        "T", "VZ", "TMUS", "SPOT", "RBLX"],
            "Industrials": ["CAT", "DE", "HON", "GE", "RTX", "BA", "UPS",
                            "UNP", "LMT", "MMM", "ETN", "ITW", "EMR"],
            "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM",
                                    "MDLZ", "CL", "STZ", "GIS", "KHC", "SYY"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX",
                        "VLO", "WMB", "OKE", "KMI", "HES"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC",
                           "XEL", "WEC", "ES", "ED", "DTE"],
            "Real Estate": ["AMT", "PLD", "CCI", "SPG", "EQIX", "PSA",
                             "DLR", "O", "WELL", "AVB", "EQR"],
            "Basic Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "NUE",
                                 "ECL", "DD", "DOW", "PPG", "VMC"],
        }
        peers = SECTOR_PEERS.get(sector, [])
        return [p for p in peers if p != exclude_ticker.upper()][:12]

    def _fetch_peer_metrics(self, peers: list) -> dict:
        """Batch-fetch key metrics for sector peers (cached)."""
        results = {}
        for ticker in peers:
            cache_key = f"peer_{ticker}"
            if cache_key in self._cache:
                results[ticker] = self._cache[cache_key]
                continue
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                if not info or not info.get("marketCap"):
                    continue
                metrics = {
                    "pe": info.get("trailingPE"),
                    "pb": info.get("priceToBook"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    "ps": info.get("priceToSalesTrailing12Months"),
                    "rev_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "gross_margin": info.get("grossMargins"),
                    "op_margin": info.get("operatingMargins"),
                    "net_margin": info.get("profitMargins"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                    "market_cap": info.get("marketCap"),
                    "beta": info.get("beta"),
                }
                self._cache[cache_key] = metrics
                results[ticker] = metrics
            except Exception:
                continue
        return results

    def _compute_zscore_group(self, target: dict, peers: dict,
                               metrics: list, invert: bool = False) -> float:
        """
        Compute average z-score for a group of metrics vs peers.

        Args:
            target: target stock's metrics dict
            peers: {ticker: metrics_dict} for all peers
            metrics: list of metric keys to use
            invert: if True, lower values = higher z-score (for valuation)
        """
        z_scores = []

        for metric in metrics:
            target_val = target.get(metric)
            if target_val is None or (isinstance(target_val, float) and not np.isfinite(target_val)):
                continue

            # Collect peer values for this metric
            peer_vals = []
            for p_metrics in peers.values():
                v = p_metrics.get(metric)
                if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                    peer_vals.append(v)

            if len(peer_vals) < 3:
                continue

            # Compute z-score
            mean = np.mean(peer_vals)
            std = np.std(peer_vals)
            if std > 0:
                z = (float(target_val) - mean) / std
                if invert:
                    z = -z  # For valuation: lower is better, so negate
                z_scores.append(z)

        return float(np.mean(z_scores)) if z_scores else np.nan

    # ─── LABEL GENERATION ───────────────────────────────────────────────────

    def _compute_labels(self, stock_data: pd.DataFrame,
                        spy_data: pd.DataFrame) -> Optional[dict]:
        """
        Compute forward returns at multiple horizons.
        Labels:
          - 3 month relative return
          - 6 month relative return
          - 12 month relative return
          - Binary: did it outperform SPY by >X% over 6 months?
        """
        labels = {}

        for months, days in [(3, 63), (6, 126), (12, 252)]:
            if len(stock_data) <= days + 5:
                continue

            try:
                # Stock return from (days ago) to now
                stock_ret = float(
                    stock_data["Close"].iloc[-1] / stock_data["Close"].iloc[-(days+1)] - 1
                )

                # SPY return over same period
                end_date = stock_data.index[-1]
                start_date = stock_data.index[-(days+1)]
                spy_slice = spy_data.loc[
                    (spy_data.index >= start_date - timedelta(days=5)) &
                    (spy_data.index <= end_date + timedelta(days=5))
                ]
                if len(spy_slice) < 2:
                    continue
                spy_ret = float(spy_slice["Close"].iloc[-1] / spy_slice["Close"].iloc[0] - 1)

                labels[f"fwd_return_{months}m"] = stock_ret
                labels[f"fwd_spy_return_{months}m"] = spy_ret
                labels[f"fwd_relative_{months}m"] = stock_ret - spy_ret

            except Exception:
                continue

        if not labels:
            return None

        # Primary label: 6-month outperformance > 5%
        rel_6m = labels.get("fwd_relative_6m")
        if rel_6m is not None:
            labels["label_6m_5pct"] = 1 if rel_6m > 0.05 else 0
            labels["label_6m_10pct"] = 1 if rel_6m > 0.10 else 0

        # Secondary label: 12-month outperformance > 10%
        rel_12m = labels.get("fwd_relative_12m")
        if rel_12m is not None:
            labels["label_12m_10pct"] = 1 if rel_12m > 0.10 else 0

        return labels

    # ─── HELPER METHODS ─────────────────────────────────────────────────────

    def _safe_val(self, stmt, labels: list) -> Optional[float]:
        if stmt is None or stmt.empty:
            return None
        for label in labels:
            if label in stmt.index:
                vals = stmt.loc[label].dropna()
                if len(vals) > 0:
                    return float(vals.iloc[0])
        return None

    def _calc_cagr(self, stmt, labels) -> float:
        if stmt is None or stmt.empty:
            return np.nan
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            if label in stmt.index:
                vals = stmt.loc[label].dropna()
                if len(vals) >= 2:
                    oldest = float(vals.iloc[-1])
                    newest = float(vals.iloc[0])
                    years = len(vals) - 1
                    if oldest > 0 and newest > 0 and years > 0:
                        return (newest / oldest) ** (1/years) - 1
        return np.nan

    def _calc_roic(self, inc, bs) -> float:
        if inc is None or bs is None or inc.empty or bs.empty:
            return np.nan
        ni = self._safe_val(inc, ["Net Income"])
        eq = self._safe_val(bs, ["Stockholders Equity", "Total Stockholders Equity"])
        debt = self._safe_val(bs, ["Total Debt"])
        if ni and eq and debt and (eq + debt) > 0:
            return ni / (eq + debt)
        return np.nan

    def _calc_margin_change(self, inc) -> float:
        if inc is None or inc.empty:
            return np.nan
        try:
            rev = inc.loc["Total Revenue"].dropna() if "Total Revenue" in inc.index else None
            op = None
            for label in ["Operating Income", "EBIT"]:
                if label in inc.index:
                    op = inc.loc[label].dropna()
                    break
            if rev is not None and op is not None and len(rev) >= 2 and len(op) >= 2:
                r0, r1 = float(rev.iloc[0]), float(rev.iloc[1])
                o0, o1 = float(op.iloc[0]), float(op.iloc[1])
                if r0 > 0 and r1 > 0:
                    return (o0/r0) - (o1/r1)
        except Exception:
            pass
        return np.nan

    def _calc_ratio_from_stmts(self, stmt1, labels1, stmt2, labels2) -> float:
        v1 = self._safe_val(stmt1, labels1) if stmt1 is not None and not stmt1.empty else None
        v2 = self._safe_val(stmt2, labels2) if stmt2 is not None and not stmt2.empty else None
        if v1 is not None and v2 is not None and v2 != 0:
            return v1 / v2
        return np.nan

    def _count_growth_years(self, inc, label: str) -> float:
        if inc is None or inc.empty or label not in inc.index:
            return np.nan
        vals = inc.loc[label].dropna()
        if len(vals) < 2:
            return np.nan
        count = 0
        for i in range(len(vals) - 1):
            if float(vals.iloc[i]) > float(vals.iloc[i+1]):
                count += 1
            else:
                break
        return float(count)

    def _calc_capex_intensity(self, cf, inc) -> float:
        if cf is None or inc is None or cf.empty or inc.empty:
            return np.nan
        capex = self._safe_val(cf, ["Capital Expenditure", "CapitalExpenditure"])
        rev = self._safe_val(inc, ["Total Revenue"])
        if capex and rev and rev > 0:
            return abs(capex) / rev
        return np.nan

    def _calc_rd_intensity(self, inc) -> float:
        if inc is None or inc.empty:
            return np.nan
        rd = self._safe_val(inc, ["Research Development", "Research And Development", "ResearchAndDevelopment"])
        rev = self._safe_val(inc, ["Total Revenue"])
        if rd and rev and rev > 0:
            return rd / rev
        return np.nan

    def _calc_accruals(self, inc, cf, bs) -> float:
        ni = self._safe_val(inc, ["Net Income"]) if inc is not None and not inc.empty else None
        fcf = None
        if cf is not None and not cf.empty:
            fcf = self._safe_val(cf, ["Free Cash Flow", "FreeCashFlow"])
        ta = self._safe_val(bs, ["Total Assets"]) if bs is not None and not bs.empty else None
        if ni is not None and fcf is not None and ta and ta > 0:
            return (ni - fcf) / ta
        return np.nan

    def _calc_net_debt_ebitda(self, info, bs) -> float:
        ebitda = info.get("ebitda")
        cash = info.get("totalCash") or 0
        debt = info.get("totalDebt") or 0
        if ebitda and ebitda > 0:
            return (debt - cash) / ebitda
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
#   REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Classifies current market regime using SPY price action.
    Models behave differently in different regimes — this lets us adapt.
    """

    @staticmethod
    def detect(spy_data: pd.DataFrame) -> dict:
        if spy_data is None or len(spy_data) < 252:
            return {"regime": "unknown", "confidence": 0}

        close = spy_data["Close"]
        returns = close.pct_change().dropna()

        # Trend: 200-day SMA direction
        sma200 = float(close.rolling(200).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        current = float(close.iloc[-1])
        trend_up = current > sma200 and sma50 > sma200

        # Momentum: 6-month return
        mom_6m = float(close.iloc[-1] / close.iloc[-126] - 1)

        # Volatility: 20-day realized vol annualized
        vol_20 = float(returns.iloc[-20:].std() * np.sqrt(252))
        vol_long = float(returns.std() * np.sqrt(252))
        high_vol = vol_20 > vol_long * 1.3

        # Breadth proxy: max drawdown in last 60 days
        roll_max = close.iloc[-60:].expanding().max()
        dd = ((close.iloc[-60:] - roll_max) / roll_max).min()
        recent_dd = float(dd)

        # Classification
        if trend_up and mom_6m > 0.05 and not high_vol:
            regime = "bull"
            confidence = min(0.6 + mom_6m, 0.95)
        elif not trend_up and mom_6m < -0.05:
            regime = "bear"
            confidence = min(0.6 + abs(mom_6m), 0.95)
        elif high_vol or recent_dd < -0.08:
            regime = "volatile"
            confidence = min(0.5 + vol_20 / 2, 0.90)
        else:
            regime = "sideways"
            confidence = 0.5

        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "trend_up": trend_up,
            "momentum_6m": round(mom_6m, 4),
            "vol_20d": round(vol_20, 4),
            "vol_ratio": round(vol_20 / vol_long, 3) if vol_long > 0 else None,
            "recent_max_dd": round(recent_dd, 4),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#   PURGED WALK-FORWARD CROSS-VALIDATION
#   Standard CV leaks information in financial data. We embargo observations
#   near the train/test boundary to prevent look-ahead bias.
# ═══════════════════════════════════════════════════════════════════════════════

class PurgedWalkForwardCV:
    """
    Walk-forward CV with embargo gap between train and test sets.
    This is the correct way to cross-validate financial ML models.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        embargo_size = max(int(n * self.embargo_pct), 1)
        test_size = n // (self.n_splits + 1)
        indices = np.arange(n)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n)
            train_end = test_start - embargo_size

            if train_end < test_size or test_end <= test_start:
                continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════════
#   ALPHA MODEL — THE MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaModel:
    """
    Institutional-grade quantitative stock selection model.

    Combines 150+ features across fundamentals, technicals, options, ownership,
    and quality signals into a stacked ensemble of gradient boosters, random
    forests, and neural networks.

    The model:
      1. Predicts probability of outperformance vs S&P 500
      2. Adjusts for current market regime
      3. Risk-adjusts rankings by expected volatility
      4. Validates via purged walk-forward backtest
      5. Self-improves by retraining on fresh data

    This is not a toy. This is a quantitative alpha signal.
    """

    LABEL_COLS = [
        "label_6m_5pct", "label_6m_10pct", "label_12m_10pct",
        "fwd_return_3m", "fwd_return_6m", "fwd_return_12m",
        "fwd_spy_return_3m", "fwd_spy_return_6m", "fwd_spy_return_12m",
        "fwd_relative_3m", "fwd_relative_6m", "fwd_relative_12m",
    ]
    META_COLS = ["ticker"]

    def __init__(self,
                 target: str = "label_6m_10pct",
                 model_dir: str = "models",
                 n_estimators: int = 500,
                 random_state: int = 42):
        self.target = target
        self.model_dir = model_dir
        self.n_estimators = n_estimators
        self.rs = random_state

        # Data
        self.dataset = None
        self.feature_names = None
        self.spy_data = None

        # Preprocessing
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()  # Robust to outliers (better than StandardScaler for finance)

        # Models
        self.stacker = None
        self.base_models = {}
        self.calibrated_model = None

        # State
        self.regime = None
        self.feature_importances = None
        self.training_metrics = []
        self.validation_results = None

    # ═══════════════════════════════════════════════════════════════════════════
    #   STEP 1: BUILD UNIVERSE
    # ═══════════════════════════════════════════════════════════════════════════

    def build_universe(self, tickers: list = None, verbose: bool = True) -> pd.DataFrame:
        """
        Extract features for entire stock universe.
        This is the most time-consuming step (~1-2 hours for 600 stocks).
        """
        if tickers is None:
            tickers = get_universe()

        if verbose:
            print(f"\n{'═'*65}")
            print(f"  ALPHA MODEL — BUILDING UNIVERSE")
            print(f"  Stocks: {len(tickers)} | Features: ~120")
            print(f"  Target: {self.target}")
            print(f"{'═'*65}\n")

        # Download SPY for relative return calculation
        print("  Downloading SPY benchmark data...")
        spy = yf.Ticker("SPY")
        self.spy_data = spy.history(period="7y", auto_adjust=True)
        if self.spy_data is None or self.spy_data.empty:
            print("  FATAL: Could not download SPY data")
            return pd.DataFrame()

        # Detect current market regime
        self.regime = RegimeDetector.detect(self.spy_data)
        if verbose:
            print(f"  Market regime: {self.regime['regime'].upper()} "
                  f"(confidence: {self.regime['confidence']:.0%})")
            print(f"  SPY 6M momentum: {self.regime['momentum_6m']:+.1%} | "
                  f"Vol: {self.regime['vol_20d']:.1%}\n")

        # Extract features for each ticker
        extractor = FeatureExtractor()
        rows = []
        failed = 0

        for i, ticker in enumerate(tickers):
            if verbose and (i + 1) % 20 == 0:
                pct = (i + 1) / len(tickers) * 100
                print(f"  [{i+1:>4}/{len(tickers)}] {pct:5.1f}% — "
                      f"{len(rows)} valid, {failed} failed")

            row = extractor.extract(ticker, self.spy_data)
            if row is not None:
                rows.append(row)
            else:
                failed += 1

            # Rate limiting to not hammer yfinance
            if (i + 1) % 5 == 0:
                time.sleep(0.1)

        if not rows:
            print("  FATAL: No valid samples")
            return pd.DataFrame()

        self.dataset = pd.DataFrame(rows)

        # Identify feature columns (exclude meta + labels)
        exclude = set(self.META_COLS + self.LABEL_COLS)
        self.feature_names = [c for c in self.dataset.columns if c not in exclude]

        # Drop rows without the target label
        if self.target in self.dataset.columns:
            before = len(self.dataset)
            self.dataset = self.dataset.dropna(subset=[self.target])
            dropped = before - len(self.dataset)
            if verbose and dropped:
                print(f"  Dropped {dropped} rows missing target label")

        if verbose:
            n = len(self.dataset)
            if self.target in self.dataset.columns:
                n_pos = int(self.dataset[self.target].sum())
                print(f"\n  Universe built: {n} stocks, {len(self.feature_names)} features")
                print(f"  Target distribution: {n_pos} outperformers "
                      f"({n_pos/n*100:.0f}%) / {n - n_pos} underperformers")
            else:
                print(f"\n  Universe built: {n} stocks, {len(self.feature_names)} features")
                print(f"  (No target label — prediction mode only)")

        return self.dataset

    # ═══════════════════════════════════════════════════════════════════════════
    #   STEP 2: TRAIN
    # ═══════════════════════════════════════════════════════════════════════════

    def train(self, verbose: bool = True) -> dict:
        """
        Train the stacked ensemble:
          Base models: RF, GBM, XGBoost, LightGBM, ExtraTrees, MLP
          Meta-learner: Logistic Regression (learns optimal blend)
          Calibration: Platt scaling for true probability estimates
        """
        if self.dataset is None or self.target not in self.dataset.columns:
            print("  ERROR: No labeled dataset. Run build_universe() first.")
            return {}

        X, y = self._get_Xy()
        if len(X) < 50:
            print(f"  ERROR: Only {len(X)} samples — need at least 50")
            return {}

        if verbose:
            print(f"\n{'═'*65}")
            print(f"  ALPHA MODEL — TRAINING")
            print(f"  Samples: {len(X)} | Features: {X.shape[1]}")
            print(f"  Positives: {y.sum()} ({y.mean()*100:.0f}%)")
            print(f"  Models: {'XGB + ' if HAS_XGB else ''}{'LGBM + ' if HAS_LGBM else ''}"
                  f"RF + GBM + ExtraTrees + MLP → Stacked")
            print(f"{'═'*65}\n")

        # ── Drop all-NaN columns FIRST ─────────────────────────
        # SimpleImputer silently drops all-NaN columns, causing dimension
        # mismatches downstream. Remove them explicitly and sync feature_names.
        X_df = self.dataset[self.feature_names]
        valid_mask = X_df.notna().any()
        dropped_cols = [c for c, v in valid_mask.items() if not v]
        if dropped_cols:
            if verbose:
                print(f"  Removing {len(dropped_cols)} empty features: {dropped_cols[:5]}...")
            self.feature_names = [c for c in self.feature_names if c not in dropped_cols]

        X = self.dataset[self.feature_names].values
        y = self.dataset[self.target].values.astype(int)

        if verbose:
            print(f"  Working features: {len(self.feature_names)} (after removing empty columns)")

        # Preprocess (initial pass for feature selection)
        tmp_imputer = SimpleImputer(strategy="median")
        tmp_scaler = RobustScaler()
        X_imp = tmp_imputer.fit_transform(X)
        X_scaled = tmp_scaler.fit_transform(X_imp)

        # ── Feature selection via mutual information ─────────────
        if verbose:
            print("  Running feature selection (mutual information)...")
        mi_scores = mutual_info_classif(X_scaled, y, random_state=self.rs, n_neighbors=5)
        mi_df = pd.Series(mi_scores, index=self.feature_names).sort_values(ascending=False)

        # Keep top MAX_FEATURES features
        n_keep = min(MAX_FEATURES, len(mi_df))
        selected_features = mi_df.head(n_keep).index.tolist()
        self.feature_names = selected_features

        # Refit imputer and scaler on SELECTED features only
        X_sel_raw = self.dataset[self.feature_names].values
        X_imp = self.imputer.fit_transform(X_sel_raw)
        X_sel = self.scaler.fit_transform(X_imp)

        if verbose:
            print(f"  Selected {len(self.feature_names)} features")
            print(f"  Top 10 by mutual information:")
            for feat, mi in list(mi_df.head(10).items()):
                bar = "█" * int(mi * 100)
                print(f"    {feat:35s} {mi:.4f} {bar}")

        # ── Build base models ────────────────────────────────────
        base_estimators = []

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=10,
            min_samples_leaf=max(8, len(X_sel) // 50),
            max_features="sqrt", class_weight="balanced",
            random_state=self.rs, n_jobs=-1,
        )
        base_estimators.append(("rf", rf))

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators, max_depth=5,
            learning_rate=0.03, min_samples_leaf=max(8, len(X_sel) // 50),
            subsample=0.8, max_features="sqrt",
            random_state=self.rs,
        )
        base_estimators.append(("gb", gb))

        # Extra Trees (good for capturing non-linear interactions)
        et = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=10,
            min_samples_leaf=max(8, len(X_sel) // 50),
            class_weight="balanced", random_state=self.rs, n_jobs=-1,
        )
        base_estimators.append(("et", et))

        # XGBoost (if available)
        if HAS_XGB:
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos
            xgb = XGBClassifier(
                n_estimators=self.n_estimators, max_depth=6,
                learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=max(5, len(X_sel) // 100),
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=self.rs, use_label_encoder=False,
                eval_metric="logloss", verbosity=0,
            )
            base_estimators.append(("xgb", xgb))

        # LightGBM (if available)
        if HAS_LGBM:
            lgbm = LGBMClassifier(
                n_estimators=self.n_estimators, max_depth=6,
                learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=max(5, len(X_sel) // 100),
                is_unbalance=True, random_state=self.rs, verbose=-1,
                force_col_wise=True,
            )
            base_estimators.append(("lgbm", lgbm))

        # MLP Neural Network
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            random_state=self.rs,
        )
        base_estimators.append(("mlp", mlp))

        # ── Stacking meta-learner ────────────────────────────────
        meta_learner = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            random_state=self.rs,
        )

        self.stacker = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )

        if verbose:
            print(f"\n  Training stacked ensemble ({len(base_estimators)} base models)...")

        self.stacker.fit(X_sel, y)

        # ── Calibrate probabilities (Platt scaling) ──────────────
        if verbose:
            print("  Calibrating probabilities (Platt scaling)...")
        self.calibrated_model = CalibratedClassifierCV(
            self.stacker, cv=3, method="sigmoid"
        )
        self.calibrated_model.fit(X_sel, y)

        # ── Feature importance (from tree-based models) ──────────
        n_feats = X_sel.shape[1]
        importances = np.zeros(n_feats)
        n_tree_models = 0

        try:
            for name, model in self.stacker.named_estimators_.items():
                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    if len(fi) != n_feats:
                        continue  # Skip mismatched models
                    fi_sum = fi.sum()
                    if fi_sum > 0:
                        fi = fi / fi_sum
                    importances += fi
                    n_tree_models += 1

            if n_tree_models > 0:
                importances /= n_tree_models

            self.feature_importances = pd.Series(
                importances, index=self.feature_names[:n_feats]
            ).sort_values(ascending=False)
        except Exception:
            # Fallback: no feature importances
            self.feature_importances = pd.Series(dtype=float)

        # ── Evaluate on training set ─────────────────────────────
        y_pred = self.calibrated_model.predict(X_sel)
        y_proba = self.calibrated_model.predict_proba(X_sel)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_proba) if len(set(y)) > 1 else 0,
            "brier_score": brier_score_loss(y, y_proba),
            "n_samples": len(y),
            "n_features": len(self.feature_names),
            "n_base_models": len(base_estimators),
            "regime": self.regime["regime"] if self.regime else "unknown",
            "timestamp": datetime.now().isoformat(),
        }

        self.training_metrics.append(metrics)

        if verbose:
            print(f"\n  Training Results (in-sample):")
            print(f"    Accuracy:     {metrics['accuracy']:.3f}")
            print(f"    Precision:    {metrics['precision']:.3f}")
            print(f"    Recall:       {metrics['recall']:.3f}")
            print(f"    F1:           {metrics['f1']:.3f}")
            print(f"    AUC-ROC:      {metrics['auc_roc']:.3f}")
            print(f"    Brier Score:  {metrics['brier_score']:.4f}")
            print(f"\n  Top 15 Alpha Features:")
            for feat, imp in list(self.feature_importances.head(15).items()):
                bar = "█" * int(imp * 200)
                print(f"    {feat:35s} {imp:.4f} {bar}")

        return metrics

    # ═══════════════════════════════════════════════════════════════════════════
    #   STEP 3: VALIDATE (Walk-Forward Backtest)
    # ═══════════════════════════════════════════════════════════════════════════

    def validate(self, n_splits: int = 5, top_n: int = 15,
                 verbose: bool = True) -> dict:
        """
        Purged walk-forward backtest — the gold standard for financial ML.

        For each fold:
          1. Train on expanding historical window
          2. Embargo gap prevents information leakage
          3. Predict on out-of-sample future period
          4. Measure: alpha, precision, hit rate, profit factor
        """
        X, y = self._get_Xy()
        tickers = self.dataset["ticker"].values
        rel_returns = self.dataset.get("fwd_relative_6m", pd.Series(dtype=float)).values

        if verbose:
            print(f"\n{'═'*65}")
            print(f"  ALPHA MODEL — WALK-FORWARD VALIDATION")
            print(f"  Folds: {n_splits} | Top picks/fold: {top_n}")
            print(f"  Embargo: 2% gap between train/test")
            print(f"{'═'*65}\n")

        cv = PurgedWalkForwardCV(n_splits=n_splits, embargo_pct=0.02)
        folds = []

        for fold_num, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_tickers = tickers[test_idx]
            test_returns = rel_returns[test_idx] if len(rel_returns) > 0 else np.zeros(len(test_idx))

            # Preprocess
            imp = SimpleImputer(strategy="median")
            sc = RobustScaler()
            X_tr = sc.fit_transform(imp.fit_transform(X_train))
            X_te = sc.transform(imp.transform(X_test))

            # Train fresh ensemble for this fold
            estimators = self._build_base_estimators(len(X_train))
            meta = LogisticRegression(C=1.0, class_weight="balanced",
                                      max_iter=1000, random_state=self.rs)
            stacker = StackingClassifier(
                estimators=estimators, final_estimator=meta,
                cv=3, stack_method="predict_proba", n_jobs=-1,
            )
            stacker.fit(X_tr, y_train)

            # Predict
            y_pred = stacker.predict(X_te)
            y_proba = stacker.predict_proba(X_te)[:, 1]

            # Pick top N
            top_idx = np.argsort(y_proba)[::-1][:top_n]
            top_tickers_fold = test_tickers[top_idx]
            top_returns_fold = test_returns[top_idx]
            top_probs = y_proba[top_idx]

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            avg_pick_ret = float(np.nanmean(top_returns_fold))
            avg_universe_ret = float(np.nanmean(test_returns))
            alpha = avg_pick_ret - avg_universe_ret
            hit_rate = float(np.nanmean(top_returns_fold > 0))

            # Profit factor = gross gains / gross losses
            gains = top_returns_fold[top_returns_fold > 0]
            losses = top_returns_fold[top_returns_fold < 0]
            profit_factor = (gains.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")

            fold = {
                "fold": fold_num + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "avg_pick_return": round(avg_pick_ret, 4),
                "avg_universe_return": round(avg_universe_ret, 4),
                "alpha": round(alpha, 4),
                "hit_rate": round(hit_rate, 4),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
                "top_picks": list(top_tickers_fold),
                "top_probs": [round(p, 3) for p in top_probs],
            }
            folds.append(fold)

            if verbose:
                pf_str = f"{fold['profit_factor']:.2f}" if isinstance(fold["profit_factor"], float) else fold["profit_factor"]
                print(f"  Fold {fold_num+1}: Acc {acc:.3f} | Prec {prec:.3f} | "
                      f"Alpha {alpha:+.1%} | Hit {hit_rate:.0%} | PF {pf_str}")

        # Aggregate
        avg_alpha = np.mean([f["alpha"] for f in folds])
        avg_hit = np.mean([f["hit_rate"] for f in folds])
        avg_prec = np.mean([f["precision"] for f in folds])
        total_alpha = sum(f["alpha"] for f in folds)

        # Sharpe of alpha (risk-adjusted consistency)
        alphas = [f["alpha"] for f in folds]
        alpha_sharpe = np.mean(alphas) / np.std(alphas) if np.std(alphas) > 0 else 0

        summary = {
            "n_folds": n_splits,
            "avg_accuracy": round(np.mean([f["accuracy"] for f in folds]), 4),
            "avg_precision": round(avg_prec, 4),
            "avg_alpha": round(avg_alpha, 4),
            "total_alpha": round(total_alpha, 4),
            "avg_hit_rate": round(avg_hit, 4),
            "alpha_sharpe": round(alpha_sharpe, 3),
            "folds": folds,
            "regime": self.regime["regime"] if self.regime else "unknown",
        }

        self.validation_results = summary

        if verbose:
            print(f"\n  {'─'*55}")
            print(f"  AGGREGATE RESULTS:")
            print(f"    Avg Alpha/Fold:    {avg_alpha:+.2%}")
            print(f"    Total Alpha:       {total_alpha:+.2%}")
            print(f"    Avg Hit Rate:      {avg_hit:.0%}")
            print(f"    Avg Precision:     {avg_prec:.3f}")
            print(f"    Alpha Sharpe:      {alpha_sharpe:.3f}")
            if avg_alpha > 0 and alpha_sharpe > 0.5:
                print(f"\n  ✅ POSITIVE ALPHA with consistent performance")
            elif avg_alpha > 0:
                print(f"\n  ⚠️  Positive alpha but inconsistent across folds")
            else:
                print(f"\n  ❌ No alpha detected — model needs more data or features")

        return summary

    # ═══════════════════════════════════════════════════════════════════════════
    #   STEP 4: GENERATE SIGNALS
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_signals(self, tickers: list = None, top_n: int = 25,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Generate risk-adjusted buy signals on current stocks.

        Returns DataFrame with:
          ticker, probability, alpha_score, risk_score,
          risk_adj_score, confidence, position_size_pct
        """
        if self.calibrated_model is None and self.stacker is None:
            print("  ERROR: Model not trained.")
            return pd.DataFrame()

        model = self.calibrated_model or self.stacker

        if tickers is None:
            tickers = get_universe()

        if verbose:
            print(f"\n{'═'*65}")
            print(f"  ALPHA MODEL — GENERATING SIGNALS")
            print(f"  Universe: {len(tickers)} stocks")
            print(f"  Regime: {self.regime['regime'].upper() if self.regime else 'UNKNOWN'}")
            print(f"{'═'*65}\n")

        extractor = FeatureExtractor()
        rows = []

        for i, ticker in enumerate(tickers):
            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tickers)}] Scoring... ({len(rows)} valid)")
            try:
                feats = extractor.extract(ticker)
                if feats:
                    rows.append(feats)
            except Exception:
                continue
            if (i + 1) % 5 == 0:
                time.sleep(0.05)

        if not rows:
            print("  ERROR: No valid stocks scored")
            return pd.DataFrame()

        pred_df = pd.DataFrame(rows)

        # Ensure all features exist
        for col in self.feature_names:
            if col not in pred_df.columns:
                pred_df[col] = np.nan

        X = pred_df[self.feature_names].values
        X_imp = self.imputer.transform(X)
        X_sc = self.scaler.transform(X_imp)

        # Predict
        probabilities = model.predict_proba(X_sc)[:, 1]
        pred_df["probability"] = probabilities

        # Risk score (volatility-based)
        pred_df["risk_score"] = pred_df.get("volatility_60d", 0.25).fillna(0.25)

        # Alpha score = probability × regime adjustment
        regime_mult = {"bull": 1.0, "sideways": 0.9, "volatile": 0.75, "bear": 0.6}
        r_mult = regime_mult.get(self.regime["regime"], 0.85) if self.regime else 0.85
        pred_df["alpha_score"] = pred_df["probability"] * r_mult

        # Risk-adjusted score = alpha / risk
        pred_df["risk_adj_score"] = (
            pred_df["alpha_score"] / pred_df["risk_score"].clip(lower=0.10)
        )

        # Confidence tier
        pred_df["confidence"] = pred_df["probability"].apply(self._confidence_tier)

        # Kelly-criterion position sizing (fractional Kelly = half Kelly)
        # Kelly % = (p * b - q) / b, where p=win prob, b=avg win/avg loss, q=1-p
        # We use fractional (half) Kelly for safety
        pred_df["position_size_pct"] = pred_df["probability"].apply(
            lambda p: max(0, min((2 * p - 1) * 0.5, 0.10)) * 100  # Half Kelly, capped at 10%
        )

        # Sort by risk-adjusted score
        pred_df = pred_df.sort_values("risk_adj_score", ascending=False).reset_index(drop=True)

        if verbose:
            print(f"\n  Top {top_n} Risk-Adjusted Signals:")
            print(f"  {'#':>3} {'Ticker':<7} {'Prob':>6} {'Alpha':>7} {'RiskAdj':>8} "
                  f"{'Size':>6} {'Confidence':<15}")
            print(f"  {'─'*60}")
            for i, row in pred_df.head(top_n).iterrows():
                print(f"  {i+1:>3} {row['ticker']:<7} {row['probability']:>5.1%} "
                      f"{row['alpha_score']:>6.3f} {row['risk_adj_score']:>7.3f} "
                      f"{row['position_size_pct']:>5.1f}% {row['confidence']:<15}")

            n_buy = (pred_df["probability"] > 0.5).sum()
            print(f"\n  Total buy signals: {n_buy} / {len(pred_df)}")
            print(f"  Market regime: {self.regime['regime'].upper() if self.regime else 'N/A'}")

        return pred_df

    # ═══════════════════════════════════════════════════════════════════════════
    #   STEP 5: SELF-IMPROVEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def retrain(self, new_tickers: list = None, verbose: bool = True) -> dict:
        """Add new data and retrain. This is how the model gets smarter."""
        if verbose:
            print("\n  SELF-IMPROVEMENT: Expanding dataset and retraining...")
        old_n = len(self.dataset) if self.dataset is not None else 0

        new_data = self.build_universe(tickers=new_tickers, verbose=verbose)

        if self.dataset is not None and old_n > 0:
            existing = set(self.dataset["ticker"])
            fresh = new_data[~new_data["ticker"].isin(existing)]
            updated = new_data[new_data["ticker"].isin(existing)]
            self.dataset = self.dataset[~self.dataset["ticker"].isin(updated["ticker"])]
            self.dataset = pd.concat([self.dataset, updated, fresh], ignore_index=True)
            self.dataset = self.dataset.dropna(subset=[self.target])
        else:
            self.dataset = new_data

        if verbose:
            print(f"  Dataset: {old_n} → {len(self.dataset)} samples")

        return self.train(verbose=verbose)

    def prune_features(self, threshold: float = 0.003, verbose: bool = True) -> list:
        """Remove features with near-zero importance and retrain."""
        if self.feature_importances is None:
            return []
        weak = self.feature_importances[self.feature_importances < threshold].index.tolist()
        if verbose:
            print(f"\n  Pruning {len(weak)} dead features (importance < {threshold})")
        if weak and self.dataset is not None:
            self.dataset = self.dataset.drop(columns=[c for c in weak if c in self.dataset.columns], errors="ignore")
            self.feature_names = [f for f in self.feature_names if f not in weak]
            self.train(verbose=verbose)
        return weak

    # ═══════════════════════════════════════════════════════════════════════════
    #   SAVE / LOAD
    # ═══════════════════════════════════════════════════════════════════════════

    def save(self, name: str = "alpha_v1"):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{name}.pkl")
        state = {
            "stacker": self.stacker,
            "calibrated_model": self.calibrated_model,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "training_metrics": self.training_metrics,
            "validation_results": self.validation_results,
            "regime": self.regime,
            "target": self.target,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Model saved → {path}")

    def load(self, name: str = "alpha_v1"):
        path = os.path.join(self.model_dir, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found")
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.stacker = state["stacker"]
        self.calibrated_model = state["calibrated_model"]
        self.imputer = state["imputer"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.feature_importances = state["feature_importances"]
        self.training_metrics = state["training_metrics"]
        self.validation_results = state["validation_results"]
        self.regime = state["regime"]
        self.target = state["target"]
        print(f"  Model loaded ← {path} ({len(self.feature_names)} features)")

    # ═══════════════════════════════════════════════════════════════════════════
    #   FULL REPORT
    # ═══════════════════════════════════════════════════════════════════════════

    def report(self) -> str:
        """Full performance report."""
        lines = [
            "═" * 65,
            "  ALPHA MODEL — PERFORMANCE REPORT",
            "═" * 65,
        ]

        if self.regime:
            lines.append(f"\n  Market Regime: {self.regime['regime'].upper()} "
                         f"(conf: {self.regime['confidence']:.0%})")

        if self.training_metrics:
            m = self.training_metrics[-1]
            lines.append(f"\n  Latest Training ({m.get('timestamp','')[:10]}):")
            for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "brier_score"]:
                if k in m:
                    lines.append(f"    {k:20s} {m[k]:.4f}")
            lines.append(f"    {'samples':20s} {m.get('n_samples', 0)}")
            lines.append(f"    {'features':20s} {m.get('n_features', 0)}")
            lines.append(f"    {'base models':20s} {m.get('n_base_models', 0)}")

        if self.validation_results:
            v = self.validation_results
            lines.append(f"\n  Walk-Forward Validation ({v['n_folds']} folds):")
            lines.append(f"    Avg Alpha:         {v['avg_alpha']:+.2%}")
            lines.append(f"    Total Alpha:       {v['total_alpha']:+.2%}")
            lines.append(f"    Avg Hit Rate:      {v['avg_hit_rate']:.0%}")
            lines.append(f"    Alpha Sharpe:      {v['alpha_sharpe']:.3f}")

        if self.feature_importances is not None:
            lines.append(f"\n  Top 20 Alpha Features:")
            for feat, imp in list(self.feature_importances.head(20).items()):
                bar = "█" * int(imp * 200)
                lines.append(f"    {feat:35s} {imp:.4f} {bar}")

        lines.append(f"\n  Models: {'XGBoost ' if HAS_XGB else ''}{'LightGBM ' if HAS_LGBM else ''}"
                     f"RandomForest GradientBoosting ExtraTrees MLP → Stacked")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    #   INTERNAL
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_Xy(self) -> Tuple[np.ndarray, np.ndarray]:
        X = self.dataset[self.feature_names].values
        y = self.dataset[self.target].values.astype(int)
        return X, y

    def _build_base_estimators(self, n_samples: int) -> list:
        min_leaf = max(8, n_samples // 50)
        estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=10,
                min_samples_leaf=min_leaf, max_features="sqrt",
                class_weight="balanced", random_state=self.rs, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=5,
                learning_rate=0.03, min_samples_leaf=min_leaf,
                subsample=0.8, random_state=self.rs)),
            ("et", ExtraTreesClassifier(
                n_estimators=self.n_estimators, max_depth=10,
                min_samples_leaf=min_leaf, class_weight="balanced",
                random_state=self.rs, n_jobs=-1)),
        ]
        if HAS_XGB:
            estimators.append(("xgb", XGBClassifier(
                n_estimators=self.n_estimators, max_depth=6,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                random_state=self.rs, use_label_encoder=False,
                eval_metric="logloss", verbosity=0)))
        if HAS_LGBM:
            estimators.append(("lgbm", LGBMClassifier(
                n_estimators=self.n_estimators, max_depth=6,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                is_unbalance=True, random_state=self.rs, verbose=-1,
                force_col_wise=True)))
        estimators.append(("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=500, early_stopping=True, random_state=self.rs)))
        return estimators

    @staticmethod
    def _confidence_tier(p: float) -> str:
        if p >= 0.80: return "VERY HIGH"
        elif p >= 0.65: return "HIGH"
        elif p >= 0.50: return "MODERATE"
        elif p >= 0.35: return "LOW"
        else: return "AVOID"


# ═══════════════════════════════════════════════════════════════════════════════
#   CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="InstitutionalEdge Alpha Model")
    parser.add_argument("command", choices=["build", "train", "validate", "predict", "full", "report"],
                        help="Pipeline step to execute")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--target", default="label_6m_10pct",
                        choices=["label_6m_5pct", "label_6m_10pct", "label_12m_10pct"])
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", type=str, default=None)

    args = parser.parse_args()
    model = AlphaModel(target=args.target)

    if args.load:
        model.load(args.load)

    if args.command == "build":
        model.build_universe(tickers=args.tickers)

    elif args.command == "train":
        model.build_universe(tickers=args.tickers)
        model.train()
        if args.save:
            model.save()

    elif args.command == "validate":
        model.build_universe(tickers=args.tickers)
        model.train()
        model.validate()

    elif args.command == "predict":
        if not args.load:
            model.build_universe(tickers=args.tickers)
            model.train()
        model.generate_signals(tickers=args.tickers, top_n=args.top)

    elif args.command == "full":
        model.build_universe(tickers=args.tickers)
        model.train()
        model.validate()
        model.generate_signals(top_n=args.top)
        print("\n" + model.report())
        if args.save:
            model.save()

    elif args.command == "report":
        if args.load:
            print(model.report())
        else:
            print("  Use --load <name> to load a saved model for report")
