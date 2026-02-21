"""
stock_screener.py — Automatic Stock Discovery Engine

Sources (all free):
- Wikipedia S&P 500 list
- Wikipedia NASDAQ-100 list  
- yfinance bulk data
- Custom screening criteria

Screens for:
- Growth stocks (high rev growth, expanding margins)
- Value stocks (low P/E, P/B, trading below intrinsic value)
- Momentum stocks (price trend, relative strength)
- Dividend stocks (yield, safety, growth)
- Small cap gems (underfollowed, high growth)
- ETF universe (for conservative investors)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from typing import Optional


# ── FREE STOCK UNIVERSES ──────────────────────────────────────────

ETF_UNIVERSE = {
    "US Broad Market": {
        "VOO":  "Vanguard S&P 500 ETF",
        "VTI":  "Vanguard Total Stock Market",
        "SPY":  "SPDR S&P 500 ETF",
        "IVV":  "iShares Core S&P 500",
        "SCHB": "Schwab US Broad Market",
    },
    "Growth": {
        "QQQ":  "Invesco NASDAQ-100",
        "VUG":  "Vanguard Growth ETF",
        "IWF":  "iShares Russell 1000 Growth",
        "SPYG": "SPDR S&P 500 Growth",
        "VONG": "Vanguard Russell 1000 Growth",
    },
    "Value": {
        "VTV":  "Vanguard Value ETF",
        "IWD":  "iShares Russell 1000 Value",
        "SPYV": "SPDR S&P 500 Value",
        "SCHV": "Schwab US Large-Cap Value",
        "VBR":  "Vanguard Small-Cap Value",
    },
    "Dividend": {
        "VYM":  "Vanguard High Dividend Yield",
        "DVY":  "iShares Select Dividend",
        "SCHD": "Schwab US Dividend Equity",
        "HDV":  "iShares Core High Dividend",
        "DGRO": "iShares Dividend Growth",
    },
    "Sector": {
        "XLK":  "Technology Select SPDR",
        "XLF":  "Financial Select SPDR",
        "XLV":  "Health Care Select SPDR",
        "XLE":  "Energy Select SPDR",
        "XLI":  "Industrial Select SPDR",
        "XLY":  "Consumer Discretionary SPDR",
        "XLP":  "Consumer Staples SPDR",
        "XLU":  "Utilities Select SPDR",
        "XLRE": "Real Estate Select SPDR",
        "XLB":  "Materials Select SPDR",
        "XLC":  "Communication Services SPDR",
    },
    "International": {
        "VEA":  "Vanguard Developed Markets",
        "VWO":  "Vanguard Emerging Markets",
        "EFA":  "iShares MSCI EAFE",
        "EEM":  "iShares MSCI Emerging Markets",
        "VXUS": "Vanguard Total International",
    },
    "Fixed Income": {
        "BND":  "Vanguard Total Bond Market",
        "AGG":  "iShares Core US Aggregate Bond",
        "TLT":  "iShares 20+ Year Treasury",
        "SHY":  "iShares 1-3 Year Treasury",
        "LQD":  "iShares Investment Grade Corp Bond",
        "HYG":  "iShares High Yield Corp Bond",
        "TIPS": "iShares TIPS Bond ETF",
    },
    "Alternatives": {
        "GLD":  "SPDR Gold Shares",
        "SLV":  "iShares Silver Trust",
        "VNQ":  "Vanguard Real Estate ETF",
        "PDBC": "Invesco Optimum Yield Commodities",
        "BITO": "ProShares Bitcoin Strategy",
    },
    "Thematic": {
        "ARKK": "ARK Innovation ETF",
        "BOTZ": "Global X Robotics & AI",
        "ICLN": "iShares Global Clean Energy",
        "SOXX": "iShares Semiconductor ETF",
        "CIBR": "First Trust Cybersecurity",
        "AIQ":  "Global X AI & Technology",
        "ROBO": "Robo Global Robotics ETF",
    },
}

SCREEN_PROFILES = {
    "aggressive_growth": {
        "min_revenue_growth": 0.20,
        "min_earnings_growth": 0.15,
        "max_pe": 80,
        "min_gross_margin": 0.30,
        "description": "High-growth companies with expanding revenues",
    },
    "garp": {  # Growth at a Reasonable Price
        "min_revenue_growth": 0.10,
        "max_pe": 35,
        "max_peg": 2.0,
        "min_gross_margin": 0.25,
        "description": "Growth companies at reasonable valuations",
    },
    "deep_value": {
        "max_pe": 15,
        "max_pb": 2.0,
        "min_roe": 0.08,
        "min_current_ratio": 1.0,
        "description": "Undervalued companies with solid fundamentals",
    },
    "quality_compounder": {
        "min_roe": 0.15,
        "min_gross_margin": 0.40,
        "max_debt_equity": 1.0,
        "min_revenue_growth": 0.05,
        "description": "High-quality businesses with durable competitive advantages",
    },
    "dividend_growth": {
        "min_dividend_yield": 0.015,
        "max_payout_ratio": 0.65,
        "min_revenue_growth": 0.03,
        "min_current_ratio": 1.0,
        "description": "Dividend-paying stocks with growth potential",
    },
    "momentum": {
        "min_revenue_growth": 0.08,
        "min_earnings_growth": 0.10,
        "description": "Stocks with strong price and earnings momentum",
    },
    "small_cap_growth": {
        "max_market_cap": 2e9,
        "min_revenue_growth": 0.15,
        "min_gross_margin": 0.30,
        "description": "Small-cap companies with high growth potential",
    },
}


class StockScreener:

    def __init__(self):
        self._universe_cache = None

    # ── UNIVERSE LOADING ─────────────────────────────────────────

    def get_sp500_tickers(self) -> list:
        """Pull S&P 500 tickers from Wikipedia (free)."""
        try:
            url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            table = pd.read_html(url, header=0)[0]
            return table["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception:
            return self._sp500_fallback()

    def get_nasdaq100_tickers(self) -> list:
        """Pull NASDAQ-100 tickers from Wikipedia (free)."""
        try:
            url   = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url, header=0)
            for t in tables:
                if "Ticker" in t.columns or "Symbol" in t.columns:
                    col = "Ticker" if "Ticker" in t.columns else "Symbol"
                    return t[col].dropna().tolist()
        except Exception:
            pass
        return []

    def get_russell2000_sample(self) -> list:
        """A curated sample of Russell 2000 small-cap tickers."""
        return [
            "SMCI","CELH","AVAV","BOOT","CAVA","BROS","SKYW","UFPT",
            "ASTS","RKLB","LUNR","IONQ","ARQT","VERV","KRTX","PRTA",
            "HIMS","JOBY","ACHR","LILM","NUVL","RXRX","ROIV","TVTX",
        ]

    def get_full_universe(self, include_small_cap: bool = False) -> list:
        """Get combined stock universe."""
        print("  Loading stock universe...")
        tickers = set(self.get_sp500_tickers())
        nasdaq  = self.get_nasdaq100_tickers()
        tickers.update(nasdaq)
        if include_small_cap:
            tickers.update(self.get_russell2000_sample())
        tickers = [t for t in tickers if t and "." not in t]
        print(f"  Universe: {len(tickers)} stocks loaded")
        return list(tickers)

    def _sp500_fallback(self) -> list:
        """Hardcoded S&P 500 sample if Wikipedia fails."""
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","JPM",
            "V","UNH","XOM","MA","JNJ","PG","MRK","HD","AVGO","CVX","ABBV","COST",
            "PEP","KO","ADBE","WMT","BAC","MCD","CRM","ACN","TMO","LIN","NFLX",
            "AMD","ORCL","TXN","DHR","ABT","PM","NEE","IBM","QCOM","HON","RTX",
            "AMGN","INTU","LOW","SPGI","GE","CAT","AMAT","BKNG","NOW","ISRG",
            "AXP","SYK","MDT","PLD","GILD","MU","VRTX","SCHW","BLK","C","REGN",
            "ADI","PANW","ZTS","CB","CI","MDLZ","SO","DE","HCA","BSX","ELV",
            "GS","MMC","KLAC","ITW","CME","TJX","WM","AON","DUK","APH","MCO",
            "F","GM","UBER","LYFT","SNAP","PINS","RBLX","COIN","HOOD","SOFI",
            "SMCI","PLTR","MSTR","IONQ","RKLB","ASTS","JOBY","ACHR",
        ]

    # ── BULK DATA FETCH ──────────────────────────────────────────

    def fetch_bulk_metrics(self, tickers: list, batch_size: int = 50) -> pd.DataFrame:
        """
        Fetch key screening metrics for all tickers in batches.
        Uses yfinance (free).
        """
        all_data = []
        batches  = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

        for i, batch in enumerate(batches):
            print(f"  Fetching batch {i+1}/{len(batches)} ({len(batch)} stocks)...")
            for ticker in batch:
                try:
                    t = yf.Ticker(ticker)
                    info = t.info or {}

                    mktcap = info.get("marketCap") or 0
                    if mktcap < 500e6:  # Skip micro caps
                        continue

                    all_data.append({
                        "ticker":           ticker,
                        "name":             info.get("shortName",""),
                        "sector":           info.get("sector",""),
                        "industry":         info.get("industry",""),
                        "market_cap":       mktcap,
                        "current_price":    info.get("currentPrice") or info.get("regularMarketPrice") or 0,
                        "pe_ratio":         info.get("trailingPE"),
                        "forward_pe":       info.get("forwardPE"),
                        "peg_ratio":        info.get("pegRatio"),
                        "pb_ratio":         info.get("priceToBook"),
                        "ps_ratio":         info.get("priceToSalesTrailing12Months"),
                        "ev_ebitda":        info.get("enterpriseToEbitda"),
                        "revenue_growth":   info.get("revenueGrowth"),
                        "earnings_growth":  info.get("earningsGrowth"),
                        "gross_margin":     info.get("grossMargins"),
                        "net_margin":       info.get("profitMargins"),
                        "roe":              info.get("returnOnEquity"),
                        "roa":              info.get("returnOnAssets"),
                        "debt_equity":      (info.get("debtToEquity") or 0) / 100,
                        "current_ratio":    info.get("currentRatio"),
                        "dividend_yield":   info.get("dividendYield") or 0,
                        "payout_ratio":     info.get("payoutRatio") or 0,
                        "beta":             info.get("beta"),
                        "52w_high":         info.get("fiftyTwoWeekHigh"),
                        "52w_low":          info.get("fiftyTwoWeekLow"),
                        "analyst_target":   info.get("targetMeanPrice"),
                        "analyst_upside":   ((info.get("targetMeanPrice",0) or 0) - (info.get("currentPrice",0) or 0)) / max(info.get("currentPrice",1) or 1, 1),
                        "recommendation":   info.get("recommendationKey","").upper(),
                        "num_analysts":     info.get("numberOfAnalystOpinions") or 0,
                        "short_float":      info.get("shortPercentOfFloat") or 0,
                        "insider_own":      info.get("heldPercentInsiders") or 0,
                        "inst_own":         info.get("heldPercentInstitutions") or 0,
                        "free_cash_flow":   info.get("freeCashflow") or 0,
                        "total_revenue":    info.get("totalRevenue") or 0,
                        "fcf_yield":        (info.get("freeCashflow") or 0) / max(mktcap, 1),
                    })
                    time.sleep(0.05)
                except Exception:
                    continue

        df = pd.DataFrame(all_data)
        print(f"  Fetched data for {len(df)} stocks")
        return df

    # ── SCREENERS ────────────────────────────────────────────────

    def screen(self, profile: str, df: pd.DataFrame = None,
               tickers: list = None, limit: int = 20) -> pd.DataFrame:
        """
        Run a named screening profile against the universe.
        """
        if df is None:
            if tickers is None:
                tickers = self.get_sp500_tickers()
            df = self.fetch_bulk_metrics(tickers)

        if df.empty:
            return df

        criteria = SCREEN_PROFILES.get(profile, {})
        mask     = pd.Series([True] * len(df), index=df.index)

        if criteria.get("min_revenue_growth"):
            mask &= df["revenue_growth"].fillna(0) >= criteria["min_revenue_growth"]
        if criteria.get("min_earnings_growth"):
            mask &= df["earnings_growth"].fillna(0) >= criteria["min_earnings_growth"]
        if criteria.get("max_pe"):
            valid_pe = df["pe_ratio"].notna() & (df["pe_ratio"] > 0)
            mask &= ~valid_pe | (df["pe_ratio"] <= criteria["max_pe"])
        if criteria.get("max_peg"):
            valid_peg = df["peg_ratio"].notna() & (df["peg_ratio"] > 0)
            mask &= ~valid_peg | (df["peg_ratio"] <= criteria["max_peg"])
        if criteria.get("max_pb"):
            valid_pb = df["pb_ratio"].notna() & (df["pb_ratio"] > 0)
            mask &= ~valid_pb | (df["pb_ratio"] <= criteria["max_pb"])
        if criteria.get("min_gross_margin"):
            mask &= df["gross_margin"].fillna(0) >= criteria["min_gross_margin"]
        if criteria.get("min_roe"):
            mask &= df["roe"].fillna(0) >= criteria["min_roe"]
        if criteria.get("max_debt_equity"):
            mask &= df["debt_equity"].fillna(99) <= criteria["max_debt_equity"]
        if criteria.get("min_current_ratio"):
            mask &= df["current_ratio"].fillna(0) >= criteria["min_current_ratio"]
        if criteria.get("min_dividend_yield"):
            mask &= df["dividend_yield"].fillna(0) >= criteria["min_dividend_yield"]
        if criteria.get("max_market_cap"):
            mask &= df["market_cap"] <= criteria["max_market_cap"]

        screened = df[mask].copy()

        # Composite ranking score
        screened["screen_score"] = self._rank_screened(screened, profile)
        screened = screened.sort_values("screen_score", ascending=False)

        return screened.head(limit)

    def custom_screen(self, df: pd.DataFrame, filters: dict, limit: int = 20) -> pd.DataFrame:
        """Apply custom filter dict to DataFrame."""
        mask = pd.Series([True] * len(df), index=df.index)
        for key, val in filters.items():
            if key.startswith("min_") and key[4:] in df.columns:
                mask &= df[key[4:]].fillna(-999) >= val
            elif key.startswith("max_") and key[4:] in df.columns:
                mask &= df[key[4:]].fillna(999) <= val
        result = df[mask].copy()
        result["screen_score"] = self._rank_screened(result, "custom")
        return result.sort_values("screen_score", ascending=False).head(limit)

    def _rank_screened(self, df: pd.DataFrame, profile: str) -> pd.Series:
        """Score each stock within screened results."""
        score = pd.Series(0.0, index=df.index)

        # Growth
        rg = df["revenue_growth"].fillna(0).clip(-1, 2)
        eg = df["earnings_growth"].fillna(0).clip(-1, 2)
        score += rg * 30 + eg * 20

        # Quality
        roe = df["roe"].fillna(0).clip(0, 1)
        gm  = df["gross_margin"].fillna(0).clip(0, 1)
        nm  = df["net_margin"].fillna(0).clip(0, 1)
        score += roe * 20 + gm * 15 + nm * 10

        # Safety
        de  = df["debt_equity"].fillna(2).clip(0, 5)
        score -= de * 5

        # Analyst conviction
        up  = df["analyst_upside"].fillna(0).clip(-0.5, 1)
        score += up * 15

        # Insider ownership
        ins = df["insider_own"].fillna(0).clip(0, 0.5)
        score += ins * 10

        return score.round(3)

    # ── ETF ANALYZER ─────────────────────────────────────────────

    def analyze_etfs(self, categories: list = None) -> pd.DataFrame:
        """Fetch performance and metrics for ETF universe."""
        if categories is None:
            categories = list(ETF_UNIVERSE.keys())

        rows = []
        for category in categories:
            etfs = ETF_UNIVERSE.get(category, {})
            for symbol, name in etfs.items():
                try:
                    t    = yf.Ticker(symbol)
                    info = t.info or {}
                    hist = t.history(period="1y", auto_adjust=True)

                    price = info.get("regularMarketPrice") or info.get("navPrice") or 0
                    aum   = info.get("totalAssets") or 0
                    exp   = info.get("annualReportExpenseRatio") or info.get("expenseRatio") or 0
                    div_y = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0

                    # Calculate returns
                    ret_1m = ret_3m = ret_1y = ytd = 0
                    if not hist.empty and len(hist) >= 2:
                        cur = float(hist["Close"].iloc[-1])
                        if len(hist) >= 21:
                            ret_1m = (cur - float(hist["Close"].iloc[-21])) / float(hist["Close"].iloc[-21])
                        if len(hist) >= 63:
                            ret_3m = (cur - float(hist["Close"].iloc[-63])) / float(hist["Close"].iloc[-63])
                        if len(hist) >= 252:
                            ret_1y = (cur - float(hist["Close"].iloc[-252])) / float(hist["Close"].iloc[-252])

                    rows.append({
                        "category":     category,
                        "symbol":       symbol,
                        "name":         name,
                        "price":        price,
                        "aum_b":        aum / 1e9 if aum else 0,
                        "expense_ratio": exp * 100 if exp and exp < 1 else exp,
                        "dividend_yield": div_y * 100 if div_y and div_y < 1 else div_y,
                        "1m_return":    ret_1m * 100,
                        "3m_return":    ret_3m * 100,
                        "1y_return":    ret_1y * 100,
                    })
                    time.sleep(0.05)
                except Exception:
                    continue

        return pd.DataFrame(rows)

    # ── MOMENTUM SCREEN ──────────────────────────────────────────

    def momentum_screen(self, tickers: list = None, top_n: int = 20) -> pd.DataFrame:
        """
        Pure price momentum screen — RS ranking (relative strength).
        Works like institutional 12-1 momentum factor.
        """
        if tickers is None:
            tickers = self.get_sp500_tickers()[:200]

        print(f"  Running momentum screen on {len(tickers)} stocks...")
        try:
            data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)["Close"]
        except Exception:
            return pd.DataFrame()

        rows = []
        for ticker in data.columns:
            prices = data[ticker].dropna()
            if len(prices) < 63:
                continue
            cur = float(prices.iloc[-1])
            try:
                ret_1m  = (cur - float(prices.iloc[-21]))  / float(prices.iloc[-21])  if len(prices) >= 21  else 0
                ret_3m  = (cur - float(prices.iloc[-63]))  / float(prices.iloc[-63])  if len(prices) >= 63  else 0
                ret_6m  = (cur - float(prices.iloc[-126])) / float(prices.iloc[-126]) if len(prices) >= 126 else 0
                ret_12m = (cur - float(prices.iloc[-252])) / float(prices.iloc[-252]) if len(prices) >= 252 else 0
                # 12-1 momentum (skip most recent month)
                ret_12_1 = ret_12m - ret_1m

                rows.append({
                    "ticker":    ticker,
                    "price":     round(cur, 2),
                    "1m_ret":    round(ret_1m  * 100, 2),
                    "3m_ret":    round(ret_3m  * 100, 2),
                    "6m_ret":    round(ret_6m  * 100, 2),
                    "12m_ret":   round(ret_12m * 100, 2),
                    "momentum":  round(ret_12_1 * 100, 2),
                })
            except Exception:
                continue

        df = pd.DataFrame(rows).sort_values("momentum", ascending=False)
        return df.head(top_n)

    # ── PRINT RESULTS ────────────────────────────────────────────

    @staticmethod
    def print_screen_results(df: pd.DataFrame, title: str = "Screen Results"):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        if df.empty:
            print("  No stocks passed the screen.")
            return

        cols = ["ticker","name","sector","market_cap","pe_ratio","revenue_growth",
                "earnings_growth","gross_margin","roe","analyst_upside","screen_score"]
        display_cols = [c for c in cols if c in df.columns]
        print(f"\n{'#':<4} {'Ticker':<8} {'Company':<28} {'Sector':<20} {'Rev Gr':<10} {'P/E':<8} {'Score'}")
        print("─"*85)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            rg  = f"{row.get('revenue_growth',0)*100:.0f}%" if row.get('revenue_growth') else "N/A"
            pe  = f"{row.get('pe_ratio',0):.1f}x" if row.get('pe_ratio') else "N/A"
            sc  = f"{row.get('screen_score',0):.2f}"
            mkt = f"${row.get('market_cap',0)/1e9:.1f}B" if row.get('market_cap') else "N/A"
            name = str(row.get('name',''))[:26]
            sector = str(row.get('sector',''))[:18]
            print(f"{i:<4} {row.get('ticker',''):<8} {name:<28} {sector:<20} {rg:<10} {pe:<8} {sc}")
