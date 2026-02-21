"""
data_fetcher.py — Centralized data retrieval using free APIs.
Sources: yfinance (stocks), CoinGecko (crypto), NewsAPI/RSS (news)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import feedparser
import warnings
warnings.filterwarnings("ignore")


class DataFetcher:
    """Fetches stock data, financial statements, and news headlines."""

    def __init__(self):
        self._cache = {}

    def get_stock_data(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Fetch OHLCV price history via yfinance."""
        cache_key = f"{ticker}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, auto_adjust=True)
            if df.empty:
                return None
            self._cache[cache_key] = df
            return df
        except Exception as e:
            print(f"  WARNING: {e}")
            return None

    def get_company_info(self, ticker: str) -> dict:
        """Get company metadata and key stats from yfinance."""
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            return info
        except Exception:
            return {}

    def get_financials(self, ticker: str) -> dict:
        """Get income statement, balance sheet, and cash flow data."""
        try:
            t = yf.Ticker(ticker)
            return {
                "income_stmt": t.income_stmt,
                "balance_sheet": t.balance_sheet,
                "cash_flow": t.cashflow,
                "quarterly_income": t.quarterly_income_stmt,
            }
        except Exception:
            return {}

    def get_earnings_history(self, ticker: str) -> pd.DataFrame:
        """Get earnings history for beat/miss tracking."""
        try:
            t = yf.Ticker(ticker)
            return t.earnings_dates
        except Exception:
            return pd.DataFrame()

    def get_news_headlines(self, ticker: str, company_name: str = "") -> list:
        """
        Fetch recent news headlines via:
        1. yfinance news
        2. Google News RSS feed
        """
        headlines = []

        # Source 1: yfinance news
        try:
            t = yf.Ticker(ticker)
            news = t.news or []
            for item in news[:15]:
                headlines.append({
                    "title": item.get("title", ""),
                    "source": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": item.get("providerPublishTime", 0),
                })
        except Exception:
            pass

        # Source 2: Google News RSS
        try:
            query = f"{ticker} stock" if not company_name else f"{company_name} stock"
            rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:10]:
                headlines.append({
                    "title": entry.get("title", ""),
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception:
            pass

        return headlines

    def get_crypto_data(self, coin_id: str, days: int = 365) -> dict:
        """Fetch crypto data from CoinGecko (free, no API key needed)."""
        base = "https://api.coingecko.com/api/v3"
        result = {}

        try:
            # Price history
            r = requests.get(
                f"{base}/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": days},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                prices = data.get("prices", [])
                result["prices"] = pd.DataFrame(prices, columns=["timestamp", "price"])
                result["prices"]["date"] = pd.to_datetime(result["prices"]["timestamp"], unit="ms")
                result["prices"].set_index("date", inplace=True)
        except Exception:
            pass

        try:
            # Coin info / market data
            r = requests.get(
                f"{base}/coins/{coin_id}",
                params={"localization": "false", "tickers": "false"},
                timeout=10,
            )
            if r.status_code == 200:
                result["info"] = r.json()
        except Exception:
            pass

        try:
            # Top crypto list for context
            r = requests.get(
                f"{base}/coins/markets",
                params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 20, "page": 1},
                timeout=10,
            )
            if r.status_code == 200:
                result["market_context"] = r.json()
        except Exception:
            pass

        return result

    def get_sector_peers(self, ticker: str, info: dict) -> list:
        """Return a rough list of sector peers for relative comparison."""
        sector_peers = {
            "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
            "Financial Services": ["JPM", "BAC", "GS", "MS", "BRK-B"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
            "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
            "Industrials": ["CAT", "DE", "HON", "GE", "BA"],
            "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
            "Real Estate": ["AMT", "PLD", "CCI", "SPG", "EQIX"],
            "Basic Materials": ["LIN", "APD", "FCX", "NEM", "NUE"],
        }
        sector = info.get("sector", "")
        peers = sector_peers.get(sector, [])
        return [p for p in peers if p != ticker.upper()][:5]
