"""
sentiment_analysis.py — News sentiment scoring using VADER (no API key needed).

Analyzes headlines from yfinance + Google News RSS to generate
a sentiment score and overall market narrative for a stock.
"""

import re
import math
from data_fetcher import DataFetcher

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


# Fallback simple keyword-based sentiment if VADER not installed
BULLISH_KEYWORDS = [
    "surge", "soar", "rally", "record", "beat", "exceed", "growth", "profit",
    "upgrade", "buy", "strong", "positive", "bullish", "outperform", "expand",
    "gains", "rises", "jumps", "top", "momentum", "breakout", "revenue growth",
    "earnings beat", "raises guidance", "raises dividend", "buyback", "acquisition",
    "partnership", "contract", "award", "innovation", "launch", "approval"
]

BEARISH_KEYWORDS = [
    "fall", "drop", "decline", "miss", "disappoint", "loss", "warning", "cut",
    "downgrade", "sell", "weak", "negative", "bearish", "underperform", "contract",
    "lawsuit", "investigation", "recall", "layoff", "restructure", "default",
    "bankruptcy", "debt", "pressure", "concern", "risk", "uncertainty", "freeze"
]


class SentimentAnalyzer:

    def __init__(self):
        self.fetcher = DataFetcher()
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    def analyze(self, ticker: str, company_name: str = "") -> dict:
        """Full sentiment analysis pipeline."""
        headlines = self.fetcher.get_news_headlines(ticker, company_name)

        if not headlines:
            return {
                "score": 50,
                "overall": "Neutral (no news found)",
                "headline_count": 0,
                "headlines": [],
                "scored_headlines": [],
            }

        scored = []
        for h in headlines:
            title = h.get("title", "")
            if not title:
                continue
            s = self._score_headline(title)
            scored.append({
                "title": title,
                "source": h.get("source", ""),
                "sentiment_score": s,
                "label": self._label(s),
            })

        if not scored:
            return {"score": 50, "overall": "Neutral", "headline_count": 0, "headlines": [], "scored_headlines": []}

        raw_scores = [s["sentiment_score"] for s in scored]
        avg_score = sum(raw_scores) / len(raw_scores)

        # Normalize to 0-100 (VADER compound is -1 to 1, keywords are -1 to 1)
        normalized = (avg_score + 1) / 2 * 100

        bullish_count = sum(1 for s in scored if s["sentiment_score"] > 0.05)
        bearish_count = sum(1 for s in scored if s["sentiment_score"] < -0.05)
        neutral_count = len(scored) - bullish_count - bearish_count

        overall = self._overall_narrative(normalized, bullish_count, bearish_count, len(scored))

        return {
            "score": round(normalized, 2),
            "overall": overall,
            "headline_count": len(scored),
            "bullish_headlines": bullish_count,
            "bearish_headlines": bearish_count,
            "neutral_headlines": neutral_count,
            "avg_compound": round(avg_score, 4),
            "headlines": [s["title"] for s in scored[:5]],
            "scored_headlines": scored[:10],
        }

    def _score_headline(self, text: str) -> float:
        """Return a compound sentiment score between -1 and 1."""
        if self.vader:
            return self.vader.polarity_scores(text)["compound"]
        else:
            return self._keyword_score(text)

    def _keyword_score(self, text: str) -> float:
        """Simple keyword-based scoring fallback."""
        text_lower = text.lower()
        bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / max(total, 1) * min(1.0, math.sqrt(total) * 0.5)

    def _label(self, score: float) -> str:
        if score > 0.15:   return "Bullish"
        elif score > 0.05: return "Slightly Bullish"
        elif score < -0.15: return "Bearish"
        elif score < -0.05: return "Slightly Bearish"
        else:              return "Neutral"

    def _overall_narrative(self, normalized: float, bull: int, bear: int, total: int) -> str:
        pct_bull = bull / total if total > 0 else 0
        pct_bear = bear / total if total > 0 else 0

        if normalized >= 70 and pct_bull >= 0.6:
            return f"Very Positive — {bull}/{total} headlines bullish"
        elif normalized >= 60:
            return f"Positive — {bull}/{total} headlines bullish"
        elif normalized <= 35 and pct_bear >= 0.6:
            return f"Very Negative — {bear}/{total} headlines bearish"
        elif normalized <= 42:
            return f"Negative — {bear}/{total} headlines bearish"
        else:
            return f"Mixed/Neutral — {bull} bullish, {bear} bearish of {total} headlines"
