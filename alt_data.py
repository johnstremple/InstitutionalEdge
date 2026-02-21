"""
alt_data.py — Alternative Data Sources

1. Reddit Sentiment (r/wallstreetbets, r/investing, r/stocks)
   - Uses Reddit's free JSON API (no key needed)
   - Mention counts, bullish/bearish sentiment
   - WSB hype score

2. Google Trends (via pytrends - free)
   - Search interest as leading indicator
   - Trending = retail interest building

3. Earnings Calendar
   - Next earnings date
   - Historical beat/miss rate
   - Estimate revision trend
   - Pre-earnings drift signal
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import re


REDDIT_HEADERS = {
    "User-Agent": "InstitutionalEdge/1.0 (financial research bot)",
    "Accept": "application/json",
}

BULLISH_WORDS  = ["bull","calls","moon","rocket","buy","long","squeeze","squeeze","breakout",
                  "upside","growth","beat","upgrade","outperform","strong","surge"]
BEARISH_WORDS  = ["bear","puts","crash","short","sell","dump","overvalued","miss","downgrade",
                  "weak","collapse","bankruptcy","fraud","lawsuit","decline","warning"]

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "SecurityAnalysis"]


class AltDataAnalyzer:

    def full_analysis(self, ticker: str, company_name: str = "") -> dict:
        print(f"    Fetching Reddit mentions & sentiment...")
        reddit   = self._reddit_sentiment(ticker, company_name)

        print(f"    Fetching earnings calendar...")
        earnings = self._earnings_calendar(ticker)

        print(f"    Fetching Google Trends...")
        trends   = self._google_trends(ticker, company_name)

        score    = self._score(reddit, trends, earnings)

        return {
            "score":            score,
            "reddit":           reddit,
            "google_trends":    trends,
            "earnings_calendar": earnings,
            "signal":           self._signal(reddit, trends, earnings),
            "summary":          self._summary(reddit, trends, earnings),
        }

    # ── REDDIT ───────────────────────────────────────────────────

    def _reddit_sentiment(self, ticker: str, company_name: str) -> dict:
        """Pull Reddit mentions using free JSON API."""
        mentions   = []
        total_bull = 0
        total_bear = 0
        total_ment = 0
        wsb_score  = 0

        search_terms = [f"${ticker}", ticker]
        if company_name:
            # Add shortened company name
            short = company_name.split()[0] if company_name else ""
            if len(short) > 3:
                search_terms.append(short)

        for subreddit in SUBREDDITS[:3]:
            for term in search_terms[:2]:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        "q":      term,
                        "sort":   "new",
                        "limit":  25,
                        "t":      "week",
                    }
                    r = requests.get(url, headers=REDDIT_HEADERS,
                                     params=params, timeout=8)
                    if r.status_code != 200:
                        continue

                    data  = r.json()
                    posts = data.get("data", {}).get("children", [])

                    for post in posts:
                        pdata = post.get("data", {})
                        title = (pdata.get("title") or "").lower()
                        body  = (pdata.get("selftext") or "").lower()
                        text  = title + " " + body
                        score_post = pdata.get("score", 0)
                        comments   = pdata.get("num_comments", 0)

                        # Check if mentions our ticker
                        if f"${ticker.lower()}" in text or ticker.lower() in text.split():
                            bull = sum(1 for w in BULLISH_WORDS if w in text)
                            bear = sum(1 for w in BEARISH_WORDS if w in text)

                            total_bull += bull
                            total_bear += bear
                            total_ment += 1

                            if subreddit == "wallstreetbets":
                                wsb_score += (score_post + comments * 2)

                            mentions.append({
                                "subreddit": subreddit,
                                "title":     pdata.get("title","")[:100],
                                "score":     score_post,
                                "comments":  comments,
                                "sentiment": "bullish" if bull > bear else "bearish" if bear > bull else "neutral",
                                "url":       f"https://reddit.com{pdata.get('permalink','')}",
                            })

                    time.sleep(0.3)  # Reddit rate limit

                except Exception:
                    continue

        # Sentiment
        if total_ment == 0:
            sentiment = "Not trending on Reddit"
            wsb_hype  = "No mentions found"
        else:
            bull_pct = total_bull / max(total_bull + total_bear, 1)
            if bull_pct > 0.65:   sentiment = "🟢 Bullish Reddit sentiment"
            elif bull_pct > 0.50: sentiment = "🟡 Slightly bullish"
            elif bull_pct < 0.35: sentiment = "🔴 Bearish Reddit sentiment"
            else:                  sentiment = "🟡 Mixed Reddit sentiment"

            if wsb_score > 10000: wsb_hype = "🔥🔥 VIRAL on WSB"
            elif wsb_score > 1000: wsb_hype = "🔥 Hot on WSB"
            elif wsb_score > 100:  wsb_hype = "Active WSB discussion"
            else:                   wsb_hype = "Low WSB activity"

        return {
            "mention_count":     total_ment,
            "bullish_posts":     total_bull,
            "bearish_posts":     total_bear,
            "bull_pct":          round(total_bull / max(total_bull+total_bear,1) * 100, 1),
            "wsb_hype":          wsb_hype,
            "wsb_score":         wsb_score,
            "sentiment":         sentiment,
            "top_posts":         sorted(mentions, key=lambda x: x["score"], reverse=True)[:5],
        }

    # ── GOOGLE TRENDS ─────────────────────────────────────────────

    def _google_trends(self, ticker: str, company_name: str) -> dict:
        """Fetch Google Trends data via pytrends (free)."""
        try:
            from pytrends.request import TrendReq
            pt = TrendReq(hl="en-US", tz=360, timeout=(10,25))

            keywords = [ticker]
            if company_name:
                keywords.append(company_name.split()[0])

            pt.build_payload(keywords[:1], cat=0, timeframe="today 3-m", geo="US")
            df = pt.interest_over_time()

            if df is None or df.empty:
                return {"status": "No trend data", "score": 50}

            col       = keywords[0]
            if col not in df.columns:
                col = df.columns[0]

            current   = float(df[col].iloc[-1])
            avg_3m    = float(df[col].mean())
            peak_3m   = float(df[col].max())
            trend_dir = "Rising" if df[col].iloc[-1] > df[col].iloc[-4] else "Falling"

            vs_avg = (current - avg_3m) / avg_3m * 100 if avg_3m > 0 else 0

            if current > avg_3m * 1.5:
                signal = "🔥 Search interest surging — retail attention building"
            elif current > avg_3m * 1.2:
                signal = "📈 Above average search interest"
            elif current < avg_3m * 0.7:
                signal = "📉 Below average interest — under the radar"
            else:
                signal = "➡️ Normal search interest level"

            return {
                "current_interest":  round(current, 1),
                "avg_3m":            round(avg_3m, 1),
                "peak_3m":           round(peak_3m, 1),
                "vs_avg_pct":        round(vs_avg, 1),
                "trend_direction":   trend_dir,
                "signal":            signal,
                "score":             min(int(current), 100),
            }

        except ImportError:
            return self._trends_fallback(ticker)
        except Exception:
            return self._trends_fallback(ticker)

    def _trends_fallback(self, ticker: str) -> dict:
        """Fallback when pytrends unavailable."""
        return {
            "current_interest": None,
            "signal": "Install pytrends for Google Trends: pip install pytrends",
            "score":  50,
        }

    # ── EARNINGS CALENDAR ─────────────────────────────────────────

    def _earnings_calendar(self, ticker: str) -> dict:
        """Fetch earnings date, EPS history, beat/miss rate."""
        try:
            t    = yf.Ticker(ticker)
            info = t.info or {}

            # Next earnings date
            next_earnings = info.get("earningsTimestamp") or info.get("earningsDate")
            if next_earnings:
                if isinstance(next_earnings, (int, float)):
                    next_dt = datetime.fromtimestamp(next_earnings)
                else:
                    next_dt = pd.Timestamp(next_earnings).to_pydatetime()
                days_to_earnings = (next_dt - datetime.now()).days
                earnings_date_str = next_dt.strftime("%Y-%m-%d")
            else:
                days_to_earnings  = None
                earnings_date_str = "Unknown"

            # Historical earnings
            try:
                earnings_hist = t.earnings_history
            except Exception:
                earnings_hist = None

            beat_rate = None
            avg_surprise = None
            recent_surprises = []

            if earnings_hist is not None and not earnings_hist.empty:
                if "epsEstimate" in earnings_hist.columns and "epsActual" in earnings_hist.columns:
                    eh = earnings_hist.dropna(subset=["epsEstimate","epsActual"])
                    beats = (eh["epsActual"] > eh["epsEstimate"]).sum()
                    beat_rate = beats / len(eh) * 100 if len(eh) > 0 else None
                    surprises = ((eh["epsActual"] - eh["epsEstimate"]) / eh["epsEstimate"].abs() * 100)
                    avg_surprise = float(surprises.mean())
                    recent_surprises = surprises.tail(4).round(1).tolist()

            # Analyst estimates
            eps_fwd   = info.get("forwardEps")
            eps_trail = info.get("trailingEps")
            rev_est   = info.get("revenueEstimateAvg")

            # Pre-earnings drift signal
            drift_signal = None
            if days_to_earnings is not None:
                if 0 < days_to_earnings <= 14:
                    drift_signal = "📅 EARNINGS IN <2 WEEKS — historically stocks drift toward consensus pre-earnings"
                elif 0 < days_to_earnings <= 30:
                    drift_signal = f"📅 Earnings in {days_to_earnings} days — watch for estimate revisions"
                elif days_to_earnings < 0:
                    drift_signal = f"📅 Earnings {abs(days_to_earnings)} days ago"

            # Beat/miss signal
            beat_signal = ""
            if beat_rate is not None:
                if beat_rate >= 80:
                    beat_signal = f"🟢 Consistent beater ({beat_rate:.0f}% beat rate, avg +{avg_surprise:.1f}% surprise)"
                elif beat_rate >= 60:
                    beat_signal = f"🟡 Usually beats ({beat_rate:.0f}% beat rate)"
                else:
                    beat_signal = f"🔴 Frequent misser ({beat_rate:.0f}% beat rate)"

            return {
                "next_earnings_date": earnings_date_str,
                "days_to_earnings":   days_to_earnings,
                "eps_forward":        eps_fwd,
                "eps_trailing":       eps_trail,
                "beat_rate_pct":      round(beat_rate, 1) if beat_rate is not None else None,
                "avg_eps_surprise":   round(avg_surprise, 1) if avg_surprise is not None else None,
                "recent_surprises":   recent_surprises,
                "drift_signal":       drift_signal,
                "beat_signal":        beat_signal,
            }

        except Exception:
            return {
                "next_earnings_date": "Unknown",
                "days_to_earnings":   None,
                "beat_signal":        "",
            }

    # ── SCORE ────────────────────────────────────────────────────

    def _score(self, reddit: dict, trends: dict, earnings: dict) -> float:
        score = 50.0

        # Reddit
        mentions = reddit.get("mention_count", 0)
        bull_pct = reddit.get("bull_pct", 50)
        if mentions > 10 and bull_pct > 60:  score += 12
        elif mentions > 5 and bull_pct > 55: score += 6
        elif bull_pct < 35:                  score -= 10

        # WSB hype
        wsb = reddit.get("wsb_score", 0)
        if wsb > 5000:   score += 10
        elif wsb > 1000: score += 5

        # Google Trends
        vs_avg = trends.get("vs_avg_pct", 0)
        if vs_avg > 50:   score += 8
        elif vs_avg > 20: score += 4
        elif vs_avg < -30: score -= 5

        # Earnings
        beat_rate = earnings.get("beat_rate_pct")
        if beat_rate:
            if beat_rate >= 80: score += 10
            elif beat_rate >= 65: score += 5
            elif beat_rate < 50: score -= 8

        dte = earnings.get("days_to_earnings")
        avg_surp = earnings.get("avg_eps_surprise", 0)
        if dte and 0 < dte <= 14 and avg_surp and avg_surp > 5:
            score += 8  # Consistent beater approaching earnings

        return round(min(max(score, 0), 100), 2)

    def _signal(self, reddit: dict, trends: dict, earnings: dict) -> str:
        parts = []
        if reddit.get("wsb_hype") and "🔥" in reddit.get("wsb_hype",""):
            parts.append(reddit["wsb_hype"])
        if trends.get("signal") and "surging" in trends.get("signal","").lower():
            parts.append("Google searches surging")
        if earnings.get("beat_signal"):
            parts.append(earnings["beat_signal"])
        if earnings.get("drift_signal"):
            parts.append(earnings["drift_signal"])
        return " | ".join(parts) if parts else "No strong alternative data signal"

    def _summary(self, reddit: dict, trends: dict, earnings: dict) -> str:
        parts = []
        mc = reddit.get("mention_count", 0)
        if mc > 0:
            parts.append(f"{mc} Reddit mentions this week ({reddit.get('bull_pct',50):.0f}% bullish)")
        if trends.get("current_interest"):
            parts.append(f"Google interest: {trends['current_interest']}/100")
        ed = earnings.get("next_earnings_date","")
        if ed and ed != "Unknown":
            parts.append(f"Next earnings: {ed}")
        return " | ".join(parts) if parts else "Limited alternative data available"
