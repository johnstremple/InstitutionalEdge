"""
risk_manager.py — Institutional risk analytics.

Metrics:
- Value at Risk (VaR): Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Sharpe, Sortino, Calmar, Information Ratio
- Max Drawdown & Drawdown duration
- Beta, Alpha, Correlation to SPY
- Volatility (realized, rolling)
- Ulcer Index
"""

import numpy as np
import pandas as pd
from typing import Optional


class RiskManager:

    TRADING_DAYS  = 252
    RISK_FREE_RATE = 0.04 / 252   # daily risk-free rate

    def __init__(self, price_data: Optional[pd.DataFrame], portfolio: Optional[pd.DataFrame] = None):
        self.df        = price_data
        self.portfolio = portfolio
        self.returns   = None
        self.prices    = None

        if price_data is not None and not price_data.empty:
            self.prices  = price_data["Close"]
            self.returns = price_data["Close"].pct_change().dropna()

    # ─── PUBLIC ────────────────────────────────────────────────────────────────

    def full_assessment(self) -> dict:
        if self.returns is None or len(self.returns) < 30:
            return {"score": 50, "error": "Insufficient data for risk assessment"}

        r = self.returns

        # Core risk metrics
        ann_vol      = float(r.std() * np.sqrt(self.TRADING_DAYS))
        ann_return   = float((1 + r.mean()) ** self.TRADING_DAYS - 1)
        ann_rf       = self.RISK_FREE_RATE * self.TRADING_DAYS
        sharpe       = (ann_return - ann_rf) / ann_vol if ann_vol > 0 else 0

        downside_r   = r[r < 0]
        downside_std = float(downside_r.std() * np.sqrt(self.TRADING_DAYS)) if len(downside_r) > 0 else ann_vol
        sortino      = (ann_return - ann_rf) / downside_std if downside_std > 0 else 0

        max_dd       = self._max_drawdown(self.prices)
        calmar       = ann_return / abs(max_dd) if max_dd != 0 else 0

        var_95_hist  = self._var_historical(r, 0.95)
        var_99_hist  = self._var_historical(r, 0.99)
        cvar_95      = self._cvar(r, 0.95)

        beta, alpha  = self._beta_alpha(r)
        ulcer_index  = self._ulcer_index(self.prices)

        score = self._risk_score(sharpe, max_dd, ann_vol, beta)

        return {
            "score":          score,
            "annual_return":  round(ann_return, 4),
            "annual_vol":     round(ann_vol, 4),
            "sharpe_ratio":   round(sharpe, 3),
            "sortino_ratio":  round(sortino, 3),
            "calmar_ratio":   round(calmar, 3),
            "max_drawdown":   round(max_dd, 4),
            "var_95":         round(var_95_hist, 4),
            "var_99":         round(var_99_hist, 4),
            "cvar_95":        round(cvar_95, 4),
            "beta":           round(beta, 3),
            "alpha":          round(alpha * self.TRADING_DAYS, 4),   # annualized alpha
            "ulcer_index":    round(ulcer_index, 4),
            "rolling_vol_1m": round(float(r.tail(21).std() * np.sqrt(self.TRADING_DAYS)), 4),
            "rolling_vol_3m": round(float(r.tail(63).std() * np.sqrt(self.TRADING_DAYS)), 4),
        }

    def portfolio_risk(self, prices_df: pd.DataFrame) -> dict:
        """Portfolio-level risk metrics."""
        returns_df = prices_df.pct_change().dropna()
        equal_w_returns = returns_df.mean(axis=1)

        ann_ret = float((1 + equal_w_returns.mean()) ** self.TRADING_DAYS - 1)
        ann_vol = float(equal_w_returns.std() * np.sqrt(self.TRADING_DAYS))
        ann_rf  = self.RISK_FREE_RATE * self.TRADING_DAYS

        corr_matrix = returns_df.corr()
        avg_corr    = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())

        max_dd = self._max_drawdown(prices_df.mean(axis=1))

        return {
            "portfolio_annual_return":  round(ann_ret * 100, 2),
            "portfolio_annual_vol":     round(ann_vol * 100, 2),
            "portfolio_sharpe":         round((ann_ret - ann_rf) / ann_vol, 3) if ann_vol > 0 else 0,
            "portfolio_max_drawdown":   round(max_dd * 100, 2),
            "average_correlation":      round(avg_corr, 3),
            "diversification_benefit":  "High" if avg_corr < 0.4 else "Moderate" if avg_corr < 0.7 else "Low",
        }

    # ─── VaR / CVaR ────────────────────────────────────────────────────────────

    def _var_historical(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical simulation VaR (daily)."""
        return float(np.percentile(returns, (1 - confidence) * 100))

    def _var_parametric(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Parametric (normal) VaR (daily)."""
        from scipy.stats import norm
        mu  = returns.mean()
        sig = returns.std()
        return float(norm.ppf(1 - confidence, mu, sig))

    def _cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) — average loss beyond VaR."""
        var = self._var_historical(returns, confidence)
        tail_losses = returns[returns <= var]
        return float(tail_losses.mean()) if len(tail_losses) > 0 else var

    # ─── DRAWDOWN ──────────────────────────────────────────────────────────────

    def _max_drawdown(self, prices: pd.Series) -> float:
        """Maximum peak-to-trough drawdown."""
        cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
        rolling_max = cumulative.cummax()
        drawdown    = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min())

    def _drawdown_duration(self, prices: pd.Series) -> int:
        """Length of longest drawdown period in days."""
        cumulative  = (1 + prices.pct_change().fillna(0)).cumprod()
        rolling_max = cumulative.cummax()
        in_drawdown = cumulative < rolling_max
        max_dur, cur_dur = 0, 0
        for flag in in_drawdown:
            if flag:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dur

    # ─── BETA / ALPHA ──────────────────────────────────────────────────────────

    def _beta_alpha(self, returns: pd.Series) -> tuple:
        """Estimate beta and alpha vs SPY using OLS."""
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY").history(period="5y", auto_adjust=True)["Close"].pct_change().dropna()
            aligned = pd.concat([returns, spy], axis=1, join="inner").dropna()
            aligned.columns = ["stock", "spy"]
            if len(aligned) < 30:
                return 1.0, 0.0
            cov = np.cov(aligned["stock"], aligned["spy"])
            beta  = cov[0, 1] / cov[1, 1]
            alpha = aligned["stock"].mean() - beta * aligned["spy"].mean()
            return float(beta), float(alpha)
        except Exception:
            return 1.0, 0.0

    # ─── ULCER INDEX ───────────────────────────────────────────────────────────

    def _ulcer_index(self, prices: pd.Series, period: int = 14) -> float:
        """Ulcer Index measures downside risk (depth and duration of drawdowns)."""
        rolling_max = prices.rolling(period).max()
        pct_drawdown = ((prices - rolling_max) / rolling_max) * 100
        return float(np.sqrt((pct_drawdown ** 2).mean()))

    # ─── RISK SCORE ────────────────────────────────────────────────────────────

    def _risk_score(self, sharpe: float, max_dd: float, vol: float, beta: float) -> float:
        """Score risk favorability from 0-100 (higher = better risk-adjusted profile)."""
        score = 50.0

        # Sharpe ratio
        if sharpe > 1.5:    score += 20
        elif sharpe > 1.0:  score += 14
        elif sharpe > 0.5:  score += 8
        elif sharpe < 0:    score -= 10

        # Max drawdown
        if max_dd > -0.15:  score += 15
        elif max_dd > -0.30: score += 8
        elif max_dd > -0.50: score += 0
        else:               score -= 10

        # Volatility
        if vol < 0.15:      score += 10
        elif vol < 0.25:    score += 5
        elif vol > 0.50:    score -= 10

        # Beta
        if 0.5 <= beta <= 1.2: score += 5
        elif beta > 2.0:       score -= 10

        return round(min(max(score, 0), 100), 2)
