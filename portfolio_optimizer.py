"""
portfolio_optimizer.py — Institutional portfolio construction.

Methods:
- Markowitz Mean-Variance Optimization
- Maximum Sharpe Ratio portfolio
- Minimum Volatility portfolio
- Risk Parity allocation
- Equal Weight benchmark
- Efficient Frontier generation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


class PortfolioOptimizer:

    RISK_FREE_RATE = 0.04   # Annualized
    TRADING_DAYS  = 252

    def __init__(self, prices: pd.DataFrame):
        """
        Args:
            prices: DataFrame with daily closing prices, columns = tickers
        """
        self.prices  = prices
        self.returns = prices.pct_change().dropna()
        self.tickers = list(prices.columns)
        self.n       = len(self.tickers)

        # Annualized stats
        self.mu    = self.returns.mean() * self.TRADING_DAYS
        self.cov   = self.returns.cov()  * self.TRADING_DAYS
        self.sigma = self.returns.std()  * np.sqrt(self.TRADING_DAYS)

    # ─── PUBLIC ────────────────────────────────────────────────────────────────

    def full_optimization(self) -> dict:
        """Run all optimization strategies and return results."""
        max_sharpe  = self.max_sharpe_portfolio()
        min_vol     = self.min_volatility_portfolio()
        risk_parity = self.risk_parity_portfolio()
        equal_wt    = self.equal_weight_portfolio()
        frontier    = self.efficient_frontier(n_points=30)

        # Correlation matrix
        corr = self.returns.corr().round(3).to_dict()

        return {
            "tickers": self.tickers,
            "optimal_weights":       max_sharpe["weights"],
            "expected_return":       max_sharpe["expected_return"],
            "expected_volatility":   max_sharpe["expected_volatility"],
            "sharpe_ratio":          max_sharpe["sharpe_ratio"],
            "max_sharpe":            max_sharpe,
            "min_volatility":        min_vol,
            "risk_parity":           risk_parity,
            "equal_weight":          equal_wt,
            "efficient_frontier":    frontier,
            "correlation_matrix":    corr,
            "individual_stats":      self._individual_stats(),
        }

    # ─── OPTIMIZATION STRATEGIES ──────────────────────────────────────────────

    def max_sharpe_portfolio(self) -> dict:
        """Maximize Sharpe ratio portfolio."""
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0.02, 0.60) for _ in range(self.n))
        x0 = np.array([1 / self.n] * self.n)

        result = minimize(
            lambda w: -self._sharpe(w),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        w = result.x if result.success else x0
        w = self._clean_weights(w)
        return self._portfolio_stats(w, label="Max Sharpe")

    def min_volatility_portfolio(self) -> dict:
        """Minimize portfolio volatility."""
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0.02, 0.60) for _ in range(self.n))
        x0 = np.array([1 / self.n] * self.n)

        result = minimize(
            lambda w: self._portfolio_vol(w),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        w = result.x if result.success else x0
        w = self._clean_weights(w)
        return self._portfolio_stats(w, label="Min Volatility")

    def risk_parity_portfolio(self) -> dict:
        """
        Risk Parity: each asset contributes equally to total portfolio risk.
        """
        n = self.n
        target_risk = np.array([1 / n] * n)

        def risk_parity_obj(w):
            w = np.array(w)
            port_var = w @ self.cov.values @ w
            marginal_contrib = self.cov.values @ w
            risk_contrib = w * marginal_contrib / port_var
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0.01, 0.60) for _ in range(n))
        x0 = np.array([1 / n] * n)

        result = minimize(
            risk_parity_obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )

        w = result.x if result.success else x0
        w = self._clean_weights(w)
        return self._portfolio_stats(w, label="Risk Parity")

    def equal_weight_portfolio(self) -> dict:
        """Naive equal-weight benchmark."""
        w = self._clean_weights(np.array([1 / self.n] * self.n))
        return self._portfolio_stats(w, label="Equal Weight")

    def efficient_frontier(self, n_points: int = 30) -> list:
        """Generate efficient frontier data points."""
        # Get return range from min-vol to max-return
        min_ret = float(self.mu.min())
        max_ret = float(self.mu.max()) * 0.99
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: self._portfolio_return(w) - t},
            ]
            bounds = tuple((0.0, 1.0) for _ in range(self.n))
            x0 = np.array([1 / self.n] * self.n)

            result = minimize(
                lambda w: self._portfolio_vol(w),
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500},
            )

            if result.success:
                w = result.x
                frontier.append({
                    "return": round(float(self._portfolio_return(w)), 4),
                    "volatility": round(float(self._portfolio_vol(w)), 4),
                    "sharpe": round(float(self._sharpe(w)), 4),
                })

        return frontier

    # ─── HELPERS ───────────────────────────────────────────────────────────────

    def _portfolio_return(self, w) -> float:
        return float(np.dot(w, self.mu.values))

    def _portfolio_vol(self, w) -> float:
        return float(np.sqrt(w @ self.cov.values @ w))

    def _sharpe(self, w) -> float:
        ret = self._portfolio_return(w)
        vol = self._portfolio_vol(w)
        if vol == 0:
            return 0
        return (ret - self.RISK_FREE_RATE) / vol

    def _portfolio_stats(self, w: np.ndarray, label: str) -> dict:
        weights_dict = {ticker: round(float(w[i]), 4) for i, ticker in enumerate(self.tickers)}
        exp_return   = round(self._portfolio_return(w), 4)
        exp_vol      = round(self._portfolio_vol(w), 4)
        sharpe       = round(self._sharpe(w), 4)
        return {
            "label":               label,
            "weights":             weights_dict,
            "expected_return":     exp_return,
            "expected_volatility": exp_vol,
            "sharpe_ratio":        sharpe,
            "expected_return_pct": round(exp_return * 100, 2),
            "expected_vol_pct":    round(exp_vol * 100, 2),
        }

    def _clean_weights(self, w: np.ndarray, threshold: float = 0.005) -> np.ndarray:
        """Zero out tiny weights and renormalize."""
        w = np.array(w, dtype=float)
        w[w < threshold] = 0
        total = w.sum()
        return w / total if total > 0 else np.array([1/len(w)] * len(w))

    def _individual_stats(self) -> dict:
        stats = {}
        for ticker in self.tickers:
            ret = float(self.mu[ticker])
            vol = float(self.sigma[ticker])
            sharpe = (ret - self.RISK_FREE_RATE) / vol if vol > 0 else 0
            stats[ticker] = {
                "annual_return_pct": round(ret * 100, 2),
                "annual_vol_pct":    round(vol * 100, 2),
                "sharpe_ratio":      round(sharpe, 3),
            }
        return stats
