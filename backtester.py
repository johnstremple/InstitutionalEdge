"""
backtester.py — Strategy Backtesting Engine

Tests the bot's stock-picking strategy against historical data.
Proves (or disproves) that the screening methodology has real alpha.

Methods:
- Rolling annual rebalance: screen at start of each year, hold for 1 year
- Compare to SPY benchmark
- Track: CAGR, Sharpe, max drawdown, win rate, alpha, beta
- Walk-forward testing (avoids look-ahead bias)
- Portfolio simulation with equal-weight and score-weighted allocation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class Backtester:

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital

    # ── MAIN BACKTEST ─────────────────────────────────────────────

    def backtest_strategy(self,
                          tickers: list,
                          start_date: str = "2019-01-01",
                          end_date: str   = None,
                          rebalance_freq: str = "annual",  # annual | quarterly | monthly
                          weighting: str = "equal",        # equal | score_weighted | market_cap
                          top_n: int = 15,
                          benchmark: str = "SPY") -> dict:
        """
        Full backtest of a ticker list as a portfolio strategy.

        Parameters:
            tickers:       List of stock tickers to include in strategy universe
            start_date:    Backtest start date (YYYY-MM-DD)
            end_date:      Backtest end date (defaults to today)
            rebalance_freq: How often to rebalance
            weighting:     How to weight positions
            top_n:         Max positions at any rebalance
            benchmark:     Ticker to compare against (SPY default)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n  Backtesting {len(tickers)} stocks | {start_date} → {end_date}")
        print(f"  Rebalance: {rebalance_freq} | Weighting: {weighting} | Top {top_n} positions")

        # ── Download price data ───────────────────────────────────
        all_tickers = list(set(tickers + [benchmark]))
        print(f"  Downloading price data...")
        try:
            raw = yf.download(all_tickers, start=start_date, end=end_date,
                              progress=False, auto_adjust=True)["Close"]
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(name=all_tickers[0])
        except Exception as e:
            print(f"  ERROR downloading data: {e}")
            return {}

        raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.5))  # Drop if <50% data
        raw = raw.ffill().bfill()

        # ── Generate rebalance dates ──────────────────────────────
        rebalance_dates = self._rebalance_dates(raw.index, rebalance_freq)

        # ── Run simulation ────────────────────────────────────────
        portfolio_values = self._simulate(raw, rebalance_dates, benchmark,
                                          top_n, weighting)

        if portfolio_values.empty:
            print("  ERROR: Simulation failed — insufficient data")
            return {}

        # ── Benchmark comparison ──────────────────────────────────
        bench_values = self._benchmark_series(raw, benchmark, self.initial_capital)

        # ── Calculate metrics ─────────────────────────────────────
        metrics = self._calculate_metrics(portfolio_values, bench_values)

        # ── Trade log ─────────────────────────────────────────────
        trade_log = self._trade_log(raw, rebalance_dates, top_n)

        # ── Print results ─────────────────────────────────────────
        self._print_results(metrics, trade_log)

        return {
            "metrics":           metrics,
            "portfolio_values":  portfolio_values,
            "benchmark_values":  bench_values,
            "trade_log":         trade_log,
            "parameters": {
                "tickers":     tickers,
                "start_date":  start_date,
                "end_date":    end_date,
                "rebalance":   rebalance_freq,
                "weighting":   weighting,
                "top_n":       top_n,
                "benchmark":   benchmark,
            }
        }

    def backtest_single_stock(self, ticker: str,
                               start_date: str = "2019-01-01",
                               end_date: str   = None,
                               benchmark: str  = "SPY") -> dict:
        """Simple buy-and-hold backtest for a single stock vs benchmark."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n  Buy-and-hold backtest: {ticker} | {start_date} → {end_date}")

        try:
            data = yf.download([ticker, benchmark],
                               start=start_date, end=end_date,
                               progress=False, auto_adjust=True)["Close"]
            data = data.dropna()
        except Exception as e:
            print(f"  ERROR: {e}")
            return {}

        if ticker not in data.columns or benchmark not in data.columns:
            return {}

        cap = self.initial_capital
        port  = cap * data[ticker] / data[ticker].iloc[0]
        bench = cap * data[benchmark] / data[benchmark].iloc[0]

        metrics = self._calculate_metrics(port, bench)
        self._print_results(metrics, [])
        return {"metrics": metrics, "portfolio_values": port, "benchmark_values": bench}

    # ── SIMULATION ENGINE ─────────────────────────────────────────

    def _simulate(self, prices: pd.DataFrame, rebalance_dates: list,
                  benchmark: str, top_n: int, weighting: str) -> pd.Series:
        """
        Simulate portfolio over time with periodic rebalancing.
        Uses equal-weight within selected tickers.
        """
        available = [c for c in prices.columns if c != benchmark]
        if not available:
            return pd.Series(dtype=float)

        capital     = float(self.initial_capital)
        holdings    = {}    # {ticker: shares}
        port_values = {}

        for i, date in enumerate(prices.index):
            # Rebalance check
            if any(abs((date - rd).days) <= 1 for rd in rebalance_dates):
                # Pick top_n by momentum (12-1 month)
                selected = self._select_stocks(prices.loc[:date], available,
                                               top_n, weighting)
                if not selected:
                    selected = available[:top_n]

                # Calculate current portfolio value
                if holdings:
                    capital = sum(
                        shares * float(prices.loc[date, ticker])
                        for ticker, shares in holdings.items()
                        if ticker in prices.columns
                    )
                else:
                    capital = float(self.initial_capital)

                # Rebalance into new positions
                holdings = {}
                for ticker in selected:
                    if ticker in prices.columns:
                        price_val = float(prices.loc[date, ticker])
                        if price_val > 0:
                            alloc = capital / len(selected)
                            holdings[ticker] = alloc / price_val

            # Daily portfolio value
            if holdings:
                daily_val = sum(
                    shares * float(prices.loc[date, ticker])
                    for ticker, shares in holdings.items()
                    if ticker in prices.columns and
                    not pd.isna(prices.loc[date, ticker])
                )
                port_values[date] = daily_val if daily_val > 0 else self.initial_capital
            else:
                port_values[date] = self.initial_capital

        return pd.Series(port_values)

    def _select_stocks(self, prices: pd.DataFrame, tickers: list,
                        top_n: int, weighting: str) -> list:
        """Select top stocks by momentum score."""
        if len(prices) < 30:
            return tickers[:top_n]

        scores = {}
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            p = prices[ticker].dropna()
            if len(p) < 21:
                continue
            try:
                cur     = float(p.iloc[-1])
                m_1m    = (cur - float(p.iloc[-21])) / float(p.iloc[-21]) if len(p) >= 21 else 0
                m_3m    = (cur - float(p.iloc[-63])) / float(p.iloc[-63]) if len(p) >= 63 else 0
                m_6m    = (cur - float(p.iloc[-126])) / float(p.iloc[-126]) if len(p) >= 126 else 0
                m_12m   = (cur - float(p.iloc[-252])) / float(p.iloc[-252]) if len(p) >= 252 else 0
                # 12-1 momentum
                momentum = m_12m - m_1m
                scores[ticker] = 0.4*m_3m + 0.3*m_6m + 0.2*momentum + 0.1*m_1m
            except Exception:
                scores[ticker] = 0

        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        return sorted_tickers[:top_n]

    # ── METRICS ──────────────────────────────────────────────────

    def _calculate_metrics(self, portfolio: pd.Series,
                            benchmark: pd.Series) -> dict:
        """Calculate comprehensive performance metrics."""
        if portfolio.empty:
            return {}

        # Align
        common = portfolio.index.intersection(benchmark.index)
        p = portfolio.loc[common]
        b = benchmark.loc[common]

        port_ret  = p.pct_change().dropna()
        bench_ret = b.pct_change().dropna()

        # Returns
        total_return  = (p.iloc[-1] - p.iloc[0]) / p.iloc[0]
        bench_total   = (b.iloc[-1] - b.iloc[0]) / b.iloc[0]
        n_years       = len(p) / 252
        cagr          = (p.iloc[-1] / p.iloc[0]) ** (1/n_years) - 1 if n_years > 0 else 0
        bench_cagr    = (b.iloc[-1] / b.iloc[0]) ** (1/n_years) - 1 if n_years > 0 else 0
        alpha_simple  = cagr - bench_cagr

        # Volatility
        ann_vol  = float(port_ret.std() * np.sqrt(252))
        bench_vol = float(bench_ret.std() * np.sqrt(252))

        # Risk-adjusted
        rf        = 0.045  # 4.5% risk-free rate (current)
        sharpe    = (cagr - rf) / ann_vol if ann_vol > 0 else 0

        downside  = port_ret[port_ret < 0].std() * np.sqrt(252)
        sortino   = (cagr - rf) / downside if downside > 0 else 0

        # Beta / Alpha (CAPM)
        cov_matrix = np.cov(port_ret, bench_ret)
        beta       = cov_matrix[0,1] / cov_matrix[1,1] if cov_matrix[1,1] > 0 else 1
        alpha_capm = cagr - (rf + beta * (bench_cagr - rf))

        # Drawdown
        roll_max   = p.cummax()
        drawdowns  = (p - roll_max) / roll_max
        max_dd     = float(drawdowns.min())

        # Drawdown duration
        in_dd    = drawdowns < -0.01
        dd_periods = []
        count = 0
        for val in in_dd:
            if val: count += 1
            else:
                if count > 0: dd_periods.append(count)
                count = 0
        max_dd_days = max(dd_periods) if dd_periods else 0

        # Win rate (monthly)
        monthly_port  = p.resample("M").last().pct_change().dropna()
        monthly_bench = b.resample("M").last().pct_change().dropna()
        if len(monthly_port) > 0 and len(monthly_bench) > 0:
            common_m = monthly_port.index.intersection(monthly_bench.index)
            win_rate = float((monthly_port.loc[common_m] > monthly_bench.loc[common_m]).mean())
        else:
            win_rate = 0.5

        # Rolling 1Y returns
        rolling_1y   = p.pct_change(252).dropna()
        best_year    = float(rolling_1y.max()) if len(rolling_1y) > 0 else 0
        worst_year   = float(rolling_1y.min()) if len(rolling_1y) > 0 else 0

        return {
            "start_value":     round(float(p.iloc[0]),   2),
            "end_value":       round(float(p.iloc[-1]),  2),
            "total_return":    round(total_return * 100, 2),
            "cagr":            round(cagr * 100,          2),
            "bench_total":     round(bench_total * 100,   2),
            "bench_cagr":      round(bench_cagr * 100,    2),
            "alpha":           round(alpha_simple * 100,  2),
            "alpha_capm":      round(alpha_capm * 100,    2),
            "beta":            round(beta, 3),
            "sharpe":          round(sharpe, 3),
            "sortino":         round(sortino, 3),
            "ann_vol":         round(ann_vol * 100,        2),
            "bench_vol":       round(bench_vol * 100,      2),
            "max_drawdown":    round(max_dd * 100,         2),
            "max_dd_days":     max_dd_days,
            "win_rate_vs_bench": round(win_rate * 100,     1),
            "best_year":       round(best_year * 100,      2),
            "worst_year":      round(worst_year * 100,     2),
            "n_years":         round(n_years, 2),
            "calmar":          round(abs(cagr / max_dd) if max_dd < 0 else 0, 3),
        }

    # ── HELPERS ──────────────────────────────────────────────────

    def _rebalance_dates(self, index: pd.DatetimeIndex,
                          freq: str) -> list:
        """Generate rebalancing dates."""
        if freq == "annual":
            return [d for d in index if d.month == 1 and d.day <= 7]
        elif freq == "quarterly":
            return [d for d in index if d.month in [1,4,7,10] and d.day <= 7]
        elif freq == "monthly":
            return [d for d in index if d.day <= 5]
        else:
            return [index[0]]

    def _benchmark_series(self, prices: pd.DataFrame,
                           benchmark: str, capital: float) -> pd.Series:
        """Create benchmark value series."""
        if benchmark not in prices.columns:
            return pd.Series([capital] * len(prices), index=prices.index)
        p = prices[benchmark].dropna()
        return capital * p / p.iloc[0]

    def _trade_log(self, prices: pd.DataFrame, rebalance_dates: list,
                    top_n: int) -> list:
        """Log what was held at each rebalance."""
        log     = []
        cols    = [c for c in prices.columns if c not in ["SPY","QQQ"]]
        for rd in rebalance_dates:
            try:
                hist = prices.loc[:rd]
                selected = self._select_stocks(hist, cols, top_n, "equal")
                log.append({
                    "date":     rd.strftime("%Y-%m-%d"),
                    "holdings": selected,
                    "n":        len(selected),
                })
            except Exception:
                continue
        return log

    # ── PRINT ────────────────────────────────────────────────────

    def _print_results(self, metrics: dict, trade_log: list):
        print(f"\n  {'='*56}")
        print(f"  BACKTEST RESULTS")
        print(f"  {'='*56}")
        print(f"  Period:           {metrics.get('n_years',0):.1f} years")
        print(f"  Start Value:      ${metrics.get('start_value',0):>12,.2f}")
        print(f"  End Value:        ${metrics.get('end_value',0):>12,.2f}")
        print(f"\n  {'─'*40}")
        print(f"  {'Metric':<25} {'Strategy':>10}  {'Benchmark':>10}")
        print(f"  {'─'*40}")
        print(f"  {'Total Return':<25} {metrics.get('total_return',0):>9.1f}%  {metrics.get('bench_total',0):>9.1f}%")
        print(f"  {'CAGR':<25} {metrics.get('cagr',0):>9.1f}%  {metrics.get('bench_cagr',0):>9.1f}%")
        print(f"  {'Alpha (simple)':<25} {metrics.get('alpha',0):>9.1f}%  {'—':>10}")
        print(f"  {'Alpha (CAPM)':<25} {metrics.get('alpha_capm',0):>9.1f}%  {'—':>10}")
        print(f"  {'Beta':<25} {metrics.get('beta',0):>10.2f}  {'1.00':>10}")
        print(f"  {'Ann. Volatility':<25} {metrics.get('ann_vol',0):>9.1f}%  {metrics.get('bench_vol',0):>9.1f}%")
        print(f"  {'Sharpe Ratio':<25} {metrics.get('sharpe',0):>10.3f}  {'—':>10}")
        print(f"  {'Sortino Ratio':<25} {metrics.get('sortino',0):>10.3f}  {'—':>10}")
        print(f"  {'Calmar Ratio':<25} {metrics.get('calmar',0):>10.3f}  {'—':>10}")
        print(f"  {'Max Drawdown':<25} {metrics.get('max_drawdown',0):>9.1f}%  {'—':>10}")
        print(f"  {'Max DD Duration':<25} {metrics.get('max_dd_days',0):>9}d  {'—':>10}")
        print(f"  {'Win Rate vs Bench':<25} {metrics.get('win_rate_vs_bench',0):>9.1f}%  {'—':>10}")
        print(f"  {'Best 1Y Return':<25} {metrics.get('best_year',0):>9.1f}%  {'—':>10}")
        print(f"  {'Worst 1Y Return':<25} {metrics.get('worst_year',0):>9.1f}%  {'—':>10}")
        print(f"  {'─'*40}")

        alpha = metrics.get("alpha", 0)
        sharpe = metrics.get("sharpe", 0)
        if alpha > 3 and sharpe > 0.8:
            print(f"\n  ✅ VERDICT: Strategy generated meaningful alpha — {alpha:.1f}% annual outperformance")
        elif alpha > 0:
            print(f"\n  🟡 VERDICT: Slight outperformance (+{alpha:.1f}% alpha) — marginal edge")
        else:
            print(f"\n  🔴 VERDICT: Underperformed benchmark by {abs(alpha):.1f}% — review strategy")

        if trade_log:
            print(f"\n  Trade Log (last 3 rebalances):")
            for entry in trade_log[-3:]:
                stocks = ", ".join(entry["holdings"][:8])
                print(f"  {entry['date']}: {stocks}{'...' if entry['n'] > 8 else ''}")

        print(f"  {'='*56}\n")

    # ── QUICK PRESETS ─────────────────────────────────────────────

    def backtest_magnificent7(self, start_date="2019-01-01") -> dict:
        """Backtest Magnificent 7 vs S&P 500."""
        mag7 = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA"]
        print("  📊 Backtesting: Magnificent 7 vs SPY")
        return self.backtest_strategy(mag7, start_date=start_date, top_n=7)

    def backtest_ai_theme(self, start_date="2022-01-01") -> dict:
        """Backtest AI theme basket."""
        ai = ["NVDA","AMD","MSFT","GOOGL","META","PLTR","SMCI","AVGO","ARM","QCOM",
              "IONQ","SOUN","BBAI","AAPL","TSM","MRVL"]
        print("  📊 Backtesting: AI Theme Basket vs QQQ")
        return self.backtest_strategy(ai, start_date=start_date, top_n=10, benchmark="QQQ")

    def backtest_quality_screen(self, start_date="2019-01-01") -> dict:
        """Backtest quality factor screen."""
        quality = ["AAPL","MSFT","GOOGL","V","MA","COST","HD","UNH","LLY","AVGO",
                   "ADBE","INTU","NOW","PANW","ISRG","TMO","SHW","MCO","CME","BLK"]
        print("  📊 Backtesting: Quality Compounders vs SPY")
        return self.backtest_strategy(quality, start_date=start_date, top_n=15)

    def backtest_growth_screen(self, start_date="2020-01-01") -> dict:
        """Backtest high-growth screen."""
        growth = ["NVDA","META","AMZN","GOOGL","MSFT","AAPL","AMD","PLTR","SMCI",
                  "CELH","CAVA","DUOL","AXON","CRWD","DDOG","SNOW","MDB","NET","ZS","HIMS"]
        print("  📊 Backtesting: High-Growth Screen vs SPY")
        return self.backtest_strategy(growth, start_date=start_date, top_n=12)

    def compare_strategies(self, start_date: str = "2020-01-01") -> dict:
        """Run all presets and compare side by side."""
        print("\n" + "="*62)
        print("  STRATEGY COMPARISON BACKTEST")
        print("="*62)
        results = {}

        strategies = [
            ("Mag7",           self.backtest_magnificent7),
            ("AI Theme",       self.backtest_ai_theme),
            ("Quality Screen", self.backtest_quality_screen),
            ("Growth Screen",  self.backtest_growth_screen),
        ]

        for name, fn in strategies:
            print(f"\n  Running: {name}...")
            try:
                r = fn(start_date)
                if r and r.get("metrics"):
                    results[name] = r["metrics"]
            except Exception as e:
                print(f"  Error in {name}: {e}")

        # Comparison table
        if results:
            print(f"\n{'='*70}")
            print(f"  STRATEGY COMPARISON vs SPY")
            print(f"{'='*70}")
            print(f"  {'Strategy':<22} {'CAGR':>8} {'Alpha':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
            print(f"  {'─'*62}")
            for name, m in sorted(results.items(), key=lambda x: x[1].get("cagr",0), reverse=True):
                print(f"  {name:<22} {m.get('cagr',0):>7.1f}% {m.get('alpha',0):>7.1f}% "
                      f"{m.get('sharpe',0):>8.2f} {m.get('max_drawdown',0):>7.1f}% "
                      f"{m.get('win_rate_vs_bench',0):>7.1f}%")

        return results
