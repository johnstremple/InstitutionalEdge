# 🏦 InstitutionalEdge

**An institutional-grade, open-source investment analysis system for long-term wealth building.**

Built for retail investors who want access to the same analytical frameworks used by professional asset managers — completely free, no paid data sources required.

---

## What It Does

InstitutionalEdge runs a full suite of quantitative models on any US stock or cryptocurrency to answer two questions:

1. **Is this a good company to own long-term?**
2. **Is now a good time to buy?**

It then gives you a composite score (0–100) and a clear signal: `STRONG BUY / BUY / HOLD / WEAK / AVOID`.

---

## Analysis Modules

### 1. Fundamental Analysis
- P/E, P/B, EV/EBITDA, P/FCF, PEG ratios
- Revenue, earnings, and FCF growth (YoY + CAGR)
- ROE, ROIC, ROA, gross/net/operating margins
- Debt/Equity, Current Ratio, Interest Coverage
- Piotroski F-Score (9-point quality assessment)
- Moat indicators

### 2. Technical Analysis
- Trend: SMA/EMA (20/50/200), ADX
- Momentum: RSI, Stochastic, Williams %R
- MACD with crossover detection
- Bollinger Bands (squeeze / expansion)
- ATR (volatility measurement)
- OBV (On-Balance Volume — accumulation/distribution)
- Golden Cross / Death Cross detection
- Support & Resistance levels

### 3. News Sentiment Analysis
- Pulls headlines from yfinance + Google News RSS (no API key)
- VADER NLP sentiment scoring on each headline
- Bullish/bearish/neutral headline counts
- Overall market narrative summary

### 4. Institutional Simulations
- **Monte Carlo** (10,000 GBM paths) — 1-year price distribution, bull/bear/median scenarios, probability of profit
- **Discounted Cash Flow (DCF)** — 2-stage growth model with WACC, intrinsic value, margin of safety entry points
- **Scenario Analysis** — Bull / Base / Bear with probability weights and 1/3/5-year price targets
- **Historical Stress Tests** — Estimate drawdowns under 2008 crash, COVID, 2022 bear market, dot-com bust
- **CAGR Projections** — Multi-rate wealth projection table (7% to 25%)

### 5. Portfolio Optimization
- **Maximum Sharpe Ratio** portfolio (Modern Portfolio Theory)
- **Minimum Volatility** portfolio
- **Risk Parity** allocation
- **Efficient Frontier** generation (30-point curve)
- Individual asset stats: return, vol, Sharpe
- Correlation matrix + diversification analysis

### 6. Risk Management
- Value at Risk (VaR): Historical simulation at 95% and 99%
- Conditional VaR / Expected Shortfall (CVaR)
- Sharpe, Sortino, and Calmar ratios
- Max Drawdown + drawdown duration
- Rolling volatility (1-month, 3-month)
- Beta and Alpha vs SPY
- Ulcer Index

### 7. Crypto Analysis
- Market data via CoinGecko (free, no API key)
- Full technical analysis on price history
- Developer activity (GitHub commits, stars, forks)
- Community metrics (Twitter, Reddit)
- Market dominance, supply inflation, ATH analysis
- Monte Carlo simulation

---

## Installation

```bash
git clone https://github.com/yourusername/InstitutionalEdge.git
cd InstitutionalEdge
pip install -r requirements.txt
```

**Requirements:** Python 3.9+

---

## Usage

### Analyze a Single Stock
```bash
python main.py analyze AAPL
python main.py analyze NVDA
python main.py analyze MSFT
```

### Optimize a Portfolio
```bash
python main.py portfolio AAPL MSFT NVDA AMZN GOOGL META
```

### Rank a Watchlist
```bash
python main.py watchlist AAPL NVDA MSFT TSLA META AMZN GOOGL
```

### Analyze Crypto
```bash
python main.py crypto bitcoin
python main.py crypto ethereum
python main.py crypto solana
```

---

## Example Output (Analyze)

```
======================================================
  INSTITUTIONAL EDGE ANALYSIS: AAPL
  2025-02-19 14:30:00
======================================================

  --- SCORES (0-100) ---
  Fundamental:  74.5/100
  Technical:    68.2/100
  Sentiment:    72.1/100
  Risk-Adj:     81.3/100
  COMPOSITE:    73.8/100

  --- SIMULATIONS (1-Year Outlook) ---
  Monte Carlo Median Return:  +14.2%
  Bull Case (90th pct):       +42.1%
  Bear Case (10th pct):       -18.3%
  DCF Intrinsic Value:        $218.40
  Current Price:              $195.20
  DCF Upside/Downside:        +11.9% (Undervalued)

  ═══════════════════════════════
  SIGNAL: 🟢 BUY
  ═══════════════════════════════
```

---

## Composite Scoring Methodology

| Component | Weight | What It Measures |
|---|---|---|
| Fundamental | 40% | Business quality, valuation, growth |
| Technical | 30% | Price trend, momentum, timing |
| Sentiment | 20% | News flow, market narrative |
| Risk-Adjusted | 10% | Sharpe, drawdown, volatility |

**Signal Thresholds:**
- `STRONG BUY`: Score ≥ 72 + MC median return > 10%
- `BUY`: Score ≥ 60 + MC median return > 5%
- `HOLD`: Score 45–60
- `WEAK`: Score 35–45
- `AVOID`: Score < 35

---

## Data Sources (All Free)

| Source | Data |
|---|---|
| yfinance | Stock prices, financials, company info |
| Google News RSS | News headlines |
| CoinGecko API | Crypto market data |
| VADER Sentiment | NLP sentiment scoring |

---

## Programmatic Usage

```python
from main import analyze_stock, analyze_portfolio, generate_watchlist

# Single stock
results = analyze_stock("NVDA")
print(results["buy_signal"])          # "STRONG BUY"
print(results["composite_score"])     # 81.3
print(results["simulations"]["dcf"]["intrinsic_value"])

# Portfolio optimization
portfolio = analyze_portfolio(["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
print(portfolio["optimal_weights"])   # {"AAPL": 0.18, "MSFT": 0.24, ...}

# Watchlist ranking
ranked = generate_watchlist(["AAPL", "NVDA", "TSLA", "META", "AMZN"])
# Returns sorted by composite score
```

---

## Architecture

```
InstitutionalEdge/
├── main.py                  # CLI entry point
├── data_fetcher.py          # Unified data retrieval
├── fundamental_analysis.py  # Valuation, growth, health, quality
├── technical_analysis.py    # 15+ indicators, signal generation
├── sentiment_analysis.py    # NLP news sentiment
├── simulations.py           # Monte Carlo, DCF, scenarios, stress tests
├── portfolio_optimizer.py   # MPT, efficient frontier, risk parity
├── risk_manager.py          # VaR, CVaR, Sharpe, beta, drawdown
├── crypto_analyzer.py       # Crypto-specific analysis
├── report_generator.py      # Text + JSON report export
└── requirements.txt
```

---

## Disclaimer

**InstitutionalEdge is for educational purposes only. It is not financial advice.**
Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions. The models used are simplified representations of complex financial systems.

---

## Contributing

Pull requests welcome. Priority improvements:
- [ ] SEC EDGAR filing integration (10-K/10-Q parsing)
- [ ] Insider trading signal detection
- [ ] Options market implied volatility integration
- [ ] Sector rotation model
- [ ] Earnings date awareness and pre-earnings analysis
- [ ] Streamlit web dashboard

---

*Built with Python, yfinance, scipy, numpy, VADER NLP, and CoinGecko.*
