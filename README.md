# InstitutionalEdge v5.0

> **Institutional-grade investment research platform built in Python — the analytical infrastructure of a hedge fund, using 100% free data.**

Built by [John Stremple](https://github.com/johnstremple) | Finance + CS @ George Mason University | Founder, Stremple Trading LLC

---

## What This Is

Retail investors get charts and P/E ratios. Institutional desks get fraud detection, factor models, alternative data, and systematic portfolio construction. This platform closes that gap.

InstitutionalEdge runs a full research pipeline — from universe screening to PDF-quality output — in a single CLI command. Built to support real portfolio management decisions for Stremple Trading LLC clients, not as a classroom exercise.

---

## Core Capabilities

### Fraud & Risk Detection
- **Beneish M-Score** — 8-variable earnings manipulation model (the same framework that flagged Enron pre-collapse). Score > -1.78 = likely manipulator
- **Altman Z-Score** — Bankruptcy probability scoring. Score < 1.81 = financial distress
- **Earnings Quality Engine** — Accruals analysis, cash flow vs. reported earnings divergence

### Quantitative Factor Model
Pulls factor data directly from Kenneth French's Dartmouth library and runs OLS regression to decompose returns into MKT, SMB, HML, RMW, CMA, and MOM. Outputs annualized alpha, factor betas, R², and style classification.

### 11-Module Stock Analysis Engine

| Module | What It Measures |
|---|---|
| Fundamental Analysis | P/E, P/B, EV/EBITDA, margins, Piotroski F-Score |
| Technical Analysis | 15+ indicators — RSI, MACD, Bollinger Bands, MAs |
| Competitive Analysis | Economic moat scoring, peer benchmarking |
| Monte Carlo Simulation | 10,000 price paths across bull/base/bear scenarios |
| DCF Valuation | Discounted cash flow with margin of safety |
| Insider Tracking | SEC Form 4 filings, cluster buying detection |
| Short Interest | Float %, days to cover, squeeze probability |
| Options Signals | P/C ratio, IV rank, max pain, unusual flow |
| Macro Regime | Economic cycle detection, sector rotation signals |
| Fraud Detection | Beneish M-Score + Altman Z-Score |
| Fama-French Factor Model | Alpha, factor betas, R², style classification |

### Automated Stock Screener
Screens the full S&P 500 + NASDAQ-100 across 7 systematic strategies — no ticker input required.

| Strategy | Logic |
|---|---|
| `aggressive_growth` | High revenue growth, expanding margins |
| `garp` | Growth at a Reasonable Price (PEG-based) |
| `deep_value` | Low P/E, P/B — trading below intrinsic value |
| `quality_compounder` | High ROE, durable competitive advantage |
| `dividend_growth` | Dividend yield + growth potential |
| `momentum` | 12-1 month price momentum factor |
| `small_cap_growth` | Under-followed small caps with high growth |

### AI Portfolio Advisor
Survey-based portfolio builder — 15 questions covering risk tolerance, time horizon, goals, and capital — outputs a personalized allocation with exact dollar amounts, 30-year projections, and tax optimization notes.

### Backtesting Engine
Walk-forward backtest with annual/quarterly rebalancing. Outputs CAGR vs benchmark, CAPM alpha, Sharpe/Sortino/Calmar ratios, max drawdown, and monthly win rate vs S&P 500.

### Natural Language Screener
```bash
python main.py ask "cheap AI stocks with insider buying"
python main.py ask "profitable dividend stocks no debt"
python main.py ask "small cap biotech high growth under PE 20"
```

---

## Sample Output
```
NVDA — NVIDIA Corporation | $4,621B Mkt Cap | Price: $189.82

12M TARGET:  $226.50  (+19.3% upside)

SCORES:
  Fund 79 | Comp 75 | Tech 62 | Sent 53 | Insider 56
  Fraud 45 | Factor 91 | Alt 80 | Options 76
  COMPOSITE: 76.6/100

ACCOUNTING QUALITY
  Beneish M-Score:  -1.02  (high growth false positive)
  Altman Z-Score:   91.10  SAFE ZONE
  Earnings Quality: 75/100 HIGH QUALITY

FACTOR MODEL: Quality Growth | Alpha +8.2%/yr | R² = 0.71

SIGNAL: STRONG BUY
```

---

## Installation
```bash
git clone https://github.com/johnstremple/InstitutionalEdge.git
cd InstitutionalEdge
pip install yfinance pandas numpy scipy requests feedparser vaderSentiment matplotlib streamlit reportlab pytrends
```

## Usage
```bash
# Full analysis with institutional PDF report
python main.py analyze NVDA --pdf

# AI advisor — builds personalized portfolio from survey
python main.py advisor

# Screen for stocks automatically
python main.py screen --profile garp
python main.py screen --profile aggressive_growth

# Backtest vs S&P 500
python main.py backtest --mode compare
python main.py backtest --mode custom --tickers NVDA AMD MSFT AAPL

# Options flow scanner
python main.py options-scan

# Portfolio risk attribution
python main.py risk-attribution AAPL:30 MSFT:25 NVDA:45

# Macro regime + sector rotation
python main.py macro

# Web dashboard
streamlit run dashboard.py
```

---

## Data Sources (all free)

| Source | Data |
|---|---|
| yfinance | Prices, financials, options chains |
| SEC EDGAR API | Form 4 insider filings, 13F holdings |
| Kenneth French Data Library | Fama-French 5-Factor + Momentum |
| Reddit JSON API | WallStreetBets sentiment |
| Wikipedia | S&P 500 and NASDAQ-100 constituents |
| Google Trends (pytrends) | Retail search interest |

**Total data cost: $0/month**

---

## Architecture
```
InstitutionalEdge/
├── main.py                  # CLI — all commands route here
├── data_fetcher.py          # Market data aggregation
├── fundamental_analysis.py  # Ratios, Piotroski F-Score
├── technical_analysis.py    # 15+ technical indicators
├── competitive_analysis.py  # Moat scoring, peer comparison
├── simulations.py           # Monte Carlo, DCF
├── risk_manager.py          # Sharpe, VaR, drawdown
├── sentiment_analysis.py    # News + Reddit sentiment (VADER)
├── insider_tracker.py       # SEC Form 4, 13F parsing
├── short_interest.py        # Squeeze probability model
├── options_signals.py       # P/C ratio, IV rank, max pain
├── macro_regime.py          # Economic cycle, sector rotation
├── price_target.py          # 5-model blended price target
├── fraud_detection.py       # Beneish M-Score, Altman Z-Score
├── factor_model.py          # Fama-French 5-Factor OLS regression
├── alt_data.py              # Reddit sentiment, earnings calendar
├── advanced_tools.py        # Options scanner, risk attribution, NL screener
├── stock_screener.py        # Universe screener (500+ stocks)
├── advisor.py               # Survey-based portfolio builder
├── backtester.py            # Walk-forward backtesting engine
├── portfolio_optimizer.py   # Mean-variance optimization (MPT)
├── crypto_analyzer.py       # Cryptocurrency analysis
├── pdf_report.py            # Institutional-style PDF report generator
├── dashboard.py             # Streamlit web dashboard
└── report_generator.py      # Text report output
```

---

## About

Built to support real portfolio management decisions for Stremple Trading LLC clients and to bridge the gap between institutional research infrastructure and accessible tooling. Not a tutorial project — actively used and maintained.

**For educational and research purposes. Not investment advice.**

---

*John Stremple | George Mason University — Finance & Computer Science | [GitHub](https://github.com/johnstremple)*
