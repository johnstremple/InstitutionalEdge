# InstitutionalEdge v5.0
### Institutional-Grade AI Financial Advisor — Built in Python, 100% Free Data

> A full-stack quantitative investment platform combining the analytical tools used by hedge funds and institutional investors — stock screening, fraud detection, factor models, backtesting, and AI-powered portfolio construction — using only free, publicly available data sources.

---

## What It Does

Most retail investors have access to basic charts and P/E ratios. InstitutionalEdge gives you the same analytical framework used by professional portfolio managers:

- **Finds stocks for you** — screens the entire S&P 500 + NASDAQ-100 automatically
- **Builds your portfolio** — survey-based AI advisor matches investments to your goals
- **Detects accounting fraud** — Beneish M-Score flags earnings manipulation before it becomes a headline
- **Quantifies alpha** — Fama-French 5-Factor model separates skill from luck
- **Proves the strategy** — backtesting engine with CAGR, Sharpe, alpha, max drawdown
- **Generates research reports** — professional PDF output that looks like sell-side equity research

---

## Features

### Stock Discovery Engine
Automatically screens 500+ stocks across 7 strategies — no ticker input required.

| Profile | Strategy |
|---------|----------|
| `aggressive_growth` | High revenue growth, expanding margins |
| `garp` | Growth at a Reasonable Price (PEG-based) |
| `deep_value` | Low P/E, P/B — trading below intrinsic value |
| `quality_compounder` | High ROE, wide moat, durable competitive advantage |
| `dividend_growth` | Dividend yield + growth potential |
| `momentum` | 12-1 month price momentum factor |
| `small_cap_growth` | Under-followed small caps with high growth |

### AI Financial Advisor
Survey-based portfolio builder — 15 questions about your risk tolerance, goals, time horizon, and capital — then auto-screens and builds a personalized portfolio with exact dollar amounts, 30-year projections, and tax optimization notes.

### 11-Module Stock Analysis Engine

| Module | What It Measures |
|--------|-----------------|
| Fundamental Analysis | P/E, P/B, EV/EBITDA, margins, growth, Piotroski F-Score |
| Technical Analysis | 15+ indicators — RSI, MACD, Bollinger Bands, moving averages |
| Competitive Analysis | Economic moat, peer comparison, competitive positioning |
| Monte Carlo Simulation | 10,000 price paths, bull/bear/base scenarios |
| DCF Valuation | Discounted cash flow with margin of safety |
| Insider Tracking | SEC Form 4 filings, cluster buying detection |
| Short Interest | Float %, days to cover, squeeze probability score |
| Options Signals | Put/call ratio, IV rank, max pain, unusual activity |
| Macro Regime | Economic cycle detection, sector rotation signals |
| Fraud Detection | Beneish M-Score + Altman Z-Score + earnings quality |
| Fama-French Factor Model | Alpha, factor betas, R-squared, style classification |

### Fraud Detection
The same models that flagged Enron and WorldCom before their collapses.

- **Beneish M-Score** — 8-variable model detecting earnings manipulation. Score > -1.78 = likely manipulator
- **Altman Z-Score** — Bankruptcy prediction. Score < 1.81 = financial distress zone
- **Earnings Quality Score** — Cash flow vs reported earnings, accruals analysis

### Fama-French 5-Factor Model
Downloads factor data free from Kenneth French's Dartmouth library and runs OLS regression to decompose returns into MKT, SMB, HML, RMW, CMA, and MOM factors. Output: annualized alpha, factor betas, R-squared, style classification.

### Backtesting Engine
Walk-forward backtest with annual/quarterly rebalancing. Outputs CAGR vs benchmark, alpha (simple + CAPM), Sharpe/Sortino/Calmar ratios, maximum drawdown, and monthly win rate vs S&P 500.

### Natural Language Screener
```bash
python main.py ask "cheap AI stocks with insider buying"
python main.py ask "profitable dividend stocks no debt"
python main.py ask "small cap biotech high growth under PE 20"
```

---

## Installation

```bash
git clone https://github.com/johnstremple/InstitutionalEdge.git
cd InstitutionalEdge
pip install yfinance pandas numpy scipy requests feedparser vaderSentiment matplotlib streamlit reportlab pytrends
```

---

## Usage

```bash
# Full stock analysis with PDF report
python main.py analyze NVDA --pdf

# AI advisor — survey builds your personal portfolio
python main.py advisor

# Find stocks automatically
python main.py screen --profile garp
python main.py screen --profile aggressive_growth

# Natural language screener
python main.py ask "profitable tech stocks with low debt"

# Backtest strategies vs S&P 500
python main.py backtest --mode compare
python main.py backtest --mode custom --tickers NVDA AMD MSFT AAPL

# Options flow scanner
python main.py options-scan

# Portfolio risk attribution
python main.py risk-attribution AAPL:30 MSFT:25 NVDA:45

# Watchlist ranking
python main.py watchlist NVDA SMCI AMD AAPL MSFT META TSLA PLTR

# Macro regime + sector rotation
python main.py macro

# Web dashboard
streamlit run dashboard.py
```

---

## Sample Output

```
NVDA — NVIDIA Corporation | $4,621B Mkt Cap | Price: $189.82

12M TARGET:  $226.50  19.3% upside

SCORES:
Fund 79 | Comp 75 | Tech 62 | Sent 53 | Insider 56
Fraud 45 | Factor 91 | Alt 80 | Options 76
COMPOSITE: 76.6/100

ACCOUNTING QUALITY
Beneish M-Score:  -1.02  (high growth false positive)
Altman Z-Score:   91.10  SAFE ZONE
Earnings Quality: 75/100 HIGH QUALITY

FACTOR MODEL: Quality Growth | Alpha +8.2%/yr | R-squared 0.71

SIGNAL: STRONG BUY
```

---

## Data Sources (all free)

| Source | Data |
|--------|------|
| yfinance | Prices, financials, options chains |
| SEC EDGAR API | Form 4 insider filings, 13F holdings |
| Kenneth French Data Library | Fama-French 5-Factor + Momentum |
| Reddit JSON API | WallStreetBets sentiment |
| Wikipedia | S&P 500 and NASDAQ-100 lists |
| Google Trends (pytrends) | Retail search interest |

**Total data cost: $0/month**

---

## Architecture

```
InstitutionalEdge/
├── main.py                  # CLI — all commands
├── data_fetcher.py          # Market data
├── fundamental_analysis.py  # Ratios, Piotroski F-Score
├── technical_analysis.py    # 15+ technical indicators
├── competitive_analysis.py  # Moat scoring, peer comparison
├── simulations.py           # Monte Carlo, DCF
├── risk_manager.py          # Sharpe, VaR, drawdown
├── sentiment_analysis.py    # News sentiment (VADER)
├── insider_tracker.py       # SEC Form 4, 13F
├── short_interest.py        # Squeeze probability model
├── options_signals.py       # P/C ratio, IV rank, max pain
├── macro_regime.py          # Economic cycle, sector rotation
├── price_target.py          # 5-model blended price target
├── fraud_detection.py       # Beneish M-Score, Altman Z-Score
├── factor_model.py          # Fama-French 5-Factor regression
├── alt_data.py              # Reddit sentiment, earnings calendar
├── advanced_tools.py        # Options scanner, risk attribution, NL screener
├── stock_screener.py        # Universe screener
├── advisor.py               # Survey-based portfolio builder
├── backtester.py            # Walk-forward backtesting
├── portfolio_optimizer.py   # Mean-variance optimization (MPT)
├── crypto_analyzer.py       # Cryptocurrency analysis
├── pdf_report.py            # Institutional PDF report generator
├── dashboard.py             # Streamlit web dashboard
└── report_generator.py      # Text report output
```

---

## Disclaimer

For educational and research purposes only. Not investment advice. Always consult a licensed financial advisor before making investment decisions.

---

*Built by John Stremple | George Mason University — Finance & Computer Science*
