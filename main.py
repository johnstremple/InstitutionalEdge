"""
InstitutionalEdge v5.0 — Full Institutional-Grade AI Financial Advisor

Modules:
  Core:        data_fetcher, fundamental_analysis, technical_analysis,
               sentiment_analysis, simulations, risk_manager
  v2:          competitive_analysis, price_target
  v3:          insider_tracker, short_interest, options_signals, macro_regime
  v4:          stock_screener, advisor, backtester
  v5 (NEW):    fraud_detection, factor_model, alt_data,
               advanced_tools, pdf_report
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ── Core modules ─────────────────────────────────────────────────
from data_fetcher          import DataFetcher
from fundamental_analysis  import FundamentalAnalyzer
from technical_analysis    import TechnicalAnalyzer
from sentiment_analysis    import SentimentAnalyzer
from simulations           import SimulationEngine
from portfolio_optimizer   import PortfolioOptimizer
from risk_manager          import RiskManager
from crypto_analyzer       import CryptoAnalyzer

# ── v2 ───────────────────────────────────────────────────────────
from competitive_analysis  import CompetitiveAnalyzer
from price_target          import PriceTargetEngine

# ── v3 ───────────────────────────────────────────────────────────
from insider_tracker       import InsiderTracker
from short_interest        import ShortInterestAnalyzer
from options_signals       import OptionsSignalAnalyzer
from macro_regime          import MacroRegimeModel

# ── v4 ───────────────────────────────────────────────────────────
from stock_screener        import StockScreener, SCREEN_PROFILES
from advisor               import InvestorAdvisor
from backtester            import Backtester

# ── v5 (new) ─────────────────────────────────────────────────────
from fraud_detection       import FraudDetector
from factor_model          import FactorModel
from alt_data              import AltDataAnalyzer
from advanced_tools        import OptionsFlowScanner, PortfolioRiskAnalyzer, NaturalLanguageScreener
from pdf_report            import ResearchReportGenerator


# ─── FULL STOCK ANALYSIS ──────────────────────────────────────────────────────

def analyze_stock(ticker: str, verbose: bool = True,
                  skip_macro: bool = False, generate_pdf: bool = False) -> dict:
    print(f"\n{'='*62}")
    print(f"  INSTITUTIONALEDGE v5.0  |  {ticker.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}\n")

    fetcher = DataFetcher()
    results = {}

    # 1. Data
    print("[1/11] Fetching market data...")
    stock_data = fetcher.get_stock_data(ticker, period="5y")
    financials = fetcher.get_financials(ticker)
    info       = fetcher.get_company_info(ticker)

    if stock_data is None or stock_data.empty:
        print(f"  ERROR: Could not fetch data for {ticker}.")
        return {}

    current_price = float(stock_data["Close"].iloc[-1])
    results.update({
        "ticker":        ticker.upper(),
        "company_name":  info.get("longName", ticker),
        "sector":        info.get("sector", "N/A"),
        "industry":      info.get("industry", "N/A"),
        "market_cap":    info.get("marketCap", 0),
        "current_price": current_price,
    })

    # 2. Fundamental
    print("[2/11] Fundamental analysis...")
    results["fundamental"] = FundamentalAnalyzer(info, financials).full_analysis()

    # 3. Technical
    print("[3/11] Technical analysis...")
    results["technical"] = TechnicalAnalyzer(stock_data).full_analysis()

    # 4. Sentiment
    print("[4/11] News sentiment...")
    results["sentiment"] = SentimentAnalyzer().analyze(ticker, info.get("longName", ticker))

    # 5. Simulations
    print("[5/11] Simulations (Monte Carlo + DCF)...")
    results["simulations"] = SimulationEngine(stock_data, info).run_all()

    # 6. Competitive
    print("[6/11] Competitive & moat analysis...")
    results["competitive"] = CompetitiveAnalyzer(ticker, info).full_analysis()

    # 7. Insider + Short + Options
    print("[7/11] Insider, short interest, options...")
    results["insider"]  = InsiderTracker().full_analysis(ticker)
    results["short"]    = ShortInterestAnalyzer(ticker, stock_data, info).full_analysis()
    results["options"]  = OptionsSignalAnalyzer(ticker, stock_data, info).full_analysis()

    # 8. Macro
    if not skip_macro:
        print("[8/11] Macro regime...")
        results["macro"] = MacroRegimeModel().full_analysis(info.get("sector",""))
    else:
        results["macro"] = None

    # 9. Risk
    print("[9/11] Risk metrics...")
    results["risk"] = RiskManager(stock_data).full_assessment()

    # 10. NEW: Fraud Detection + Factor Model + Alt Data
    print("[10/11] Fraud detection, factor model, alt data...")
    results["fraud_detection"] = FraudDetector(ticker, info, financials).full_analysis()
    results["factor_model"]    = FactorModel().full_analysis(ticker, stock_data)
    results["alt_data"]        = AltDataAnalyzer().full_analysis(
                                     ticker, info.get("longName", ticker))

    # 11. Price Target
    print("[11/11] Price target...")
    results["price_target"] = PriceTargetEngine(
        ticker, current_price, info,
        results["simulations"], results["technical"], results["competitive"]
    ).generate()

    # Composite score + signal
    results["composite_score"] = _composite_score(results)
    results["buy_signal"]      = _buy_signal(results)

    if verbose:
        _print_analysis(results)

    # PDF report
    if generate_pdf:
        try:
            rg = ResearchReportGenerator()
            pdf_path = rg.generate(results)
            results["pdf_path"] = pdf_path
        except Exception as e:
            print(f"  PDF generation error: {e}")

    return results


# ─── COMPOSITE SCORING ────────────────────────────────────────────────────────

def _composite_score(r: dict) -> float:
    fund    = r.get("fundamental",    {}).get("score", 50)
    tech    = r.get("technical",      {}).get("score", 50)
    sent    = r.get("sentiment",      {}).get("score", 50)
    risk    = r.get("risk",           {}).get("score", 50)
    comp    = r.get("competitive",    {}).get("score", 50)
    insider = r.get("insider",        {}).get("score", 50)
    short_s = r.get("short",          {}).get("score", 50)
    options = r.get("options",        {}).get("score", 50)
    fraud   = r.get("fraud_detection",{}).get("score", 70)
    factor  = r.get("factor_model",   {}).get("score", 50)
    alt     = r.get("alt_data",       {}).get("score", 50)

    macro_adj = 0
    macro = r.get("macro")
    if macro:
        macro_adj = macro.get("sector_fit",{}).get("macro_score_adjustment", 0)

    base = (
        fund    * 0.25 +
        comp    * 0.18 +
        tech    * 0.14 +
        sent    * 0.08 +
        risk    * 0.08 +
        insider * 0.07 +
        fraud   * 0.07 +   # NEW
        factor  * 0.05 +   # NEW
        options * 0.04 +
        alt     * 0.02 +   # NEW
        short_s * 0.02
    )
    return round(min(max(base + macro_adj, 0), 100), 2)


def _buy_signal(r: dict) -> str:
    score = r.get("composite_score", 50)
    mc    = r.get("simulations",{}).get("monte_carlo",{}).get("median_return_1y", 0)
    pt    = r.get("price_target",{}).get("upside_downside", 0)
    fraud = r.get("fraud_detection",{}).get("overall_risk",{}).get("composite_score", 70)

    # Fraud penalty
    if fraud < 30: return "AVOID"  # High fraud risk → always avoid

    if score >= 73 and (mc > 0.10 or pt > 10):  return "STRONG BUY"
    elif score >= 61 and (mc > 0.05 or pt > 5): return "BUY"
    elif score >= 46:                            return "HOLD"
    elif score >= 34:                            return "WEAK"
    else:                                        return "AVOID"


# ─── PRINT ────────────────────────────────────────────────────────────────────

def _print_analysis(r: dict):
    pt      = r.get("price_target",    {})
    fund    = r.get("fundamental",     {})
    tech    = r.get("technical",       {})
    sent    = r.get("sentiment",       {})
    risk    = r.get("risk",            {})
    comp    = r.get("competitive",     {})
    sims    = r.get("simulations",     {})
    insider = r.get("insider",         {})
    short   = r.get("short",           {})
    options = r.get("options",         {})
    macro   = r.get("macro")
    fraud   = r.get("fraud_detection", {})
    factor  = r.get("factor_model",    {})
    alt     = r.get("alt_data",        {})
    mc      = sims.get("monte_carlo",  {})
    dcf     = sims.get("dcf",          {})
    mktcap  = r.get("market_cap", 0)

    print(f"\n{'='*62}")
    print(f"  {r['ticker']} — {r['company_name']}")
    print(f"  {r.get('sector')} | {r.get('industry')}")
    if mktcap:
        print(f"  ${mktcap/1e9:.2f}B Mkt Cap | Price: ${r.get('current_price',0):.2f}")

    # Price Target
    if pt.get("price_target_12m"):
        up = pt.get("upside_downside", 0)
        print(f"\n  12M TARGET:  ${pt['price_target_12m']:.2f}  {'▲' if up>0 else '▼'}{abs(up):.1f}%")
        print(f"  Range:       ${pt.get('confidence_low',0):.2f} — ${pt.get('confidence_high',0):.2f}")
        print(f"  Conviction:  {pt.get('conviction','')}")
        print(f"\n  Model Breakdown:")
        for nm, m in pt.get("models",{}).items():
            if m.get("price"):
                print(f"    {m.get('source',nm):<38} ${m['price']:<8.2f} ({m.get('upside_pct',0):+.1f}%)")

    # Scores
    print(f"\n  SCORES:")
    print(f"  Fund {fund.get('score',0):.0f} | Comp {comp.get('score',0):.0f} | Tech {tech.get('score',0):.0f} | "
          f"Sent {sent.get('score',0):.0f} | Insider {insider.get('score',0):.0f}")
    print(f"  Fraud {fraud.get('score',0):.0f} | Factor {factor.get('score',0):.0f} | "
          f"Alt {alt.get('score',0):.0f} | Options {options.get('score',0):.0f}")
    print(f"  ─────────────────────────────────")
    print(f"  COMPOSITE: {r.get('composite_score',0):.1f}/100")

    # Thesis
    if pt.get("thesis"):
        print(f"\n  THESIS: {pt['thesis'][:250]}...")

    # Fraud / Accounting
    print(f"\n  {'─'*50}")
    print(f"  ACCOUNTING QUALITY")
    b  = fraud.get("beneish",{})
    az = fraud.get("altman",{})
    eq = fraud.get("earnings_quality",{})
    print(f"  Beneish M-Score:  {b.get('m_score','N/A')} — {b.get('verdict','')}")
    print(f"  Altman Z-Score:   {az.get('z_score','N/A')} — {az.get('zone','')}")
    print(f"  Earnings Quality: {eq.get('score','N/A')}/100 — {eq.get('verdict','')}")
    for flag in fraud.get("overall_risk",{}).get("all_flags",[])[:3]:
        print(f"  ⚠️  {flag}")

    # Factor Model
    fm = factor.get("factor_exposures",{})
    sb = factor.get("smart_beta",{})
    if sb.get("classification"):
        print(f"\n  FACTOR MODEL: {sb.get('classification','')} | "
              f"Alpha {fm.get('alpha_annual','N/A')}%/yr | "
              f"R² {fm.get('r_squared','N/A')}")
        betas = fm.get("factor_betas",{})
        if betas:
            beta_str = " | ".join([f"{k} {v:.2f}" for k,v in betas.items()])
            print(f"  Betas: {beta_str}")

    # Alt Data
    print(f"\n  ALT DATA:")
    reddit = alt.get("reddit",{})
    earn   = alt.get("earnings_calendar",{})
    if reddit.get("mention_count",0) > 0:
        print(f"  Reddit: {reddit.get('mention_count',0)} mentions | {reddit.get('wsb_hype','')} | {reddit.get('sentiment','')}")
    if earn.get("next_earnings_date","") not in ["Unknown",""]:
        print(f"  Earnings: {earn.get('next_earnings_date','')} ({earn.get('days_to_earnings','?')} days)")
    if earn.get("beat_signal"):
        print(f"  {earn.get('beat_signal','')}")

    # Macro
    if macro:
        print(f"\n  MACRO: {macro.get('regime','')} — {macro.get('sector_fit',{}).get('assessment','')}")

    # Insider + Short + Options
    print(f"\n  INSIDER:  {insider.get('signal','')}")
    sf = short.get("short_float_pct")
    sq = short.get("squeeze_score",0)
    if sf: print(f"  SHORT:    {sf:.1f}% float | Squeeze {sq}/100 — {short.get('squeeze_risk','')}")
    pcr = options.get("put_call_ratio",{})
    if pcr.get("pcr_volume"):
        print(f"  OPTIONS:  P/C {pcr['pcr_volume']:.2f} — {pcr.get('sentiment','')}")

    # Fundamentals
    print(f"\n  FUNDAMENTALS:")
    for k, v in fund.get("highlights",{}).items():
        print(f"  {k:<28} {v}")
    q = fund.get("quality_score",{})
    if q: print(f"  F-Score: {q.get('f_score','N/A')}/9 — {q.get('interpretation','')}")

    # Simulations
    if mc:
        print(f"\n  MC: Median {mc.get('median_return_1y',0)*100:.1f}% | "
              f"Bull {mc.get('bull_case',0)*100:.1f}% | Bear {mc.get('bear_case',0)*100:.1f}%")
    if dcf and dcf.get("intrinsic_value"):
        print(f"  DCF: ${dcf['intrinsic_value']:.2f} intrinsic — {dcf.get('valuation_status','')}")

    # Risk
    print(f"\n  Risk: Sharpe {risk.get('sharpe_ratio',0):.2f} | "
          f"Max DD {risk.get('max_drawdown',0)*100:.1f}% | "
          f"Beta {risk.get('beta',0):.2f} | VaR {risk.get('var_95',0)*100:.2f}%")

    # Catalysts + Risks
    for c in pt.get("catalysts",[])[:3]: print(f"  🚀 {c}")
    for rk in pt.get("risks",[])[:3]:   print(f"  ⚠️  {rk}")

    # Final signal
    signal  = r.get("buy_signal","HOLD")
    display = {"STRONG BUY":"🟢🟢 STRONG BUY","BUY":"🟢  BUY","HOLD":"🟡  HOLD",
               "WEAK":"🟠  WEAK","AVOID":"🔴  AVOID"}.get(signal, signal)
    print(f"\n  {'═'*50}")
    print(f"  SIGNAL:    {display}")
    print(f"  COMPOSITE: {r.get('composite_score',0):.1f}/100")
    if pt.get("price_target_12m"):
        up = pt.get("upside_downside",0)
        print(f"  12M TARGET: ${pt['price_target_12m']:.2f} ({'▲' if up>0 else '▼'}{abs(up):.1f}%)")
    print(f"  {'═'*50}\n")
    print(f"  ⚠  Educational only. Not financial advice.\n")


# ─── OTHER COMMANDS ───────────────────────────────────────────────────────────

def run_screener(profile="garp", include_small_cap=False, limit=20):
    print(f"\n{'='*62}\n  STOCK SCREENER — {profile.upper()}\n{'='*62}")
    screener = StockScreener()
    tickers  = screener.get_full_universe(include_small_cap)
    df       = screener.fetch_bulk_metrics(tickers)
    results  = screener.screen(profile, df, limit=limit)
    screener.print_screen_results(results, f"Top {limit} — {profile}")
    return results.to_dict("records") if not results.empty else []


def run_momentum_screen(limit=20):
    print(f"\n{'='*62}\n  MOMENTUM SCREEN\n{'='*62}\n")
    screener = StockScreener()
    df       = screener.momentum_screen(top_n=limit)
    if not df.empty:
        print(f"  {'#':<4}{'Ticker':<8}{'Price':<10}{'1M':>8}{'3M':>8}{'12M':>8}{'Momentum':>10}")
        print("  " + "─"*55)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"  {i:<4}{row['ticker']:<8}${row['price']:<9.2f}"
                  f"{row['1m_ret']:>7.1f}%{row['3m_ret']:>7.1f}%"
                  f"{row['12m_ret']:>7.1f}%{row['momentum']:>9.1f}%")
    return df.to_dict("records") if not df.empty else []


def run_advisor(auto_screen=True):
    advisor  = InvestorAdvisor()
    profile  = advisor.run_survey()
    screened = []
    if auto_screen and not profile.prefers_etfs:
        sp = {"growth":"aggressive_growth","income":"dividend_growth",
              "preservation":"deep_value","speculation":"aggressive_growth","balanced":"garp"
              }.get(profile.primary_goal, "garp")
        screener = StockScreener()
        df       = screener.fetch_bulk_metrics(screener.get_sp500_tickers())
        res      = screener.screen(sp, df, limit=20)
        screened = res.to_dict("records") if not res.empty else []
    rec = advisor.build_portfolio_recommendation(profile, screened)
    advisor.print_recommendation(rec)
    with open(f"{profile.name.lower()}_portfolio.json","w") as f:
        json.dump(rec, f, indent=2, default=str)
    return rec


def run_backtest(mode="compare", tickers=None, start="2019-01-01"):
    bt = Backtester(100_000)
    if mode == "compare":   return bt.compare_strategies(start)
    elif mode == "mag7":    return bt.backtest_magnificent7(start)
    elif mode == "ai":      return bt.backtest_ai_theme(start)
    elif mode == "quality": return bt.backtest_quality_screen(start)
    elif mode == "growth":  return bt.backtest_growth_screen(start)
    elif mode == "custom" and tickers:
        return bt.backtest_strategy(tickers, start_date=start, top_n=min(15,len(tickers)))
    return {}


def run_macro():
    macro = MacroRegimeModel().full_analysis()
    print(f"\n  Regime: {macro.get('regime')} | {macro.get('asset_bias','')}")
    print(f"  {macro.get('description','')}")
    print(f"  Favored: {', '.join(macro.get('favored_sectors',[]))}")
    for sec, d in sorted(macro.get("sector_rotation",{}).items(),
                          key=lambda x: x[1].get("momentum_score",0), reverse=True):
        print(f"  {d.get('signal','')} {sec:<28} 1M:{d.get('1m_return',0):+.1f}%  3M:{d.get('3m_return',0):+.1f}%")
    return macro


def run_options_scan(tickers=None):
    if tickers is None:
        tickers = ["AAPL","MSFT","NVDA","AMD","TSLA","META","AMZN","GOOGL",
                   "SMCI","PLTR","COIN","HOOD","MSTR","SPY","QQQ"]
    scanner = OptionsFlowScanner()
    alerts  = scanner.scan(tickers)
    scanner.print_flow(alerts)
    return alerts


def run_risk_attribution(holdings: dict):
    """holdings = {"AAPL":0.30, "MSFT":0.25, ...}"""
    analyzer = PortfolioRiskAnalyzer()
    result   = analyzer.analyze(holdings)
    analyzer.print_attribution(result)
    return result


def run_nl_screen(query: str):
    nl = NaturalLanguageScreener()
    return nl.run(query)


def generate_watchlist(tickers):
    print(f"\n{'='*62}\n  WATCHLIST: {len(tickers)} stocks\n{'='*62}\n")
    scored = []
    for ticker in tickers:
        print(f"  Analyzing {ticker}...")
        r = analyze_stock(ticker, verbose=False, skip_macro=True)
        if r:
            pt = r.get("price_target",{})
            fraud = r.get("fraud_detection",{}).get("beneish",{})
            scored.append({
                "ticker":          ticker,
                "company":         r.get("company_name",ticker)[:26],
                "composite_score": r.get("composite_score",0),
                "buy_signal":      r.get("buy_signal","HOLD"),
                "price_target":    pt.get("price_target_12m"),
                "upside":          pt.get("upside_downside"),
                "moat":            r.get("competitive",{}).get("moat_score",{}).get("width","N/A"),
                "m_score":         fraud.get("m_score","N/A"),
                "z_score":         r.get("fraud_detection",{}).get("altman",{}).get("z_score","N/A"),
            })
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    print(f"\n  {'#':<4}{'Ticker':<8}{'Company':<27}{'Score':<8}{'Signal':<12}{'Target':<10}{'Upside':<10}{'M-Score':>8}")
    print("  " + "─"*80)
    for i, s in enumerate(scored, 1):
        tgt = f"${s['price_target']:.2f}" if s.get("price_target") else "N/A"
        up  = f"{s['upside']:.1f}%" if s.get("upside") is not None else "N/A"
        ms  = f"{s['m_score']:.2f}" if isinstance(s.get("m_score"), float) else "N/A"
        print(f"  {i:<4}{s['ticker']:<8}{s['company']:<27}{s['composite_score']:<8.1f}"
              f"{s['buy_signal']:<12}{tgt:<10}{up:<10}{ms:>8}")
    return scored


def screen_etfs():
    screener = StockScreener()
    df       = screener.analyze_etfs()
    if df.empty:
        print("  Could not fetch ETF data."); return
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat].sort_values("1y_return", ascending=False)
        print(f"\n  📂 {cat}")
        print(f"  {'Symbol':<8}{'Name':<35}{'Expense':>9}{'Div':>8}{'1M':>8}{'1Y':>8}")
        print("  " + "─"*80)
        for _, row in cat_df.iterrows():
            exp = f"{row['expense_ratio']:.2f}%" if row.get("expense_ratio") else "N/A"
            dy  = f"{row['dividend_yield']:.2f}%" if row.get("dividend_yield") else "0%"
            m1  = f"{row['1m_return']:+.1f}%" if row.get("1m_return") else "N/A"
            y1  = f"{row['1y_return']:+.1f}%" if row.get("1y_return") else "N/A"
            print(f"  {row['symbol']:<8}{str(row['name'])[:33]:<35}{exp:>9}{dy:>8}{m1:>8}{y1:>8}")


def analyze_portfolio(tickers):
    import pandas as pd
    fetcher    = DataFetcher()
    price_data = {}
    for t in tickers:
        d = fetcher.get_stock_data(t, period="3y")
        if d is not None and not d.empty:
            price_data[t] = d["Close"]
    if len(price_data) < 2:
        print("  Need at least 2 valid tickers."); return {}
    prices_df = pd.DataFrame(price_data).dropna()
    results   = PortfolioOptimizer(prices_df).full_optimization()
    print(f"\n  Max Sharpe Weights:")
    for t, w in results.get("optimal_weights",{}).items():
        print(f"  {t}: {w*100:.1f}%")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def print_help():
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║        INSTITUTIONALEDGE v5.0 — COMMAND GUIDE               ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  STOCK ANALYSIS                                              ║
  ║  python main.py analyze NVDA                                 ║
  ║  python main.py analyze NVDA --pdf          (+ PDF report)   ║
  ║  python main.py watchlist NVDA AAPL MSFT SMCI PLTR           ║
  ║  python main.py portfolio AAPL MSFT NVDA AMZN                ║
  ║  python main.py macro                                        ║
  ║  python main.py crypto bitcoin                               ║
  ║                                                              ║
  ║  AI ADVISOR (survey → auto portfolio)                        ║
  ║  python main.py advisor                                      ║
  ║                                                              ║
  ║  STOCK DISCOVERY                                             ║
  ║  python main.py screen --profile garp                        ║
  ║  python main.py screen --profile aggressive_growth           ║
  ║  python main.py screen --profile deep_value                  ║
  ║  python main.py screen --profile quality_compounder          ║
  ║  python main.py screen --profile dividend_growth             ║
  ║  python main.py momentum                                     ║
  ║  python main.py etfs                                         ║
  ║                                                              ║
  ║  NATURAL LANGUAGE SCREENER (type what you want)              ║
  ║  python main.py ask "cheap AI stocks with insider buying"    ║
  ║  python main.py ask "profitable tech under PE 30"            ║
  ║  python main.py ask "dividend stocks yielding over 4%"       ║
  ║                                                              ║
  ║  BACKTESTING                                                 ║
  ║  python main.py backtest --mode compare                      ║
  ║  python main.py backtest --mode mag7                         ║
  ║  python main.py backtest --mode ai                           ║
  ║  python main.py backtest --mode custom --tickers NVDA AMD    ║
  ║                                                              ║
  ║  ADVANCED TOOLS                                              ║
  ║  python main.py options-scan                                 ║
  ║  python main.py risk-attribution AAPL:30 MSFT:25 NVDA:45    ║
  ║                                                              ║
  ║  WEB DASHBOARD                                               ║
  ║  streamlit run dashboard.py                                  ║
  ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstitutionalEdge v5.0", add_help=False)
    subs   = parser.add_subparsers(dest="command")

    a = subs.add_parser("analyze")
    a.add_argument("ticker")
    a.add_argument("--pdf", action="store_true")

    w = subs.add_parser("watchlist")
    w.add_argument("tickers", nargs="+")

    p = subs.add_parser("portfolio")
    p.add_argument("tickers", nargs="+")

    subs.add_parser("macro")

    c = subs.add_parser("crypto")
    c.add_argument("symbol")

    sc = subs.add_parser("screen")
    sc.add_argument("--profile", default="garp", choices=list(SCREEN_PROFILES.keys()))
    sc.add_argument("--limit", type=int, default=20)
    sc.add_argument("--small-cap", action="store_true")

    subs.add_parser("momentum")
    subs.add_parser("etfs")

    subs.add_parser("advisor")

    bt = subs.add_parser("backtest")
    bt.add_argument("--mode", default="compare",
                    choices=["compare","custom","mag7","ai","quality","growth"])
    bt.add_argument("--tickers", nargs="*")
    bt.add_argument("--start", default="2019-01-01")

    ask_p = subs.add_parser("ask")
    ask_p.add_argument("query", nargs="+")

    os_p = subs.add_parser("options-scan")
    os_p.add_argument("--tickers", nargs="*")

    ra_p = subs.add_parser("risk-attribution")
    ra_p.add_argument("holdings", nargs="+",
                       help="Format: TICKER:WEIGHT e.g. AAPL:30 MSFT:25 NVDA:45")

    subs.add_parser("help")

    args = parser.parse_args()

    if args.command == "analyze":
        r = analyze_stock(args.ticker, generate_pdf=args.pdf)
        with open(f"{args.ticker.upper()}_analysis.json","w") as f:
            json.dump(r, f, indent=2, default=str)

    elif args.command == "watchlist":
        r = generate_watchlist(args.tickers)
        with open("watchlist.json","w") as f:
            json.dump(r, f, indent=2, default=str)

    elif args.command == "portfolio":
        r = analyze_portfolio(args.tickers)

    elif args.command == "macro":
        r = run_macro()

    elif args.command == "crypto":
        ca = CryptoAnalyzer()
        r  = ca.full_analysis(args.symbol)
        ca.print_summary(r)

    elif args.command == "screen":
        r = run_screener(args.profile, args.small_cap, args.limit)

    elif args.command == "momentum":
        r = run_momentum_screen()

    elif args.command == "etfs":
        screen_etfs()

    elif args.command == "advisor":
        r = run_advisor()

    elif args.command == "backtest":
        r = run_backtest(args.mode, args.tickers, args.start)
        with open("backtest_results.json","w") as f:
            json.dump(r, f, indent=2, default=str)

    elif args.command == "ask":
        query = " ".join(args.query)
        r = run_nl_screen(query)

    elif args.command == "options-scan":
        r = run_options_scan(args.tickers)

    elif args.command == "risk-attribution":
        holdings = {}
        for h in args.holdings:
            parts = h.split(":")
            if len(parts) == 2:
                holdings[parts[0].upper()] = float(parts[1])
        if holdings:
            r = run_risk_attribution(holdings)
        else:
            print("  Format: python main.py risk-attribution AAPL:30 MSFT:25 NVDA:45")

    else:
        print_help()
