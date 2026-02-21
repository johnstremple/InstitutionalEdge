"""
report_generator.py — Generate and export analysis reports.

Outputs:
- Formatted text report (always)
- JSON export (structured data)
- HTML report (optional, for browser viewing)
- Price chart with technical indicators (requires matplotlib)
"""

import json
import os
from datetime import datetime
from typing import Optional
import pandas as pd


class ReportGenerator:

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_text_report(self, results: dict, filename: Optional[str] = None) -> str:
        """Generate a clean plain-text report."""
        ticker = results.get("ticker", "UNKNOWN")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"{ticker}_{ts}_report.txt"
        filepath = os.path.join(self.output_dir, filename)

        lines = []
        lines.append("=" * 70)
        lines.append(f"  INSTITUTIONALEDGE — INVESTMENT ANALYSIS REPORT")
        lines.append(f"  {ticker} | {results.get('company_name', '')}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)

        # Company overview
        lines.append(f"\nSECTOR: {results.get('sector', 'N/A')}  |  INDUSTRY: {results.get('industry', 'N/A')}")
        mktcap = results.get("market_cap", 0)
        if mktcap:
            lines.append(f"MARKET CAP: ${mktcap/1e9:.2f}B")

        # Scores
        lines.append(f"\n{'─'*40}")
        lines.append("COMPOSITE SCORES")
        lines.append(f"{'─'*40}")
        fund = results.get("fundamental", {})
        tech = results.get("technical", {})
        sent = results.get("sentiment", {})
        risk = results.get("risk", {})
        lines.append(f"Fundamental:   {fund.get('score', 'N/A'):.1f}/100")
        lines.append(f"Technical:     {tech.get('score', 'N/A'):.1f}/100")
        lines.append(f"Sentiment:     {sent.get('score', 'N/A'):.1f}/100")
        lines.append(f"Risk-Adjusted: {risk.get('score', 'N/A'):.1f}/100")
        lines.append(f"COMPOSITE:     {results.get('composite_score', 0):.1f}/100")
        lines.append(f"SIGNAL:        {results.get('buy_signal', 'N/A')}")

        # Fundamentals
        lines.append(f"\n{'─'*40}")
        lines.append("FUNDAMENTAL ANALYSIS")
        lines.append(f"{'─'*40}")
        for k, v in fund.get("highlights", {}).items():
            lines.append(f"  {k:<25} {v}")

        quality = fund.get("quality_score", {})
        lines.append(f"\nPiotroski F-Score: {quality.get('f_score', 'N/A')}/9 — {quality.get('interpretation', '')}")
        for sig in quality.get("signals", []):
            lines.append(f"  ✓ {sig}")

        # Technical
        lines.append(f"\n{'─'*40}")
        lines.append("TECHNICAL ANALYSIS")
        lines.append(f"{'─'*40}")
        for k, v in tech.get("signals", {}).items():
            lines.append(f"  {k}: {v}")
        if tech.get("support"):
            lines.append(f"\n  Support Level:    ${tech['support']:.2f}")
        if tech.get("resistance"):
            lines.append(f"  Resistance Level: ${tech['resistance']:.2f}")
        lines.append(f"  Volume Trend:     {tech.get('volume_trend', 'N/A')}")

        # Sentiment
        lines.append(f"\n{'─'*40}")
        lines.append("NEWS SENTIMENT")
        lines.append(f"{'─'*40}")
        lines.append(f"  Overall: {sent.get('overall', 'N/A')}")
        lines.append(f"  Score:   {sent.get('score', 'N/A'):.1f}/100")
        headlines = sent.get("headlines", [])
        if headlines:
            lines.append(f"\n  Recent Headlines:")
            for h in headlines[:5]:
                lines.append(f"    • {h[:80]}")

        # Simulations
        lines.append(f"\n{'─'*40}")
        lines.append("INSTITUTIONAL SIMULATIONS")
        lines.append(f"{'─'*40}")

        sims = results.get("simulations", {})
        mc = sims.get("monte_carlo", {})
        if mc:
            lines.append(f"\n  Monte Carlo (10,000 paths, 1 Year):")
            lines.append(f"    Current Price:       ${mc.get('current_price', 0):.2f}")
            lines.append(f"    Median Price (1Y):   ${mc.get('median_price_1y', 0):.2f}")
            lines.append(f"    Median Return:       {mc.get('median_return_1y', 0)*100:.1f}%")
            lines.append(f"    Bull Case (P90):     {mc.get('bull_case', 0)*100:.1f}%")
            lines.append(f"    Bear Case (P10):     {mc.get('bear_case', 0)*100:.1f}%")
            lines.append(f"    Prob of Profit:      {mc.get('probability_profit', 0)*100:.0f}%")
            lines.append(f"    Prob +20%:           {mc.get('prob_up_20pct', 0)*100:.0f}%")
            lines.append(f"    Prob -20%:           {mc.get('prob_down_20pct', 0)*100:.0f}%")
            lines.append(f"    Annualized Vol:      {mc.get('annualized_vol', 0)*100:.1f}%")

        dcf = sims.get("dcf", {})
        if dcf and dcf.get("intrinsic_value"):
            lines.append(f"\n  DCF Valuation:")
            lines.append(f"    Current Price:       ${dcf.get('current_price', 0):.2f}")
            lines.append(f"    Intrinsic Value:     ${dcf.get('intrinsic_value', 0):.2f}")
            lines.append(f"    Upside/Downside:     {dcf.get('upside_downside', 0)*100:.1f}%")
            lines.append(f"    Status:              {dcf.get('valuation_status', 'N/A')}")
            lines.append(f"    WACC Used:           {dcf.get('wacc', 0)*100:.1f}%")
            lines.append(f"    MOS 20% Entry:       ${dcf.get('margin_of_safety_20', 0):.2f}")
            lines.append(f"    MOS 30% Entry:       ${dcf.get('margin_of_safety_30', 0):.2f}")

        scenarios = sims.get("scenarios", {})
        if scenarios and "bull" in scenarios:
            lines.append(f"\n  Scenario Analysis:")
            for name in ["bull", "base", "bear"]:
                sc = scenarios.get(name, {})
                lines.append(f"    {name.upper():<6} ({int(sc.get('probability',0)*100)}% prob): "
                              f"1Y={sc.get('return_pct_1y',0):.1f}%  "
                              f"5Y=${sc.get('price_5y',0):.2f}")

        stress = sims.get("stress_test", {})
        if stress:
            lines.append(f"\n  Historical Stress Tests (Beta={stress.get('beta_used','N/A')}):")
            for scenario, sc_data in stress.get("scenarios", {}).items():
                lines.append(f"    {scenario:<30} {sc_data.get('estimated_drawdown_pct',0):.1f}%")

        cagr = sims.get("cagr_projection", {})
        hist_cagr = cagr.get("historical_cagr_pct", {})
        if hist_cagr:
            lines.append(f"\n  Historical CAGR:")
            for period, rate in hist_cagr.items():
                lines.append(f"    {period}: {rate:.1f}%")

        # Risk
        lines.append(f"\n{'─'*40}")
        lines.append("RISK METRICS")
        lines.append(f"{'─'*40}")
        lines.append(f"  Annual Return:      {risk.get('annual_return', 0)*100:.2f}%")
        lines.append(f"  Annual Volatility:  {risk.get('annual_vol', 0)*100:.2f}%")
        lines.append(f"  Sharpe Ratio:       {risk.get('sharpe_ratio', 'N/A'):.3f}")
        lines.append(f"  Sortino Ratio:      {risk.get('sortino_ratio', 'N/A'):.3f}")
        lines.append(f"  Calmar Ratio:       {risk.get('calmar_ratio', 'N/A'):.3f}")
        lines.append(f"  Max Drawdown:       {risk.get('max_drawdown', 0)*100:.1f}%")
        lines.append(f"  VaR 95% (1D):       {risk.get('var_95', 0)*100:.2f}%")
        lines.append(f"  CVaR 95% (1D):      {risk.get('cvar_95', 0)*100:.2f}%")
        lines.append(f"  Beta:               {risk.get('beta', 'N/A'):.3f}")
        lines.append(f"  Alpha (Annual):     {risk.get('alpha', 0)*100:.2f}%")

        lines.append(f"\n{'='*70}")
        lines.append("  DISCLAIMER: For educational purposes only. Not financial advice.")
        lines.append("  Always conduct your own due diligence before investing.")
        lines.append(f"{'='*70}\n")

        text = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"\n  Report saved: {filepath}")
        return filepath

    def save_json(self, results: dict, filename: Optional[str] = None) -> str:
        """Save full results as JSON."""
        ticker = results.get("ticker", "UNKNOWN")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"{ticker}_{ts}_analysis.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return filepath

    def generate_chart(self, price_data: pd.DataFrame, ticker: str) -> Optional[str]:
        """Generate price chart with technical indicators."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from technical_analysis import TechnicalAnalyzer

            tech = TechnicalAnalyzer(price_data)
            df = tech._indicators

            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(f"{ticker} — Technical Analysis", fontsize=14, fontweight="bold")
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

            # Price + MAs + BB
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(df.index, df["Close"], label="Price", linewidth=1.5, color="black")
            ax1.plot(df.index, df["SMA_20"],  label="SMA20",  linewidth=1, alpha=0.8, color="blue")
            ax1.plot(df.index, df["SMA_50"],  label="SMA50",  linewidth=1, alpha=0.8, color="orange")
            ax1.plot(df.index, df["SMA_200"], label="SMA200", linewidth=1, alpha=0.8, color="red")
            ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.1, color="gray", label="Bollinger Bands")
            ax1.set_ylabel("Price ($)")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(alpha=0.3)

            # RSI
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(df.index, df["RSI"], color="purple", linewidth=1)
            ax2.axhline(70, color="red",   linestyle="--", alpha=0.5, label="Overbought")
            ax2.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold")
            ax2.set_ylabel("RSI")
            ax2.set_ylim(0, 100)
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(alpha=0.3)

            # MACD
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(df.index, df["MACD"],        color="blue",   linewidth=1, label="MACD")
            ax3.plot(df.index, df["MACD_Signal"], color="orange", linewidth=1, label="Signal")
            ax3.bar(df.index, df["MACD_Hist"], color=df["MACD_Hist"].apply(lambda x: "green" if x > 0 else "red"), alpha=0.5)
            ax3.set_ylabel("MACD")
            ax3.legend(loc="upper left", fontsize=8)
            ax3.grid(alpha=0.3)

            # Only show last 252 days
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(df.index[-252], df.index[-1])

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"{ticker}_{ts}_chart.png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Chart saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"  Chart generation skipped: {e}")
            return None
