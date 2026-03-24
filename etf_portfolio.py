"""
etf_portfolio.py — InstitutionalEdge Default 3-ETF Portfolio
VOO (S&P 500) | QQQM (Nasdaq-100) | SCHD (Dividend Growth)
Data pulled from each ETF's inception date.
"""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Palette ────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#21262d"
TEXT   = "#e6edf3"
SUBTEXT= "#8b949e"

ETF_COLORS = {
    "VOO":  "#58a6ff",
    "QQQM": "#3fb950",
    "SCHD": "#f0883e",
}

ETF_NAMES = {
    "VOO":  "VOO — S&P 500",
    "QQQM": "QQQM — Nasdaq-100",
    "SCHD": "SCHD — Dividend Growth",
}

# Real inception dates
INCEPTION = {
    "VOO":  "2010-09-07",
    "QQQM": "2020-10-13",
    "SCHD": "2011-10-20",
}

TICKERS = ["VOO", "QQQM", "SCHD"]
WEIGHTS = {"VOO": 0.40, "QQQM": 0.35, "SCHD": 0.25}


# ── Data Fetching ──────────────────────────────────────────────────────────────
def fetch_since_inception() -> dict:
    """Pull each ETF from its own inception date to today."""
    today = datetime.today().strftime("%Y-%m-%d")
    series = {}
    for t in TICKERS:
        raw = yf.download(t, start=INCEPTION[t], end=today,
                          auto_adjust=True, progress=False)
        series[t] = raw["Close"].squeeze().dropna()
        series[t].name = t
    return series


def calc_stats(series: dict) -> dict:
    stats = {}
    for t, s in series.items():
        total_ret  = (s.iloc[-1] / s.iloc[0]) - 1
        n_years    = len(s) / 252
        cagr       = (1 + total_ret) ** (1 / n_years) - 1
        daily_ret  = s.pct_change().dropna()
        volatility = daily_ret.std() * np.sqrt(252)
        sharpe     = (cagr - 0.045) / volatility
        drawdowns  = (s / s.cummax()) - 1
        max_dd     = drawdowns.min()
        stats[t] = {
            "total_return": total_ret,
            "cagr":         cagr,
            "volatility":   volatility,
            "sharpe":       sharpe,
            "max_drawdown": max_dd,
            "current":      s.iloc[-1],
            "inception":    INCEPTION[t],
            "years":        round(n_years, 1),
        }
    return stats


def build_blended(series: dict):
    """
    Blended portfolio aligned to QQQM inception (newest ETF)
    so all three contribute from day 1 of the shared window.
    """
    df = pd.DataFrame(series).dropna()
    norm = df / df.iloc[0]
    blend = sum(norm[t] * WEIGHTS[t] for t in TICKERS)
    blend.name = "Blend"
    return blend, df


# ── Chart ──────────────────────────────────────────────────────────────────────
def run_etf_portfolio(save: bool = False):
    print("\n  Fetching data from each ETF's inception date...\n")
    series = fetch_since_inception()
    stats  = calc_stats(series)
    blend, aligned = build_blended(series)

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle(
        "INSTITUTIONALEDGE  ·  Default 3-ETF Portfolio  ·  Since Inception",
        color=TEXT, fontsize=13, fontweight="bold", y=0.97,
        fontfamily="monospace"
    )

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.05, right=0.97,
        top=0.91, bottom=0.07
    )

    # ── 1. Individual cumulative returns (each from own inception) ──────────────
    ax_ind = fig.add_subplot(gs[0:2, 0:2])
    ax_ind.set_facecolor(PANEL)
    for sp in ax_ind.spines.values(): sp.set_color(BORDER)

    for t, s in series.items():
        norm = (s / s.iloc[0] - 1) * 100
        ax_ind.plot(norm.index, norm.values,
                    color=ETF_COLORS[t], linewidth=1.8,
                    label=f"{ETF_NAMES[t]}  (since {INCEPTION[t][:7]})")

    ax_ind.axhline(0, color=BORDER, linewidth=0.8)
    ax_ind.set_title("Cumulative Return — Each ETF from Inception",
                     color=TEXT, fontsize=10, pad=8)
    ax_ind.set_ylabel("Return (%)", color=SUBTEXT, fontsize=9)
    ax_ind.tick_params(colors=SUBTEXT, labelsize=8)
    ax_ind.legend(loc="upper left", fontsize=8,
                  facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax_ind.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)

    # ── 2. Blended portfolio since QQQM inception ───────────────────────────────
    ax_blend = fig.add_subplot(gs[0:2, 2:4])
    ax_blend.set_facecolor(PANEL)
    for sp in ax_blend.spines.values(): sp.set_color(BORDER)

    norm_aligned = aligned / aligned.iloc[0]
    for t in TICKERS:
        ax_blend.plot(norm_aligned.index, (norm_aligned[t] - 1) * 100,
                      color=ETF_COLORS[t], linewidth=1.5, alpha=0.7,
                      label=ETF_NAMES[t])

    ax_blend.plot(blend.index, (blend - 1) * 100,
                  color="white", linewidth=2.5, linestyle="--",
                  alpha=0.9, label="Blended (40/35/25)")

    ax_blend.axhline(0, color=BORDER, linewidth=0.8)
    ax_blend.set_title(f"Blended Portfolio  ·  Since QQQM Inception ({INCEPTION['QQQM'][:7]})",
                       color=TEXT, fontsize=10, pad=8)
    ax_blend.set_ylabel("Return (%)", color=SUBTEXT, fontsize=9)
    ax_blend.tick_params(colors=SUBTEXT, labelsize=8)
    ax_blend.legend(loc="upper left", fontsize=8,
                    facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax_blend.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)

    # ── 3. CAGR bar ─────────────────────────────────────────────────────────────
    ax_cagr = fig.add_subplot(gs[2, 0])
    ax_cagr.set_facecolor(PANEL)
    for sp in ax_cagr.spines.values(): sp.set_color(BORDER)

    cagr_vals = [stats[t]["cagr"] * 100 for t in TICKERS]
    bars = ax_cagr.bar(TICKERS, cagr_vals,
                       color=[ETF_COLORS[t] for t in TICKERS], width=0.5, zorder=3)
    for bar, val in zip(bars, cagr_vals):
        ax_cagr.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2,
                     f"{val:.1f}%", ha="center", va="bottom",
                     color=TEXT, fontsize=9, fontweight="bold")
    ax_cagr.set_title("CAGR Since Inception", color=TEXT, fontsize=9, pad=6)
    ax_cagr.set_ylim(0, max(cagr_vals) * 1.3)
    ax_cagr.tick_params(colors=SUBTEXT, labelsize=8)
    ax_cagr.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6, zorder=0)

    # ── 4. Volatility bar ────────────────────────────────────────────────────────
    ax_vol = fig.add_subplot(gs[2, 1])
    ax_vol.set_facecolor(PANEL)
    for sp in ax_vol.spines.values(): sp.set_color(BORDER)

    vol_vals = [stats[t]["volatility"] * 100 for t in TICKERS]
    bars2 = ax_vol.bar(TICKERS, vol_vals,
                       color=[ETF_COLORS[t] for t in TICKERS], width=0.5,
                       zorder=3, alpha=0.85)
    for bar, val in zip(bars2, vol_vals):
        ax_vol.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color=TEXT, fontsize=9, fontweight="bold")
    ax_vol.set_title("Annualized Volatility", color=TEXT, fontsize=9, pad=6)
    ax_vol.set_ylim(0, max(vol_vals) * 1.3)
    ax_vol.tick_params(colors=SUBTEXT, labelsize=8)
    ax_vol.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6, zorder=0)

    # ── 5. Total return bar ─────────────────────────────────────────────────────
    ax_tot = fig.add_subplot(gs[2, 2])
    ax_tot.set_facecolor(PANEL)
    for sp in ax_tot.spines.values(): sp.set_color(BORDER)

    tot_vals = [stats[t]["total_return"] * 100 for t in TICKERS]
    bars3 = ax_tot.bar(TICKERS, tot_vals,
                       color=[ETF_COLORS[t] for t in TICKERS], width=0.5, zorder=3)
    for bar, val in zip(bars3, tot_vals):
        ax_tot.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{val:.0f}%", ha="center", va="bottom",
                    color=TEXT, fontsize=9, fontweight="bold")
    ax_tot.set_title("Total Return (Inception → Now)", color=TEXT, fontsize=9, pad=6)
    ax_tot.set_ylim(0, max(tot_vals) * 1.25)
    ax_tot.tick_params(colors=SUBTEXT, labelsize=8)
    ax_tot.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6, zorder=0)

    # ── 6. Stats table ──────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 3])
    ax_tbl.set_facecolor(PANEL)
    ax_tbl.axis("off")
    ax_tbl.set_title("Key Metrics", color=TEXT, fontsize=9, pad=6)

    row_labels = ["Price", "CAGR", "Total Ret", "Vol", "Sharpe", "Max DD", "Inception", "History"]
    metrics = [
        ("current",      lambda v: f"${v:.2f}"),
        ("cagr",         lambda v: f"{v*100:.1f}%"),
        ("total_return", lambda v: f"{v*100:.0f}%"),
        ("volatility",   lambda v: f"{v*100:.1f}%"),
        ("sharpe",       lambda v: f"{v:.2f}"),
        ("max_drawdown", lambda v: f"{v*100:.1f}%"),
        ("inception",    lambda v: v[:7]),
        ("years",        lambda v: f"{v}y"),
    ]
    table_data = [[fmt(stats[t][key]) for t in TICKERS] for key, fmt in metrics]

    tbl = ax_tbl.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=TICKERS,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=TEXT)
        if r == 0:
            cell.set_facecolor(BORDER)
            if c >= 0:
                col = list(ETF_COLORS.values())[c] if c < len(TICKERS) else TEXT
                cell.set_text_props(color=col, fontweight="bold")
        if c == -1:
            cell.set_facecolor(BG)
            cell.set_text_props(color=SUBTEXT, fontstyle="italic")

    # ── footer ──────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Data: Yahoo Finance  ·  VOO since 2010-09  ·  SCHD since 2011-10  ·  QQQM since 2020-10  ·  "
        "Blended portfolio aligned to QQQM inception  ·  Past performance != future results  ·  "
        f"Generated {datetime.today().strftime('%b %d, %Y')}",
        ha="center", color=SUBTEXT, fontsize=7, fontfamily="monospace"
    )

    _print_summary(stats)

    if save:
        path = "etf_portfolio.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"\n  Chart saved -> {path}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _print_summary(stats: dict):
    print("  ╔═══════════════════════════════════════════════════════════════════════╗")
    print("  ║            DEFAULT 3-ETF PORTFOLIO — SINCE INCEPTION                 ║")
    print("  ╠═══════════════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'ETF':<6} {'Price':>8}  {'Total Ret':>10}  {'CAGR':>7}  {'Vol':>7}  {'Sharpe':>7}  {'Max DD':>7}  {'History':>7}  ║")
    print("  ╠═══════════════════════════════════════════════════════════════════════╣")
    for t in TICKERS:
        s = stats[t]
        print(f"  ║  {t:<6} ${s['current']:>7.2f}  {s['total_return']*100:>9.0f}%  "
              f"{s['cagr']*100:>6.1f}%  {s['volatility']*100:>6.1f}%  "
              f"{s['sharpe']:>7.2f}  {s['max_drawdown']*100:>6.1f}%  "
              f"{s['years']:>5.1f}y  ║")
    print("  ╠═══════════════════════════════════════════════════════════════════════╣")
    print("  ║  ALLOCATION:  VOO 40%  ·  QQQM 35%  ·  SCHD 25%                     ║")
    print("  ║  INCEPTION:   VOO Sep 2010  ·  SCHD Oct 2011  ·  QQQM Oct 2020       ║")
    print("  ║  NOTE: Blended chart aligned to QQQM inception (newest of the three) ║")
    print("  ╚═══════════════════════════════════════════════════════════════════════╝\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_etf_portfolio()
