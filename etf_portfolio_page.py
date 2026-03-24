"""
etf_portfolio_page.py — InstitutionalEdge ETF Portfolio Page
VOO | QQQM | SCHD — Equal weight (33/33/33)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

TICKERS  = ["VOO", "QQQM", "SCHD"]
WEIGHTS  = {"VOO": 1/3, "QQQM": 1/3, "SCHD": 1/3}
INCEPTION = {"VOO": "2010-09-07", "QQQM": "2020-10-13", "SCHD": "2011-10-20"}

ETF_META = {
    "VOO":  {
        "name":    "Vanguard S&P 500 ETF",
        "color":   "#16A34A",
        "expense": "0.03%",
        "focus":   "Large-cap U.S. equities — tracks the S&P 500",
        "why":     "The foundation. Tracks the 500 largest U.S. companies at an ultra-low cost. The benchmark most active managers fail to beat long-term.",
    },
    "QQQM": {
        "name":    "Invesco Nasdaq-100 ETF",
        "color":   "#2563EB",
        "expense": "0.15%",
        "focus":   "Top 100 Nasdaq companies — tech-heavy growth",
        "why":     "The growth engine. Apple, Microsoft, Nvidia, Meta, Amazon. Higher risk, higher ceiling. QQQM is the lower-cost version of QQQ.",
    },
    "SCHD": {
        "name":    "Schwab U.S. Dividend Equity ETF",
        "color":   "#D97706",
        "expense": "0.06%",
        "focus":   "High-quality dividend-paying stocks",
        "why":     "The stabilizer. Strong fundamentals, consistent dividends, lower volatility. Balances out the risk from QQQM and provides income.",
    },
}

SAMPLE = {
    "VOO":  {"cagr":13.9,"total":521,"vol":17.2,"sharpe":0.86,"maxdd":-23.9,"price":548.32,"years":14.5},
    "QQQM": {"cagr":17.1,"total":108,"vol":21.8,"sharpe":0.93,"maxdd":-32.6,"price":511.67,"years":4.4},
    "SCHD": {"cagr":11.2,"total":287,"vol":14.6,"sharpe":0.80,"maxdd":-26.1,"price":28.14, "years":13.4},
}

@st.cache_data(ttl=3600)
def load_data():
    today = datetime.today().strftime("%Y-%m-%d")
    series = {}
    for t in TICKERS:
        raw = yf.download(t, start=INCEPTION[t], end=today, auto_adjust=True, progress=False)
        s = raw["Close"].squeeze().dropna()
        s.name = t
        series[t] = s
    return series

def calc_stats(series):
    stats = {}
    for t, s in series.items():
        total_ret = (s.iloc[-1] / s.iloc[0]) - 1
        n_years   = len(s) / 252
        cagr      = (1 + total_ret) ** (1 / n_years) - 1
        daily_ret = s.pct_change().dropna()
        vol       = daily_ret.std() * np.sqrt(252)
        sharpe    = (cagr - 0.045) / vol
        max_dd    = ((s / s.cummax()) - 1).min()
        stats[t]  = {
            "cagr":  round(cagr * 100, 1),
            "total": round(total_ret * 100, 0),
            "vol":   round(vol * 100, 1),
            "sharpe":round(sharpe, 2),
            "maxdd": round(max_dd * 100, 1),
            "price": round(float(s.iloc[-1]), 2),
            "years": round(n_years, 1),
        }
    return stats

def render():
    st.markdown("## ETF Portfolio — VOO / QQQM / SCHD")
    st.markdown("<div style='color:#6B7280;margin-bottom:20px;'>A simple, low-cost, diversified starting point. Three ETFs. Equal weight. Covers the full market, tech growth, and dividend income.</div>", unsafe_allow_html=True)

    # ETF cards
    st.markdown("### Why These Three?")
    cols = st.columns(3)
    for col, t in zip(cols, TICKERS):
        m = ETF_META[t]
        with col:
            st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-top:3px solid {m['color']};border-radius:10px;padding:18px;height:100%;">
    <div style="font-size:1.3rem;font-weight:800;color:{m['color']};font-family:monospace;">{t}</div>
    <div style="font-size:0.78rem;color:#6B7280;margin-bottom:10px;">{m['name']}</div>
    <div style="font-size:0.82rem;color:#374151;line-height:1.6;margin-bottom:12px;">{m['why']}</div>
    <div style="font-size:0.72rem;color:#9CA3AF;">
        <strong style="color:#374151;">Expense:</strong> {m['expense']}<br>
        <strong style="color:#374151;">Focus:</strong> {m['focus']}
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Allocation bar
    st.markdown("### Allocation")
    st.markdown("""
<div style="display:flex;border-radius:8px;overflow:hidden;height:44px;margin-bottom:20px;">
    <div style="flex:33.3;background:#16A34A;display:flex;align-items:center;justify-content:center;color:white;font-size:0.85rem;font-weight:700;">VOO 33%</div>
    <div style="flex:33.3;background:#2563EB;display:flex;align-items:center;justify-content:center;color:white;font-size:0.85rem;font-weight:700;">QQQM 33%</div>
    <div style="flex:33.4;background:#D97706;display:flex;align-items:center;justify-content:center;color:white;font-size:0.85rem;font-weight:700;">SCHD 34%</div>
</div>""", unsafe_allow_html=True)

    # Live data
    st.markdown("### Performance Metrics")
    live_stats = None
    with st.spinner("Loading live data..."):
        try:
            live_data  = load_data()
            live_stats = calc_stats(live_data)
        except Exception:
            pass

    stats = live_stats if live_stats else SAMPLE
    if not live_stats:
        st.info("Showing estimated figures. Live data unavailable.", icon="ℹ️")

    m_cols = st.columns(3)
    for col, t in zip(m_cols, TICKERS):
        s = stats[t]; m = ETF_META[t]
        with col:
            st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;padding:16px;text-align:center;">
    <div style="font-size:1rem;font-weight:700;color:{m['color']};margin-bottom:4px;">{t}</div>
    <div style="font-size:0.7rem;color:#9CA3AF;margin-bottom:12px;">Since {INCEPTION[t][:7]}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div style="background:#F9FAFB;border-radius:6px;padding:8px;">
            <div style="font-size:0.62rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;">Price</div>
            <div style="font-size:1rem;font-weight:700;color:#111827;font-family:monospace;">${s['price']}</div>
        </div>
        <div style="background:#F9FAFB;border-radius:6px;padding:8px;">
            <div style="font-size:0.62rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;">CAGR</div>
            <div style="font-size:1rem;font-weight:700;color:#16A34A;font-family:monospace;">+{s['cagr']}%</div>
        </div>
        <div style="background:#F9FAFB;border-radius:6px;padding:8px;">
            <div style="font-size:0.62rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;">Total Return</div>
            <div style="font-size:1rem;font-weight:700;color:#16A34A;font-family:monospace;">+{int(s['total'])}%</div>
        </div>
        <div style="background:#F9FAFB;border-radius:6px;padding:8px;">
            <div style="font-size:0.62rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;">Volatility</div>
            <div style="font-size:1rem;font-weight:700;color:#111827;font-family:monospace;">{s['vol']}%</div>
        </div>
        <div style="background:#F9FAFB;border-radius:6px;padding:8px;">
            <div style="font-size:0.62rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;">Sharpe</div>
            <div style="font-size:1rem;font-weight:700;color:#2563EB;font-family:monospace;">{s['sharpe']}</div>
        </div>
        <div style="background:#F9FAFB;border-radius:6px;padding:8px;">
            <div style="font-size:0.62rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;">Max DD</div>
            <div style="font-size:1rem;font-weight:700;color:#DC2626;font-family:monospace;">{s['maxdd']}%</div>
        </div>
    </div>
    <div style="margin-top:10px;font-size:0.7rem;color:#9CA3AF;">{s['years']} years of history</div>
</div>""", unsafe_allow_html=True)

    # Charts
    if live_stats:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Performance Charts")
        tab1, tab2 = st.tabs(["Individual (Full History)", "Blended Portfolio"])

        with tab1:
            fig = go.Figure()
            for t in TICKERS:
                s = live_data[t]
                norm = (s / s.iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=norm.index, y=norm.values.round(2),
                    name=f"{t} (since {INCEPTION[t][:7]})",
                    line=dict(color=ETF_META[t]["color"], width=2),
                    hovertemplate=f"<b>{t}</b><br>%{{x|%b %Y}}<br>+%{{y:.1f}}%<extra></extra>",
                ))
            fig.update_layout(
                template="plotly_white", height=380,
                hovermode="x unified", margin=dict(l=10,r=10,t=20,b=10),
                yaxis_title="Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each ETF shown from its own inception. VOO: Sep 2010 · SCHD: Oct 2011 · QQQM: Oct 2020")

        with tab2:
            df    = pd.DataFrame(live_data).dropna()
            norm  = df / df.iloc[0]
            blend = sum(norm[t] * WEIGHTS[t] for t in TICKERS)
            fig2  = go.Figure()
            for t in TICKERS:
                fig2.add_trace(go.Scatter(
                    x=norm.index, y=((norm[t]-1)*100).round(2),
                    name=t, line=dict(color=ETF_META[t]["color"], width=1.5),
                    opacity=0.6,
                ))
            fig2.add_trace(go.Scatter(
                x=blend.index, y=((blend-1)*100).round(2),
                name="Blended (33/33/33)",
                line=dict(color="#111827", width=2.5, dash="dash"),
            ))
            fig2.update_layout(
                template="plotly_white", height=380,
                hovermode="x unified", margin=dict(l=10,r=10,t=20,b=10),
                yaxis_title="Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"Blended portfolio aligned to QQQM inception (Oct 2020) — the common window for all three ETFs.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:14px 18px;font-size:0.78rem;color:#6B7280;line-height:1.7;">
    <strong style="color:#374151;">Disclaimer:</strong> Data via Yahoo Finance. Past performance does not guarantee future results.
    This is not financial advice. The equal-weight allocation is a starting point — adjust to your own risk tolerance and time horizon.
</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(page_title="ETF Portfolio", layout="wide")
    render()
