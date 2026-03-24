"""
InstitutionalEdge — Professional Investment Research Platform
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="InstitutionalEdge — Investment Research",
    page_icon="IE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background-color: #080C10 !important;
    color: #E2E8F0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0C1117 !important;
    border-right: 1px solid #1E2A36;
}
[data-testid="stSidebar"] .block-container { padding-top: 0 !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }
header { display: none !important; }
footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.02em; color: #F1F5F9 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; letter-spacing: -0.01em; color: #E2E8F0 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #CBD5E1 !important; }

[data-testid="stMetric"] {
    background: #0F1923;
    border: 1px solid #1E2A36;
    border-radius: 10px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; color: #F1F5F9 !important; font-family: 'DM Mono', monospace; }

.stButton > button {
    background: #0F1923 !important;
    border: 1px solid #2D3F50 !important;
    color: #94A3B8 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #162030 !important;
    border-color: #3B82F6 !important;
    color: #E2E8F0 !important;
}
.stButton > button[kind="primary"] {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2563EB !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0F1923 !important;
    border: 1px solid #2D3F50 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

[data-testid="stDataFrame"] { border: 1px solid #1E2A36 !important; border-radius: 10px !important; }

.stTabs [data-baseweb="tab-list"] { background: #0C1117; border-bottom: 1px solid #1E2A36; gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom-color: #3B82F6 !important;
    background: transparent !important;
}

hr { border-color: #1E2A36 !important; margin: 1rem 0 !important; }

.ie-card { background: #0F1923; border: 1px solid #1E2A36; border-radius: 12px; padding: 20px 24px; margin-bottom: 12px; }
.ie-badge { display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; }
.badge-strong-buy { background: #052E16; color: #4ADE80; border: 1px solid #166534; }
.badge-buy        { background: #052E16; color: #86EFAC; border: 1px solid #15803D; }
.badge-hold       { background: #1C1917; color: #FCD34D; border: 1px solid #92400E; }
.badge-weak       { background: #1C0A00; color: #FB923C; border: 1px solid #9A3412; }
.badge-avoid      { background: #1C0000; color: #F87171; border: 1px solid #991B1B; }

.score-num { font-size: 2.4rem; font-weight: 800; line-height: 1; font-family: 'DM Mono', monospace; letter-spacing: -0.02em; }
.score-sub { font-size: 0.62rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

.pillar-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.pillar-bar-bg { flex: 1; height: 6px; background: #1E2A36; border-radius: 3px; overflow: hidden; }
.pillar-bar-fill { height: 100%; border-radius: 3px; }
.pillar-label { color: #94A3B8; min-width: 160px; font-size: 0.8rem; }
.pillar-val { color: #CBD5E1; min-width: 28px; text-align: right; font-family: 'DM Mono', monospace; font-size: 0.8rem; }

.module-chip { display: inline-flex; flex-direction: column; align-items: center; background: #0F1923; border: 1px solid #1E2A36; border-radius: 10px; padding: 10px 14px; min-width: 80px; }
.chip-score { font-size: 1.2rem; font-weight: 700; font-family: 'DM Mono', monospace; }
.chip-label { font-size: 0.6rem; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

.home-hero { padding: 40px 0 20px; text-align: center; }
.feature-card { background: #0F1923; border: 1px solid #1E2A36; border-radius: 12px; padding: 20px; height: 100%; }
.feature-icon { font-size: 1.4rem; margin-bottom: 10px; }
.feature-title { font-size: 0.95rem; font-weight: 600; color: #E2E8F0; margin-bottom: 6px; }
.feature-desc { font-size: 0.82rem; color: #64748B; line-height: 1.5; }
.step-pill { display: inline-block; background: #162030; color: #3B82F6; border: 1px solid #1E3A5F; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; margin-bottom: 8px; }
.beginner-tip { background: #0A1628; border: 1px solid #1E3A5F; border-radius: 10px; padding: 14px 18px; margin: 10px 0; font-size: 0.83rem; color: #93C5FD; line-height: 1.6; }
.beginner-tip strong { color: #BFDBFE; }
.section-header { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #1E2A36; }
.section-header-title { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ────────────────────────────────────────────────────────────────────
def score_color(s: float) -> str:
    if s >= 70: return "#4ADE80"
    if s >= 55: return "#86EFAC"
    if s >= 45: return "#FCD34D"
    if s >= 35: return "#FB923C"
    return "#F87171"

def signal_badge(sig: str) -> str:
    cls = {"STRONG BUY":"badge-strong-buy","BUY":"badge-buy","HOLD":"badge-hold","WEAK":"badge-weak","AVOID":"badge-avoid"}.get(sig,"badge-hold")
    return f"<span class='ie-badge {cls}'>{sig}</span>"

def fmt_cap(v):
    if not v: return "—"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def pillar_bar(label, val, color):
    pct = min(max(float(val), 0), 100)
    return f"""<div class='pillar-row'>
        <span class='pillar-label'>{label}</span>
        <div class='pillar-bar-bg'><div class='pillar-bar-fill' style='width:{pct}%;background:{color};'></div></div>
        <span class='pillar-val'>{int(pct)}</span>
    </div>"""

def module_chip(label, val):
    c = score_color(float(val))
    return f"<div class='module-chip'><span class='chip-score' style='color:{c}'>{int(val)}</span><span class='chip-label'>{label}</span></div>"

def run_analysis(ticker):
    from main import analyze_stock
    return analyze_stock(ticker, verbose=False, skip_macro=False)

def section(title, icon=""):
    st.markdown(f"<div class='section-header'><span style='font-size:1rem'>{icon}</span><span class='section-header-title'>{title}</span></div>", unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────────────────────────
for k, v in [("page","Home"),("results",None),("last_ticker",""),("macro_result",None),("crypto_result",None),("pending_ticker","")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 16px 16px;border-bottom:1px solid #1E2A36;margin-bottom:8px;'>
        <div style='font-size:1.1rem;font-weight:800;letter-spacing:-0.02em;color:#F1F5F9;'>InstitutionalEdge</div>
        <div style='font-size:0.7rem;color:#475569;margin-top:3px;letter-spacing:0.05em;text-transform:uppercase;'>Investment Research Platform</div>
    </div>""", unsafe_allow_html=True)

    nav_pages = [("🏠", "Home"), ("🔍", "Stock Analysis"), ("💼", "Portfolio Builder"),
                 ("📈", "ETF Portfolio"), ("🤖", "ML Alpha Signals"), ("🌐", "Macro & Crypto"), ("📊", "Screener")]
    for icon, name in nav_pages:
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
            st.session_state.page = name
            st.rerun()

    st.markdown("<div style='margin-top:16px;padding-top:16px;border-top:1px solid #1E2A36;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;'>Quick Analyze</div>", unsafe_allow_html=True)
    q_ticker = st.text_input("", placeholder="NVDA, AAPL, MSFT...", label_visibility="collapsed", key="quick_ticker").upper().strip()
    if st.button("Analyze Stock", type="primary", use_container_width=True, key="quick_analyze"):
        if q_ticker:
            st.session_state.page = "Stock Analysis"
            st.session_state.pending_ticker = q_ticker
            st.rerun()

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class='home-hero'>
        <div style='font-size:2.2rem;font-weight:800;letter-spacing:-0.03em;color:#F1F5F9;margin-bottom:12px;'>
            Research like an institution.<br>Invest with confidence.
        </div>
        <div style='color:#64748B;font-size:1rem;max-width:520px;margin:0 auto;'>
            InstitutionalEdge gives beginner investors access to the same analytical frameworks used by professional fund managers.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    features = [
        ("🔍","Stock Analysis","Full 12-module breakdown — fundamentals, technicals, insider activity, options flow, risk metrics, and a 12-month price target."),
        ("💼","Portfolio Builder","Answer 10 questions. Get a personalized portfolio with specific tickers, position sizes, and 30-year projections."),
        ("📈","ETF Portfolio","New to investing? Start with our 3-ETF core portfolio and see its full performance history since inception."),
        ("🌐","Macro Analysis","Understand the current economic regime and which sectors to favor right now."),
        ("📊","Screener","Find stocks automatically: GARP, deep value, quality compounder, dividend growth, momentum."),
        ("🤖","ML Signals","Machine learning models rank stocks by risk-adjusted return probability."),
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        col = row1[i] if i < 3 else row2[i-3]
        with col:
            st.markdown(f"<div class='feature-card'><div class='feature-icon'>{icon}</div><div class='feature-title'>{title}</div><div class='feature-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### New to investing? Start here.")
    g1, g2, g3, g4 = st.columns(4)
    steps = [
        ("01","Learn the basics","Visit ETF Portfolio to understand a simple, diversified starting point used by millions of investors."),
        ("02","Research a stock","Type any ticker into Stock Analysis — is it a good buy right now?"),
        ("03","Build your portfolio","Use Portfolio Builder to get a personalized plan based on your goals."),
        ("04","Find new ideas","Use the Screener to discover stocks matching your investment style."),
    ]
    for col, (num, title, desc) in zip([g1,g2,g3,g4], steps):
        with col:
            st.markdown(f"<div class='ie-card'><div class='step-pill'>Step {num}</div><div style='font-size:0.9rem;font-weight:600;color:#E2E8F0;margin-bottom:6px;'>{title}</div><div style='font-size:0.8rem;color:#64748B;line-height:1.5;'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;padding:16px;border-top:1px solid #1E2A36;font-size:0.72rem;color:#334155;'>InstitutionalEdge — For educational purposes only. Not financial advice. All investments carry risk.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Stock Analysis":

    if st.session_state.pending_ticker:
        t = st.session_state.pending_ticker
        st.session_state.pending_ticker = ""
        with st.spinner(f"Analyzing {t}..."):
            try:
                st.session_state.results = run_analysis(t)
                st.session_state.last_ticker = t
            except Exception as e:
                st.error(f"Could not analyze {t}: {e}")

    col_s, col_b, _ = st.columns([2,1,4])
    with col_s:
        ticker_in = st.text_input("", placeholder="Enter ticker (e.g. NVDA)", label_visibility="collapsed", key="main_ticker")
    with col_b:
        go = st.button("Analyze", type="primary", use_container_width=True)

    if go and ticker_in.strip():
        t = ticker_in.strip().upper()
        with st.spinner(f"Running full analysis on {t}..."):
            try:
                st.session_state.results = run_analysis(t)
                st.session_state.last_ticker = t
            except Exception as e:
                st.error(f"Could not analyze {t}: {e}")

    r = st.session_state.results
    if not r:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;padding:60px 20px;'><div style='font-size:2.5rem;margin-bottom:16px;'>🔍</div><div style='font-size:1.1rem;font-weight:600;color:#CBD5E1;margin-bottom:8px;'>Search for any stock</div><div style='font-size:0.9rem;color:#475569;'>Enter a ticker above or click a popular stock below</div></div>", unsafe_allow_html=True)
        st.markdown("**Popular:**")
        p_cols = st.columns(6)
        for col, t in zip(p_cols, ["NVDA","AAPL","MSFT","AMZN","TSLA","GOOGL"]):
            with col:
                if st.button(t, use_container_width=True, key=f"pop_{t}"):
                    with st.spinner(f"Analyzing {t}..."):
                        try:
                            st.session_state.results = run_analysis(t)
                            st.session_state.last_ticker = t
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
    else:
        pt      = r.get("price_target", {})
        fund    = r.get("fundamental", {})
        tech    = r.get("technical", {})
        sent    = r.get("sentiment", {})
        risk    = r.get("risk", {})
        comp    = r.get("competitive", {})
        insider = r.get("insider", {})
        short   = r.get("short", {})
        options = r.get("options", {})
        macro   = r.get("macro", {})
        sims    = r.get("simulations", {})
        mc      = sims.get("monte_carlo", {})
        dcf     = sims.get("dcf", {})
        score   = r.get("composite_score", 0)
        signal  = r.get("buy_signal", "HOLD")
        buffett = r.get("buffett_score", {})

        # Header
        h1, h2, h3, h4, h5 = st.columns([3,1,1,1,1])
        with h1:
            st.markdown(f"<div style='padding:8px 0;'><div style='font-size:1.5rem;font-weight:800;color:#F1F5F9;letter-spacing:-0.02em;'>{r.get('ticker','')} <span style='font-size:1rem;font-weight:400;color:#64748B;'>{r.get('company_name','')}</span></div><div style='font-size:0.78rem;color:#475569;margin-top:3px;'>{r.get('sector','')} · {r.get('industry','')}</div></div>", unsafe_allow_html=True)
        with h2: st.metric("Price", f"${r.get('current_price',0):.2f}")
        with h3: st.metric("Market Cap", fmt_cap(r.get("market_cap",0)))
        with h4:
            c = score_color(score)
            st.markdown(f"<div style='text-align:center;padding:14px;'><div class='score-num' style='color:{c};'>{score:.0f}</div><div class='score-sub'>Composite</div></div>", unsafe_allow_html=True)
        with h5:
            st.markdown(f"<div style='padding:14px;text-align:center;'>{signal_badge(signal)}<div style='font-size:0.68rem;color:#475569;margin-top:6px;text-transform:uppercase;letter-spacing:0.05em;'>Signal</div></div>", unsafe_allow_html=True)

        signal_explain = {
            "STRONG BUY":"Multiple indicators suggest this stock is significantly undervalued with strong fundamentals.",
            "BUY":"Analysis indicates this stock is attractive at the current price with good risk/reward.",
            "HOLD":"Fairly valued. Good to hold if you own it, but not a compelling entry point right now.",
            "WEAK":"Some concerns identified. Risk may outweigh reward at this price.",
            "AVOID":"Multiple red flags across fundamentals, technicals, or valuation. High risk.",
        }.get(signal,"")
        if signal_explain:
            st.markdown(f"<div class='beginner-tip'><strong>What does {signal} mean?</strong> {signal_explain}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Module scores
        section("Score Breakdown","📊")
        score_items = [("Fund.",fund.get("score",0)),("Comp.",comp.get("score",0)),("Tech.",tech.get("score",0)),("Sent.",sent.get("score",0)),("Risk",risk.get("score",0)),("Insider",insider.get("score",0)),("Options",options.get("score",0)),("Short",short.get("score",0))]
        chips_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;'>" + "".join(module_chip(l,v) for l,v in score_items) + "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

        # Conviction score
        if buffett:
            bscore = buffett.get("score", 0)
            bsig   = buffett.get("signal","")
            pillars = buffett.get("pillars",{})
            bc = score_color(bscore)
            section("Conviction Score","🏆")
            cv1, cv2 = st.columns([1,2])
            with cv1:
                st.markdown(f"<div class='ie-card' style='text-align:center;'><div class='score-num' style='color:{bc};font-size:3rem;'>{bscore}</div><div class='score-sub'>/ 100 Conviction</div><div style='margin-top:10px;'>{signal_badge(bsig)}</div></div>", unsafe_allow_html=True)
                if buffett.get("summary"): st.caption(buffett["summary"])
            with cv2:
                if pillars:
                    bars_html = "<div style='padding:8px 0;'>" + "".join(pillar_bar(n,v,score_color(float(v))) for n,v in pillars.items()) + "</div>"
                    st.markdown(bars_html, unsafe_allow_html=True)
                for b in buffett.get("bull_points",[])[:3]: st.success(f"↑ {b}")
                for b in buffett.get("bear_points",[])[:2]: st.warning(f"↓ {b}")

        st.markdown("---")

        # Price target
        if pt.get("price_target_12m"):
            section("12-Month Price Target","🎯")
            pt1,pt2,pt3,pt4 = st.columns(4)
            up = pt.get("upside_downside",0)
            pt1.metric("Target Price",  f"${pt['price_target_12m']:.2f}", f"{up:+.1f}%")
            pt2.metric("Bear Case",     f"${pt.get('confidence_low',0):.2f}")
            pt3.metric("Bull Case",     f"${pt.get('confidence_high',0):.2f}")
            pt4.metric("Conviction",    pt.get("conviction","N/A").split()[0])
            model_rows = [{"Model":m.get("source",n),"Target":f"${m['price']:.2f}","Expected Return":f"{m.get('upside_pct',0):+.1f}%","Confidence":f"{m.get('confidence',0)}%"} for n,m in pt.get("models",{}).items() if m.get("price")]
            if model_rows: st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
            if pt.get("thesis"):
                with st.expander("Investment Thesis"): st.write(pt["thesis"])
            cats, risks2 = pt.get("catalysts",[]), pt.get("risks",[])
            if cats or risks2:
                cc2, rc2 = st.columns(2)
                with cc2:
                    if cats:
                        st.markdown("**Upside Catalysts**")
                        for c in cats: st.success(c)
                with rc2:
                    if risks2:
                        st.markdown("**Key Risks**")
                        for rk in risks2: st.error(rk)
            st.markdown("---")

        # Tabs
        tab1,tab2,tab3,tab4,tab5 = st.tabs(["📋 Fundamentals","📉 Market Data","🎲 Simulations","📰 Sentiment","⚠️ Risk"])

        with tab1:
            f1,f2 = st.columns(2)
            with f1:
                section("Fundamentals","📋")
                highlights = fund.get("highlights",{})
                if highlights: st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in highlights.items()]), use_container_width=True, hide_index=True)
                q = fund.get("quality_score",{})
                if q:
                    st.metric("Piotroski F-Score", f"{q.get('f_score','N/A')}/9", delta=q.get("interpretation",""))
                    st.markdown("<div class='beginner-tip'><strong>Piotroski F-Score</strong> rates financial health 0-9. A score of 7+ means financially strong. 8-9 = excellent quality.</div>", unsafe_allow_html=True)
            with f2:
                section("Competitive Position","🏛️")
                moat = comp.get("moat_score",{})
                st.metric("Economic Moat", moat.get("width","N/A"), delta=f"{moat.get('score',0)}/100")
                st.markdown("<div class='beginner-tip'><strong>Economic Moat</strong> = the company's long-term competitive advantage. Wide moat = hard for competitors to take market share.</div>", unsafe_allow_html=True)
                st.metric("Competitive Position", comp.get("competitive_position","N/A"))
                for a in comp.get("key_advantages",[])[:3]: st.success(a[:90])

        with tab2:
            m1,m2,m3,m4 = st.columns(4)
            with m1:
                section("Short Interest","📉")
                sq_score = short.get("squeeze_score",0)
                sf = short.get("short_float_pct"); dtc = short.get("days_to_cover")
                st.markdown(f"<div style='color:{score_color(sq_score)};font-size:1rem;font-weight:700;'>{short.get('squeeze_risk','N/A')}</div>", unsafe_allow_html=True)
                if sf:  st.metric("Short Float",   f"{sf:.1f}%")
                if dtc: st.metric("Days to Cover", f"{dtc:.1f}")
                st.metric("Squeeze Score", f"{sq_score}/100")
                st.markdown("<div class='beginner-tip'>High short interest with rising price can trigger a squeeze — rapid price spike as short sellers rush to cover.</div>", unsafe_allow_html=True)
            with m2:
                section("Options Flow","⚙️")
                pcr = options.get("put_call_ratio",{}); ivr = options.get("iv_rank",{}); mp = options.get("max_pain",{})
                if pcr.get("pcr_volume"): st.metric("Put/Call Ratio", f"{pcr['pcr_volume']:.2f}"); st.caption(pcr.get("sentiment",""))
                if ivr.get("iv_rank") is not None: st.metric("IV Rank", f"{ivr['iv_rank']:.0f}/100")
                if mp.get("max_pain_price"): st.metric("Max Pain", f"${mp['max_pain_price']:.2f}")
            with m3:
                section("Insider Activity","👤")
                st.markdown(insider.get("signal","N/A"))
                inst = insider.get("institutional",{})
                st.metric("Institutional", f"{inst.get('institutional_ownership_pct',0):.1f}%")
                st.metric("Insider Own.", f"{inst.get('insider_ownership_pct',0):.1f}%")
                st.markdown("<div class='beginner-tip'>When executives buy stock with personal money, that's a bullish signal — they believe the price will rise.</div>", unsafe_allow_html=True)
            with m4:
                section("Technical","📈")
                for k,v in list(tech.get("signals",{}).items())[:4]: st.markdown(f"**{k}:** {v}")
                if tech.get("support"):
                    st.metric("Support",    f"${tech['support']:.2f}")
                    st.metric("Resistance", f"${tech.get('resistance',0):.2f}")

        with tab3:
            s1,s2 = st.columns(2)
            with s1:
                section("Monte Carlo Simulation","🎲")
                if mc:
                    mc1,mc2,mc3 = st.columns(3)
                    mc1.metric("Median 1Y", f"{mc.get('median_return_1y',0)*100:.1f}%")
                    mc2.metric("Bull Case", f"{mc.get('bull_case',0)*100:.1f}%")
                    mc3.metric("Bear Case", f"{mc.get('bear_case',0)*100:.1f}%")
                    st.metric("Probability of Profit", f"{mc.get('probability_profit',0)*100:.0f}%")
                    st.markdown("<div class='beginner-tip'><strong>Monte Carlo</strong> runs thousands of simulated scenarios. The median is the middle expected outcome across all scenarios.</div>", unsafe_allow_html=True)
            with s2:
                section("DCF Valuation","💲")
                if dcf and dcf.get("intrinsic_value"):
                    st.metric("Intrinsic Value", f"${dcf['intrinsic_value']:.2f}")
                    st.metric("Status", dcf.get("valuation_status",""))
                    st.metric("Margin of Safety (20%)", f"${dcf.get('margin_of_safety_20',0):.2f}")
                    st.markdown("<div class='beginner-tip'><strong>Intrinsic value</strong> = what the stock is theoretically worth based on future cash flows. Price below intrinsic value = potentially undervalued.</div>", unsafe_allow_html=True)
            stress = sims.get("stress_test",{})
            if stress and stress.get("scenarios"):
                section("Historical Stress Tests","⚡")
                rows = [{"Scenario":sc,"Est. Drawdown":f"{sd.get('estimated_drawdown_pct',0):.1f}%","Stressed Price":f"${sd.get('estimated_price',0):.2f}"} for sc,sd in stress["scenarios"].items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.markdown("<div class='beginner-tip'>Shows how this stock might perform in historical market crashes (2008, COVID, etc.).</div>", unsafe_allow_html=True)

        with tab4:
            section("News Sentiment","📰")
            sn1,sn2,sn3 = st.columns(3)
            sn1.metric("Sentiment Score", f"{sent.get('score',0):.0f}/100")
            sn2.metric("Bullish Headlines", sent.get("bullish_headlines",0))
            sn3.metric("Bearish Headlines", sent.get("bearish_headlines",0))
            st.caption(sent.get("overall",""))
            for h in sent.get("headlines",[])[:6]: st.markdown(f"— {h}")

        with tab5:
            section("Risk Metrics","⚠️")
            r1,r2,r3,r4,r5,r6 = st.columns(6)
            r1.metric("Sharpe",      f"{risk.get('sharpe_ratio',0):.2f}")
            r2.metric("Sortino",     f"{risk.get('sortino_ratio',0):.2f}")
            r3.metric("Calmar",      f"{risk.get('calmar_ratio',0):.2f}")
            r4.metric("Max Drawdown",f"{risk.get('max_drawdown',0)*100:.1f}%")
            r5.metric("VaR 95%",     f"{risk.get('var_95',0)*100:.2f}%")
            r6.metric("Beta",        f"{risk.get('beta',0):.2f}")
            st.markdown("""<div class='beginner-tip'>
                <strong>Sharpe Ratio</strong> = return per unit of risk (above 1.0 is good) ·
                <strong>Max Drawdown</strong> = worst peak-to-trough decline ·
                <strong>Beta</strong> = market sensitivity (2.0 = moves 2x the market)
            </div>""", unsafe_allow_html=True)
            if macro:
                st.markdown("---")
                section("Macro Context","🌐")
                st.markdown(f"**Regime:** {macro.get('regime','N/A')} — {macro.get('asset_bias','')}")
                st.caption(macro.get("description",""))
                fit = macro.get("sector_fit",{})
                if fit.get("assessment"): st.info(fit["assessment"])


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Builder":
    st.markdown("## 💼 Portfolio Builder")
    st.markdown("<div style='color:#64748B;margin-bottom:20px;'>Answer a few questions and we'll build a personalized investment portfolio with specific tickers, allocation percentages, and long-term projections.</div>", unsafe_allow_html=True)

    with st.form("portfolio_form"):
        st.markdown("### Your Profile")
        c1,c2,c3 = st.columns(3)
        with c1: experience = st.selectbox("Experience Level", ["Beginner — Just getting started","Intermediate — Some experience","Advanced — Active investor"])
        with c2: account    = st.selectbox("Account Type", ["Taxable Brokerage","Roth IRA","Traditional IRA","401(k)"])
        with c3: goal       = st.selectbox("Primary Goal", ["Long-term wealth building","Retirement in 20+ years","Retirement in 10-20 years","Income / Dividends","Short-term growth (5 years)"])

        st.markdown("### Risk Tolerance")
        risk_tol = st.slider("How would you react if your portfolio dropped 30%?", 1, 10, 5, help="1 = Sell everything  |  10 = Buy more aggressively")
        risk_labels = {1:"Sell everything",3:"Very nervous",5:"Hold steady",7:"Buy a little more",10:"Buy aggressively"}
        nearest = min(risk_labels.keys(), key=lambda x: abs(x-risk_tol))
        st.caption(f"Your selection: {risk_labels[nearest]}")

        st.markdown("### Capital & Timeline")
        cap1,cap2,cap3 = st.columns(3)
        with cap1: capital = st.number_input("Starting Capital ($)", min_value=500, max_value=10_000_000, value=10_000, step=500)
        with cap2: monthly = st.number_input("Monthly Contribution ($)", min_value=0, max_value=50_000, value=500, step=100)
        with cap3: horizon = st.selectbox("Investment Horizon", ["5 years","10 years","20 years","30+ years"])

        st.markdown("### Style Preferences")
        pr1,pr2 = st.columns(2)
        with pr1: style = st.selectbox("Investment Style", ["Balanced (growth + stability)","Aggressive growth","Conservative / income","Value focused","Dividend growth"])
        with pr2: sectors = st.multiselect("Preferred Sectors (optional)", ["Technology","Healthcare","Finance","Energy","Consumer Staples","Real Estate","Industrials","No preference"], default=["No preference"])

        submitted = st.form_submit_button("Build My Portfolio", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Building your personalized portfolio..."):
            try:
                from advisor import PortfolioAdvisor
                result = PortfolioAdvisor().build_portfolio_recommendation({
                    "risk_tolerance": risk_tol, "experience": experience,
                    "goal": goal, "account_type": account,
                    "starting_capital": capital, "monthly_contribution": monthly,
                    "time_horizon": horizon, "preferred_sectors": sectors,
                    "investment_style": style,
                })
                st.markdown("---")
                st.markdown("## Your Portfolio")
                profile = result.get("profile",{})
                if profile:
                    st.markdown(f"**Profile:** {profile.get('type','')}")
                    st.info(profile.get("description",""))
                alloc = result.get("allocation",{})
                if alloc:
                    st.markdown("### Asset Allocation")
                    a_cols = st.columns(len(alloc))
                    for col, (asset, pct) in zip(a_cols, alloc.items()): col.metric(asset, f"{pct:.0f}%")
                holdings = result.get("holdings",[])
                if holdings:
                    st.markdown("### Recommended Holdings")
                    rows = [{"Ticker":h.get("ticker",""),"Name":h.get("name",""),"Allocation":f"{h.get('weight',0):.1f}%","Amount":f"${h.get('dollar_amount',0):,.0f}","Category":h.get("category",""),"Rationale":h.get("rationale","")[:60]} for h in holdings]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                proj = result.get("projections",{})
                if proj:
                    st.markdown("### Long-Term Projections")
                    p1,p2,p3 = st.columns(3)
                    p1.metric("10-Year Value", f"${proj.get('10y',0):,.0f}")
                    p2.metric("20-Year Value", f"${proj.get('20y',0):,.0f}")
                    p3.metric("30-Year Value", f"${proj.get('30y',0):,.0f}")
                    st.markdown("<div class='beginner-tip'>These projections assume consistent contributions and historical average returns. The power of compound interest means starting early dramatically increases your final wealth.</div>", unsafe_allow_html=True)
                for w in result.get("warnings",[]): st.warning(w)
            except Exception as e:
                st.error(f"Portfolio builder error: {e}")
                st.info("Make sure advisor.py is in your InstitutionalEdge folder.")


# ══════════════════════════════════════════════════════════════════════════════
#  ETF PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ETF Portfolio":
    try:
        import etf_portfolio_page
        etf_portfolio_page.render()
    except ImportError:
        st.error("etf_portfolio_page.py not found. Make sure it's in your InstitutionalEdge folder.")


# ══════════════════════════════════════════════════════════════════════════════
#  MACRO & CRYPTO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Macro & Crypto":
    tab_macro, tab_crypto = st.tabs(["🌐 Macro Regime","₿ Crypto"])

    with tab_macro:
        st.markdown("## 🌐 Macro Regime Analysis")
        if st.button("Run Macro Analysis", type="primary"):
            with st.spinner("Fetching macro data..."):
                try:
                    from macro_regime import MacroRegimeModel
                    st.session_state.macro_result = MacroRegimeModel().full_analysis()
                except Exception as e:
                    st.error(f"Macro error: {e}")

        macro = st.session_state.get("macro_result")
        if macro:
            st.markdown(f"### Regime: {macro.get('regime','N/A')}")
            st.info(f"{macro.get('description','')} · Bias: **{macro.get('asset_bias','')}**")
            fc,ac = st.columns(2)
            with fc:
                st.markdown("**Favored Sectors**")
                for s in macro.get("favored_sectors",[]): st.success(s)
            with ac:
                st.markdown("**Avoid Sectors**")
                for s in macro.get("avoid_sectors",[]): st.error(s)
            rot = macro.get("sector_rotation",{})
            if rot:
                st.markdown("### Sector Rotation")
                rows = [{"Sector":sec,"ETF":d.get("etf",""),"1M":f"{d.get('1m_return',0):+.1f}%","3M":f"{d.get('3m_return',0):+.1f}%","6M":f"{d.get('6m_return',0):+.1f}%","Signal":d.get("signal","")} for sec,d in sorted(rot.items(),key=lambda x:x[1].get("momentum_score",0),reverse=True)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            ind = macro.get("indicators",{})
            yc = ind.get("yield_curve",{}); vix = ind.get("vix",{}); mkt = ind.get("market_trend",{}); infl = ind.get("inflation",{})
            if any([yc,vix,mkt,infl]):
                st.markdown("### Key Indicators")
                ic = st.columns(4)
                if yc:   ic[0].metric("10Y-3M Spread", f"{yc.get('spread_10y_3m',0):.2f}%", delta="Inverted" if yc.get("inverted") else "Normal")
                if vix:  ic[1].metric("VIX", f"{vix.get('current',0):.1f}", delta=vix.get("regime",""))
                if mkt:  ic[2].metric("SPY 1Y", f"{mkt.get('1y_return',0):+.1f}%", delta=mkt.get("trend",""))
                if infl: ic[3].metric("Inflation", infl.get("inflation_trend",""), delta=infl.get("signal","")[:30])

    with tab_crypto:
        st.markdown("## ₿ Crypto Analysis")
        cc1,cc2 = st.columns([2,1])
        with cc1: crypto_in = st.text_input("Coin ID", value="bitcoin", placeholder="bitcoin, ethereum, solana...")
        with cc2: crypto_go = st.button("Analyze", type="primary", use_container_width=True, key="crypto_go")
        if crypto_go:
            with st.spinner(f"Analyzing {crypto_in}..."):
                try:
                    from crypto_analyzer import CryptoAnalyzer
                    st.session_state.crypto_result = CryptoAnalyzer().full_analysis(crypto_in)
                except Exception as e:
                    st.error(f"Crypto error: {e}")
        cr = st.session_state.get("crypto_result")
        if cr:
            market = cr.get("market",{}); mc_c = cr.get("simulations",{}).get("monte_carlo",{})
            st.markdown(f"### {market.get('name',crypto_in)} ({market.get('symbol','').upper()})")
            cx1,cx2,cx3,cx4 = st.columns(4)
            price = market.get("current_price")
            cx1.metric("Price",      f"${price:,.2f}" if price else "N/A")
            cx2.metric("Market Cap", fmt_cap(market.get("market_cap")))
            cx3.metric("Rank",       f"#{market.get('rank','N/A')}")
            cx4.metric("Tier",       market.get("tier","N/A"))
            cp1,cp2,cp3 = st.columns(3)
            cp1.metric("24H", f"{market.get('price_change_24h',0):.2f}%")
            cp2.metric("7D",  f"{market.get('price_change_7d',0):.2f}%")
            cp3.metric("30D", f"{market.get('price_change_30d',0):.2f}%")
            if mc_c:
                st.markdown("### Monte Carlo (1 Year)")
                cm1,cm2,cm3,cm4 = st.columns(4)
                cm1.metric("Median",       f"{mc_c.get('median_return_1y',0)*100:.1f}%")
                cm2.metric("Bull Case",    f"{mc_c.get('bull_case',0)*100:.1f}%")
                cm3.metric("Bear Case",    f"{mc_c.get('bear_case',0)*100:.1f}%")
                cm4.metric("Prob. Profit", f"{mc_c.get('probability_profit',0)*100:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  ML ALPHA SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Alpha Signals":
    st.markdown("## 🤖 ML Alpha Signals")
    st.markdown("""<div style='color:#64748B;margin-bottom:16px;'>
        Stacked ensemble of 6 ML models (Random Forest, Gradient Boosting, XGBoost, LightGBM, ExtraTrees, Neural Network)
        trained on 125+ fundamental, technical, and alternative data features.
        Predicts which stocks will outperform the S&P 500 by 10%+ over the next 6 months.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='beginner-tip'><strong>How to read this:</strong> The model scores every stock in its universe and ranks them by probability of outperforming the market. The higher the score, the more the model believes the stock will beat the S&P 500. It also detects the current market regime (bull, bear, sideways) and adjusts recommendations accordingly.</div>", unsafe_allow_html=True)

    ml1, ml2 = st.columns([3,1])
    with ml1:
        st.markdown("First run takes 10–20 minutes to train. After that the model saves locally and loads in seconds.")
    with ml2:
        top_n = st.number_input("Show top N picks", min_value=5, max_value=50, value=25)

    run_ml = st.button("🚀 Run Alpha Model", type="primary", use_container_width=True)

    if run_ml:
        try:
            from alpha_model import AlphaModel
            model = AlphaModel(target="label_6m_10pct")
            saved = os.path.join("models", "alpha_v1.pkl")

            if os.path.exists(saved):
                with st.spinner("Loading trained model..."):
                    model.load("alpha_v1")
                with st.spinner("Scoring stocks..."):
                    signals = model.generate_signals(top_n=top_n, verbose=False)
            else:
                progress = st.empty()
                progress.info("Building universe — extracting features for ~170 stocks...")
                df = model.build_universe()
                if df is None or df.empty:
                    st.error("Could not build stock universe. Check your internet connection.")
                    st.stop()
                progress.info("Training ensemble (6 models)... this takes ~15 minutes on first run")
                model.train(verbose=False)
                progress.info("Running walk-forward validation...")
                val = model.validate(verbose=False)
                progress.info("Generating signals...")
                signals = model.generate_signals(top_n=top_n, verbose=False)
                model.save("alpha_v1")
                progress.success("Model trained and saved. Future runs load instantly.")

            if signals:
                # Regime banner
                regime_info = signals[0] if signals else {}
                regime = regime_info.get("regime", "UNKNOWN")
                regime_color = {"BULL":"#4ADE80","BEAR":"#F87171","SIDEWAYS":"#FCD34D"}.get(regime,"#94A3B8")
                st.markdown(f"""<div style='background:#0F1923;border:1px solid #1E2A36;border-radius:10px;padding:14px 20px;margin:12px 0;display:flex;gap:24px;align-items:center;'>
                    <div><span style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;'>Market Regime</span><br>
                    <span style='font-size:1.2rem;font-weight:700;color:{regime_color};'>{regime}</span></div>
                    <div><span style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;'>Stocks Scored</span><br>
                    <span style='font-size:1.2rem;font-weight:700;color:#E2E8F0;'>{regime_info.get("universe_size",0)}</span></div>
                    <div><span style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;'>Buy Signals</span><br>
                    <span style='font-size:1.2rem;font-weight:700;color:#4ADE80;'>{regime_info.get("buy_signals",0)}</span></div>
                </div>""", unsafe_allow_html=True)

                # Backtest results
                val_data = getattr(model, "validation_results", {})
                if val_data:
                    section("Walk-Forward Backtest Results", "📊")
                    v1,v2,v3,v4 = st.columns(4)
                    v1.metric("Avg Alpha",    f"+{val_data.get('avg_alpha',0)*100:.1f}%")
                    v2.metric("Hit Rate",     f"{val_data.get('hit_rate',0)*100:.0f}%")
                    v3.metric("Alpha Sharpe", f"{val_data.get('alpha_sharpe',0):.2f}")
                    v4.metric("CV Folds",     val_data.get("n_folds",5))
                    st.markdown("<div class='beginner-tip'><strong>Hit Rate</strong> = % of picks that beat the S&P 500. <strong>Alpha Sharpe</strong> = risk-adjusted outperformance. Above 1.0 is excellent.</div>", unsafe_allow_html=True)

                # Top picks table
                section(f"Top {len(signals)} Picks", "🏆")
                rows = []
                for i, s in enumerate(signals):
                    rows.append({
                        "Rank":         i + 1,
                        "Ticker":       s.get("ticker",""),
                        "Probability":  f"{s.get('probability',0):.3f}",
                        "Alpha Score":  f"{s.get('alpha_score',0):.3f}",
                        "Risk-Adj":     f"{s.get('risk_adj_score',0):.3f}",
                        "Position":     f"{s.get('position_size',0)*100:.1f}%",
                        "Confidence":   s.get("confidence_tier",""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Feature importance
                fi = getattr(model, "feature_importance_", None)
                if fi is not None and len(fi) > 0:
                    section("Top Features (What the Model Learned)", "🧠")
                    try:
                        fi_df = pd.DataFrame(fi).head(15) if isinstance(fi, list) else fi.head(15)
                        st.dataframe(fi_df, use_container_width=True, hide_index=True)
                    except Exception:
                        pass
                    st.markdown("<div class='beginner-tip'>These are the data signals the model found most predictive. Momentum, price vs moving averages, and valuation metrics tend to be the top drivers.</div>", unsafe_allow_html=True)

        except ImportError:
            st.error("alpha_model.py not found in your InstitutionalEdge folder.")
        except Exception as e:
            st.error(f"ML model error: {e}")
            st.info("Make sure scikit-learn, xgboost, and lightgbm are installed: pip install scikit-learn xgboost lightgbm")


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Screener":
    st.markdown("## 📊 Stock Screener")
    stab1,stab2 = st.tabs(["🔭 Strategy Screener","📋 Watchlist Ranker"])

    with stab1:
        sc1,sc2 = st.columns([2,1])
        with sc1:
            profile = st.selectbox("Select Strategy", ["garp","aggressive_growth","deep_value","quality_compounder","dividend_growth","momentum"],
                format_func=lambda x: {"garp":"GARP — Growth at a Reasonable Price","aggressive_growth":"Aggressive Growth","deep_value":"Deep Value — Undervalued stocks","quality_compounder":"Quality Compounder — Wide moat","dividend_growth":"Dividend Growth — Income + appreciation","momentum":"Momentum — Price leaders"}[x])
        with sc2:
            screen_go = st.button("Screen Now", type="primary", use_container_width=True)
        descriptions = {"garp":"Finds companies growing faster than the market but not at bubble valuations.","aggressive_growth":"Targets highest-growth companies. Higher risk, higher potential reward.","deep_value":"Stocks trading well below intrinsic value. Classic bargain hunting.","quality_compounder":"Wide moat companies with durable competitive advantages.","dividend_growth":"Companies with growing dividends and strong balance sheets.","momentum":"Stocks with strong recent price performance."}
        st.markdown(f"<div class='beginner-tip'>{descriptions[profile]}</div>", unsafe_allow_html=True)
        if screen_go:
            with st.spinner(f"Screening for {profile} stocks..."):
                try:
                    from main import run_screener
                    results = run_screener(profile=profile)
                    if results:
                        st.markdown(f"### Results ({len(results)} stocks found)")
                        rows = [{"Ticker":s.get("ticker",""),"Company":s.get("company","")[:30],"Score":s.get("composite_score",0),"Signal":s.get("buy_signal",""),"Target":f"${s['price_target_12m']:.2f}" if s.get("price_target_12m") else "N/A","Upside":f"{s['upside_pct']:.1f}%" if s.get("upside_pct") is not None else "N/A","Moat":s.get("moat","N/A")} for s in results]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else: st.warning("No results. Try a different strategy.")
                except Exception as e: st.error(f"Screener error: {e}")

    with stab2:
        wl_input = st.text_area("Tickers (one per line)", value="NVDA\nAAPL\nMSFT\nSMCI\nAMZN", height=150)
        wl_go = st.button("Rank Watchlist", type="primary", use_container_width=True)
        if wl_go:
            tickers = [t.strip().upper() for t in wl_input.split("\n") if t.strip()]
            if tickers:
                with st.spinner(f"Analyzing {len(tickers)} stocks..."):
                    try:
                        from main import generate_watchlist
                        ranked = generate_watchlist(tickers)
                        st.markdown(f"### Rankings — {len(ranked)} Stocks")
                        rows = [{"Rank":i+1,"Ticker":s["ticker"],"Company":s.get("company","")[:25],"Score":s["composite_score"],"Signal":s["buy_signal"],"Target":f"${s['price_target_12m']:.2f}" if s.get("price_target_12m") else "N/A","Upside":f"{s['upside_pct']:.1f}%" if s.get("upside_pct") is not None else "N/A","Moat":s.get("moat","N/A")} for i,s in enumerate(ranked)]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception as e: st.error(f"Watchlist error: {e}")
            else: st.warning("Enter at least one ticker.")
