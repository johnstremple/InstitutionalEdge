"""
InstitutionalEdge — Professional Investment Research Platform
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="InstitutionalEdge",
    page_icon="IE",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, .stApp { background:#F7F8FA !important; color:#111827; font-family:'Inter',sans-serif; }
header[data-testid="stHeader"] { display:none !important; }
footer { display:none !important; }
[data-testid="stDecoration"] { display:none !important; }
#MainMenu { display:none !important; }

[data-testid="stSidebar"] {
    background:#FFFFFF !important;
    border-right:1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top:0 !important; }

.block-container { padding:2rem 2.5rem !important; max-width:1280px !important; }

h1 { font-size:1.6rem !important; font-weight:800 !important; color:#111827 !important; letter-spacing:-0.02em; }
h2 { font-size:1.2rem !important; font-weight:700 !important; color:#1F2937 !important; }
h3 { font-size:0.95rem !important; font-weight:600 !important; color:#374151 !important; }

[data-testid="stMetric"] {
    background:#FFFFFF; border:1px solid #E5E7EB;
    border-radius:10px; padding:14px 16px !important;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] { font-size:0.68rem !important; font-weight:600 !important; color:#6B7280 !important; text-transform:uppercase; letter-spacing:0.07em; }
[data-testid="stMetricValue"] { font-size:1.4rem !important; font-weight:700 !important; color:#111827 !important; font-family:'IBM Plex Mono',monospace; }

.stButton > button {
    background:#FFFFFF !important; border:1px solid #D1D5DB !important;
    color:#374151 !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important; font-size:0.84rem !important;
    font-weight:500 !important; transition:all 0.15s !important;
    box-shadow:0 1px 2px rgba(0,0,0,0.04) !important;
}
.stButton > button:hover { border-color:#16A34A !important; color:#15803D !important; background:#F0FDF4 !important; }
.stButton > button[kind="primary"] {
    background:#16A34A !important; border-color:#16A34A !important;
    color:#FFFFFF !important; font-weight:600 !important;
    box-shadow:0 1px 3px rgba(22,163,74,0.3) !important;
}
.stButton > button[kind="primary"]:hover { background:#15803D !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background:#FFFFFF !important; border:1px solid #D1D5DB !important;
    color:#111827 !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important; font-size:0.875rem !important;
}
.stTextInput > div > div > input:focus { border-color:#16A34A !important; box-shadow:0 0 0 3px rgba(22,163,74,0.1) !important; }

.stTabs [data-baseweb="tab-list"] { background:transparent; border-bottom:2px solid #E5E7EB; gap:0; }
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#6B7280 !important;
    border:none !important; border-bottom:2px solid transparent !important;
    margin-bottom:-2px !important; font-size:0.84rem !important;
    font-weight:500 !important; padding:10px 18px !important;
}
.stTabs [aria-selected="true"] { color:#16A34A !important; border-bottom-color:#16A34A !important; font-weight:600 !important; }

[data-testid="stDataFrame"] { border:1px solid #E5E7EB !important; border-radius:10px !important; background:#FFFFFF; }
hr { border-color:#E5E7EB !important; margin:1.25rem 0 !important; }

.stSelectbox > div > div > div {
    background:#FFFFFF !important; border:1px solid #D1D5DB !important;
    color:#111827 !important; border-radius:8px !important;
}

/* ── Custom components ── */
.ie-brand { padding:22px 20px 16px; border-bottom:1px solid #F3F4F6; margin-bottom:6px; }
.ie-brand-name { font-size:0.95rem; font-weight:800; color:#111827; letter-spacing:-0.01em; }
.ie-brand-sub  { font-size:0.65rem; color:#9CA3AF; margin-top:2px; text-transform:uppercase; letter-spacing:0.07em; }
.ie-nav-label  { font-size:0.63rem; font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em; padding:10px 20px 4px; }

.ie-card { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:20px 24px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.ie-tip  { background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:12px 16px; margin:10px 0; font-size:0.82rem; color:#166534; line-height:1.6; }
.ie-tip strong { color:#14532D; }

.ie-badge { display:inline-block; padding:3px 10px; border-radius:5px; font-size:0.74rem; font-weight:700; letter-spacing:0.04em; }
.badge-strong-buy { background:#DCFCE7; color:#15803D; border:1px solid #86EFAC; }
.badge-buy        { background:#DCFCE7; color:#16A34A; border:1px solid #A7F3D0; }
.badge-hold       { background:#FEF9C3; color:#854D0E; border:1px solid #FDE68A; }
.badge-weak       { background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; }
.badge-avoid      { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }

.ie-score { font-size:2.8rem; font-weight:800; line-height:1; font-family:'IBM Plex Mono',monospace; letter-spacing:-0.02em; }
.ie-score-lbl { font-size:0.62rem; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px; }

.ie-chip { display:inline-flex; flex-direction:column; align-items:center; background:#FFFFFF; border:1px solid #E5E7EB; border-radius:10px; padding:10px 14px; min-width:74px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.ie-chip-val { font-size:1.15rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.ie-chip-lbl { font-size:0.58rem; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.06em; margin-top:2px; }

.ie-pillar { display:flex; align-items:center; gap:10px; margin:5px 0; }
.ie-pillar-lbl { color:#6B7280; min-width:165px; font-size:0.79rem; }
.ie-pillar-bg  { flex:1; height:5px; background:#F3F4F6; border-radius:3px; overflow:hidden; }
.ie-pillar-fill { height:100%; border-radius:3px; }
.ie-pillar-val { color:#374151; min-width:26px; text-align:right; font-family:'IBM Plex Mono',monospace; font-size:0.77rem; }

.ie-section { font-size:0.7rem; font-weight:600; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.08em; border-bottom:1px solid #F3F4F6; padding-bottom:8px; margin-bottom:14px; }

.ie-feature { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:20px; height:100%; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.ie-feature-tag   { display:inline-block; background:#F0FDF4; color:#16A34A; border:1px solid #BBF7D0; border-radius:4px; font-size:0.63rem; font-weight:600; padding:2px 7px; margin-bottom:8px; letter-spacing:0.04em; }
.ie-feature-title { font-size:0.9rem; font-weight:700; color:#111827; margin-bottom:5px; }
.ie-feature-desc  { font-size:0.79rem; color:#6B7280; line-height:1.55; }

.ie-step { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:18px 20px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.ie-step-num   { font-size:0.65rem; font-weight:700; color:#16A34A; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:20px; padding:2px 8px; display:inline-block; margin-bottom:8px; }
.ie-step-title { font-size:0.86rem; font-weight:600; color:#111827; margin-bottom:5px; }
.ie-step-desc  { font-size:0.78rem; color:#6B7280; line-height:1.5; }

.ie-stat-row { display:flex; align-items:center; justify-content:space-between; padding:8px 0; border-bottom:1px solid #F9FAFB; }
.ie-stat-lbl { font-size:0.72rem; color:#9CA3AF; }
.ie-stat-val { font-size:0.88rem; font-weight:600; font-family:'IBM Plex Mono',monospace; color:#111827; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def sc(s):
    if s >= 70: return "#16A34A"
    if s >= 55: return "#22C55E"
    if s >= 45: return "#CA8A04"
    if s >= 35: return "#D97706"
    return "#DC2626"

def badge(sig):
    cls = {"STRONG BUY":"badge-strong-buy","BUY":"badge-buy","HOLD":"badge-hold","WEAK":"badge-weak","AVOID":"badge-avoid"}.get(sig,"badge-hold")
    return f"<span class='ie-badge {cls}'>{sig}</span>"

def fmt_cap(v):
    if not v: return "—"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def chip(lbl, val):
    c = sc(float(val))
    return f"<div class='ie-chip'><span class='ie-chip-val' style='color:{c}'>{int(val)}</span><span class='ie-chip-lbl'>{lbl}</span></div>"

def pillar_bar(lbl, val):
    c = sc(float(val)); pct = min(max(float(val),0),100)
    return f"<div class='ie-pillar'><span class='ie-pillar-lbl'>{lbl}</span><div class='ie-pillar-bg'><div class='ie-pillar-fill' style='width:{pct}%;background:{c};'></div></div><span class='ie-pillar-val'>{int(pct)}</span></div>"

def section(title):
    st.markdown(f"<div class='ie-section'>{title}</div>", unsafe_allow_html=True)

def tip(text):
    st.markdown(f"<div class='ie-tip'>{text}</div>", unsafe_allow_html=True)

def run_analysis(ticker):
    from main import analyze_stock
    return analyze_stock(ticker, verbose=False, skip_macro=False)


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("page","Home"),("results",None),("last_ticker",""),
             ("macro_result",None),("crypto_result",None),("pending_ticker","")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='ie-brand'><div class='ie-brand-name'>InstitutionalEdge</div><div class='ie-brand-sub'>Investment Research Platform</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='ie-nav-label'>Navigation</div>", unsafe_allow_html=True)

    for label, key in [("Home","Home"),("Stock Analysis","Stock Analysis"),
                       ("Portfolio Builder","Portfolio Builder"),("ETF Portfolio","ETF Portfolio"),
                       ("ML Alpha Signals","ML Alpha Signals"),("Macro & Crypto","Macro & Crypto"),
                       ("Screener","Screener")]:
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown("<div class='ie-nav-label'>Quick Analyze</div>", unsafe_allow_html=True)
    q = st.text_input("", placeholder="NVDA, AAPL, MSFT...", label_visibility="collapsed", key="qt")
    if st.button("Analyze", type="primary", use_container_width=True, key="qa"):
        if q.strip():
            st.session_state.pending_ticker = q.strip().upper()
            st.session_state.page = "Stock Analysis"
            st.rerun()
    st.markdown("<div style='padding-top:12px;font-size:0.68rem;color:#9CA3AF;line-height:1.6;'>For educational purposes only.<br>Not financial advice.</div>", unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div style='padding:32px 0 24px;'>
        <div style='display:inline-block;background:#F0FDF4;color:#16A34A;border:1px solid #BBF7D0;border-radius:20px;font-size:0.75rem;font-weight:600;padding:4px 12px;margin-bottom:16px;letter-spacing:0.04em;'>
            Institutional-Grade Research
        </div>
        <div style='font-size:2.2rem;font-weight:800;color:#111827;letter-spacing:-0.03em;line-height:1.18;margin-bottom:14px;'>
            Invest without blind spots.
        </div>
        <div style='font-size:1rem;color:#6B7280;max-width:500px;line-height:1.7;'>
            InstitutionalEdge gives every investor access to the same analytical frameworks used by professional fund managers — for free.
        </div>
    </div>
    """, unsafe_allow_html=True)

    r1 = st.columns(3)
    r2 = st.columns(3)
    features = [
        ("Stock Analysis",    "Full 12-module breakdown — fundamentals, technicals, insider activity, options flow, and a 12-month price target with conviction score.", "12 Modules"),
        ("Portfolio Builder",  "Answer 10 questions. Get a personalized portfolio with specific tickers, position sizes, and 30-year projections.", "Personalized"),
        ("ETF Portfolio",      "New to investing? Start with a core 3-ETF portfolio — VOO, QQQM, and SCHD — with full performance history since inception.", "Beginner"),
        ("ML Alpha Signals",   "6-model ensemble trained on 125+ features. Predicts which stocks will outperform the S&P 500 over the next 6 months.", "Machine Learning"),
        ("Macro Analysis",     "Detect the current economic regime — which sectors to favor, where to avoid, and how the cycle affects your holdings.", "Real-time"),
        ("Screener",           "Find stocks automatically using proven strategies: GARP, deep value, quality compounder, dividend growth, and momentum.", "7 Strategies"),
    ]
    for i, (title, desc, tag) in enumerate(features):
        col = r1[i] if i < 3 else r2[i-3]
        with col:
            st.markdown(f"<div class='ie-feature'><div class='ie-feature-tag'>{tag}</div><div class='ie-feature-title'>{title}</div><div class='ie-feature-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.1rem;font-weight:700;color:#111827;margin-bottom:14px;'>New to investing? Start here.</div>", unsafe_allow_html=True)
    sc_cols = st.columns(4)
    steps = [("01","Learn the basics","Visit ETF Portfolio to understand a simple, diversified starting point used by millions of investors."),
             ("02","Research a stock","Enter any ticker into Stock Analysis — get a full breakdown and find out if it's worth buying."),
             ("03","Build your portfolio","Use Portfolio Builder to get a personalized allocation based on your goals and timeline."),
             ("04","Find new ideas","Use the Screener to automatically discover stocks matching your investment style.")]
    for col, (num, title, desc) in zip(sc_cols, steps):
        with col:
            st.markdown(f"<div class='ie-step'><div class='ie-step-num'>Step {num}</div><div class='ie-step-title'>{title}</div><div class='ie-step-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:40px;padding-top:20px;border-top:1px solid #E5E7EB;font-size:0.72rem;color:#9CA3AF;'>InstitutionalEdge — For educational purposes only. Not financial advice. All investments carry risk of loss.</div>", unsafe_allow_html=True)


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
                st.error(str(e))

    cs, cb, _ = st.columns([2,1,4])
    with cs: ticker_in = st.text_input("", placeholder="Enter ticker symbol", label_visibility="collapsed", key="main_ticker")
    with cb: go = st.button("Analyze", type="primary", use_container_width=True)

    if go and ticker_in.strip():
        t = ticker_in.strip().upper()
        with st.spinner(f"Analyzing {t}..."):
            try:
                st.session_state.results = run_analysis(t)
                st.session_state.last_ticker = t
            except Exception as e:
                st.error(str(e))

    r = st.session_state.results
    if not r:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;padding:60px 20px;'><div style='font-size:1.1rem;font-weight:600;color:#374151;margin-bottom:8px;'>Enter a ticker symbol to begin</div><div style='font-size:0.88rem;color:#9CA3AF;'>Try NVDA, AAPL, MSFT, AMZN, or any U.S. public company</div></div>", unsafe_allow_html=True)
        st.markdown("**Quick picks:**")
        pc = st.columns(6)
        for col, t in zip(pc, ["NVDA","AAPL","MSFT","AMZN","TSLA","GOOGL"]):
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
        pt      = r.get("price_target",{})
        fund    = r.get("fundamental",{})
        tech    = r.get("technical",{})
        sent    = r.get("sentiment",{})
        risk    = r.get("risk",{})
        comp    = r.get("competitive",{})
        insider = r.get("insider",{})
        short   = r.get("short",{})
        options = r.get("options",{})
        macro   = r.get("macro",{})
        sims    = r.get("simulations",{})
        mc      = sims.get("monte_carlo",{})
        dcf     = sims.get("dcf",{})
        score   = r.get("composite_score",0)
        signal  = r.get("buy_signal","HOLD")
        buffett = r.get("buffett_score",{})

        # Header row
        h1,h2,h3,h4,h5 = st.columns([3,1,1,1,1])
        with h1:
            st.markdown(f"<div style='padding:8px 0;'><div style='font-size:1.5rem;font-weight:800;color:#111827;letter-spacing:-0.02em;'>{r.get('ticker','')} <span style='font-size:0.95rem;font-weight:400;color:#9CA3AF;'>{r.get('company_name','')}</span></div><div style='font-size:0.76rem;color:#6B7280;margin-top:3px;'>{r.get('sector','')} &nbsp;·&nbsp; {r.get('industry','')}</div></div>", unsafe_allow_html=True)
        with h2: st.metric("Price", f"${r.get('current_price',0):.2f}")
        with h3: st.metric("Market Cap", fmt_cap(r.get("market_cap",0)))
        with h4:
            c = sc(score)
            st.markdown(f"<div style='text-align:center;padding:14px 0;'><div class='ie-score' style='color:{c};'>{score:.0f}</div><div class='ie-score-lbl'>Composite</div></div>", unsafe_allow_html=True)
        with h5:
            st.markdown(f"<div style='text-align:center;padding:18px 0;'>{badge(signal)}<div style='font-size:0.64rem;color:#9CA3AF;margin-top:6px;text-transform:uppercase;letter-spacing:0.06em;'>Signal</div></div>", unsafe_allow_html=True)

        explain = {"STRONG BUY":"Multiple indicators suggest this stock is significantly undervalued with strong fundamentals — high conviction opportunity.",
                   "BUY":"Analysis indicates the stock is attractively priced with favorable risk/reward at current levels.",
                   "HOLD":"Fairly valued at current price. Good to hold if you own it, but not a compelling new entry point.",
                   "WEAK":"Some concerns identified. Risk may outweigh reward — consider waiting for a better entry.",
                   "AVOID":"Multiple red flags across fundamentals, technicals, or valuation. High risk of loss."}.get(signal,"")
        if explain:
            tip(f"<strong>What does {signal} mean?</strong> {explain}")

        st.markdown("---")
        section("Score Breakdown")
        score_items = [("Fund.",fund.get("score",0)),("Comp.",comp.get("score",0)),("Tech.",tech.get("score",0)),
                       ("Sent.",sent.get("score",0)),("Risk",risk.get("score",0)),("Insider",insider.get("score",0)),
                       ("Options",options.get("score",0)),("Short",short.get("score",0))]
        st.markdown("<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:4px;'>" + "".join(chip(l,v) for l,v in score_items) + "</div>", unsafe_allow_html=True)

        # Conviction score
        if buffett:
            st.markdown("---")
            section("Conviction Score")
            cv1, cv2 = st.columns([1,2])
            bscore = buffett.get("score",0)
            bc = sc(bscore)
            with cv1:
                st.markdown(f"<div class='ie-card' style='text-align:center;'><div class='ie-score' style='color:{bc};font-size:3rem;'>{bscore}</div><div class='ie-score-lbl'>/ 100 Conviction</div><div style='margin-top:12px;'>{badge(buffett.get('signal',''))}</div></div>", unsafe_allow_html=True)
                if buffett.get("summary"): st.caption(buffett["summary"])
            with cv2:
                pillars = buffett.get("pillars",{})
                if pillars:
                    st.markdown("<div style='padding:8px 0;'>" + "".join(pillar_bar(n,v) for n,v in pillars.items()) + "</div>", unsafe_allow_html=True)
                for b in buffett.get("bull_points",[])[:3]: st.success(f"+ {b}")
                for b in buffett.get("bear_points",[])[:2]: st.warning(f"- {b}")

        st.markdown("---")

        # Price target
        if pt.get("price_target_12m"):
            section("12-Month Price Target")
            pa,pb,pc2,pd2 = st.columns(4)
            up = pt.get("upside_downside",0)
            pa.metric("Target Price",  f"${pt['price_target_12m']:.2f}", f"{up:+.1f}%")
            pb.metric("Bear Case",     f"${pt.get('confidence_low',0):.2f}")
            pc2.metric("Bull Case",    f"${pt.get('confidence_high',0):.2f}")
            pd2.metric("Conviction",   pt.get("conviction","N/A").split()[0])
            mrows = [{"Model":m.get("source",n),"Target":f"${m['price']:.2f}","Expected Return":f"{m.get('upside_pct',0):+.1f}%","Confidence":f"{m.get('confidence',0)}%"} for n,m in pt.get("models",{}).items() if m.get("price")]
            if mrows: st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)
            if pt.get("thesis"):
                with st.expander("Investment Thesis"): st.write(pt["thesis"])
            cats, risks2 = pt.get("catalysts",[]), pt.get("risks",[])
            if cats or risks2:
                cc2, rc2 = st.columns(2)
                with cc2:
                    if cats: st.markdown("**Upside Catalysts**"); [st.success(c) for c in cats]
                with rc2:
                    if risks2: st.markdown("**Key Risks**"); [st.error(rk) for rk in risks2]
            st.markdown("---")

        # Detail tabs
        tab1,tab2,tab3,tab4,tab5 = st.tabs(["Fundamentals","Market Data","Simulations","Sentiment","Risk"])

        with tab1:
            f1,f2 = st.columns(2)
            with f1:
                section("Financial Health")
                hi = fund.get("highlights",{})
                if hi: st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in hi.items()]), use_container_width=True, hide_index=True)
                q2 = fund.get("quality_score",{})
                if q2:
                    st.metric("Piotroski F-Score", f"{q2.get('f_score','N/A')}/9", delta=q2.get("interpretation",""))
                    tip("<strong>Piotroski F-Score</strong> rates financial health from 0–9. A score of 7+ means the company is financially strong. 8–9 is excellent.")
            with f2:
                section("Competitive Position")
                moat = comp.get("moat_score",{})
                st.metric("Economic Moat", moat.get("width","N/A"), delta=f"{moat.get('score',0)}/100")
                tip("<strong>Economic Moat</strong> = the company's long-term competitive advantage. Wide moat = hard for competitors to take market share.")
                st.metric("Market Position", comp.get("competitive_position","N/A"))
                for a in comp.get("key_advantages",[])[:3]: st.success(a[:90])

        with tab2:
            m1,m2,m3,m4 = st.columns(4)
            with m1:
                section("Short Interest")
                sq = short.get("squeeze_score",0)
                st.markdown(f"<div style='color:{sc(sq)};font-weight:700;font-size:0.95rem;margin-bottom:8px;'>{short.get('squeeze_risk','N/A')}</div>", unsafe_allow_html=True)
                sf = short.get("short_float_pct"); dtc = short.get("days_to_cover")
                if sf: st.metric("Short Float", f"{sf:.1f}%")
                if dtc: st.metric("Days to Cover", f"{dtc:.1f}")
                st.metric("Squeeze Score", f"{sq}/100")
                tip("High short interest + rising price = potential short squeeze — a rapid price spike.")
            with m2:
                section("Options Flow")
                pcr = options.get("put_call_ratio",{}); ivr = options.get("iv_rank",{}); mp = options.get("max_pain",{})
                if pcr.get("pcr_volume"): st.metric("Put/Call Ratio", f"{pcr['pcr_volume']:.2f}"); st.caption(pcr.get("sentiment",""))
                if ivr.get("iv_rank") is not None: st.metric("IV Rank", f"{ivr['iv_rank']:.0f}/100")
                if mp.get("max_pain_price"): st.metric("Max Pain", f"${mp['max_pain_price']:.2f}")
            with m3:
                section("Insider Activity")
                st.markdown(insider.get("signal","N/A"))
                inst = insider.get("institutional",{})
                st.metric("Institutional", f"{inst.get('institutional_ownership_pct',0):.1f}%")
                st.metric("Insider Own.", f"{inst.get('insider_ownership_pct',0):.1f}%")
                tip("When executives buy stock with personal money, that's a strong bullish signal.")
            with m4:
                section("Technical")
                for k,v in list(tech.get("signals",{}).items())[:4]: st.markdown(f"**{k}:** {v}")
                if tech.get("support"):
                    st.metric("Support",    f"${tech['support']:.2f}")
                    st.metric("Resistance", f"${tech.get('resistance',0):.2f}")

        with tab3:
            s1,s2 = st.columns(2)
            with s1:
                section("Monte Carlo Simulation")
                if mc:
                    mc1,mc2,mc3 = st.columns(3)
                    mc1.metric("Median 1Y", f"{mc.get('median_return_1y',0)*100:.1f}%")
                    mc2.metric("Bull Case", f"{mc.get('bull_case',0)*100:.1f}%")
                    mc3.metric("Bear Case", f"{mc.get('bear_case',0)*100:.1f}%")
                    st.metric("Probability of Profit", f"{mc.get('probability_profit',0)*100:.0f}%")
                    tip("<strong>Monte Carlo</strong> runs thousands of simulated scenarios to estimate the range of possible outcomes over 12 months.")
            with s2:
                section("DCF Valuation")
                if dcf and dcf.get("intrinsic_value"):
                    st.metric("Intrinsic Value", f"${dcf['intrinsic_value']:.2f}")
                    st.metric("Status", dcf.get("valuation_status",""))
                    st.metric("Margin of Safety (20%)", f"${dcf.get('margin_of_safety_20',0):.2f}")
                    tip("<strong>Intrinsic value</strong> = what the stock is theoretically worth based on future cash flows. Price below intrinsic value may indicate undervaluation.")
            stress = sims.get("stress_test",{})
            if stress and stress.get("scenarios"):
                section("Historical Stress Tests")
                rows = [{"Scenario":sc2,"Est. Drawdown":f"{sd.get('estimated_drawdown_pct',0):.1f}%","Stressed Price":f"${sd.get('estimated_price',0):.2f}"} for sc2,sd in stress["scenarios"].items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                tip("Shows how this stock might have performed during past market crashes (2008 financial crisis, COVID-19, etc.).")

        with tab4:
            section("News Sentiment")
            sn1,sn2,sn3 = st.columns(3)
            sn1.metric("Sentiment Score",   f"{sent.get('score',0):.0f}/100")
            sn2.metric("Bullish Headlines", sent.get("bullish_headlines",0))
            sn3.metric("Bearish Headlines", sent.get("bearish_headlines",0))
            st.caption(sent.get("overall",""))
            for h in sent.get("headlines",[])[:6]: st.markdown(f"— {h}")

        with tab5:
            section("Risk Metrics")
            r1c,r2c,r3c,r4c,r5c,r6c = st.columns(6)
            r1c.metric("Sharpe",      f"{risk.get('sharpe_ratio',0):.2f}")
            r2c.metric("Sortino",     f"{risk.get('sortino_ratio',0):.2f}")
            r3c.metric("Calmar",      f"{risk.get('calmar_ratio',0):.2f}")
            r4c.metric("Max Drawdown",f"{risk.get('max_drawdown',0)*100:.1f}%")
            r5c.metric("VaR 95%",     f"{risk.get('var_95',0)*100:.2f}%")
            r6c.metric("Beta",        f"{risk.get('beta',0):.2f}")
            tip("<strong>Sharpe Ratio</strong> = return per unit of risk (above 1.0 is good) · <strong>Max Drawdown</strong> = worst peak-to-trough loss · <strong>Beta</strong> = market sensitivity (2.0 = moves 2× the market)")
            if macro:
                st.markdown("---")
                section("Macro Regime Context")
                st.markdown(f"**Regime:** {macro.get('regime','N/A')} — {macro.get('asset_bias','')}")
                st.caption(macro.get("description",""))
                fit = macro.get("sector_fit",{})
                if fit.get("assessment"): st.info(fit["assessment"])


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Builder":
    st.markdown("## Portfolio Builder")
    st.markdown("<div style='color:#6B7280;margin-bottom:20px;'>Answer a few questions and we'll build a personalized investment portfolio with specific tickers, allocation percentages, and long-term projections.</div>", unsafe_allow_html=True)

    with st.form("portfolio_form"):
        st.markdown("### Your Profile")
        c1,c2,c3 = st.columns(3)
        with c1: experience = st.selectbox("Experience Level", ["Beginner — Just getting started","Intermediate — Some experience","Advanced — Active investor"])
        with c2: account    = st.selectbox("Account Type", ["Taxable Brokerage","Roth IRA","Traditional IRA","401(k)"])
        with c3: goal       = st.selectbox("Primary Goal", ["Long-term wealth building","Retirement in 20+ years","Retirement in 10-20 years","Income / Dividends","Short-term growth (5 years)"])
        st.markdown("### Risk Tolerance")
        risk_tol = st.slider("How would you react if your portfolio dropped 30%?", 1, 10, 5, help="1 = Sell everything  |  10 = Buy more aggressively")
        nearest = min({1:"Sell everything",3:"Very nervous",5:"Hold steady",7:"Buy more",10:"Buy aggressively"}.keys(), key=lambda x: abs(x-risk_tol))
        st.caption(f"Your selection: {  {1:'Sell everything',3:'Very nervous',5:'Hold steady',7:'Buy more',10:'Buy aggressively'}[nearest]}")
        st.markdown("### Capital & Timeline")
        cap1,cap2,cap3 = st.columns(3)
        with cap1: capital = st.number_input("Starting Capital ($)", min_value=500, max_value=10_000_000, value=10_000, step=500)
        with cap2: monthly = st.number_input("Monthly Contribution ($)", min_value=0, max_value=50_000, value=500, step=100)
        with cap3: horizon = st.selectbox("Investment Horizon", ["5 years","10 years","20 years","30+ years"])
        st.markdown("### Style")
        pr1,pr2 = st.columns(2)
        with pr1: style   = st.selectbox("Investment Style", ["Balanced (growth + stability)","Aggressive growth","Conservative / income","Value focused","Dividend growth"])
        with pr2: sectors = st.multiselect("Preferred Sectors", ["Technology","Healthcare","Finance","Energy","Consumer Staples","Real Estate","Industrials","No preference"], default=["No preference"])
        submitted = st.form_submit_button("Build My Portfolio", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Building your personalized portfolio..."):
            try:
                from advisor import PortfolioAdvisor
                result = PortfolioAdvisor().build_portfolio_recommendation({
                    "risk_tolerance":risk_tol,"experience":experience,"goal":goal,
                    "account_type":account,"starting_capital":capital,
                    "monthly_contribution":monthly,"time_horizon":horizon,
                    "preferred_sectors":sectors,"investment_style":style,
                })
                st.markdown("---")
                st.markdown("## Your Portfolio")
                profile = result.get("profile",{})
                if profile: st.markdown(f"**Profile:** {profile.get('type','')}"); st.info(profile.get("description",""))
                alloc = result.get("allocation",{})
                if alloc:
                    st.markdown("### Asset Allocation")
                    a_cols = st.columns(len(alloc))
                    for col,(asset,pct) in zip(a_cols,alloc.items()): col.metric(asset,f"{pct:.0f}%")
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
                    tip("These projections assume consistent contributions and historical average returns. Compound interest means starting early has an enormous impact on your final wealth.")
                for w in result.get("warnings",[]): st.warning(w)
            except Exception as e:
                st.error(f"Portfolio builder error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ETF PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ETF Portfolio":
    try:
        import etf_portfolio_page
        etf_portfolio_page.render()
    except ImportError:
        st.error("etf_portfolio_page.py not found in your InstitutionalEdge folder.")


# ══════════════════════════════════════════════════════════════════════════════
#  ML ALPHA SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Alpha Signals":
    st.markdown("## ML Alpha Signals")
    st.markdown("<div style='color:#6B7280;margin-bottom:16px;'>Stacked ensemble of 6 ML models trained on 125+ features. Predicts which stocks will outperform the S&P 500 by 10%+ over the next 6 months. Regime-aware and risk-adjusted.</div>", unsafe_allow_html=True)
    tip("<strong>How to read this:</strong> The model scores every stock and ranks them by probability of outperforming the market. It detects the current market regime (bull, bear, sideways) and adjusts recommendations accordingly.")

    ml1,ml2 = st.columns([3,1])
    with ml1: st.markdown("First run trains the model (~15 min). After that it saves locally and loads in seconds.")
    with ml2: top_n = st.number_input("Top N picks", min_value=5, max_value=50, value=25)

    if st.button("Run Alpha Model", type="primary", use_container_width=True):
        try:
            from alpha_model import AlphaModel
            model = AlphaModel(target="label_6m_10pct")
            saved = os.path.join("models","alpha_v1.pkl")
            if os.path.exists(saved):
                with st.spinner("Loading trained model..."): model.load("alpha_v1")
                with st.spinner("Scoring stocks..."): signals = model.generate_signals(top_n=top_n, verbose=False)
            else:
                prog = st.empty()
                prog.info("Building universe — extracting features for ~170 stocks...")
                df = model.build_universe()
                if df is None or df.empty: st.error("Could not build universe."); st.stop()
                prog.info("Training 6-model ensemble (~15 min first run)...")
                model.train(verbose=False)
                prog.info("Running walk-forward validation...")
                model.validate(verbose=False)
                prog.info("Generating signals...")
                signals = model.generate_signals(top_n=top_n, verbose=False)
                model.save("alpha_v1")
                prog.success("Model trained and saved.")

            if signals:
                ri = signals[0] if signals else {}
                regime = ri.get("regime","UNKNOWN")
                regime_color = {"BULL":"#16A34A","BEAR":"#DC2626","SIDEWAYS":"#CA8A04"}.get(regime,"#6B7280")
                st.markdown(f"<div class='ie-card' style='display:flex;gap:32px;align-items:center;'><div><div style='font-size:0.65rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.06em;'>Market Regime</div><div style='font-size:1.2rem;font-weight:800;color:{regime_color};font-family:monospace;'>{regime}</div></div><div><div style='font-size:0.65rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.06em;'>Stocks Scored</div><div style='font-size:1.2rem;font-weight:700;color:#111827;font-family:monospace;'>{ri.get('universe_size',0)}</div></div><div><div style='font-size:0.65rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.06em;'>Buy Signals</div><div style='font-size:1.2rem;font-weight:700;color:#16A34A;font-family:monospace;'>{ri.get('buy_signals',0)}</div></div></div>", unsafe_allow_html=True)

                val_data = getattr(model,"validation_results",{})
                if val_data:
                    section("Walk-Forward Backtest")
                    v1,v2,v3,v4 = st.columns(4)
                    v1.metric("Avg Alpha",    f"+{val_data.get('avg_alpha',0)*100:.1f}%")
                    v2.metric("Hit Rate",     f"{val_data.get('hit_rate',0)*100:.0f}%")
                    v3.metric("Alpha Sharpe", f"{val_data.get('alpha_sharpe',0):.2f}")
                    v4.metric("CV Folds",     val_data.get("n_folds",5))
                    tip("<strong>Hit Rate</strong> = % of picks that beat the S&P 500. <strong>Alpha Sharpe</strong> = risk-adjusted outperformance. Above 1.0 is strong.")

                section(f"Top {len(signals)} Picks")
                rows = [{"Rank":i+1,"Ticker":s.get("ticker",""),"Probability":f"{s.get('probability',0):.3f}","Alpha Score":f"{s.get('alpha_score',0):.3f}","Risk-Adj":f"{s.get('risk_adj_score',0):.3f}","Position":f"{s.get('position_size',0)*100:.1f}%","Confidence":s.get("confidence_tier","")} for i,s in enumerate(signals)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        except ImportError:
            st.error("alpha_model.py not found.")
        except Exception as e:
            st.error(f"ML model error: {e}")
            st.info("Make sure scikit-learn, xgboost, and lightgbm are installed: pip install scikit-learn xgboost lightgbm")


# ══════════════════════════════════════════════════════════════════════════════
#  MACRO & CRYPTO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Macro & Crypto":
    tab_macro, tab_crypto = st.tabs(["Macro Regime","Crypto Analysis"])

    with tab_macro:
        st.markdown("## Macro Regime Analysis")
        st.markdown("<div style='color:#6B7280;margin-bottom:16px;'>Understand the current economic environment and which sectors to favor.</div>", unsafe_allow_html=True)
        if st.button("Run Macro Analysis", type="primary"):
            with st.spinner("Fetching macro data..."):
                try:
                    from macro_regime import MacroRegimeModel
                    st.session_state.macro_result = MacroRegimeModel().full_analysis()
                except Exception as e: st.error(f"Macro error: {e}")
        macro = st.session_state.get("macro_result")
        if macro:
            st.markdown(f"### Regime: {macro.get('regime','N/A')}")
            st.info(f"{macro.get('description','')} · Bias: **{macro.get('asset_bias','')}**")
            fc,ac = st.columns(2)
            with fc: st.markdown("**Favored Sectors**"); [st.success(s) for s in macro.get("favored_sectors",[])]
            with ac: st.markdown("**Avoid Sectors**");   [st.error(s)   for s in macro.get("avoid_sectors",[])]
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
        st.markdown("## Crypto Analysis")
        cc1,cc2 = st.columns([2,1])
        with cc1: crypto_in = st.text_input("Coin ID", value="bitcoin", placeholder="bitcoin, ethereum, solana...")
        with cc2: crypto_go = st.button("Analyze", type="primary", use_container_width=True, key="crypto_go")
        if crypto_go:
            with st.spinner(f"Analyzing {crypto_in}..."):
                try:
                    from crypto_analyzer import CryptoAnalyzer
                    st.session_state.crypto_result = CryptoAnalyzer().full_analysis(crypto_in)
                except Exception as e: st.error(f"Crypto error: {e}")
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
#  SCREENER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Screener":
    st.markdown("## Stock Screener")
    stab1,stab2 = st.tabs(["Strategy Screener","Watchlist Ranker"])

    with stab1:
        sc1,sc2 = st.columns([2,1])
        with sc1:
            profile = st.selectbox("Select Strategy",["garp","aggressive_growth","deep_value","quality_compounder","dividend_growth","momentum"],
                format_func=lambda x:{"garp":"GARP — Growth at a Reasonable Price","aggressive_growth":"Aggressive Growth","deep_value":"Deep Value — Undervalued stocks","quality_compounder":"Quality Compounder — Wide moat","dividend_growth":"Dividend Growth — Income + appreciation","momentum":"Momentum — Price momentum leaders"}[x])
        with sc2: screen_go = st.button("Screen Now", type="primary", use_container_width=True)
        tip({"garp":"Finds companies growing faster than the market but not at bubble valuations — balances growth and price discipline.","aggressive_growth":"Targets highest-growth companies. Higher risk, higher potential reward.","deep_value":"Stocks trading well below intrinsic value. Classic bargain hunting.","quality_compounder":"Wide moat companies with durable competitive advantages and consistent returns on capital.","dividend_growth":"Companies with growing dividends and strong balance sheets — ideal for income-focused investors.","momentum":"Stocks with strong recent price performance. Trend-following approach."}[profile])
        if screen_go:
            with st.spinner(f"Screening for {profile} stocks..."):
                try:
                    from main import run_screener
                    results = run_screener(profile=profile)
                    if results:
                        st.markdown(f"### Results — {len(results)} stocks found")
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
