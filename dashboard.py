"""
dashboard.py — InstitutionalEdge Streamlit Web Dashboard

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

# Page config
st.set_page_config(
    page_title="InstitutionalEdge",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #2d2d3e;
    margin-bottom: 12px;
}
.score-ring {
    font-size: 2.4rem;
    font-weight: 800;
    text-align: center;
}
.signal-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
}
.strong-buy  { background: #1a472a; color: #00ff7f; }
.buy         { background: #1a3a1a; color: #7fff7f; }
.hold        { background: #3a3a1a; color: #ffff7f; }
.weak        { background: #3a2a1a; color: #ffaa7f; }
.avoid       { background: #3a1a1a; color: #ff7f7f; }
.sticker {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ──────────────────────────────────────────────────────

def run_analysis(ticker: str) -> dict:
    from main import analyze_stock
    with st.spinner(f"Running institutional analysis on {ticker.upper()}..."):
        return analyze_stock(ticker, verbose=False, skip_macro=False)


def score_color(score: float) -> str:
    if score >= 70: return "#00ff7f"
    elif score >= 55: return "#7fff7f"
    elif score >= 45: return "#ffff7f"
    elif score >= 35: return "#ffaa7f"
    else: return "#ff7f7f"


def signal_class(signal: str) -> str:
    return {
        "STRONG BUY": "strong-buy",
        "BUY":        "buy",
        "HOLD":       "hold",
        "WEAK":       "weak",
        "AVOID":      "avoid",
    }.get(signal, "hold")


def fmt_mktcap(v):
    if not v: return "N/A"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


# ── SIDEBAR ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 InstitutionalEdge")
    st.markdown("*Institutional-grade investment analysis*")
    st.divider()

    ticker_input = st.text_input("Stock Ticker", value="NVDA", max_chars=10).upper()
    run_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Watchlist**")
    watchlist_input = st.text_area("Tickers (one per line)", value="NVDA\nAAPL\nMSFT\nSMCI")
    watchlist_btn = st.button("📊 Rank Watchlist", use_container_width=True)

    st.divider()
    macro_btn = st.button("🌍 Macro Regime", use_container_width=True)

    st.divider()
    st.markdown("**Crypto**")
    crypto_input = st.text_input("Coin ID (e.g. bitcoin)", value="bitcoin")
    crypto_btn = st.button("₿ Analyze Crypto", use_container_width=True)

    st.divider()
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ── MAIN AREA ────────────────────────────────────────────────────

st.title("🏦 InstitutionalEdge v3.0")
st.caption("Institutional-level investment analysis — free, open source")

# ── STOCK ANALYSIS ────────────────────────────────────────────────
if run_btn or ("results" in st.session_state and st.session_state.get("last_ticker") == ticker_input):
    if run_btn:
        st.session_state["results"]      = run_analysis(ticker_input)
        st.session_state["last_ticker"]  = ticker_input

    r = st.session_state.get("results", {})
    if not r:
        st.error("Could not fetch data. Check the ticker and try again.")
        st.stop()

    pt      = r.get("price_target", {})
    fund    = r.get("fundamental",  {})
    tech    = r.get("technical",    {})
    sent    = r.get("sentiment",    {})
    risk    = r.get("risk",         {})
    comp    = r.get("competitive",  {})
    insider = r.get("insider",      {})
    short   = r.get("short",        {})
    options = r.get("options",      {})
    macro   = r.get("macro",        {})
    sims    = r.get("simulations",  {})
    mc      = sims.get("monte_carlo", {})
    dcf     = sims.get("dcf", {})
    score   = r.get("composite_score", 0)
    signal  = r.get("buy_signal", "HOLD")
    mktcap  = r.get("market_cap", 0)

    # ── Header ───────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns([2,1,1,1,1])
    with col1:
        st.markdown(f"## {r['ticker']} — {r.get('company_name','')}")
        st.caption(f"{r.get('sector','')} | {r.get('industry','')}")
    with col2:
        st.metric("Price", f"${r.get('current_price',0):.2f}")
    with col3:
        st.metric("Market Cap", fmt_mktcap(mktcap))
    with col4:
        color = score_color(score)
        st.markdown(f"<div style='text-align:center'><div style='color:{color};font-size:2rem;font-weight:800'>{score:.1f}</div><div style='color:gray;font-size:0.8rem'>COMPOSITE SCORE</div></div>", unsafe_allow_html=True)
    with col5:
        cls = signal_class(signal)
        st.markdown(f"<div style='text-align:center;margin-top:8px'><span class='signal-badge {cls}'>{signal}</span></div>", unsafe_allow_html=True)

    st.divider()

    # ── Price Target ─────────────────────────────────────────────
    if pt.get("price_target_12m"):
        st.subheader("📈 12-Month Price Target")
        c1, c2, c3, c4 = st.columns(4)
        up = pt.get("upside_downside", 0)
        c1.metric("Price Target", f"${pt['price_target_12m']:.2f}", f"{up:+.1f}%")
        c2.metric("Low Estimate",  f"${pt.get('confidence_low',0):.2f}")
        c3.metric("High Estimate", f"${pt.get('confidence_high',0):.2f}")
        c4.metric("Conviction",    pt.get("conviction","N/A").split()[0])

        # Model breakdown
        model_rows = []
        for name, m in pt.get("models", {}).items():
            if m.get("price"):
                model_rows.append({
                    "Model":   m.get("source", name),
                    "Target":  f"${m['price']:.2f}",
                    "Upside":  f"{m.get('upside_pct',0):+.1f}%",
                    "Confidence": f"{m.get('confidence',0)}%",
                })
        if model_rows:
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        if pt.get("thesis"):
            with st.expander("📝 Investment Thesis"):
                st.write(pt["thesis"])

    # ── Score Breakdown ───────────────────────────────────────────
    st.subheader("📊 Score Breakdown")
    cols = st.columns(8)
    score_items = [
        ("Fundamental", fund.get("score",0), "28%"),
        ("Competitive", comp.get("score",0), "22%"),
        ("Technical",   tech.get("score",0), "17%"),
        ("Sentiment",   sent.get("score",0), "10%"),
        ("Risk-Adj",    risk.get("score",0), "8%"),
        ("Insider",     insider.get("score",0), "7%"),
        ("Options",     options.get("score",0), "5%"),
        ("Short",       short.get("score",0), "3%"),
    ]
    for col, (label, val, weight) in zip(cols, score_items):
        color = score_color(float(val))
        col.markdown(f"<div style='text-align:center'><div style='color:{color};font-size:1.5rem;font-weight:800'>{val:.0f}</div><div style='font-size:0.72rem;color:gray'>{label}<br>{weight}</div></div>", unsafe_allow_html=True)

    st.divider()

    # ── Macro + Insider + Short + Options row ─────────────────────
    col_macro, col_insider, col_short, col_options = st.columns(4)

    with col_macro:
        st.subheader("🌍 Macro Regime")
        if macro:
            regime = macro.get("regime","N/A")
            bias   = macro.get("asset_bias","")
            colors_regime = {"Expansion":"🟢","Overheating":"🟠","Stagflation":"🔴","Recession":"🔴","Recovery":"🟡"}
            st.markdown(f"**{colors_regime.get(regime,'⚪')} {regime}** — {bias}")
            st.caption(macro.get("description",""))
            fit = macro.get("sector_fit",{})
            st.markdown(fit.get("assessment",""))
            favored = macro.get("favored_sectors",[])
            if favored:
                st.caption(f"Favored: {', '.join(favored[:3])}")
        else:
            st.caption("Macro data unavailable")

    with col_insider:
        st.subheader("🏛️ Insider/Inst.")
        st.markdown(insider.get("signal","N/A"))
        inst = insider.get("institutional",{})
        st.metric("Inst. Ownership", f"{inst.get('institutional_ownership_pct',0):.1f}%")
        st.metric("Insider Own.",    f"{inst.get('insider_ownership_pct',0):.1f}%")
        if insider.get("institutional",{}).get("top_institutions"):
            top = insider["institutional"]["top_institutions"][:2]
            for h in top:
                name = h.get("Holder") or h.get("holder","")
                if name: st.caption(f"• {name}")

    with col_short:
        st.subheader("📉 Short Interest")
        sq_score = short.get("squeeze_score",0)
        sq_risk  = short.get("squeeze_risk","N/A")
        sq_color = score_color(sq_score)
        st.markdown(f"<div style='color:{sq_color};font-size:1.4rem;font-weight:800'>{sq_risk}</div>", unsafe_allow_html=True)
        sf  = short.get("short_float_pct")
        dtc = short.get("days_to_cover")
        if sf:  st.metric("Short Float",    f"{sf:.1f}%")
        if dtc: st.metric("Days to Cover",  f"{dtc:.1f}")
        st.metric("Squeeze Score", f"{sq_score}/100")

    with col_options:
        st.subheader("⚙️ Options Flow")
        pcr = options.get("put_call_ratio",{})
        ivr = options.get("iv_rank",{})
        mp  = options.get("max_pain",{})
        if pcr.get("pcr_volume"):
            st.metric("P/C Ratio",  f"{pcr['pcr_volume']:.2f}")
            st.caption(pcr.get("sentiment",""))
        if ivr.get("iv_rank") is not None:
            st.metric("IV Rank",    f"{ivr['iv_rank']:.0f}/100")
            st.caption(ivr.get("assessment","")[:50])
        if mp.get("max_pain_price"):
            st.metric("Max Pain",   f"${mp['max_pain_price']:.2f}")

    st.divider()

    # ── Fundamentals + Competitive ────────────────────────────────
    col_f, col_c = st.columns(2)

    with col_f:
        st.subheader("📋 Fundamentals")
        highlights = fund.get("highlights",{})
        if highlights:
            df_fund = pd.DataFrame([{"Metric": k, "Value": v} for k,v in highlights.items()])
            st.dataframe(df_fund, use_container_width=True, hide_index=True)
        q = fund.get("quality_score",{})
        if q:
            st.metric("Piotroski F-Score", f"{q.get('f_score','N/A')}/9",
                      delta=q.get("interpretation",""))

    with col_c:
        st.subheader("⚔️ Competitive")
        moat = comp.get("moat_score",{})
        st.metric("Economic Moat",      moat.get("width","N/A"),
                  delta=f"{moat.get('score',0)}/100")
        st.metric("Competitive Position", comp.get("competitive_position","N/A"))
        rel = comp.get("relative_valuation",{})
        if rel.get("overall"):
            st.info(f"vs Peers: {rel['overall']}")
        adv = comp.get("key_advantages",[])
        for a in adv[:2]:
            st.success(f"✅ {a[:80]}")
        crs = comp.get("key_risks",[])
        for cr in crs[:2]:
            st.warning(f"⚠️ {cr[:80]}")

    # ── Technical + Simulations ───────────────────────────────────
    col_t, col_s = st.columns(2)

    with col_t:
        st.subheader("📉 Technical Analysis")
        for k, v in tech.get("signals",{}).items():
            st.markdown(f"**{k}:** {v}")
        if tech.get("support"):
            c1, c2 = st.columns(2)
            c1.metric("Support",    f"${tech['support']:.2f}")
            c2.metric("Resistance", f"${tech.get('resistance',0):.2f}")
        st.caption(f"Volume Trend: {tech.get('volume_trend','N/A')}")

    with col_s:
        st.subheader("🎲 Simulations")
        if mc:
            c1, c2, c3 = st.columns(3)
            c1.metric("Median 1Y",   f"{mc.get('median_return_1y',0)*100:.1f}%")
            c2.metric("Bull Case",   f"{mc.get('bull_case',0)*100:.1f}%")
            c3.metric("Bear Case",   f"{mc.get('bear_case',0)*100:.1f}%")
            st.metric("Prob Profit",  f"{mc.get('probability_profit',0)*100:.0f}%")

        if dcf and dcf.get("intrinsic_value"):
            st.markdown("**DCF Valuation**")
            c1, c2 = st.columns(2)
            c1.metric("Intrinsic Value", f"${dcf['intrinsic_value']:.2f}")
            c2.metric("Status",          dcf.get("valuation_status",""))
            c1.metric("MOS 20%",         f"${dcf.get('margin_of_safety_20',0):.2f}")
            c2.metric("MOS 30%",         f"${dcf.get('margin_of_safety_30',0):.2f}")

    # ── Stress Tests ──────────────────────────────────────────────
    stress = sims.get("stress_test",{})
    if stress and stress.get("scenarios"):
        st.subheader("🚨 Historical Stress Tests")
        rows = [{"Scenario": sc, "Estimated Drawdown": f"{sd.get('estimated_drawdown_pct',0):.1f}%",
                 "Stressed Price": f"${sd.get('estimated_price',0):.2f}"}
                for sc, sd in stress["scenarios"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Catalysts & Risks ─────────────────────────────────────────
    col_cat, col_risk = st.columns(2)
    cats = pt.get("catalysts",[])
    rks  = pt.get("risks",[])
    with col_cat:
        if cats:
            st.subheader("🚀 Catalysts")
            for c in cats:
                st.success(f"🚀 {c}")
    with col_risk:
        if rks:
            st.subheader("⚠️ Risk Factors")
            for rk in rks:
                st.error(f"⚠️ {rk}")

    # ── Sentiment ─────────────────────────────────────────────────
    st.subheader("📰 News Sentiment")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment Score",     f"{sent.get('score',0):.1f}/100")
    c2.metric("Bullish Headlines",   sent.get("bullish_headlines",0))
    c3.metric("Bearish Headlines",   sent.get("bearish_headlines",0))
    st.caption(sent.get("overall",""))
    for h in sent.get("headlines",[])[:5]:
        st.markdown(f"• {h}")

    # ── Risk ─────────────────────────────────────────────────────
    st.subheader("⚖️ Risk Metrics")
    rc1,rc2,rc3,rc4,rc5,rc6 = st.columns(6)
    rc1.metric("Sharpe",      f"{risk.get('sharpe_ratio',0):.2f}")
    rc2.metric("Sortino",     f"{risk.get('sortino_ratio',0):.2f}")
    rc3.metric("Calmar",      f"{risk.get('calmar_ratio',0):.2f}")
    rc4.metric("Max Drawdown", f"{risk.get('max_drawdown',0)*100:.1f}%")
    rc5.metric("VaR 95%",     f"{risk.get('var_95',0)*100:.2f}%")
    rc6.metric("Beta",        f"{risk.get('beta',0):.2f}")


# ── MACRO PAGE ────────────────────────────────────────────────────
elif macro_btn:
    from macro_regime import MacroRegimeModel
    with st.spinner("Fetching macro data..."):
        macro = MacroRegimeModel().full_analysis()

    st.subheader(f"🌍 Macro Regime: {macro.get('regime','N/A')}")
    st.info(f"{macro.get('description','')} | **Bias: {macro.get('asset_bias','')}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Favored Sectors**")
        for s in macro.get("favored_sectors",[]): st.success(s)
    with c2:
        st.markdown("**🚫 Avoid Sectors**")
        for s in macro.get("avoid_sectors",[]): st.error(s)

    rot = macro.get("sector_rotation",{})
    if rot:
        st.subheader("🔄 Sector Rotation")
        rows = [{
            "Sector": sec,
            "ETF":    d.get("etf",""),
            "1M":     f"{d.get('1m_return',0):+.1f}%",
            "3M":     f"{d.get('3m_return',0):+.1f}%",
            "6M":     f"{d.get('6m_return',0):+.1f}%",
            "Signal": d.get("signal",""),
        } for sec, d in sorted(rot.items(), key=lambda x: x[1].get("momentum_score",0), reverse=True)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    ind = macro.get("indicators",{})
    yc   = ind.get("yield_curve",{})
    vix  = ind.get("vix",{})
    mkt  = ind.get("market_trend",{})
    infl = ind.get("inflation",{})
    if yc or vix or mkt:
        st.subheader("📊 Key Indicators")
        cols = st.columns(4)
        if yc:
            cols[0].metric("10Y-3M Spread", f"{yc.get('spread_10y_3m',0):.2f}%",
                           delta="Inverted ⚠️" if yc.get("inverted") else "Normal")
        if vix:
            cols[1].metric("VIX", f"{vix.get('current',0):.1f}",
                           delta=vix.get("regime",""))
        if mkt:
            cols[2].metric("SPY 1Y Return", f"{mkt.get('1y_return',0):+.1f}%",
                           delta=mkt.get("trend",""))
        if infl:
            cols[3].metric("Inflation Trend", infl.get("inflation_trend",""),
                           delta=infl.get("signal","")[:30])


# ── CRYPTO PAGE ────────────────────────────────────────────────────
elif crypto_btn:
    from crypto_analyzer import CryptoAnalyzer
    with st.spinner(f"Analyzing {crypto_input}..."):
        crypto = CryptoAnalyzer()
        cr = crypto.full_analysis(crypto_input)

    market = cr.get("market",{})
    tech_c = cr.get("technical",{})
    fund_c = cr.get("fundamentals",{})
    mc_c   = cr.get("simulations",{}).get("monte_carlo",{})

    st.subheader(f"₿ {market.get('name',crypto_input)} ({market.get('symbol','')})")
    c1,c2,c3,c4 = st.columns(4)
    price = market.get("current_price")
    c1.metric("Price", f"${price:,.2f}" if price else "N/A")
    mktcap_c = market.get("market_cap")
    c2.metric("Market Cap", fmt_mktcap(mktcap_c))
    c3.metric("Rank", f"#{market.get('rank','N/A')}")
    c4.metric("Tier", market.get("tier","N/A"))

    c1,c2,c3 = st.columns(3)
    c1.metric("24H",  f"{market.get('price_change_24h',0):.2f}%")
    c2.metric("7D",   f"{market.get('price_change_7d',0):.2f}%")
    c3.metric("30D",  f"{market.get('price_change_30d',0):.2f}%")

    if mc_c:
        st.subheader("Monte Carlo (1 Year)")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Median Return", f"{mc_c.get('median_return_1y',0)*100:.1f}%")
        c2.metric("Bull Case",     f"{mc_c.get('bull_case',0)*100:.1f}%")
        c3.metric("Bear Case",     f"{mc_c.get('bear_case',0)*100:.1f}%")
        c4.metric("Prob Profit",   f"{mc_c.get('probability_profit',0)*100:.0f}%")


# ── WATCHLIST ─────────────────────────────────────────────────────
elif watchlist_btn:
    tickers = [t.strip().upper() for t in watchlist_input.split("\n") if t.strip()]
    if tickers:
        from main import generate_watchlist
        with st.spinner(f"Analyzing {len(tickers)} stocks..."):
            ranked = generate_watchlist(tickers)

        st.subheader(f"📊 Watchlist Rankings — {len(ranked)} Stocks")
        rows = []
        for s in ranked:
            rows.append({
                "Rank":    ranked.index(s)+1,
                "Ticker":  s["ticker"],
                "Company": s["company"],
                "Score":   s["composite_score"],
                "Signal":  s["buy_signal"],
                "Target":  f"${s['price_target_12m']:.2f}" if s.get("price_target_12m") else "N/A",
                "Upside":  f"{s['upside_pct']:.1f}%" if s.get("upside_pct") is not None else "N/A",
                "Moat":    s.get("moat","N/A"),
                "Squeeze": s.get("squeeze",""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Add at least one ticker to the watchlist.")

else:
    # Landing page
    st.markdown("""
    ## Welcome to InstitutionalEdge 🏦

    The same analytical frameworks used by professional asset managers — completely free.

    ### What This Tool Analyzes:
    | Module | What It Does |
    |---|---|
    | 📋 **Fundamental Analysis** | P/E, growth, margins, Piotroski F-Score |
    | ⚔️ **Competitive Analysis** | Moat scoring, peer comparison, relative valuation |
    | 📉 **Technical Analysis** | 15+ indicators, trend, momentum, support/resistance |
    | 🏛️ **Insider Trading** | SEC Form 4 filings, institutional 13F ownership |
    | 📉 **Short Interest** | Short float, days to cover, squeeze probability |
    | ⚙️ **Options Signals** | P/C ratio, IV rank, max pain, unusual activity |
    | 🌍 **Macro Regime** | Economic cycle detection, sector rotation signals |
    | 🎲 **Simulations** | Monte Carlo, DCF, scenario analysis, stress tests |
    | ⚖️ **Risk Metrics** | Sharpe, VaR, CVaR, beta, max drawdown |
    | 📈 **Price Target** | 5-model blended 12-month target with conviction |

    ### Quick Start:
    1. Enter a ticker in the sidebar (e.g. **NVDA**, **AAPL**, **SMCI**)
    2. Click **Analyze**
    3. View full institutional-grade analysis in seconds

    ---
    *⚠️ For educational purposes only. Not financial advice.*
    """)
