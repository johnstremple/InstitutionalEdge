"""
InstitutionalEdge — Web Platform
Run: streamlit run dashboard_v6.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="InstitutionalEdge",
    page_icon="IE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
    h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .score-lg {
        font-size: 2.4rem; font-weight: 800; text-align: center;
        line-height: 1; letter-spacing: -0.02em;
    }
    .score-sm {
        font-size: 0.7rem; color: #8b949e; text-align: center;
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;
    }
    .signal-badge {
        display: inline-block; padding: 5px 14px; border-radius: 4px;
        font-weight: 700; font-size: 0.85rem; letter-spacing: 0.03em;
    }
    .badge-strong-buy { background: #0d3320; color: #34d399; border: 1px solid #065f46; }
    .badge-buy        { background: #14332a; color: #6ee7b7; border: 1px solid #065f46; }
    .badge-hold       { background: #332e14; color: #fcd34d; border: 1px solid #78350f; }
    .badge-weak       { background: #33201a; color: #fdba74; border: 1px solid #7c2d12; }
    .badge-avoid      { background: #331a1a; color: #fca5a5; border: 1px solid #7f1d1d; }
    .insight-box {
        background: #0d1117; border-left: 3px solid #3b82f6;
        padding: 12px 16px; margin: 8px 0; border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def score_color(s):
    if s >= 70: return "#34d399"
    if s >= 55: return "#6ee7b7"
    if s >= 45: return "#fcd34d"
    if s >= 35: return "#fdba74"
    return "#fca5a5"

def signal_badge(sig):
    cls = {"STRONG BUY":"badge-strong-buy","BUY":"badge-buy","HOLD":"badge-hold",
           "WEAK":"badge-weak","AVOID":"badge-avoid"}.get(sig,"badge-hold")
    return f"<span class='signal-badge {cls}'>{sig}</span>"

def fmt_mktcap(v):
    if not v: return "-"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def score_html(val, label):
    c = score_color(float(val))
    return (f"<div style='text-align:center'>"
            f"<div class='score-lg' style='color:{c}'>{val:.0f}</div>"
            f"<div class='score-sm'>{label}</div></div>")


# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### InstitutionalEdge")
    st.caption("Investment Research Platform")
    st.markdown("---")
    page = st.radio("", ["Home","Stock Analysis","Portfolio Builder",
                          "Alpha Signals","Screener"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("For educational and research purposes only. "
               "Not financial advice. All investing involves risk.")
    st.caption(f"v6.0 · {datetime.now().strftime('%b %d, %Y')}")


# ═════════════════════════════════════════════════════════════════
#   HOME
# ═════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("InstitutionalEdge")
    st.markdown("**Institutional-grade investment research — open and free.**")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Analysis Modules","12")
    c2.metric("ML Features","125+")
    c3.metric("Ensemble Models","6")
    c4.metric("Data Cost","$0/mo")
    st.markdown("")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### Stock Analysis")
        st.markdown("Full breakdown on any public equity: fundamental scoring, "
                    "technical indicators, Monte Carlo simulations, DCF valuation, "
                    "insider tracking, options flow, fraud detection, factor model, "
                    "and a 6-pillar conviction framework.")
        st.markdown("#### Alpha Model")
        st.markdown("Stacked ensemble of 6 ML models trained on 125+ features "
                    "to predict which stocks will outperform the S&P 500. "
                    "Regime-aware, risk-adjusted, walk-forward validated.")
    with col2:
        st.markdown("#### Portfolio Builder")
        st.markdown("Detailed financial survey covering age, income, risk tolerance, "
                    "goals, time horizon, and debt situation. Outputs a personalized "
                    "portfolio with specific tickers, sizing, and growth projections.")
        st.markdown("#### Screener")
        st.markdown("Screen 500+ stocks across 7 systematic strategies: "
                    "aggressive growth, GARP, deep value, quality compounder, "
                    "dividend growth, momentum, and small-cap growth.")
    st.markdown("---")
    st.markdown("Built by **John Stremple** · Finance & CS, George Mason University · "
                "[GitHub](https://github.com/johnstremple)")


# ═════════════════════════════════════════════════════════════════
#   STOCK ANALYSIS
# ═════════════════════════════════════════════════════════════════
elif page == "Stock Analysis":
    st.title("Stock Analysis")
    with st.form("analyze_form"):
        col_in,col_btn,_ = st.columns([2,1,5])
        with col_in:
            ticker_input = st.text_input("Ticker",value="NVDA",max_chars=10,
                                          label_visibility="collapsed",placeholder="Enter ticker")
        with col_btn:
            run_btn = st.form_submit_button("Run Analysis",type="primary",use_container_width=True)

    if run_btn:
        ticker = ticker_input.strip().upper()
        if not ticker:
            st.warning("Enter a ticker symbol."); st.stop()
        try:
            from main import analyze_stock
            with st.spinner(f"Analyzing {ticker}..."):
                results = analyze_stock(ticker,verbose=False,skip_macro=False)
            if not results:
                st.error(f"Could not retrieve data for {ticker}. Check the symbol and try again.")
                st.stop()
            st.session_state["results"] = results
            st.session_state["last_ticker"] = ticker
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)[:200]}")
            st.stop()

    if st.session_state.get("results") and st.session_state.get("last_ticker"):
        r = st.session_state["results"]
        ticker = st.session_state["last_ticker"]

        pt=r.get("price_target",{}); fund=r.get("fundamental",{})
        tech=r.get("technical",{}); sent=r.get("sentiment",{})
        risk=r.get("risk",{}); comp=r.get("competitive",{})
        insider=r.get("insider",{}); short=r.get("short",{})
        options=r.get("options",{}); sims=r.get("simulations",{})
        fraud=r.get("fraud_detection",{}); factor=r.get("factor_model",{})
        conv=r.get("conviction",{}); mc=sims.get("monte_carlo",{})
        dcf=sims.get("dcf",{}); score=r.get("composite_score",0)
        signal=r.get("buy_signal","HOLD")

        # Header
        h1,h2,h3,h4,h5 = st.columns([3,1,1,1,1])
        with h1:
            st.markdown(f"## {ticker} — {r.get('company_name','')}")
            st.caption(f"{r.get('sector','')} · {r.get('industry','')}")
        h2.metric("Price",f"${r.get('current_price',0):.2f}")
        h3.metric("Mkt Cap",fmt_mktcap(r.get("market_cap",0)))
        with h4: st.markdown(score_html(score,"Composite"),unsafe_allow_html=True)
        with h5: st.markdown(f"<div style='text-align:center;margin-top:10px'>{signal_badge(signal)}</div>",unsafe_allow_html=True)
        st.markdown("---")

        # Conviction
        if conv and conv.get("pillars"):
            st.markdown(f"### Conviction Score: {conv.get('score',0):.0f}/100 — {conv.get('conviction_label','')}")
            pillars=conv.get("pillars",{})
            pcols=st.columns(len(pillars))
            for col,(_,data) in zip(pcols,pillars.items()):
                col.markdown(score_html(data.get("score",0),data.get("label","")),unsafe_allow_html=True)
            thesis=conv.get("thesis",{})
            if thesis.get("verdict"): st.info(thesis["verdict"])
            tc1,tc2=st.columns(2)
            with tc1:
                for bp in thesis.get("bull_case",[])[:3]: st.success(bp)
            with tc2:
                for bp in thesis.get("bear_case",[])[:3]: st.error(bp)
            st.markdown("---")

        # Price Target
        if pt.get("price_target_12m"):
            st.markdown("### 12-Month Price Target")
            tc1,tc2,tc3,tc4=st.columns(4)
            up=pt.get("upside_downside",0)
            tc1.metric("Target",f"${pt['price_target_12m']:.2f}",f"{up:+.1f}%")
            tc2.metric("Low",f"${pt.get('confidence_low',0):.2f}")
            tc3.metric("High",f"${pt.get('confidence_high',0):.2f}")
            tc4.metric("Conviction",(pt.get("conviction","").split()[0] if pt.get("conviction") else "-"))
            models=pt.get("models",{})
            if models:
                rows=[{"Model":m.get("source",n),"Target":f"${m['price']:.2f}",
                       "Upside":f"{m.get('upside_pct',0):+.1f}%"}
                      for n,m in models.items() if m.get("price")]
                if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        # Module Scores
        st.markdown("### Module Scores")
        mods=[("Fundamental",fund),("Competitive",comp),("Technical",tech),
              ("Sentiment",sent),("Risk",risk),("Insider",insider),
              ("Fraud",fraud),("Factor",factor),("Options",options),("Short",short)]
        scols=st.columns(len(mods))
        for col,(label,mod) in zip(scols,mods):
            col.markdown(score_html(mod.get("score",50),label),unsafe_allow_html=True)
        st.markdown("---")

        # Fundamentals + Simulations
        fc1,fc2=st.columns(2)
        with fc1:
            st.markdown("### Fundamentals")
            hl=fund.get("highlights",{})
            if hl:
                st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in hl.items()]),
                             use_container_width=True,hide_index=True)
            q=fund.get("quality_score",{})
            if q: st.metric("Piotroski F-Score",f"{q.get('f_score','-')}/9",delta=q.get("interpretation",""))
        with fc2:
            st.markdown("### Simulations")
            if mc:
                m1,m2,m3=st.columns(3)
                m1.metric("Median 1Y",f"{mc.get('median_return_1y',0)*100:.1f}%")
                m2.metric("Bull",f"{mc.get('bull_case',0)*100:.1f}%")
                m3.metric("Bear",f"{mc.get('bear_case',0)*100:.1f}%")
                st.metric("Prob. Profit",f"{mc.get('probability_profit',0)*100:.0f}%")
            if dcf and dcf.get("intrinsic_value"):
                d1,d2=st.columns(2)
                d1.metric("DCF Value",f"${dcf['intrinsic_value']:.2f}")
                d2.metric("Valuation",dcf.get("valuation_status",""))

        # Institutional Data
        ic1,ic2,ic3,ic4=st.columns(4)
        with ic1:
            st.markdown("### Insider Activity")
            st.markdown(insider.get("signal","No signal"))
            inst=insider.get("institutional",{})
            st.metric("Inst. Ownership",f"{inst.get('institutional_ownership_pct',0):.1f}%")
        with ic2:
            st.markdown("### Short Interest")
            sf=short.get("short_float_pct")
            if sf: st.metric("Short Float",f"{sf:.1f}%")
            st.metric("Squeeze Score",f"{short.get('squeeze_score',0)}/100")
        with ic3:
            st.markdown("### Options Flow")
            pcr=options.get("put_call_ratio",{})
            if pcr.get("pcr_volume"):
                st.metric("Put/Call",f"{pcr['pcr_volume']:.2f}")
                st.caption(pcr.get("sentiment",""))
        with ic4:
            st.markdown("### Risk")
            st.metric("Sharpe",f"{risk.get('sharpe_ratio',0):.2f}")
            st.metric("Max Drawdown",f"{risk.get('max_drawdown',0)*100:.1f}%")
            st.metric("Beta",f"{risk.get('beta',0):.2f}")

        # Factor + Accounting
        f1,f2=st.columns(2)
        with f1:
            st.markdown("### Factor Model")
            fm=factor.get("factor_exposures",{}); sb=factor.get("smart_beta",{})
            if sb.get("classification"):
                st.markdown(f"**{sb['classification']}**")
                a1,a2=st.columns(2)
                a1.metric("Alpha",f"{fm.get('alpha_annual','-')}%/yr")
                a2.metric("R²",f"{fm.get('r_squared','-')}")
        with f2:
            st.markdown("### Accounting Quality")
            b=fraud.get("beneish",{}); az=fraud.get("altman",{})
            if b.get("m_score"):
                st.metric("Beneish M-Score",f"{b['m_score']:.2f}",delta=b.get("verdict",""),delta_color="off")
            if az.get("z_score"):
                st.metric("Altman Z-Score",f"{az['z_score']:.2f}",delta=az.get("zone",""),delta_color="off")

        cats=pt.get("catalysts",[]); rks=pt.get("risks",[])
        if cats or rks:
            cr1,cr2=st.columns(2)
            with cr1:
                if cats:
                    st.markdown("### Catalysts")
                    for c in cats[:4]: st.success(c)
            with cr2:
                if rks:
                    st.markdown("### Risk Factors")
                    for rk in rks[:4]: st.warning(rk)


# ═════════════════════════════════════════════════════════════════
#   PORTFOLIO BUILDER
# ═════════════════════════════════════════════════════════════════
elif page == "Portfolio Builder":
    st.title("Portfolio Builder")
    st.markdown("Complete the financial profile below to receive a personalized "
                "portfolio recommendation with specific holdings and projections.")
    st.markdown("---")

    with st.form("survey"):
        st.markdown("### Personal Information")
        c1,c2,c3=st.columns(3)
        with c1:
            name=st.text_input("First name",placeholder="John")
            age=st.number_input("Age",16,100,25)
        with c2:
            employment=st.selectbox("Employment",[
                "Employed (W-2)","Self-employed","Student","Retired","Between jobs"])
            experience=st.selectbox("Investment experience",[
                "None","Beginner (< 1 year)","Intermediate (1-3 years)",
                "Advanced (3+ years)","Professional"])
        with c3:
            dependents=st.selectbox("Dependents",["None","1","2-3","4+"])
            income_stability=st.selectbox("Income stability",[
                "Very stable","Mostly stable","Variable","Unstable"])

        st.markdown("---")
        st.markdown("### Financial Situation")
        c1,c2,c3=st.columns(3)
        with c1:
            annual_income=st.number_input("Annual income (pre-tax)",0,10000000,60000,5000,format="%d")
            monthly_expenses=st.number_input("Monthly expenses",0,100000,2500,250,format="%d")
        with c2:
            starting_capital=st.number_input("Amount to invest",100,50000000,10000,1000,format="%d")
            monthly_contrib=st.number_input("Monthly investment",0,100000,500,100,format="%d")
        with c3:
            net_worth=st.number_input("Net worth (excl. home)",0,100000000,25000,5000,format="%d")
            existing_inv=st.number_input("Existing portfolio",0,50000000,0,1000,format="%d")
        c1,c2=st.columns(2)
        with c1: emergency=st.radio("Emergency fund?",["Yes","Building","No"],horizontal=True)
        with c2: debt=st.radio("High-interest debt?",["None","Under $5K","$5K-$25K","$25K+"],horizontal=True)

        st.markdown("---")
        st.markdown("### Accounts")
        c1,c2=st.columns(2)
        with c1:
            account_type=st.selectbox("Account type",[
                "Taxable brokerage","Traditional IRA","Roth IRA","401(k)","Not sure"])
        with c2:
            has_401k=st.checkbox("I have a 401(k)")
            employer_match=st.checkbox("Employer matches contributions")
            has_ira=st.checkbox("I have an IRA")

        st.markdown("---")
        st.markdown("### Risk Tolerance")
        risk_tolerance=st.slider("Risk comfort (1=conservative, 10=aggressive)",1,10,5)
        market_crash=st.selectbox("Market drops 35%. Your $100K becomes $65K. You:",[
            "Sell everything","Sell half","Hold and wait","Buy more","Go all in"])
        loss_tol=st.selectbox("Max annual loss you can tolerate:",["5%","10%","20%","35%","50%+"])

        st.markdown("---")
        st.markdown("### Goals")
        c1,c2=st.columns(2)
        with c1:
            goal=st.selectbox("Primary goal",[
                "Long-term wealth","Passive income","Specific goal (house, education)",
                "Capital preservation","Aggressive growth","Speculation"])
            ret_age=st.number_input("Target retirement age",30,85,65)
        with c2:
            horizon=st.slider("Time horizon (years)",1,40,10)
            goals=st.multiselect("Saving for:",["Retirement","Home","Education",
                                                  "Financial independence","Business","General wealth"])

        st.markdown("---")
        st.markdown("### Preferences")
        c1,c2=st.columns(2)
        with c1:
            style=st.selectbox("Approach",["ETFs only","Mix of ETFs and stocks","Individual stocks"])
        with c2:
            sectors=st.multiselect("Sector interest (blank=all)",[
                "Technology","Healthcare","Financials","Energy","Consumer","Industrials","Real Estate"])
        exclusions=st.multiselect("Exclude:",["Tobacco","Weapons","Gambling","Fossil fuels","None"])

        st.markdown("---")
        st.caption("Educational portfolio suggestions only. Not licensed financial advice. "
                   "All investing involves risk including loss of principal.")
        submitted=st.form_submit_button("Build Portfolio",type="primary",use_container_width=True)

    if submitted:
        try:
            from advisor import InvestorProfile, InvestorAdvisor
            goal_map={"Long-term":"growth","Passive":"income","Specific":"balanced",
                      "Capital":"preservation","Aggressive":"growth","Speculation":"speculation"}
            acct_map={"Taxable":"taxable","Traditional":"ira","Roth":"roth_ira","401":"401k","Not":"taxable"}
            crash_map={"Sell everything":1,"Sell half":2,"Hold":3,"Buy":4,"Go":5}
            emp_map={"Employed":"employed","Self":"self_employed","Student":"student",
                     "Retired":"retired","Between":"unemployed"}
            stab_map={"Very":"stable","Mostly":"stable","Variable":"variable","Unstable":"unstable"}
            dep_map={"None":0,"1":1,"2":2,"4":4}
            debt_map={"None":0,"Under":2500,"$5K":15000,"$25K":40000}
            def m(s,d):
                for k,v in d.items():
                    if s.startswith(k): return v
                return list(d.values())[0]

            profile=InvestorProfile(
                name=name or "Investor",age=age,risk_tolerance=risk_tolerance,
                time_horizon_yrs=horizon,retirement_age=ret_age,
                primary_goal=m(goal,goal_map),specific_goals=goals,
                starting_capital=float(starting_capital),monthly_contrib=float(monthly_contrib),
                annual_income=float(annual_income),net_worth=float(net_worth),
                monthly_expenses=float(monthly_expenses),emergency_fund=(emergency=="Yes"),
                has_debt=(debt!="None"),debt_amount=m(debt,debt_map),
                has_dependents=(dependents!="None"),num_dependents=m(dependents,dep_map),
                employment_status=m(employment,emp_map),income_stability=m(income_stability,stab_map),
                has_401k=has_401k,has_ira=has_ira,employer_match=employer_match,
                existing_portfolio_value=float(existing_inv),
                prefers_etfs=style.startswith("ETF"),sector_focus=sectors,
                exclude_sectors=[s for s in exclusions if s!="None"],
                experience_level=("beginner" if "None" in experience or "Beginner" in experience
                                  else "advanced" if "Advanced" in experience or "Professional" in experience
                                  else "intermediate"),
                account_type=m(account_type,acct_map),
            )
            advisor=InvestorAdvisor()
            profile=advisor._calculate_profile(profile,m(market_crash,crash_map))

            with st.spinner("Building portfolio..."):
                screened=[]
                if not profile.prefers_etfs:
                    try:
                        from stock_screener import StockScreener
                        sc=StockScreener()
                        sp_map={"growth":"aggressive_growth","income":"dividend_growth",
                                "preservation":"deep_value","speculation":"aggressive_growth","balanced":"garp"}
                        df=sc.fetch_bulk_metrics(sc.get_sp500_tickers()[:80])
                        res=sc.screen(sp_map.get(profile.primary_goal,"garp"),df,limit=15)
                        screened=res.to_dict("records") if not res.empty else []
                    except Exception: pass
                rec=advisor.build_portfolio_recommendation(profile,screened)

            st.markdown("---")
            st.markdown(f"### Portfolio for {rec.get('investor_name','')}")
            pc1,pc2,pc3,pc4,pc5=st.columns(5)
            pc1.metric("Profile",rec.get("profile_type","").replace("_"," ").title())
            pc2.metric("Risk Score",f"{rec.get('composite_risk',5)}/10")
            pc3.metric("Age",str(profile.age))
            pc4.metric("Horizon",f"{rec.get('time_horizon_yrs',10)} yrs")
            pc5.metric("Rebalance",rec.get("rebalance","Quarterly"))
            st.info(rec.get("description",""))

            # Insights
            insights=[]
            if profile.age<30 and profile.time_horizon_yrs>20:
                insights.append("Your biggest advantage is time. With 20+ years ahead, compounding does the heavy lifting.")
            if profile.has_debt and profile.debt_amount>10000:
                insights.append(f"Consider addressing ${profile.debt_amount:,.0f} in high-interest debt first — the interest likely exceeds market returns.")
            if not profile.emergency_fund:
                insights.append(f"Build an emergency fund of ${profile.monthly_expenses*3:,.0f}-${profile.monthly_expenses*6:,.0f} before investing aggressively.")
            if profile.employer_match and not profile.has_401k:
                insights.append("Your employer matches 401(k) contributions — that is a guaranteed return. Prioritize maxing the match.")
            if profile.starting_capital>profile.net_worth*0.5 and profile.net_worth>0:
                insights.append("You are investing a large share of your net worth. Consider maintaining more liquid reserves.")
            if insights:
                st.markdown("### Key Considerations")
                for ins in insights:
                    st.markdown(f"<div class='insight-box'>{ins}</div>",unsafe_allow_html=True)
                st.markdown("")

            st.markdown("### Asset Allocation")
            alloc=rec.get("asset_allocation",{}); ad=rec.get("allocation_dollars",{})
            a1,a2,a3,a4=st.columns(4)
            a1.metric("Equities",f"{alloc.get('stocks',0)*100:.0f}%",delta=f"${ad.get('stocks',0):,.0f}")
            a2.metric("Fixed Income",f"{alloc.get('bonds',0)*100:.0f}%",delta=f"${ad.get('bonds',0):,.0f}")
            a3.metric("Alternatives",f"{alloc.get('alternatives',0)*100:.0f}%",delta=f"${ad.get('alternatives',0):,.0f}")
            a4.metric("Cash",f"{alloc.get('cash',0)*100:.0f}%",delta=f"${ad.get('cash',0):,.0f}")

            holdings=rec.get("holdings",[])
            if holdings:
                st.markdown("### Holdings")
                rows=[{"Ticker":h.get("ticker",""),"Type":h.get("type",""),
                       "Weight":f"{h.get('weight',0)*100:.1f}%",
                       "Amount":f"${h.get('dollar',0):,.2f}",
                       "Role":h.get("role","")} for h in holdings]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

            proj=rec.get("projections",{})
            if proj:
                st.markdown("### Growth Projections")
                st.caption(f"${starting_capital:,.0f} initial + ${monthly_contrib:,.0f}/mo")
                p1,p2,p3=st.columns(3)
                for col,(label,key) in zip([p1,p2,p3],[("10 Year","10y"),("20 Year","20y"),("30 Year","30y")]):
                    p=proj.get(key,{})
                    if p:
                        col.metric(f"{label}",f"${p.get('expected',0):,.0f}")
                        col.caption(f"Conservative: ${p.get('conservative',0):,.0f}")
                        col.caption(f"Optimistic: ${p.get('aggressive',0):,.0f}")

            w=rec.get("warnings",[]); t=rec.get("tax_notes",[])
            if w:
                with st.expander("Warnings"):
                    for x in w: st.warning(x)
            if t:
                with st.expander("Tax Notes"):
                    for x in t: st.info(x)
        except Exception as e:
            st.error(f"Error: {str(e)[:300]}")


# ═════════════════════════════════════════════════════════════════
#   ALPHA SIGNALS
# ═════════════════════════════════════════════════════════════════
elif page == "Alpha Signals":
    st.title("Alpha Model")
    st.markdown("Stacked ensemble of 6 ML models trained on 125+ features. "
                "Predicts 6-month outperformance vs. S&P 500.")
    st.markdown("---")
    c1,c2=st.columns([3,1])
    with c1:
        st.markdown("First run takes 10-20 minutes. The trained model saves locally for instant reuse.")
    with c2:
        top_n=st.number_input("Top picks",5,50,25)

    if st.button("Run Model",type="primary",use_container_width=True):
        try:
            from alpha_model import AlphaModel
            model=AlphaModel(target="label_6m_10pct")
            saved=os.path.join("models","alpha_v1.pkl")
            if os.path.exists(saved):
                with st.spinner("Loading model..."):
                    model.load("alpha_v1")
                with st.spinner("Scoring..."):
                    signals=model.generate_signals(top_n=top_n,verbose=False)
            else:
                p=st.empty()
                p.info("Building universe...")
                df=model.build_universe()
                if df is None or df.empty:
                    st.error("Could not build universe."); st.stop()
                p.info("Training (6 models)...")
                model.train(verbose=False)
                p.info("Validating...")
                val=model.validate(verbose=False)
                p.info("Generating signals...")
                signals=model.generate_signals(top_n=top_n,verbose=False)
                model.save("alpha_v1"); p.success("Done. Model saved.")
                if val:
                    st.markdown("### Backtest (Walk-Forward)")
                    v1,v2,v3,v4=st.columns(4)
                    v1.metric("Avg Alpha",f"{val.get('avg_alpha',0)*100:+.1f}%")
                    v2.metric("Hit Rate",f"{val.get('avg_hit_rate',0)*100:.0f}%")
                    v3.metric("Alpha Sharpe",f"{val.get('alpha_sharpe',0):.2f}")
                    v4.metric("Folds",val.get("n_folds",0))

            if signals is not None and not signals.empty:
                st.markdown(f"### Top {top_n} Picks")
                if model.regime:
                    r=model.regime
                    st.info(f"**Regime: {r['regime'].upper()}** · "
                            f"SPY 6M: {r.get('momentum_6m',0)*100:+.1f}% · "
                            f"Vol: {r.get('vol_20d',0)*100:.1f}%")
                cols=["ticker","probability","alpha_score","risk_adj_score","position_size_pct","confidence"]
                avail=[c for c in cols if c in signals.columns]
                out=signals.head(top_n)[avail].copy()
                out.columns=[c.replace("_"," ").title() for c in avail]
                for col in out.columns:
                    if any(k in col for k in ["Prob","Alpha","Risk"]):
                        out[col]=out[col].apply(lambda x:f"{x:.3f}" if pd.notna(x) else "")
                    if "Size" in col:
                        out[col]=out[col].apply(lambda x:f"{x:.1f}%" if pd.notna(x) else "")
                out.index=range(1,len(out)+1); out.index.name="#"
                st.dataframe(out,use_container_width=True)
                st.caption(f"{int((signals['probability']>0.5).sum())} buy signals / {len(signals)} scored")
                if model.feature_importances is not None:
                    with st.expander("Feature Importance"):
                        fi=model.feature_importances.head(20)
                        st.dataframe(pd.DataFrame({"Feature":fi.index,"Importance":fi.values}),
                                     use_container_width=True,hide_index=True)
        except Exception as e:
            st.error(f"Error: {str(e)[:300]}")
    elif os.path.exists(os.path.join("models","alpha_v1.pkl")):
        st.info("Trained model found. Click Run Model to load and score.")


# ═════════════════════════════════════════════════════════════════
#   SCREENER
# ═════════════════════════════════════════════════════════════════
elif page == "Screener":
    st.title("Stock Screener")
    tab1,tab2=st.tabs(["Strategy Screen","Watchlist"])

    with tab1:
        c1,c2=st.columns([3,1])
        with c1:
            profile=st.selectbox("Strategy",["garp","aggressive_growth","deep_value",
                "quality_compounder","dividend_growth","momentum","small_cap_growth"],
                format_func=lambda x:x.replace("_"," ").title())
        with c2:
            limit=st.number_input("Results",5,50,20)
        if st.button("Run Screen",type="primary"):
            try:
                from stock_screener import StockScreener,SCREEN_PROFILES
                sc=StockScreener()
                with st.spinner("Screening..."):
                    tickers=sc.get_full_universe()
                    df=sc.fetch_bulk_metrics(tickers)
                    results=sc.screen(profile,df,limit=limit)
                if not results.empty:
                    desc=SCREEN_PROFILES.get(profile,{}).get("description","")
                    st.markdown(f"### {profile.replace('_',' ').title()} — {len(results)} Results")
                    if desc: st.caption(desc)
                    st.dataframe(results,use_container_width=True,hide_index=True)
                else:
                    st.warning("No matches.")
            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")

    with tab2:
        wl=st.text_area("Tickers",value="NVDA, AAPL, MSFT, META, AMZN, GOOGL, TSLA, AMD")
        if st.button("Rank",type="primary"):
            raw=wl.replace(",","\n").replace(" ","\n")
            tickers=[t.strip().upper() for t in raw.split("\n") if t.strip()]
            if tickers:
                try:
                    from main import generate_watchlist
                    with st.spinner(f"Analyzing {len(tickers)} stocks..."):
                        ranked=generate_watchlist(tickers)
                    if ranked:
                        st.markdown(f"### Rankings — {len(ranked)} Stocks")
                        rows=[{"#":i+1,"Ticker":s.get("ticker",""),
                               "Company":s.get("company",""),"Score":s.get("composite_score",0),
                               "Signal":s.get("buy_signal",""),
                               "Target":(f"${s['price_target_12m']:.2f}" if s.get("price_target_12m") else "-"),
                               "Upside":(f"{s.get('upside_pct',0):.1f}%" if s.get("upside_pct") is not None else "")}
                              for i,s in enumerate(ranked)]
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                except Exception as e:
                    st.error(f"Error: {str(e)[:200]}")
