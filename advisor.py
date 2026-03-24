"""
advisor.py — AI Financial Advisor with Investor Profile Survey

The survey determines:
- Risk tolerance (1-10)
- Time horizon
- Investment goals (growth, income, preservation, speculation)
- Financial situation (income, net worth, emergency fund)
- Experience level
- Sector preferences / exclusions
- ETF vs stocks preference
- Starting capital

From this it builds a complete portfolio recommendation with:
- Asset allocation (stocks/bonds/alternatives)
- Specific ticker recommendations (screened and scored)
- Position sizing
- Rebalancing schedule
- Expected return range
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class InvestorProfile:
    # Identity
    name:              str   = "Investor"
    age:               int   = 30
    
    # Risk
    risk_tolerance:    int   = 5      # 1-10 (1=very conservative, 10=very aggressive)
    risk_capacity:     int   = 5      # Ability to absorb losses financially
    
    # Time
    time_horizon_yrs:  int   = 10     # Years until you need the money
    retirement_age:    int   = 65     # Target retirement age
    
    # Goals
    primary_goal:      str   = "growth"  # growth | income | preservation | speculation | balanced
    specific_goals:    list  = field(default_factory=list)  # ["retirement","house","education","wealth"]
    
    # Financial situation
    starting_capital:  float = 10000.0
    monthly_contrib:   float = 500.0
    annual_income:     float = 75000.0
    net_worth:         float = 50000.0    # Excluding primary residence
    emergency_fund:    bool  = True       # 3-6 months expenses saved?
    has_debt:          bool  = False
    debt_amount:       float = 0.0        # Total high-interest debt
    monthly_expenses:  float = 3000.0
    has_dependents:    bool  = False
    num_dependents:    int   = 0
    employment_status: str   = "employed" # employed | self_employed | retired | student
    income_stability:  str   = "stable"   # stable | variable | unstable
    
    # Existing investments
    has_401k:          bool  = False
    has_ira:           bool  = False
    employer_match:    bool  = False       # Does employer match 401k?
    existing_portfolio_value: float = 0.0
    
    # Preferences
    prefers_etfs:      bool  = False   # ETFs over individual stocks?
    sector_focus:      list  = field(default_factory=list)
    exclude_sectors:   list  = field(default_factory=list)
    exclude_tickers:   list  = field(default_factory=list)
    
    # Experience
    experience_level:  str   = "intermediate"  # beginner | intermediate | advanced
    
    # Tax
    account_type:      str   = "taxable"  # taxable | ira | 401k | roth_ira
    tax_bracket:       str   = "middle"   # low | middle | high
    
    # Derived (set automatically)
    composite_risk:    int   = 5
    profile_type:      str   = "balanced"
    asset_allocation:  dict  = field(default_factory=dict)
    years_to_retirement: int = 35
    savings_rate:      float = 0.0        # monthly_contrib / monthly_income


class InvestorAdvisor:
    """
    Runs the investor survey, builds the profile,
    and generates a complete portfolio recommendation.
    """

    PROFILE_TYPES = {
        "ultra_conservative": {
            "description": "Capital preservation with minimal risk",
            "stocks": 0.10, "bonds": 0.70, "alternatives": 0.10, "cash": 0.10,
            "stock_style": "dividend",
            "etf_mix": ["BND","AGG","SHY","VYM","TIPS","LQD"],
        },
        "conservative": {
            "description": "Income-focused with low risk",
            "stocks": 0.30, "bonds": 0.55, "alternatives": 0.10, "cash": 0.05,
            "stock_style": "dividend",
            "etf_mix": ["BND","VYM","SCHD","AGG","DVY","VTV","HDV"],
        },
        "moderate_conservative": {
            "description": "Balanced with defensive tilt",
            "stocks": 0.50, "bonds": 0.35, "alternatives": 0.10, "cash": 0.05,
            "stock_style": "quality_compounder",
            "etf_mix": ["VOO","BND","VYM","VTV","SCHD","VEA","AGG"],
        },
        "balanced": {
            "description": "Classic 60/40 portfolio",
            "stocks": 0.60, "bonds": 0.30, "alternatives": 0.08, "cash": 0.02,
            "stock_style": "garp",
            "etf_mix": ["VOO","QQQ","VTI","BND","VEA","VWO","GLD"],
        },
        "moderate_aggressive": {
            "description": "Growth-oriented with some stability",
            "stocks": 0.75, "bonds": 0.15, "alternatives": 0.10, "cash": 0.00,
            "stock_style": "garp",
            "etf_mix": ["VOO","QQQ","VTI","VUG","VEA","XLK","GLD"],
        },
        "aggressive": {
            "description": "Maximum growth, higher volatility",
            "stocks": 0.90, "bonds": 0.05, "alternatives": 0.05, "cash": 0.00,
            "stock_style": "aggressive_growth",
            "etf_mix": ["QQQ","SOXX","XLK","VUG","VTI","ARKK","BOTZ"],
        },
        "ultra_aggressive": {
            "description": "Speculative, all-in on growth and themes",
            "stocks": 0.95, "bonds": 0.00, "alternatives": 0.05, "cash": 0.00,
            "stock_style": "aggressive_growth",
            "etf_mix": ["QQQ","SOXX","ARKK","BOTZ","CIBR","AIQ","BITO"],
        },
        "income": {
            "description": "Dividend income and yield focus",
            "stocks": 0.55, "bonds": 0.35, "alternatives": 0.10, "cash": 0.00,
            "stock_style": "dividend_growth",
            "etf_mix": ["VYM","SCHD","DVY","HDV","DGRO","BND","LQD"],
        },
    }

    def run_survey(self) -> InvestorProfile:
        """Interactive command-line survey."""
        print("\n" + "="*60)
        print("  💼 INSTITUTIONALEDGE — FINANCIAL ADVISOR")
        print("  Personal Investment Profile Survey")
        print("="*60)
        print("  Answer each question honestly for the best results.")
        print("  Takes about 2 minutes.\n")

        profile = InvestorProfile()

        # ── Personal ─────────────────────────────────────────────
        print("─"*50)
        print("SECTION 1: ABOUT YOU")
        print("─"*50)
        profile.name = input("  What's your first name? ").strip() or "Investor"

        exp = self._ask_choice(
            "  Experience level?",
            ["Beginner (learning basics)", "Intermediate (some experience)",
             "Advanced (active investor/trader)"],
            ["beginner", "intermediate", "advanced"]
        )
        profile.experience_level = exp

        # ── Risk ─────────────────────────────────────────────────
        print("\n─"*50)
        print("SECTION 2: RISK TOLERANCE")
        print("─"*50)
        print("  On a scale of 1-10, where:")
        print("  1 = I can't sleep if my portfolio drops 5%")
        print("  5 = I can handle 20-30% drops without panic-selling")
        print("  10 = I'm fine watching my portfolio drop 50%+ for future gains")

        risk_tol = self._ask_int("  Your risk tolerance (1-10): ", 1, 10, 5)
        profile.risk_tolerance = risk_tol

        print("\n  Scenario: The market drops 30% tomorrow. You:")
        scenario = self._ask_choice(
            "  What do you do?",
            ["Sell everything — I need to protect what I have",
             "Sell some — reduce exposure",
             "Hold — stay the course",
             "Buy more — this is a buying opportunity",
             "Go all-in — I love discounts"],
            [1, 2, 3, 4, 5]
        )
        risk_behavior = scenario

        # ── Financial Situation ───────────────────────────────────
        print("\n─"*50)
        print("SECTION 3: FINANCIAL SITUATION")
        print("─"*50)
        ef = self._ask_yes_no("  Do you have 3-6 months of expenses saved as emergency fund?")
        profile.emergency_fund = ef

        debt = self._ask_yes_no("  Do you have high-interest debt (credit cards, personal loans)?")
        profile.has_debt = debt

        capital_str = input("  How much are you starting with? (e.g. 1000, 10000, 50000) $").strip()
        try:
            profile.starting_capital = float(capital_str.replace(",","").replace("$",""))
        except Exception:
            profile.starting_capital = 10000

        contrib_str = input("  How much can you invest monthly? $").strip()
        try:
            profile.monthly_contrib = float(contrib_str.replace(",","").replace("$",""))
        except Exception:
            profile.monthly_contrib = 0

        account = self._ask_choice(
            "  Account type?",
            ["Taxable brokerage", "Traditional IRA", "Roth IRA", "401(k)"],
            ["taxable", "ira", "roth_ira", "401k"]
        )
        profile.account_type = account

        # ── Time Horizon ──────────────────────────────────────────
        print("\n─"*50)
        print("SECTION 4: GOALS & TIME HORIZON")
        print("─"*50)
        horizon = self._ask_int("  When do you need this money? (years from now): ", 1, 50, 10)
        profile.time_horizon_yrs = horizon

        goal = self._ask_choice(
            "  Primary investment goal?",
            ["Build wealth for retirement",
             "Generate passive income now",
             "Preserve capital (protect what I have)",
             "Speculate — I want high-risk high-reward",
             "All of the above (balanced)"],
            ["growth", "income", "preservation", "speculation", "balanced"]
        )
        profile.primary_goal = goal

        # ── Preferences ───────────────────────────────────────────
        print("\n─"*50)
        print("SECTION 5: PREFERENCES")
        print("─"*50)
        prefer_etfs = self._ask_choice(
            "  Investment style preference?",
            ["Individual stocks — I want to pick winners",
             "ETFs only — I want diversification with less work",
             "Mix — some ETFs as core, some individual stocks"],
            ["stocks", "etfs", "mixed"]
        )
        profile.prefers_etfs = (prefer_etfs == "etfs")

        if prefer_etfs != "etfs":
            print("\n  Sectors you're most interested in (enter numbers, comma-separated):")
            sector_choices = [
                "Technology", "Healthcare", "Financials", "Energy",
                "Consumer", "Industrials", "Real Estate", "All sectors"
            ]
            for i, s in enumerate(sector_choices, 1):
                print(f"  {i}. {s}")
            sec_input = input("  Your choices (e.g. 1,3,8): ").strip()
            try:
                idxs = [int(x.strip())-1 for x in sec_input.split(",") if x.strip()]
                selected = [sector_choices[i] for i in idxs if 0 <= i < len(sector_choices)]
                if "All sectors" in selected or not selected:
                    profile.sector_focus = []
                else:
                    profile.sector_focus = selected
            except Exception:
                profile.sector_focus = []

        exclude = input("\n  Any sectors/stocks to exclude? (e.g. tobacco, gambling, or leave blank): ").strip()
        if exclude:
            profile.exclude_sectors = [s.strip() for s in exclude.split(",")]

        # ── Build Profile ─────────────────────────────────────────
        profile = self._calculate_profile(profile, risk_behavior)

        print("\n" + "="*60)
        print(f"  ✅ Profile complete for {profile.name}!")
        print(f"  Profile type: {profile.profile_type.upper().replace('_',' ')}")
        print(f"  Composite risk score: {profile.composite_risk}/10")
        print("="*60)

        return profile

    def _calculate_profile(self, profile: InvestorProfile, risk_behavior: int) -> InvestorProfile:
        """Calculate composite risk and profile type from all survey answers."""

        # ── Derived fields ───────────────────────────────────────
        profile.years_to_retirement = max(0, profile.retirement_age - profile.age)
        if profile.annual_income > 0:
            profile.savings_rate = (profile.monthly_contrib * 12) / profile.annual_income
        
        # Tax bracket estimate
        inc = profile.annual_income
        if inc < 45000:
            profile.tax_bracket = "low"
        elif inc < 100000:
            profile.tax_bracket = "middle"
        elif inc < 200000:
            profile.tax_bracket = "upper_middle"
        else:
            profile.tax_bracket = "high"

        # ── Auto-calculate time horizon from age if not set manually ──
        if profile.time_horizon_yrs == 10 and profile.years_to_retirement > 0:
            # Default to years-to-retirement unless user set it differently
            profile.time_horizon_yrs = profile.years_to_retirement

        # ── Composite risk score ─────────────────────────────────
        # Start with stated risk tolerance
        base_risk = profile.risk_tolerance

        # Age adjustment: younger = can afford more risk
        if profile.age < 25:
            age_adj = 1.5
        elif profile.age < 35:
            age_adj = 1.0
        elif profile.age < 45:
            age_adj = 0.0
        elif profile.age < 55:
            age_adj = -0.5
        elif profile.age < 65:
            age_adj = -1.0
        else:
            age_adj = -2.0

        # Behavior adjustment (how they react to crashes)
        behavior_adj = {1: -2, 2: -1, 3: 0, 4: 1, 5: 2}.get(risk_behavior, 0) * 0.5

        # Time horizon: more time = can take more risk
        horizon_adj = min((profile.time_horizon_yrs - 5) / 5, 2) * 0.5

        # Financial cushion: emergency fund, debt, dependents
        debt_adj = -1.5 if profile.has_debt else 0
        ef_adj = -1 if not profile.emergency_fund else 0
        dependent_adj = -0.5 * min(profile.num_dependents, 4)

        # Income stability
        stability_adj = {"stable": 0, "variable": -0.5, "unstable": -1.5}.get(
            profile.income_stability, 0)

        # Net worth relative to investment (can they afford to lose this?)
        if profile.net_worth > 0 and profile.starting_capital > 0:
            invest_pct = profile.starting_capital / profile.net_worth
            if invest_pct > 0.5:
                cushion_adj = -1.0  # Investing most of their net worth — reduce risk
            elif invest_pct > 0.3:
                cushion_adj = -0.5
            else:
                cushion_adj = 0.5   # This is a small fraction — can afford more risk
        else:
            cushion_adj = 0

        # Goal adjustment
        goal_adj = {"speculation": 2, "growth": 1, "balanced": 0,
                     "income": -1, "preservation": -2}.get(profile.primary_goal, 0) * 0.5

        # Composite
        composite = round(
            base_risk + age_adj + behavior_adj + horizon_adj +
            debt_adj + ef_adj + dependent_adj + stability_adj +
            cushion_adj + goal_adj
        )
        profile.composite_risk = max(1, min(10, composite))

        # Risk capacity (separate from willingness — ability to absorb losses)
        capacity_score = 5
        if profile.age < 35 and profile.emergency_fund and not profile.has_debt:
            capacity_score = 8
        elif profile.age > 55 or profile.has_dependents:
            capacity_score = 4
        if profile.net_worth > 500000:
            capacity_score = min(capacity_score + 2, 10)
        if profile.income_stability == "unstable":
            capacity_score = max(capacity_score - 2, 1)
        profile.risk_capacity = capacity_score

        # If risk capacity is much lower than tolerance, cap it
        # (they WANT risk but CAN'T afford it)
        if profile.risk_capacity < profile.composite_risk - 2:
            profile.composite_risk = profile.risk_capacity + 1

        # ── Map to profile type ──────────────────────────────────
        cr = profile.composite_risk
        if cr <= 2:
            ptype = "ultra_conservative"
        elif cr <= 3:
            ptype = "conservative"
        elif cr <= 4:
            ptype = "moderate_conservative"
        elif cr <= 5:
            ptype = "balanced"
        elif cr <= 6:
            ptype = "moderate_aggressive"
        elif cr <= 7:
            ptype = "aggressive"
        elif cr <= 8:
            ptype = "aggressive"
        else:
            ptype = "ultra_aggressive"

        # Override for specific goals / life situations
        if profile.primary_goal == "income" and cr <= 6:
            ptype = "income"
        if profile.primary_goal == "preservation" and cr <= 4:
            ptype = "conservative"
        if profile.age >= 60 and cr <= 5:
            ptype = "conservative"  # Near retirement, be careful
        if profile.age < 25 and profile.primary_goal == "growth" and cr >= 5:
            ptype = "aggressive"  # Young with long horizon

        profile.profile_type = ptype
        profile.asset_allocation = self.PROFILE_TYPES[ptype]

        return profile

    def build_portfolio_recommendation(self, profile: InvestorProfile,
                                        screened_stocks: list = None) -> dict:
        """
        Build a complete portfolio recommendation from the investor profile.
        """
        alloc   = self.PROFILE_TYPES.get(profile.profile_type, self.PROFILE_TYPES["balanced"])
        capital = profile.starting_capital
        monthly = profile.monthly_contrib

        # Asset allocation in dollars
        stock_dollar  = capital * alloc["stocks"]
        bond_dollar   = capital * alloc["bonds"]
        alt_dollar    = capital * alloc["alternatives"]
        cash_dollar   = capital * alloc.get("cash", 0)

        # Build holdings
        holdings = []

        if profile.prefers_etfs or profile.primary_goal == "preservation":
            holdings = self._build_etf_portfolio(profile, alloc, capital)
        else:
            holdings = self._build_mixed_portfolio(profile, alloc, capital, screened_stocks)

        # Project future value
        projections = self._project_growth(capital, monthly, profile)

        # Rebalancing schedule
        rebalance_freq = "Quarterly" if profile.composite_risk <= 5 else "Semi-annually"

        return {
            "investor_name":     profile.name,
            "profile_type":      profile.profile_type,
            "description":       alloc.get("description",""),
            "composite_risk":    profile.composite_risk,
            "time_horizon_yrs":  profile.time_horizon_yrs,
            "primary_goal":      profile.primary_goal,
            "starting_capital":  capital,
            "monthly_contrib":   monthly,
            "asset_allocation": {
                "stocks":       alloc["stocks"],
                "bonds":        alloc["bonds"],
                "alternatives": alloc["alternatives"],
                "cash":         alloc.get("cash", 0),
            },
            "allocation_dollars": {
                "stocks":       round(stock_dollar, 2),
                "bonds":        round(bond_dollar, 2),
                "alternatives": round(alt_dollar, 2),
                "cash":         round(cash_dollar, 2),
            },
            "holdings":          holdings,
            "projections":       projections,
            "rebalance":         rebalance_freq,
            "tax_notes":         self._tax_notes(profile),
            "warnings":          self._warnings(profile),
        }

    def _build_etf_portfolio(self, profile: InvestorProfile,
                              alloc: dict, capital: float) -> list:
        """Build ETF-only portfolio."""
        etfs     = alloc.get("etf_mix", ["VOO","BND","GLD"])
        holdings = []
        n        = len(etfs)

        # Weight distribution (not equal — core gets more)
        weights  = self._smart_etf_weights(etfs, profile)

        for etf, weight in zip(etfs, weights):
            dollar_amount = capital * alloc["stocks"] * weight  # within equity allocation
            holdings.append({
                "ticker":      etf,
                "type":        "ETF",
                "weight":      round(weight, 3),
                "dollar":      round(dollar_amount, 2),
                "role":        self._etf_role(etf),
                "rationale":   self._etf_rationale(etf, profile),
            })

        # Add bond ETF
        if alloc["bonds"] > 0:
            bond_etf = "BND" if profile.account_type in ["ira","roth_ira"] else "SCHD"
            holdings.append({
                "ticker":    bond_etf if alloc["bonds"] >= 0.15 else "SHY",
                "type":      "ETF",
                "weight":    alloc["bonds"],
                "dollar":    round(capital * alloc["bonds"], 2),
                "role":      "Fixed Income / Stability",
                "rationale": "Reduces portfolio volatility and provides income",
            })

        # Add alternatives
        if alloc["alternatives"] > 0:
            holdings.append({
                "ticker":    "GLD",
                "type":      "ETF",
                "weight":    alloc["alternatives"],
                "dollar":    round(capital * alloc["alternatives"], 2),
                "role":      "Inflation Hedge / Alternative",
                "rationale": "Gold as portfolio hedge against inflation and tail risk",
            })

        return holdings

    def _build_mixed_portfolio(self, profile: InvestorProfile, alloc: dict,
                                capital: float, screened_stocks: list = None) -> list:
        """Build mixed ETF + individual stock portfolio."""
        holdings = []
        stock_capital = capital * alloc["stocks"]

        # Core ETF (40% of equity)
        core_etf_pct  = 0.40
        stock_pick_pct = 0.60

        core_etf = "VOO" if profile.composite_risk <= 6 else "QQQ"
        holdings.append({
            "ticker":    core_etf,
            "type":      "ETF",
            "weight":    alloc["stocks"] * core_etf_pct,
            "dollar":    round(stock_capital * core_etf_pct, 2),
            "role":      "Core Market Exposure",
            "rationale": f"{'S&P 500' if core_etf=='VOO' else 'NASDAQ-100'} index core — low-cost, diversified base",
        })

        # Individual stocks
        if screened_stocks:
            # Filter by sector preference
            candidates = screened_stocks
            if profile.sector_focus:
                pref_candidates = [s for s in candidates
                                   if any(sec.lower() in str(s.get("sector","")).lower()
                                          for sec in profile.sector_focus)]
                if len(pref_candidates) >= 3:
                    candidates = pref_candidates

            # Filter exclusions
            if profile.exclude_sectors:
                candidates = [s for s in candidates
                              if not any(ex.lower() in str(s.get("sector","")).lower()
                                         for ex in profile.exclude_sectors)]
            if profile.exclude_tickers:
                candidates = [s for s in candidates
                              if s.get("ticker") not in profile.exclude_tickers]

            # Pick top N
            n_stocks   = min(8, max(3, len(candidates)))
            candidates = candidates[:n_stocks]
            total_stock_dollars = stock_capital * stock_pick_pct

            # Tapered weights that sum exactly to 1.0
            raw_tapers = [1 - (i * 0.08) for i in range(n_stocks)]
            raw_tapers = [max(t, 0.5) for t in raw_tapers]
            taper_sum  = sum(raw_tapers)
            norm_tapers = [t / taper_sum for t in raw_tapers]

            for i, stock in enumerate(candidates):
                dollar = total_stock_dollars * norm_tapers[i]
                weight = dollar / capital

                holdings.append({
                    "ticker":    stock.get("ticker", ""),
                    "name":      stock.get("name",""),
                    "type":      "Stock",
                    "sector":    stock.get("sector",""),
                    "weight":    round(weight, 3),
                    "dollar":    round(dollar, 2),
                    "role":      "Active Selection",
                    "score":     stock.get("screen_score", 0),
                    "pe":        stock.get("pe_ratio"),
                    "rev_growth": stock.get("revenue_growth"),
                    "analyst_upside": stock.get("analyst_upside"),
                    "rationale": self._stock_rationale(stock, profile),
                })
        else:
            # No screened stocks — use sector ETFs based on preference
            sector_etf_map = {
                "Technology":   "XLK", "Healthcare": "XLV",
                "Financials":   "XLF", "Energy":     "XLE",
                "Consumer":     "XLY", "Industrials":"XLI",
            }
            sectors   = profile.sector_focus[:3] if profile.sector_focus else ["Technology","Healthcare","Financials"]
            per_sector = stock_capital * stock_pick_pct / max(len(sectors), 1)
            for sec in sectors:
                etf = sector_etf_map.get(sec, "XLK")
                holdings.append({
                    "ticker":    etf,
                    "type":      "Sector ETF",
                    "weight":    round(per_sector / capital, 3),
                    "dollar":    round(per_sector, 2),
                    "role":      f"{sec} Exposure",
                    "rationale": f"Sector ETF for {sec} exposure without single-stock risk",
                })

        # Bonds
        if alloc["bonds"] > 0:
            bond_etf = "BND" if alloc["bonds"] >= 0.20 else "SHY"
            holdings.append({
                "ticker":    bond_etf,
                "type":      "ETF",
                "weight":    alloc["bonds"],
                "dollar":    round(capital * alloc["bonds"], 2),
                "role":      "Fixed Income",
                "rationale": "Bond allocation for stability and drawdown protection",
            })

        # Alternatives
        if alloc["alternatives"] > 0:
            holdings.append({
                "ticker":    "GLD",
                "type":      "ETF",
                "weight":    alloc["alternatives"],
                "dollar":    round(capital * alloc["alternatives"], 2),
                "role":      "Alternative / Hedge",
                "rationale": "Gold hedge for inflation and macro uncertainty",
            })

        return holdings

    def _project_growth(self, capital: float, monthly: float,
                         profile: InvestorProfile) -> dict:
        """Project portfolio value over time at different CAGR scenarios."""
        projections   = {}
        profile_cagrs = {
            "ultra_conservative":    {"bear": 0.02, "base": 0.04, "bull": 0.06},
            "conservative":          {"bear": 0.02, "base": 0.05, "bull": 0.08},
            "moderate_conservative": {"bear": 0.03, "base": 0.06, "bull": 0.10},
            "balanced":              {"bear": 0.02, "base": 0.07, "bull": 0.12},
            "moderate_aggressive":   {"bear": 0.00, "base": 0.09, "bull": 0.15},
            "aggressive":            {"bear": -0.05,"base": 0.10, "bull": 0.18},
            "ultra_aggressive":      {"bear": -0.10,"base": 0.12, "bull": 0.25},
            "income":                {"bear": 0.02, "base": 0.06, "bull": 0.09},
        }
        cagrs = profile_cagrs.get(profile.profile_type, {"bear":0.04,"base":0.07,"bull":0.12})

        for scenario, cagr in cagrs.items():
            vals = {}
            for yr in [1, 3, 5, 10, 20, 30]:
                if yr <= profile.time_horizon_yrs + 5:
                    # FV = PV*(1+r)^n + PMT*[((1+r)^n - 1)/r]
                    pv_fv  = capital * (1 + cagr) ** yr
                    if cagr != 0:
                        pmt_fv = monthly * 12 * (((1 + cagr) ** yr - 1) / cagr)
                    else:
                        pmt_fv = monthly * 12 * yr
                    total = pv_fv + pmt_fv
                    vals[f"{yr}yr"] = round(total, 2)
            projections[scenario] = {"cagr": cagr, "values": vals}

        return projections

    def _smart_etf_weights(self, etfs: list, profile: InvestorProfile) -> list:
        """Non-equal weights — core gets more allocation."""
        n = len(etfs)
        if n == 0:
            return []
        # Taper weights: first ETF gets most
        raw = [1.0 - i * (0.5/n) for i in range(n)]
        total = sum(raw)
        return [r/total for r in raw]

    def _etf_role(self, etf: str) -> str:
        roles = {
            "VOO":"Core — S&P 500", "VTI":"Core — Total Market", "QQQ":"Core — NASDAQ Growth",
            "BND":"Fixed Income", "AGG":"Fixed Income", "TLT":"Long-Term Bonds",
            "VYM":"Dividend Income", "SCHD":"Dividend Growth", "GLD":"Gold Hedge",
            "VEA":"International Developed", "VWO":"Emerging Markets",
            "XLK":"Tech Sector", "XLF":"Financials","XLV":"Healthcare",
            "ARKK":"Innovation/Thematic", "SOXX":"Semiconductors","BOTZ":"AI & Robotics",
        }
        return roles.get(etf, "Portfolio Diversifier")

    def _etf_rationale(self, etf: str, profile: InvestorProfile) -> str:
        rationales = {
            "VOO": "Broad S&P 500 exposure at 0.03% expense ratio — Warren Buffett's recommendation",
            "QQQ": "Top 100 NASDAQ companies — tech-heavy growth at reasonable cost",
            "SCHD": "High-quality dividend growth stocks — consistent income with inflation protection",
            "BND": "Total US bond market — ballast against stock market volatility",
            "GLD": "Physical gold ETF — hedge against inflation and dollar weakness",
            "VEA": "International developed markets — geographic diversification",
            "SOXX": "Semiconductor sector — direct AI infrastructure exposure",
            "ARKK": "High-conviction innovation bets — high risk/reward thematic",
        }
        return rationales.get(etf, f"{etf} — portfolio diversifier for your {profile.profile_type} profile")

    def _stock_rationale(self, stock: dict, profile: InvestorProfile) -> str:
        rg  = (stock.get("revenue_growth") or 0) * 100
        pe  = stock.get("pe_ratio")
        up  = (stock.get("analyst_upside") or 0) * 100
        parts = []
        if rg > 15: parts.append(f"{rg:.0f}% revenue growth")
        if pe:      parts.append(f"P/E {pe:.0f}x")
        if up > 10: parts.append(f"analysts see {up:.0f}% upside")
        return ", ".join(parts) if parts else "Selected by institutional screening criteria"

    def _tax_notes(self, profile: InvestorProfile) -> list:
        notes = []
        if profile.account_type == "taxable":
            notes.append("In a taxable account — hold ETFs >1 year for long-term capital gains rates")
            notes.append("Consider tax-loss harvesting opportunities at year-end")
            notes.append("Dividend-paying stocks are less tax-efficient in taxable accounts")
        elif profile.account_type in ["ira","roth_ira"]:
            notes.append("Tax-advantaged account — can hold dividend stocks and bonds without tax drag")
            notes.append("Roth IRA: pay taxes now, all growth is tax-free" if profile.account_type == "roth_ira" else
                         "Traditional IRA: tax-deferred — consider Roth conversion if income is low this year")
        return notes

    def _warnings(self, profile: InvestorProfile) -> list:
        warnings = []
        if profile.has_debt:
            warnings.append("⚠️  HIGH-INTEREST DEBT: Consider paying off credit cards/personal loans before investing. A guaranteed 20%+ return beats most investments.")
        if not profile.emergency_fund:
            warnings.append("⚠️  NO EMERGENCY FUND: Build 3-6 months of expenses in a high-yield savings account FIRST before investing.")
        if profile.starting_capital < 1000:
            warnings.append("⚠️  SMALL CAPITAL: Start with low-cost ETFs like VOO — avoid individual stocks until you have $5,000+")
        if profile.composite_risk >= 8 and profile.time_horizon_yrs <= 3:
            warnings.append("⚠️  MISMATCH: Very aggressive portfolio with short time horizon is dangerous. You may need this money during a downturn.")
        return warnings

    # ── HELPERS ──────────────────────────────────────────────────

    def _ask_choice(self, prompt: str, options: list, values: list):
        print(f"\n  {prompt}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        while True:
            try:
                choice = int(input("  Enter number: ").strip())
                if 1 <= choice <= len(options):
                    return values[choice-1]
            except Exception:
                pass
            print(f"  Please enter a number 1-{len(options)}")

    def _ask_int(self, prompt: str, min_val: int, max_val: int, default: int) -> int:
        while True:
            try:
                val = input(f"  {prompt}").strip()
                if not val:
                    return default
                val = int(val)
                if min_val <= val <= max_val:
                    return val
            except Exception:
                pass
            print(f"  Please enter a number between {min_val} and {max_val}")

    def _ask_yes_no(self, prompt: str) -> bool:
        ans = input(f"  {prompt} (y/n): ").strip().lower()
        return ans.startswith("y")

    def print_recommendation(self, rec: dict):
        """Print portfolio recommendation to terminal."""
        print(f"\n{'='*62}")
        print(f"  💼 PORTFOLIO RECOMMENDATION FOR {rec['investor_name'].upper()}")
        print(f"{'='*62}")
        print(f"  Profile:     {rec['profile_type'].upper().replace('_',' ')}")
        print(f"  Description: {rec['description']}")
        print(f"  Risk Score:  {rec['composite_risk']}/10")
        print(f"  Goal:        {rec['primary_goal'].upper()}")
        print(f"  Horizon:     {rec['time_horizon_yrs']} years")

        # Warnings
        for w in rec.get("warnings", []):
            print(f"\n  {w}")

        # Allocation
        print(f"\n  {'─'*56}")
        print(f"  ASSET ALLOCATION (Starting Capital: ${rec['starting_capital']:,.0f})")
        print(f"  {'─'*56}")
        alloc   = rec["asset_allocation"]
        dollars = rec["allocation_dollars"]
        for asset, pct in alloc.items():
            print(f"  {asset.title():<16} {pct*100:.0f}%   ${dollars.get(asset,0):>10,.0f}")

        # Holdings
        print(f"\n  {'─'*56}")
        print(f"  PORTFOLIO HOLDINGS")
        print(f"  {'─'*56}")
        print(f"  {'Ticker':<8} {'Type':<12} {'Weight':<8} {'Amount':<12} {'Role'}")
        print("  " + "─"*56)
        for h in rec.get("holdings",[]):
            print(f"  {h['ticker']:<8} {h['type']:<12} {h['weight']*100:<8.1f}% ${h['dollar']:<11,.0f} {h.get('role','')[:30]}")
            if h.get("rationale"):
                print(f"    ↳ {h['rationale'][:70]}")

        # Projections
        print(f"\n  {'─'*56}")
        print(f"  PORTFOLIO PROJECTIONS (${rec['monthly_contrib']:,.0f}/month contributions)")
        print(f"  {'─'*56}")
        proj = rec.get("projections", {})
        print(f"  {'Year':<8} {'Bear Case':<15} {'Base Case':<15} {'Bull Case'}")
        print("  " + "─"*50)
        years = ["1yr","3yr","5yr","10yr","20yr"]
        for yr in years:
            bear = proj.get("bear",{}).get("values",{}).get(yr,0)
            base = proj.get("base",{}).get("values",{}).get(yr,0)
            bull = proj.get("bull",{}).get("values",{}).get(yr,0)
            if base > 0:
                print(f"  {yr:<8} ${bear:<14,.0f} ${base:<14,.0f} ${bull:,.0f}")

        bear_cagr = proj.get("bear",{}).get("cagr",0)
        base_cagr = proj.get("base",{}).get("cagr",0)
        bull_cagr = proj.get("bull",{}).get("cagr",0)
        print(f"\n  CAGR assumptions: Bear {bear_cagr*100:.0f}% | Base {base_cagr*100:.0f}% | Bull {bull_cagr*100:.0f}%")

        # Tax notes
        if rec.get("tax_notes"):
            print(f"\n  {'─'*56}")
            print(f"  TAX CONSIDERATIONS")
            print(f"  {'─'*56}")
            for note in rec["tax_notes"]:
                print(f"  📋 {note}")

        print(f"\n  {'─'*56}")
        print(f"  Rebalance: {rec['rebalance']}")
        print(f"\n  ⚠  For educational purposes only. Not financial advice.")
        print(f"     Consult a licensed financial advisor for personalized advice.")
        print(f"{'='*62}\n")
