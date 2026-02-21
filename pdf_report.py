"""
pdf_report.py — Institutional Research Report Generator

Generates a professional PDF report for any stock analysis.
Looks like a real sell-side equity research report.

Uses reportlab (free, pure Python).
Install: pip install reportlab
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os


# ── COLOR PALETTE ─────────────────────────────────────────────────
DARK_BLUE    = colors.HexColor("#0D1B2A")
MID_BLUE     = colors.HexColor("#1B3A5C")
ACCENT_BLUE  = colors.HexColor("#1E88E5")
GREEN        = colors.HexColor("#00C853")
RED          = colors.HexColor("#FF1744")
YELLOW       = colors.HexColor("#FFD600")
LIGHT_GRAY   = colors.HexColor("#F5F5F5")
MID_GRAY     = colors.HexColor("#BDBDBD")
DARK_GRAY    = colors.HexColor("#424242")
WHITE        = colors.white


class ResearchReportGenerator:

    def generate(self, analysis: dict, output_path: str = None) -> str:
        """
        Generate a full institutional research report PDF.
        analysis = result dict from analyze_stock()
        Returns: path to generated PDF file
        """
        ticker  = analysis.get("ticker", "UNKNOWN")
        company = analysis.get("company_name", ticker)

        if output_path is None:
            output_path = f"{ticker}_research_report.pdf"

        print(f"  Generating PDF report for {ticker}...")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.65*inch,
            leftMargin=0.65*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )

        styles = self._build_styles()
        story  = []

        # ── Cover Section ─────────────────────────────────────────
        story += self._cover(analysis, styles)
        story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE))
        story.append(Spacer(1, 8))

        # ── Key Stats Bar ─────────────────────────────────────────
        story += self._key_stats_bar(analysis, styles)
        story.append(Spacer(1, 10))

        # ── Investment Thesis ─────────────────────────────────────
        story += self._thesis_section(analysis, styles)

        # ── Price Target ──────────────────────────────────────────
        story += self._price_target_section(analysis, styles)

        # ── Score Dashboard ───────────────────────────────────────
        story += self._score_dashboard(analysis, styles)

        # ── Fundamental Analysis ──────────────────────────────────
        story += self._fundamental_section(analysis, styles)

        # ── Accounting Quality (Beneish / Altman) ─────────────────
        story += self._accounting_section(analysis, styles)

        # ── Competitive Analysis ──────────────────────────────────
        story += self._competitive_section(analysis, styles)

        # ── Technical Analysis ────────────────────────────────────
        story += self._technical_section(analysis, styles)

        # ── Insider & Institutional ───────────────────────────────
        story += self._insider_section(analysis, styles)

        # ── Short Interest & Options ──────────────────────────────
        story += self._short_options_section(analysis, styles)

        # ── Simulations ───────────────────────────────────────────
        story += self._simulation_section(analysis, styles)

        # ── Macro Regime ──────────────────────────────────────────
        story += self._macro_section(analysis, styles)

        # ── Catalysts & Risks ─────────────────────────────────────
        story += self._catalyst_risk_section(analysis, styles)

        # ── Disclaimer ────────────────────────────────────────────
        story += self._disclaimer(styles)

        doc.build(story, onFirstPage=self._header_footer,
                  onLaterPages=self._header_footer)

        print(f"  ✅ Report saved: {output_path}")
        return output_path

    # ── STYLES ───────────────────────────────────────────────────

    def _build_styles(self) -> dict:
        base = getSampleStyleSheet()
        return {
            "cover_title": ParagraphStyle("cover_title",
                fontName="Helvetica-Bold", fontSize=22,
                textColor=DARK_BLUE, leading=26, spaceAfter=4),
            "cover_sub": ParagraphStyle("cover_sub",
                fontName="Helvetica", fontSize=12,
                textColor=DARK_GRAY, leading=16, spaceAfter=2),
            "section_header": ParagraphStyle("section_header",
                fontName="Helvetica-Bold", fontSize=11,
                textColor=WHITE, leading=14,
                backColor=MID_BLUE, borderPadding=(4,6,4,6)),
            "body": ParagraphStyle("body",
                fontName="Helvetica", fontSize=9,
                textColor=DARK_GRAY, leading=13, spaceAfter=4),
            "body_bold": ParagraphStyle("body_bold",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=DARK_BLUE, leading=13),
            "small": ParagraphStyle("small",
                fontName="Helvetica", fontSize=7.5,
                textColor=DARK_GRAY, leading=11),
            "signal_buy": ParagraphStyle("signal_buy",
                fontName="Helvetica-Bold", fontSize=14,
                textColor=WHITE, leading=18, alignment=TA_CENTER),
            "metric_label": ParagraphStyle("metric_label",
                fontName="Helvetica", fontSize=7.5,
                textColor=DARK_GRAY, alignment=TA_CENTER),
            "metric_value": ParagraphStyle("metric_value",
                fontName="Helvetica-Bold", fontSize=11,
                textColor=DARK_BLUE, alignment=TA_CENTER),
            "disclaimer": ParagraphStyle("disclaimer",
                fontName="Helvetica", fontSize=7,
                textColor=MID_GRAY, leading=10),
            "thesis": ParagraphStyle("thesis",
                fontName="Helvetica", fontSize=9.5,
                textColor=DARK_GRAY, leading=14,
                leftIndent=10, rightIndent=10,
                backColor=LIGHT_GRAY, borderPadding=(8,8,8,8)),
        }

    # ── PAGE TEMPLATE ────────────────────────────────────────────

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        w, h = letter

        # Top bar
        canvas.setFillColor(DARK_BLUE)
        canvas.rect(0, h-0.45*inch, w, 0.45*inch, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(0.65*inch, h-0.28*inch, "INSTITUTIONALEDGE")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w-0.65*inch, h-0.28*inch,
            f"CONFIDENTIAL — FOR EDUCATIONAL PURPOSES ONLY")

        # Footer
        canvas.setFillColor(MID_GRAY)
        canvas.rect(0, 0, w, 0.35*inch, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(0.65*inch, 0.13*inch,
            f"Generated {datetime.now().strftime('%B %d, %Y %H:%M')} | InstitutionalEdge v4.0")
        canvas.drawRightString(w-0.65*inch, 0.13*inch,
            f"Page {doc.page}")

        canvas.restoreState()

    # ── COVER ────────────────────────────────────────────────────

    def _cover(self, a: dict, s: dict) -> list:
        ticker  = a.get("ticker","")
        company = a.get("company_name", ticker)
        sector  = a.get("sector","")
        price   = a.get("current_price", 0)
        mktcap  = a.get("market_cap", 0)
        signal  = a.get("buy_signal","HOLD")
        score   = a.get("composite_score", 50)
        pt      = a.get("price_target",{})
        target  = pt.get("price_target_12m")
        upside  = pt.get("upside_downside",0)

        signal_color = {
            "STRONG BUY": GREEN, "BUY": GREEN,
            "HOLD": YELLOW, "WEAK": YELLOW, "AVOID": RED,
        }.get(signal, YELLOW)

        story = [Spacer(1, 12)]

        # Company header table
        header_data = [[
            Paragraph(f"<b>{company}</b>", ParagraphStyle("ch",
                fontName="Helvetica-Bold", fontSize=20,
                textColor=DARK_BLUE, leading=24)),
            Paragraph(f"<b>{signal}</b>", ParagraphStyle("sig",
                fontName="Helvetica-Bold", fontSize=16,
                textColor=WHITE, alignment=TA_CENTER)),
        ]]
        header_table = Table(header_data, colWidths=[4.5*inch, 1.8*inch])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (1,0), (1,0), signal_color),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("ROWPADDING", (0,0), (-1,-1), 8),
            ("ROUNDEDCORNERS", [6]),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 6))

        story.append(Paragraph(
            f"{ticker} | {sector} | ${price:,.2f} | "
            f"Mkt Cap: ${'%.1fB' % (mktcap/1e9) if mktcap >= 1e9 else '%.0fM' % (mktcap/1e6)} | "
            f"Score: {score:.1f}/100",
            s["cover_sub"]
        ))

        if target:
            story.append(Paragraph(
                f"12-Month Price Target: <b>${target:.2f}</b> "
                f"({'▲' if upside > 0 else '▼'}{abs(upside):.1f}% {'upside' if upside > 0 else 'downside'}) "
                f"| Conviction: {pt.get('conviction','')}",
                s["cover_sub"]
            ))

        story.append(Spacer(1, 8))
        return story

    # ── KEY STATS ────────────────────────────────────────────────

    def _key_stats_bar(self, a: dict, s: dict) -> list:
        fund = a.get("fundamental",{})
        risk = a.get("risk",{})
        tech = a.get("technical",{})
        info_h = fund.get("highlights",{})

        def kv(label, value):
            return [
                Paragraph(label, s["metric_label"]),
                Paragraph(str(value), s["metric_value"]),
            ]

        pe  = next((v for k,v in info_h.items() if "P/E" in k), "N/A")
        rg  = next((v for k,v in info_h.items() if "Revenue Growth" in k), "N/A")
        gm  = next((v for k,v in info_h.items() if "Gross Margin" in k), "N/A")

        data = [[
            kv("P/E Ratio",     pe),
            kv("Rev Growth",    rg),
            kv("Gross Margin",  gm),
            kv("Sharpe Ratio",  f"{risk.get('sharpe_ratio',0):.2f}"),
            kv("Max Drawdown",  f"{risk.get('max_drawdown',0)*100:.1f}%"),
            kv("Beta",          f"{risk.get('beta',0):.2f}"),
        ]]
        # Flatten for single row
        flat = [[item for pair in data[0] for item in pair]]
        t = Table(flat, colWidths=[1.05*inch]*6)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), LIGHT_GRAY),
            ("GRID",          (0,0), (-1,-1), 0.5, MID_GRAY),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("ROWPADDING",    (0,0), (-1,-1), 6),
        ]))
        return [t, Spacer(1, 8)]

    # ── THESIS ───────────────────────────────────────────────────

    def _thesis_section(self, a: dict, s: dict) -> list:
        pt     = a.get("price_target",{})
        thesis = pt.get("thesis","")
        if not thesis:
            return []
        story  = [Paragraph("  INVESTMENT THESIS", s["section_header"]),
                  Spacer(1, 6)]
        story.append(Paragraph(thesis, s["thesis"]))
        story.append(Spacer(1, 10))
        return story

    # ── PRICE TARGET ─────────────────────────────────────────────

    def _price_target_section(self, a: dict, s: dict) -> list:
        pt = a.get("price_target",{})
        if not pt.get("price_target_12m"):
            return []

        story = [Paragraph("  12-MONTH PRICE TARGET", s["section_header"]),
                 Spacer(1, 6)]

        models = pt.get("models",{})
        rows   = [["Model", "Target", "Upside", "Weight", "Confidence"]]
        for nm, m in models.items():
            if m.get("price"):
                rows.append([
                    m.get("source", nm),
                    f"${m['price']:.2f}",
                    f"{m.get('upside_pct',0):+.1f}%",
                    f"{m.get('weight',0)*100:.0f}%",
                    f"{m.get('confidence',0)}%",
                ])

        rows.append([
            "BLENDED TARGET",
            f"${pt['price_target_12m']:.2f}",
            f"{pt.get('upside_downside',0):+.1f}%",
            "100%",
            pt.get("conviction",""),
        ])

        t = Table(rows, colWidths=[2.2*inch, 1*inch, 1*inch, 0.8*inch, 2.1*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  MID_BLUE),
            ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("BACKGROUND",  (0,-1),(-1,-1), DARK_BLUE),
            ("TEXTCOLOR",   (0,-1),(-1,-1), WHITE),
            ("FONTNAME",    (0,-1),(-1,-1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1),(-1,-2), [WHITE, LIGHT_GRAY]),
            ("GRID",        (0,0), (-1,-1), 0.5, MID_GRAY),
            ("ALIGN",       (1,0), (-1,-1), "CENTER"),
            ("FONTSIZE",    (0,0), (-1,-1), 8.5),
            ("ROWPADDING",  (0,0), (-1,-1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
        return story

    # ── SCORE DASHBOARD ──────────────────────────────────────────

    def _score_dashboard(self, a: dict, s: dict) -> list:
        story = [Paragraph("  COMPOSITE SCORE BREAKDOWN", s["section_header"]),
                 Spacer(1, 6)]

        scores = [
            ("Fundamental Analysis",    a.get("fundamental",{}).get("score",50),  "28%"),
            ("Competitive / Moat",      a.get("competitive",{}).get("score",50),  "22%"),
            ("Technical Analysis",      a.get("technical",{}).get("score",50),    "17%"),
            ("News Sentiment",          a.get("sentiment",{}).get("score",50),    "10%"),
            ("Risk-Adjusted Metrics",   a.get("risk",{}).get("score",50),         "8%"),
            ("Insider / Institutional", a.get("insider",{}).get("score",50),      "7%"),
            ("Options Flow",            a.get("options",{}).get("score",50),      "5%"),
            ("Short Interest",          a.get("short",{}).get("score",50),        "3%"),
        ]

        rows = [["Module", "Weight", "Score", "Bar"]]
        for name, score, weight in scores:
            bar_len = int(score / 5)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            rows.append([name, weight, f"{score:.0f}/100", bar])

        rows.append(["COMPOSITE SCORE", "100%",
                     f"{a.get('composite_score',50):.1f}/100", ""])

        t = Table(rows, colWidths=[2.4*inch, 0.6*inch, 0.8*inch, 2.85*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  MID_BLUE),
            ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("BACKGROUND",  (0,-1),(-1,-1), DARK_BLUE),
            ("TEXTCOLOR",   (0,-1),(-1,-1), WHITE),
            ("FONTNAME",    (0,-1),(-1,-1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1),(-1,-2), [WHITE, LIGHT_GRAY]),
            ("GRID",        (0,0), (-1,-1), 0.5, MID_GRAY),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("FONTNAME",    (0,1), (0,-2),  "Helvetica"),
            ("ROWPADDING",  (0,0), (-1,-1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
        return story

    # ── SECTIONS ─────────────────────────────────────────────────

    def _fundamental_section(self, a: dict, s: dict) -> list:
        fund = a.get("fundamental",{})
        highlights = fund.get("highlights",{})
        q          = fund.get("quality_score",{})
        if not highlights:
            return []

        story = [Paragraph("  FUNDAMENTAL ANALYSIS", s["section_header"]),
                 Spacer(1, 6)]

        rows  = [["Metric", "Value"]]
        for k, v in highlights.items():
            rows.append([k, str(v)])
        if q:
            rows.append([f"Piotroski F-Score", f"{q.get('f_score','N/A')}/9 — {q.get('interpretation','')}"])

        t = Table(rows, colWidths=[3*inch, 4.15*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0), MID_BLUE),
            ("TEXTCOLOR",      (0,0), (-1,0), WHITE),
            ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
            ("GRID",           (0,0), (-1,-1), 0.5, MID_GRAY),
            ("FONTSIZE",       (0,0), (-1,-1), 8.5),
            ("ROWPADDING",     (0,0), (-1,-1), 4),
        ]))
        story += [t, Spacer(1, 10)]
        return story

    def _accounting_section(self, a: dict, s: dict) -> list:
        fraud = a.get("fraud_detection",{})
        if not fraud:
            return []

        story = [Paragraph("  ACCOUNTING QUALITY & FRAUD DETECTION", s["section_header"]),
                 Spacer(1, 6)]

        beneish = fraud.get("beneish",{})
        altman  = fraud.get("altman",{})
        eq      = fraud.get("earnings_quality",{})

        data = [
            ["Beneish M-Score", f"{beneish.get('m_score','N/A')}",
             beneish.get("verdict","")[:60]],
            ["Altman Z-Score",  f"{altman.get('z_score','N/A')}",
             altman.get("zone","")[:60]],
            ["Earnings Quality", f"{eq.get('score','N/A')}/100",
             eq.get("verdict","")[:60]],
        ]
        rows = [["Test", "Score", "Assessment"]] + data

        t = Table(rows, colWidths=[1.8*inch, 1*inch, 4.35*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0), MID_BLUE),
            ("TEXTCOLOR",      (0,0), (-1,0), WHITE),
            ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
            ("GRID",           (0,0), (-1,-1), 0.5, MID_GRAY),
            ("FONTSIZE",       (0,0), (-1,-1), 8.5),
            ("ROWPADDING",     (0,0), (-1,-1), 4),
        ]))

        story += [t, Spacer(1, 6)]

        flags = fraud.get("overall_risk",{}).get("all_flags",[])
        if flags:
            story.append(Paragraph("<b>Risk Flags:</b>", s["body_bold"]))
            for f in flags[:4]:
                story.append(Paragraph(f"  • {f}", s["small"]))

        story.append(Spacer(1, 10))
        return story

    def _competitive_section(self, a: dict, s: dict) -> list:
        comp = a.get("competitive",{})
        if not comp:
            return []
        story = [Paragraph("  COMPETITIVE ANALYSIS & ECONOMIC MOAT", s["section_header"]),
                 Spacer(1, 6)]
        moat  = comp.get("moat_score",{})
        story.append(Paragraph(
            f"<b>Moat:</b> {moat.get('width','N/A')} ({moat.get('score',0)}/100) | "
            f"<b>Position:</b> {comp.get('competitive_position','N/A')} | "
            f"<b>vs Peers:</b> {comp.get('relative_valuation',{}).get('overall','')}",
            s["body"]
        ))
        for adv in comp.get("key_advantages",[])[:3]:
            story.append(Paragraph(f"  ✅ {adv}", s["body"]))
        for risk in comp.get("key_risks",[])[:3]:
            story.append(Paragraph(f"  ⚠️  {risk}", s["body"]))
        story.append(Spacer(1, 10))
        return story

    def _technical_section(self, a: dict, s: dict) -> list:
        tech = a.get("technical",{})
        if not tech:
            return []
        story = [Paragraph("  TECHNICAL ANALYSIS", s["section_header"]),
                 Spacer(1, 6)]
        for k, v in tech.get("signals",{}).items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", s["body"]))
        sup = tech.get("support")
        res = tech.get("resistance")
        if sup and res:
            story.append(Paragraph(
                f"<b>Support:</b> ${sup:.2f} | <b>Resistance:</b> ${res:.2f}", s["body"]))
        story.append(Spacer(1, 10))
        return story

    def _insider_section(self, a: dict, s: dict) -> list:
        insider = a.get("insider",{})
        if not insider:
            return []
        story = [Paragraph("  INSIDER TRADING & INSTITUTIONAL OWNERSHIP", s["section_header"]),
                 Spacer(1, 6)]
        story.append(Paragraph(insider.get("signal",""), s["body"]))
        story.append(Paragraph(insider.get("summary",""), s["body"]))
        inst = insider.get("institutional",{})
        if inst:
            story.append(Paragraph(
                f"Institutional: {inst.get('institutional_ownership_pct',0):.1f}% | "
                f"Insider: {inst.get('insider_ownership_pct',0):.1f}%",
                s["body"]
            ))
        story.append(Spacer(1, 10))
        return story

    def _short_options_section(self, a: dict, s: dict) -> list:
        short   = a.get("short",{})
        options = a.get("options",{})
        story   = [Paragraph("  SHORT INTEREST & OPTIONS SIGNALS", s["section_header"]),
                   Spacer(1, 6)]
        if short:
            story.append(Paragraph(
                f"<b>Short Interest:</b> {short.get('signal','')} | "
                f"Float: {short.get('short_float_pct','N/A')}% | "
                f"Days to Cover: {short.get('days_to_cover','N/A')} | "
                f"Squeeze Score: {short.get('squeeze_score',0)}/100",
                s["body"]
            ))
        if options:
            pcr = options.get("put_call_ratio",{})
            ivr = options.get("iv_rank",{})
            if pcr.get("pcr_volume"):
                story.append(Paragraph(
                    f"<b>Options:</b> P/C Ratio {pcr['pcr_volume']:.2f} ({pcr.get('sentiment','')}) | "
                    f"IV Rank {ivr.get('iv_rank','N/A')}/100",
                    s["body"]
                ))
        story.append(Spacer(1, 10))
        return story

    def _simulation_section(self, a: dict, s: dict) -> list:
        sims = a.get("simulations",{})
        mc   = sims.get("monte_carlo",{})
        dcf  = sims.get("dcf",{})
        risk = a.get("risk",{})
        if not mc and not dcf:
            return []

        story = [Paragraph("  SIMULATIONS & RISK METRICS", s["section_header"]),
                 Spacer(1, 6)]

        if mc:
            story.append(Paragraph(
                f"<b>Monte Carlo (1-Year):</b> Median {mc.get('median_return_1y',0)*100:.1f}% | "
                f"Bull {mc.get('bull_case',0)*100:.1f}% | Bear {mc.get('bear_case',0)*100:.1f}% | "
                f"Prob Profit {mc.get('probability_profit',0)*100:.0f}%",
                s["body"]
            ))
        if dcf and dcf.get("intrinsic_value"):
            story.append(Paragraph(
                f"<b>DCF Valuation:</b> Intrinsic ${dcf['intrinsic_value']:.2f} — "
                f"{dcf.get('valuation_status','')} | "
                f"MOS 20%: ${dcf.get('margin_of_safety_20',0):.2f}",
                s["body"]
            ))
        if risk:
            story.append(Paragraph(
                f"<b>Risk:</b> Sharpe {risk.get('sharpe_ratio',0):.2f} | "
                f"Sortino {risk.get('sortino_ratio',0):.2f} | "
                f"Max DD {risk.get('max_drawdown',0)*100:.1f}% | "
                f"VaR(95%) {risk.get('var_95',0)*100:.2f}%",
                s["body"]
            ))
        story.append(Spacer(1, 10))
        return story

    def _macro_section(self, a: dict, s: dict) -> list:
        macro = a.get("macro")
        if not macro:
            return []
        story = [Paragraph("  MACRO REGIME", s["section_header"]),
                 Spacer(1, 6)]
        story.append(Paragraph(
            f"<b>Regime:</b> {macro.get('regime','')} ({macro.get('asset_bias','')}) | "
            f"{macro.get('description','')}",
            s["body"]
        ))
        fit = macro.get("sector_fit",{})
        story.append(Paragraph(f"<b>Sector Fit:</b> {fit.get('assessment','')}", s["body"]))
        story.append(Spacer(1, 10))
        return story

    def _catalyst_risk_section(self, a: dict, s: dict) -> list:
        pt   = a.get("price_target",{})
        cats = pt.get("catalysts",[])
        risks = pt.get("risks",[])
        if not cats and not risks:
            return []

        story = [Paragraph("  CATALYSTS & RISK FACTORS", s["section_header"]),
                 Spacer(1, 6)]

        data  = []
        left  = ["<b>Upside Catalysts</b>"] + [f"+ {c}" for c in cats[:5]]
        right = ["<b>Risk Factors</b>"]     + [f"- {r}" for r in risks[:5]]
        max_r = max(len(left), len(right))
        left  += [""] * (max_r - len(left))
        right += [""] * (max_r - len(right))

        rows  = [[Paragraph(l, s["body"]), Paragraph(r, s["body"])]
                 for l, r in zip(left, right)]
        t = Table(rows, colWidths=[3.5*inch, 3.65*inch])
        t.setStyle(TableStyle([
            ("VALIGN",   (0,0), (-1,-1), "TOP"),
            ("ROWPADDING", (0,0), (-1,-1), 3),
            ("LINEAFTER", (0,0), (0,-1), 0.5, MID_GRAY),
        ]))
        story += [t, Spacer(1, 10)]
        return story

    def _disclaimer(self, s: dict) -> list:
        return [
            HRFlowable(width="100%", thickness=1, color=MID_GRAY),
            Spacer(1, 6),
            Paragraph(
                "DISCLAIMER: This report is generated by InstitutionalEdge and is for educational "
                "and informational purposes only. It does not constitute investment advice, a "
                "recommendation to buy or sell any security, or an offer to provide investment "
                "advisory services. Past performance is not indicative of future results. All "
                "investments involve risk, including the possible loss of principal. The information "
                "contained herein has been obtained from sources believed to be reliable but is not "
                "guaranteed for accuracy or completeness. Always consult a licensed financial advisor "
                "before making investment decisions. InstitutionalEdge is not a registered investment "
                "advisor.",
                s["disclaimer"]
            )
        ]
