"""
streamlit_app.py  –  Retail Sales & Customer Intelligence Analytics
IBM SkillsBuild Academic Internship — Data Analytics with AI

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.feature_engineering import load_processed
from src.analytics import (
    sales_summary, sales_by_year, monthly_sales_trend, sales_by_category,
    sales_by_subcategory, sales_by_region, profitability_by_subcategory,
    discount_profit_analysis,
)
from src.customer_analysis import customer_revenue_table, compute_rfm
from src.ai_insights import build_insight_context, generate_insights

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design tokens ────────────────────────────────────────────────────────────
BG        = "#0B0F14"
SIDEBAR   = "#11161D"
CARD      = "#151B23"
CARD2     = "#1B222C"
BORDER    = "#2A323D"
TEXT      = "#F3F4F6"
MUTED     = "#9CA3AF"
ACCENT    = "#F97316"
GREEN     = "#22C55E"
RED       = "#EF4444"
BLUE      = "#3B82F6"
CHART_PAL = [ACCENT, BLUE, GREEN, "#A78BFA", "#38BDF8"]

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Reset & base ─────────────────────────────── */
html, body, [data-testid="stApp"] {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 14px;
}}

/* ── Content area padding ─────────────────────── */
.block-container {{
    padding: 1.25rem 1.5rem 2.5rem !important;
    max-width: 100% !important;
}}

/* ── Hide default Streamlit header/footer ─────── */
#MainMenu, footer, [data-testid="stDeployButton"] {{ display: none !important; }}
[data-testid="stSidebarNav"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}

/* ── Sidebar ──────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {{
    background: {SIDEBAR};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .block-container {{
    padding: 0 !important;
}}

/* ── KPI metric tiles ─────────────────────────── */
[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 12px 16px 10px !important;
    min-width: 0;
}}
[data-testid="stMetricLabel"] > div {{
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {MUTED} !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 22px !important;
    font-weight: 700 !important;
    color: {TEXT} !important;
    line-height: 1.2 !important;
    white-space: nowrap;
}}
[data-testid="stMetricDelta"] {{
    font-size: 11px !important;
}}
/* Remove the default bottom delta container */
[data-testid="stMetricDeltaIcon"] {{ display: none; }}

/* ── Sidebar radio ────────────────────────────── */
[data-testid="stSidebarContent"] [data-testid="stRadio"] > label {{
    display: none;
}}
[data-testid="stSidebarContent"] [data-testid="stRadio"] > div {{
    gap: 2px !important;
}}
[data-testid="stSidebarContent"] [data-testid="stRadio"] label {{
    font-size: 13px !important;
    color: {MUTED} !important;
    padding: 7px 12px !important;
    border-radius: 5px !important;
    cursor: pointer;
    transition: background 0.15s;
}}
[data-testid="stSidebarContent"] [data-testid="stRadio"] label:hover {{
    background: {CARD2} !important;
    color: {TEXT} !important;
}}
/* Selected nav item */
[data-testid="stSidebarContent"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebarContent"] [data-testid="stRadio"] input:checked ~ div label {{
    color: {TEXT} !important;
    background: {CARD2} !important;
    border-left: 3px solid {ACCENT} !important;
    padding-left: 9px !important;
}}

/* ── Multiselect inputs ───────────────────────── */
[data-baseweb="select"] > div {{
    background: {CARD2} !important;
    border-color: {BORDER} !important;
    border-radius: 5px !important;
    min-height: 34px !important;
    font-size: 12px !important;
}}
[data-baseweb="select"] * {{ color: {TEXT} !important; }}
[data-baseweb="select"] [data-baseweb="tag"] {{
    background: {BORDER} !important;
    border: none !important;
    border-radius: 3px !important;
    height: 20px !important;
    padding: 0 6px !important;
}}
[data-baseweb="select"] [data-baseweb="tag"] span {{
    font-size: 11px !important;
    color: {MUTED} !important;
}}
/* Multiselect label */
[data-testid="stSidebarContent"] .stMultiSelect label {{
    font-size: 11px !important;
    color: {MUTED} !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 2px;
}}

/* ── Buttons ──────────────────────────────────── */
button[kind="primary"], [data-testid="baseButton-primary"] {{
    background: {ACCENT} !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 5px !important;
    padding: 6px 16px !important;
}}
button[kind="secondary"], [data-testid="baseButton-secondary"] {{
    background: transparent !important;
    border: 1px solid {BORDER} !important;
    color: {MUTED} !important;
    font-size: 12px !important;
    border-radius: 5px !important;
    padding: 4px 12px !important;
}}
button[kind="secondary"]:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
}}

/* ── Dataframes ───────────────────────────────── */
[data-testid="stDataFrame"] iframe {{
    border-radius: 0 !important;
}}
.dvn-scroller {{ background: {CARD} !important; }}

/* ── Spinners ─────────────────────────────────── */
[data-testid="stSpinner"] > div > div {{ border-top-color: {ACCENT} !important; }}

/* ── Alert boxes ──────────────────────────────── */
[data-testid="stAlert"] {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 5px !important;
    font-size: 13px !important;
}}

/* ── Checkbox ─────────────────────────────────── */
[data-testid="stCheckbox"] label span {{
    font-size: 12px !important;
    color: {MUTED} !important;
}}

/* ── Column gap fix ───────────────────────────── */
[data-testid="column"] {{ min-width: 0; }}
</style>
""", unsafe_allow_html=True)

# ─── Chart card counter (unique CSS keys per card) ────────────────────────────
_card_counter = [0]

def _next_card_key() -> str:
    _card_counter[0] += 1
    return f"cc{_card_counter[0]}"


# ─── Reusable UI helpers ──────────────────────────────────────────────────────

def fmt_currency(val: float) -> str:
    v = float(val)
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


def fmt_pct(val: float, d: int = 1) -> str:
    return f"{float(val):.{d}f}%"


def page_header(title: str, subtitle: str = "") -> None:
    sub_html = (f'<div style="font-size:12px;color:{MUTED};margin-top:3px;'
                f'line-height:1.4">{subtitle}</div>') if subtitle else ""
    st.markdown(
        f'<div style="padding-bottom:12px;margin-bottom:16px;'
        f'border-bottom:1px solid {BORDER}">'
        f'<span style="font-size:22px;font-weight:700;color:{TEXT};'
        f'letter-spacing:-0.01em;line-height:1.2">{title}</span>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_label(title: str, margin_top: int = 20) -> None:
    st.markdown(
        f'<div style="margin:{margin_top}px 0 10px;font-size:11px;font-weight:700;'
        f'color:{MUTED};text-transform:uppercase;letter-spacing:0.09em;'
        f'border-left:2px solid {ACCENT};padding-left:8px">{title}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list) -> None:
    cols = st.columns(len(metrics), gap="small")
    for col, m in zip(cols, metrics):
        col.metric(m["label"], m["value"])


def chart_card(fig: go.Figure, title: str, height: int = 300) -> None:
    """
    Render a Plotly chart inside a styled card container.

    Because Streamlit renders each markdown and plotly call independently,
    we use an injected CSS rule that targets the NEXT sibling .stPlotlyChart
    using a unique wrapper class, giving us reliable card appearance.
    """
    key = _next_card_key()
    # Inject: a wrapper div with unique class + the title
    st.markdown(
        f'<div class="ra-card" id="card-{key}" style="'
        f'background:{CARD};border:1px solid {BORDER};border-radius:7px;'
        f'padding:14px 14px 2px;margin-bottom:0">'
        f'<div style="font-size:11px;font-weight:700;color:{MUTED};'
        f'text-transform:uppercase;letter-spacing:0.08em;'
        f'margin-bottom:6px">{title}</div>',
        unsafe_allow_html=True,
    )
    _style_chart(fig, height)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "responsive": True})
    st.markdown("</div>", unsafe_allow_html=True)
    # CSS: zero out the default Streamlit element gap inside cards
    st.markdown(
        f"<style>#card-{key} + div[data-testid='stVerticalBlock'] {{"
        f"margin-top:-8px}}</style>",
        unsafe_allow_html=True,
    )


def _style_chart(fig: go.Figure, height: int) -> None:
    """Apply the unified dark chart theme."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Segoe UI', system-ui, sans-serif",
                  size=11, color=MUTED),
        margin=dict(l=4, r=48, t=4, b=52),  # b=52: room for legend below x-axis
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10, color=MUTED),
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="left", x=0,
        ),
        xaxis=dict(
            showgrid=False,
            gridcolor=BORDER,
            linecolor=BORDER,
            tickfont=dict(size=10, color=MUTED),
            title_font=dict(size=10, color=MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=BORDER,
            gridwidth=1,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=10, color=MUTED),
            title_font=dict(size=10, color=MUTED),
            zeroline=False,
        ),
        colorway=CHART_PAL,
        hoverlabel=dict(
            bgcolor=CARD2,
            bordercolor=BORDER,
            font_size=12,
            font_color=TEXT,
        ),
    )


def render_table(df: pd.DataFrame, col_fmt: dict | None = None,
                 height: int = 380) -> None:
    reset = df.reset_index(drop=True)
    if col_fmt:
        styled = reset.style.format(col_fmt).hide(axis="index")
        st.dataframe(styled, width="stretch", height=height)
    else:
        st.dataframe(reset, width="stretch", height=height, hide_index=True)


def spacer(px: int = 8) -> None:
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return load_processed()


@st.cache_data(show_spinner="Running forecast models…")
def get_forecast(_df: pd.DataFrame):
    from src.forecasting import run_forecasting
    return run_forecasting(_df)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

NAV = [
    ("Overview",              "overview"),
    ("Customer Analytics",    "customers"),
    ("Product & Profitability","products"),
    ("Sales Prediction",      "forecast"),
    ("AI Business Insights",  "ai"),
]


def render_sidebar(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    with st.sidebar:
        # Brand header
        st.markdown(
            f'<div style="padding:18px 16px 14px;'
            f'border-bottom:1px solid {BORDER}">'
            f'<div style="font-size:13px;font-weight:700;color:{TEXT};'
            f'letter-spacing:0.04em;text-transform:uppercase">Retail Analytics</div>'
            f'<div style="font-size:10px;color:{MUTED};margin-top:2px">'
            f'IBM SkillsBuild Capstone</div></div>',
            unsafe_allow_html=True,
        )

        # Navigation section label
        st.markdown(
            f'<div style="padding:14px 16px 4px;font-size:10px;font-weight:700;'
            f'color:{MUTED};text-transform:uppercase;letter-spacing:0.08em">'
            f'Navigation</div>',
            unsafe_allow_html=True,
        )
        page_labels = [n[0] for n in NAV]
        page_ids    = [n[1] for n in NAV]
        sel_label = st.radio("__nav__", page_labels, label_visibility="collapsed")
        sel_id    = page_ids[page_labels.index(sel_label)]

        # Filters section label
        st.markdown(
            f'<div style="padding:14px 16px 6px;font-size:10px;font-weight:700;'
            f'color:{MUTED};text-transform:uppercase;letter-spacing:0.08em;'
            f'border-top:1px solid {BORDER};margin-top:6px">Filters</div>',
            unsafe_allow_html=True,
        )

        years_all   = sorted(df["Year"].unique().tolist())
        regions_all = sorted(df["Region"].unique().tolist())
        cats_all    = sorted(df["Category"].unique().tolist())
        segs_all    = sorted(df["Segment"].unique().tolist())

        # Default = None (no pills shown; fallback = all)
        with st.container():
            years_sel   = st.multiselect("Year",     years_all,   default=None,
                                          placeholder="All years",   key="f_year")
            regions_sel = st.multiselect("Region",   regions_all, default=None,
                                          placeholder="All regions",  key="f_region")
            cats_sel    = st.multiselect("Category", cats_all,    default=None,
                                          placeholder="All categories", key="f_cat")
            segs_sel    = st.multiselect("Segment",  segs_all,    default=None,
                                          placeholder="All segments", key="f_seg")

        spacer(4)
        if st.button("Reset Filters", use_container_width=True, type="secondary"):
            for k in ("f_year", "f_region", "f_cat", "f_seg"):
                st.session_state.pop(k, None)
            st.rerun()

        # Resolve: empty list → all
        ys = years_sel   or years_all
        rs = regions_sel or regions_all
        cs = cats_sel    or cats_all
        ss = segs_sel    or segs_all

        mask = (
            df["Year"].isin(ys)
            & df["Region"].isin(rs)
            & df["Category"].isin(cs)
            & df["Segment"].isin(ss)
        )
        filtered = df[mask]

        # Footer
        spacer(12)
        st.markdown(
            f'<div style="padding:10px 16px;border-top:1px solid {BORDER};'
            f'font-size:10px;color:{MUTED};line-height:1.6">'
            f'Tableau Sample Superstore<br>'
            f'<span style="color:{TEXT}">{len(filtered):,}</span> rows selected'
            f'</div>',
            unsafe_allow_html=True,
        )

    return sel_id, filtered


# ─── Page 1: Executive Overview ───────────────────────────────────────────────

def page_overview(df: pd.DataFrame) -> None:
    page_header(
        "Executive Overview",
        "Business performance across sales, profitability, and geography.",
    )

    kpis = sales_summary(df)
    kpi_row([
        {"label": "Total Sales",     "value": fmt_currency(kpis["total_sales"])},
        {"label": "Total Profit",    "value": fmt_currency(kpis["total_profit"])},
        {"label": "Orders",          "value": f"{kpis['total_orders']:,}"},
        {"label": "Customers",       "value": f"{kpis['unique_customers']:,}"},
        {"label": "Avg Order Value", "value": fmt_currency(kpis["avg_order_value"])},
        {"label": "Profit Margin",   "value": fmt_pct(kpis["overall_profit_margin_pct"])},
    ])

    spacer(10)

    # Row 1: monthly trend | category performance
    c1, c2 = st.columns(2, gap="small")

    with c1:
        monthly = monthly_sales_trend(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Year_Month"], y=monthly["Sales"],
            name="Sales", mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.06)",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Year_Month"], y=monthly["Profit"],
            name="Profit", mode="lines",
            line=dict(color=GREEN, width=1.6, dash="dot"),
        ))
        fig.update_xaxes(tickangle=-45, nticks=10, showgrid=False)
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        # Extra bottom margin: angled x-labels + legend need more space
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=82))
        chart_card(fig, "Monthly Sales & Profit", height=320)

    with c2:
        cat = sales_by_category(df)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Sales",  x=cat["Category"], y=cat["Sales"],
            marker_color=ACCENT, opacity=0.88,
            hovertemplate="%{x}<br>Sales: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Profit", x=cat["Category"], y=cat["Profit"],
            marker_color=GREEN, opacity=0.88,
            hovertemplate="%{x}<br>Profit: $%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        chart_card(fig, "Category Performance", height=290)

    spacer(6)

    # Row 2: region sales | region profitability
    c3, c4 = st.columns(2, gap="small")
    reg = sales_by_region(df)

    with c3:
        fig = go.Figure(go.Bar(
            x=reg["Region"], y=reg["Sales"],
            marker_color=CHART_PAL[:len(reg)],
            text=[fmt_currency(v) for v in reg["Sales"]],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig.update_yaxes(tickprefix="$", tickformat="~s", showgrid=True)
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
        chart_card(fig, "Sales by Region", height=250)

    with c4:
        reg["Margin_Pct"] = reg["Profit_Margin"] * 100
        bar_colors = [GREEN if v >= 10 else ACCENT if v >= 5 else RED
                      for v in reg["Margin_Pct"]]
        fig = go.Figure(go.Bar(
            x=reg["Region"], y=reg["Margin_Pct"],
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in reg["Margin_Pct"]],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        ))
        fig.update_yaxes(ticksuffix="%", showgrid=True)
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
        chart_card(fig, "Regional Profitability", height=250)

    spacer(6)

    # Row 3: year-over-year (full width)
    yoy = sales_by_year(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Sales",  x=yoy["Year"].astype(str), y=yoy["Sales"],
        marker_color=ACCENT, opacity=0.88,
        hovertemplate="%{x}<br>Sales: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Profit", x=yoy["Year"].astype(str), y=yoy["Profit"],
        marker_color=GREEN, opacity=0.88,
        hovertemplate="%{x}<br>Profit: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    chart_card(fig, "Year-over-Year Performance", height=240)


# ─── Page 2: Customer Analytics ───────────────────────────────────────────────

def page_customers(df: pd.DataFrame) -> None:
    page_header(
        "Customer Analytics",
        "Customer value distribution, purchasing behavior, and RFM segmentation.",
    )

    rfm  = compute_rfm(df)
    cust = customer_revenue_table(df)

    champs   = int(rfm["RFM_Segment"].value_counts().get("Champions", 0))
    at_risk  = int(rfm["RFM_Segment"].value_counts().get("At Risk", 0))
    top_rev  = float(cust["Total_Sales"].iloc[0]) if len(cust) else 0

    kpi_row([
        {"label": "Customers",           "value": f"{len(cust):,}"},
        {"label": "Avg Revenue / Cust",  "value": fmt_currency(cust["Total_Sales"].mean())},
        {"label": "Top Customer",        "value": fmt_currency(top_rev)},
        {"label": "Champions",           "value": str(champs)},
        {"label": "At-Risk",             "value": str(at_risk)},
        {"label": "Avg Orders / Cust",   "value": f"{cust['Total_Orders'].mean():.1f}"},
    ])

    spacer(10)

    # Row 1: donut + segment revenue bar (equal height)
    seg_palette = {
        "Champions":           GREEN,
        "Loyal Customers":     BLUE,
        "Potential Loyalists": ACCENT,
        "At Risk":             "#F59E0B",
        "Lost / Inactive":     RED,
    }

    c1, c2 = st.columns(2, gap="small")

    with c1:
        sc = rfm["RFM_Segment"].value_counts().reset_index()
        sc.columns = ["Segment", "Count"]
        colors = [seg_palette.get(s, ACCENT) for s in sc["Segment"]]
        fig = go.Figure(go.Pie(
            labels=sc["Segment"], values=sc["Count"],
            marker_colors=colors,
            hole=0.48,
            textinfo="percent",
            textfont=dict(size=10, color=TEXT),
            insidetextorientation="radial",
            hovertemplate="%{label}<br>%{value} customers (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            margin=dict(l=4, r=4, t=4, b=4),
            legend=dict(
                orientation="v", x=1.0, y=0.5,
                xanchor="left", yanchor="middle",
                font=dict(size=10, color=MUTED),
            ),
        )
        chart_card(fig, "RFM Segment Distribution", height=300)

    with c2:
        sr = rfm.groupby("RFM_Segment")["Monetary"].sum().reset_index()
        sr = sr.sort_values("Monetary", ascending=True)
        bar_c = [seg_palette.get(s, ACCENT) for s in sr["RFM_Segment"]]
        fig = go.Figure(go.Bar(
            x=sr["Monetary"], y=sr["RFM_Segment"],
            orientation="h",
            marker_color=bar_c,
            text=[fmt_currency(v) for v in sr["Monetary"]],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
        ))
        fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True)
        fig.update_layout(margin=dict(l=4, r=60, t=4, b=4))
        chart_card(fig, "Revenue by RFM Segment", height=300)

    spacer(6)

    # Row 2: top customers bar (full width)
    top15 = cust.head(15)
    seg_c = {"Consumer": ACCENT, "Corporate": BLUE, "Home Office": GREEN}
    bc = [seg_c.get(s, ACCENT) for s in top15["Segment"]]
    fig = go.Figure(go.Bar(
        x=top15["Total_Sales"],
        y=top15["Customer_Name"],
        orientation="h",
        marker_color=bc,
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
    chart_card(fig, "Top 15 Customers by Revenue", height=370)

    spacer(6)

    # Row 3: revenue distribution | RFM score histogram (equal columns)
    c3, c4 = st.columns(2, gap="small")

    with c3:
        fig = go.Figure(go.Histogram(
            x=cust["Total_Sales"], nbinsx=32,
            marker_color=ACCENT, opacity=0.82,
            hovertemplate="$%{x:,.0f}<br>%{y} customers<extra></extra>",
        ))
        fig.update_xaxes(tickprefix="$", tickformat="~s",
                         title_text="Revenue per Customer")
        fig.update_yaxes(title_text="Customers")
        chart_card(fig, "Customer Revenue Distribution", height=230)

    with c4:
        fig = go.Figure(go.Histogram(
            x=rfm["RFM_Score"], nbinsx=10,
            marker_color=BLUE, opacity=0.82,
            hovertemplate="Score %{x}<br>%{y} customers<extra></extra>",
        ))
        fig.update_xaxes(title_text="RFM Score (3–12)", dtick=1)
        fig.update_yaxes(title_text="Customers")
        chart_card(fig, "RFM Score Distribution", height=230)

    spacer(6)

    # RFM detail table
    section_label("Customer Detail — Top 25 by Revenue")
    tbl = rfm[["Customer_Name","Segment","Recency","Frequency",
               "Monetary","R_Score","F_Score","M_Score",
               "RFM_Score","RFM_Segment"]].head(25)
    render_table(tbl,
                 {"Monetary": "${:,.0f}", "Recency": "{:.0f}"},
                 height=400)


# ─── Page 3: Product & Profitability ──────────────────────────────────────────

def page_products(df: pd.DataFrame) -> None:
    page_header(
        "Product & Profitability",
        "Sub-category profit drivers, discount impact, and product-level performance.",
    )

    cat  = sales_by_category(df)
    sub  = profitability_by_subcategory(df)
    disc = discount_profit_analysis(df)

    # KPI row: 3 category margins + 2 counts = 5 tiles
    loss_n   = int((sub["Profit"] < 0).sum())
    profit_n = int((sub["Profit"] >= 0).sum())
    cat_kpis = [
        {"label": f"{row['Category']} Margin",
         "value": fmt_pct(row["Profit_Margin"] * 100)}
        for _, row in cat.iterrows()
    ] + [
        {"label": "Loss-Making",  "value": str(loss_n)},
        {"label": "Profitable",   "value": str(profit_n)},
    ]
    kpi_row(cat_kpis)

    spacer(10)

    # Row 1: sub-category profit | scatter
    c1, c2 = st.columns(2, gap="small")

    with c1:
        ss = sub.sort_values("Profit")
        cs = [RED if v < 0 else GREEN for v in ss["Profit"]]
        fig = go.Figure(go.Bar(
            x=ss["Profit"], y=ss["Sub_Category"],
            orientation="h",
            marker_color=cs,
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=BORDER, line_width=1.2)
        fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True, zeroline=False)
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
        chart_card(fig, "Profit by Sub-Category", height=460)

    with c2:
        sub2 = sales_by_subcategory(df)
        cmap = {"Furniture": ACCENT, "Office Supplies": BLUE, "Technology": GREEN}
        pt_c = [cmap.get(c, ACCENT) for c in sub2["Category"]]
        fig = go.Figure()
        # Invisible legend proxies
        for nm, col in cmap.items():
            if nm in sub2["Category"].values:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=8, color=col), name=nm,
                ))
        fig.add_trace(go.Scatter(
            x=sub2["Sales"], y=sub2["Profit"],
            mode="markers",
            marker=dict(color=pt_c, size=9, opacity=0.80,
                        line=dict(width=0.5, color=BORDER)),
            text=sub2["Sub_Category"],
            hovertemplate="<b>%{text}</b><br>Sales: $%{x:,.0f}<br>"
                          "Profit: $%{y:,.0f}<extra></extra>",
            showlegend=False,
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
        fig.add_vline(x=float(sub2["Sales"].median()),
                      line_dash="dot", line_color=BORDER, line_width=1)
        fig.update_xaxes(tickprefix="$", tickformat="~s")
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        chart_card(fig, "Sales vs Profit by Sub-Category", height=460)

    spacer(6)

    # Discount impact chart (full width)
    disc["Avg_Margin_Pct"] = disc["Avg_Profit_Margin"] * 100
    dc = [RED if v < 0 else GREEN for v in disc["Avg_Margin_Pct"]]
    fig = go.Figure(go.Bar(
        x=disc["Discount_Bin"].astype(str),
        y=disc["Avg_Margin_Pct"],
        marker_color=dc,
        text=[f"{v:.1f}%" for v in disc["Avg_Margin_Pct"]],
        textposition="outside",
        textfont=dict(size=10, color=MUTED),
        hovertemplate="Discount %{x}<br>Avg Margin: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=BORDER, line_width=1.2)
    fig.update_xaxes(title_text="Discount Range")
    fig.update_yaxes(ticksuffix="%", showgrid=True)
    fig.update_layout(margin=dict(l=4, r=48, t=4, b=4))
    chart_card(fig, "Discount Level vs Average Profit Margin", height=250)

    spacer(6)

    section_label("Sub-Category Performance Table")
    sub_tbl = sales_by_subcategory(df).copy()
    sub_tbl["Margin_%"] = (sub_tbl["Profit_Margin"] * 100).round(1)
    sub_tbl = sub_tbl.drop(columns=["Profit_Margin"]).sort_values("Profit")

    def _cp(v):
        return f"color: {RED}" if isinstance(v, (int, float)) and v < 0 else ""

    styled = (
        sub_tbl[["Category","Sub_Category","Sales","Profit","Margin_%","Quantity"]]
        .style
        .format({"Sales":"${:,.0f}", "Profit":"${:,.0f}",
                 "Margin_%":"{:.1f}%", "Quantity":"{:,}"})
        .map(_cp, subset=["Profit","Margin_%"])
        .hide(axis="index")
    )
    st.dataframe(styled, width="stretch", height=360)


# ─── Page 4: Sales Prediction ─────────────────────────────────────────────────

def page_forecast(df: pd.DataFrame) -> None:
    page_header(
        "Sales Prediction",
        "Monthly sales modelled with time-index regression. Last 12 months = test set.",
    )

    # Always forecast on the full dataset — filtering would remove training data
    # and produce inconsistent metrics. The full dataset is stored in session state.
    df_for_forecast = st.session_state.get("_df_full", df)
    forecast = get_forecast(df_for_forecast)
    train    = forecast["train"]
    test     = forecast["test"]
    best     = forecast["best_model"]

    m_df     = pd.DataFrame(forecast["metrics"]).sort_values("RMSE")
    best_row = m_df.iloc[0]

    kpi_row([
        {"label": "Best Model",      "value": best},
        {"label": "MAE",             "value": fmt_currency(best_row["MAE"])},
        {"label": "RMSE",            "value": fmt_currency(best_row["RMSE"])},
        {"label": "R²",              "value": f"{best_row['R2']:.3f}"},
        {"label": "Train Months",    "value": str(len(train))},
        {"label": "Test Months",     "value": str(len(test))},
    ])

    spacer(10)

    # Forecast chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train["Year_Month"], y=train["Sales"],
        name="Historical", mode="lines",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["Year_Month"], y=test["Sales"],
        name="Actual (test)", mode="lines",
        line=dict(color=BLUE, width=2, dash="dot"),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["Year_Month"], y=test["Pred_LR"],
        name="Linear Regression", mode="lines",
        line=dict(color=ACCENT, width=2),
        hovertemplate="%{x}<br>LR: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["Year_Month"], y=test["Pred_RF"],
        name="Random Forest", mode="lines",
        line=dict(color=GREEN, width=1.6),
        hovertemplate="%{x}<br>RF: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["Year_Month"], y=test["Pred_Naive"],
        name="Naive Baseline", mode="lines",
        line=dict(color=MUTED, width=1.2, dash="dot"),
        hovertemplate="%{x}<br>Naive: $%{y:,.0f}<extra></extra>",
    ))
    if len(train):
        # add_vline doesn't work on string/categorical axes — use a shape instead.
        # x_ref="x" with a string category value requires the exact label.
        split_label = train["Year_Month"].iloc[-1]
        all_months = list(train["Year_Month"]) + list(test["Year_Month"])
        split_idx = all_months.index(split_label)
        # Normalise to [0,1] paper coords
        split_x = split_idx / (len(all_months) - 1)
        fig.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=split_x, x1=split_x,
            y0=0, y1=1,
            line=dict(color=BORDER, width=1, dash="dot"),
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=split_x + 0.01, y=0.98,
            text="train / test",
            showarrow=False,
            font=dict(size=9, color=MUTED),
            xanchor="left",
        )
    fig.update_xaxes(tickangle=-45, nticks=14)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    fig.update_layout(margin=dict(l=4, r=8, t=4, b=72))
    chart_card(fig, "Monthly Sales — Historical vs Predicted", height=400)

    spacer(8)

    # Compact model comparison table
    section_label("Model Evaluation — Test Set")
    render_table(
        m_df[["model","MAE","RMSE","R2"]].rename(
            columns={"model":"Model","R2":"R²"}),
        {"MAE":"${:,.0f}", "RMSE":"${:,.0f}", "R²":"{:.4f}"},
        height=148,
    )

    spacer(8)

    # Limitations card
    section_label("Model Limitations")
    st.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-radius:6px;padding:14px 18px;font-size:12px;'
        f'line-height:1.75;color:{MUTED}">'
        f'<ul style="margin:0;padding-left:16px">'
        f'<li>~36 training months — insufficient for robust seasonal decomposition.</li>'
        f'<li>Single time-index feature; no external regressors or holiday effects.</li>'
        f'<li>Random Forest cannot extrapolate beyond training range (negative R² expected on trend data).</li>'
        f'<li>R²&nbsp;=&nbsp;{best_row["R2"]:.3f} — model captures trend direction, not month-level variance.</li>'
        f'<li>Treat forecasts as directional indicators only, not precise projections.</li>'
        f'</ul></div>',
        unsafe_allow_html=True,
    )


# ─── Page 5: AI Business Insights ─────────────────────────────────────────────

def page_ai_insights(df: pd.DataFrame) -> None:
    page_header(
        "AI Business Insights",
        "AI-assisted interpretation of verified analytical results. "
        "Numbers are computed deterministically; AI provides language interpretation only.",
    )

    kpis      = sales_summary(df)
    cat_df    = sales_by_category(df)
    region_df = sales_by_region(df)
    rfm_df    = compute_rfm(df)
    disc_df   = discount_profit_analysis(df)
    # Forecast on full dataset for consistent metrics regardless of sidebar filters
    df_for_forecast = st.session_state.get("_df_full", df)
    forecast  = get_forecast(df_for_forecast)

    # ── Verified metrics ──────────────────────────────────────────────────────
    section_label("Verified Analytical Metrics", margin_top=0)
    st.markdown(
        f'<div style="font-size:11px;color:{MUTED};margin-bottom:8px">'
        f'Computed by deterministic analytics code — not generated by AI.</div>',
        unsafe_allow_html=True,
    )

    top_cat   = cat_df.sort_values("Sales", ascending=False).iloc[0]
    worst_cat = cat_df.sort_values("Profit_Margin").iloc[0]
    top_reg   = region_df.sort_values("Sales", ascending=False).iloc[0]
    worst_reg = region_df.sort_values("Profit_Margin").iloc[0]
    champs    = int(rfm_df["RFM_Segment"].value_counts().get("Champions", 0))
    at_risk   = int(rfm_df["RFM_Segment"].value_counts().get("At Risk", 0))

    kpi_row([
        {"label": "Total Sales",     "value": fmt_currency(kpis["total_sales"])},
        {"label": "Total Profit",    "value": fmt_currency(kpis["total_profit"])},
        {"label": "Profit Margin",   "value": fmt_pct(kpis["overall_profit_margin_pct"])},
        {"label": "Orders",          "value": f"{kpis['total_orders']:,}"},
        {"label": "Customers",       "value": f"{kpis['unique_customers']:,}"},
        {"label": "Avg Order Value", "value": fmt_currency(kpis["avg_order_value"])},
    ])
    spacer(6)
    kpi_row([
        {"label": "Top Category",
         "value": top_cat["Category"]},
        {"label": "Lowest Margin Category",
         "value": f"{worst_cat['Category']} ({fmt_pct(worst_cat['Profit_Margin']*100)})"},
        {"label": "Top Region",
         "value": top_reg["Region"]},
        {"label": "Weakest Margin Region",
         "value": f"{worst_reg['Region']} ({fmt_pct(worst_reg['Profit_Margin']*100)})"},
        {"label": "Champions",
         "value": str(champs)},
        {"label": "At-Risk Customers",
         "value": str(at_risk)},
    ])

    spacer(16)
    st.markdown(f"<div style='border-top:1px solid {BORDER}'></div>",
                unsafe_allow_html=True)
    spacer(12)

    # ── AI generation ─────────────────────────────────────────────────────────
    section_label("AI-Generated Interpretation")

    prefer_llm = st.checkbox(
        "Use IBM Granite (requires `transformers` + local model download)",
        value=False,
    )
    spacer(4)

    col_btn, col_note = st.columns([1, 4], gap="small")
    with col_btn:
        run_ai = st.button("Generate Insights", type="primary",
                           use_container_width=True)
    with col_note:
        st.markdown(
            f'<div style="font-size:11px;color:{MUTED};padding-top:10px">'
            f'Click to generate AI-assisted interpretation from the metrics above.</div>',
            unsafe_allow_html=True,
        )

    if run_ai:
        with st.spinner("Generating…"):
            ctx = build_insight_context(
                kpis=kpis, cat_df=cat_df, region_df=region_df,
                rfm_df=rfm_df, forecast_metrics=forecast["metrics"],
                disc_df=disc_df,
            )
            text, mode = generate_insights(ctx, prefer_llm=prefer_llm)

        mode_label = ("IBM Granite" if mode == "granite" else "Template / Demo Mode")
        st.markdown(
            f'<span style="display:inline-block;background:{CARD2};'
            f'border:1px solid {BORDER};border-radius:3px;'
            f'padding:2px 8px;font-size:10px;color:{MUTED};'
            f'text-transform:uppercase;letter-spacing:0.05em;'
            f'margin-bottom:10px">Mode: {mode_label}</span>',
            unsafe_allow_html=True,
        )
        _render_ai_sections(text)

        rp = Path("outputs/reports/ai_insights.md")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(text, encoding="utf-8")
        st.markdown(
            f'<div style="font-size:10px;color:{MUTED};margin-top:8px">'
            f'Saved to {rp}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};'
            f'border-radius:6px;padding:20px 24px;font-size:13px;'
            f'color:{MUTED}">Press <strong style="color:{TEXT}">Generate Insights</strong> '
            f'to produce AI-assisted analysis of the verified metrics above.</div>',
            unsafe_allow_html=True,
        )

    # Responsible AI notice
    spacer(14)
    st.markdown(
        f'<div style="background:{CARD2};border:1px solid {BORDER};'
        f'border-left:2px solid {ACCENT};border-radius:0 5px 5px 0;'
        f'padding:10px 14px;font-size:11px;color:{MUTED}">'
        f'<strong style="color:{TEXT}">Responsible AI</strong> — '
        f'AI output is interpretation, not audit-grade evidence. '
        f'All numbers originate from deterministic code. '
        f'Template mode is rule-based, not a trained LLM.'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_ai_sections(text: str) -> None:
    """Parse AI markdown and render each section as a styled card."""
    SECTION_MAP = {
        "### Key Findings":          "Key Findings",
        "### Business Implications": "Business Implications",
        "### Areas Requiring Further Investigation": "Areas for Investigation",
    }
    cur_title = None
    cur_block: list[str] = []

    def _flush(title: str | None, block: list[str]) -> None:
        if title is None:
            return
        content = "\n".join(
            l for l in block
            if not l.startswith("#") and "---" not in l
            and "Demo/Template Mode" not in l
            and "AI-Generated Business Insights" not in l
        ).strip()
        if not content:
            return
        section_label(title)
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};'
            f'border-radius:6px;padding:14px 18px;font-size:13px;'
            f'line-height:1.75;color:{TEXT}">{_md_lite(content)}</div>',
            unsafe_allow_html=True,
        )
        spacer(6)

    for line in text.splitlines():
        stripped = line.strip()
        matched  = False
        for md_h, disp in SECTION_MAP.items():
            if stripped == md_h:
                _flush(cur_title, cur_block)
                cur_title, cur_block = disp, []
                matched = True
                break
        if not matched:
            cur_block.append(line)

    _flush(cur_title, cur_block)


def _md_lite(text: str) -> str:
    """Minimal markdown → HTML: bold, bullets, numbered lists, blockquotes."""
    import re
    lines  = text.splitlines()
    out    = []
    in_ul  = False

    for line in lines:
        s = line.strip()
        if s.startswith(("- ", "* ")):
            if not in_ul:
                out.append("<ul style='margin:6px 0;padding-left:16px'>")
                in_ul = True
            c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s[2:])
            out.append(f"<li style='margin-bottom:3px'>{c}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if s.startswith("> "):
                out.append(
                    f'<div style="font-size:10px;color:{MUTED};'
                    f'border-top:1px solid {BORDER};margin-top:10px;'
                    f'padding-top:8px">'
                    + re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s[2:])
                    + "</div>"
                )
            elif s:
                c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
                m = re.match(r"^(\d+)\.\s+(.*)", c)
                if m:
                    out.append(
                        f"<p style='margin:5px 0'>"
                        f"<strong>{m.group(1)}.</strong> {m.group(2)}</p>"
                    )
                else:
                    out.append(f"<p style='margin:5px 0'>{c}</p>")

    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Reset chart counter each run
    _card_counter[0] = 0

    df_full = load_data()
    sel_id, df_filtered = render_sidebar(df_full)

    # Always run forecast on full dataset (historical model, not filtered)
    # Expose df_full via session state for the forecast page
    st.session_state["_df_full"] = df_full

    if len(df_filtered) == 0:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {RED};'
            f'border-radius:6px;padding:18px 20px;color:{RED};font-size:13px">'
            f'No data matches the current filters. '
            f'Adjust or reset filters in the sidebar.</div>',
            unsafe_allow_html=True,
        )
        return

    {
        "overview":  page_overview,
        "customers": page_customers,
        "products":  page_products,
        "forecast":  page_forecast,
        "ai":        page_ai_insights,
    }[sel_id](df_filtered)


if __name__ == "__main__":
    main()
