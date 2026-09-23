"""
Retail_Sales_Analytics.py
-------------------------
Single-file consolidated execution for Retail Sales AI Analytics.
"""

import sys
import os
import textwrap
from pathlib import Path
from datetime import timedelta
import sqlite3
import re
from typing import Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
st.set_page_config(
    page_title="Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
[data-testid="stSidebar"] .block-container {{ padding: 0 !important; }}
/* ── KPI metric tiles ─────────────────────────── */
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 12px 16px 10px !important; min-width: 0;
}}
[data-testid="stMetricLabel"] > div {{
    font-size: 10px !important; font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: 0.07em; color: {MUTED} !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 22px !important; font-weight: 700 !important; color: {TEXT} !important; line-height: 1.2 !important; white-space: nowrap;
}}
[data-testid="stMetricDelta"] {{ font-size: 11px !important; }}
[data-testid="stMetricDeltaIcon"] {{ display: none; }}
/* ── Sidebar radio ────────────────────────────── */
[data-testid="stSidebarContent"] [data-testid="stRadio"] > label {{ display: none; }}
[data-testid="stSidebarContent"] [data-testid="stRadio"] > div {{ gap: 2px !important; }}
[data-testid="stSidebarContent"] [data-testid="stRadio"] label {{
    font-size: 13px !important; color: {MUTED} !important; padding: 7px 12px !important;
    border-radius: 5px !important; cursor: pointer; transition: background 0.15s;
}}
[data-testid="stSidebarContent"] [data-testid="stRadio"] label:hover {{
    background: {CARD2} !important; color: {TEXT} !important;
}}
[data-testid="stSidebarContent"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebarContent"] [data-testid="stRadio"] input:checked ~ div label {{
    color: {TEXT} !important; background: {CARD2} !important;
    border-left: 3px solid {ACCENT} !important; padding-left: 9px !important;
}}
/* ── Multiselect inputs ───────────────────────── */
[data-baseweb="select"] > div {{
    background: {CARD2} !important; border-color: {BORDER} !important;
    border-radius: 5px !important; min-height: 34px !important; font-size: 12px !important;
}}
[data-baseweb="select"] * {{ color: {TEXT} !important; }}
[data-baseweb="select"] [data-baseweb="tag"] {{
    background: {BORDER} !important; border: none !important; border-radius: 3px !important;
    height: 20px !important; padding: 0 6px !important;
}}
[data-baseweb="select"] [data-baseweb="tag"] span {{ font-size: 11px !important; color: {MUTED} !important; }}
[data-testid="stSidebarContent"] .stMultiSelect label {{
    font-size: 11px !important; color: {MUTED} !important; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 2px;
}}
/* ── Buttons ──────────────────────────────────── */
button[kind="primary"], [data-testid="baseButton-primary"] {{
    background: {ACCENT} !important; border: none !important; color: #fff !important;
    font-weight: 600 !important; font-size: 13px !important; border-radius: 5px !important; padding: 6px 16px !important;
}}
button[kind="secondary"], [data-testid="baseButton-secondary"] {{
    background: transparent !important; border: 1px solid {BORDER} !important; color: {MUTED} !important;
    font-size: 12px !important; border-radius: 5px !important; padding: 4px 12px !important;
}}
button[kind="secondary"]:hover {{ border-color: {ACCENT} !important; color: {ACCENT} !important; }}
/* ── Dataframes ───────────────────────────────── */
[data-testid="stDataFrame"] iframe {{ border-radius: 0 !important; }}
.dvn-scroller {{ background: {CARD} !important; }}
/* ── Spinners ─────────────────────────────────── */
[data-testid="stSpinner"] > div > div {{ border-top-color: {ACCENT} !important; }}
/* ── Alert boxes ──────────────────────────────── */
[data-testid="stAlert"] {{ background: {CARD2} !important; border: 1px solid {BORDER} !important; border-radius: 5px !important; font-size: 13px !important; }}
/* ── Checkbox ─────────────────────────────────── */
[data-testid="stCheckbox"] label span {{ font-size: 12px !important; color: {MUTED} !important; }}
/* ── Column gap fix ───────────────────────────── */
[data-testid="column"] {{ min-width: 0; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# UI UTILITIES
# ============================================================
_card_counter = [0]
def _next_card_key() -> str:
    _card_counter[0] += 1
    return f"cc{_card_counter[0]}"

def fmt_currency(val: float) -> str:
    v = float(val)
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000: return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def fmt_pct(val: float, d: int = 1) -> str:
    return f"{float(val):.{d}f}%"

def page_header(title: str, subtitle: str = "") -> None:
    sub_html = (f'<div style="font-size:12px;color:{MUTED};margin-top:3px;'
                f'line-height:1.4">{subtitle}</div>') if subtitle else ""
    st.markdown(
        f'<div style="padding-bottom:12px;margin-bottom:16px;border-bottom:1px solid {BORDER}">'
        f'<span style="font-size:22px;font-weight:700;color:{TEXT};letter-spacing:-0.01em;line-height:1.2">{title}</span>'
        f'{sub_html}</div>', unsafe_allow_html=True
    )

def section_label(title: str, margin_top: int = 20) -> None:
    st.markdown(
        f'<div style="margin:{margin_top}px 0 10px;font-size:11px;font-weight:700;'
        f'color:{MUTED};text-transform:uppercase;letter-spacing:0.09em;'
        f'border-left:2px solid {ACCENT};padding-left:8px">{title}</div>',
        unsafe_allow_html=True
    )

def kpi_row(metrics: list) -> None:
    cols = st.columns(len(metrics), gap="small")
    for col, m in zip(cols, metrics):
        col.metric(m["label"], m["value"])

def chart_card(fig: go.Figure, title: str, height: int = 300) -> None:
    key = _next_card_key()
    st.markdown(
        f'<div class="ra-card" id="card-{key}" style="'
        f'background:{CARD};border:1px solid {BORDER};border-radius:7px;'
        f'padding:14px 14px 2px;margin-bottom:0">'
        f'<div style="font-size:11px;font-weight:700;color:{MUTED};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">{title}</div>',
        unsafe_allow_html=True
    )
    _style_chart(fig, height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f"<style>#card-{key} + div[data-testid='stVerticalBlock'] {{margin-top:-8px}}</style>",
        unsafe_allow_html=True
    )

def _style_chart(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Segoe UI', system-ui, sans-serif", size=11, color=MUTED),
        margin=dict(l=4, r=48, t=4, b=52),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=10, color=MUTED), orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        xaxis=dict(showgrid=False, gridcolor=BORDER, linecolor=BORDER, tickfont=dict(size=10, color=MUTED), title_font=dict(size=10, color=MUTED), zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER, gridwidth=1, linecolor="rgba(0,0,0,0)", tickfont=dict(size=10, color=MUTED), title_font=dict(size=10, color=MUTED), zeroline=False),
        colorway=CHART_PAL,
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font_size=12, font_color=TEXT),
    )

def render_table(df: pd.DataFrame, col_fmt: dict | None = None, height: int = 380) -> None:
    reset = df.reset_index(drop=True).copy()
    if col_fmt:
        for col, fmt in col_fmt.items():
            if col in reset.columns:
                reset[col] = reset[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else x)
    th_style = f"padding:10px; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:0.05em; position:sticky; top:0; background:{CARD2}; border-bottom:1px solid {BORDER}; z-index:1;"
    html = f"""
    <div style="height:{height}px; overflow-y:auto; background:{CARD}; border:1px solid {BORDER}; border-radius:5px;">
        <table style="width:100%; border-collapse:collapse; font-size:12px; color:{TEXT}; text-align:left;">
            <thead><tr>{"".join(f'<th style="{th_style}">{c}</th>' for c in reset.columns)}</tr></thead>
            <tbody>{"".join(f'<tr style="border-bottom:1px solid {BORDER};">' + "".join(f'<td style="padding:10px;">{val}</td>' for val in row) + '</tr>' for row in reset.itertuples(index=False))}</tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def spacer(px: int = 8) -> None:
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

# ============================================================
# DATA PROCESSING
# ============================================================
_COLUMN_RENAMES = {
    "Country/Region": "Country", "State/Province": "State", "Sub-Category": "Sub_Category",
    "Order Date": "Order_Date", "Ship Date": "Ship_Date", "Ship Mode": "Ship_Mode",
    "Order ID": "Order_ID", "Customer ID": "Customer_ID", "Customer Name": "Customer_Name",
    "Product ID": "Product_ID", "Product Name": "Product_Name", "Row ID": "Row_ID",
    "Postal Code": "Postal_Code",
}

def load_and_process_data() -> pd.DataFrame:
    # 1. Resolve Path
    candidates = [
        Path("data/raw/sample_-_superstore.xls"), Path("data/raw/sample_-_superstore.xlsx"),
        Path("data/superstore.xls"), Path("data/superstore.xlsx"),
        Path("data/raw/superstore.xls"), Path("data/raw/superstore.xlsx"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        raise FileNotFoundError("Superstore dataset not found in expected 'data/' or 'data/raw/' paths.")
    
    # 2. Load
    df = pd.read_excel(path, sheet_name="Orders")
    
    # 3. Clean
    rename_map = {c: _COLUMN_RENAMES[c] for c in df.columns if c in _COLUMN_RENAMES}
    df.rename(columns=rename_map, inplace=True)
    for date_col in ("Order_Date", "Ship_Date"):
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            if col not in ("Order_Date", "Ship_Date"):
                df[col] = df[col].astype(str).str.strip()
    for num_col in ("Sales", "Quantity", "Discount", "Profit"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    core_cols = [c for c in ("Sales", "Quantity", "Profit", "Order_Date") if c in df.columns]
    df.dropna(subset=core_cols, inplace=True)
    df.drop_duplicates(inplace=True)
    if "Discount" in df.columns: df["Discount"] = df["Discount"].clip(0, 1)
    df.reset_index(drop=True, inplace=True)
    
    # 4. Feature Engineering
    df["Year"] = df["Order_Date"].dt.year.astype(int)
    df["Month"] = df["Order_Date"].dt.month.astype(int)
    df["Quarter"] = df["Order_Date"].dt.quarter.astype(int)
    df["Month_Name"] = df["Order_Date"].dt.strftime("%b")
    df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
    df["Year_Quarter"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)
    if "Ship_Date" in df.columns:
        df["Days_to_Ship"] = (df["Ship_Date"] - df["Order_Date"]).dt.days.astype("Int64")
    df["Profit_Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
    df["Sales_per_Unit"] = df["Sales"] / df["Quantity"]
    df["Profit_per_Unit"] = df["Profit"] / df["Quantity"]
    df["Is_Profitable"] = df["Profit"] > 0
    
    return df

@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return load_and_process_data()
# ============================================================
# DATABASE & SQL LOGIC
# ============================================================

def build_database_in_memory(df: pd.DataFrame) -> sqlite3.Connection:
    """Creates an in-memory SQLite database for the SQL queries."""
    export = df.copy()
    for col in export.select_dtypes(include=["datetime64"]).columns:
        export[col] = export[col].dt.strftime("%Y-%m-%d")
    for col in export.columns:
        if hasattr(export[col], "dtype") and str(export[col].dtype).startswith("Int"):
            export[col] = export[col].astype("float64")
    
    conn = sqlite3.connect(":memory:")
    export.to_sql("orders", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_order_id ON orders (Order_ID)")
    conn.execute("CREATE INDEX idx_customer_id ON orders (Customer_ID)")
    conn.execute("CREATE INDEX idx_order_date ON orders (Order_Date)")
    return conn

@st.cache_resource(show_spinner="Building SQL database…")
def get_sql_connection(_df: pd.DataFrame):
    return build_database_in_memory(_df)

BUSINESS_QUERIES = {
    "total_sales_profit_by_category": (
        "Sales & Profit by Category",
        """
        SELECT
            Category,
            ROUND(SUM(Sales), 2)         AS Total_Sales,
            ROUND(SUM(Profit), 2)        AS Total_Profit,
            COUNT(DISTINCT Order_ID)     AS Order_Count,
            ROUND(SUM(Profit)/SUM(Sales)*100, 2) AS Profit_Margin_Pct
        FROM orders
        GROUP BY Category
        ORDER BY Total_Sales DESC;
        """
    ),
    "regional_performance": (
        "Regional Performance",
        """
        SELECT
            Region,
            ROUND(SUM(Sales), 2)  AS Total_Sales,
            ROUND(SUM(Profit), 2) AS Total_Profit,
            COUNT(DISTINCT Order_ID) AS Orders,
            COUNT(DISTINCT Customer_ID) AS Customers,
            ROUND(SUM(Profit)/SUM(Sales)*100, 2) AS Profit_Margin_Pct
        FROM orders
        GROUP BY Region
        ORDER BY Total_Sales DESC;
        """
    ),
    "top_10_customers_by_revenue": (
        "Top 10 Customers by Revenue",
        """
        SELECT
            Customer_Name,
            Segment,
            ROUND(SUM(Sales), 2)  AS Total_Sales,
            ROUND(SUM(Profit), 2) AS Total_Profit,
            COUNT(DISTINCT Order_ID) AS Orders
        FROM orders
        GROUP BY Customer_ID, Customer_Name, Segment
        ORDER BY Total_Sales DESC
        LIMIT 10;
        """
    ),
    "loss_making_subcategories": (
        "Loss-Making Sub-Categories",
        """
        SELECT
            Category,
            Sub_Category,
            ROUND(SUM(Sales), 2)  AS Total_Sales,
            ROUND(SUM(Profit), 2) AS Total_Profit,
            ROUND(SUM(Profit)/SUM(Sales)*100, 2) AS Profit_Margin_Pct
        FROM orders
        GROUP BY Category, Sub_Category
        HAVING Total_Profit < 0
        ORDER BY Total_Profit ASC;
        """
    ),
    "avg_discount_by_category": (
        "Average Discount by Category",
        """
        SELECT
            Category,
            ROUND(AVG(Discount) * 100, 2)  AS Avg_Discount_Pct,
            ROUND(SUM(Sales), 2)           AS Total_Sales,
            ROUND(SUM(Profit), 2)          AS Total_Profit
        FROM orders
        GROUP BY Category
        ORDER BY Avg_Discount_Pct DESC;
        """
    )
}

def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, conn)


# ============================================================
# PANDAS ANALYTICS
# ============================================================

def sales_summary(df: pd.DataFrame) -> dict:
    total_orders = df["Order_ID"].nunique()
    unique_customers = df["Customer_ID"].nunique()
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    avg_order_val = df.groupby("Order_ID")["Sales"].sum().mean()
    overall_margin = total_profit / total_sales if total_sales else 0
    return {
        "total_sales": round(total_sales, 2),
        "total_profit": round(total_profit, 2),
        "total_orders": total_orders,
        "unique_customers": unique_customers,
        "avg_order_value": round(avg_order_val, 2),
        "overall_profit_margin_pct": round(overall_margin * 100, 2),
    }

def sales_by_year(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Year").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order_ID", "nunique")).reset_index()

def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby("Year_Month").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    monthly["Year_Month"] = pd.PeriodIndex(monthly["Year_Month"], freq="M")
    monthly = monthly.sort_values("Year_Month").reset_index(drop=True)
    monthly["Year_Month"] = monthly["Year_Month"].astype(str)
    return monthly

def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order_ID", "nunique")) \
             .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"]).reset_index().sort_values("Sales", ascending=False)

def sales_by_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Category", "Sub_Category"]).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum")) \
             .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"]).reset_index().sort_values("Sales", ascending=False)

def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Region").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order_ID", "nunique")) \
             .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"]).reset_index().sort_values("Sales", ascending=False)

def profitability_by_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.groupby(["Category", "Sub_Category"]).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum")) \
            .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"]).reset_index().sort_values("Profit", ascending=False)
    sub["Is_Loss"] = sub["Profit"] < 0
    return sub

def discount_profit_analysis(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
    labels = ["0%", "1-9%", "10-20%", "21-30%", "31-40%", "41-50%", "51-60%", "61-80%", ">80%"]
    d = df.copy()
    d["Discount_Bin"] = pd.cut(d["Discount"], bins=bins, labels=labels, right=False)
    return d.groupby("Discount_Bin", observed=True).agg(
        Avg_Profit_Margin=("Profit_Margin", "mean"), Avg_Profit=("Profit", "mean"), Avg_Sales=("Sales", "mean"), Count=("Sales", "count")
    ).reset_index()


# ============================================================
# CUSTOMER ANALYTICS
# ============================================================
def customer_revenue_table(df: pd.DataFrame) -> pd.DataFrame:
    cust = df.groupby(["Customer_ID", "Customer_Name", "Segment"]).agg(
        Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"),
        Total_Orders=("Order_ID", "nunique"), Total_Quantity=("Quantity", "sum")
    ).reset_index()
    cust["Profit_Margin"] = cust["Total_Profit"] / cust["Total_Sales"]
    cust["Avg_Order_Value"] = cust["Total_Sales"] / cust["Total_Orders"]
    return cust.sort_values("Total_Sales", ascending=False).reset_index(drop=True)

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["Order_Date"].max() + timedelta(days=1)
    rfm = df.groupby(["Customer_ID", "Customer_Name", "Segment"]).agg(
        Recency=("Order_Date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Order_ID", "nunique"), Monetary=("Sales", "sum")
    ).reset_index()

    def safe_qcut(series, labels):
        try: return pd.qcut(series, q=4, labels=labels, duplicates="drop")
        except ValueError: return pd.qcut(series.rank(method="first"), q=4, labels=labels, duplicates="drop")
    
    rfm["R_Score"] = safe_qcut(rfm["Recency"], labels=[4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = safe_qcut(rfm["Frequency"], labels=[1, 2, 3, 4]).astype(int)
    rfm["M_Score"] = safe_qcut(rfm["Monetary"], labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def assign_segment(score: int) -> str:
        if score >= 11: return "Champions"
        elif score >= 9: return "Loyal Customers"
        elif score >= 7: return "Potential Loyalists"
        elif score >= 5: return "At Risk"
        else: return "Lost / Inactive"
        
    rfm["RFM_Segment"] = rfm["RFM_Score"].apply(assign_segment)
    return rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)
# ============================================================
# FORECASTING
# ============================================================
def prepare_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby("Year_Month")["Sales"].sum().reset_index()
    monthly["_period"] = pd.PeriodIndex(monthly["Year_Month"], freq="M")
    monthly = monthly.sort_values("_period").reset_index(drop=True)
    monthly["Year_Month"] = monthly["_period"].astype(str)
    monthly.drop(columns=["_period"], inplace=True)
    monthly["time_idx"] = np.arange(len(monthly))
    return monthly

def train_test_split_time(monthly: pd.DataFrame, test_months: int = 12):
    split = len(monthly) - test_months
    if split < 6: raise ValueError(f"Dataset too short to split: only {len(monthly)} months available.")
    return monthly.iloc[:split].copy(), monthly.iloc[split:].copy()

def naive_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    return np.full(n_test, y_train[-1])

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    return {
        "model": model_name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }

def run_forecasting(df: pd.DataFrame) -> dict:
    monthly = prepare_monthly_series(df)
    train, test = train_test_split_time(monthly, test_months=12)
    X_train, y_train = train[["time_idx"]].values, train["Sales"].values
    X_test, y_test = test[["time_idx"]].values, test["Sales"].values

    pred_naive = naive_baseline(y_train, len(test))
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    metrics = [
        evaluate(y_test, pred_naive, "Naive Baseline"),
        evaluate(y_test, pred_lr, "Linear Regression"),
        evaluate(y_test, pred_rf, "Random Forest"),
    ]
    metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
    best_model = metrics_df.iloc[0]["model"]

    test = test.copy()
    test["Pred_Naive"] = pred_naive
    test["Pred_LR"] = pred_lr
    test["Pred_RF"] = pred_rf

    return {
        "monthly": monthly, "train": train, "test": test,
        "metrics": metrics, "metrics_df": metrics_df, "best_model": best_model,
    }


# ============================================================
# AI INSIGHTS
# ============================================================
def build_insight_context(kpis: dict, cat_df, region_df, rfm_df, forecast_metrics: list, disc_df) -> dict:
    ctx: dict[str, Any] = {}
    for k in ["total_sales", "total_profit", "total_orders", "unique_customers", "avg_order_value"]:
        ctx[k] = kpis.get(k, 0)
    ctx["overall_margin_pct"] = kpis.get("overall_profit_margin_pct", 0)

    if cat_df is not None and len(cat_df):
        top_cat = cat_df.sort_values("Sales", ascending=False).iloc[0]
        worst_cat = cat_df.sort_values("Profit_Margin").iloc[0]
        ctx["top_category_by_sales"] = top_cat["Category"]
        ctx["top_category_sales"] = round(float(top_cat["Sales"]), 2)
        ctx["worst_margin_category"] = worst_cat["Category"]
        ctx["worst_margin_pct"] = round(float(worst_cat["Profit_Margin"]) * 100, 2)

    if region_df is not None and len(region_df):
        top_reg = region_df.sort_values("Sales", ascending=False).iloc[0]
        worst_reg = region_df.sort_values("Profit_Margin").iloc[0]
        ctx["top_region"] = top_reg["Region"]
        ctx["top_region_sales"] = round(float(top_reg["Sales"]), 2)
        ctx["worst_margin_region"] = worst_reg["Region"]
        ctx["worst_margin_region_pct"] = round(float(worst_reg["Profit_Margin"]) * 100, 2)

    if rfm_df is not None and len(rfm_df):
        seg_counts = rfm_df["RFM_Segment"].value_counts().to_dict()
        ctx["rfm_segments"] = seg_counts
        ctx["champions_count"] = seg_counts.get("Champions", 0)
        ctx["at_risk_count"] = seg_counts.get("At Risk", 0)

    if forecast_metrics:
        best = min(forecast_metrics, key=lambda x: x["RMSE"])
        ctx["best_forecast_model"] = best["model"]
        ctx["best_forecast_rmse"] = best["RMSE"]
        ctx["best_forecast_r2"] = best["R2"]

    if disc_df is not None and len(disc_df):
        loss_disc = disc_df[disc_df["Avg_Profit_Margin"] < 0]
        if len(loss_disc):
            worst_disc = loss_disc.sort_values("Avg_Profit_Margin").iloc[0]
            ctx["worst_discount_bin"] = str(worst_disc["Discount_Bin"])
            ctx["worst_discount_margin"] = round(float(worst_disc["Avg_Profit_Margin"]) * 100, 2)

    return ctx

def _generate_template_insights(ctx: dict) -> str:
    lines = [
        "## AI-Generated Business Insights", "*(Demo/Template Mode — interpretations are derived from computed metrics)*", "", "---", "", "### Key Findings", "",
        f"- **Overall Business Health**: The store generated **${ctx.get('total_sales', 0):,.0f}** in total sales across **{ctx.get('total_orders', 0):,} orders** from **{ctx.get('unique_customers', 0):,} unique customers**, achieving an overall profit margin of **{ctx.get('overall_margin_pct', 0):.1f}%**.", "",
        f"- **Category Leadership**: The **{ctx.get('top_category_by_sales', 'N/A')}** category leads in revenue at **${ctx.get('top_category_sales', 0):,.0f}**. The **{ctx.get('worst_margin_category', 'N/A')}** category has the lowest profit margin at **{ctx.get('worst_margin_pct', 0):.1f}%**, suggesting pricing or discount control issues.", "",
        f"- **Regional Performance**: The **{ctx.get('top_region', 'N/A')}** region contributes the highest sales (**${ctx.get('top_region_sales', 0):,.0f}**). The **{ctx.get('worst_margin_region', 'N/A')}** region has the weakest profit margin at **{ctx.get('worst_margin_region_pct', 0):.1f}%**, which may warrant regional cost or pricing review.", ""
    ]
    if "worst_discount_bin" in ctx:
        lines += [f"- **Discount Impact**: Transactions with discounts in the **{ctx['worst_discount_bin']}** range show an average profit margin of **{ctx['worst_discount_margin']:.1f}%**, indicating that aggressive discounting in this band is destroying value rather than driving volume.", ""]
    if "champions_count" in ctx:
        lines += [f"- **Customer Segmentation**: RFM analysis identified **{ctx['champions_count']}** Champions and **{ctx.get('at_risk_count', 0)}** At-Risk customers. Re-engagement campaigns targeting At-Risk customers could recover a meaningful share of lapsed revenue.", ""]
    if "best_forecast_model" in ctx:
        r2 = ctx.get("best_forecast_r2", 0)
        quality = "a reasonably good fit" if r2 > 0.5 else "a limited fit"
        lines += [f"- **Sales Forecasting**: The best-performing model is **{ctx['best_forecast_model']}** (RMSE = ${ctx['best_forecast_rmse']:,.0f}, R² = {r2:.3f}), indicating {quality} for the available monthly data volume.", ""]

    lines += [
        "---", "", "### Business Implications", "",
        "1. **Discount Policy Review**: The data clearly shows that high discounts correlate with negative profit margins.  A structured discount ceiling policy — especially for Office Supplies and Furniture — could meaningfully improve profitability without sacrificing competitive positioning.", "",
        "2. **Customer Retention**: Prioritise re-engagement of At-Risk and Lost/Inactive customer segments through targeted outreach, as their historical purchase behaviour indicates willingness to buy.", "",
        "3. **Product Mix Optimisation**: Loss-making sub-categories should be reviewed for cost structure, supplier negotiations, or pricing adjustments rather than increased discounting.", "",
        "4. **Regional Strategy**: The highest-sales region should be studied as a benchmark for operating practices that could be replicated in lower-margin regions.", "",
        "---", "", "### Areas Requiring Further Investigation", "",
        "- **Seasonality**: Monthly sales show variation; a longer historical series would enable more robust seasonal decomposition and planning.",
        "- **Returns Data**: Returns data was not fully integrated into this analysis; returns can mask true profitability.",
        "- **Customer Lifetime Value**: The current RFM model uses transactional proxies for value.  A formal CLV model would require longer customer history.",
        "- **Product-level Margin Drivers**: Cost of goods data is not included; validation against actual cost records is recommended.", "",
        "---", "", "> ⚠️  **Disclaimer**: The interpretations above are generated by a template-based AI module from pre-computed analytical metrics. They represent plausible business interpretations, not audit-grade findings. All numeric values cited are derived from the dataset and computed by deterministic code — not hallucinated by an AI model."
    ]
    return "\n".join(lines)

def _try_granite_insights(ctx: dict, model_name: str = "ibm-granite/granite-3.3-2b-instruct") -> str | None:
    model_name = os.environ.get("GRANITE_MODEL_NAME", model_name)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        return None
    
    prompt = textwrap.dedent(f"""
        You are a retail business analyst.  Below are verified statistical findings
        from a Superstore retail dataset analysis.  Interpret these findings and provide:
        1. Key business findings (3-5 bullet points)
        2. Business implications
        3. Areas for further investigation

        Do NOT invent any numbers.  Only reference the figures provided.

        === VERIFIED METRICS ===
        Total Sales: ${ctx.get('total_sales', 0):,.0f}
        Total Profit: ${ctx.get('total_profit', 0):,.0f}
        Overall Profit Margin: {ctx.get('overall_margin_pct', 0):.1f}%
        Total Orders: {ctx.get('total_orders', 0):,}
        Unique Customers: {ctx.get('unique_customers', 0):,}
        Average Order Value: ${ctx.get('avg_order_value', 0):,.2f}
        Top Category by Sales: {ctx.get('top_category_by_sales', 'N/A')} (${ctx.get('top_category_sales', 0):,.0f})
        Worst Margin Category: {ctx.get('worst_margin_category', 'N/A')} ({ctx.get('worst_margin_pct', 0):.1f}%)
        Top Region by Sales: {ctx.get('top_region', 'N/A')} (${ctx.get('top_region_sales', 0):,.0f})
        Champions Customers: {ctx.get('champions_count', 'N/A')}
        At-Risk Customers: {ctx.get('at_risk_count', 'N/A')}
        Best Forecast Model: {ctx.get('best_forecast_model', 'N/A')} (RMSE=${ctx.get('best_forecast_rmse', 0):,.0f}, R²={ctx.get('best_forecast_r2', 0):.3f})

        === BEGIN ANALYSIS ===
    """).strip()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded[len(prompt):].strip()
        header = "## AI-Generated Business Insights\n*(IBM Granite — ibm-granite/granite-3.3-2b-instruct)*\n\n---\n\n"
        footer = "\n\n---\n\n> ⚠️  **Disclaimer**: These insights were generated by IBM Granite from pre-computed verified metrics. Numbers cited are from deterministic analysis code, not LLM inference. Treat interpretations as AI-assisted analysis, not authoritative business conclusions."
        return header + response + footer
    except Exception:
        return None

def generate_insights(ctx: dict, prefer_llm: bool = False) -> tuple[str, str]:
    if prefer_llm:
        result = _try_granite_insights(ctx)
        if result: return result, "granite"
    return _generate_template_insights(ctx), "template"
# ============================================================
# APP PAGES & UI LOGIC
# ============================================================

def render_sidebar(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    with st.sidebar:
        st.markdown(
            f'<div style="padding:18px 16px 14px;border-bottom:1px solid {BORDER}">'
            f'<div style="font-size:13px;font-weight:700;color:{TEXT};letter-spacing:0.04em;text-transform:uppercase">Retail Analytics</div>'
            f'<div style="font-size:10px;color:{MUTED};margin-top:2px">IBM SkillsBuild Capstone</div></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div style="padding:14px 16px 4px;font-size:10px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:0.08em">Navigation</div>',
            unsafe_allow_html=True
        )
        NAV = [
            ("Overview", "overview"),
            ("Customer Analytics", "customers"),
            ("Product & Profitability", "products"),
            ("Sales Prediction", "forecast"),
            ("SQL Business Analysis", "sql"),
            ("AI Business Insights", "ai"),
        ]
        page_labels = [n[0] for n in NAV]
        page_ids    = [n[1] for n in NAV]
        sel_label = st.radio("__nav__", page_labels, label_visibility="collapsed")
        sel_id    = page_ids[page_labels.index(sel_label)]

        st.markdown(
            f'<div style="padding:14px 16px 6px;font-size:10px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:0.08em;border-top:1px solid {BORDER};margin-top:6px">Filters</div>',
            unsafe_allow_html=True
        )

        years_all   = sorted(df["Year"].unique().tolist())
        regions_all = sorted(df["Region"].unique().tolist())
        cats_all    = sorted(df["Category"].unique().tolist())
        segs_all    = sorted(df["Segment"].unique().tolist())

        with st.container():
            years_sel   = st.multiselect("Year", years_all, default=None, placeholder="All years", key="f_year")
            regions_sel = st.multiselect("Region", regions_all, default=None, placeholder="All regions", key="f_region")
            cats_sel    = st.multiselect("Category", cats_all, default=None, placeholder="All categories", key="f_cat")
            segs_sel    = st.multiselect("Segment", segs_all, default=None, placeholder="All segments", key="f_seg")

        spacer(4)
        if st.button("Reset Filters", width="stretch", type="secondary"):
            for k in ("f_year", "f_region", "f_cat", "f_seg"): st.session_state.pop(k, None)
            st.rerun()

        ys = years_sel or years_all
        rs = regions_sel or regions_all
        cs = cats_sel or cats_all
        ss = segs_sel or segs_all

        mask = df["Year"].isin(ys) & df["Region"].isin(rs) & df["Category"].isin(cs) & df["Segment"].isin(ss)
        filtered = df[mask]

        spacer(12)
        st.markdown(
            f'<div style="padding:10px 16px;border-top:1px solid {BORDER};font-size:10px;color:{MUTED};line-height:1.6">'
            f'Tableau Sample Superstore<br><span style="color:{TEXT}">{len(filtered):,}</span> rows selected</div>',
            unsafe_allow_html=True
        )
    return sel_id, filtered

def page_overview(df: pd.DataFrame) -> None:
    page_header("Executive Overview", "Business performance across sales, profitability, and geography.")
    kpis = sales_summary(df)
    kpi_row([
        {"label": "Total Sales", "value": fmt_currency(kpis["total_sales"])},
        {"label": "Total Profit", "value": fmt_currency(kpis["total_profit"])},
        {"label": "Orders", "value": f"{kpis['total_orders']:,}"},
        {"label": "Customers", "value": f"{kpis['unique_customers']:,}"},
        {"label": "Avg Order Value", "value": fmt_currency(kpis["avg_order_value"])},
        {"label": "Profit Margin", "value": fmt_pct(kpis["overall_profit_margin_pct"])},
    ])
    spacer(10)
    c1, c2 = st.columns(2, gap="small")
    with c1:
        monthly = monthly_sales_trend(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Year_Month"], y=monthly["Sales"], name="Sales", mode="lines", line=dict(color=ACCENT, width=2), fill="tozeroy", fillcolor="rgba(249,115,22,0.06)"))
        fig.add_trace(go.Scatter(x=monthly["Year_Month"], y=monthly["Profit"], name="Profit", mode="lines", line=dict(color=GREEN, width=1.6, dash="dot")))
        fig.update_xaxes(tickangle=-45, nticks=10, showgrid=False)
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=82))
        chart_card(fig, "Monthly Sales & Profit", height=320)
    with c2:
        cat = sales_by_category(df)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Sales", x=cat["Category"], y=cat["Sales"], marker_color=ACCENT, opacity=0.88, hovertemplate="%{x}<br>Sales: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Bar(name="Profit", x=cat["Category"], y=cat["Profit"], marker_color=GREEN, opacity=0.88, hovertemplate="%{x}<br>Profit: $%{y:,.0f}<extra></extra>"))
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        chart_card(fig, "Category Performance", height=290)
    spacer(6)
    c3, c4 = st.columns(2, gap="small")
    reg = sales_by_region(df)
    with c3:
        fig = go.Figure(go.Bar(x=reg["Region"], y=reg["Sales"], marker_color=CHART_PAL[:len(reg)], text=[fmt_currency(v) for v in reg["Sales"]], textposition="outside", textfont=dict(size=10, color=MUTED), hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(tickprefix="$", tickformat="~s", showgrid=True)
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
        chart_card(fig, "Sales by Region", height=250)
    with c4:
        reg["Margin_Pct"] = reg["Profit_Margin"] * 100
        bar_colors = [GREEN if v >= 10 else ACCENT if v >= 5 else RED for v in reg["Margin_Pct"]]
        fig = go.Figure(go.Bar(x=reg["Region"], y=reg["Margin_Pct"], marker_color=bar_colors, text=[f"{v:.1f}%" for v in reg["Margin_Pct"]], textposition="outside", textfont=dict(size=10, color=MUTED), hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>"))
        fig.update_yaxes(ticksuffix="%", showgrid=True)
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
        chart_card(fig, "Regional Profitability", height=250)
    spacer(6)
    yoy = sales_by_year(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sales", x=yoy["Year"].astype(str), y=yoy["Sales"], marker_color=ACCENT, opacity=0.88, hovertemplate="%{x}<br>Sales: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Profit", x=yoy["Year"].astype(str), y=yoy["Profit"], marker_color=GREEN, opacity=0.88, hovertemplate="%{x}<br>Profit: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    chart_card(fig, "Year-over-Year Performance", height=240)

def page_customers(df: pd.DataFrame) -> None:
    page_header("Customer Analytics", "Customer value distribution, purchasing behavior, and RFM segmentation.")
    rfm, cust = compute_rfm(df), customer_revenue_table(df)
    champs = int(rfm["RFM_Segment"].value_counts().get("Champions", 0))
    at_risk = int(rfm["RFM_Segment"].value_counts().get("At Risk", 0))
    top_rev = float(cust["Total_Sales"].iloc[0]) if len(cust) else 0
    kpi_row([
        {"label": "Customers", "value": f"{len(cust):,}"},
        {"label": "Avg Revenue / Cust", "value": fmt_currency(cust["Total_Sales"].mean())},
        {"label": "Top Customer", "value": fmt_currency(top_rev)},
        {"label": "Champions", "value": str(champs)},
        {"label": "At-Risk", "value": str(at_risk)},
        {"label": "Avg Orders / Cust", "value": f"{cust['Total_Orders'].mean():.1f}"},
    ])
    spacer(10)
    seg_palette = {"Champions": GREEN, "Loyal Customers": BLUE, "Potential Loyalists": ACCENT, "At Risk": "#F59E0B", "Lost / Inactive": RED}
    c1, c2 = st.columns(2, gap="small")
    with c1:
        sc = rfm["RFM_Segment"].value_counts().reset_index()
        sc.columns = ["Segment", "Count"]
        colors = [seg_palette.get(s, ACCENT) for s in sc["Segment"]]
        fig = go.Figure(go.Pie(labels=sc["Segment"], values=sc["Count"], marker_colors=colors, hole=0.48, textinfo="percent", textfont=dict(size=10, color=TEXT), insidetextorientation="radial", hovertemplate="%{label}<br>%{value} customers (%{percent})<extra></extra>"))
        fig.update_layout(margin=dict(l=4, r=4, t=4, b=4), legend=dict(orientation="v", x=1.0, y=0.5, xanchor="left", yanchor="middle", font=dict(size=10, color=MUTED)))
        chart_card(fig, "RFM Segment Distribution", height=300)
    with c2:
        sr = rfm.groupby("RFM_Segment")["Monetary"].sum().reset_index().sort_values("Monetary", ascending=True)
        bar_c = [seg_palette.get(s, ACCENT) for s in sr["RFM_Segment"]]
        fig = go.Figure(go.Bar(x=sr["Monetary"], y=sr["RFM_Segment"], orientation="h", marker_color=bar_c, text=[fmt_currency(v) for v in sr["Monetary"]], textposition="outside", textfont=dict(size=10, color=MUTED), hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
        fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True)
        fig.update_layout(margin=dict(l=4, r=60, t=4, b=4))
        chart_card(fig, "Revenue by RFM Segment", height=300)
    spacer(6)
    top15 = cust.head(15)
    seg_c = {"Consumer": ACCENT, "Corporate": BLUE, "Home Office": GREEN}
    bc = [seg_c.get(s, ACCENT) for s in top15["Segment"]]
    fig = go.Figure(go.Bar(x=top15["Total_Sales"], y=top15["Customer_Name"], orientation="h", marker_color=bc, hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
    chart_card(fig, "Top 15 Customers by Revenue", height=370)
    spacer(6)
    c3, c4 = st.columns(2, gap="small")
    with c3:
        fig = go.Figure(go.Histogram(x=cust["Total_Sales"], nbinsx=32, marker_color=ACCENT, opacity=0.82, hovertemplate="$%{x:,.0f}<br>%{y} customers<extra></extra>"))
        fig.update_xaxes(tickprefix="$", tickformat="~s", title_text="Revenue per Customer")
        fig.update_yaxes(title_text="Customers")
        chart_card(fig, "Customer Revenue Distribution", height=230)
    with c4:
        fig = go.Figure(go.Histogram(x=rfm["RFM_Score"], nbinsx=10, marker_color=BLUE, opacity=0.82, hovertemplate="Score %{x}<br>%{y} customers<extra></extra>"))
        fig.update_xaxes(title_text="RFM Score (3–12)", dtick=1)
        fig.update_yaxes(title_text="Customers")
        chart_card(fig, "RFM Score Distribution", height=230)
    spacer(6)
    section_label("Customer Detail — Top 25 by Revenue")
    tbl = rfm[["Customer_Name", "Segment", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Segment"]].head(25)
    render_table(tbl, {"Monetary": "${:,.0f}", "Recency": "{:.0f}"}, height=400)

def page_products(df: pd.DataFrame) -> None:
    page_header("Product & Profitability", "Sub-category profit drivers, discount impact, and product-level performance.")
    cat, sub, disc = sales_by_category(df), profitability_by_subcategory(df), discount_profit_analysis(df)
    loss_n, profit_n = int((sub["Profit"] < 0).sum()), int((sub["Profit"] >= 0).sum())
    cat_kpis = [{"label": f"{row['Category']} Margin", "value": fmt_pct(row["Profit_Margin"] * 100)} for _, row in cat.iterrows()] + [{"label": "Loss-Making", "value": str(loss_n)}, {"label": "Profitable", "value": str(profit_n)}]
    kpi_row(cat_kpis)
    spacer(10)
    c1, c2 = st.columns(2, gap="small")
    with c1:
        ss = sub.sort_values("Profit")
        cs = [RED if v < 0 else GREEN for v in ss["Profit"]]
        fig = go.Figure(go.Bar(x=ss["Profit"], y=ss["Sub_Category"], orientation="h", marker_color=cs, hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
        fig.add_vline(x=0, line_color=BORDER, line_width=1.2)
        fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True, zeroline=False)
        fig.update_layout(margin=dict(l=4, r=8, t=4, b=4))
        chart_card(fig, "Profit by Sub-Category", height=460)
    with c2:
        sub2 = sales_by_subcategory(df)
        cmap = {"Furniture": ACCENT, "Office Supplies": BLUE, "Technology": GREEN}
        pt_c = [cmap.get(c, ACCENT) for c in sub2["Category"]]
        fig = go.Figure()
        for nm, col in cmap.items():
            if nm in sub2["Category"].values:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=8, color=col), name=nm))
        fig.add_trace(go.Scatter(x=sub2["Sales"], y=sub2["Profit"], mode="markers", marker=dict(color=pt_c, size=9, opacity=0.80, line=dict(width=0.5, color=BORDER)), text=sub2["Sub_Category"], hovertemplate="<b>%{text}</b><br>Sales: $%{x:,.0f}<br>Profit: $%{y:,.0f}<extra></extra>", showlegend=False))
        fig.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
        fig.add_vline(x=float(sub2["Sales"].median()), line_dash="dot", line_color=BORDER, line_width=1)
        fig.update_xaxes(tickprefix="$", tickformat="~s")
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        chart_card(fig, "Sales vs Profit by Sub-Category", height=460)
    spacer(6)
    disc["Avg_Margin_Pct"] = disc["Avg_Profit_Margin"] * 100
    dc = [RED if v < 0 else GREEN for v in disc["Avg_Margin_Pct"]]
    fig = go.Figure(go.Bar(x=disc["Discount_Bin"].astype(str), y=disc["Avg_Margin_Pct"], marker_color=dc, text=[f"{v:.1f}%" for v in disc["Avg_Margin_Pct"]], textposition="outside", textfont=dict(size=10, color=MUTED), hovertemplate="Discount %{x}<br>Avg Margin: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=BORDER, line_width=1.2)
    fig.update_xaxes(title_text="Discount Range")
    fig.update_yaxes(ticksuffix="%", showgrid=True)
    fig.update_layout(margin=dict(l=4, r=48, t=4, b=4))
    chart_card(fig, "Discount Level vs Average Profit Margin", height=250)
    spacer(6)
    section_label("Sub-Category Performance Table")
    sub_tbl = sub2.copy()
    sub_tbl["Margin_%"] = (sub_tbl["Profit_Margin"] * 100).round(1)
    sub_tbl = sub_tbl.drop(columns=["Profit_Margin"]).sort_values("Profit")
    disp_df = sub_tbl[["Category", "Sub_Category", "Sales", "Profit", "Margin_%", "Quantity"]].copy()
    fmts = {"Sales":"${:,.0f}", "Profit":"${:,.0f}", "Margin_%":"{:.1f}%", "Quantity":"{:,}"}
    for col, fmt in fmts.items():
        disp_df[col] = disp_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else x)
    render_table(disp_df, height=360)

def page_forecast(df: pd.DataFrame) -> None:
    page_header("Sales Prediction", "Monthly sales modelled with time-index regression. Last 12 months = test set.")
    df_for_forecast = st.session_state.get("_df_full", df)
    forecast = run_forecasting(df_for_forecast)
    train, test, best, m_df = forecast["train"], forecast["test"], forecast["best_model"], forecast["metrics_df"]
    best_row = m_df.iloc[0]
    kpi_row([
        {"label": "Best Model", "value": best.replace(" Regression", " Reg.")},
        {"label": "MAE", "value": f"{best_row['MAE']:,.0f}"},
        {"label": "RMSE", "value": f"{best_row['RMSE']:,.0f}"},
        {"label": "R²", "value": f"{best_row['R2']:.3f}"},
        {"label": "Train Months", "value": str(len(train))},
        {"label": "Test Months", "value": str(len(test))},
    ])
    spacer(10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["Year_Month"], y=train["Sales"], name="Historical", mode="lines", line=dict(color=BLUE, width=2), hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Sales"], name="Actual (test)", mode="lines", line=dict(color=BLUE, width=2, dash="dot"), hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Pred_LR"], name="Linear Regression", mode="lines", line=dict(color=ACCENT, width=2), hovertemplate="%{x}<br>LR: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Pred_RF"], name="Random Forest", mode="lines", line=dict(color=GREEN, width=1.6), hovertemplate="%{x}<br>RF: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Pred_Naive"], name="Naive Baseline", mode="lines", line=dict(color=MUTED, width=1.2, dash="dot"), hovertemplate="%{x}<br>Naive: $%{y:,.0f}<extra></extra>"))
    if len(train):
        split_label = train["Year_Month"].iloc[-1]
        all_months = list(train["Year_Month"]) + list(test["Year_Month"])
        split_x = all_months.index(split_label) / (len(all_months) - 1)
        fig.add_shape(type="line", xref="paper", yref="paper", x0=split_x, x1=split_x, y0=0, y1=1, line=dict(color=BORDER, width=1, dash="dot"))
        fig.add_annotation(xref="paper", yref="paper", x=split_x + 0.01, y=0.98, text="train / test", showarrow=False, font=dict(size=9, color=MUTED), xanchor="left")
    fig.update_xaxes(tickangle=-45, nticks=14)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    fig.update_layout(margin=dict(l=4, r=8, t=4, b=72))
    chart_card(fig, "Monthly Sales — Historical vs Predicted", height=400)
    spacer(8)
    section_label("Model Evaluation — Test Set")
    render_table(m_df[["model", "MAE", "RMSE", "R2"]].rename(columns={"model": "Model", "R2": "R²"}), {"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R²": "{:.4f}"}, height=148)
    spacer(8)
    section_label("Model Limitations")
    st.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:14px 18px;font-size:12px;line-height:1.75;color:{MUTED}">'
        f'<ul style="margin:0;padding-left:16px">'
        f'<li>~36 training months — insufficient for robust seasonal decomposition.</li>'
        f'<li>Single time-index feature; no external regressors or holiday effects.</li>'
        f'<li>Random Forest cannot extrapolate beyond training range.</li>'
        f'<li>Treat forecasts as directional indicators only, not precise projections.</li>'
        f'</ul></div>', unsafe_allow_html=True
    )

def page_sql_analysis(df: pd.DataFrame) -> None:
    page_header("SQL Business Analysis", "Run SQL queries against the local SQLite database to verify data consistency.")
    
    conn = get_sql_connection(st.session_state.get("_df_full", df))
    
    st.markdown(
        f'<div style="background:{CARD2};border:1px solid {BORDER};border-radius:6px;padding:14px 18px;font-size:13px;color:{TEXT}">'
        f'This section executes raw SQL queries against a programmatically generated <code>sqlite3</code> database containing the cleaned dataset.'
        f' This verifies that the pandas-based dashboard metrics can be independently validated via SQL.'
        f'</div>', unsafe_allow_html=True
    )
    spacer(12)
    
    query_keys = list(BUSINESS_QUERIES.keys())
    selected_query_key = st.selectbox("Select Business Query to Execute:", query_keys, format_func=lambda k: BUSINESS_QUERIES[k][0])
    
    if selected_query_key:
        title, sql = BUSINESS_QUERIES[selected_query_key]
        section_label(f"Executing: {title}")
        
        with st.expander("View Raw SQL Query"):
            st.code(textwrap.dedent(sql).strip(), language="sql")
        
        with st.spinner("Executing query..."):
            try:
                results_df = run_query(conn, sql)
                st.markdown(f"**Results ({len(results_df)} rows):**")
                
                # Render common column formats
                fmts = {}
                for col in results_df.columns:
                    if "Sales" in col or "Profit" in col: fmts[col] = "${:,.2f}"
                    if "Pct" in col or "Margin" in col: fmts[col] = "{:.2f}%"
                    
                render_table(results_df, fmts, height=400)
                
            except Exception as e:
                st.error(f"Error executing query: {e}")


def page_ai_insights(df: pd.DataFrame) -> None:
    page_header("AI Business Insights", "AI-assisted interpretation of verified analytical results. Numbers are computed deterministically; AI provides language interpretation only.")
    kpis, cat_df, region_df, rfm_df, disc_df = sales_summary(df), sales_by_category(df), sales_by_region(df), compute_rfm(df), discount_profit_analysis(df)
    forecast = run_forecasting(st.session_state.get("_df_full", df))
    section_label("Verified Analytical Metrics", margin_top=0)
    st.markdown(f'<div style="font-size:11px;color:{MUTED};margin-bottom:8px">Computed by deterministic analytics code — not generated by AI.</div>', unsafe_allow_html=True)
    top_cat, worst_cat = cat_df.sort_values("Sales", ascending=False).iloc[0], cat_df.sort_values("Profit_Margin").iloc[0]
    top_reg, worst_reg = region_df.sort_values("Sales", ascending=False).iloc[0], region_df.sort_values("Profit_Margin").iloc[0]
    champs, at_risk = int(rfm_df["RFM_Segment"].value_counts().get("Champions", 0)), int(rfm_df["RFM_Segment"].value_counts().get("At Risk", 0))
    kpi_row([
        {"label": "Total Sales", "value": fmt_currency(kpis["total_sales"])},
        {"label": "Total Profit", "value": fmt_currency(kpis["total_profit"])},
        {"label": "Profit Margin", "value": fmt_pct(kpis["overall_profit_margin_pct"])},
        {"label": "Orders", "value": f"{kpis['total_orders']:,}"},
        {"label": "Customers", "value": f"{kpis['unique_customers']:,}"},
        {"label": "Avg Order Value", "value": fmt_currency(kpis["avg_order_value"])},
    ])
    spacer(6)
    kpi_row([
        {"label": "Top Category", "value": top_cat["Category"]},
        {"label": "Lowest Margin Category", "value": f"{worst_cat['Category']} ({fmt_pct(worst_cat['Profit_Margin']*100)})"},
        {"label": "Top Region", "value": top_reg["Region"]},
        {"label": "Weakest Margin Region", "value": f"{worst_reg['Region']} ({fmt_pct(worst_reg['Profit_Margin']*100)})"},
        {"label": "Champions", "value": str(champs)},
        {"label": "At-Risk Customers", "value": str(at_risk)},
    ])
    spacer(16)
    st.markdown(f"<div style='border-top:1px solid {BORDER}'></div>", unsafe_allow_html=True)
    spacer(12)
    section_label("AI-Generated Interpretation")
    prefer_llm = st.checkbox("Use IBM Granite (requires `transformers` + local model download)", value=False)
    spacer(4)
    col_btn, col_note = st.columns([1, 4], gap="small")
    with col_btn: run_ai = st.button("Generate Insights", type="primary", width="stretch")
    with col_note: st.markdown(f'<div style="font-size:11px;color:{MUTED};padding-top:10px">Click to generate AI-assisted interpretation from the metrics above.</div>', unsafe_allow_html=True)
    if run_ai:
        with st.spinner("Generating…"):
            ctx = build_insight_context(kpis, cat_df, region_df, rfm_df, forecast["metrics"], disc_df)
            text, mode = generate_insights(ctx, prefer_llm=prefer_llm)
        mode_label = "IBM Granite" if mode == "granite" else "Template / Demo Mode"
        st.markdown(f'<span style="display:inline-block;background:{CARD2};border:1px solid {BORDER};border-radius:3px;padding:2px 8px;font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px">Mode: {mode_label}</span>', unsafe_allow_html=True)
        
        def _md_lite(t: str) -> str:
            lines, out, in_ul = t.splitlines(), [], False
            for line in lines:
                s = line.strip()
                if s.startswith(("- ", "* ")):
                    if not in_ul: out.append("<ul style='margin:6px 0;padding-left:16px'>"); in_ul = True
                    c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s[2:])
                    out.append(f"<li style='margin-bottom:3px'>{c}</li>")
                else:
                    if in_ul: out.append("</ul>"); in_ul = False
                    if s.startswith("> "): out.append(f'<div style="font-size:10px;color:{MUTED};border-top:1px solid {BORDER};margin-top:10px;padding-top:8px">' + re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s[2:]) + "</div>")
                    elif s:
                        c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
                        m = re.match(r"^(\d+)\.\s+(.*)", c)
                        if m: out.append(f"<p style='margin:5px 0'><strong>{m.group(1)}.</strong> {m.group(2)}</p>")
                        else: out.append(f"<p style='margin:5px 0'>{c}</p>")
            if in_ul: out.append("</ul>")
            return "\n".join(out)
            
        def _render_ai_sections(t: str) -> None:
            SECTION_MAP = {"### Key Findings": "Key Findings", "### Business Implications": "Business Implications", "### Areas Requiring Further Investigation": "Areas for Investigation"}
            cur_title, cur_block = None, []
            def _flush(title, block):
                if not title: return
                content = "\n".join(l for l in block if not l.startswith("#") and "---" not in l and "Demo/Template Mode" not in l and "AI-Generated Business Insights" not in l).strip()
                if not content: return
                section_label(title)
                st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:14px 18px;font-size:13px;line-height:1.75;color:{TEXT}">{_md_lite(content)}</div>', unsafe_allow_html=True)
                spacer(6)
            for line in t.splitlines():
                s = line.strip()
                matched = False
                for md_h, disp in SECTION_MAP.items():
                    if s == md_h:
                        _flush(cur_title, cur_block)
                        cur_title, cur_block = disp, []
                        matched = True
                        break
                if not matched: cur_block.append(line)
            _flush(cur_title, cur_block)

        _render_ai_sections(text)
    else:
        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:20px 24px;font-size:13px;color:{MUTED}">Press <strong style="color:{TEXT}">Generate Insights</strong> to produce AI-assisted analysis of the verified metrics above.</div>', unsafe_allow_html=True)
    spacer(14)
    st.markdown(f'<div style="background:{CARD2};border:1px solid {BORDER};border-left:2px solid {ACCENT};border-radius:0 5px 5px 0;padding:10px 14px;font-size:11px;color:{MUTED}"><strong style="color:{TEXT}">Responsible AI</strong> — AI output is interpretation, not audit-grade evidence. All numbers originate from deterministic code. Template mode is rule-based, not a trained LLM.</div>', unsafe_allow_html=True)


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main() -> None:
    _card_counter[0] = 0
    try:
        df_full = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return
        
    st.session_state["_df_full"] = df_full
    sel_id, df_filtered = render_sidebar(df_full)

    if len(df_filtered) == 0:
        st.markdown(f'<div style="background:{CARD};border:1px solid {RED};border-radius:6px;padding:18px 20px;color:{RED};font-size:13px">No data matches the current filters. Adjust or reset filters in the sidebar.</div>', unsafe_allow_html=True)
        return

    pages = {
        "overview": page_overview,
        "customers": page_customers,
        "products": page_products,
        "forecast": page_forecast,
        "sql": page_sql_analysis,
        "ai": page_ai_insights,
    }
    pages[sel_id](df_filtered)

if __name__ == "__main__":
    main()
