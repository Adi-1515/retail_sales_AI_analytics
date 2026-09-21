"""
analytics.py
------------
Core exploratory data analysis (EDA) and business analytics functions.

All functions accept a processed DataFrame (output of feature_engineering)
and return either aggregated DataFrames or produce Matplotlib/Seaborn figures
saved to outputs/figures/.

Design: every public function is independently callable and does NOT assume
any global state — making them reusable in notebooks, scripts, and Streamlit.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts/Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared plot style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted", 10)


def _save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ============================================================
# SALES ANALYSIS
# ============================================================

def sales_summary(df: pd.DataFrame) -> dict:
    """High-level KPI summary."""
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
    return (
        df.groupby("Year")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order_ID", "nunique"))
        .reset_index()
    )


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("Year_Month")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    monthly["Year_Month"] = pd.PeriodIndex(monthly["Year_Month"], freq="M")
    monthly = monthly.sort_values("Year_Month").reset_index(drop=True)
    monthly["Year_Month"] = monthly["Year_Month"].astype(str)
    return monthly


def quarterly_sales(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Year", "Quarter", "Year_Quarter"])
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
        .sort_values(["Year", "Quarter"])
        .reset_index(drop=True)
    )


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Category")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order_ID", "nunique"))
        .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"])
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def sales_by_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Category", "Sub_Category"])
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"))
        .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"])
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Region")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order_ID", "nunique"))
        .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"])
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def sales_by_state(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("State")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"])
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


# ============================================================
# PROFITABILITY ANALYSIS
# ============================================================

def profitability_by_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    sub = (
        df.groupby(["Category", "Sub_Category"])
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"))
        .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"])
        .reset_index()
        .sort_values("Profit", ascending=False)
    )
    sub["Is_Loss"] = sub["Profit"] < 0
    return sub


def discount_profit_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Bin orders by discount range and show avg profit margin."""
    bins = [0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
    labels = ["0%", "1-10%", "11-20%", "21-30%", "31-40%", "41-50%", "51-60%", "61-80%", ">80%"]
    d = df.copy()
    d["Discount_Bin"] = pd.cut(d["Discount"], bins=bins, labels=labels, right=False)
    return (
        d.groupby("Discount_Bin", observed=True)
        .agg(
            Avg_Profit_Margin=("Profit_Margin", "mean"),
            Avg_Profit=("Profit", "mean"),
            Avg_Sales=("Sales", "mean"),
            Count=("Sales", "count"),
        )
        .reset_index()
    )


# ============================================================
# FIGURE GENERATION
# ============================================================

def plot_monthly_sales(df: pd.DataFrame) -> Path:
    monthly = monthly_sales_trend(df)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(monthly["Year_Month"], monthly["Sales"] / 1e3, marker="o", linewidth=1.8,
            markersize=3, color="#2563eb", label="Sales ($k)")
    ax.plot(monthly["Year_Month"], monthly["Profit"] / 1e3, marker="s", linewidth=1.5,
            markersize=3, color="#16a34a", label="Profit ($k)", linestyle="--")
    step = max(1, len(monthly) // 12)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly["Year_Month"].iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("USD (thousands)")
    ax.set_title("Monthly Sales and Profit Trend")
    ax.legend()
    return _save(fig, "monthly_sales_trend.png")


def plot_sales_by_category(df: pd.DataFrame) -> Path:
    cat = sales_by_category(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=cat, x="Category", y="Sales", ax=axes[0],
                hue="Category", palette="Blues_d", legend=False)
    axes[0].set_title("Total Sales by Category")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    axes[0].set_xlabel("")

    sns.barplot(data=cat, x="Category", y="Profit", ax=axes[1],
                hue="Category", palette="Greens_d", legend=False)
    axes[1].set_title("Total Profit by Category")
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    axes[1].set_xlabel("")
    plt.tight_layout()
    return _save(fig, "sales_profit_by_category.png")


def plot_subcategory_profit(df: pd.DataFrame) -> Path:
    sub = profitability_by_subcategory(df).sort_values("Profit")
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in sub["Profit"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(sub["Sub_Category"], sub["Profit"] / 1e3, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Profit (USD thousands)")
    ax.set_title("Profit by Sub-Category (red = loss-making)")
    plt.tight_layout()
    return _save(fig, "profit_by_subcategory.png")


def plot_discount_vs_profit(df: pd.DataFrame) -> Path:
    disc = discount_profit_analysis(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in disc["Avg_Profit_Margin"]]
    ax.bar(disc["Discount_Bin"].astype(str), disc["Avg_Profit_Margin"] * 100, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Discount Range")
    ax.set_ylabel("Avg Profit Margin (%)")
    ax.set_title("Average Profit Margin by Discount Level")
    plt.tight_layout()
    return _save(fig, "discount_vs_profit_margin.png")


def plot_region_performance(df: pd.DataFrame) -> Path:
    reg = sales_by_region(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=reg, x="Region", y="Sales", ax=axes[0],
                hue="Region", palette="Blues_d", legend=False)
    axes[0].set_title("Sales by Region")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    axes[0].set_xlabel("")

    sns.barplot(data=reg, x="Region", y="Profit_Margin", ax=axes[1],
                hue="Region", palette="Greens_d", legend=False)
    axes[1].set_title("Profit Margin by Region")
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    axes[1].set_xlabel("")
    plt.tight_layout()
    return _save(fig, "region_performance.png")


def run_all_eda(df: pd.DataFrame) -> dict:
    """
    Execute all EDA analytics and save all figures.
    Returns a dict of aggregated DataFrames for downstream use.
    """
    print("[analytics] Running full EDA...")
    results = {
        "kpis": sales_summary(df),
        "sales_by_year": sales_by_year(df),
        "monthly_trend": monthly_sales_trend(df),
        "quarterly_sales": quarterly_sales(df),
        "sales_by_category": sales_by_category(df),
        "sales_by_subcategory": sales_by_subcategory(df),
        "sales_by_region": sales_by_region(df),
        "sales_by_state": sales_by_state(df),
        "profitability_by_subcategory": profitability_by_subcategory(df),
        "discount_profit": discount_profit_analysis(df),
    }
    # Generate and save all figures
    plot_monthly_sales(df)
    plot_sales_by_category(df)
    plot_subcategory_profit(df)
    plot_discount_vs_profit(df)
    plot_region_performance(df)
    print("[analytics] EDA complete. Figures saved to outputs/figures/")
    return results
