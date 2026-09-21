"""
streamlit_app.py
----------------
Interactive Streamlit dashboard for the Retail Sales AI Analytics project.

Sections:
  1. Executive Overview      — KPIs + high-level charts
  2. Customer Analytics      — RFM segments, top customers
  3. Product & Profitability — category/sub-category drill-down
  4. Sales Prediction        — forecast vs actual
  5. AI Business Insights    — AI-generated interpretation

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Allow imports from project root when running from app/ directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)   # ensure relative data paths resolve correctly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.feature_engineering import load_processed
from src.analytics import (
    sales_summary, sales_by_year, monthly_sales_trend, sales_by_category,
    sales_by_subcategory, sales_by_region, profitability_by_subcategory,
    discount_profit_analysis, quarterly_sales,
)
from src.customer_analysis import customer_revenue_table, compute_rfm
from src.forecasting import prepare_monthly_series, train_test_split_time
from src.ai_insights import build_insight_context, generate_insights

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Sales AI Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    return load_processed()


@st.cache_data(show_spinner="Running forecasting models…")
def get_forecast(df: pd.DataFrame):
    from src.forecasting import run_forecasting
    return run_forecasting(df)


# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("🔍 Filters")

    years = sorted(df["Year"].unique())
    selected_years = st.sidebar.multiselect("Year", years, default=years)

    regions = sorted(df["Region"].unique())
    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

    categories = sorted(df["Category"].unique())
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

    segments = sorted(df["Segment"].unique())
    selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)

    mask = (
        df["Year"].isin(selected_years)
        & df["Region"].isin(selected_regions)
        & df["Category"].isin(selected_categories)
        & df["Segment"].isin(selected_segments)
    )
    filtered = df[mask]

    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Rows", f"{len(filtered):,}")
    return filtered


# ---------------------------------------------------------------------------
# Page 1 — Executive Overview
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame):
    st.title("📊 Executive Overview")

    kpis = sales_summary(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Sales", f"${kpis['total_sales']:,.0f}")
    c2.metric("Total Profit", f"${kpis['total_profit']:,.0f}")
    c3.metric("Total Orders", f"{kpis['total_orders']:,}")
    c4.metric("Unique Customers", f"{kpis['unique_customers']:,}")
    c5.metric("Avg Order Value", f"${kpis['avg_order_value']:,.0f}")
    c6.metric("Profit Margin", f"{kpis['overall_profit_margin_pct']:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Sales & Profit Trend")
        monthly = monthly_sales_trend(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Year_Month"], y=monthly["Sales"],
                                 mode="lines+markers", name="Sales",
                                 line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(x=monthly["Year_Month"], y=monthly["Profit"],
                                 mode="lines+markers", name="Profit",
                                 line=dict(color="#16a34a", width=2, dash="dash")))
        fig.update_layout(height=300, margin=dict(t=10, b=10),
                          xaxis_tickangle=-45, legend=dict(orientation="h"))
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sales & Profit by Category")
        cat = sales_by_category(df)
        fig = px.bar(cat, x="Category", y=["Sales", "Profit"],
                     barmode="group",
                     color_discrete_map={"Sales": "#2563eb", "Profit": "#16a34a"},
                     height=300)
        fig.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"))
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Sales by Region")
        reg = sales_by_region(df)
        fig = px.bar(reg, x="Region", y="Sales", color="Region",
                     color_discrete_sequence=px.colors.qualitative.Safe, height=280)
        fig.update_layout(margin=dict(t=10, b=10), showlegend=False)
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Profit Margin by Region")
        reg["Margin_Pct"] = reg["Profit_Margin"] * 100
        fig = px.bar(reg, x="Region", y="Margin_Pct", color="Region",
                     color_discrete_sequence=px.colors.qualitative.Pastel, height=280)
        fig.update_layout(margin=dict(t=10, b=10), showlegend=False)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Year-over-Year Performance")
    yoy = sales_by_year(df)
    fig = px.bar(yoy, x="Year", y=["Sales", "Profit"], barmode="group",
                 color_discrete_map={"Sales": "#2563eb", "Profit": "#16a34a"})
    fig.update_layout(height=280, margin=dict(t=10, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Customer Analytics
# ---------------------------------------------------------------------------

def page_customers(df: pd.DataFrame):
    st.title("👥 Customer Analytics")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("RFM Customer Segments")
        rfm = compute_rfm(df)
        seg_counts = rfm["RFM_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(seg_counts, names="Segment", values="Count",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     height=350)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by RFM Segment")
        seg_rev = rfm.groupby("RFM_Segment")["Monetary"].sum().reset_index()
        fig = px.bar(seg_rev, x="RFM_Segment", y="Monetary",
                     color="RFM_Segment",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     height=350)
        fig.update_layout(margin=dict(t=10, b=10), showlegend=False,
                          xaxis_tickangle=-20)
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 Customers by Revenue")
    cust = customer_revenue_table(df).head(15)
    fig = px.bar(cust, x="Total_Sales", y="Customer_Name", orientation="h",
                 color="Segment",
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 height=450)
    fig.update_layout(margin=dict(t=10, b=10), yaxis={"categoryorder": "total ascending"})
    fig.update_xaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Customer Revenue Distribution")
        cust_all = customer_revenue_table(df)
        fig = px.histogram(cust_all, x="Total_Sales", nbins=40,
                           color_discrete_sequence=["#2563eb"], height=280)
        fig.update_layout(margin=dict(t=10, b=10))
        fig.update_xaxes(tickprefix="$", title="Total Revenue per Customer")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("RFM Score Distribution")
        fig = px.histogram(rfm, x="RFM_Score", nbins=10,
                           color_discrete_sequence=["#16a34a"], height=280)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("RFM Table (Top 20 by Monetary)")
    st.dataframe(
        rfm[["Customer_Name", "Segment", "Recency", "Frequency",
             "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Segment"]]
        .head(20)
        .style.format({"Monetary": "${:,.0f}", "Recency": "{:.0f} days"}),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Page 3 — Product & Profitability
# ---------------------------------------------------------------------------

def page_products(df: pd.DataFrame):
    st.title("📦 Product & Profitability")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sub-Category Profit (Loss in red)")
        sub = profitability_by_subcategory(df).sort_values("Profit")
        colors = ["#ef4444" if v < 0 else "#22c55e" for v in sub["Profit"]]
        fig = go.Figure(go.Bar(
            x=sub["Profit"],
            y=sub["Sub_Category"],
            orientation="h",
            marker_color=colors,
        ))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(height=500, margin=dict(t=10, b=10))
        fig.update_xaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sales vs Profit by Sub-Category")
        sub2 = sales_by_subcategory(df)
        fig = px.scatter(sub2, x="Sales", y="Profit", color="Category",
                         hover_data=["Sub_Category"],
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         height=500)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_xaxes(tickprefix="$")
        fig.update_yaxes(tickprefix="$")
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Discount vs. Average Profit Margin")
    disc = discount_profit_analysis(df)
    disc["Avg_Profit_Margin_Pct"] = disc["Avg_Profit_Margin"] * 100
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in disc["Avg_Profit_Margin_Pct"]]
    fig = go.Figure(go.Bar(
        x=disc["Discount_Bin"].astype(str),
        y=disc["Avg_Profit_Margin_Pct"],
        marker_color=colors,
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=300, margin=dict(t=10, b=10),
                      xaxis_title="Discount Range", yaxis_title="Avg Profit Margin (%)")
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category Performance Table")
    sub_tbl = sales_by_subcategory(df)
    sub_tbl["Profit_Margin_Pct"] = (sub_tbl["Profit_Margin"] * 100).round(2)
    st.dataframe(
        sub_tbl[["Category", "Sub_Category", "Sales", "Profit", "Profit_Margin_Pct", "Quantity"]]
        .style.format({
            "Sales": "${:,.0f}", "Profit": "${:,.0f}",
            "Profit_Margin_Pct": "{:.1f}%", "Quantity": "{:,}",
        })
        .applymap(lambda v: "color: red" if isinstance(v, (int, float)) and v < 0 else "",
                  subset=["Profit", "Profit_Margin_Pct"]),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Page 4 — Sales Prediction
# ---------------------------------------------------------------------------

def page_forecast(df: pd.DataFrame):
    st.title("📈 Sales Prediction")

    st.info(
        "**Methodology**: Monthly sales aggregates are modelled using a numeric time "
        "index. The last 12 months are held out as a test set (no random split). "
        "Models evaluated: Naive Baseline, Linear Regression, and Random Forest. "
        "Metrics: MAE, RMSE, R².",
        icon="ℹ️"
    )

    forecast = get_forecast(df)
    train = forecast["train"]
    test = forecast["test"]
    metrics = forecast["metrics"]

    # Metrics table
    st.subheader("Model Evaluation (Test Set)")
    m_df = pd.DataFrame(metrics).sort_values("RMSE")
    st.dataframe(
        m_df.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.4f}"}),
        use_container_width=True,
    )

    # Forecast chart
    st.subheader("Actual vs Predicted Monthly Sales")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["Year_Month"], y=train["Sales"],
                             name="Training Actual", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Sales"],
                             name="Test Actual", line=dict(color="#2563eb", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Pred_LR"],
                             name="Linear Regression", line=dict(color="#f97316", width=1.8)))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Pred_RF"],
                             name="Random Forest", line=dict(color="#16a34a", width=1.8)))
    fig.add_trace(go.Scatter(x=test["Year_Month"], y=test["Pred_Naive"],
                             name="Naive Baseline",
                             line=dict(color="#9ca3af", width=1.2, dash="dot")))
    fig.add_vline(
        x=len(train) - 1,
        line_dash="dot", line_color="gray",
        annotation_text="Train/Test split"
    )
    fig.update_layout(height=420, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=-0.2),
                      xaxis_tickangle=-45)
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ Model Limitations")
    st.markdown("""
    - This dataset spans only ~4 years, giving limited monthly data points for
      robust forecasting.
    - Only a numeric time index is used as a feature — no explicit seasonality dummies
      to avoid overfitting on short series.
    - Forecasts beyond the observed date range should not be treated as business
      projections without further validation.
    - The store may have one-off events (promotions, external disruptions) not captured
      by any model trained on this data alone.
    """)


# ---------------------------------------------------------------------------
# Page 5 — AI Business Insights
# ---------------------------------------------------------------------------

def page_ai_insights(df: pd.DataFrame):
    st.title("🤖 AI Business Insights")

    st.warning(
        "All numeric values in the AI interpretation are derived from pre-computed "
        "deterministic analytics — they are NOT generated by the AI model.  "
        "The AI layer provides natural-language interpretation only.",
        icon="⚠️"
    )

    kpis = sales_summary(df)
    cat_df = sales_by_category(df)
    region_df = sales_by_region(df)
    rfm_df = compute_rfm(df)
    disc_df = discount_profit_analysis(df)

    forecast = get_forecast(df)
    forecast_metrics = forecast["metrics"]

    st.subheader("📋 Verified Analytical Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"${kpis['total_sales']:,.0f}")
        st.metric("Total Profit", f"${kpis['total_profit']:,.0f}")
        st.metric("Profit Margin", f"{kpis['overall_profit_margin_pct']:.1f}%")
    with col2:
        st.metric("Total Orders", f"{kpis['total_orders']:,}")
        st.metric("Unique Customers", f"{kpis['unique_customers']:,}")
        st.metric("Avg Order Value", f"${kpis['avg_order_value']:,.0f}")
    with col3:
        top_cat = cat_df.sort_values("Sales", ascending=False).iloc[0]
        st.metric("Top Category", top_cat["Category"])
        top_reg = region_df.sort_values("Sales", ascending=False).iloc[0]
        st.metric("Top Region", top_reg["Region"])
        champ = rfm_df["RFM_Segment"].value_counts().get("Champions", 0)
        st.metric("Champion Customers", champ)

    st.markdown("---")
    st.subheader("🤖 AI-Generated Interpretation")

    prefer_llm = st.checkbox(
        "Attempt IBM Granite (local LLM) — requires `transformers` + model downloaded",
        value=False,
    )

    if st.button("Generate AI Insights", type="primary"):
        with st.spinner("Generating insights…"):
            ctx = build_insight_context(
                kpis=kpis,
                cat_df=cat_df,
                region_df=region_df,
                rfm_df=rfm_df,
                forecast_metrics=forecast_metrics,
                disc_df=disc_df,
            )
            insight_text, mode = generate_insights(ctx, prefer_llm=prefer_llm)

        mode_label = "IBM Granite (Local LLM)" if mode == "granite" else "Template / Demo Mode"
        st.caption(f"Generation mode: **{mode_label}**")
        st.markdown(insight_text)

        # Save to file
        report_path = Path("outputs/reports/ai_insights.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(insight_text, encoding="utf-8")
        st.success(f"Insights saved to {report_path}")
    else:
        st.caption("Click the button above to generate AI-assisted insights.")


# ---------------------------------------------------------------------------
# Main app routing
# ---------------------------------------------------------------------------

def main():
    df_full = load_data()

    # Navigation
    pages = {
        "📊 Executive Overview": page_overview,
        "👥 Customer Analytics": page_customers,
        "📦 Product & Profitability": page_products,
        "📈 Sales Prediction": page_forecast,
        "🤖 AI Business Insights": page_ai_insights,
    }

    st.sidebar.title("🛒 Retail Analytics")
    st.sidebar.markdown("IBM SkillsBuild Capstone Project")
    st.sidebar.markdown("---")

    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

    # Apply global filters
    df_filtered = sidebar_filters(df_full)

    if len(df_filtered) == 0:
        st.error("No data matches the current filter selection. Please adjust the filters.")
        return

    # Render selected page
    pages[selected_page](df_filtered)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: Tableau Sample Superstore\n\n"
        "IBM SkillsBuild Academic Internship\n"
        "Data Analytics with AI"
    )


if __name__ == "__main__":
    main()
