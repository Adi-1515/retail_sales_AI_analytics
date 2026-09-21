"""
pipeline.py
-----------
End-to-end pipeline runner for the Retail Sales AI Analytics project.

Running this script executes:
  1. Data loading and quality check
  2. Data cleaning
  3. Feature engineering and persistence
  4. EDA and figure generation
  5. Customer analytics (RFM)
  6. SQLite database creation + all business queries
  7. Sales forecasting
  8. AI-assisted insight generation

Usage:
    python pipeline.py
"""

from pathlib import Path
import sys
import os

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_raw_orders, generate_data_quality_report, print_data_quality_report
from src.data_cleaning import clean_orders
from src.feature_engineering import engineer_features, save_processed
from src.analytics import run_all_eda
from src.customer_analysis import run_customer_analysis
from src.database import build_database, run_all_business_queries
from src.forecasting import run_forecasting
from src.ai_insights import build_insight_context, generate_insights

BANNER = "=" * 64


def main():
    print(f"\n{BANNER}")
    print("  RETAIL SALES AI ANALYTICS — FULL PIPELINE")
    print(f"{BANNER}\n")

    # ------------------------------------------------------------------
    # STEP 1: Load raw data
    # ------------------------------------------------------------------
    print(f"\n--- STEP 1: Data Loading ---")
    df_raw = load_raw_orders()
    report = generate_data_quality_report(df_raw)
    print_data_quality_report(report)

    # ------------------------------------------------------------------
    # STEP 2: Data cleaning
    # ------------------------------------------------------------------
    print(f"\n--- STEP 2: Data Cleaning ---")
    df_clean = clean_orders(df_raw)

    # ------------------------------------------------------------------
    # STEP 3: Feature engineering + save
    # ------------------------------------------------------------------
    print(f"\n--- STEP 3: Feature Engineering ---")
    df = engineer_features(df_clean)
    save_processed(df)

    # ------------------------------------------------------------------
    # STEP 4: EDA
    # ------------------------------------------------------------------
    print(f"\n--- STEP 4: Exploratory Data Analysis ---")
    eda_results = run_all_eda(df)

    # ------------------------------------------------------------------
    # STEP 5: Customer analytics
    # ------------------------------------------------------------------
    print(f"\n--- STEP 5: Customer Analytics (RFM) ---")
    cust_results = run_customer_analysis(df)

    # ------------------------------------------------------------------
    # STEP 6: SQLite database + business queries
    # ------------------------------------------------------------------
    print(f"\n--- STEP 6: Database & SQL Analysis ---")
    build_database(df)
    sql_results = run_all_business_queries()
    for name, result_df in sql_results.items():
        print(f"\n  [{name}]")
        print(result_df.to_string(index=False))

    # ------------------------------------------------------------------
    # STEP 7: Forecasting
    # ------------------------------------------------------------------
    print(f"\n--- STEP 7: Sales Forecasting ---")
    forecast_results = run_forecasting(df)

    # ------------------------------------------------------------------
    # STEP 8: AI Insights
    # ------------------------------------------------------------------
    print(f"\n--- STEP 8: AI Business Insights ---")
    ctx = build_insight_context(
        kpis=eda_results["kpis"],
        cat_df=eda_results["sales_by_category"],
        region_df=eda_results["sales_by_region"],
        rfm_df=cust_results["rfm"],
        forecast_metrics=forecast_results["metrics"],
        disc_df=eda_results["discount_profit"],
    )
    prefer_llm = os.environ.get("USE_GRANITE", "0") == "1"
    insight_text, mode = generate_insights(ctx, prefer_llm=prefer_llm)
    print(f"  Mode: {mode}")
    print(insight_text[:800] + "…" if len(insight_text) > 800 else insight_text)

    # Save insights to file
    report_path = Path("outputs/reports/ai_insights.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(insight_text, encoding="utf-8")
    print(f"\n  AI insights saved -> {report_path}")

    print(f"\n{BANNER}")
    print("  PIPELINE COMPLETE")
    print(f"{BANNER}\n")


if __name__ == "__main__":
    main()
