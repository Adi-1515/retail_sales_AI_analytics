"""
feature_engineering.py
-----------------------
Creates new analytical features from the cleaned Superstore orders DataFrame.

All derived columns are documented below.  The raw/cleaned columns are never
overwritten — only additive columns are created.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and business-metric features to the cleaned DataFrame.

    New columns
    -----------
    Year            : int   – calendar year of Order_Date
    Month           : int   – month number (1-12) of Order_Date
    Quarter         : int   – quarter (1-4) of Order_Date
    Month_Name      : str   – abbreviated month name ('Jan', 'Feb', …)
    Year_Month      : str   – 'YYYY-MM' period string (for time-series plotting)
    Year_Quarter    : str   – 'YYYY-Q#' string
    Days_to_Ship    : int   – Ship_Date - Order_Date in calendar days
    Profit_Margin   : float – Profit / Sales (NaN where Sales == 0)
    Sales_per_Unit  : float – Sales / Quantity
    Profit_per_Unit : float – Profit / Quantity
    Is_Profitable   : bool  – True where Profit > 0

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned orders from data_cleaning.clean_orders().

    Returns
    -------
    pd.DataFrame
        DataFrame with all original columns plus new feature columns.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Time-based features
    # ------------------------------------------------------------------
    df["Year"] = df["Order_Date"].dt.year.astype(int)
    df["Month"] = df["Order_Date"].dt.month.astype(int)
    df["Quarter"] = df["Order_Date"].dt.quarter.astype(int)
    df["Month_Name"] = df["Order_Date"].dt.strftime("%b")
    df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
    df["Year_Quarter"] = (
        df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)
    )

    # Days to ship (may reveal logistics patterns)
    if "Ship_Date" in df.columns:
        df["Days_to_Ship"] = (df["Ship_Date"] - df["Order_Date"]).dt.days.astype("Int64")

    # ------------------------------------------------------------------
    # Business-metric features
    # ------------------------------------------------------------------
    # Profit margin (unit: fraction, not percent).
    # Guard against division by zero — Sales=0 rows get NaN margin.
    df["Profit_Margin"] = np.where(
        df["Sales"] != 0,
        df["Profit"] / df["Sales"],
        np.nan,
    )

    df["Sales_per_Unit"] = df["Sales"] / df["Quantity"]
    df["Profit_per_Unit"] = df["Profit"] / df["Quantity"]
    df["Is_Profitable"] = df["Profit"] > 0

    print(
        f"[feature_engineering] Added features: Year, Month, Quarter, "
        "Month_Name, Year_Month, Year_Quarter, Days_to_Ship, "
        "Profit_Margin, Sales_per_Unit, Profit_per_Unit, Is_Profitable."
    )
    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed/orders_processed.parquet") -> None:
    """
    Persist the processed DataFrame to a Parquet file for downstream use.

    We use Parquet (not CSV) so that dtypes — especially dates and
    nullable integers — are preserved exactly across load/save cycles.

    Parameters
    ----------
    df   : processed DataFrame
    path : output file path (relative or absolute)
    """
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[feature_engineering] Processed dataset saved -> {out}  ({len(df):,} rows)")


def load_processed(path: str = "data/processed/orders_processed.parquet") -> pd.DataFrame:
    """Load the processed Parquet file."""
    from pathlib import Path
    import sys

    p = Path(path)
    if not p.exists():
        print("[feature_engineering] Processed file not found — running full pipeline...")
        # Lazy import to avoid circular dependencies at module level
        from src.data_loader import load_raw_orders
        from src.data_cleaning import clean_orders

        df_raw = load_raw_orders()
        df_clean = clean_orders(df_raw)
        df = engineer_features(df_clean)
        save_processed(df, path)
        return df

    df = pd.read_parquet(p)
    print(f"[feature_engineering] Loaded processed dataset: {len(df):,} rows × {len(df.columns)} cols.")
    return df
