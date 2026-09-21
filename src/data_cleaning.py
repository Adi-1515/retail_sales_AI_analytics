"""
data_cleaning.py
----------------
Cleans the raw Superstore orders DataFrame.

Design principles
-----------------
* The raw DataFrame is NEVER modified in-place.
* Every decision is documented in docstrings and inline comments.
* Returns a new, fully cleaned DataFrame ready for feature engineering.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Column name normalisation map
# Superstore column names differ slightly across Tableau versions.
# We map whatever the file has to our internal canonical names.
# ---------------------------------------------------------------------------
_COLUMN_RENAMES = {
    # Variations seen in different Tableau exports
    "Country/Region": "Country",
    "State/Province": "State",
    "Sub-Category": "Sub_Category",
    "Order Date": "Order_Date",
    "Ship Date": "Ship_Date",
    "Ship Mode": "Ship_Mode",
    "Order ID": "Order_ID",
    "Customer ID": "Customer_ID",
    "Customer Name": "Customer_Name",
    "Product ID": "Product_ID",
    "Product Name": "Product_Name",
    "Row ID": "Row_ID",
    "Postal Code": "Postal_Code",
    # already canonical — no-ops included for clarity
    "Region": "Region",
    "Segment": "Segment",
    "Category": "Category",
    "Sales": "Sales",
    "Quantity": "Quantity",
    "Discount": "Discount",
    "Profit": "Profit",
}


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw orders DataFrame.

    Steps
    -----
    1. Copy to avoid mutating original.
    2. Rename columns to canonical names.
    3. Parse date columns.
    4. Strip whitespace from string columns.
    5. Ensure numeric columns have correct types.
    6. Handle missing values (none expected, but guarded).
    7. Remove confirmed duplicate rows.
    8. Validate that key business fields are non-negative / in range.

    Parameters
    ----------
    df : pd.DataFrame
        Raw orders from data_loader.load_raw_orders().

    Returns
    -------
    pd.DataFrame
        Cleaned, typed DataFrame.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Rename columns to canonical names
    # ------------------------------------------------------------------
    rename_map = {c: _COLUMN_RENAMES[c] for c in df.columns if c in _COLUMN_RENAMES}
    df.rename(columns=rename_map, inplace=True)

    # ------------------------------------------------------------------
    # 2. Parse date columns (already datetime in newer pandas+xlrd, but
    #    guard against object dtype from older parsers)
    # ------------------------------------------------------------------
    for date_col in ("Order_Date", "Ship_Date"):
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], infer_format=True, errors="coerce")

    # ------------------------------------------------------------------
    # 3. Strip leading/trailing whitespace from all string columns
    # ------------------------------------------------------------------
    str_cols = df.select_dtypes(include="object").columns.tolist()
    # Also handle pandas StringDtype (ArrowStringArray) columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            if col not in ("Order_Date", "Ship_Date"):
                df[col] = df[col].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 4. Ensure numeric types
    # ------------------------------------------------------------------
    for num_col in ("Sales", "Quantity", "Discount", "Profit"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # ------------------------------------------------------------------
    # 5. Handle missing values
    #    The Superstore dataset has no missing values, but we protect
    #    against edge cases in user-supplied versions.
    # ------------------------------------------------------------------
    before = len(df)
    # Drop rows where core business fields are NaN (unrecoverable)
    core_cols = [c for c in ("Sales", "Quantity", "Profit", "Order_Date") if c in df.columns]
    df.dropna(subset=core_cols, inplace=True)
    dropped_na = before - len(df)
    if dropped_na:
        print(f"[data_cleaning] Dropped {dropped_na} rows with missing core values.")

    # ------------------------------------------------------------------
    # 6. Remove fully duplicated rows (same Row_ID or full row match)
    # ------------------------------------------------------------------
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped_dupes = before - len(df)
    if dropped_dupes:
        print(f"[data_cleaning] Dropped {dropped_dupes} fully duplicate rows.")

    # ------------------------------------------------------------------
    # 7. Business-rule validation / flagging
    #    - Discount must be in [0, 1].  Values outside this range indicate
    #      data entry error; we clip rather than drop to preserve the row.
    #    - Quantity must be >= 1.
    #    - Sales must be >= 0.
    # ------------------------------------------------------------------
    if "Discount" in df.columns:
        bad_disc = ((df["Discount"] < 0) | (df["Discount"] > 1)).sum()
        if bad_disc:
            print(f"[data_cleaning] Clipping {bad_disc} out-of-range Discount values to [0,1].")
            df["Discount"] = df["Discount"].clip(0, 1)

    if "Quantity" in df.columns:
        bad_qty = (df["Quantity"] < 1).sum()
        if bad_qty:
            print(f"[data_cleaning] {bad_qty} rows with Quantity < 1 detected (keeping).")

    if "Sales" in df.columns:
        bad_sales = (df["Sales"] < 0).sum()
        if bad_sales:
            print(f"[data_cleaning] {bad_sales} rows with negative Sales detected (keeping as-is).")

    # ------------------------------------------------------------------
    # 8. Reset index
    # ------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    print(f"[data_cleaning] Clean dataset: {len(df):,} rows × {len(df.columns)} columns.")
    return df
