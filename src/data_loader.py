"""
data_loader.py
--------------
Responsible for locating and loading the Superstore dataset from disk.
Supports .xls, .xlsx automatically.  The dataset is never modified here.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Candidate locations, checked in order.
_CANDIDATE_PATHS = [
    Path("data/raw/sample_-_superstore.xls"),
    Path("data/raw/sample_-_superstore.xlsx"),
    Path("data/superstore.xls"),
    Path("data/superstore.xlsx"),
    Path("data/raw/superstore.xls"),
    Path("data/raw/superstore.xlsx"),
]

_ORDERS_SHEET = "Orders"


def _resolve_dataset_path() -> Path:
    """Return the first existing candidate path for the Superstore dataset."""
    for p in _CANDIDATE_PATHS:
        if p.exists():
            return p
    searched = "\n  ".join(str(p) for p in _CANDIDATE_PATHS)
    raise FileNotFoundError(
        f"Superstore dataset not found.  Searched:\n  {searched}\n"
        "Place the file at data/raw/sample_-_superstore.xls (or .xlsx)."
    )


def load_raw_orders(dataset_path: Path | None = None) -> pd.DataFrame:
    """
    Load the 'Orders' sheet of the Superstore Excel file.

    Parameters
    ----------
    dataset_path : Path or None
        Explicit path to the .xls/.xlsx file.  If None, the function
        auto-discovers the file from the candidate list.

    Returns
    -------
    pd.DataFrame
        Raw orders DataFrame — no transformations applied.
    """
    path = dataset_path or _resolve_dataset_path()
    print(f"[data_loader] Loading dataset from: {path}")
    df = pd.read_excel(path, sheet_name=_ORDERS_SHEET)
    print(f"[data_loader] Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df


def load_returns(dataset_path: Path | None = None) -> pd.DataFrame:
    """Load the 'Returns' sheet (if present)."""
    path = dataset_path or _resolve_dataset_path()
    try:
        df = pd.read_excel(path, sheet_name="Returns")
        print(f"[data_loader] Returns sheet: {len(df):,} rows.")
        return df
    except Exception:
        print("[data_loader] Returns sheet not available; returning empty DataFrame.")
        return pd.DataFrame(columns=["Returned", "Order ID"])


def generate_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a basic data-quality summary for the given DataFrame.

    Returns
    -------
    dict with keys:
        shape, column_dtypes, missing_counts, missing_pct,
        duplicate_rows, numeric_summary
    """
    missing_counts = df.isnull().sum()
    report = {
        "shape": df.shape,
        "column_dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": missing_counts.to_dict(),
        "missing_pct": (missing_counts / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df.select_dtypes(include="number").describe().to_dict(),
    }
    return report


def print_data_quality_report(report: dict) -> None:
    """Pretty-print the data-quality report to stdout."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Shape        : {report['shape'][0]:,} rows × {report['shape'][1]} columns")
    print(f"Duplicates   : {report['duplicate_rows']}")
    print("\nColumn Types:")
    for col, dtype in report["column_dtypes"].items():
        missing = report["missing_pct"].get(col, 0)
        flag = f"  ← {missing:.1f}% missing" if missing > 0 else ""
        print(f"  {col:<30} {dtype}{flag}")
    print("=" * 60)
