"""
database.py
-----------
Creates a SQLite database from the processed Superstore DataFrame and
exposes helper functions for running SQL queries programmatically.

The database is stored at: data/processed/superstore.db
It is recreated fresh on each call to build_database() so that it stays
in sync with the processed data.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/processed/superstore.db")


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------

def build_database(df: pd.DataFrame, db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Persist the processed DataFrame into a SQLite database.

    Creates (or replaces) a table named `orders` containing all processed
    columns.  Date columns are stored as ISO-8601 text strings, which SQLite
    handles transparently via strftime / date() functions.

    Parameters
    ----------
    df      : processed DataFrame (from feature_engineering.load_processed)
    db_path : path for the SQLite file

    Returns
    -------
    sqlite3.Connection
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare a copy with SQLite-friendly types
    export = df.copy()
    for col in export.select_dtypes(include=["datetime64"]).columns:
        export[col] = export[col].dt.strftime("%Y-%m-%d")

    # Handle pandas nullable integer types (Int64 etc.)
    for col in export.columns:
        if hasattr(export[col], "dtype") and str(export[col].dtype).startswith("Int"):
            export[col] = export[col].astype("float64")  # nullable int -> float OK for SQLite

    conn = sqlite3.connect(str(db_path))
    export.to_sql("orders", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_order_id ON orders (Order_ID)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON orders (Customer_ID)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON orders (Order_Date)")
    conn.commit()

    row_count = pd.read_sql("SELECT COUNT(*) AS n FROM orders", conn).iloc[0, 0]
    print(f"[database] SQLite database built at: {db_path}  ({row_count:,} rows)")
    return conn


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the existing database (must be built first)."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}.  Call build_database() first."
        )
    return sqlite3.connect(str(db_path))


def run_query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.

    Parameters
    ----------
    sql     : SQL query string
    db_path : path to the SQLite database

    Returns
    -------
    pd.DataFrame
    """
    conn = get_connection(db_path)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Business queries (callable individually or run all at once)
# ---------------------------------------------------------------------------

BUSINESS_QUERIES = {
    "total_sales_profit_by_category": """
        SELECT
            Category,
            ROUND(SUM(Sales), 2)         AS Total_Sales,
            ROUND(SUM(Profit), 2)        AS Total_Profit,
            COUNT(DISTINCT Order_ID)     AS Order_Count,
            ROUND(SUM(Profit)/SUM(Sales)*100, 2) AS Profit_Margin_Pct
        FROM orders
        GROUP BY Category
        ORDER BY Total_Sales DESC;
    """,

    "monthly_sales": """
        SELECT
            strftime('%Y-%m', Order_Date) AS Year_Month,
            ROUND(SUM(Sales), 2)          AS Monthly_Sales,
            ROUND(SUM(Profit), 2)         AS Monthly_Profit
        FROM orders
        GROUP BY Year_Month
        ORDER BY Year_Month;
    """,

    "regional_performance": """
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
    """,

    "top_10_customers_by_revenue": """
        SELECT
            Customer_ID,
            Customer_Name,
            Segment,
            ROUND(SUM(Sales), 2)  AS Total_Sales,
            ROUND(SUM(Profit), 2) AS Total_Profit,
            COUNT(DISTINCT Order_ID) AS Orders
        FROM orders
        GROUP BY Customer_ID, Customer_Name, Segment
        ORDER BY Total_Sales DESC
        LIMIT 10;
    """,

    "top_10_products_by_profit": """
        SELECT
            Product_ID,
            Product_Name,
            Category,
            Sub_Category,
            ROUND(SUM(Sales), 2)  AS Total_Sales,
            ROUND(SUM(Profit), 2) AS Total_Profit,
            ROUND(SUM(Profit)/SUM(Sales)*100, 2) AS Profit_Margin_Pct
        FROM orders
        GROUP BY Product_ID, Product_Name, Category, Sub_Category
        ORDER BY Total_Profit DESC
        LIMIT 10;
    """,

    "loss_making_subcategories": """
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
    """,

    "avg_discount_by_category": """
        SELECT
            Category,
            ROUND(AVG(Discount) * 100, 2)  AS Avg_Discount_Pct,
            ROUND(SUM(Sales), 2)           AS Total_Sales,
            ROUND(SUM(Profit), 2)          AS Total_Profit
        FROM orders
        GROUP BY Category
        ORDER BY Avg_Discount_Pct DESC;
    """,

    "sales_profit_by_segment": """
        SELECT
            Segment,
            ROUND(SUM(Sales), 2)         AS Total_Sales,
            ROUND(SUM(Profit), 2)        AS Total_Profit,
            COUNT(DISTINCT Customer_ID)  AS Customers,
            COUNT(DISTINCT Order_ID)     AS Orders,
            ROUND(SUM(Profit)/SUM(Sales)*100, 2) AS Profit_Margin_Pct
        FROM orders
        GROUP BY Segment
        ORDER BY Total_Sales DESC;
    """,

    "high_sales_low_profit_products": """
        SELECT
            Product_Name,
            Category,
            Sub_Category,
            ROUND(SUM(Sales), 2)   AS Total_Sales,
            ROUND(SUM(Profit), 2)  AS Total_Profit,
            ROUND(SUM(Profit)/SUM(Sales)*100, 2) AS Profit_Margin_Pct
        FROM orders
        GROUP BY Product_Name, Category, Sub_Category
        HAVING Total_Sales > 1000 AND Profit_Margin_Pct < 5
        ORDER BY Total_Sales DESC
        LIMIT 20;
    """,

    "yearly_performance": """
        SELECT
            strftime('%Y', Order_Date) AS Year,
            ROUND(SUM(Sales), 2)  AS Total_Sales,
            ROUND(SUM(Profit), 2) AS Total_Profit,
            COUNT(DISTINCT Order_ID)    AS Orders,
            COUNT(DISTINCT Customer_ID) AS Customers
        FROM orders
        GROUP BY Year
        ORDER BY Year;
    """,

    "ship_mode_analysis": """
        SELECT
            Ship_Mode,
            COUNT(DISTINCT Order_ID)  AS Orders,
            ROUND(SUM(Sales), 2)      AS Total_Sales,
            ROUND(AVG(Days_to_Ship), 1) AS Avg_Days_to_Ship
        FROM orders
        GROUP BY Ship_Mode
        ORDER BY Orders DESC;
    """,
}


def run_all_business_queries(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    """
    Execute all predefined business queries and return results as a dict.

    Keys are query names; values are DataFrames.
    """
    results = {}
    for name, sql in BUSINESS_QUERIES.items():
        try:
            results[name] = run_query(sql, db_path)
            print(f"[database] Query '{name}' -> {len(results[name])} rows")
        except Exception as exc:
            print(f"[database] Query '{name}' FAILED: {exc}")
    return results
