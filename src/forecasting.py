"""
forecasting.py
--------------
Simple sales forecasting for the Superstore dataset.

Methodology
-----------
1. Aggregate daily order-level data to monthly totals.
2. Create a numeric time index (months since dataset start) as the sole
   feature.  This is intentionally minimal — the dataset spans only ~4 years
   and a complex feature set would overfit.
3. Time-based train/test split: the last 12 months form the test set.
   Random splitting of time-series data is methodologically invalid and is
   explicitly NOT used here.
4. Models evaluated:
     a. Naive Baseline  – repeats the last training-month value for every
                          test month.  This is a zero-parameter benchmark.
     b. Linear Regression – captures a linear growth/decline trend.
     c. Random Forest Regressor – captures non-linear seasonal patterns
                                   if present.
5. Metrics: MAE, RMSE, R² (computed on test set).

Limitations (explicit, honest)
--------------------------------
* With only ~48 monthly data points a test set of 12 is reasonable but
  leaves the model with limited training data.
* No explicit seasonality features are added (month-of-year dummies would
  help but risk overfitting given the short history).
* The dataset covers a single retail store chain; results are not
  generalisable to other contexts without retraining.
* Forecasts beyond the observed date range should not be treated as reliable
  business projections.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def _save(fig, filename: str) -> Path:
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the processed orders DataFrame to monthly Sales totals.

    Returns
    -------
    pd.DataFrame with columns: Year_Month (str), Sales (float), time_idx (int)
    """
    monthly = (
        df.groupby("Year_Month")["Sales"]
        .sum()
        .reset_index()
    )
    # Sort chronologically
    monthly["_period"] = pd.PeriodIndex(monthly["Year_Month"], freq="M")
    monthly = monthly.sort_values("_period").reset_index(drop=True)
    monthly["Year_Month"] = monthly["_period"].astype(str)
    monthly.drop(columns=["_period"], inplace=True)

    # Numeric time index (0, 1, 2, …) — sole feature for trend models
    monthly["time_idx"] = np.arange(len(monthly))
    return monthly


def train_test_split_time(monthly: pd.DataFrame, test_months: int = 12):
    """
    Split a monthly time series into train and test sets chronologically.

    Parameters
    ----------
    monthly    : output of prepare_monthly_series()
    test_months: number of most-recent months held out for testing

    Returns
    -------
    train, test DataFrames
    """
    split = len(monthly) - test_months
    if split < 6:
        raise ValueError(
            f"Dataset too short to split: only {len(monthly)} months available."
        )
    return monthly.iloc[:split].copy(), monthly.iloc[split:].copy()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _fit_predict(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict(X_test)


def naive_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Repeat the last training value for all test periods."""
    return np.full(n_test, y_train[-1])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }


def run_forecasting(df: pd.DataFrame) -> dict:
    """
    Run the full forecasting pipeline.

    Returns
    -------
    dict with keys:
        monthly       : full monthly DataFrame (with all predictions)
        train         : training DataFrame
        test          : test DataFrame with prediction columns
        metrics       : list of dicts (one per model)
        best_model    : name of the model with lowest RMSE
    """
    print("[forecasting] Preparing monthly sales series...")
    monthly = prepare_monthly_series(df)
    train, test = train_test_split_time(monthly, test_months=12)

    X_train = train[["time_idx"]].values
    y_train = train["Sales"].values
    X_test = test[["time_idx"]].values
    y_test = test["Sales"].values

    # --- Naive baseline -------------------------------------------------------
    pred_naive = naive_baseline(y_train, len(test))

    # --- Linear Regression ----------------------------------------------------
    lr = LinearRegression()
    pred_lr = _fit_predict(lr, X_train, y_train, X_test)

    # --- Random Forest --------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE
    )
    pred_rf = _fit_predict(rf, X_train, y_train, X_test)

    # --- Metrics --------------------------------------------------------------
    metrics = [
        evaluate(y_test, pred_naive, "Naive Baseline"),
        evaluate(y_test, pred_lr, "Linear Regression"),
        evaluate(y_test, pred_rf, "Random Forest"),
    ]
    metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
    best_model = metrics_df.iloc[0]["model"]

    # --- Attach predictions to test set ---------------------------------------
    test = test.copy()
    test["Pred_Naive"] = pred_naive
    test["Pred_LR"] = pred_lr
    test["Pred_RF"] = pred_rf

    print("[forecasting] Model evaluation complete.")
    for m in metrics:
        print(f"  {m['model']:<22} MAE={m['MAE']:>10,.2f}  RMSE={m['RMSE']:>10,.2f}  R²={m['R2']:>6.4f}")
    print(f"  -> Best model (lowest RMSE): {best_model}")

    # --- Figure ---------------------------------------------------------------
    plot_forecast(train, test)

    return {
        "monthly": monthly,
        "train": train,
        "test": test,
        "metrics": metrics,
        "metrics_df": metrics_df,
        "best_model": best_model,
        "lr_model": lr,
        "rf_model": rf,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_forecast(train: pd.DataFrame, test: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train["Year_Month"], train["Sales"] / 1e3, label="Training Sales", color="#2563eb",
            linewidth=1.8)
    ax.plot(test["Year_Month"], test["Sales"] / 1e3, label="Actual Sales (test)", color="#2563eb",
            linewidth=1.8, linestyle="--")
    ax.plot(test["Year_Month"], test["Pred_LR"] / 1e3, label="Linear Regression",
            color="#f97316", linewidth=1.5)
    ax.plot(test["Year_Month"], test["Pred_RF"] / 1e3, label="Random Forest",
            color="#16a34a", linewidth=1.5)
    ax.plot(test["Year_Month"], test["Pred_Naive"] / 1e3, label="Naive Baseline",
            color="#9ca3af", linewidth=1.2, linestyle=":")

    n = len(train) + len(test)
    step = max(1, n // 12)
    all_months = list(train["Year_Month"]) + list(test["Year_Month"])
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(all_months[::step], rotation=45, ha="right", fontsize=8)

    ax.axvline(x=len(train) - 1, color="gray", linestyle=":", linewidth=1, label="Train/Test split")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales (USD thousands)")
    ax.set_title("Monthly Sales: Actual vs Predicted")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.0f}k"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, "sales_forecast.png")
