# Retail Sales & Customer Intelligence Analytics
## IBM SkillsBuild Academic Internship — Data Analytics with AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

This is an end-to-end data analytics and AI project built on the **Tableau Sample Superstore** retail dataset. It demonstrates a complete data analytics workflow: data ingestion, cleaning, exploratory data analysis (EDA), SQL-based business analysis, customer segmentation (RFM), product profitability analysis, sales forecasting, AI-assisted insight generation, and an interactive Streamlit dashboard.

The project was developed as a capstone submission for the IBM SkillsBuild Academic Internship in Data Analytics with AI.

---

## Business Problem

Retail organisations generate large volumes of transactional data but often lack the analytical infrastructure to extract actionable intelligence. Key business questions this project addresses:

- Which product categories and sub-categories are most and least profitable?
- Which customer segments and individual customers drive the most revenue?
- What is the impact of discount policies on profit margins?
- How have sales trended over time, and can we forecast near-future performance?
- Which customers are at risk of churning?

---

## Objectives

1. Build a reproducible data analytics pipeline from raw Excel data to insights.
2. Perform EDA to understand sales, profit, regional, and customer patterns.
3. Implement RFM-based customer segmentation.
4. Create a SQLite-based analytical database with meaningful business queries.
5. Build simple sales forecasting models with honest evaluation.
6. Integrate an AI-assisted insight module that interprets verified computed metrics.
7. Deliver findings through an interactive Streamlit dashboard.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Name** | Tableau Sample Superstore |
| **Source** | [Tableau Public](https://public.tableau.com/app/learn/sample-data) |
| **Format** | `.xls` Excel file — `Orders` sheet |
| **Rows** | 10,194 order-line items |
| **Columns** | 21 (dates, IDs, dimensions, metrics) |
| **Date Range** | 2023–2026 |
| **Customers** | 804 unique |
| **Products** | 1,862 unique |
| **Categories** | Furniture, Office Supplies, Technology |

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Static visualizations |
| Plotly | Interactive charts |
| Scikit-learn | ML models (Linear Regression, Random Forest) |
| SQLite | Analytical database |
| Streamlit | Interactive dashboard |
| Jupyter Notebooks | Exploratory notebooks |

---

## Project Architecture

```
Raw Excel Data
     │
     ▼
data_loader.py   ──→  Raw DataFrame (read-only)
     │
     ▼
data_cleaning.py ──→  Cleaned DataFrame (column renames, type fixes)
     │
     ▼
feature_engineering.py ──→  Processed DataFrame (time features, margin, etc.)
     │                       Saved to data/processed/orders_processed.parquet
     ├──→ analytics.py         ──→  EDA aggregations + figures
     ├──→ customer_analysis.py ──→  RFM + customer charts
     ├──→ database.py          ──→  SQLite DB + business queries
     ├──→ forecasting.py       ──→  Monthly forecast models + metrics
     └──→ ai_insights.py       ──→  Verified metrics → AI interpretation
                                     │
                                     ▼
                              Streamlit Dashboard (app/streamlit_app.py)
```

---

## Folder Structure

```
retail-sales-ai-analytics/
├── data/
│   ├── raw/                     ← Place superstore.xls here
│   └── processed/               ← Auto-generated parquet + SQLite
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_customer_analysis.ipynb
│   └── 04_sales_prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── analytics.py
│   ├── customer_analysis.py
│   ├── forecasting.py
│   ├── ai_insights.py
│   └── database.py
├── sql/
│   └── business_analysis.sql
├── app/
│   └── streamlit_app.py
├── outputs/
│   ├── figures/                 ← Auto-generated charts
│   └── reports/                 ← AI insights markdown
├── pipeline.py                  ← Full pipeline runner
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Adi-1515/retail_sales_AI_analytics.git
cd retail_sales_AI_analytics
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the **Tableau Sample Superstore** dataset:

1. Visit https://public.tableau.com/app/learn/sample-data
2. Download `Sample - Superstore.xls` (or `.xlsx`)
3. Place the file at: `data/raw/sample_-_superstore.xls`

The data loader will automatically find it. If your filename differs slightly, the loader checks several common variants.

---

## Running the Project

### Full pipeline (recommended first run)

```bash
python pipeline.py
```

This executes all steps: loading, cleaning, feature engineering, EDA, customer analysis, database creation, SQL queries, forecasting, and AI insights.

### Individual modules

```bash
python -c "from src.data_loader import load_raw_orders; df = load_raw_orders(); print(df.shape)"
python -c "from src.forecasting import run_forecasting; from src.feature_engineering import load_processed; r = run_forecasting(load_processed()); print(r['metrics_df'])"
```

---

## Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard sections:
1. **Executive Overview** — KPIs, monthly trends, category and region performance
2. **Customer Analytics** — RFM segments, top customers, revenue distribution
3. **Product & Profitability** — Sub-category profit, discount analysis
4. **Sales Prediction** — Forecast vs actual, model metrics
5. **AI Business Insights** — Verified metrics + AI interpretation

Use the sidebar filters to drill down by year, region, category, and customer segment.

---

## Jupyter Notebooks

```bash
jupyter notebook
```

Open notebooks in order:
1. `notebooks/01_data_understanding.ipynb` — dataset inspection
2. `notebooks/02_data_cleaning_eda.ipynb` — cleaning + EDA
3. `notebooks/03_customer_analysis.ipynb` — RFM segmentation
4. `notebooks/04_sales_prediction.ipynb` — forecasting models

---

## SQL Analysis

The SQLite database is built automatically by the pipeline. To run queries manually:

```bash
python -c "
from src.database import run_all_business_queries
results = run_all_business_queries()
for name, df in results.items():
    print(f'\n=== {name} ===')
    print(df.to_string(index=False))
"
```

Or open the database directly:

```bash
# Using sqlite3 CLI
sqlite3 data/processed/superstore.db
.read sql/business_analysis.sql
```

SQL queries cover:
- Sales/profit by category
- Monthly sales trend
- Regional performance
- Top 10 customers by revenue
- Top 10 products by profit
- Loss-making sub-categories
- Discount analysis by category
- Segment performance
- High-sales/low-profit products
- Year-over-year performance
- Shipping analysis

---

## AI Component

The AI insight module (`src/ai_insights.py`) follows this workflow:

```
analytics.py / customer_analysis.py / forecasting.py
  → verified numerical metrics
  → build_insight_context()         ← assembles structured dict
  → generate_insights()             ← passes context to insight generator
  → natural-language interpretation (template or LLM)
```

**Demo/Template Mode** (default): A structured template fills computed metrics into business-language sentences. This mode requires no external services, API keys, or model downloads.

**IBM Granite Mode** (optional): If `transformers` is installed and the `ibm-granite/granite-3.3-2b-instruct` model is available locally, the same context is passed as a prompt. Enable with:

```bash
USE_GRANITE=1 python pipeline.py
```

**Critical**: AI output is clearly labelled as AI-generated interpretation. All numbers in AI output originate from deterministic analytics code, not the language model.

---

## Model Methodology

### RFM Segmentation

| Dimension | Definition |
|-----------|-----------|
| Recency | Days since last order (lower = better, score 4=best) |
| Frequency | Distinct order count (higher = better) |
| Monetary | Total sales revenue (higher = better) |

Scores are computed via quartile cuts (1-4 per dimension). Combined score (3-12) maps to: Champions, Loyal Customers, Potential Loyalists, At Risk, Lost/Inactive.

### Sales Forecasting

| Model | Description |
|-------|-------------|
| Naive Baseline | Repeats last training-month value |
| Linear Regression | Fits linear trend on time index |
| Random Forest | Non-linear ensemble on time index |

Train/test split: **last 12 months** held out as test set. No random splitting of time-series data.

Feature: single numeric time index (months since dataset start).

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error — average absolute prediction error in USD |
| RMSE | Root Mean Squared Error — penalises large errors |
| R² | Coefficient of determination — proportion of variance explained |

---

## Limitations

- Dataset spans only ~4 years, limiting forecasting robustness.
- No cost-of-goods data; profit is taken as given from the dataset.
- Returns data exists in a separate sheet but is not fully integrated.
- RFM segments are relative within this dataset only.
- Template-mode AI insights are rule-based, not a trained language model.
- Forecasting models use only a linear time trend; no external variables.

---

## Responsible AI Considerations

1. **No hallucinated data**: The AI module receives only pre-computed, verified metrics. It cannot invent numbers.
2. **Transparent labelling**: All AI output is clearly labelled as AI-generated interpretation, not factual evidence.
3. **Demo mode is honest**: Template mode never claims to be a real LLM call.
4. **No paid APIs**: The project works fully offline. The optional Granite model uses a free HuggingFace-hosted model.
5. **Reproducibility**: Fixed random seeds (42) are used throughout.

---

## Deployment (Streamlit Cloud)

To deploy this application to [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push the repository to GitHub (ensure the dataset file is committed to `data/raw/`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select the repository, branch (`main`), and entry point (`app/streamlit_app.py`).
4. The dataset (`data/raw/sample_-_superstore.xls`) must be present in the repository for the deployed app to work. It is included because it is sample/demo data from Tableau Public.
5. No API keys or secrets are required for the default (template) AI mode.

**If using IBM Granite (optional)**:

- Do not commit API keys or model credentials to the repository.
- If using Streamlit secrets, add them via the Streamlit Cloud dashboard under **App > Settings > Secrets**.
- Reference secrets via `st.secrets["YOUR_KEY"]` — never hardcode them.

---

## Future Improvements

- Integrate returns data to compute true net profitability.
- Add explicit seasonality features (month-of-year dummies) to forecasting.
- Extend RFM to a formal CLV (Customer Lifetime Value) model.
- Add a SHAP-based feature importance analysis for the Random Forest model.
- Add automated report generation (PDF export from the dashboard).

---

## License

MIT License — see [LICENSE](LICENSE)
