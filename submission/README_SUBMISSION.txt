Retail Sales AI Analytics - Single-File Submission
--------------------------------------------------

This folder contains the consolidated, single-file submission for the IBM SkillsBuild Academic Internship (Data Analytics with AI).

The main project repository (on GitHub) employs a modular, professional architecture (separating data loading, cleaning, feature engineering, analytics, AI, forecasting, and UI across multiple directories). 

To meet the portal's requirement for a SINGLE uploaded Code File (< 10 MB), the entire application pipeline has been consolidated into:
`Retail_Sales_Analytics.py`

This standalone file:
1. Loads the dataset directly from `data/raw/sample_-_superstore.xls`
2. Cleans and processes the data
3. Performs predictive forecasting and RFM segmentation
4. Exposes the AI Insights (template/demo mode)
5. Generates a SQLite database programmatically to execute the required SQL queries on a dedicated "SQL Business Analysis" page.
6. Renders the complete Streamlit dashboard without importing any project-local modules.

How to run:
-----------
1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   streamlit run Retail_Sales_Analytics.py
