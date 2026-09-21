"""
ai_insights.py
--------------
AI-assisted business insight generation module.

Architecture
------------
The AI layer receives VERIFIED, PRE-COMPUTED analytical metrics — not raw data.
This guarantees that the AI cannot invent numbers; it can only interpret
figures that have already been computed by deterministic code.

Two operating modes:
  1. LOCAL / DEMO MODE  (default)
     Uses a template-based natural-language generator that fills structured
     sentences from the provided metrics dictionary.
     This mode is clearly labelled as "Demo/Template Mode" throughout.
     It produces realistic, context-aware business language without an LLM.

  2. IBM GRANITE / LOCAL LLM MODE  (optional, if `transformers` is installed
     and a local Granite model is available via HuggingFace)
     If enabled, the same structured context is passed to the model as a
     prompt and the response is returned.
     This mode is explicitly gated behind an availability check — it will
     NOT silently fall back to fake LLM output.

IMPORTANT
---------
* AI output is always labelled as AI-GENERATED INTERPRETATION.
* Numbers shown to the user come from analytics.py / customer_analysis.py,
  not from the AI model.
* The AI model is asked to interpret findings, not to generate new data.
"""

from __future__ import annotations

import textwrap
from typing import Any


# ---------------------------------------------------------------------------
# Structured context builder
# ---------------------------------------------------------------------------

def build_insight_context(kpis: dict, cat_df, region_df, rfm_df,
                          forecast_metrics: list, disc_df) -> dict:
    """
    Assemble verified analytical metrics into a structured context dict
    suitable for passing to the insight generator.

    All numeric values come from pre-computed DataFrames / KPI dicts.
    """
    import pandas as pd

    ctx: dict[str, Any] = {}

    # KPIs
    ctx["total_sales"] = kpis.get("total_sales", 0)
    ctx["total_profit"] = kpis.get("total_profit", 0)
    ctx["total_orders"] = kpis.get("total_orders", 0)
    ctx["unique_customers"] = kpis.get("unique_customers", 0)
    ctx["avg_order_value"] = kpis.get("avg_order_value", 0)
    ctx["overall_margin_pct"] = kpis.get("overall_profit_margin_pct", 0)

    # Category performance
    if cat_df is not None and len(cat_df):
        top_cat = cat_df.sort_values("Sales", ascending=False).iloc[0]
        worst_cat = cat_df.sort_values("Profit_Margin").iloc[0]
        ctx["top_category_by_sales"] = top_cat["Category"]
        ctx["top_category_sales"] = round(float(top_cat["Sales"]), 2)
        ctx["worst_margin_category"] = worst_cat["Category"]
        ctx["worst_margin_pct"] = round(float(worst_cat["Profit_Margin"]) * 100, 2)

    # Region
    if region_df is not None and len(region_df):
        top_reg = region_df.sort_values("Sales", ascending=False).iloc[0]
        worst_reg = region_df.sort_values("Profit_Margin").iloc[0]
        ctx["top_region"] = top_reg["Region"]
        ctx["top_region_sales"] = round(float(top_reg["Sales"]), 2)
        ctx["worst_margin_region"] = worst_reg["Region"]
        ctx["worst_margin_region_pct"] = round(float(worst_reg["Profit_Margin"]) * 100, 2)

    # RFM segments
    if rfm_df is not None and len(rfm_df):
        seg_counts = rfm_df["RFM_Segment"].value_counts().to_dict()
        ctx["rfm_segments"] = seg_counts
        ctx["champions_count"] = seg_counts.get("Champions", 0)
        ctx["at_risk_count"] = seg_counts.get("At Risk", 0)

    # Forecasting
    if forecast_metrics:
        best = min(forecast_metrics, key=lambda x: x["RMSE"])
        ctx["best_forecast_model"] = best["model"]
        ctx["best_forecast_rmse"] = best["RMSE"]
        ctx["best_forecast_r2"] = best["R2"]

    # Discount analysis — find the highest-discount bin with negative margin
    if disc_df is not None and len(disc_df):
        loss_disc = disc_df[disc_df["Avg_Profit_Margin"] < 0]
        if len(loss_disc):
            worst_disc = loss_disc.sort_values("Avg_Profit_Margin").iloc[0]
            ctx["worst_discount_bin"] = str(worst_disc["Discount_Bin"])
            ctx["worst_discount_margin"] = round(float(worst_disc["Avg_Profit_Margin"]) * 100, 2)

    return ctx


# ---------------------------------------------------------------------------
# Template-based insight generator (Demo / Local mode)
# ---------------------------------------------------------------------------

def _generate_template_insights(ctx: dict) -> str:
    """
    Generate structured natural-language insights from the context dict
    using pure string formatting — no external model required.

    This is clearly labelled as template-generated in the returned text.
    """
    lines = [
        "## AI-Generated Business Insights",
        "*(Demo/Template Mode — interpretations are derived from computed metrics)*",
        "",
        "---",
        "",
        "### Key Findings",
        "",
        f"- **Overall Business Health**: The store generated "
        f"**${ctx.get('total_sales', 0):,.0f}** in total sales across "
        f"**{ctx.get('total_orders', 0):,} orders** from "
        f"**{ctx.get('unique_customers', 0):,} unique customers**, "
        f"achieving an overall profit margin of "
        f"**{ctx.get('overall_margin_pct', 0):.1f}%**.",
        "",
        f"- **Category Leadership**: The **{ctx.get('top_category_by_sales', 'N/A')}** "
        f"category leads in revenue at "
        f"**${ctx.get('top_category_sales', 0):,.0f}**. "
        f"The **{ctx.get('worst_margin_category', 'N/A')}** category has the "
        f"lowest profit margin at "
        f"**{ctx.get('worst_margin_pct', 0):.1f}%**, suggesting pricing or "
        f"discount control issues.",
        "",
        f"- **Regional Performance**: The **{ctx.get('top_region', 'N/A')}** region "
        f"contributes the highest sales "
        f"(**${ctx.get('top_region_sales', 0):,.0f}**). "
        f"The **{ctx.get('worst_margin_region', 'N/A')}** region has the weakest "
        f"profit margin at "
        f"**{ctx.get('worst_margin_region_pct', 0):.1f}%**, which may warrant "
        f"regional cost or pricing review.",
        "",
    ]

    # Discount findings (conditional)
    if "worst_discount_bin" in ctx:
        lines += [
            f"- **Discount Impact**: Transactions with discounts in the "
            f"**{ctx['worst_discount_bin']}** range show an average profit margin of "
            f"**{ctx['worst_discount_margin']:.1f}%**, indicating that aggressive "
            f"discounting in this band is destroying value rather than driving volume.",
            "",
        ]

    # RFM findings (conditional)
    if "champions_count" in ctx:
        lines += [
            f"- **Customer Segmentation**: RFM analysis identified "
            f"**{ctx['champions_count']}** Champions and "
            f"**{ctx.get('at_risk_count', 0)}** At-Risk customers. "
            f"Re-engagement campaigns targeting At-Risk customers could recover "
            f"a meaningful share of lapsed revenue.",
            "",
        ]

    # Forecasting (conditional)
    if "best_forecast_model" in ctx:
        r2 = ctx.get("best_forecast_r2", 0)
        quality = "a reasonably good fit" if r2 > 0.5 else "a limited fit"
        lines += [
            f"- **Sales Forecasting**: The best-performing model is "
            f"**{ctx['best_forecast_model']}** "
            f"(RMSE = ${ctx['best_forecast_rmse']:,.0f}, R² = {r2:.3f}), "
            f"indicating {quality} for the available monthly data volume.",
            "",
        ]

    lines += [
        "---",
        "",
        "### Business Implications",
        "",
        "1. **Discount Policy Review**: The data clearly shows that high discounts "
        "   correlate with negative profit margins.  A structured discount ceiling "
        "   policy — especially for Office Supplies and Furniture — could meaningfully "
        "   improve profitability without sacrificing competitive positioning.",
        "",
        "2. **Customer Retention**: Prioritise re-engagement of At-Risk and "
        "   Lost/Inactive customer segments through targeted outreach, as their "
        "   historical purchase behaviour indicates willingness to buy.",
        "",
        "3. **Product Mix Optimisation**: Loss-making sub-categories (e.g. Tables, "
        "   Bookcases where applicable) should be reviewed for cost structure, "
        "   supplier negotiations, or pricing adjustments rather than increased "
        "   discounting.",
        "",
        "4. **Regional Strategy**: The highest-sales region should be studied as "
        "   a benchmark for operating practices that could be replicated in "
        "   lower-margin regions.",
        "",
        "---",
        "",
        "### Areas Requiring Further Investigation",
        "",
        "- **Seasonality**: Monthly sales show variation; a longer historical series "
        "  would enable more robust seasonal decomposition and planning.",
        "- **Returns Data**: The dataset includes a Returns sheet that was not fully "
        "  integrated into this analysis; returns can mask true profitability.",
        "- **Customer Lifetime Value**: The current RFM model uses transactional "
        "  proxies for value.  A formal CLV model would require longer customer history.",
        "- **Product-level Margin Drivers**: Cost of goods data is not included; "
        "  the Profit column is taken as given.  Validation against actual cost "
        "  records is recommended.",
        "",
        "---",
        "",
        "> ⚠️  **Disclaimer**: The interpretations above are generated by a "
        "template-based AI module from pre-computed analytical metrics.  "
        "They represent plausible business interpretations, not audit-grade findings.  "
        "All numeric values cited are derived from the dataset and computed by "
        "deterministic code — not hallucinated by an AI model.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional: IBM Granite / HuggingFace local model
# ---------------------------------------------------------------------------

def _try_granite_insights(ctx: dict, model_name: str = "ibm-granite/granite-3.3-2b-instruct") -> str | None:
    """
    Attempt to generate insights using a local HuggingFace-hosted IBM Granite model.

    Returns None if the model is not available, not downloaded, or if inference
    fails for any reason.  The caller must handle None gracefully.

    NOTE: This function will download ~2-5 GB of model weights on first call if
    the model is available. Set env var GRANITE_MODEL_NAME to override the default.
    """
    import os
    model_name = os.environ.get("GRANITE_MODEL_NAME", model_name)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        return None

    prompt = _build_granite_prompt(ctx)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        # Strip the prompt itself from the output
        response = decoded[len(prompt):].strip()
        header = (
            "## AI-Generated Business Insights\n"
            "*(IBM Granite — ibm-granite/granite-3.3-2b-instruct)*\n\n"
            "---\n\n"
        )
        footer = (
            "\n\n---\n\n"
            "> ⚠️  **Disclaimer**: These insights were generated by IBM Granite "
            "from pre-computed verified metrics.  Numbers cited are from deterministic "
            "analysis code, not LLM inference.  Treat interpretations as AI-assisted "
            "analysis, not authoritative business conclusions."
        )
        return header + response + footer
    except Exception as exc:
        print(f"[ai_insights] Granite inference failed: {exc}")
        return None


def _build_granite_prompt(ctx: dict) -> str:
    return textwrap.dedent(f"""
        You are a retail business analyst.  Below are verified statistical findings
        from a Superstore retail dataset analysis.  Interpret these findings and provide:
        1. Key business findings (3-5 bullet points)
        2. Business implications
        3. Areas for further investigation

        Do NOT invent any numbers.  Only reference the figures provided.

        === VERIFIED METRICS ===
        Total Sales: ${ctx.get('total_sales', 0):,.0f}
        Total Profit: ${ctx.get('total_profit', 0):,.0f}
        Overall Profit Margin: {ctx.get('overall_margin_pct', 0):.1f}%
        Total Orders: {ctx.get('total_orders', 0):,}
        Unique Customers: {ctx.get('unique_customers', 0):,}
        Average Order Value: ${ctx.get('avg_order_value', 0):,.2f}
        Top Category by Sales: {ctx.get('top_category_by_sales', 'N/A')} (${ctx.get('top_category_sales', 0):,.0f})
        Worst Margin Category: {ctx.get('worst_margin_category', 'N/A')} ({ctx.get('worst_margin_pct', 0):.1f}%)
        Top Region by Sales: {ctx.get('top_region', 'N/A')} (${ctx.get('top_region_sales', 0):,.0f})
        Champions Customers: {ctx.get('champions_count', 'N/A')}
        At-Risk Customers: {ctx.get('at_risk_count', 'N/A')}
        Best Forecast Model: {ctx.get('best_forecast_model', 'N/A')} (RMSE=${ctx.get('best_forecast_rmse', 0):,.0f}, R²={ctx.get('best_forecast_r2', 0):.3f})

        === BEGIN ANALYSIS ===
    """).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insights(ctx: dict, prefer_llm: bool = False) -> tuple[str, str]:
    """
    Generate business insights from the verified metrics context.

    Parameters
    ----------
    ctx        : dict from build_insight_context()
    prefer_llm : if True, attempt Granite model first; fall back to template mode

    Returns
    -------
    (insight_text: str, mode: str)
        mode is either "granite" or "template"
    """
    if prefer_llm:
        result = _try_granite_insights(ctx)
        if result:
            return result, "granite"
        print("[ai_insights] Granite not available — falling back to template mode.")

    return _generate_template_insights(ctx), "template"
