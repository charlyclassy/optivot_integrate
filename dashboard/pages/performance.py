"""Model Performance: baseline vs XGBoost on the held-out test period."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard import services
from dashboard import theme as ui
from evaluation import evaluate, fmt_hours, fmt_pct

HOLDOUT = "chronological_holdout"
LODO = "leave_one_depot_out"


def comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Format one row per model for display."""
    return pd.DataFrame({
        "Model": metrics["model"],
        "MAE (kW)": metrics["mae_kw"].round(2),
        "RMSE (kW)": metrics["rmse_kw"].round(2),
        "Breach recall": metrics["breach_recall"].map(fmt_pct),
        "False-alarm ratio": metrics["false_alarm_ratio"].map(fmt_pct),
        "Actual breaches": metrics["actual_breaches"].astype(int),
        "Alerts issued": metrics["alerts"].astype(int),
        "Lead time (mean / median / shortest)": [
            "N/A" if pd.isna(a) else f"{fmt_hours(a)} / {fmt_hours(b)} / {fmt_hours(c)}"
            for a, b, c in zip(metrics["lead_time_mean_h"], metrics["lead_time_median_h"], metrics["lead_time_min_h"])
        ],
    })


def _day(value: str | None) -> str:
    """'2022-04-01' -> '1 Apr 2022'."""
    try:
        return f"{pd.Timestamp(value):%d %b %Y}".lstrip("0")
    except (TypeError, ValueError):
        return "n/a"


def _improvement(metrics: pd.DataFrame, column: str) -> str:
    try:
        xgb = float(metrics.loc[metrics["model"] == "XGBoost", column].iloc[0])
        base = float(metrics.loc[metrics["model"] != "XGBoost", column].iloc[0])
        change = (xgb - base) / base
        return f"{abs(change):.0%} {'lower' if change < 0 else 'higher'} than baseline"
    except (IndexError, ZeroDivisionError, ValueError):
        return ""


def render() -> None:
    ctx = services.safe_context(title="Model Performance")
    if ctx is None:
        return
    bundle, meta = ctx.bundle, ctx.bundle.metadata
    split, evaluation = meta.get("split", {}), meta.get("evaluation", {})
    ui.page_header("Model Performance", "Baseline and XGBoost evaluated on the same held-out period",
                   ctx.organisation, [("Depot", ctx.depot["name"]), ("Data", "Real data"), ("Model", f"v{bundle.version}")])

    predictions = bundle.test_predictions
    depot_id = ctx.depot["data_id"]
    in_test = predictions is not None and (predictions["depot_id"] == depot_id).any()
    if in_test:
        depot_predictions = predictions[predictions["depot_id"] == depot_id]
        period_text = (f"Held-out test period {depot_predictions['timestamp'].min():%d %b %Y} to "
                       f"{depot_predictions['timestamp'].max():%d %b %Y}")
    else:
        # A depot the model was not trained on: every day of its data is out-of-sample.
        depot_predictions = services.get_replay(ctx.depot, days=60)
        period_text = "Day-ahead forecasts replayed on this depot's own data (the model was not trained on this depot)"
    if depot_predictions.empty:
        ui.empty_state("Not enough data to evaluate", "This depot needs more history for a held-out comparison.")
        return

    ui.section(f"{ctx.depot['name']}: baseline vs XGBoost")
    metrics = evaluate(depot_predictions, ctx.capacity_kw, ctx.alpha)
    xgb = metrics[metrics["model"] == "XGBoost"].iloc[0]
    ui.cards([
        ui.card("XGBoost MAE", f"{xgb['mae_kw']:.2f}", "kW", _improvement(metrics, "mae_kw")),
        ui.card("XGBoost RMSE", f"{xgb['rmse_kw']:.2f}", "kW", _improvement(metrics, "rmse_kw")),
        ui.card("Breach recall", fmt_pct(xgb["breach_recall"]), "",
                f"{int(xgb['tp'])} of {int(xgb['actual_breaches'])} breach days detected" if xgb["actual_breaches"] else "No actual breaches in the period"),
        ui.card("False-alarm ratio", fmt_pct(xgb["false_alarm_ratio"]), "",
                f"{int(xgb['fp'])} of {int(xgb['alerts'])} alerts were false" if xgb["alerts"] else "No alerts issued"),
        ui.card("Mean lead time", fmt_hours(xgb["lead_time_mean_h"]), "", "Forecast issue to first actual breach"),
    ])
    st.dataframe(comparison_table(metrics), hide_index=True, width="stretch")
    ui.note(f"{period_text}. {int(xgb['windows'])} day-ahead forecasts, each issued at 00:00 for the next "
            f"{evaluation.get('horizon_hours', 24)} hours; both models are scored on the same forecasts. Breach metrics use "
            f"this depot's current capacity ({ui.fmt_kw(ctx.capacity_kw)} kW) and warning threshold ({ctx.alpha:.0%}): an alert "
            "is a forecast that reaches the warning threshold. Breach recall = TP / (TP + FN). "
            "False-alarm ratio = FP / (TP + FP). N/A means there were no events to measure.", title="How these metrics are calculated")

    st.write("")
    ui.section("Actual vs predicted over the test period")
    st.plotly_chart(charts.test_period_chart(depot_predictions, ctx.capacity_kw, ctx.alpha), config=charts.CONFIG,
                    key="test_chart")

    left, right = st.columns([1.15, 1])
    with left:
        ui.section("All depots, held-out test period")
        stored = bundle.metrics
        if "evaluation" in stored.columns:
            holdout = stored[stored["evaluation"] == HOLDOUT]
            pooled = holdout[holdout["scope"] == "All depots"]
            if not pooled.empty:
                st.dataframe(comparison_table(pooled)[["Model", "MAE (kW)", "RMSE (kW)", "Breach recall",
                                                       "False-alarm ratio"]], hide_index=True, width="stretch")
            per_depot = holdout[holdout["scope"] != "All depots"].pivot_table(
                index="scope", columns="model", values="mae_kw", sort=False)
            if not per_depot.empty:
                per_depot = per_depot.rename(columns=lambda c: f"{c} MAE (kW)").round(2).reset_index(names="Depot")
                st.dataframe(per_depot, hide_index=True, width="stretch")
            ui.note("Stored with the model package at training time, using each depot's default capacity and an "
                    f"{evaluation.get('warning_alpha', 0.8):.0%} warning threshold.", title="About these figures")
        else:
            st.dataframe(stored, hide_index=True, width="stretch")
    with right:
        ui.section("Feature importance (gain)")
        importance = meta.get("feature_importance_gain") or {}
        if importance:
            st.plotly_chart(charts.importance_chart(importance), config=charts.CONFIG, key="importance_chart")
        else:
            ui.empty_state("Not available", "This model package does not include feature importance.")

    with st.expander("How the model was validated"):
        st.markdown(
            f"- **Model:** {bundle.name}, version {bundle.version}, trained {bundle.trained_at} on "
            f"{meta.get('training_data', {}).get('n_depots', 'n/a')} real depots ({meta.get('training_rows', 0):,} rows)"
        )
        st.markdown(
            f"- **Split rule:** {split.get('rule', 'n/a')}\n"
            f"- **Training:** {split.get('train_start')} to {split.get('train_end')} (end-exclusive)\n"
            f"- **Validation (early stopping only):** {split.get('validation_start')} to {split.get('validation_end')}\n"
            f"- **Test:** {split.get('test_start')} to {split.get('test_end')}\n"
            f"- **Sampling interval:** {meta.get('sampling_minutes')} minutes\n"
            f"- **Method:** {evaluation.get('method', 'n/a')}\n"
            f"- **Baseline:** {evaluation.get('baseline', 'n/a')}\n"
            f"- **Target:** {meta.get('target_description', meta.get('target'))}"
        )
    stored = bundle.metrics
    if "evaluation" in stored.columns and (stored["evaluation"] == LODO).any():
        with st.expander("Generalisation to depots the model has not seen"):
            lodo = stored[stored["evaluation"] == LODO]
            table = lodo.pivot_table(index="scope", columns="model", values="mae_kw", sort=False)
            table = table.rename(columns=lambda c: f"{c} MAE (kW)").round(2).reset_index(names="Held-out depot")
            st.dataframe(table, hide_index=True, width="stretch")
            ui.note("Leave-one-depot-out: for each row the model was retrained without that depot and then asked to "
                    "forecast it over the test period. This indicates how the model behaves on a newly uploaded depot.")
    ui.footer(bundle.version, ctx.generated_at, ctx.series.index[-1], ctx.source_label)
