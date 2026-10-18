"""History: previous forecasts, actual outcomes, errors, alerts, scenarios and models."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
import config
import model_loader
from dashboard import services
from dashboard import database as storage
from dashboard import theme as ui
from evaluation import mae, rmse, window_events
from dashboard.pages import shared as common


def _forecast_outcomes(ctx) -> None:
    days = st.segmented_control("Period", [7, 14, 30, 60], default=config.REPLAY_DAYS, format_func=lambda d: f"{d} days",
                                key="history_days") or config.REPLAY_DAYS
    replay = services.get_replay(ctx.depot, days=days)
    if replay.empty:
        ui.empty_state("No previous forecasts", "This depot does not have enough history yet.")
        return
    events = window_events(replay, "xgboost_kw", ctx.capacity_kw, ctx.alpha)
    errors = replay.assign(abs_err=(replay["actual_kw"] - replay["xgboost_kw"]).abs(),
                           bias=replay["xgboost_kw"] - replay["actual_kw"])
    daily = errors.groupby("origin").agg(mae=("abs_err", "mean"), bias=("bias", "mean"))
    alerts = int(events["alert"].sum())
    ui.cards([
        ui.card("Forecasts replayed", str(len(events)), "", "One day-ahead forecast per day"),
        ui.card("Mean absolute error", f"{mae(replay['actual_kw'], replay['xgboost_kw']):.2f}", "kW",
                f"RMSE {rmse(replay['actual_kw'], replay['xgboost_kw']):.2f} kW"),
        ui.card("Mean peak error", f"{(events['forecast_peak_kw'] - events['actual_peak_kw']).mean():+.1f}", "kW",
                "Forecast peak minus actual peak"),
        ui.card("Alerts issued", str(alerts), "", f"{int((events['outcome'] == 'Detected breach').sum())} confirmed, "
                f"{int((events['outcome'] == 'False alarm').sum())} false alarms"),
        ui.card("Missed breaches", str(int((events["outcome"] == "Missed breach").sum())), "",
                f"of {int(events['actual_breach'].sum())} actual breach days"),
    ])
    st.plotly_chart(charts.replay_chart(replay, ctx.capacity_kw, ctx.alpha), config=charts.CONFIG, key="replay_chart")
    table = pd.DataFrame({
        "Forecast issued": events["origin"],
        "Forecast peak (kW)": events["forecast_peak_kw"],
        "Actual peak (kW)": events["actual_peak_kw"],
        "Peak error (kW)": events["forecast_peak_kw"] - events["actual_peak_kw"],
        "MAE (kW)": events["origin"].map(daily["mae"]),
        "Bias (kW)": events["origin"].map(daily["bias"]),
        "Outcome": events["outcome"],
    }).sort_values("Forecast issued", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(table)), column_config={
        "Forecast issued": st.column_config.DatetimeColumn(format="ddd D MMM YYYY, HH:mm"),
        **{c: st.column_config.NumberColumn(format="%.2f") for c in
           ("Forecast peak (kW)", "Actual peak (kW)", "Peak error (kW)", "MAE (kW)", "Bias (kW)")},
    })
    ui.note("Previous forecasts are produced by replaying the active model on the observed real data: each forecast "
            "sees only data before its issue time and is then compared with what actually happened.", title="About previous forecasts")


def _table(frame: pd.DataFrame, columns: dict[str, str], empty_title: str, empty_text: str) -> None:
    if frame.empty:
        ui.empty_state(empty_title, empty_text)
        return
    shown = frame[list(columns)].rename(columns=columns)
    st.dataframe(shown, hide_index=True, width="stretch", height=min(460, 38 + 35 * len(shown)))


def render() -> None:
    ctx = services.safe_context(title="History")
    if ctx is None:
        return
    ui.page_header("History", "Previous forecasts, outcomes, alerts and scenario runs", ctx.organisation, ctx.chips())
    tabs = st.tabs(["Forecasts and outcomes", "Forecast log", "Alert log", "Scenario runs", "Model versions"])

    with tabs[0]:
        _forecast_outcomes(ctx)
    with tabs[1]:
        runs = storage.read_table(ctx.org_id, "forecast_runs", ctx.depot["name"])
        if not runs.empty:
            runs["lead"] = runs["lead_minutes"].map(lambda m: "" if pd.isna(m) else ui.fmt_duration(pd.Timedelta(minutes=m)))
            runs["status"] = runs["status"].map(common.status_text)
        _table(runs, {"created_at": "Generated", "issued_at": "Forecast from", "horizon_hours": "Horizon (h)",
                      "current_kw": "Demand at issue (kW)", "peak_kw": "Predicted peak (kW)", "peak_time": "Peak time",
                      "capacity_kw": "Capacity (kW)", "status": "Status", "breach_time": "Expected breach",
                      "lead": "Lead time", "model_version": "Model", "user": "User"},
               "No forecasts logged yet", "Forecasts are logged automatically when they are generated.")
    with tabs[2]:
        alerts = storage.read_table(ctx.org_id, "alerts", ctx.depot["name"])
        if not alerts.empty:
            alerts["lead"] = alerts["lead_minutes"].map(lambda m: "" if pd.isna(m) else ui.fmt_duration(pd.Timedelta(minutes=m)))
            alerts["level"] = alerts["level"].map(common.status_text)
        _table(alerts, {"created_at": "Raised", "issued_at": "Forecast from", "level": "Level", "event_time": "Expected at",
                        "peak_kw": "Forecast peak (kW)", "capacity_kw": "Capacity (kW)", "lead": "Lead time",
                        "status": "Alert status", "acknowledged_by": "Acknowledged by", "acknowledged_at": "Acknowledged"},
               "No alerts logged", "Warnings and breaches raised for this depot will be listed here.")
    with tabs[3]:
        scenarios = storage.read_table(ctx.org_id, "scenario_runs", ctx.depot["name"])
        _table(scenarios, {"created_at": "Saved", "growth_pct": "Growth (%)", "utilisation_factor": "Utilisation factor",
                           "capacity_kw": "Capacity (kW)", "horizon_hours": "Horizon (h)", "base_peak_kw": "Base peak (kW)",
                           "scenario_peak_kw": "Scenario peak (kW)", "base_status": "Base status",
                           "scenario_status": "Scenario status", "scenario_breach_time": "Scenario breach time",
                           "note": "Note", "user": "User"},
               "No scenario runs saved", "Use Save scenario run on the Scenario Analysis page to keep a record.")
    with tabs[4]:
        meta = ctx.bundle.metadata
        ui.cards([
            ui.card("Active model", f"v{ctx.bundle.version}", "", ui.esc(ctx.bundle.name), "good"),
            ui.card("Trained on", ctx.bundle.trained_at or "n/a", "", f"XGBoost {meta.get('xgboost_version', '')}"),
            ui.card("Archived versions", str(len(model_loader.list_archive())), "", "Kept by the platform for rollback"),
        ])
        events = storage.read_model_events()
        _table(events, {"created_at": "When", "action": "Action", "model_version": "Model version", "detail": "Detail",
                        "user": "User"},
               "No model changes recorded", "Model replacements and rollbacks will be listed here.")
    ui.footer(ctx.bundle.version, ctx.generated_at, ctx.series.index[-1], ctx.source_label)
