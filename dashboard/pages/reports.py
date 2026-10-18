"""Reports: downloadable Excel, CSV and PDF outputs with depot and date filters."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import reports
from dashboard import services
from dashboard import database as storage
from dashboard import theme as ui
from evaluation import evaluate, fmt_hours, fmt_pct, window_events
from forecasting import ForecastError

SECTION_CHOICES = {
    "forecast_summary": "Forecast summary",
    "capacity_risk": "Capacity-risk summary",
    "model_performance": "Model-performance summary",
    "scenario_summary": "Scenario summary",
    "history": "Forecast history and outcomes",
    "forecast_detail": "Forecast detail (every interval)",
}


def build_sections(ctx, depot_keys: list[str], start, end, wanted: list[str]) -> dict[str, pd.DataFrame]:
    """Assemble the report tables for the chosen depots and dates."""
    summary, risk_rows, detail, perf, history = [], [], [], [], []
    names = [ctx.settings["depots"][k]["name"] for k in depot_keys]
    for key in depot_keys:
        depot = ctx.settings["depots"][key]
        alpha = storage.depot_alpha(ctx.settings, key)
        try:
            series, forecast, risk, issued_at = services.depot_risk(ctx.settings, key, ctx.horizon_hours, ctx.bundle)
        except (ForecastError, ValueError, KeyError):
            continue
        summary.append({
            "Depot": depot["name"], "Data through": series.index[-1], "Forecast from": issued_at,
            "Horizon (h)": ctx.horizon_hours, "Current demand (kW)": float(series.iloc[-1]),
            "Predicted peak (kW)": risk.peak_kw, "Peak time": risk.peak_time,
            "Mean forecast (kW)": float(forecast["forecast_kw"].mean()),
            "Forecast energy (kWh)": float(forecast["forecast_kw"].sum() * ctx.bundle.cfg.sampling_minutes / 60),
            "Model version": ctx.bundle.version,
        })
        risk_rows.append({
            "Depot": depot["name"], "Status": risk.status, "Capacity (kW)": risk.capacity_kw,
            "Warning threshold (kW)": risk.warning_kw, "Predicted peak (kW)": risk.peak_kw,
            "Peak / capacity (%)": risk.peak_utilisation * 100, "Headroom (kW)": risk.headroom_kw,
            "First warning": risk.warning_time, "Expected breach": risk.breach_time,
            "Lead time (min)": risk.lead_time, "Warning intervals": risk.warning_intervals,
            "Breach intervals": risk.breach_intervals,
        })
        detail.append(forecast.assign(Depot=depot["name"])[["Depot", "timestamp", "forecast_kw", "baseline_kw"]].rename(
            columns={"timestamp": "Time", "forecast_kw": "XGBoost forecast (kW)", "baseline_kw": "Baseline (kW)"}))

        predictions = ctx.bundle.test_predictions
        if predictions is not None and (predictions["depot_id"] == depot["data_id"]).any():
            evaluated = predictions[predictions["depot_id"] == depot["data_id"]]
        else:
            evaluated = services.get_replay(depot, days=60)
        if not evaluated.empty:
            for _, row in evaluate(evaluated, float(depot["capacity_kw"]), alpha).iterrows():
                perf.append({
                    "Depot": depot["name"], "Model": row["model"], "MAE (kW)": row["mae_kw"], "RMSE (kW)": row["rmse_kw"],
                    "Forecasts": int(row["windows"]), "Actual breaches": int(row["actual_breaches"]),
                    "Alerts": int(row["alerts"]), "Breach recall": fmt_pct(row["breach_recall"]),
                    "False-alarm ratio": fmt_pct(row["false_alarm_ratio"]),
                    "Mean lead time": fmt_hours(row["lead_time_mean_h"]),
                    "Period": f"{evaluated['timestamp'].min():%Y-%m-%d} to {evaluated['timestamp'].max():%Y-%m-%d}",
                })

        replay = services.get_replay(depot, days=max(1, (series.index[-1].date() - start).days + 1))
        replay = replay[(replay["origin"].dt.date >= start) & (replay["origin"].dt.date <= end)]
        if not replay.empty:
            events = window_events(replay, "xgboost_kw", float(depot["capacity_kw"]), alpha)
            daily = replay.assign(err=(replay["actual_kw"] - replay["xgboost_kw"]).abs()).groupby("origin")["err"].mean()
            events["Depot"] = depot["name"]
            events["MAE (kW)"] = events["origin"].map(daily)
            history.append(events.rename(columns={
                "origin": "Forecast issued", "forecast_peak_kw": "Forecast peak (kW)",
                "actual_peak_kw": "Actual peak (kW)", "outcome": "Outcome"})[
                ["Depot", "Forecast issued", "Forecast peak (kW)", "Actual peak (kW)", "MAE (kW)", "Outcome"]])

    scenarios = storage.read_table(ctx.org_id, "scenario_runs")
    if not scenarios.empty:
        created = pd.to_datetime(scenarios["created_at"])
        scenarios = scenarios[scenarios["depot"].isin(names)]
        scenarios = scenarios.assign(created_at=created).rename(columns={
            "created_at": "Saved", "depot": "Depot", "growth_pct": "Growth (%)", "utilisation_factor": "Utilisation factor",
            "capacity_kw": "Capacity (kW)", "base_peak_kw": "Base peak (kW)", "scenario_peak_kw": "Scenario peak (kW)",
            "base_status": "Base status", "scenario_status": "Scenario status",
            "scenario_breach_time": "Scenario breach time", "note": "Note"})[
            ["Saved", "Depot", "Growth (%)", "Utilisation factor", "Capacity (kW)", "Base peak (kW)",
             "Scenario peak (kW)", "Base status", "Scenario status", "Scenario breach time", "Note"]]

    built = {
        "forecast_summary": pd.DataFrame(summary),
        "capacity_risk": pd.DataFrame(risk_rows),
        "model_performance": pd.DataFrame(perf),
        "scenario_summary": scenarios,
        "history": pd.concat(history, ignore_index=True) if history else pd.DataFrame(),
        "forecast_detail": pd.concat(detail, ignore_index=True) if detail else pd.DataFrame(),
    }
    return {key: built[key] for key in SECTION_CHOICES if key in wanted}


def render() -> None:
    ctx = services.safe_context(title="Reports")
    if ctx is None:
        return
    settings = ctx.settings
    ui.page_header("Reports", "Download forecast, capacity-risk, performance and scenario summaries", ctx.organisation,
                   [("Depot", ctx.depot["name"]), ("Model", f"v{ctx.bundle.version}")])

    last_day = ctx.series.index[-1].date()
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        depot_keys = c1.multiselect("Depots", list(settings["depots"]), default=[ctx.depot_key],
                                    format_func=lambda k: settings["depots"][k]["name"])
        dates = c2.date_input("History dates", value=(last_day - pd.Timedelta(days=13), last_day),
                              min_value=ctx.series.index[0].date(), max_value=last_day,
                              help="Date range for the forecast history section.")
        wanted = st.pills("Sections", list(SECTION_CHOICES), selection_mode="multi", format_func=SECTION_CHOICES.get,
                          default=[k for k in SECTION_CHOICES if k != "forecast_detail" or settings["report"].get("include_forecast_table", True)])
    if not depot_keys:
        ui.empty_state("Choose at least one depot", "Select the depots to include in the report.")
        return
    if not wanted:
        ui.empty_state("Choose at least one section", "Select the sections to include in the report.")
        return
    start, end = (dates if isinstance(dates, tuple) and len(dates) == 2 else (last_day - pd.Timedelta(days=13), last_day))

    with st.spinner("Preparing report"):
        sections = build_sections(ctx, depot_keys, start, end, wanted)
    meta = {
        "Organisation": ctx.organisation,
        "Depots": ", ".join(settings["depots"][k]["name"] for k in depot_keys),
        "History dates": f"{start:%d %b %Y} to {end:%d %b %Y}",
        "Model": f"{ctx.bundle.name} v{ctx.bundle.version}",
        "Generated": f"{datetime.now():%d %b %Y %H:%M}",
    }
    if settings["report"].get("prepared_by"):
        meta["Prepared by"] = settings["report"]["prepared_by"]

    stem = "depot_report_" + ("_".join(depot_keys) if len(depot_keys) <= 2 else f"{len(depot_keys)}_depots")
    builders = {
        "Excel": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", reports.build_excel),
        "PDF": ("pdf", "application/pdf", reports.build_pdf),
        "CSV": ("zip", "application/zip", reports.build_csv_zip),
    }
    order = sorted(builders, key=lambda k: k != settings["report"].get("default_format", "Excel"))
    ui.section("Download")
    columns = st.columns(len(order) + 2)
    for column, label in zip(columns, order):
        extension, mime, builder = builders[label]
        try:
            content = builder(sections, meta)
        except Exception as exc:
            column.error(f"{label} export failed ({type(exc).__name__}).")
            continue
        column.download_button(
            f"{label} report" if label != "CSV" else "CSV files (.zip)", content, file_name=f"{stem}.{extension}",
            mime=mime, icon=":material/download:", width="stretch", type="primary" if label == order[0] else "secondary",
            on_click=reports.save_export, args=(content, stem, extension), key=f"download_{label}",
        )

    st.write("")
    ui.section("Preview")
    tabs = st.tabs([SECTION_CHOICES[k] for k in sections])
    for tab, (key, frame) in zip(tabs, sections.items()):
        with tab:
            if frame.empty:
                ui.empty_state("Nothing to report", "There are no records for the selected depots and dates."
                               if key != "scenario_summary" else "Save a scenario on the Scenario Analysis page to include it here.")
            else:
                shown = reports.preview(frame)
                st.dataframe(shown, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(shown)))
    ui.footer(ctx.bundle.version, ctx.generated_at, ctx.series.index[-1], ctx.source_label)
