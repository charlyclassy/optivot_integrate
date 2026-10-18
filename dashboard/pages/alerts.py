"""Alerts: current warnings/breaches for every depot and recent alert outcomes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from dashboard import services
from dashboard import database as storage
from dashboard import theme as ui
from evaluation import window_events
from forecasting import STATUS_BREACH, STATUS_OK, STATUS_WARNING, ForecastError
from dashboard.pages import shared as common


def _active_alerts(ctx) -> pd.DataFrame:
    """Alerts raised by the latest forecast of each depot (logged so they can be acknowledged)."""
    rows = []
    for key, depot in ctx.settings["depots"].items():
        try:
            series, forecast, risk, issued_at = services.depot_risk(ctx.settings, key, ctx.horizon_hours, ctx.bundle)
        except (ForecastError, ValueError, KeyError):
            continue
        if risk.status == STATUS_OK:
            continue
        alpha = storage.depot_alpha(ctx.settings, key)
        storage.log_forecast_run(ctx.org_id, depot["name"], issued_at, ctx.horizon_hours, ctx.bundle.version, risk,
                                 float(series.iloc[-1]), forecast, services.user_email())
        logged = storage.find_alert(ctx.org_id, depot["name"], issued_at, ctx.horizon_hours, ctx.bundle.version,
                                    risk.capacity_kw, alpha) or {}
        event_time = risk.breach_time if risk.breach_time is not None else risk.warning_time
        rows.append({
            "id": logged.get("id"), "Issued": issued_at, "Depot": depot["name"], "Level": risk.status,
            "Event time": event_time, "Forecast peak (kW)": risk.peak_kw, "Capacity (kW)": risk.capacity_kw,
            "Lead time": risk.lead_time, "Alert status": logged.get("status", "Open"),
            "Acknowledged by": logged.get("acknowledged_by") or "",
        })
    return pd.DataFrame(rows)


def _recent_alerts(ctx, depot_keys: list[str]) -> pd.DataFrame:
    """Day-ahead alerts over the last days, with what actually happened."""
    frames = []
    for key in depot_keys:
        depot = ctx.settings["depots"][key]
        try:
            replay = services.get_replay(depot)
        except (ForecastError, ValueError, KeyError):
            continue
        if replay.empty:
            continue
        capacity, alpha = float(depot["capacity_kw"]), storage.depot_alpha(ctx.settings, key)
        events = window_events(replay, "xgboost_kw", capacity, alpha)
        events = events[events["outcome"] != "No event"].copy()
        if events.empty:
            continue
        events["Depot"] = depot["name"]
        events["Capacity (kW)"] = capacity
        events["Level"] = [
            "Not alerted" if not alert else STATUS_BREACH if peak >= capacity else STATUS_WARNING
            for alert, peak in zip(events["alert"], events["forecast_peak_kw"])
        ]
        events["Lead time"] = events["first_alert"] - events["origin"]
        frames.append(events)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_values("origin", ascending=False)
    return pd.DataFrame({
        "Issued": out["origin"], "Depot": out["Depot"], "Level": out["Level"].map(common.status_text),
        "Forecast peak (kW)": out["forecast_peak_kw"], "Actual peak (kW)": out["actual_peak_kw"],
        "Capacity (kW)": out["Capacity (kW)"], "Lead time": out["Lead time"].map(common.lead_text),
        "Outcome": out["outcome"],
    })


def render() -> None:
    ctx = services.safe_context(title="Alerts")
    if ctx is None:
        return
    ui.page_header("Alerts", "Capacity warnings and breaches across your depots", ctx.organisation,
                   [("Depots", str(len(ctx.settings["depots"]))), ("Data", "Real data + Forecast"),
                    ("Model", f"v{ctx.bundle.version}")])

    with st.spinner("Checking depot forecasts"):
        active = _active_alerts(ctx)
    breaches = int((active["Level"] == STATUS_BREACH).sum()) if not active.empty else 0
    warnings = int((active["Level"] == STATUS_WARNING).sum()) if not active.empty else 0
    open_count = int((active["Alert status"] == "Open").sum()) if not active.empty else 0
    ui.cards([
        ui.card("Breach alerts", str(breaches), "", "Forecast at or above capacity", "critical" if breaches else "neutral"),
        ui.card("Warning alerts", str(warnings), "", "Forecast at or above the warning threshold", "warning" if warnings else "neutral"),
        ui.card("Awaiting acknowledgement", str(open_count), "", "Open alerts from the latest forecasts"),
        ui.card("Depots within capacity", str(len(ctx.settings["depots"]) - breaches - warnings), "",
                f"of {len(ctx.settings['depots'])} depots", "good"),
    ])

    ui.section(f"Active alerts: next {ctx.horizon_hours} hours")
    if active.empty:
        ui.empty_state("No active alerts", "Every depot is forecast to stay below its warning threshold.")
    else:
        order = {STATUS_BREACH: 0, STATUS_WARNING: 1}
        active = active.sort_values(by="Level", key=lambda s: s.map(order), kind="stable")
        shown = pd.DataFrame({
            "Issued": active["Issued"].map(common.time_text), "Depot": active["Depot"],
            "Level": active["Level"].map(common.status_text),
            "Expected at": active["Event time"].map(common.time_text),
            "Forecast peak (kW)": active["Forecast peak (kW)"], "Capacity (kW)": active["Capacity (kW)"],
            "Lead time": active["Lead time"].map(common.lead_text), "Alert status": active["Alert status"],
            "Acknowledged by": active["Acknowledged by"],
        })
        st.dataframe(shown, hide_index=True, width="stretch", column_config={
            "Forecast peak (kW)": st.column_config.NumberColumn(format="%.1f"),
            "Capacity (kW)": st.column_config.NumberColumn(format="%.0f"),
        })
        pending = active[(active["Alert status"] == "Open") & active["id"].notna()]
        if not pending.empty:
            left, right = st.columns([3, 1], vertical_alignment="bottom")
            labels = {int(r["id"]): f"{r['Depot']}: {r['Level']} expected {common.time_text(r['Event time'])}"
                      for _, r in pending.iterrows()}
            chosen = left.multiselect("Acknowledge alerts", list(labels), format_func=labels.get,
                                      placeholder="Choose alerts to acknowledge")
            if right.button("Acknowledge", type="primary", icon=":material/done_all:", width="stretch", disabled=not chosen):
                for alert_id in chosen:
                    storage.acknowledge_alert(ctx.org_id, alert_id, services.user_email() or "user")
                st.rerun()

    st.write("")
    ui.section(f"Recent alerts and outcomes: last {config.REPLAY_DAYS} days")
    scope = st.segmented_control("Depots", ["All depots", "Selected depot"], default="All depots",
                                 key="alert_scope", label_visibility="collapsed") or "All depots"
    keys = [ctx.depot_key] if scope == "Selected depot" else list(ctx.settings["depots"])
    recent = _recent_alerts(ctx, keys)
    if recent.empty:
        ui.empty_state("No alerts in this period", "No warnings, breaches or missed breaches in the last "
                       f"{config.REPLAY_DAYS} days for the selected depots.")
    else:
        st.dataframe(recent, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(recent)), column_config={
            "Issued": st.column_config.DatetimeColumn(format="ddd D MMM, HH:mm"),
            "Forecast peak (kW)": st.column_config.NumberColumn(format="%.1f"),
            "Actual peak (kW)": st.column_config.NumberColumn(format="%.1f"),
            "Capacity (kW)": st.column_config.NumberColumn(format="%.0f"),
        })
        ui.note("Each row is a day-ahead forecast issued at 00:00 by the active model, replayed on the observed real "
                "data and compared with what actually happened that day. Detected breach: an alert was issued and "
                "demand reached capacity. False alarm: an alert was issued but demand stayed below capacity. "
                "Missed breach: demand reached capacity without an alert.", title="How outcomes are determined")
    ui.footer(ctx.bundle.version, ctx.generated_at, ctx.series.index[-1], ctx.source_label)
