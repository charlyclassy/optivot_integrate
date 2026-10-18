"""Helpers shared by several screens."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import theme as ui
from forecasting import STATUS_BREACH, STATUS_OK, STATUS_WARNING

STATUS_LABEL = {STATUS_OK: "✓ Within Capacity", STATUS_WARNING: "▲ Warning", STATUS_BREACH: "● Breach"}
TABLE_TIME = "%a %d %b %H:%M"


def status_text(status: str) -> str:
    return STATUS_LABEL.get(status, status)


def time_text(value) -> str:
    return "" if value is None or pd.isna(value) else pd.Timestamp(value).strftime(TABLE_TIME)


def lead_text(value) -> str:
    return "" if value is None or pd.isna(value) else ui.fmt_duration(value)


def fleet_dataframe(fleet: pd.DataFrame) -> None:
    """Status table for all depots."""
    ok = fleet[fleet["Status"] != "Unavailable"].copy()
    if ok.empty:
        ui.empty_state("No depot forecasts available", "Add a depot with at least 15 days of data on the Depots page.")
        return
    order = {STATUS_BREACH: 0, STATUS_WARNING: 1, STATUS_OK: 2}
    ok = ok.sort_values(by="Status", key=lambda s: s.map(order), kind="stable")
    table = pd.DataFrame({
        "Depot": ok["Depot"],
        "Status": ok["Status"].map(status_text),
        "Now (kW)": ok["Current (kW)"],
        "Peak (kW)": ok["Predicted peak (kW)"],
        "Capacity": ok["Capacity (kW)"],
        "Peak vs capacity": ok["Peak utilisation"] * 100,
        "Peak time": ok["Peak time"].map(time_text),
        "Breach": ok["Breach time"].map(time_text).replace("", "None"),
        "Lead time": ok["Lead time"].map(lead_text).replace("", "n/a"),
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Now (kW)": st.column_config.NumberColumn(format="%.1f", help="Latest measured demand"),
            "Peak (kW)": st.column_config.NumberColumn(format="%.1f", help="Predicted peak demand in the horizon"),
            "Capacity": st.column_config.NumberColumn(format="%.0f kW"),
            "Peak vs capacity": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=150),
            "Breach": st.column_config.TextColumn(help="Expected breach time"),
        },
    )
    failed = fleet[fleet["Status"] == "Unavailable"]
    for _, row in failed.iterrows():
        st.warning(f"{row['Depot']}: forecast unavailable. {row['error']}", icon=":material/warning:")


def banner_text(ctx) -> str:
    risk = ctx.risk
    peak = f"{ui.fmt_kw(risk.peak_kw)} kW at {ui.fmt_time(risk.peak_time)}"
    if risk.status == STATUS_BREACH:
        return (f"{ctx.depot['name']} is forecast to reach its {ui.fmt_kw(risk.capacity_kw)} kW capacity at "
                f"{ui.fmt_time(risk.breach_time)}, {ui.fmt_duration(risk.lead_time)} after the latest reading. "
                f"Forecast peak {peak}.")
    if risk.status == STATUS_WARNING:
        return (f"{ctx.depot['name']} is forecast to pass the warning level of {ui.fmt_kw(risk.warning_kw)} kW at "
                f"{ui.fmt_time(risk.warning_time)} ({ui.fmt_duration(risk.lead_time)} after the latest reading) "
                f"but to stay under capacity. Forecast peak {peak}.")
    return (f"{ctx.depot['name']} is forecast to stay below the warning level of {ui.fmt_kw(risk.warning_kw)} kW "
            f"for the next {ctx.horizon_hours} hours. Forecast peak {peak}.")
