"""Live Forecast: recent observed demand plus the genuine future forecast."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
import config
from dashboard import services
from dashboard import theme as ui
from forecasting import classify
from dashboard.pages import shared as common


def render() -> None:
    settings = services.get_settings()
    horizon = int(st.session_state.get("live_horizon_pick") or settings["horizon_hours"])
    ctx = services.safe_context(horizon, title="Depot Demand Forecast")
    if ctx is None:
        return
    ui.page_header("Depot Demand Forecast", "Observed demand and the XGBoost forecast against grid capacity",
                   ctx.organisation, ctx.chips())

    left, middle, right = st.columns([2.2, 2.2, 1.2], vertical_alignment="bottom")
    with left:
        st.segmented_control("Forecast horizon", list(config.HORIZON_CHOICES), default=horizon,
                             format_func=lambda h: f"{h} h", key="live_horizon_pick")
    with middle:
        history_hours = st.segmented_control(
            "History shown", [24, 48, 72, 168], default=int(settings["history_window_hours"]),
            format_func=lambda h: f"{h} h" if h < 168 else "7 days", key="live_history")
    with right:
        show_baseline = st.toggle("Baseline", value=False, help="Show the baseline: previous-day same-time demand")
    history_hours = history_hours or int(settings["history_window_hours"])

    ui.status_banner(ctx.risk.status, common.banner_text(ctx), ctx.risk.peak_utilisation, ctx.alpha)
    ui.risk_cards(ctx.risk, ctx.current_kw)

    history = ctx.series[ctx.series.index >= ctx.issued_at - pd.Timedelta(hours=history_hours)]
    fig = charts.forecast_chart(history, ctx.forecast, ctx.capacity_kw, ctx.alpha, ctx.issued_at, ctx.step,
                                show_baseline=show_baseline, height=470)
    st.plotly_chart(fig, config=charts.CONFIG, key="live_chart")
    ui.note(f"Forecast horizon {ctx.horizon_hours} h from {ui.fmt_time(ctx.issued_at)} at "
            f"{ctx.bundle.cfg.sampling_minutes}-minute resolution. Demand is shown as measured, without smoothing. "
            "Amber shading: forecast at or above the warning threshold. Red shading: forecast at or above capacity.", title="About this chart")

    table = pd.DataFrame({
        "Time": ctx.forecast["timestamp"],
        "Forecast (kW)": ctx.forecast["forecast_kw"].round(2),
        "Baseline (kW)": ctx.forecast["baseline_kw"].round(2),
        "Share of capacity": (ctx.forecast["forecast_kw"] / ctx.capacity_kw * 100).round(1),
        "Status": [common.status_text(s) for s in classify(ctx.forecast["forecast_kw"], ctx.capacity_kw, ctx.alpha)],
    })
    with st.expander("Forecast table and download"):
        st.dataframe(table, hide_index=True, width="stretch", height=320, column_config={
            "Time": st.column_config.DatetimeColumn(format="ddd D MMM, HH:mm"),
            "Share of capacity": st.column_config.NumberColumn(format="%.1f%%"),
        })
        st.download_button("Download forecast (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{ctx.depot_key}_{ctx.issued_at:%Y%m%d_%H%M}.csv", mime="text/csv",
                           icon=":material/download:")
    ui.footer(ctx.bundle.version, ctx.generated_at, ctx.series.index[-1], ctx.source_label)
