"""Home / Overview: is the depot safe, at risk, or predicted to breach capacity?"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard import services
from dashboard import theme as ui
from dashboard.pages import shared as common


def render() -> None:
    ctx = services.safe_context(title="Overview")
    if ctx is None:
        return
    ui.page_header("Overview", f"Capacity outlook for the next {ctx.horizon_hours} hours", ctx.organisation, ctx.chips())
    ui.status_banner(ctx.risk.status, common.banner_text(ctx), ctx.risk.peak_utilisation, ctx.alpha)
    ui.risk_cards(ctx.risk, ctx.current_kw)

    ui.section(f"{ctx.depot['name']}: recent demand and forecast")
    history = ctx.series[ctx.series.index >= ctx.issued_at - pd.Timedelta(hours=24)]
    fig = charts.forecast_chart(history, ctx.forecast, ctx.capacity_kw, ctx.alpha, ctx.issued_at, ctx.step, height=380)
    st.plotly_chart(fig, config=charts.CONFIG, key="overview_chart")

    ui.section("All depots")
    with st.spinner("Updating depot forecasts"):
        fleet = services.fleet_table(ctx.settings, ctx.bundle, ctx.horizon_hours)
    common.fleet_dataframe(fleet)
    ui.footer(ctx.bundle.version, ctx.generated_at, ctx.series.index[-1], ctx.source_label)
