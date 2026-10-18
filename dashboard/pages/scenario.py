"""Scenario Analysis: hypothetical growth applied to the real XGBoost forecast."""
from __future__ import annotations

import streamlit as st

from dashboard import charts
import config
from dashboard import services
from dashboard import database as storage
from dashboard import theme as ui
from forecasting import STATUS_BREACH, apply_scenario, assess_risk, max_growth_before
from dashboard.pages import shared as common


def render() -> None:
    settings = services.get_settings()
    horizon = int(st.session_state.get("scenario_horizon") or settings["horizon_hours"])
    ctx = services.safe_context(horizon, title="Scenario Analysis")
    if ctx is None:
        return
    ui.page_header("Scenario Analysis", "Test fleet growth and capacity changes against the model forecast",
                   ctx.organisation, ctx.chips("Forecast + Scenario"))

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        growth = c1.slider("Demand / fleet growth (%)", -30, 200, 25, 5,
                           help="Hypothetical change in charging demand, for example from adding vehicles.")
        utilisation = c2.slider("Utilisation factor", 0.50, 1.50, 1.00, 0.05,
                                help="Optional extra scaling, for example 0.9 if 10% of charging is shifted off-peak.")
        capacity = c3.number_input("Grid / depot capacity (kW)", min_value=1.0, value=float(ctx.capacity_kw), step=5.0,
                                   help="Capacity assumed in this scenario. The depot's configured capacity is not changed.")
        c4.selectbox("Forecast horizon", list(config.HORIZON_CHOICES), index=list(config.HORIZON_CHOICES).index(horizon),
                     format_func=lambda h: f"{h} hours", key="scenario_horizon")

    base = ctx.forecast["forecast_kw"].to_numpy()
    scenario_kw = apply_scenario(base, growth, utilisation)
    times = ctx.forecast["timestamp"]
    base_risk = assess_risk(times, base, capacity, ctx.alpha, ctx.issued_at)
    risk = assess_risk(times, scenario_kw, capacity, ctx.alpha, ctx.issued_at)
    tone = ui.STATUS_TONE[risk.status]

    multiplier = (1 + growth / 100) * utilisation
    growth_to_warning = max_growth_before(base, ctx.alpha * capacity)
    growth_to_breach = max_growth_before(base, capacity)
    ui.cards([
        ui.card("Base forecast peak", ui.fmt_kw(base_risk.peak_kw), "kW", ui.status_pill(base_risk.status),
                ui.STATUS_TONE[base_risk.status]),
        ui.card("Scenario peak", ui.fmt_kw(risk.peak_kw), "kW",
                f"{ui.fmt_time(risk.peak_time)} &middot; {risk.peak_utilisation:.0%} of capacity", tone),
        ui.card("Scenario status", risk.status, "", ui.status_pill(risk.status), tone),
        ui.card("Expected breach time", ui.fmt_time(risk.breach_time), "",
                f"Lead time {ui.fmt_duration(risk.lead_time)}" if risk.lead_time is not None else "Scenario stays below the warning level",
                "critical" if risk.status == STATUS_BREACH else "neutral"),
        ui.card("Headroom at peak", ui.fmt_kw(risk.headroom_kw), "kW",
                "Capacity minus scenario peak", "critical" if risk.headroom_kw < 0 else "neutral"),
        ui.card("Growth before breach", f"{growth_to_breach:+.0f}" if growth_to_breach is not None else "n/a", "%",
                f"Warning level at {growth_to_warning:+.0f}% growth" if growth_to_warning is not None else ""),
    ])

    history = ctx.series[ctx.series.index >= ctx.issued_at - ctx.step * (24 * 60 // ctx.bundle.cfg.sampling_minutes)]
    fig = charts.forecast_chart(history, ctx.forecast, capacity, ctx.alpha, ctx.issued_at, ctx.step,
                                scenario_kw=scenario_kw, height=450)
    st.plotly_chart(fig, config=charts.CONFIG, key="scenario_chart")
    ui.note(f"The dashed indigo line is the XGBoost forecast from real depot data. The teal line is that forecast multiplied "
            f"by {multiplier:.2f} (growth {growth:+d}%, utilisation factor {utilisation:.2f}); it is a hypothetical "
            "scenario, not a prediction. Shading shows where the scenario reaches the warning level or capacity.", title="About this scenario")

    left, right = st.columns([3, 1], vertical_alignment="bottom")
    note = left.text_input("Scenario note (optional)", placeholder="For example: add 20 vans in Q3", max_chars=120)
    if right.button("Save scenario run", type="primary", icon=":material/save:", width="stretch"):
        storage.log_scenario_run(ctx.org_id, {
            "user": services.user_email(), "depot": ctx.depot["name"], "issued_at": ctx.issued_at,
            "horizon_hours": horizon, "growth_pct": float(growth), "utilisation_factor": float(utilisation),
            "capacity_kw": float(capacity), "alpha": ctx.alpha, "base_peak_kw": base_risk.peak_kw,
            "scenario_peak_kw": risk.peak_kw, "base_status": base_risk.status, "scenario_status": risk.status,
            "scenario_breach_time": risk.breach_time, "model_version": ctx.bundle.version, "note": note,
        })
        st.success("Scenario saved. It is listed under History and can be included in reports.", icon=":material/check_circle:")
    ui.footer(ctx.bundle.version, ctx.generated_at, ctx.series.index[-1], ctx.source_label + " + hypothetical scenario")
