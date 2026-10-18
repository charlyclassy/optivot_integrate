"""Plotly charts. One visual language across every screen.

Series colours identify *what a line is*; the status colours (amber, red) are
reserved for the warning threshold, the capacity line and shaded risk periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from forecasting import STATUS_BREACH, classify, status_periods

INK = "#0f172a"          # observed demand
INK_MUTED = "#94a3b8"
FORECAST = "#4f46e5"     # XGBoost forecast
FORECAST_FILL = "rgba(79, 70, 229, 0.10)"
SCENARIO = "#0d9488"     # scenario-adjusted forecast
SCENARIO_FILL = "rgba(13, 148, 136, 0.10)"
BASELINE = "#94a3b8"     # baseline (context)
WARNING = "#d97706"
WARNING_FILL = "#f59e0b"
CRITICAL = "#ef4444"
GRID = "#eef1f6"
FONT = 'Inter, "Segoe UI", system-ui, -apple-system, Arial, sans-serif'
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _layout(fig: go.Figure, height: int, y_title: str = "Demand (kW)") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=56, r=24, t=48, b=40),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family=FONT, size=13, color="#475569"),
        hovermode="x unified", hoverlabel=dict(bgcolor="#0f172a", bordercolor="#0f172a", font=dict(family=FONT, size=12.5, color="#ffffff")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12.5)),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0", ticks="outside", tickcolor="#e2e8f0", ticklen=4,
                     showspikes=True, spikemode="across", spikethickness=1, spikecolor="#cbd5e1", spikedash="solid",
                     tickformat="%H:%M\n%a %d %b", hoverformat="%a %d %b %Y, %H:%M")
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, rangemode="tozero", title_standoff=8)
    return fig


def _line(x, y, name: str, color: str, dash: str | None = None, width: float = 2.0) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash),
                      hovertemplate="%{y:,.1f} kW")


def _thresholds(fig: go.Figure, x0, x1, capacity_kw: float, alpha: float) -> None:
    fig.add_trace(go.Scatter(x=[x0, x1], y=[alpha * capacity_kw] * 2, mode="lines", hoverinfo="skip",
                             name=f"Warning threshold ({alpha * capacity_kw:,.0f} kW)",
                             line=dict(color=WARNING, width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=[x0, x1], y=[capacity_kw] * 2, mode="lines", hoverinfo="skip",
                             name=f"Grid capacity ({capacity_kw:,.0f} kW)",
                             line=dict(color=CRITICAL, width=1.8)))


def _shade_periods(fig: go.Figure, times, values, capacity_kw: float, alpha: float, step: pd.Timedelta) -> None:
    status = classify(values, capacity_kw, alpha)
    for start, end, kind in status_periods(pd.DatetimeIndex(times), status, step):
        breach = kind == STATUS_BREACH
        fig.add_vrect(x0=start, x1=end, line_width=0, layer="below",
                      fillcolor=CRITICAL if breach else WARNING_FILL, opacity=0.16 if breach else 0.18)


def forecast_chart(history: pd.Series, forecast: pd.DataFrame, capacity_kw: float, alpha: float,
                   issued_at: pd.Timestamp, step: pd.Timedelta, scenario_kw: np.ndarray | None = None,
                   show_baseline: bool = False, height: int = 440) -> go.Figure:
    """Observed demand, the future forecast, thresholds and shaded warning/breach periods."""
    fig = go.Figure()
    # Start the forecast line at the last observed point so the two lines meet.
    join_x = [history.index[-1], *forecast["timestamp"]]
    risk_values = scenario_kw if scenario_kw is not None else forecast["forecast_kw"].to_numpy()
    _shade_periods(fig, forecast["timestamp"], risk_values, capacity_kw, alpha, step)

    fig.add_trace(_line(history.index, history.to_numpy(), "Actual demand (observed)", INK))
    if show_baseline:
        fig.add_trace(_line(join_x, [history.iloc[-1], *forecast["baseline_kw"]], "Baseline (previous day)", BASELINE, "dot", 1.6))
    base_name = "Base XGBoost forecast" if scenario_kw is not None else "XGBoost forecast"
    base = _line(join_x, [history.iloc[-1], *forecast["forecast_kw"]], base_name, FORECAST,
                 "dash" if scenario_kw is not None else None, 2.4)
    if scenario_kw is None:
        base.update(fill="tozeroy", fillcolor=FORECAST_FILL)  # tint the forecast period
    fig.add_trace(base)
    if scenario_kw is not None:
        scenario = _line(forecast["timestamp"], scenario_kw, "Scenario-adjusted forecast (hypothetical)", SCENARIO, None, 2.6)
        scenario.update(fill="tozeroy", fillcolor=SCENARIO_FILL)
        fig.add_trace(scenario)
    _thresholds(fig, history.index[0], forecast["timestamp"].iloc[-1], capacity_kw, alpha)

    fig.add_vline(x=issued_at, line=dict(color=INK_MUTED, width=1.2, dash="dot"))
    fig.add_annotation(x=issued_at, y=1.0, yref="paper", text="Forecast start", showarrow=False,
                       xanchor="left", xshift=6, yanchor="top", font=dict(size=12, color="#475569"))
    top = max(float(history.max()), float(np.max(risk_values)), float(forecast["forecast_kw"].max()), capacity_kw)
    _layout(fig, height)
    fig.update_yaxes(range=[0, top * 1.12])
    return fig


def test_period_chart(predictions: pd.DataFrame, capacity_kw: float | None, alpha: float | None,
                      height: int = 430) -> go.Figure:
    """Actual vs predicted over the whole held-out test period."""
    df = predictions.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(_line(df["timestamp"], df["actual_kw"], "Actual demand (observed)", INK, None, 1.6))
    fig.add_trace(_line(df["timestamp"], df["baseline_kw"], "Baseline (previous day)", BASELINE, "dot", 1.3))
    fig.add_trace(_line(df["timestamp"], df["xgboost_kw"], "XGBoost (day-ahead)", FORECAST, None, 1.8))
    if capacity_kw:
        _thresholds(fig, df["timestamp"].iloc[0], df["timestamp"].iloc[-1], capacity_kw, alpha or 1.0)
    _layout(fig, height)
    # Open on the last two weeks; the slider below the chart covers the whole test period.
    last = df["timestamp"].iloc[-1]
    first = max(df["timestamp"].iloc[0], last - pd.Timedelta(days=14))
    fig.update_xaxes(tickformat="%d %b", range=[first, last], rangeslider=dict(visible=True, thickness=0.07))
    fig.update_yaxes(fixedrange=False)
    return fig


def replay_chart(replay: pd.DataFrame, capacity_kw: float, alpha: float, height: int = 380) -> go.Figure:
    """Previous forecasts against what actually happened."""
    df = replay.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(_line(df["timestamp"], df["actual_kw"], "Actual outcome (observed)", INK))
    fig.add_trace(_line(df["timestamp"], df["xgboost_kw"], "Day-ahead forecast (issued 00:00)", FORECAST))
    _thresholds(fig, df["timestamp"].iloc[0], df["timestamp"].iloc[-1], capacity_kw, alpha)
    _layout(fig, height)
    fig.update_xaxes(tickformat="%a %d %b")
    return fig


def importance_chart(importance: dict[str, float], top: int = 12, height: int = 400) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker=dict(color=FORECAST, cornerradius=5), width=0.62,
        text=[f"{v:.1%}" for _, v in items], textposition="outside", textfont=dict(size=12, color="#475569"),
        hovertemplate="%{y}: %{x:.1%} of total gain<extra></extra>", cliponaxis=False,
    ))
    fig.update_layout(height=height, margin=dict(l=150, r=50, t=16, b=36), paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", font=dict(family=FONT, size=13, color="#475569"), showlegend=False)
    fig.update_xaxes(title="Share of total gain", tickformat=".0%", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig
