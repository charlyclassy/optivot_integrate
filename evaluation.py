"""Evaluation: rolling-origin backtests, error metrics and breach/alert metrics.

Definitions used everywhere (code, dashboard, README):

* A *forecast window* is one multi-step forecast issued at a forecast origin.
* Actual breach event: actual demand >= capacity C at any time in the window.
* Alert: the forecast reaches the warning threshold, i.e. forecast demand >= alpha * C
  at any time in the window (forecast status Warning or Breach).
* TP = alert and actual breach, FP = alert without a breach, FN = breach without alert.
* Breach recall (detection rate) = TP / (TP + FN); N/A when there were no actual breaches.
* False-alarm ratio = FP / (TP + FP), the share of issued alerts that were false;
  N/A when no alerts were issued.
* Warning lead time = first actual breach time - forecast issue time, for detected
  breaches (TP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import FeatureConfig
from forecasting import baseline_forecast, recursive_forecast

MODEL_COLUMNS = {"XGBoost": "xgboost_kw", "Baseline (previous day)": "baseline_kw"}


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def rolling_origin_backtest(
    model,
    cfg: FeatureConfig,
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon_hours: int = 24,
    origin_hour: int = 0,
) -> pd.DataFrame:
    """Replay forecasts over ``[start, end)``: one forecast per day at ``origin_hour``.

    For every origin the model and the baseline see only data before the origin
    and forecast the next ``horizon_hours``. Both are scored on the same actuals.
    Returns long-format rows: origin, timestamp, actual_kw, xgboost_kw, baseline_kw.
    """
    step = pd.Timedelta(minutes=cfg.sampling_minutes)
    steps = int(horizon_hours * 60 // cfg.sampling_minutes)
    look = cfg.lookback_steps
    values = series.to_numpy(dtype=np.float64)
    t0 = series.index[0]

    first_possible = t0 + look * step
    origin = max(pd.Timestamp(start), first_possible).normalize() + pd.Timedelta(hours=origin_hour)
    if origin < max(pd.Timestamp(start), first_possible):
        origin += pd.Timedelta(days=1)
    last_origin = min(pd.Timestamp(end), series.index[-1] + step) - steps * step
    origins = pd.date_range(origin, last_origin, freq="D") if origin <= last_origin else pd.DatetimeIndex([])
    if len(origins) == 0:
        return pd.DataFrame(columns=["origin", "timestamp", "actual_kw", "xgboost_kw", "baseline_kw"])

    pos = ((origins - t0) / step).astype(int).to_numpy()
    windows = np.stack([values[p - look:p] for p in pos])
    actual = np.stack([values[p:p + steps] for p in pos])
    xgb = recursive_forecast(model, cfg, windows, origins, steps)
    base = baseline_forecast(cfg, windows, steps)

    offsets = pd.to_timedelta(np.arange(steps) * cfg.sampling_minutes, unit="min").to_numpy()
    return pd.DataFrame({
        "origin": np.repeat(origins.to_numpy(), steps),
        "timestamp": (origins.to_numpy()[:, None] + offsets[None, :]).ravel(),
        "actual_kw": actual.ravel(),
        "xgboost_kw": xgb.ravel(),
        "baseline_kw": base.ravel(),
    })


def window_events(backtest: pd.DataFrame, column: str, capacity_kw: float, alpha: float) -> pd.DataFrame:
    """One row per forecast window with the breach / alert outcome for ``column``."""
    rows = []
    for origin, g in backtest.groupby("origin", sort=True):
        actual_breach = g["actual_kw"].to_numpy() >= capacity_kw
        alert = g[column].to_numpy() >= alpha * capacity_kw
        first_breach = g["timestamp"].to_numpy()[actual_breach][0] if actual_breach.any() else pd.NaT
        first_alert = g["timestamp"].to_numpy()[alert][0] if alert.any() else pd.NaT
        has_breach, has_alert = bool(actual_breach.any()), bool(alert.any())
        outcome = (
            "Detected breach" if has_breach and has_alert
            else "Missed breach" if has_breach
            else "False alarm" if has_alert
            else "No event"
        )
        rows.append({
            "origin": origin,
            "actual_peak_kw": float(g["actual_kw"].max()),
            "forecast_peak_kw": float(g[column].max()),
            "actual_breach": has_breach,
            "alert": has_alert,
            "first_actual_breach": pd.Timestamp(first_breach) if has_breach else pd.NaT,
            "first_alert": pd.Timestamp(first_alert) if has_alert else pd.NaT,
            "outcome": outcome,
        })
    return pd.DataFrame(rows)


def breach_metrics(events: pd.DataFrame) -> dict:
    """Recall, false-alarm ratio and lead-time statistics. ``None`` means N/A."""
    if events.empty:
        return {"windows": 0, "actual_breaches": 0, "alerts": 0, "tp": 0, "fp": 0, "fn": 0,
                "breach_recall": None, "false_alarm_ratio": None,
                "lead_time_mean_h": None, "lead_time_median_h": None, "lead_time_min_h": None}
    tp = int((events["actual_breach"] & events["alert"]).sum())
    fn = int((events["actual_breach"] & ~events["alert"]).sum())
    fp = int((~events["actual_breach"] & events["alert"]).sum())
    detected = events[events["actual_breach"] & events["alert"]]
    lead_h = ((detected["first_actual_breach"] - detected["origin"]) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    return {
        "windows": int(len(events)),
        "actual_breaches": tp + fn,
        "alerts": tp + fp,
        "tp": tp, "fp": fp, "fn": fn,
        "breach_recall": tp / (tp + fn) if (tp + fn) else None,
        "false_alarm_ratio": fp / (tp + fp) if (tp + fp) else None,
        "lead_time_mean_h": float(lead_h.mean()) if len(lead_h) else None,
        "lead_time_median_h": float(np.median(lead_h)) if len(lead_h) else None,
        "lead_time_min_h": float(lead_h.min()) if len(lead_h) else None,
    }


def evaluate(backtest: pd.DataFrame, capacity_kw: float, alpha: float) -> pd.DataFrame:
    """Baseline vs XGBoost on the same windows: one row per model."""
    rows = []
    for label, column in MODEL_COLUMNS.items():
        if backtest.empty:
            continue
        row = {
            "model": label,
            "mae_kw": mae(backtest["actual_kw"], backtest[column]),
            "rmse_kw": rmse(backtest["actual_kw"], backtest[column]),
            "peak_mae_kw": None,
        }
        events = window_events(backtest, column, capacity_kw, alpha)
        row["peak_mae_kw"] = mae(events["actual_peak_kw"], events["forecast_peak_kw"])
        row.update(breach_metrics(events))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_pct(value) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.0%}"


def fmt_hours(value) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.1f} h"
