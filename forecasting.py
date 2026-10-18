"""Forecast generation, the baseline, capacity/warning logic and scenarios."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_engineering import FeatureConfig, build_features

STATUS_OK = "Within Capacity"
STATUS_WARNING = "Warning"
STATUS_BREACH = "Breach"


class ForecastError(ValueError):
    """Raised with a user-readable message when a forecast cannot be produced."""


# =============================================================================
# XGBoost: explicit recursive multi-step forecast
# =============================================================================
def recursive_forecast(
    model, cfg: FeatureConfig, windows: np.ndarray, first_times: pd.DatetimeIndex, steps: int
) -> np.ndarray:
    """Forecast ``steps`` intervals ahead for one or many forecast origins.

    The model predicts one interval ahead. Each prediction is appended to the
    lookback window and becomes a lag input for the next step, so beyond the
    first step the lag/rolling features are built from the model's own output
    and never from future observations.

    windows      (n, lookback_steps) observed demand up to each origin, in kW.
    first_times  (n,) timestamp of the first interval to forecast for each origin.
    Returns      (n, steps) forecast demand in kW.
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, look = windows.shape
    first_times = pd.DatetimeIndex(first_times)
    step = pd.Timedelta(minutes=cfg.sampling_minutes)
    buffer = np.empty((n, look + steps))
    buffer[:, :look] = windows
    for h in range(steps):
        X, scale = build_features(buffer[:, h:h + look], first_times + h * step, cfg)
        ratio = model.predict(X)
        buffer[:, look + h] = np.maximum(ratio * scale, 0.0)
    return buffer[:, look:]


def baseline_forecast(cfg: FeatureConfig, windows: np.ndarray, steps: int) -> np.ndarray:
    """Baseline: previous-day same-time demand.

    The forecast for a time of day is the demand observed at that time on the
    last full day before the forecast was issued (repeated for horizons > 24 h).
    """
    windows = np.asarray(windows, dtype=np.float64)
    spd = cfg.steps_per_day
    last_day = windows[:, -spd:]
    return last_day[:, np.arange(steps) % spd]


def forecast_series(model, cfg: FeatureConfig, series: pd.Series, horizon_hours: int) -> pd.DataFrame:
    """Genuine future forecast for the timestamps after the end of ``series``."""
    look = cfg.lookback_steps
    if len(series) < look:
        have = len(series) * cfg.sampling_minutes / 1440
        raise ForecastError(
            f"The active model needs {cfg.lookback_days} days of history to forecast; "
            f"this depot has {have:.1f} days."
        )
    steps = int(horizon_hours * 60 // cfg.sampling_minutes)
    step = pd.Timedelta(minutes=cfg.sampling_minutes)
    first = series.index[-1] + step
    window = series.to_numpy(dtype=np.float64)[-look:][None, :]
    values = recursive_forecast(model, cfg, window, pd.DatetimeIndex([first]), steps)[0]
    base = baseline_forecast(cfg, window, steps)[0]
    times = pd.date_range(first, periods=steps, freq=step)
    return pd.DataFrame({"timestamp": times, "forecast_kw": values, "baseline_kw": base})


# =============================================================================
# Capacity and warning logic
# =============================================================================
def classify(demand_kw, capacity_kw: float, alpha: float) -> np.ndarray:
    """Per-interval status.

    Breach:           demand >= C
    Warning:          alpha * C <= demand < C
    Within Capacity:  demand < alpha * C
    """
    demand = np.asarray(demand_kw, dtype=np.float64)
    status = np.full(demand.shape, STATUS_OK, dtype=object)
    status[demand >= alpha * capacity_kw] = STATUS_WARNING
    status[demand >= capacity_kw] = STATUS_BREACH
    return status


@dataclass
class RiskAssessment:
    status: str
    peak_kw: float
    peak_time: pd.Timestamp
    capacity_kw: float
    warning_kw: float
    headroom_kw: float            # capacity minus forecast peak (negative when breached)
    peak_utilisation: float       # forecast peak / capacity
    warning_time: pd.Timestamp | None   # first interval at or above the warning threshold
    breach_time: pd.Timestamp | None    # first interval at or above capacity
    lead_time: pd.Timedelta | None      # from forecast issue to the first breach (else first warning)
    warning_intervals: int
    breach_intervals: int


def assess_risk(
    times: pd.DatetimeIndex, demand_kw, capacity_kw: float, alpha: float, issued_at: pd.Timestamp
) -> RiskAssessment:
    """Summarise a forecast against capacity C and warning threshold alpha * C."""
    demand = np.asarray(demand_kw, dtype=np.float64)
    times = pd.DatetimeIndex(times)
    status = classify(demand, capacity_kw, alpha)
    breach = status == STATUS_BREACH
    warn_or_breach = status != STATUS_OK
    breach_time = times[breach][0] if breach.any() else None
    warning_time = times[warn_or_breach][0] if warn_or_breach.any() else None
    first_event = breach_time if breach_time is not None else warning_time
    i_peak = int(np.argmax(demand))
    overall = STATUS_BREACH if breach.any() else STATUS_WARNING if warn_or_breach.any() else STATUS_OK
    return RiskAssessment(
        status=overall,
        peak_kw=float(demand[i_peak]),
        peak_time=times[i_peak],
        capacity_kw=float(capacity_kw),
        warning_kw=float(alpha * capacity_kw),
        headroom_kw=float(capacity_kw - demand[i_peak]),
        peak_utilisation=float(demand[i_peak] / capacity_kw) if capacity_kw else float("nan"),
        warning_time=warning_time,
        breach_time=breach_time,
        lead_time=(first_event - issued_at) if first_event is not None else None,
        warning_intervals=int((status == STATUS_WARNING).sum()),
        breach_intervals=int(breach.sum()),
    )


def status_periods(times: pd.DatetimeIndex, status: np.ndarray, step: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Contiguous warning / breach periods as (start, end, status) for chart shading."""
    periods = []
    start = None
    current = STATUS_OK
    for t, s in zip(times, status):
        if s != current:
            if current != STATUS_OK:
                periods.append((start, t, current))
            start, current = t, s
    if current != STATUS_OK and len(times):
        periods.append((start, times[-1] + step, current))
    return periods


# =============================================================================
# Scenario analysis (hypothetical adjustment of the real XGBoost forecast)
# =============================================================================
def apply_scenario(base_forecast_kw, growth_pct: float = 0.0, utilisation_factor: float = 1.0) -> np.ndarray:
    """Scenario demand = base XGBoost forecast x (1 + growth) x utilisation factor."""
    base = np.asarray(base_forecast_kw, dtype=np.float64)
    return base * (1.0 + growth_pct / 100.0) * utilisation_factor


def max_growth_before(base_forecast_kw, threshold_kw: float) -> float | None:
    """Largest growth (%) that keeps the base forecast peak below ``threshold_kw``."""
    peak = float(np.max(base_forecast_kw))
    if peak <= 0:
        return None
    return (threshold_kw / peak - 1.0) * 100.0
