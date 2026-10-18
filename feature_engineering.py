"""Feature engineering shared by training and inference.

One function, ``build_features``, turns a *lookback window* of past demand into
the model inputs for the next interval. Training and live forecasting both call
it, so there is no train/serve skew, and because a window ends at ``t - 1`` the
features for time ``t`` can never contain information from ``t`` or later.

Everything is driven by ``feature_config.json`` (lags expressed in minutes and
converted to steps from the sampling interval), so a replacement model with a
different lag set works without code changes.

Scale handling: depots differ in size, so demand is divided by the mean demand
of the lookback window before it reaches the model, and the model predicts the
same ratio. The scale is recomputed from each window, i.e. it is causal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

FEATURE_SCHEMA_VERSION = 1
CALENDAR_FEATURES = ("hour", "hour_sin", "hour_cos", "day_of_week", "is_weekend", "month")
DEPOT_FEATURES = ("depot_load_factor", "depot_scale_log")
MIN_SCALE_KW = 1e-3


@dataclass(frozen=True)
class FeatureConfig:
    sampling_minutes: int = 15
    lookback_days: int = 14
    target: str = "demand_ratio_next_step"
    lag_minutes: tuple[int, ...] = (15, 30, 45, 60, 120, 1440, 2880, 10080)
    delta_minutes: tuple[int, ...] = (15, 60)
    rolling_mean_minutes: tuple[int, ...] = (60, 240, 1440)
    rolling_std_minutes: tuple[int, ...] = (60, 1440)
    rolling_max_minutes: tuple[int, ...] = (1440,)
    rolling_min_minutes: tuple[int, ...] = (1440,)
    same_time_mean_days: tuple[int, ...] = (7,)
    calendar: tuple[str, ...] = CALENDAR_FEATURES
    depot_features: tuple[str, ...] = DEPOT_FEATURES
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = field(default=())

    # ------------------------------------------------------------------ helpers
    def steps(self, minutes: int) -> int:
        if minutes % self.sampling_minutes:
            raise ValueError(f"{minutes} min is not a multiple of the {self.sampling_minutes} min sampling interval.")
        return minutes // self.sampling_minutes

    @property
    def steps_per_day(self) -> int:
        return self.steps(1440)

    @property
    def lookback_steps(self) -> int:
        return self.lookback_days * self.steps_per_day

    def derive_feature_names(self) -> list[str]:
        names = [f"lag_{m}min" for m in self.lag_minutes]
        names += [f"delta_{m}min" for m in self.delta_minutes]
        names += [f"roll_mean_{m}min" for m in self.rolling_mean_minutes]
        names += [f"roll_std_{m}min" for m in self.rolling_std_minutes]
        names += [f"roll_max_{m}min" for m in self.rolling_max_minutes]
        names += [f"roll_min_{m}min" for m in self.rolling_min_minutes]
        names += [f"same_time_mean_{d}d" for d in self.same_time_mean_days]
        names += list(self.calendar)
        names += list(self.depot_features)
        return names

    def validate(self) -> None:
        """Raise ValueError if this configuration cannot be served by this code."""
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Feature schema version {self.feature_schema_version} is not supported "
                f"(this application supports version {FEATURE_SCHEMA_VERSION})."
            )
        if self.sampling_minutes <= 0 or 1440 % self.sampling_minutes:
            raise ValueError(f"Sampling interval of {self.sampling_minutes} min must divide a day evenly.")
        unknown = set(self.calendar) - set(CALENDAR_FEATURES)
        unknown |= set(self.depot_features) - set(DEPOT_FEATURES)
        if unknown:
            raise ValueError(f"Unknown feature(s): {', '.join(sorted(unknown))}.")
        longest = max(
            [*self.lag_minutes, *(m + self.sampling_minutes for m in self.delta_minutes),
             *self.rolling_mean_minutes, *self.rolling_std_minutes,
             *self.rolling_max_minutes, *self.rolling_min_minutes,
             *(d * 1440 for d in self.same_time_mean_days), self.sampling_minutes]
        )
        for minutes in (*self.lag_minutes, *self.delta_minutes, *self.rolling_mean_minutes,
                        *self.rolling_std_minutes, *self.rolling_max_minutes, *self.rolling_min_minutes):
            self.steps(minutes)
        if longest > self.lookback_days * 1440:
            raise ValueError(f"Features need {longest} min of history but the lookback is {self.lookback_days} days.")
        derived = self.derive_feature_names()
        if self.feature_names and list(self.feature_names) != derived:
            raise ValueError(
                "Declared feature_names do not match the names (and order) derived from the feature "
                f"settings. Declared: {list(self.feature_names)}. Derived: {derived}."
            )

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        data = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        data["feature_names"] = list(self.feature_names) or self.derive_feature_names()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureConfig":
        known = set(cls.__dataclass_fields__)
        missing = {"sampling_minutes", "lookback_days", "target", "lag_minutes", "feature_names"} - set(data)
        if missing:
            raise ValueError(f"feature_config.json is missing: {', '.join(sorted(missing))}.")
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in known}
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg


def build_features(
    windows: np.ndarray, target_times: pd.DatetimeIndex, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Build model inputs for the interval that follows each lookback window.

    windows       (n, lookback_steps) demand in kW; ``windows[:, -1]`` is the most
                  recent observed interval (t - 1), ``windows[:, -k]`` is t - k.
    target_times  (n,) timestamp of the interval being predicted (t).
    Returns (X, scale): X has one column per ``cfg.derive_feature_names()``;
    ``scale`` (kW) is what the model's ratio output must be multiplied by.
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 2 or windows.shape[1] != cfg.lookback_steps:
        raise ValueError(f"Expected windows of {cfg.lookback_steps} steps, got shape {windows.shape}.")
    target_times = pd.DatetimeIndex(target_times)

    window_mean = windows.mean(axis=1)
    scale = np.maximum(window_mean, MIN_SCALE_KW)
    cols: list[np.ndarray] = []

    def lag(minutes: int) -> np.ndarray:
        return windows[:, -cfg.steps(minutes)] / scale

    def tail(minutes: int) -> np.ndarray:
        return windows[:, -cfg.steps(minutes):]

    for m in cfg.lag_minutes:
        cols.append(lag(m))
    for m in cfg.delta_minutes:
        cols.append(lag(cfg.sampling_minutes) - lag(m + cfg.sampling_minutes))
    for m in cfg.rolling_mean_minutes:
        cols.append(tail(m).mean(axis=1) / scale)
    for m in cfg.rolling_std_minutes:
        cols.append(tail(m).std(axis=1) / scale)
    for m in cfg.rolling_max_minutes:
        cols.append(tail(m).max(axis=1) / scale)
    for m in cfg.rolling_min_minutes:
        cols.append(tail(m).min(axis=1) / scale)
    spd = cfg.steps_per_day
    for d in cfg.same_time_mean_days:
        same_time = windows[:, [-spd * k for k in range(1, d + 1)]]
        cols.append(same_time.mean(axis=1) / scale)

    hour = (target_times.hour + target_times.minute / 60.0).to_numpy(dtype=np.float64)
    dow = target_times.dayofweek.to_numpy()
    calendar = {
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "day_of_week": dow.astype(np.float64),
        "is_weekend": (dow >= 5).astype(np.float64),
        "month": target_times.month.to_numpy(dtype=np.float64),
    }
    for name in cfg.calendar:
        cols.append(calendar[name])

    window_max = windows.max(axis=1)
    depot = {
        # Depot descriptors derived from the depot's own recent history. They play the
        # role of a site identifier but also apply to depots the model has never seen.
        "depot_load_factor": window_mean / np.maximum(window_max, MIN_SCALE_KW),
        "depot_scale_log": np.log10(scale),
    }
    for name in cfg.depot_features:
        cols.append(depot[name])

    return np.column_stack(cols).astype(np.float32), scale


def make_supervised(series: pd.Series, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Turn a gap-free demand series into (X, y_ratio, scale, target_times).

    Row i predicts ``series[i + lookback]`` from ``series[i : i + lookback]``.
    """
    values = series.to_numpy(dtype=np.float64)
    look = cfg.lookback_steps
    if len(values) <= look:
        raise ValueError(f"Need more than {cfg.lookback_days} days of history, got {len(values)} intervals.")
    windows = sliding_window_view(values, look)[:-1]
    times = series.index[look:]
    X, scale = build_features(windows, times, cfg)
    y = values[look:] / scale
    return X, y, scale, times
