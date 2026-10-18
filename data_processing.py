"""Data preparation: raw sessions -> demand series, and client-upload validation.

Canonical processed schema (one row per interval, per depot):

    timestamp   local wall-clock time at the START of the interval (no timezone)
    depot_id    depot / site identifier
    demand_kw   mean electrical demand over the interval, in kW
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import COL_DEMAND, COL_DEPOT, COL_TIME, SAMPLING_MINUTES

UNIT_FACTORS = {"kW": 1.0, "W": 0.001, "MW": 1000.0}
UNIT_CHOICES = ("kW", "W", "MW", "kWh per interval")
MAX_INTERP_STEPS = 4  # gaps up to this many intervals are linearly interpolated
MIN_HISTORY_DAYS = 15  # 14-day model lookback plus one day of margin

TIME_HINTS = ("timestamp", "datetime", "date_time", "date", "time", "period", "interval")
DEMAND_HINTS = ("demand_kw", "demand", "load", "power", "kw", "consumption", "energy", "kwh")
DEPOT_HINTS = ("depot_id", "depot", "site_id", "site", "location", "station")


class DataValidationError(ValueError):
    """Raised with a user-readable message when a dataset cannot be used."""


# =============================================================================
# Raw meter readings -> demand time series (used for the bundled real depots)
# =============================================================================
def meter_readings_to_demand(
    timestamps_utc: pd.Series, load_kw: pd.Series, minutes: int = SAMPLING_MINUTES,
    timezone: str = "Europe/London",
) -> tuple[pd.Series, dict]:
    """Convert irregular meter readings (UTC) into a regular local-time demand series.

    Readings are averaged into ``minutes`` intervals, converted to local wall-clock
    time (the daylight-saving transitions are repaired) and gaps are filled by
    ``regularise``. Returns the series and gap statistics. Use ``observed_mask`` on
    the same inputs to know which intervals were measured rather than filled.
    """
    raw = pd.Series(pd.to_numeric(load_kw, errors="coerce").to_numpy(),
                    index=pd.DatetimeIndex(timestamps_utc)).dropna().sort_index()
    raw = raw.clip(lower=0)
    demand = raw.resample(f"{minutes}min").mean()
    demand.index = demand.index.tz_convert(timezone).tz_localize(None)
    demand = demand[~demand.index.duplicated(keep="first")]
    demand, gaps, filled, longest = regularise(demand, minutes)
    demand = demand.round(3)
    demand.name = COL_DEMAND
    stats = {"readings": int(len(raw)), "gap_count": gaps, "filled_intervals": filled,
             "longest_gap_hours": longest}
    return demand, stats


def observed_mask(timestamps_utc: pd.Series, index: pd.DatetimeIndex, minutes: int = SAMPLING_MINUTES,
                  timezone: str = "Europe/London") -> pd.Series:
    """True for intervals of ``index`` (local time) that contain at least one meter reading."""
    seen = pd.DatetimeIndex(timestamps_utc).floor(f"{minutes}min").unique()
    seen = seen.tz_convert(timezone).tz_localize(None)
    return pd.Series(index.isin(seen), index=index)


def trim_to_full_days(series: pd.Series) -> pd.Series:
    """Drop the partial first and last day."""
    first = series.index.min().ceil("D")
    last = series.index.max().floor("D")
    return series[(series.index >= first) & (series.index < last)]


# =============================================================================
# Processed-data IO
# =============================================================================
def save_processed(series_by_depot: dict[str, pd.Series], path: Path,
                   observed: dict[str, pd.Series] | None = None) -> None:
    """Write the canonical CSV. ``observed`` adds a 1/0 column (1 = measured, 0 = gap-filled)."""
    frames = []
    for depot, series in series_by_depot.items():
        frame = pd.DataFrame({COL_TIME: series.index, COL_DEPOT: depot, COL_DEMAND: series.to_numpy()})
        if observed is not None:
            frame["observed"] = observed[depot].to_numpy().astype(int)
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(path, index=False)


def load_processed(path: Path, column: str = COL_DEMAND) -> dict[str, pd.Series]:
    """Load a canonical processed CSV into ``{depot_id: series}`` (demand by default)."""
    df = pd.read_csv(path, parse_dates=[COL_TIME], dtype={COL_DEPOT: str})
    out = {}
    for depot, group in df.groupby(COL_DEPOT, sort=False):
        out[str(depot)] = group.set_index(COL_TIME)[column].astype(float).sort_index()
    return out


# =============================================================================
# Client upload: read, map columns, validate, harmonise
# =============================================================================
@dataclass
class QualityReport:
    rows_read: int = 0
    rows_used: int = 0
    invalid_timestamps: int = 0
    non_numeric: int = 0
    missing_values: int = 0
    negative_values: int = 0
    duplicate_timestamps: int = 0
    detected_interval_min: float | None = None
    target_interval_min: int = SAMPLING_MINUTES
    resampled: str = "none"
    gap_count: int = 0
    longest_gap_hours: float = 0.0
    filled_points: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    depots: list[str] = field(default_factory=list)
    unit: str = "kW"
    warnings: list[str] = field(default_factory=list)

    def as_table(self) -> pd.DataFrame:
        interval = "n/a" if self.detected_interval_min is None else f"{self.detected_interval_min:g} min"
        rows = [
            ("Rows read", f"{self.rows_read:,}"),
            ("Rows used", f"{self.rows_used:,}"),
            ("Depots found", ", ".join(self.depots) or "n/a"),
            ("Date range", f"{self.start:%d %b %Y %H:%M} to {self.end:%d %b %Y %H:%M}" if self.start is not None else "n/a"),
            ("Detected sampling interval", interval),
            ("Harmonised to", f"{self.target_interval_min} min ({self.resampled})"),
            ("Unit supplied", self.unit),
            ("Invalid timestamps removed", f"{self.invalid_timestamps:,}"),
            ("Non-numeric demand values", f"{self.non_numeric:,}"),
            ("Missing demand values", f"{self.missing_values:,}"),
            ("Negative demand values", f"{self.negative_values:,}"),
            ("Duplicate timestamps", f"{self.duplicate_timestamps:,}"),
            ("Gaps in the series", f"{self.gap_count:,} (longest {self.longest_gap_hours:.1f} h)"),
            ("Intervals filled", f"{self.filled_points:,}"),
        ]
        return pd.DataFrame(rows, columns=["Check", "Result"])


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame of raw columns."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(io.BytesIO(content))
        elif suffix in (".csv", ".txt"):
            df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        else:
            raise DataValidationError(f"Unsupported file type '{suffix}'. Please upload a CSV or Excel file.")
    except DataValidationError:
        raise
    except Exception as exc:  # unreadable / corrupt file
        raise DataValidationError(f"The file could not be read: {exc}") from exc
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty or df.shape[1] < 2:
        raise DataValidationError("The file needs at least a timestamp column and a demand column.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_column(columns: Iterable[str], hints: Iterable[str]) -> str | None:
    cols = list(columns)
    lowered = {c: c.lower().replace(" ", "_") for c in cols}
    for hint in hints:
        for col, low in lowered.items():
            if low == hint:
                return col
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    return None


def guess_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    time_col = guess_column(df.columns, TIME_HINTS)
    rest = [c for c in df.columns if c != time_col]
    depot_col = guess_column(rest, DEPOT_HINTS)
    demand_col = guess_column([c for c in rest if c != depot_col], DEMAND_HINTS)
    unit = "kW"
    if demand_col:
        low = demand_col.lower()
        if "kwh" in low:
            unit = "kWh per interval"
        elif "mw" in low:
            unit = "MW"
        elif low.endswith("_w") or "(w)" in low or "watts" in low:
            unit = "W"
    return {"time": time_col, "demand": demand_col, "depot": depot_col, "unit": unit}


def _detect_interval_minutes(index: pd.DatetimeIndex) -> float:
    diffs = pd.Series(index).diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise DataValidationError("The sampling interval could not be determined (fewer than two valid timestamps).")
    return float(diffs.mode().iloc[0].total_seconds() / 60.0)


def regularise(series: pd.Series, minutes: int) -> tuple[pd.Series, int, int, float]:
    """Put a series on a gap-free ``minutes`` grid.

    Short gaps are interpolated; longer gaps are filled with the typical value
    for that time of week. Returns (series, gap_count, filled_points, longest_gap_h).
    """
    full = pd.date_range(series.index.min(), series.index.max(), freq=f"{minutes}min")
    out = series.reindex(full)
    missing = out.isna()
    filled = int(missing.sum())
    if not filled:
        return out, 0, 0, 0.0
    run_id = (missing != missing.shift()).cumsum()[missing]
    run_lengths = run_id.value_counts()
    out = out.interpolate(limit=MAX_INTERP_STEPS, limit_area="inside")
    # Undo partial interpolation of long gaps, then fill those from the weekly profile.
    long_runs = run_lengths[run_lengths > MAX_INTERP_STEPS].index
    out[run_id[run_id.isin(long_runs)].index] = np.nan
    if out.isna().any():
        key = [out.index.dayofweek, out.index.hour, out.index.minute]
        profile = out.groupby(key).transform("median")
        out = out.fillna(profile).interpolate(limit_direction="both").fillna(0.0)
    return out, int(len(run_lengths)), filled, float(run_lengths.max() * minutes / 60.0)


def validate_and_prepare(
    df: pd.DataFrame,
    time_col: str,
    demand_col: str,
    depot_col: str | None = None,
    unit: str = "kW",
    default_depot: str = "depot",
    target_minutes: int = SAMPLING_MINUTES,
    day_first: bool = False,
    min_days: float = MIN_HISTORY_DAYS,
) -> tuple[dict[str, pd.Series], QualityReport]:
    """Validate a raw upload and return harmonised demand series per depot.

    Raises ``DataValidationError`` with a clear message if the data is unusable.
    """
    report = QualityReport(rows_read=len(df), unit=unit, target_interval_min=target_minutes)
    for col, label in ((time_col, "timestamp"), (demand_col, "demand")):
        if not col or col not in df.columns:
            raise DataValidationError(f"Please choose the {label} column.")
    if time_col == demand_col:
        raise DataValidationError("The timestamp and demand columns must be different.")
    if unit not in UNIT_CHOICES:
        raise DataValidationError(f"Unknown unit '{unit}'.")

    work = pd.DataFrame()
    try:
        ts = pd.to_datetime(df[time_col], errors="coerce", dayfirst=day_first)
    except Exception:
        ts = pd.to_datetime(df[time_col].astype(str), errors="coerce", dayfirst=day_first, utc=True)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    work["ts"] = ts
    raw_demand = df[demand_col]
    work["demand"] = pd.to_numeric(raw_demand, errors="coerce")
    work["depot"] = df[depot_col].astype(str).str.strip() if depot_col and depot_col in df.columns else default_depot

    report.invalid_timestamps = int(work["ts"].isna().sum())
    report.missing_values = int(raw_demand.isna().sum())
    report.non_numeric = int(work["demand"].isna().sum()) - report.missing_values
    if report.invalid_timestamps > 0.5 * len(work):
        raise DataValidationError(
            f"Column '{time_col}' does not look like timestamps: "
            f"{report.invalid_timestamps:,} of {len(work):,} values could not be read as dates."
        )
    if work["demand"].notna().sum() < 0.5 * len(work):
        raise DataValidationError(
            f"Column '{demand_col}' does not look like demand readings: most values are missing or not numeric."
        )
    work = work.dropna(subset=["ts", "demand"])
    report.negative_values = int((work["demand"] < 0).sum())
    work.loc[work["demand"] < 0, "demand"] = 0.0

    series_by_depot: dict[str, pd.Series] = {}
    intervals = []
    for depot, group in work.groupby("depot", sort=True):
        dup = int(group["ts"].duplicated().sum())
        report.duplicate_timestamps += dup
        series = group.groupby("ts")["demand"].mean().sort_index()
        interval = _detect_interval_minutes(series.index)
        intervals.append(interval)

        if unit == "kWh per interval":
            series = series / (interval / 60.0)
        else:
            series = series * UNIT_FACTORS[unit]

        coarser = interval > target_minutes
        grid_minutes = max(int(round(interval)), 1) if coarser else target_minutes
        if interval < target_minutes:
            report.resampled = "averaged from finer readings"
        # Snap readings onto the grid; empty bins stay missing and are handled below.
        series = series.resample(f"{grid_minutes}min").mean()
        series, gaps, filled, longest = regularise(series, grid_minutes)
        if coarser:
            # Coarser data: hold each reading across its interval.
            end = series.index.max() + pd.Timedelta(minutes=grid_minutes - target_minutes)
            fine = pd.date_range(series.index.min().floor(f"{target_minutes}min"), end, freq=f"{target_minutes}min")
            series = series.reindex(fine, method="ffill").bfill()
            report.resampled = "coarser readings held across each interval"
            report.warnings.append(
                f"Data is sampled every {interval:g} min but the model works at {target_minutes} min. "
                "Short peaks inside each interval cannot be recovered, so peak demand may be understated."
            )
        report.gap_count += gaps
        report.filled_points += filled
        report.longest_gap_hours = max(report.longest_gap_hours, longest)

        span_days = (series.index.max() - series.index.min()) / pd.Timedelta(days=1)
        if span_days < min_days:
            raise DataValidationError(
                f"Depot '{depot}' has only {span_days:.1f} days of data. "
                f"At least {min_days:g} days of continuous history are needed to forecast."
            )
        series.name = COL_DEMAND
        series_by_depot[str(depot)] = series.astype(float)

    if not series_by_depot:
        raise DataValidationError("No usable rows were found after validation.")

    report.detected_interval_min = float(np.median(intervals))
    report.depots = list(series_by_depot)
    report.rows_used = int(sum(len(s) for s in series_by_depot.values()))
    report.start = min(s.index.min() for s in series_by_depot.values())
    report.end = max(s.index.max() for s in series_by_depot.values())

    # Soft checks -> warnings shown in the quality summary.
    all_values = pd.concat(series_by_depot.values())
    peak = float(all_values.max())
    if peak <= 0:
        raise DataValidationError("All demand values are zero, so there is nothing to forecast.")
    if unit == "kW" and peak > 50_000:
        report.warnings.append(f"Peak demand is {peak:,.0f} kW, which is very high for a depot. Check the unit (values may be in W).")
    if unit == "kW" and peak < 1:
        report.warnings.append(f"Peak demand is only {peak:.3f} kW. Check the unit (values may be in MW).")
    if report.negative_values:
        report.warnings.append(f"{report.negative_values:,} negative readings were set to 0 kW.")
    if report.duplicate_timestamps:
        report.warnings.append(f"{report.duplicate_timestamps:,} duplicate timestamps were averaged.")
    if report.non_numeric:
        report.warnings.append(f"{report.non_numeric:,} non-numeric demand values were ignored.")
    if report.filled_points:
        share = report.filled_points / max(report.rows_used, 1)
        report.warnings.append(
            f"{report.filled_points:,} missing intervals ({share:.1%}) were filled: short gaps by interpolation, "
            "longer gaps from the typical value for that time of week."
        )
    return series_by_depot, report


def merge_readings(existing: pd.Series, new: pd.Series, minutes: int = SAMPLING_MINUTES) -> tuple[pd.Series, dict]:
    """Add newly uploaded readings to a depot's stored series.

    Where both have a value for the same interval the new reading wins (a correction).
    The result is put back on a gap-free grid. Returns (series, summary).
    """
    combined = pd.concat([existing[~existing.index.isin(new.index)], new]).sort_index()
    merged, gaps, filled, longest = regularise(combined, minutes)
    merged.name = COL_DEMAND
    summary = {
        "added": int((~new.index.isin(existing.index)).sum()),
        "replaced": int(new.index.isin(existing.index).sum()),
        "filled": filled,
        "longest_gap_hours": longest,
        "extended_to": merged.index.max(),
        "previous_end": existing.index.max(),
    }
    return merged.astype(float), summary
