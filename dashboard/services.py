"""Glue between the Streamlit pages and the forecasting code (cached, no training)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from dashboard import database
from dashboard import theme as ui
from data_processing import load_processed
from evaluation import rolling_origin_backtest
from forecasting import ForecastError, RiskAssessment, assess_risk, forecast_series
from model_loader import ModelBundle, ModelPackageError, load_bundle

PROCESSED_PATH = config.PROCESSED_DIR / "depot_demand.csv"


class NoDepotsError(ForecastError):
    """The organisation has no depots yet."""


# ----------------------------------------------------------------- signed-in user
def current_user() -> dict:
    return st.session_state.get("user") or {}


def org_id() -> int:
    return int(current_user()["org_id"])


def user_email() -> str | None:
    return current_user().get("email")


def is_admin() -> bool:
    """Administrator of the signed-in user's own organisation."""
    return current_user().get("role") == "admin"


def is_owner() -> bool:
    """Platform owner (the operator of the service)."""
    return bool(current_user().get("platform_admin"))


# ----------------------------------------------------------------------- model
def model_signature() -> float:
    """Changes whenever the active model package changes, which invalidates the caches."""
    files = [config.ACTIVE_MODEL_DIR / f for f in config.REQUIRED_PACKAGE_FILES]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


@st.cache_resource(show_spinner=False)
def _bundle(signature: float) -> ModelBundle:
    return load_bundle(config.ACTIVE_MODEL_DIR)


def get_bundle() -> ModelBundle:
    return _bundle(model_signature())


# ------------------------------------------------------------------------ data
def depot_path(depot: dict) -> Path:
    """Where a depot's demand series is stored (bundled sample data or the organisation's upload)."""
    if depot["source"] == "bundled":
        return PROCESSED_PATH
    return config.UPLOAD_DIR / depot["data_file"]


def upload_file_for(org: int, depot_key: str) -> str:
    """Relative path (under the upload folder) for an organisation's uploaded depot."""
    return f"{int(org)}/{depot_key}.csv"


@st.cache_data(show_spinner=False)
def _series(source: str, data_id: str, path: str, mtime: float) -> pd.Series:
    data = load_processed(Path(path))
    return data[data_id] if source == "bundled" else next(iter(data.values()))


def _series_args(depot: dict) -> tuple[str, str, str, float]:
    path = depot_path(depot)
    if not path.is_file():
        raise ForecastError(f"The data file for {depot['name']} is missing.")
    return depot["source"], depot["data_id"], str(path), path.stat().st_mtime


def get_series(depot: dict) -> pd.Series:
    return _series(*_series_args(depot))


# -------------------------------------------------------------------- forecasts
@st.cache_data(show_spinner=False)
def _forecast(source: str, data_id: str, path: str, mtime: float, signature: float, horizon_hours: int) -> pd.DataFrame:
    bundle = _bundle(signature)
    return forecast_series(bundle.model, bundle.cfg, _series(source, data_id, path, mtime), horizon_hours)


def get_forecast(depot: dict, horizon_hours: int) -> pd.DataFrame:
    return _forecast(*_series_args(depot), model_signature(), int(horizon_hours))


@st.cache_data(show_spinner=False)
def _replay(source: str, data_id: str, path: str, mtime: float, signature: float, days: int) -> pd.DataFrame:
    bundle = _bundle(signature)
    series = _series(source, data_id, path, mtime)
    end = series.index[-1] + pd.Timedelta(minutes=bundle.cfg.sampling_minutes)
    start = end.normalize() - pd.Timedelta(days=days)
    return rolling_origin_backtest(bundle.model, bundle.cfg, series, start, end,
                                   config.EVAL_HORIZON_HOURS, config.EVAL_ORIGIN_HOUR)


def get_replay(depot: dict, days: int = config.REPLAY_DAYS) -> pd.DataFrame:
    """Day-ahead forecasts replayed over the most recent days, with the actual outcomes."""
    return _replay(*_series_args(depot), model_signature(), int(days))


# ---------------------------------------------------------------------- context
@dataclass
class Context:
    settings: dict
    org_id: int
    depot_key: str
    depot: dict
    bundle: ModelBundle
    series: pd.Series
    forecast: pd.DataFrame
    risk: RiskAssessment
    capacity_kw: float
    alpha: float
    horizon_hours: int
    issued_at: pd.Timestamp       # first forecast interval = end of the observed data
    current_kw: float
    generated_at: datetime        # wall-clock time this forecast was produced
    step: pd.Timedelta

    @property
    def organisation(self) -> str:
        return self.settings["organisation"]

    @property
    def source_label(self) -> str:
        return self.depot.get("data_label", "Real data")

    def chips(self, kind: str = "Forecast") -> list[tuple[str, str]]:
        return [("Depot", self.depot["name"]), ("Data", f"Real data + {kind}"), ("Model", f"v{self.bundle.version}")]


def get_settings() -> dict:
    """The signed-in organisation's settings (loaded once per session)."""
    if "settings" not in st.session_state:
        try:
            metadata = get_bundle().metadata
        except ModelPackageError:
            metadata = None
        st.session_state["settings"] = database.load_settings(org_id(), metadata)
    return st.session_state["settings"]


def save_settings() -> None:
    database.save_settings(org_id(), st.session_state["settings"])


def audit(action: str, detail: str = "") -> None:
    database.audit(org_id(), user_email(), action, detail)


def depot_risk(settings: dict, depot_key: str, horizon_hours: int, bundle: ModelBundle):
    """Forecast and risk for one depot. Returns (series, forecast, risk, issued_at)."""
    depot = settings["depots"][depot_key]
    series = get_series(depot)
    forecast = get_forecast(depot, horizon_hours)
    issued_at = forecast["timestamp"].iloc[0]
    risk = assess_risk(forecast["timestamp"], forecast["forecast_kw"], float(depot["capacity_kw"]),
                       database.depot_alpha(settings, depot_key), issued_at)
    return series, forecast, risk, issued_at


def build_context(horizon_hours: int | None = None) -> Context:
    """Everything a page needs for the selected depot. Logs the forecast once."""
    settings = get_settings()
    if not settings["depots"] or settings["selected_depot"] not in settings["depots"]:
        raise NoDepotsError("No depots yet.")
    bundle = get_bundle()
    key = settings["selected_depot"]
    depot = settings["depots"][key]
    horizon = int(horizon_hours or settings["horizon_hours"])
    series, forecast, risk, issued_at = depot_risk(settings, key, horizon, bundle)
    current_kw = float(series.iloc[-1])

    stamp_key = (org_id(), key, horizon, model_signature(), str(issued_at))
    stamps = st.session_state.setdefault("forecast_stamps", {})
    if stamp_key not in stamps:
        stamps[stamp_key] = datetime.now()
    run_id = database.log_forecast_run(org_id(), depot["name"], issued_at, horizon, bundle.version, risk, current_kw,
                                       forecast, user_email())
    if run_id is not None and horizon == int(settings["horizon_hours"]):
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = forecast.assign(depot=depot["name"], model_version=bundle.version,
                              capacity_kw=risk.capacity_kw, warning_kw=risk.warning_kw)
        out.round({"forecast_kw": 3, "baseline_kw": 3}).to_csv(config.LATEST_FORECAST_PATH, index=False)
    return Context(
        settings=settings, org_id=org_id(), depot_key=key, depot=depot, bundle=bundle, series=series,
        forecast=forecast, risk=risk, capacity_kw=float(depot["capacity_kw"]),
        alpha=database.depot_alpha(settings, key), horizon_hours=horizon, issued_at=issued_at,
        current_kw=current_kw, generated_at=stamps[stamp_key],
        step=pd.Timedelta(minutes=bundle.cfg.sampling_minutes),
    )


def fleet_table(settings: dict, bundle: ModelBundle, horizon_hours: int) -> pd.DataFrame:
    """Status of every depot, for the overview and the alerts screen."""
    rows = []
    for key, depot in settings["depots"].items():
        try:
            series, forecast, risk, issued_at = depot_risk(settings, key, horizon_hours, bundle)
        except (ForecastError, ValueError, KeyError) as exc:
            rows.append({"key": key, "Depot": depot["name"], "Status": "Unavailable", "error": str(exc)})
            continue
        rows.append({
            "key": key, "Depot": depot["name"], "Status": risk.status,
            "Current (kW)": float(series.iloc[-1]), "Predicted peak (kW)": risk.peak_kw,
            "Capacity (kW)": risk.capacity_kw, "Peak utilisation": risk.peak_utilisation,
            "Peak time": risk.peak_time, "Breach time": risk.breach_time, "Warning time": risk.warning_time,
            "Lead time": risk.lead_time, "Issued": issued_at,
        })
    return pd.DataFrame(rows, columns=["key", "Depot", "Status", "Current (kW)", "Predicted peak (kW)",
                                       "Capacity (kW)", "Peak utilisation", "Peak time", "Breach time",
                                       "Warning time", "Lead time", "Issued", "error"])


def safe_context(horizon_hours: int | None = None, title: str | None = None) -> Context | None:
    """``build_context`` with user-facing error handling. Returns None after showing a message."""
    try:
        return build_context(horizon_hours)
    except NoDepotsError:
        if title:
            ui.page_header(title, "Add a depot to get started", get_settings()["organisation"])
        ui.empty_state("No depots yet",
                       "Open the Depots page to upload your depot's meter data or to load the sample depots.")
    except ModelPackageError as exc:
        st.error(f"The forecasting model could not be loaded. {exc}")
    except ForecastError as exc:
        st.error(str(exc))
    except Exception as exc:  # never show a traceback in the customer UI
        st.error(f"The forecast could not be produced for this depot ({type(exc).__name__}). "
                 "Check the depot's data on the Depots page.")
    return None
