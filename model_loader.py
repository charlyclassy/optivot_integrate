"""Model package loading, validation, replacement and rollback.

A model package is a folder (or a .zip of it) containing:

    xgboost_model.json     trained XGBoost regressor (native JSON format)
    model_metadata.json    name, version, training date, data and split description
    feature_config.json    feature settings + ordered feature_names
    model_metrics.csv      baseline vs XGBoost metrics from training
    test_predictions.csv   (optional) held-out predictions for the performance charts

The active package lives in ``models/global_model``. Activating a replacement
moves the current one to ``models/archive`` so it can be rolled back.
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (ACTIVE_MODEL_DIR, FEATURE_CONFIG_FILE, METADATA_FILE, METRICS_FILE,
                    MODEL_ARCHIVE_DIR, MODEL_FILE, REQUIRED_PACKAGE_FILES, SUPPORTED_MODEL_TYPES,
                    SUPPORTED_TARGETS, TEST_PREDICTIONS_FILE)
from feature_engineering import FeatureConfig, build_features

REQUIRED_METADATA_KEYS = ("model_name", "model_version", "model_type", "target", "sampling_minutes", "trained_at")
REQUIRED_METRIC_COLUMNS = ("model", "mae_kw", "rmse_kw")


class ModelPackageError(ValueError):
    """Raised with a user-readable message when a model package is unusable."""


@dataclass
class ModelBundle:
    model: xgb.XGBRegressor
    cfg: FeatureConfig
    metadata: dict
    metrics: pd.DataFrame
    test_predictions: pd.DataFrame | None
    path: Path

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def name(self) -> str:
        return str(self.metadata.get("model_name", "XGBoost model"))

    @property
    def trained_at(self) -> str:
        return str(self.metadata.get("trained_at", ""))[:10]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


# --------------------------------------------------------------------- validation
def validate_package(folder: Path) -> list[Check]:
    """Run every compatibility check on a package folder. Never raises."""
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str) -> bool:
        checks.append(Check(name, passed, detail))
        return passed

    folder = Path(folder)
    missing = [f for f in REQUIRED_PACKAGE_FILES if not (folder / f).is_file()]
    if not add("Required files", not missing,
               "All four package files are present." if not missing else f"Missing: {', '.join(missing)}."):
        return checks

    try:
        metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
        absent = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
        if absent:
            raise ValueError(f"missing keys: {', '.join(absent)}")
        add("Model metadata", True, f"{metadata['model_name']} version {metadata['model_version']}, "
                                    f"trained {str(metadata['trained_at'])[:10]}.")
    except Exception as exc:
        add("Model metadata", False, f"model_metadata.json is not valid: {exc}.")
        return checks

    add("Model type", metadata["model_type"] in SUPPORTED_MODEL_TYPES,
        f"'{metadata['model_type']}' (supported: {', '.join(SUPPORTED_MODEL_TYPES)}).")
    add("Target definition", metadata["target"] in SUPPORTED_TARGETS,
        f"'{metadata['target']}' (supported: {', '.join(SUPPORTED_TARGETS)}).")

    cfg = None
    try:
        cfg = FeatureConfig.from_dict(json.loads((folder / FEATURE_CONFIG_FILE).read_text(encoding="utf-8")))
        add("Feature configuration", True,
            f"{len(cfg.feature_names)} features; names and order match the feature settings.")
    except Exception as exc:
        add("Feature configuration", False, f"{exc}")

    if cfg is not None:
        same_interval = int(metadata["sampling_minutes"]) == cfg.sampling_minutes
        add("Sampling interval", same_interval,
            f"{cfg.sampling_minutes} min." if same_interval else
            f"Metadata says {metadata['sampling_minutes']} min but feature_config says {cfg.sampling_minutes} min.")
        add("Target consistency", metadata["target"] == cfg.target,
            "Metadata and feature configuration agree on the target." if metadata["target"] == cfg.target else
            f"Metadata target '{metadata['target']}' differs from feature_config target '{cfg.target}'.")

    try:
        model = xgb.XGBRegressor()
        model.load_model(folder / MODEL_FILE)
        booster = model.get_booster()
        add("XGBoost model file", True, f"Loaded; {booster.num_boosted_rounds()} boosted rounds.")
    except Exception as exc:
        add("XGBoost model file", False, f"xgboost_model.json could not be loaded: {exc}.")
        return checks

    if cfg is not None:
        expected = list(cfg.feature_names)
        n_ok = booster.num_features() == len(expected)
        add("Feature count", n_ok, f"Model expects {booster.num_features()} features; configuration lists {len(expected)}.")
        if booster.feature_names is not None:
            same = list(booster.feature_names) == expected
            add("Feature names and order", same,
                "Model feature names match the configuration, in order." if same else
                "Feature names stored in the model differ from feature_config.json (names or order).")
        if n_ok:
            try:
                ramp = np.linspace(1.0, 20.0, cfg.lookback_steps)[None, :]
                X, _ = build_features(ramp, pd.DatetimeIndex(["2024-01-01 00:00"]), cfg)
                out = float(model.predict(X)[0])
                add("Test prediction", bool(np.isfinite(out)), "The model returns a finite prediction.")
            except Exception as exc:
                add("Test prediction", False, f"Prediction failed: {exc}.")

    try:
        metrics = pd.read_csv(folder / METRICS_FILE)
        absent = [c for c in REQUIRED_METRIC_COLUMNS if c not in metrics.columns]
        add("Model metrics", not absent and not metrics.empty,
            f"{len(metrics)} rows." if not absent else f"model_metrics.csv is missing columns: {', '.join(absent)}.")
    except Exception as exc:
        add("Model metrics", False, f"model_metrics.csv could not be read: {exc}.")
    return checks


def is_valid(checks: list[Check]) -> bool:
    return bool(checks) and all(c.passed for c in checks)


# ------------------------------------------------------------------------ loading
def load_bundle(folder: Path = ACTIVE_MODEL_DIR) -> ModelBundle:
    """Load a saved model package for inference. No training happens here."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ModelPackageError(f"No model package found at {folder}. Run: python scripts/train_model.py")
    checks = validate_package(folder)
    if not is_valid(checks):
        failed = "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.passed)
        raise ModelPackageError(f"The model package at {folder} is not usable. {failed}")
    model = xgb.XGBRegressor()
    model.load_model(folder / MODEL_FILE)
    predictions_path = folder / TEST_PREDICTIONS_FILE
    predictions = None
    if predictions_path.is_file():
        predictions = pd.read_csv(predictions_path, parse_dates=["origin", "timestamp"], dtype={"depot_id": str})
    return ModelBundle(
        model=model,
        cfg=FeatureConfig.from_dict(json.loads((folder / FEATURE_CONFIG_FILE).read_text(encoding="utf-8"))),
        metadata=json.loads((folder / METADATA_FILE).read_text(encoding="utf-8")),
        metrics=pd.read_csv(folder / METRICS_FILE),
        test_predictions=predictions,
        path=folder,
    )


# ---------------------------------------------------------- replacement / rollback
def extract_package(zip_bytes: bytes, destination: Path) -> Path:
    """Unpack an uploaded .zip into ``destination`` (flat). Only known files are kept."""
    wanted = set(REQUIRED_PACKAGE_FILES) | {TEST_PREDICTIONS_FILE}
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ModelPackageError("The uploaded file is not a valid .zip archive.") from exc
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with archive:
        for info in archive.infolist():
            name = Path(info.filename).name  # ignore any folder structure inside the zip
            if info.is_dir() or name not in wanted:
                continue
            (destination / name).write_bytes(archive.read(info))
    return destination


def package_zip_bytes(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(folder).iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    return buffer.getvalue()


def _archive_name(folder: Path) -> str:
    try:
        version = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8")).get("model_version", "unknown")
    except Exception:
        version = "unknown"
    safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in str(version))
    return f"v{safe}__archived_{datetime.now():%Y%m%d_%H%M%S_%f}"


def list_archive(archive_dir: Path = MODEL_ARCHIVE_DIR) -> list[dict]:
    """Archived packages, most recently archived first."""
    items = []
    if not archive_dir.is_dir():
        return items
    # The archive time is part of the folder name; a moved folder keeps its old modified time.
    folders = [f for f in archive_dir.iterdir() if f.is_dir() and "__archived_" in f.name]
    for folder in sorted(folders, key=lambda f: f.name.split("__archived_")[-1], reverse=True):
        try:
            meta = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        items.append({
            "folder": folder,
            "model_name": meta.get("model_name", "unknown"),
            "model_version": meta.get("model_version", "unknown"),
            "trained_at": str(meta.get("trained_at", ""))[:10],
            "archived": folder.name.split("__archived_")[-1],
        })
    return items


def activate_package(source: Path, active_dir: Path = ACTIVE_MODEL_DIR,
                     archive_dir: Path = MODEL_ARCHIVE_DIR) -> Path | None:
    """Make ``source`` the active model. The current model is archived, not deleted.

    Raises ``ModelPackageError`` (and changes nothing) if ``source`` is incompatible.
    Returns the archive folder of the previous model, if there was one.
    """
    checks = validate_package(source)
    if not is_valid(checks):
        failed = "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.passed)
        raise ModelPackageError(f"Model package rejected. {failed}")
    archived = None
    if active_dir.is_dir():
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / _archive_name(active_dir)
        shutil.move(str(active_dir), str(archived))
    shutil.copytree(source, active_dir)
    return archived


def rollback(archived_folder: Path, active_dir: Path = ACTIVE_MODEL_DIR,
             archive_dir: Path = MODEL_ARCHIVE_DIR) -> Path | None:
    """Re-activate an archived model; the model it replaces is archived in turn."""
    staging = archive_dir / "_rollback_staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(archived_folder, staging)
    try:
        previous = activate_package(staging, active_dir, archive_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(archived_folder, ignore_errors=True)
    return previous
