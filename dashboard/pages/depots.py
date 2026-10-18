"""Depots: depot list, per-depot setup, data upload and adding new readings."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from dashboard import database as storage
from dashboard import services
from dashboard import theme as ui
from dashboard.pages import shared as common
from data_processing import (UNIT_CHOICES, DataValidationError, guess_mapping, merge_readings, read_table,
                             save_processed, validate_and_prepare)

NONE = "(none)"


def _select(settings: dict, key: str | None) -> None:
    """Make ``key`` the selected depot; the sidebar selector picks it up on the next run."""
    settings["selected_depot"] = key
    st.session_state["pending_depot"] = key


def _uploader_key(name: str) -> str:
    """Uploader keys carry a counter so a finished upload can be cleared."""
    return f"{name}_{st.session_state.get('upload_round', 0)}"


def _clear_uploaders() -> None:
    st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1


def _mapping_form(raw: pd.DataFrame, prefix: str):
    """Column mapping controls shared by both upload flows."""
    guess = guess_mapping(raw)
    columns = list(raw.columns)
    with st.container(border=True):
        st.markdown("**Column mapping**")
        c1, c2, c3, c4 = st.columns(4)
        time_col = c1.selectbox("Timestamp column", columns, key=f"{prefix}_time",
                                index=columns.index(guess["time"]) if guess["time"] in columns else 0)
        demand_col = c2.selectbox("Demand column", columns, key=f"{prefix}_demand",
                                  index=columns.index(guess["demand"]) if guess["demand"] in columns else min(1, len(columns) - 1))
        depot_options = [NONE, *columns]
        depot_col = c3.selectbox("Depot identifier (optional)", depot_options, key=f"{prefix}_depot",
                                 index=depot_options.index(guess["depot"]) if guess["depot"] in columns else 0)
        unit = c4.selectbox("Demand unit", list(UNIT_CHOICES), index=list(UNIT_CHOICES).index(guess["unit"]),
                            key=f"{prefix}_unit")
        day_first = st.checkbox("Dates are written day first (DD/MM/YYYY)", value=False, key=f"{prefix}_dayfirst")
        st.dataframe(raw.head(5), hide_index=True, width="stretch")
    return time_col, demand_col, None if depot_col == NONE else depot_col, unit, day_first


def _depot_setup(settings: dict) -> None:
    key = settings["selected_depot"]
    depot = settings["depots"][key]
    ui.section(f"Depot setup: {depot['name']}")
    try:
        series = services.get_series(depot)
        data_text = f"{len(series):,} intervals from {series.index.min():%d %b %Y} to {series.index.max():%d %b %Y %H:%M}"
    except Exception:
        data_text = "Data file not available"
    admin = services.is_admin()
    with st.form("depot_setup", border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("Depot name", value=depot["name"], max_chars=60)
        capacity = c2.number_input("Grid capacity (kW)", min_value=1.0, value=float(depot["capacity_kw"]), step=1.0,
                                   help="Capacity available to this depot. Forecasts at or above it are a Breach.")
        own_alpha = depot.get("warning_alpha")
        alpha_pct = c3.number_input("Warning threshold (% of capacity)", min_value=50, max_value=99,
                                    value=int(round((own_alpha or settings["warning_alpha"]) * 100)), step=1,
                                    help="Forecasts at or above this share of capacity raise a Warning.")
        st.caption(f"Data source: {depot.get('data_label', 'Real data')}. {data_text}."
                   + (f" Recorded site connection: {depot['connection_kva']:,.0f} kVA." if depot.get("connection_kva") else "")
                   + (f" Vehicles: {depot['vehicles']}." if depot.get("vehicles") else ""))
        saved = st.form_submit_button("Save depot", type="primary", icon=":material/save:", disabled=not admin)
    if not admin:
        st.caption("Sign in as an administrator to change depot settings.")
        return
    if saved:
        if not name.strip():
            st.error("The depot name cannot be empty.", icon=":material/error:")
            return
        depot["name"] = name.strip()
        depot["capacity_kw"] = float(capacity)
        depot["warning_alpha"] = None if abs(alpha_pct / 100 - settings["warning_alpha"]) < 1e-9 else alpha_pct / 100
        services.save_settings()
        services.audit("Depot updated", f"{depot['name']}: capacity {capacity:g} kW, warning {alpha_pct}%")
        st.success(f"{depot['name']} saved.", icon=":material/check_circle:")
        st.rerun()
    with st.popover("Remove this depot", icon=":material/delete:"):
        st.write(f"Remove **{depot['name']}** from your organisation?"
                 + (" Its uploaded data will be deleted." if depot["source"] == "upload" else " Sample depots can be restored later."))
        if st.button("Remove depot", type="primary"):
            if depot["source"] == "upload":
                services.depot_path(depot).unlink(missing_ok=True)
            del settings["depots"][key]
            _select(settings, next(iter(settings["depots"]), None))
            services.save_settings()
            services.audit("Depot removed", depot["name"])
            st.rerun()


def _add_readings(settings: dict) -> None:
    """Keep an uploaded depot current by adding its latest meter readings."""
    key = settings["selected_depot"]
    depot = settings["depots"][key]
    if depot["source"] != "upload" or not services.is_admin():
        return
    ui.section(f"Add new readings: {depot['name']}")
    ui.note("Upload the latest meter readings for this depot. They are appended to the stored history; readings for "
            "intervals that already exist replace the old values. The forecast then starts from the new end of the data.", title="How adding readings works")
    uploaded = st.file_uploader("New readings", type=["csv", "xlsx", "xls"], key=_uploader_key("readings_upload"),
                                label_visibility="collapsed")
    if uploaded is None:
        return
    try:
        raw = read_table(uploaded.getvalue(), uploaded.name)
        time_col, demand_col, depot_col, unit, day_first = _mapping_form(raw, f"append_{uploaded.file_id}")
        prepared, report = validate_and_prepare(raw, time_col, demand_col, None, unit, default_depot=depot["name"],
                                                day_first=day_first, min_days=0)
        new = next(iter(prepared.values()))
        existing = services.get_series(depot)
        if new.index.max() < existing.index.min():
            raise DataValidationError("These readings are older than the depot's stored history.")
        merged, summary = merge_readings(existing, new)
    except DataValidationError as exc:
        st.error(f"This file cannot be used. {exc}", icon=":material/error:")
        return
    except Exception as exc:
        st.error(f"This file could not be validated ({type(exc).__name__}). Check the column mapping and try again.",
                 icon=":material/error:")
        return
    ui.cards([
        ui.card("New intervals", f"{summary['added']:,}", "", "Added to the history", icon_name="upload"),
        ui.card("Replaced intervals", f"{summary['replaced']:,}", "", "Existing values updated", icon_name="layers"),
        ui.card("Data will run to", f"{summary['extended_to']:%d %b %Y %H:%M}", "",
                f"Currently {summary['previous_end']:%d %b %Y %H:%M}", icon_name="calendar"),
        ui.card("Gaps filled", f"{summary['filled']:,}", "", f"Longest gap {summary['longest_gap_hours']:.1f} h",
                "warning" if summary["longest_gap_hours"] > 24 else "neutral", "alert"),
    ])
    for warning in report.warnings:
        st.warning(warning, icon=":material/warning:")
    if st.button("Add these readings", type="primary", icon=":material/add:"):
        save_processed({depot["name"]: merged}, services.depot_path(depot))
        services.audit("Readings added", f"{depot['name']}: {summary['added']} new, {summary['replaced']} replaced")
        _clear_uploaders()
        st.session_state["depot_notice"] = (f"{summary['added']:,} new intervals added to {depot['name']}. "
                                            f"Data now runs to {summary['extended_to']:%d %b %Y %H:%M}.")
        st.rerun()


def _upload(settings: dict, lookback_days: int) -> None:
    ui.section("Upload Depot Data")
    ui.note("Add a depot from your own metered data. Upload a CSV or Excel file with a timestamp column and a demand "
            f"column (kW), optionally a depot identifier. At least {lookback_days + 1} days of history are needed. "
            "The data is validated before it is used; nothing is trained. Only your organisation can see it.", title="What file do I need?")
    if not services.is_admin():
        st.info("Sign in as an administrator to upload depot data.", icon=":material/lock:")
        return
    sample = config.SAMPLE_DATA_PATH
    if sample.is_file():
        st.download_button("Download example file", sample.read_bytes(), file_name="sample_processed_data.csv",
                           mime="text/csv", icon=":material/download:")
    uploaded = st.file_uploader("Client data file", type=["csv", "xlsx", "xls"], key=_uploader_key("depot_upload"),
                                label_visibility="collapsed")
    if uploaded is None:
        return
    try:
        raw = read_table(uploaded.getvalue(), uploaded.name)
    except DataValidationError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    time_col, demand_col, depot_col, unit, day_first = _mapping_form(raw, f"new_{uploaded.file_id}")
    try:
        prepared, report = validate_and_prepare(
            raw, time_col, demand_col, depot_col, unit, default_depot=uploaded.name.rsplit(".", 1)[0],
            day_first=day_first, min_days=lookback_days + 1)
    except DataValidationError as exc:
        st.error(f"This file cannot be used. {exc}", icon=":material/error:")
        return
    except Exception as exc:
        st.error(f"This file could not be validated ({type(exc).__name__}). Check the column mapping and try again.",
                 icon=":material/error:")
        return

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Data-quality summary**")
        st.dataframe(report.as_table(), hide_index=True, width="stretch", height=38 + 35 * 14)
    with right:
        if report.warnings:
            for warning in report.warnings:
                st.warning(warning, icon=":material/warning:")
        else:
            st.success("All checks passed. The data is ready to use.", icon=":material/check_circle:")
        st.markdown("**Add to your depots**")
        with st.form("add_depot", border=False):
            entries = {}
            for depot_id, series in prepared.items():
                a, b = st.columns([2, 1])
                default_capacity = float(round(series.resample("D").max().quantile(config.DEFAULT_CAPACITY_QUANTILE)))
                entries[depot_id] = (
                    a.text_input("Depot name", value=str(depot_id).replace("_", " ").title(), key=f"name_{depot_id}"),
                    b.number_input("Grid capacity (kW)", min_value=1.0, value=max(default_capacity, 1.0), step=1.0,
                                   key=f"cap_{depot_id}"),
                )
            add = st.form_submit_button("Add depot" if len(prepared) == 1 else f"Add {len(prepared)} depots",
                                        type="primary", icon=":material/add:")
        if add:
            last_key = None
            for depot_id, (name, capacity) in entries.items():
                name = name.strip() or str(depot_id)
                key = storage.slugify(name)
                while key in settings["depots"]:
                    key += "_2"
                data_file = services.upload_file_for(services.org_id(), key)
                save_processed({name: prepared[depot_id]}, config.UPLOAD_DIR / data_file)
                settings["depots"][key] = {
                    "name": name, "data_id": key, "data_file": data_file, "source": "upload",
                    "data_label": f"Real data: your upload ({uploaded.name})",
                    "capacity_kw": float(capacity), "warning_alpha": None, "vehicles": None, "connection_kva": None,
                }
                services.audit("Depot added", f"{name} from {uploaded.name}, {len(prepared[depot_id]):,} intervals")
                last_key = key
            _select(settings, last_key)
            _clear_uploaders()
            services.save_settings()
            st.session_state["depot_notice"] = "Depot added. It is now the selected depot."
            st.rerun()


def _sample_depots(settings: dict, metadata: dict | None) -> None:
    """Offer the bundled sample depots when some (or all) are not in the workspace."""
    samples = storage.sample_depots(metadata)
    missing = {k: v for k, v in samples.items() if k not in settings["depots"]}
    if not missing or not services.is_admin():
        return
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.markdown(f'<div class="dw-note">{len(missing)} sample depot(s) with real Royal Mail charging data are available '
                  "to explore the product before you upload your own data.</div>", unsafe_allow_html=True)
    if right.button("Load sample depots", icon=":material/dataset:", width="stretch"):
        settings["depots"].update(missing)
        if settings["selected_depot"] not in settings["depots"]:
            _select(settings, next(iter(settings["depots"])))
        services.save_settings()
        services.audit("Sample depots loaded", f"{len(missing)} depot(s)")
        st.rerun()


def render() -> None:
    settings = services.get_settings()
    try:
        bundle = services.get_bundle()
    except Exception as exc:
        bundle = None
        st.error(f"The forecasting model could not be loaded. {exc}")
    selected = settings["depots"].get(settings["selected_depot"] or "")
    chips = [("Depots", str(len(settings["depots"])))]
    if selected:
        chips.insert(0, ("Depot", selected["name"]))
    ui.page_header("Depots", "Depot list, capacity settings and data upload", settings["organisation"], chips)
    notice = st.session_state.pop("depot_notice", None)
    if notice:
        st.success(notice, icon=":material/check_circle:")

    if not settings["depots"]:
        ui.empty_state("No depots yet", "Upload your depot's meter data below, or load the sample depots to explore.")
    elif bundle is not None:
        ui.section("Your depots")
        with st.spinner("Updating depot forecasts"):
            fleet = services.fleet_table(settings, bundle, int(settings["horizon_hours"]))
        common.fleet_dataframe(fleet)
        sources = pd.DataFrame([
            {"Depot": d["name"], "Data source": d.get("data_label", ""), "Vehicles": d.get("vehicles"),
             "Recorded connection (kVA)": d.get("connection_kva")}
            for d in settings["depots"].values()
        ])
        with st.expander("Data sources"):
            st.dataframe(sources, hide_index=True, width="stretch")
            ui.note("Grid capacity is the limit used for warnings in this application and is set per depot. For the "
                    "sample depots it defaults to a planning limit derived from historic charging peaks; the recorded "
                    "site connection also serves the building load, so it is shown for reference only.")
    _sample_depots(settings, bundle.metadata if bundle else None)
    if selected:
        st.write("")
        _depot_setup(settings)
        _add_readings(settings)
    st.write("")
    _upload(settings, bundle.cfg.lookback_days if bundle else 14)
