"""Platform: the service owner's console (customers, sign-up, the shared model, activity).

Only the platform owner sees this page. Customer administrators manage their own
organisation under Settings and cannot change anything here.
"""
from __future__ import annotations

import shutil

import pandas as pd
import streamlit as st

import config
import model_loader
from dashboard import database as storage
from dashboard import services
from dashboard import theme as ui
from evaluation import fmt_pct


def _model(bundle, admin: bool) -> None:
    meta = bundle.metadata
    pooled = bundle.metrics
    if "scope" in pooled.columns:
        pooled = pooled[(pooled["scope"] == "All depots") & (pooled.get("evaluation", "chronological_holdout") == "chronological_holdout")]
    xgb = pooled[pooled["model"] == "XGBoost"]
    base = pooled[pooled["model"] != "XGBoost"]
    mae_text = f"{xgb['mae_kw'].iloc[0]:.2f}" if len(xgb) else "n/a"
    ui.cards([
        ui.card("Active model", f"v{bundle.version}", "", ui.esc(bundle.name), "good"),
        ui.card("Trained on", bundle.trained_at or "n/a", "", f"{meta.get('model_type', '')}"),
        ui.card("Test MAE", mae_text, "kW", f"Baseline {base['mae_kw'].iloc[0]:.2f} kW" if len(base) else ""),
        ui.card("Test RMSE", f"{xgb['rmse_kw'].iloc[0]:.2f}" if len(xgb) else "n/a", "kW",
                f"Baseline {base['rmse_kw'].iloc[0]:.2f} kW" if len(base) else ""),
        ui.card("Breach recall", fmt_pct(xgb["breach_recall"].iloc[0]) if len(xgb) and "breach_recall" in xgb else "n/a", "",
                f"False-alarm ratio {fmt_pct(xgb['false_alarm_ratio'].iloc[0])}" if len(xgb) and "false_alarm_ratio" in xgb else ""),
    ])
    with st.expander("Active model details"):
        st.markdown(
            f"- **Target:** `{meta.get('target')}` at {meta.get('sampling_minutes')}-minute intervals\n"
            f"- **Features:** {len(bundle.cfg.feature_names)} ({', '.join(bundle.cfg.feature_names[:8])}, ...)\n"
            f"- **Training data:** {meta.get('training_data', {}).get('source', 'n/a')}, "
            f"{meta.get('training_data', {}).get('n_depots', 'n/a')} depots\n"
            f"- **Package folder:** `{bundle.path.relative_to(config.ROOT)}`"
        )
        st.download_button("Download active model package (.zip)", model_loader.package_zip_bytes(bundle.path),
                           file_name=f"global_model_v{bundle.version}.zip", mime="application/zip",
                           icon=":material/download:")

    st.markdown("**Replace the model**")
    ui.note("Upload a .zip containing xgboost_model.json, model_metadata.json, feature_config.json and "
            "model_metrics.csv. The package is checked before it can be activated; the current model is kept for rollback.")
    uploaded = st.file_uploader("Model package (.zip)", type=["zip"], key="model_upload", label_visibility="collapsed")
    staging = config.MODELS_DIR / "_staging"
    if uploaded is None:
        shutil.rmtree(staging, ignore_errors=True)
    else:
        try:
            model_loader.extract_package(uploaded.getvalue(), staging)
            checks = model_loader.validate_package(staging)
        except model_loader.ModelPackageError as exc:
            st.error(str(exc), icon=":material/error:")
            checks = []
        if checks:
            st.dataframe(pd.DataFrame([{"Check": c.name, "Result": "✓ Passed" if c.passed else "✕ Failed",
                                        "Detail": c.detail} for c in checks]), hide_index=True, width="stretch",
                         height=38 + 35 * len(checks))
            if model_loader.is_valid(checks):
                new_version = next((c.detail for c in checks if c.name == "Model metadata"), "")
                st.success(f"Compatible model package: {new_version}", icon=":material/check_circle:")
                if st.button("Activate this model", type="primary", icon=":material/published_with_changes:"):
                    try:
                        model_loader.activate_package(staging)
                        activated = model_loader.load_bundle(config.ACTIVE_MODEL_DIR)
                        storage.log_model_event("Activated", activated.version,
                                                f"Replaced v{bundle.version} with uploaded package {uploaded.name}",
                                                services.user_email())
                        shutil.rmtree(staging, ignore_errors=True)
                        st.session_state.pop("model_upload", None)
                        st.session_state["model_notice"] = f"Model v{activated.version} is now active. v{bundle.version} was archived."
                        st.rerun()
                    except model_loader.ModelPackageError as exc:
                        st.error(str(exc), icon=":material/error:")
            else:
                st.error("This model package is not compatible and was rejected. The active model is unchanged.",
                         icon=":material/error:")
                shutil.rmtree(staging, ignore_errors=True)

    st.markdown("**Previous models (rollback)**")
    archive = model_loader.list_archive()
    if not archive:
        ui.empty_state("No previous models", "When a model is replaced, the previous one is kept here for rollback.")
        return
    labels = {i: f"v{a['model_version']} ({a['model_name']}), trained {a['trained_at']}, archived {a['archived'][:8]}"
              for i, a in enumerate(archive)}
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    pick = left.selectbox("Archived model", list(labels), format_func=labels.get)
    if right.button("Roll back", icon=":material/undo:", width="stretch"):
        try:
            target = archive[pick]
            model_loader.rollback(target["folder"])
            storage.log_model_event("Rolled back", str(target["model_version"]),
                                    f"Re-activated v{target['model_version']}; v{bundle.version} archived",
                                    services.user_email())
            st.session_state["model_notice"] = f"Rolled back to model v{target['model_version']}."
            st.rerun()
        except model_loader.ModelPackageError as exc:
            st.error(str(exc), icon=":material/error:")


def _organisations() -> None:
    orgs = storage.list_organisations()
    customers = orgs[orgs["platform"] != 1]
    ui.cards([
        ui.card("Customer organisations", str(len(customers)), "", f"{int(customers['active'].sum())} active", icon_name="layers"),
        ui.card("User accounts", str(int(orgs["users"].sum())), "", "Across all organisations", icon_name="users"),
        ui.card("Depots", str(int(customers["depots"].fillna(0).sum())), "", "Configured by customers", icon_name="cube"),
        ui.card("Sign-up", "Open" if storage.signup_allowed() else "Closed", "", "Public Create account page", icon_name="shield"),
    ])
    notice = st.session_state.pop("platform_notice", None)
    if notice:
        st.success(notice, icon=":material/check_circle:")
    shown = pd.DataFrame({
        "Organisation": orgs["name"],
        "Type": orgs["platform"].map(lambda p: "Platform" if p == 1 else "Customer"),
        "Status": orgs["active"].map({1: "Active", 0: "Suspended"}),
        "Users": orgs["users"], "Administrators": orgs["admins"],
        "Depots": orgs["depots"].map(lambda d: "Not opened yet" if pd.isna(d) else str(int(d))),
        "Created": orgs["created_at"].str[:16], "Last sign-in": orgs["last_sign_in"].fillna("Never").str[:16],
    })
    st.dataframe(shown, hide_index=True, width="stretch")

    if customers.empty:
        ui.empty_state("No customers yet", "Organisations appear here when someone creates an account.")
        return
    names = customers.set_index("id")["name"].to_dict()
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    target = left.selectbox("Customer organisation", list(names), format_func=names.get)
    active = bool(customers.set_index("id").loc[target, "active"])
    if right.button("Suspend access" if active else "Reactivate access", width="stretch",
                    icon=":material/block:" if active else ":material/check_circle:"):
        try:
            storage.set_organisation_active(target, not active, services.user_email())
            st.session_state["platform_notice"] = f"{names[target]} {'suspended' if active else 'reactivated'}."
            st.rerun()
        except storage.AccountError as exc:
            st.error(str(exc), icon=":material/error:")
    ui.note("Suspending an organisation signs its users out and blocks sign-in until it is reactivated. "
            "Its data is kept.")


def _signup() -> None:
    allowed = storage.signup_allowed()
    choice = st.toggle("Allow new customers to create an account from the public page", value=allowed)
    if choice != allowed:
        storage.set_signup_allowed(choice, services.user_email())
        st.rerun()
    ui.note("When this is on, anyone can create a new organisation and becomes its administrator. They start with "
            "the sample depots and can upload their own data. When it is off, the Create account page is closed "
            "and only existing users can sign in.", title="What this controls")


def _activity() -> None:
    log = storage.read_platform_audit()
    if log.empty:
        ui.empty_state("No activity recorded yet", "Sign-ins, account changes and model changes appear here.")
        return
    st.dataframe(log.rename(columns={"created_at": "When", "organisation": "Organisation", "user": "User",
                                     "action": "Action", "detail": "Detail"}),
                 hide_index=True, width="stretch", height=min(560, 38 + 35 * len(log)))


def render() -> None:
    if not services.is_owner():
        st.error("This page is only available to the platform owner.", icon=":material/lock:")
        return
    try:
        bundle = services.get_bundle()
    except model_loader.ModelPackageError as exc:
        bundle = None
        st.error(f"The forecasting model could not be loaded. {exc}")
    ui.page_header("Platform", "Customers, sign-up, the shared forecasting model and activity", config.PRODUCT_NAME,
                   [("Role", "Platform owner")] + ([("Model", f"v{bundle.version}")] if bundle else []))
    notice = st.session_state.pop("model_notice", None)
    if notice:
        st.success(notice, icon=":material/check_circle:")
    tabs = st.tabs(["Organisations", "Sign-up", "Model", "Activity"])
    with tabs[0]:
        _organisations()
    with tabs[1]:
        _signup()
    with tabs[2]:
        if bundle is not None:
            _model(bundle, True)
    with tabs[3]:
        _activity()
