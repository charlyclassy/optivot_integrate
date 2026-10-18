"""Settings: an organisation's thresholds, capacities, users, data sources and reports."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from dashboard import services
from dashboard import database as storage
from dashboard import theme as ui


ROLE_LABEL = {"admin": "Administrator", "viewer": "Viewer"}


def _general(settings: dict, admin: bool) -> None:
    with st.form("general_settings", border=True):
        st.markdown("**Organisation and forecast defaults**")
        c1, c2 = st.columns(2)
        organisation = c1.text_input("Organisation name", value=settings["organisation"], max_chars=80)
        alpha = c2.slider("Default warning threshold (% of capacity)", 50, 99, int(round(settings["warning_alpha"] * 100)),
                          help="Warning when forecast demand reaches this share of capacity. Depots can override it.")
        c3, c4 = st.columns(2)
        horizon = c3.selectbox("Default forecast horizon", list(config.HORIZON_CHOICES),
                               index=list(config.HORIZON_CHOICES).index(int(settings["horizon_hours"])),
                               format_func=lambda h: f"{h} hours")
        window = c4.selectbox("Recent history shown on charts", [24, 48, 72, 168],
                              index=[24, 48, 72, 168].index(int(settings["history_window_hours"])),
                              format_func=lambda h: f"{h} hours" if h < 168 else "7 days")
        saved = st.form_submit_button("Save settings", type="primary", icon=":material/save:", disabled=not admin)
    if saved:
        new_name = organisation.strip() or settings["organisation"]
        if new_name != settings["organisation"]:
            storage.rename_organisation(services.org_id(), new_name, services.user_email())
            st.session_state["user"]["org_name"] = new_name
        settings.update(organisation=new_name, warning_alpha=alpha / 100,
                        horizon_hours=int(horizon), history_window_hours=int(window))
        services.save_settings()
        services.audit("Settings changed", f"warning threshold {alpha}%, horizon {horizon} h")
        st.success("Settings saved.", icon=":material/check_circle:")
        st.rerun()
    ui.note("Status rules: Breach when forecast demand is at or above capacity C; Warning when it is at or above "
            "the warning threshold but below C; Within Capacity otherwise.", title="Status rules")


def _capacity(settings: dict, admin: bool) -> None:
    st.markdown("**Capacity and warning threshold per depot**")
    table = pd.DataFrame([
        {"key": key, "Depot": d["name"], "Grid capacity (kW)": float(d["capacity_kw"]),
         "Warning threshold (%)": int(round(storage.depot_alpha(settings, key) * 100)),
         "Data source": d.get("data_label", "")}
        for key, d in settings["depots"].items()
    ]).set_index("key")
    edited = st.data_editor(
        table, hide_index=True, width="stretch", disabled=["Depot", "Data source"] if admin else True,
        column_config={
            "Grid capacity (kW)": st.column_config.NumberColumn(min_value=1.0, step=1.0, format="%.0f"),
            "Warning threshold (%)": st.column_config.NumberColumn(min_value=50, max_value=99, step=1),
        }, key="capacity_editor",
    )
    if st.button("Save capacities", type="primary", icon=":material/save:", disabled=not admin):
        for key, row in edited.iterrows():
            depot = settings["depots"][key]
            depot["capacity_kw"] = float(row["Grid capacity (kW)"])
            alpha = float(row["Warning threshold (%)"]) / 100
            depot["warning_alpha"] = None if abs(alpha - settings["warning_alpha"]) < 1e-9 else alpha
        services.save_settings()
        services.audit("Capacities changed", f"{len(edited)} depot(s)")
        st.success("Capacities saved. Forecast status and alerts now use the new values.", icon=":material/check_circle:")
        st.rerun()
    ui.note("Depot names, new depots and data uploads are managed on the Depots page.")


def _data_sources(settings: dict) -> None:
    st.markdown("**Data source per depot**")
    rows = []
    for depot in settings["depots"].values():
        try:
            series = services.get_series(depot)
            rows.append({"Depot": depot["name"], "Type": "Sample real data" if depot["source"] == "bundled" else "Your upload",
                         "Source": depot.get("data_label", ""), "From": series.index.min(), "Through": series.index.max(),
                         "Intervals": len(series), "Status": "Real Data"})
        except Exception:
            rows.append({"Depot": depot["name"], "Type": depot["source"], "Source": depot.get("data_label", ""),
                         "Status": "Data file missing"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "From": st.column_config.DatetimeColumn(format="D MMM YYYY"),
        "Through": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
    })
    ui.note(f"Sample depots use {config.DATA_SOURCE_NAME} (CC BY 4.0): metered EV charging load at Royal Mail depots, "
            "averaged to 15-minute intervals. Your own depots use the meter data you upload. No simulated demand is "
            "used anywhere in the application. Forecasts start where each depot's observed data ends.", title="About the data")


def _users(settings: dict, admin: bool) -> None:
    """Your organisation's accounts: who can sign in and with which role."""
    ui.note("<b>Administrator</b>: manages this organisation's depots, data, thresholds and users. "
            "<b>Viewer</b>: read-only access to forecasts, alerts, reports and history. "
            "People you add here can only see this organisation's data.", title="What each role can do")
    if not admin:
        st.info("Only administrators can manage accounts.", icon=":material/lock:")
        return
    org, me = services.org_id(), services.user_email()
    notice = st.session_state.pop("user_notice", None)
    if notice:
        st.success(notice, icon=":material/check_circle:")

    users = storage.list_users(org)
    shown = pd.DataFrame({
        "Name": users["name"], "Email": users["email"], "Role": users["role"].map(ROLE_LABEL),
        "Status": users["active"].map({1: "Active", 0: "Inactive"}),
        "Created": users["created_at"].str[:16], "Created by": users["created_by"],
        "Last sign-in": users["last_login"].fillna("Never").str[:16],
    })
    st.dataframe(shown, hide_index=True, width="stretch")

    left, right = st.columns(2, gap="large")
    with left, st.form("create_user", border=True, clear_on_submit=True):
        st.markdown("**Add a colleague**")
        name = st.text_input("Full name")
        email = st.text_input("Email")
        a, b = st.columns(2)
        role = a.selectbox("Role", list(storage.ROLES), index=1, format_func=ROLE_LABEL.get)
        password = b.text_input("Temporary password", type="password",
                                help=f"At least {storage.MIN_PASSWORD_LENGTH} characters, with letters and numbers or symbols.")
        if st.form_submit_button("Create account", type="primary", icon=":material/person_add:"):
            try:
                created = storage.create_user(org, email, name, password, role, created_by=me)
                st.session_state["user_notice"] = f"Account created for {created['email']} ({ROLE_LABEL[role]})."
                st.rerun()
            except storage.AccountError as exc:
                st.error(str(exc), icon=":material/error:")

    with right, st.container(border=True):
        st.markdown("**Manage an account**")
        by_email = users.set_index("email")
        target = st.selectbox("Account", users["email"].tolist(), format_func=lambda e: f"{by_email.loc[e, 'name']} ({e})")
        row = by_email.loc[target]
        a, b = st.columns(2)
        new_role = a.selectbox("Role", list(storage.ROLES), index=list(storage.ROLES).index(row["role"]),
                               format_func=ROLE_LABEL.get, key=f"role_{target}")
        new_active = b.selectbox("Status", [True, False], index=0 if row["active"] else 1,
                                 format_func=lambda v: "Active" if v else "Inactive", key=f"active_{target}")
        new_password = st.text_input("New password (leave empty to keep the current one)", type="password",
                                     key=f"password_{target}")
        save, remove = st.columns(2)
        if save.button("Save changes", type="primary", icon=":material/save:", width="stretch"):
            try:
                if target == me and (new_role != "admin" or not new_active):
                    raise storage.AccountError("You cannot remove your own administrator access.")
                storage.update_user(org, target, role=new_role, active=bool(new_active), actor=me)
                if new_password:
                    storage.reset_password(org, target, new_password, actor=me)
                st.session_state["user_notice"] = f"Account {target} updated."
                st.rerun()
            except storage.AccountError as exc:
                st.error(str(exc), icon=":material/error:")
        if remove.button("Delete account", icon=":material/delete:", width="stretch", disabled=target == me):
            try:
                storage.delete_user(org, target, actor=me)
                st.session_state["user_notice"] = f"Account {target} deleted."
                st.rerun()
            except storage.AccountError as exc:
                st.error(str(exc), icon=":material/error:")


def _account() -> None:
    """The signed-in user's own profile and password."""
    user = services.current_user()
    ui.cards([
        ui.card("Signed in as", user["name"], "", ui.esc(user["email"]), icon_name="users"),
        ui.card("Role", "Platform owner" if user.get("platform_admin") else ROLE_LABEL[user["role"]], "",
                ui.esc(user.get("org_name", "")), icon_name="shield"),
    ], 2)
    with st.form("change_password", border=True, clear_on_submit=True):
        st.markdown("**Change your password**")
        current = st.text_input("Current password", type="password")
        a, b = st.columns(2)
        new = a.text_input("New password", type="password",
                           help=f"At least {storage.MIN_PASSWORD_LENGTH} characters, with letters and numbers or symbols.")
        confirm = b.text_input("Confirm new password", type="password")
        if st.form_submit_button("Change password", type="primary", icon=":material/lock_reset:"):
            if new != confirm:
                st.error("The two new passwords do not match.", icon=":material/error:")
            else:
                try:
                    storage.change_password(user["email"], current, new)
                    st.success("Your password has been changed.", icon=":material/check_circle:")
                except storage.AccountError as exc:
                    st.error(str(exc), icon=":material/error:")


def _activity(admin: bool) -> None:
    """Audit log of this organisation."""
    if not admin:
        st.info("Only administrators can view the activity log.", icon=":material/lock:")
        return
    log = storage.read_audit(services.org_id())
    if log.empty:
        ui.empty_state("No activity recorded yet", "Sign-ins, user changes, data uploads and settings changes appear here.")
        return
    st.dataframe(log.rename(columns={"created_at": "When", "user": "User", "action": "Action", "detail": "Detail"}),
                 hide_index=True, width="stretch", height=min(520, 38 + 35 * len(log)))
    ui.note("The most recent 500 events for your organisation.")


def _reports(settings: dict, admin: bool) -> None:
    with st.form("report_settings", border=True):
        st.markdown("**Report and export settings**")
        c1, c2 = st.columns(2)
        formats = ["Excel", "PDF", "CSV"]
        default_format = c1.selectbox("Default report format", formats,
                                      index=formats.index(settings["report"].get("default_format", "Excel")))
        prepared_by = c2.text_input("Prepared by (shown on reports)", value=settings["report"].get("prepared_by", ""))
        include = st.checkbox("Include interval-level forecast detail by default",
                              value=bool(settings["report"].get("include_forecast_table", True)))
        saved = st.form_submit_button("Save report settings", type="primary", icon=":material/save:", disabled=not admin)
    if saved:
        settings["report"] = {"default_format": default_format, "prepared_by": prepared_by.strip(),
                              "include_forecast_table": include}
        services.save_settings()
        st.success("Report settings saved.", icon=":material/check_circle:")
    exports = sorted(config.REPORT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if config.REPORT_DIR.is_dir() else []
    ui.note(f"Exports are saved to <code>outputs/report_exports</code> ({len(exports)} file(s) so far). "
            "The latest forecast is written to <code>outputs/latest_forecast.csv</code>. ")


def render() -> None:
    settings = services.get_settings()
    admin = services.is_admin()
    chips = [("Role", "Administrator" if admin else "Viewer")]
    ui.page_header("Settings", "Capacity, thresholds, users, data sources and report settings",
                   settings["organisation"], chips)
    if not admin:
        st.info("You are signed in as a viewer. Organisation settings are read-only.", icon=":material/lock:")

    tabs = st.tabs(["General", "Capacity and thresholds", "Users", "My account", "Data sources",
                    "Reports and exports", "Activity log"])
    with tabs[0]:
        _general(settings, admin)
    with tabs[1]:
        _capacity(settings, admin)
    with tabs[2]:
        _users(settings, admin)
    with tabs[3]:
        _account()
    with tabs[4]:
        _data_sources(settings)
    with tabs[5]:
        _reports(settings, admin)
    with tabs[6]:
        _activity(admin)
