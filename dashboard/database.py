"""Application database (SQLite): organisations, users, sessions, settings and history.

Everything the application remembers lives in one file, ``<state dir>/depotwatt.db``:

    organisations    one row per customer workspace
    users            sign-in accounts (salted PBKDF2 hashes, role, organisation, lockout state)
    sessions         "remember me" tokens
    app_settings     per-organisation settings (depots, thresholds, report options) as JSON,
                     plus platform-wide settings
    forecast_runs    every forecast generated, with the settings in force
    forecast_points  the interval values of each forecast run
    alerts           warnings / breaches raised by forecast runs, with acknowledgement
    scenario_runs    saved scenario analyses
    model_events     model activations and rollbacks (platform-wide)
    audit_log        who did what and when

Every organisation sees only its own rows. SQLite needs no server and is created on first run.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config

SCHEMA_VERSION = 2
ROLES = ("admin", "viewer")
SESSION_DAYS = 14
MIN_PASSWORD_LENGTH = 8
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS organisations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organisations (id),
    name TEXT NOT NULL, role TEXT NOT NULL CHECK (role IN ('admin', 'viewer')),
    platform_admin INTEGER NOT NULL DEFAULT 0,
    salt TEXT NOT NULL, password_hash TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
    failed_logins INTEGER NOT NULL DEFAULT 0, locked_until TEXT,
    created_at TEXT NOT NULL, created_by TEXT, last_login TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, email TEXT NOT NULL, expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL,
    created_at TEXT NOT NULL, user TEXT, depot TEXT NOT NULL, issued_at TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL, model_version TEXT NOT NULL,
    capacity_kw REAL NOT NULL, alpha REAL NOT NULL,
    current_kw REAL, peak_kw REAL, peak_time TEXT, status TEXT,
    warning_time TEXT, breach_time TEXT, lead_minutes REAL,
    UNIQUE (org_id, depot, issued_at, horizon_hours, model_version, capacity_kw, alpha)
);
CREATE TABLE IF NOT EXISTS forecast_points (
    run_id INTEGER NOT NULL, timestamp TEXT NOT NULL, forecast_kw REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_points_run ON forecast_points (run_id);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL,
    run_id INTEGER UNIQUE, created_at TEXT NOT NULL, depot TEXT NOT NULL, issued_at TEXT NOT NULL,
    level TEXT NOT NULL, event_time TEXT, peak_kw REAL, capacity_kw REAL, lead_minutes REAL,
    status TEXT NOT NULL DEFAULT 'Open', acknowledged_by TEXT, acknowledged_at TEXT
);
CREATE TABLE IF NOT EXISTS scenario_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER NOT NULL,
    created_at TEXT NOT NULL, user TEXT, depot TEXT NOT NULL, issued_at TEXT NOT NULL,
    horizon_hours INTEGER, growth_pct REAL, utilisation_factor REAL, capacity_kw REAL, alpha REAL,
    base_peak_kw REAL, scenario_peak_kw REAL, base_status TEXT, scenario_status TEXT,
    scenario_breach_time TEXT, model_version TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS model_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, user TEXT, action TEXT NOT NULL, model_version TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, org_id INTEGER,
    created_at TEXT NOT NULL, user TEXT, action TEXT NOT NULL, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_log (org_id, id);
"""
ORG_TABLES = ("forecast_runs", "alerts", "scenario_runs")


class AccountError(ValueError):
    """Raised with a user-readable message when an account operation is not allowed."""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ts(value) -> str | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def _retire_old_database(path: Path) -> None:
    """A database from an earlier schema is kept as a backup and a new one is started."""
    if not path.is_file():
        return
    conn = sqlite3.connect(path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
    finally:
        conn.close()
    if has_tables and version != SCHEMA_VERSION:
        path.replace(path.with_name(f"{path.stem}_v{version}_backup_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"))


@contextmanager
def connect(path: Path | None = None):
    """Open the database, creating the schema (and demo workspace in demo mode) on first use."""
    path = path or config.DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    _retire_old_database(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        if config.DEMO_MODE and conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            _seed_demo_workspace(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


# =============================================================================
# Audit log
# =============================================================================
def _audit(conn, org_id: int | None, user: str | None, action: str, detail: str = "") -> None:
    conn.execute("INSERT INTO audit_log (org_id, created_at, user, action, detail) VALUES (?,?,?,?,?)",
                 (org_id, _now(), user, action, detail))


def audit(org_id: int | None, user: str | None, action: str, detail: str = "", path: Path | None = None) -> None:
    with connect(path) as conn:
        _audit(conn, org_id, user, action, detail)


def read_audit(org_id: int, limit: int = 500, path: Path | None = None) -> pd.DataFrame:
    with connect(path) as conn:
        return pd.read_sql_query(
            "SELECT created_at, user, action, detail FROM audit_log WHERE org_id = ? ORDER BY id DESC LIMIT ?",
            conn, params=[int(org_id), int(limit)])


# =============================================================================
# Organisations and users
# =============================================================================
def _insert_user(conn, org_id: int, email: str, name: str, role: str, password: str, created_by: str | None,
                 platform_admin: bool = False) -> None:
    salt = secrets.token_hex(16)
    conn.execute(
        "INSERT INTO users (email, org_id, name, role, platform_admin, salt, password_hash, active, created_at, created_by) "
        "VALUES (?,?,?,?,?,?,?,1,?,?)",
        (email, org_id, name, role, int(platform_admin), salt, _hash_password(password, salt), _now(), created_by),
    )


def _insert_org(conn, name: str) -> int:
    return conn.execute("INSERT INTO organisations (name, created_at) VALUES (?,?)", (name, _now())).lastrowid


def _seed_demo_workspace(conn) -> None:
    """Demo mode only: a platform owner plus one customer organisation with an admin and a viewer."""
    platform = _insert_org(conn, config.PLATFORM_ORGANISATION)
    _insert_user(conn, platform, config.DEMO_OWNER_EMAIL, "Platform Owner", "admin", config.DEMO_OWNER_PASSWORD,
                 "system", platform_admin=True)
    org_id = _insert_org(conn, config.DEFAULT_ORGANISATION)
    _insert_user(conn, org_id, config.DEMO_ADMIN_EMAIL, "Depot Administrator", "admin", config.DEMO_ADMIN_PASSWORD,
                 "system")
    _insert_user(conn, org_id, config.DEMO_VIEWER_EMAIL, "Operations Viewer", "viewer", config.DEMO_VIEWER_PASSWORD,
                 "system")


def _public(conn, row) -> dict:
    org = conn.execute("SELECT name FROM organisations WHERE id = ?", (row["org_id"],)).fetchone()
    return {"email": row["email"], "name": row["name"], "role": row["role"], "org_id": row["org_id"],
            "org_name": org["name"] if org else "", "platform_admin": bool(row["platform_admin"])}


def _check_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AccountError(f"The password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if password.isalpha() or password.isdigit():
        raise AccountError("The password must contain both letters and numbers or symbols.")


def _clean_identity(email: str, name: str) -> tuple[str, str]:
    email, name = (email or "").strip().lower(), (name or "").strip()
    if not EMAIL_PATTERN.match(email):
        raise AccountError("Enter a valid email address.")
    if not name:
        raise AccountError("Enter your name.")
    return email, name


def create_organisation(org_name: str, email: str, name: str, password: str, platform_admin: bool = False,
                        path: Path | None = None) -> dict:
    """Sign-up: a new customer workspace with its first user as administrator."""
    org_name = (org_name or "").strip()
    if len(org_name) < 2:
        raise AccountError("Enter your organisation's name.")
    email, name = _clean_identity(email, name)
    _check_password(password)
    with connect(path) as conn:
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise AccountError("An account with this email already exists. Sign in instead.")
        org_id = _insert_org(conn, org_name)
        _insert_user(conn, org_id, email, name, "admin", password, "sign-up", platform_admin)
        _audit(conn, org_id, email, "Workspace created", org_name)
        return _public(conn, conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone())


def create_user(org_id: int, email: str, name: str, password: str, role: str = "viewer",
                created_by: str | None = None, path: Path | None = None) -> dict:
    """An administrator adds a colleague to their own organisation."""
    email, name = _clean_identity(email, name)
    if role not in ROLES:
        raise AccountError(f"Unknown role '{role}'.")
    _check_password(password)
    with connect(path) as conn:
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise AccountError("An account with this email already exists.")
        _insert_user(conn, int(org_id), email, name, role, password, created_by)
        _audit(conn, int(org_id), created_by, "User created", f"{email} ({role})")
    return {"email": email, "name": name, "role": role, "org_id": int(org_id)}


def authenticate(email: str, password: str, path: Path | None = None) -> dict | None:
    """Return the signed-in user for valid credentials of an active account, else None.

    After ``MAX_FAILED_LOGINS`` wrong passwords the account is locked for
    ``LOCKOUT_MINUTES``; ``AccountError`` is raised while it is locked.
    """
    email = (email or "").strip().lower()
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not row or not password or not row["active"]:
            return None
        org = conn.execute("SELECT active FROM organisations WHERE id = ?", (row["org_id"],)).fetchone()
        if not org or not org["active"]:
            raise AccountError("This organisation's access has been suspended. Contact support.")
        if row["locked_until"] and row["locked_until"] > _now():
            raise AccountError("Too many failed attempts. This account is locked for "
                               f"{LOCKOUT_MINUTES} minutes; try again later or ask an administrator to reset it.")
        if not hmac.compare_digest(_hash_password(password, row["salt"]), row["password_hash"]):
            failures = row["failed_logins"] + 1
            locked = None
            if failures >= MAX_FAILED_LOGINS:
                locked = (datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
                failures = 0
                _audit(conn, row["org_id"], email, "Account locked", f"{MAX_FAILED_LOGINS} failed sign-in attempts")
            conn.execute("UPDATE users SET failed_logins = ?, locked_until = ? WHERE email = ?", (failures, locked, email))
            conn.commit()
            return None
        conn.execute("UPDATE users SET last_login = ?, failed_logins = 0, locked_until = NULL WHERE email = ?",
                     (_now(), email))
        _audit(conn, row["org_id"], email, "Signed in")
        return _public(conn, row)


def list_users(org_id: int, path: Path | None = None) -> pd.DataFrame:
    with connect(path) as conn:
        return pd.read_sql_query(
            "SELECT email, name, role, active, created_at, created_by, last_login FROM users WHERE org_id = ? "
            "ORDER BY created_at", conn, params=[int(org_id)])


def _member(conn, org_id: int, email: str):
    row = conn.execute("SELECT * FROM users WHERE email = ? AND org_id = ?", (email, int(org_id))).fetchone()
    if not row:
        raise AccountError("Account not found in your organisation.")
    return row


def _guard_last_admin(conn, org_id: int, email: str) -> None:
    others = conn.execute("SELECT COUNT(*) FROM users WHERE org_id = ? AND role = 'admin' AND active = 1 AND email != ?",
                          (int(org_id), email)).fetchone()[0]
    if others == 0:
        raise AccountError("This is the only active administrator. Create another administrator first.")


def update_user(org_id: int, email: str, role: str | None = None, active: bool | None = None,
                actor: str | None = None, path: Path | None = None) -> None:
    """Change a colleague's role and/or active flag. The last active administrator is protected."""
    with connect(path) as conn:
        row = _member(conn, org_id, email)
        new_role = role if role is not None else row["role"]
        new_active = int(active) if active is not None else row["active"]
        if new_role not in ROLES:
            raise AccountError(f"Unknown role '{new_role}'.")
        if row["role"] == "admin" and row["active"] and (new_role != "admin" or not new_active):
            _guard_last_admin(conn, org_id, email)
        conn.execute("UPDATE users SET role = ?, active = ?, failed_logins = 0, locked_until = NULL WHERE email = ?",
                     (new_role, new_active, email))
        if not new_active:
            conn.execute("DELETE FROM sessions WHERE email = ?", (email,))
        _audit(conn, org_id, actor, "User updated", f"{email}: role {new_role}, {'active' if new_active else 'inactive'}")


def reset_password(org_id: int, email: str, new_password: str, actor: str | None = None,
                   path: Path | None = None) -> None:
    """Set a new password (administrator reset). Ends the user's sessions and clears any lockout."""
    _check_password(new_password)
    salt = secrets.token_hex(16)
    with connect(path) as conn:
        _member(conn, org_id, email)
        conn.execute("UPDATE users SET salt = ?, password_hash = ?, failed_logins = 0, locked_until = NULL WHERE email = ?",
                     (salt, _hash_password(new_password, salt), email))
        conn.execute("DELETE FROM sessions WHERE email = ?", (email,))
        _audit(conn, org_id, actor, "Password reset", email)


def change_password(email: str, current_password: str, new_password: str, path: Path | None = None) -> None:
    """A user changes their own password after proving the current one."""
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not row or not hmac.compare_digest(_hash_password(current_password or "", row["salt"]), row["password_hash"]):
            raise AccountError("The current password is not correct.")
        _check_password(new_password)
        if new_password == current_password:
            raise AccountError("Choose a password that is different from the current one.")
        salt = secrets.token_hex(16)
        conn.execute("UPDATE users SET salt = ?, password_hash = ? WHERE email = ?",
                     (salt, _hash_password(new_password, salt), email))
        _audit(conn, row["org_id"], email, "Password changed")


def delete_user(org_id: int, email: str, actor: str | None = None, path: Path | None = None) -> None:
    with connect(path) as conn:
        row = _member(conn, org_id, email)
        if row["role"] == "admin" and row["active"]:
            _guard_last_admin(conn, org_id, email)
        conn.execute("DELETE FROM users WHERE email = ?", (email,))
        conn.execute("DELETE FROM sessions WHERE email = ?", (email,))
        _audit(conn, org_id, actor, "User deleted", email)


def list_organisations(path: Path | None = None) -> pd.DataFrame:
    """Platform owner view: every customer organisation with its size and last activity."""
    with connect(path) as conn:
        orgs = pd.read_sql_query(
            "SELECT o.id, o.name, o.active, o.created_at, COUNT(u.email) AS users, "
            "SUM(CASE WHEN u.role = 'admin' THEN 1 ELSE 0 END) AS admins, MAX(u.last_login) AS last_sign_in, "
            "MAX(u.platform_admin) AS platform FROM organisations o LEFT JOIN users u ON u.org_id = o.id "
            "GROUP BY o.id ORDER BY o.id", conn)
        depots = []
        for org_id in orgs["id"]:
            settings = _read_setting(conn, f"org:{int(org_id)}")
            depots.append(len(settings["depots"]) if settings else None)
    orgs["depots"] = depots
    return orgs


def set_organisation_active(org_id: int, active: bool, actor: str | None = None, path: Path | None = None) -> None:
    """Suspend or reactivate a customer organisation. Suspension ends all of its sessions."""
    with connect(path) as conn:
        row = conn.execute("SELECT MAX(u.platform_admin) AS platform FROM users u WHERE u.org_id = ?",
                           (int(org_id),)).fetchone()
        if row and row["platform"] and not active:
            raise AccountError("The platform owner's organisation cannot be suspended.")
        conn.execute("UPDATE organisations SET active = ? WHERE id = ?", (int(active), int(org_id)))
        if not active:
            conn.execute("DELETE FROM sessions WHERE email IN (SELECT email FROM users WHERE org_id = ?)", (int(org_id),))
        _audit(conn, int(org_id), actor, "Organisation " + ("reactivated" if active else "suspended"))


def read_platform_audit(limit: int = 500, path: Path | None = None) -> pd.DataFrame:
    """Platform owner view: recent activity across all organisations."""
    with connect(path) as conn:
        return pd.read_sql_query(
            "SELECT a.created_at, COALESCE(o.name, 'Platform') AS organisation, a.user, a.action, a.detail "
            "FROM audit_log a LEFT JOIN organisations o ON o.id = a.org_id ORDER BY a.id DESC LIMIT ?",
            conn, params=[int(limit)])


def rename_organisation(org_id: int, name: str, actor: str | None = None, path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE organisations SET name = ? WHERE id = ?", (name, int(org_id)))
        _audit(conn, org_id, actor, "Organisation renamed", name)


# =============================================================================
# "Remember me" sessions (a token kept in the page address so a reload stays signed in)
# =============================================================================
def create_session(email: str, path: Path | None = None) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(days=SESSION_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    with connect(path) as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
        conn.execute("INSERT INTO sessions (token, email, expires_at) VALUES (?,?,?)", (token, email, expires))
    return token


def resolve_session(token: str | None, path: Path | None = None) -> dict | None:
    """Return the user for a valid, unexpired session token of an active account."""
    if not token:
        return None
    with connect(path) as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.email = s.email JOIN organisations o ON o.id = u.org_id "
            "WHERE s.token = ? AND s.expires_at >= ? AND u.active = 1 AND o.active = 1", (token, _now())).fetchone()
        return _public(conn, row) if row else None


def drop_session(token: str | None, path: Path | None = None) -> None:
    if token:
        with connect(path) as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# =============================================================================
# Settings (per organisation: depots, thresholds, report options)
# =============================================================================
def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "depot"


def sample_depots(model_metadata: dict | None = None) -> dict:
    """The bundled real depots (sample data) with their default planning limits."""
    capacities = (model_metadata or {}).get("evaluation", {}).get("capacity_kw", {})
    info = {d["depot_id"]: d for d in (model_metadata or {}).get("training_data", {}).get("depots", [])}
    depots = {}
    for name in config.BUNDLED_DEPOTS.values():
        meta = info.get(name, {})
        depots[slugify(name)] = {
            "name": name,
            "data_id": name,
            "source": "bundled",
            "data_label": f"Sample real data: {config.DATA_SOURCE_NAME}",
            "capacity_kw": float(capacities.get(name, 0.0)),
            "warning_alpha": None,  # None = use the organisation default
            "vehicles": meta.get("vehicles"),
            "connection_kva": meta.get("recorded_connection_kva"),
        }
    return depots


def default_settings(model_metadata: dict | None = None, organisation: str | None = None) -> dict:
    """First-run settings for a workspace: sample depots so there is something to explore."""
    depots = sample_depots(model_metadata)
    return {
        "organisation": organisation or config.DEFAULT_ORGANISATION,
        "warning_alpha": config.DEFAULT_WARNING_ALPHA,
        "horizon_hours": config.DEFAULT_HORIZON_HOURS,
        "history_window_hours": config.DEFAULT_HISTORY_WINDOW_HOURS,
        "selected_depot": "mount_pleasant" if "mount_pleasant" in depots else next(iter(depots), None),
        "depots": depots,
        "report": {"default_format": "Excel", "prepared_by": "", "include_forecast_table": True},
    }


def _read_setting(conn, key: str):
    row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else None


def _write_setting(conn, key: str, value) -> None:
    conn.execute("INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                 (key, json.dumps(value), _now()))


def save_settings(org_id: int, settings: dict, path: Path | None = None) -> None:
    with connect(path) as conn:
        _write_setting(conn, f"org:{int(org_id)}", settings)


def load_settings(org_id: int, model_metadata: dict | None = None, path: Path | None = None) -> dict:
    with connect(path) as conn:
        stored = _read_setting(conn, f"org:{int(org_id)}")
        org = conn.execute("SELECT name FROM organisations WHERE id = ?", (int(org_id),)).fetchone()
    defaults = default_settings(model_metadata, org["name"] if org else None)
    if stored is None:
        save_settings(org_id, defaults, path)
        return defaults
    for key, value in defaults.items():  # tolerate settings saved by an older version
        if key != "depots":
            stored.setdefault(key, value)
    stored.setdefault("depots", {})
    for key, value in defaults["report"].items():
        stored["report"].setdefault(key, value)
    if stored["selected_depot"] not in stored["depots"]:
        stored["selected_depot"] = next(iter(stored["depots"]), None)
    return stored


def signup_allowed(path: Path | None = None) -> bool:
    """Platform-wide: whether the public "Create account" page is enabled."""
    with connect(path) as conn:
        platform = _read_setting(conn, "platform") or {}
    return bool(platform.get("allow_signup", config.ALLOW_SELF_SIGNUP))


def set_signup_allowed(allowed: bool, actor: str | None = None, path: Path | None = None) -> None:
    with connect(path) as conn:
        platform = _read_setting(conn, "platform") or {}
        platform["allow_signup"] = bool(allowed)
        _write_setting(conn, "platform", platform)
        _audit(conn, None, actor, "Sign-up " + ("enabled" if allowed else "disabled"))


def depot_alpha(settings: dict, depot_key: str) -> float:
    value = settings["depots"][depot_key].get("warning_alpha")
    return float(value) if value else float(settings["warning_alpha"])


# =============================================================================
# History (forecast runs, alerts, scenario runs, model events)
# =============================================================================
def log_forecast_run(org_id: int, depot: str, issued_at, horizon_hours: int, model_version: str, risk,
                     current_kw: float, forecast: pd.DataFrame, user: str | None, path: Path | None = None) -> int | None:
    """Record a forecast (once per depot / issue time / settings) and its alert, if any.

    Returns the new run id, or None if this exact forecast was already logged.
    """
    lead = None if risk.lead_time is None else risk.lead_time.total_seconds() / 60.0
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO forecast_runs (org_id, created_at, user, depot, issued_at, horizon_hours, "
            "model_version, capacity_kw, alpha, current_kw, peak_kw, peak_time, status, warning_time, breach_time, "
            "lead_minutes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (int(org_id), _now(), user, depot, _ts(issued_at), int(horizon_hours), model_version,
             float(risk.capacity_kw), round(risk.warning_kw / risk.capacity_kw, 4) if risk.capacity_kw else 0.0,
             float(current_kw), risk.peak_kw, _ts(risk.peak_time), risk.status, _ts(risk.warning_time),
             _ts(risk.breach_time), lead),
        )
        if cur.rowcount == 0:
            return None
        run_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO forecast_points (run_id, timestamp, forecast_kw) VALUES (?,?,?)",
            [(run_id, _ts(t), float(v)) for t, v in zip(forecast["timestamp"], forecast["forecast_kw"])],
        )
        if risk.status != "Within Capacity":
            event_time = risk.breach_time if risk.breach_time is not None else risk.warning_time
            conn.execute(
                "INSERT OR IGNORE INTO alerts (org_id, run_id, created_at, depot, issued_at, level, event_time, "
                "peak_kw, capacity_kw, lead_minutes) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (int(org_id), run_id, _now(), depot, _ts(issued_at), risk.status, _ts(event_time), risk.peak_kw,
                 float(risk.capacity_kw), lead),
            )
        return run_id


def find_alert(org_id: int, depot: str, issued_at, horizon_hours: int, model_version: str, capacity_kw: float,
               alpha: float, path: Path | None = None) -> dict | None:
    """The alert (if any) raised by one specific forecast run."""
    with connect(path) as conn:
        row = conn.execute(
            "SELECT a.* FROM alerts a JOIN forecast_runs r ON r.id = a.run_id WHERE r.org_id=? AND r.depot=? "
            "AND r.issued_at=? AND r.horizon_hours=? AND r.model_version=? AND r.capacity_kw=? AND r.alpha=?",
            (int(org_id), depot, _ts(issued_at), int(horizon_hours), model_version, float(capacity_kw),
             round(float(alpha), 4)),
        ).fetchone()
        return dict(row) if row else None


def log_scenario_run(org_id: int, row: dict, path: Path | None = None) -> None:
    row = {**row, "org_id": int(org_id), "created_at": _now()}
    for key in ("issued_at", "scenario_breach_time"):
        row[key] = _ts(row.get(key))
    columns = ", ".join(row)
    with connect(path) as conn:
        conn.execute(f"INSERT INTO scenario_runs ({columns}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


def acknowledge_alert(org_id: int, alert_id: int, user: str, path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE alerts SET status='Acknowledged', acknowledged_by=?, acknowledged_at=? "
                     "WHERE id=? AND org_id=?", (user, _now(), int(alert_id), int(org_id)))
        _audit(conn, org_id, user, "Alert acknowledged", f"alert {int(alert_id)}")


def read_table(org_id: int, table: str, depot: str | None = None, path: Path | None = None) -> pd.DataFrame:
    """One organisation's rows of a history table, newest first."""
    if table not in ORG_TABLES:
        raise ValueError(f"Unknown table {table}")
    query, params = f"SELECT * FROM {table} WHERE org_id = ?", [int(org_id)]
    if depot:
        query += " AND depot = ?"
        params.append(depot)
    with connect(path) as conn:
        return pd.read_sql_query(query + " ORDER BY id DESC", conn, params=params)


def log_model_event(action: str, model_version: str, detail: str, user: str | None, path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.execute("INSERT INTO model_events (created_at, user, action, model_version, detail) VALUES (?,?,?,?,?)",
                     (_now(), user, action, model_version, detail))
        _audit(conn, None, user, f"Model {action.lower()}", f"v{model_version}: {detail}")


def read_model_events(path: Path | None = None) -> pd.DataFrame:
    with connect(path) as conn:
        return pd.read_sql_query("SELECT * FROM model_events ORDER BY id DESC", conn)
