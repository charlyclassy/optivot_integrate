"""Shared look and feel: stylesheet, icons and small HTML components."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from forecasting import STATUS_BREACH, STATUS_OK, STATUS_WARNING

# Status colours are reserved for operational state and always paired with an icon + label.
TONES = {
    "good": {"fg": "#047857", "bg": "#ecfdf5", "line": "#10b981", "soft": "#d1fae5", "icon": "check"},
    "warning": {"fg": "#b45309", "bg": "#fffbeb", "line": "#f59e0b", "soft": "#fde68a", "icon": "alert"},
    "critical": {"fg": "#b91c1c", "bg": "#fef2f2", "line": "#ef4444", "soft": "#fecaca", "icon": "bolt"},
    "neutral": {"fg": "#475569", "bg": "#f1f5f9", "line": "#94a3b8", "soft": "#e2e8f0", "icon": ""},
    "brand": {"fg": "#4338ca", "bg": "#eef2ff", "line": "#4f46e5", "soft": "#c7d2fe", "icon": ""},
}
STATUS_TONE = {STATUS_OK: "good", STATUS_WARNING: "warning", STATUS_BREACH: "critical"}
STATUS_HEADLINE = {
    STATUS_OK: "Within capacity",
    STATUS_WARNING: "Warning: demand is forecast to approach capacity",
    STATUS_BREACH: "Breach predicted: demand is forecast to exceed capacity",
}
GAUGE_MAX = 1.25  # the gauge runs from 0 to 125% of capacity

# Outline icons (24x24, stroke). Kept inline so the app has no icon-font dependency.
_ICON_PATHS = {
    "bolt": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.500-1-8-4-8-9V6z"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "timer": '<path d="M10 2h4"/><path d="M12 14l3-3"/><circle cx="12" cy="14" r="8"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "alert": '<path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.700-3L13.700 3.900a2 2 0 0 0-3.400 0z"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.300 21a1.940 1.940 0 0 0 3.400 0"/>',
    "layers": '<path d="M12 2 2 7l10 5 10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
    "cube": '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="M3 8l9 5 9-5"/><path d="M12 13v8"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.870M16 3.130a4 4 0 0 1 0 7.750"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/>',
}


def icon(name: str, size: int = 18) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{_ICON_PATHS.get(name, "")}</svg>')


LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48" role="img" '
    'aria-label="DepotWatt logo"><defs><linearGradient id="dwg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>'
    '<rect width="48" height="48" rx="13" fill="url(#dwg)"/><path d="M27 7 13 27h9l-2 14 15-21h-9z" fill="#fff"/></svg>'
)

CSS = """
<style>
:root { --ink:#0f172a; --ink-2:#475569; --ink-3:#94a3b8; --line:#e6eaf2; --card:#ffffff; --bg:#f4f6fb;
  --brand:#4f46e5; --brand-2:#6366f1; --radius:16px; --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10); }
.block-container { padding: 2.2rem clamp(1rem, 3.2vw, 3.2rem) 3rem; max-width: 1400px; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.02em; }

/* ---------- page header ---------- */
.dw-header { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:22px; }
.dw-header .org { font-size:12px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--brand); }
.dw-header h1 { font-size:clamp(23px, 2.4vw, 30px); font-weight:700; margin:4px 0 0 0; padding:0; line-height:1.15; letter-spacing:-0.03em; }
.dw-header .sub { font-size:14.5px; color:var(--ink-2); margin-top:5px; }
.dw-header > div:first-child { min-width:0; }
.dw-chips { display:flex; gap:6px; flex-wrap:wrap; }
.dw-chip { display:inline-flex; align-items:center; gap:6px; font-size:12.5px; font-weight:500; color:var(--ink-2);
  background:#fff; border:1px solid var(--line); border-radius:999px; padding:6px 12px; white-space:nowrap;
  box-shadow:0 1px 2px rgba(15,23,42,.04); }
.dw-chip b { color:var(--ink); font-weight:600; }

/* ---------- status hero ---------- */
.dw-hero { display:flex; align-items:center; justify-content:space-between; gap:28px; flex-wrap:wrap;
  border-radius:18px; padding:18px 22px; margin-bottom:16px; border:1px solid var(--tone-soft);
  background:linear-gradient(120deg, var(--tone-bg) 0%, #ffffff 70%); box-shadow:var(--shadow); }
.dw-hero .left { display:flex; align-items:center; gap:16px; flex:1 1 380px; min-width:0; }
.dw-hero .badge { width:44px; height:44px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  color:#fff; flex:0 0 44px; background:var(--tone-line); box-shadow:0 8px 18px -8px var(--tone-line); }
.dw-hero .eyebrow { font-size:11.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--tone-fg); }
.dw-hero .title { font-size:clamp(17px, 1.7vw, 20px); font-weight:700; line-height:1.25; color:var(--ink); letter-spacing:-0.02em; margin-top:2px; }
.dw-hero .text { font-size:13.5px; margin-top:3px; color:var(--ink-2); line-height:1.5; }
.dw-gauge { flex:1 1 260px; max-width:360px; min-width:0; }
.dw-gauge .g-top { display:flex; justify-content:space-between; align-items:baseline; font-size:12.5px; color:var(--ink-2); }
.dw-gauge .g-top b { font-size:21px; font-weight:700; color:var(--ink); letter-spacing:-0.02em; }
.dw-gauge .track { position:relative; height:10px; border-radius:999px; background:#e9edf5; margin:8px 0 6px 0; }
.dw-gauge .fill { position:absolute; left:0; top:0; bottom:0; border-radius:999px; background:var(--tone-line); }
.dw-gauge .mark { position:absolute; top:-4px; bottom:-4px; width:2px; border-radius:2px; }
.dw-gauge .g-foot { display:flex; gap:14px; justify-content:flex-end; font-size:11.5px; color:var(--ink-2); }
.dw-gauge .g-foot span { display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }
.dw-gauge .g-foot i { width:3px; height:11px; border-radius:2px; display:inline-block; }

/* ---------- stats strip: one panel, cells divided by hairlines, reflows by width ---------- */
.dw-cards-wrap { container-type:inline-size; margin-bottom:18px; }
.dw-cards { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); background:var(--card);
  border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow); }
/* Column counts always divide the number of cells evenly, so a row is never left half empty. */
@container (min-width: 460px) { .dw-cards.n3, .dw-cards.n5, .dw-cards.n6 { grid-template-columns:repeat(3, minmax(0, 1fr)); } }
@container (min-width: 620px) { .dw-cards.n4 { grid-template-columns:repeat(4, minmax(0, 1fr)); } }
@container (min-width: 860px) { .dw-cards.n5 { grid-template-columns:repeat(5, minmax(0, 1fr)); } }
@container (min-width: 960px) { .dw-cards.n6 { grid-template-columns:repeat(6, minmax(0, 1fr)); } }
.dw-cards.n1 { grid-template-columns:1fr; }
.dw-card { padding:16px 18px; min-width:0; box-shadow:1px 0 0 0 var(--line), 0 1px 0 0 var(--line); }
.dw-card .label { display:flex; align-items:center; gap:7px; font-size:12.5px; font-weight:500; color:var(--ink-2);
  line-height:1.3; }
.dw-card .dot { width:7px; height:7px; border-radius:50%; flex:0 0 7px; }
.dw-card .value { font-size:25px; font-weight:700; color:var(--ink); line-height:1.2; margin-top:8px;
  letter-spacing:-0.03em; overflow-wrap:anywhere; }
.dw-card .value.text { font-size:17px; letter-spacing:-0.01em; line-height:1.3; }
.dw-card .value small { font-size:13px; font-weight:600; color:var(--ink-3); margin-left:4px; letter-spacing:0; }
.dw-card .sub { font-size:12.5px; color:var(--ink-3); margin-top:4px; line-height:1.4; }
.dw-card .sub:empty { display:none; }

.dw-pill { display:inline-flex; align-items:center; gap:5px; font-size:12px; font-weight:600; border-radius:999px;
  padding:3px 10px 3px 8px; white-space:nowrap; }
.dw-pill svg { flex:0 0 auto; }
.dw-section { font-size:16px; font-weight:700; color:var(--ink); margin:8px 0 10px 0; letter-spacing:-0.01em; }
.dw-note { font-size:13px; color:var(--ink-2); line-height:1.55; margin-bottom:10px; }
details.dw-details { font-size:13px; color:var(--ink-2); line-height:1.55; margin:2px 0 12px 0; }
details.dw-details summary { cursor:pointer; color:var(--ink-3); font-weight:500; width:fit-content; list-style:none; }
details.dw-details summary::-webkit-details-marker { display:none; }
details.dw-details summary::before { content:"?"; display:inline-flex; align-items:center; justify-content:center;
  width:16px; height:16px; border-radius:50%; border:1px solid #cbd5e1; font-size:10.5px; font-weight:700; margin-right:7px; }
details.dw-details summary:hover { color:var(--brand); }
details.dw-details[open] summary { margin-bottom:6px; }
details.dw-details > div { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px 14px; max-width:920px; }
.dw-note code { background:#eef1f7; padding:1px 6px; border-radius:6px; font-size:12px; }
.dw-footer { margin-top:30px; padding-top:14px; border-top:1px solid var(--line); font-size:12px; color:var(--ink-3);
  display:flex; gap:8px 20px; flex-wrap:wrap; }
.dw-footer b { color:var(--ink-2); font-weight:600; }
.dw-empty { border:1px dashed #cdd5e3; border-radius:var(--radius); padding:30px; text-align:center; color:var(--ink-2);
  background:#fafbfe; font-size:14px; }
.dw-empty b { display:block; color:var(--ink); font-size:15px; margin-bottom:4px; }

/* ---------- Streamlit widgets ---------- */
div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid var(--line); border-radius:var(--radius);
  padding:10px 10px 4px 10px; box-shadow:var(--shadow); }
div[data-testid="stDataFrame"] { border-radius:14px; box-shadow:var(--shadow); }
div[data-testid="stExpander"] details { border-radius:14px; border-color:var(--line); background:#fff; }
div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:var(--radius); }
.stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius:10px; font-weight:600;
  transition:transform .08s ease, box-shadow .15s ease; }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"], .stFormSubmitButton button[kind="primaryFormSubmit"] {
  background:linear-gradient(135deg, #4f46e5, #6366f1); border:0; box-shadow:0 8px 18px -10px #4f46e5; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover { transform:translateY(-1px); }
[data-testid="stTabs"] [role="tablist"] { gap:4px; background:#e9edf5; padding:4px; border-radius:12px; width:fit-content;
  border-bottom:0 !important; box-shadow:none !important; margin-bottom:8px; }
[data-testid="stTabs"] [role="tablist"]::before, [data-testid="stTabs"] [role="tablist"]::after { display:none !important; }
[data-testid="stTabs"] [role="tab"] { border-radius:9px; padding:7px 14px; height:auto; border:0 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background:#fff; box-shadow:0 1px 3px rgba(15,23,42,.12); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { border-right:0; }
[data-testid="stSidebarNav"] a { border-radius:10px; margin:1px 0; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:linear-gradient(135deg, rgba(99,102,241,.38), rgba(99,102,241,.16)); }
.dw-user { display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:12px; background:rgba(255,255,255,.06);
  margin-top:4px; }
.dw-user .avatar { width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#22d3ee); color:#fff;
  font-weight:700; font-size:13px; display:flex; align-items:center; justify-content:center; flex:0 0 34px; }
.dw-user .who { font-size:13px; font-weight:600; color:#f1f5f9; line-height:1.25; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.dw-user .role { font-size:11.5px; color:#94a3b8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
/* ---------- responsive ---------- */
[data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 118px); }
[data-testid="stMainBlockContainer"] [data-testid="stSegmentedControl"] { max-width:100%; overflow-x:auto; }
@media (max-width: 1100px) {
  .dw-header { align-items:flex-start; flex-direction:column; gap:10px; margin-bottom:16px; }
  .dw-gauge { max-width:none; }
}
@media (max-width: 640px) {
  .block-container { padding-top:1.2rem; }
  .dw-card { padding:13px 14px; }
  .dw-card .value { font-size:21px; }
  .dw-hero { padding:15px 16px; gap:16px; }
  .dw-hero .badge { display:none; }
  .dw-footer { gap:4px 14px; }
}
.dw-side-label { font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#64748b; margin:6px 0 -6px 2px; }

/* Optivolt single permanent dark-blue navigation sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
  background: linear-gradient(180deg, #11213e 0%, #1b2858 54%, #292477 100%) !important;
}
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.11) !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
[data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
[data-testid="stSidebar"] [data-testid="stSidebarNav"] svg { color: #ecf3ff !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
  background: rgba(255,255,255,.14) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]:hover {
  background: linear-gradient(100deg, #514af0, #6764ed) !important;
  color: white !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] * { color: white !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] { background: transparent !important; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { background: transparent !important; }
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg { color:#e2e8f0 !important; }
[data-testid="stSidebar"] .dw-user { background: rgba(255,255,255,.09); }
[data-testid="stSidebar"] .dw-side-label { color: #b9c8e9 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #bfcbdf !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def _tone_vars(tone: str) -> str:
    t = TONES[tone]
    return f"--tone-fg:{t['fg']};--tone-bg:{t['bg']};--tone-line:{t['line']};--tone-soft:{t['soft']}"


# ------------------------------------------------------------------- formatting
def fmt_kw(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.1f}"


def fmt_time(value, with_date: bool = True) -> str:
    if value is None or pd.isna(value):
        return "None expected"
    return pd.Timestamp(value).strftime("%a %d %b, %H:%M" if with_date else "%H:%M")


def fmt_duration(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    minutes = int(round(pd.Timedelta(value).total_seconds() / 60.0))
    hours, mins = divmod(minutes, 60)
    return f"{hours} h {mins:02d} min" if hours else f"{mins} min"


# ------------------------------------------------------------------- components
def pill(text: str, tone: str) -> str:
    t = TONES[tone]
    mark = icon(t["icon"], 12) if t["icon"] else ""
    return f'<span class="dw-pill" style="color:{t["fg"]};background:{t["bg"]}">{mark}{esc(text)}</span>'


def status_pill(status: str) -> str:
    return pill(status, STATUS_TONE.get(status, "neutral"))


def page_header(title: str, subtitle: str, organisation: str, chips: list[tuple[str, str]] | None = None) -> None:
    chip_html = "".join(f'<span class="dw-chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in (chips or []))
    st.markdown(
        f'<div class="dw-header"><div><div class="org">{esc(organisation)}</div><h1>{esc(title)}</h1>'
        f'<div class="sub">{esc(subtitle)}</div></div><div class="dw-chips">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def _gauge(utilisation: float, alpha: float, tone: str) -> str:
    """Forecast peak as a share of capacity, with the warning threshold and capacity marked."""
    def pos(share: float) -> float:
        return max(0.0, min(share / GAUGE_MAX, 1.0)) * 100

    return (
        '<div class="dw-gauge"><div class="g-top"><span>Forecast peak vs capacity</span>'
        f'<b>{utilisation:.0%}</b></div><div class="track">'
        f'<div class="fill" style="width:{pos(utilisation):.1f}%"></div>'
        f'<div class="mark" style="left:{pos(alpha):.1f}%;background:{TONES["warning"]["line"]}"></div>'
        f'<div class="mark" style="left:{pos(1.0):.1f}%;background:{TONES["critical"]["line"]}"></div></div>'
        f'<div class="g-foot"><span><i style="background:{TONES["warning"]["line"]}"></i>Warning at {alpha:.0%}</span>'
        f'<span><i style="background:{TONES["critical"]["line"]}"></i>Capacity at 100%</span></div></div>'
    )


def status_banner(status: str, text: str, utilisation: float | None = None, alpha: float | None = None) -> None:
    """Headline status. With ``utilisation`` and ``alpha`` it also shows the capacity gauge."""
    tone = STATUS_TONE[status]
    gauge = _gauge(utilisation, alpha, tone) if utilisation is not None and alpha is not None else ""
    st.markdown(
        f'<div class="dw-hero" style="{_tone_vars(tone)}"><div class="left">'
        f'<div class="badge">{icon(TONES[tone]["icon"], 26)}</div>'
        f'<div><div class="eyebrow">Operational status</div><div class="title">{esc(STATUS_HEADLINE[status])}</div>'
        f'<div class="text">{esc(text)}</div></div></div>{gauge}</div>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str, unit: str = "", sub: str = "", tone: str = "neutral", icon_name: str = "") -> str:
    """One cell of a stats strip. A coloured dot marks a non-neutral operational state."""
    unit_html = f"<small>{esc(unit)}</small>" if unit else ""
    value_class = " text" if len(str(value)) > 9 else ""  # words, dates and times use a smaller size
    dot = f'<i class="dot" style="background:{TONES[tone]["line"]}"></i>' if tone != "neutral" else ""
    value_html = f'<div class="value{value_class}">{esc(value)}{unit_html}</div>' if value != "" else ""
    return (f'<div class="dw-card"><div class="label">{dot}{esc(label)}</div>{value_html}'
            f'<div class="sub">{sub}</div></div>')


def cards(items: list[str], columns: int | None = None) -> None:
    """A strip of stats in one panel. Cells reflow to as many columns as the width allows."""
    st.markdown(f'<div class="dw-cards-wrap"><div class="dw-cards n{min(len(items), 6)}">{"".join(items)}</div></div>',
                unsafe_allow_html=True)


def risk_cards(risk, current_kw: float, columns: int = 6) -> None:
    """The six headline figures used on Overview and Live Forecast."""
    tone = STATUS_TONE[risk.status]
    lead_label = ("to first breach" if risk.breach_time is not None
                  else "to warning level" if risk.warning_time is not None else "No warning in the horizon")
    breach_sub = (f"{risk.breach_intervals} interval(s) at or above capacity" if risk.breach_time is not None
                  else f"Warning level from {fmt_time(risk.warning_time)}" if risk.warning_time is not None else "")
    cards([
        card("Current demand", fmt_kw(current_kw), "kW",
             f"{current_kw / risk.capacity_kw:.0%} of capacity" if risk.capacity_kw else ""),
        card("Predicted peak", fmt_kw(risk.peak_kw), "kW", fmt_time(risk.peak_time), tone),
        card("Grid capacity", fmt_kw(risk.capacity_kw), "kW", f"Warning at {fmt_kw(risk.warning_kw)} kW"),
        card("Status", "", "", status_pill(risk.status)),
        card("Expected breach time", fmt_time(risk.breach_time) if risk.breach_time is not None else "None", "",
             breach_sub, "critical" if risk.breach_time is not None else "neutral"),
        card("Warning lead time", fmt_duration(risk.lead_time) if risk.lead_time is not None else "n/a", "", lead_label,
             tone if risk.lead_time is not None else "neutral"),
    ], columns)


def section(title: str) -> None:
    st.markdown(f'<div class="dw-section">{esc(title)}</div>', unsafe_allow_html=True)


def note(text: str, title: str | None = None) -> None:
    """Supporting text. With ``title`` it is collapsed behind a small help link to keep the page clean."""
    if title:
        st.markdown(f'<details class="dw-details"><summary>{esc(title)}</summary><div>{text}</div></details>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="dw-note">{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, text: str) -> None:
    st.markdown(f'<div class="dw-empty"><b>{esc(title)}</b>{esc(text)}</div>', unsafe_allow_html=True)


def footer(model_version: str, generated_at: datetime, data_through, source_label: str) -> None:
    st.markdown(
        '<div class="dw-footer">'
        f"<span><b>{esc(config.PRODUCT_NAME)}</b></span>"
        f"<span>Model version {esc(model_version)}</span>"
        f"<span>Forecast generated {generated_at:%d %b %Y %H:%M}</span>"
        f"<span>Data through {pd.Timestamp(data_through):%d %b %Y %H:%M}</span>"
        f"<span>{esc(source_label)}</span></div>",
        unsafe_allow_html=True,
    )
