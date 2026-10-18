"""Public landing page shown before sign-in."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from dashboard import charts, database, services
from dashboard import theme as ui

PUBLIC_CSS = """
<style>
/* The public navigation sidebar remains visible. */
.block-container { max-width: 1240px; padding-top: 1.4rem; }
.lp-brand { display:flex; align-items:center; gap:10px; font-size:20px; font-weight:700; color:#0f172a; letter-spacing:-0.02em; }
.st-key-lp_hero { background: radial-gradient(1200px 500px at 85% -10%, rgba(34,211,238,.28), transparent 60%),
  linear-gradient(135deg, #0b1020 0%, #1e1b4b 55%, #3730a3 100%); border-radius:28px; padding:54px 48px 48px 48px;
  margin-top:14px; box-shadow:0 30px 60px -30px rgba(30,27,75,.6); }
.lp-eyebrow { display:inline-flex; align-items:center; gap:8px; font-size:12.5px; font-weight:600; color:#c7d2fe;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); border-radius:999px; padding:6px 12px; }
.lp-eyebrow i { width:7px; height:7px; border-radius:50%; background:#34d399; display:inline-block; }
.lp-h1 { font-size:50px; line-height:1.06; font-weight:800; color:#fff; letter-spacing:-0.04em; margin:18px 0 0 0; }
.lp-h1 span { background:linear-gradient(90deg,#a5b4fc,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent; }
.lp-lead { font-size:17px; line-height:1.6; color:#cbd5e1; margin:18px 0 22px 0; max-width:520px; }
.lp-trust { font-size:12.5px; color:#94a3b8; margin-top:14px; }
.st-key-lp_hero .stButton button[kind="secondary"] { background:rgba(255,255,255,.08); color:#fff; border:1px solid rgba(255,255,255,.22); }
.st-key-lp_hero .stButton button[kind="primary"] { background:#fff; color:#312e81; box-shadow:0 12px 24px -12px rgba(255,255,255,.55); }
.st-key-lp_hero div[data-testid="stPlotlyChart"] { border:0; border-radius:18px; box-shadow:0 30px 60px -20px rgba(2,6,23,.7); padding:14px 12px 6px 12px; }
.lp-preview-head { display:flex; justify-content:space-between; align-items:center; color:#e2e8f0; font-size:13px; font-weight:600; margin-bottom:10px; }
.lp-preview-head span.live { font-size:11.5px; font-weight:600; color:#a7f3d0; background:rgba(16,185,129,.16); border-radius:999px; padding:3px 10px; }
.lp-stats { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; margin:26px 0 8px 0; }
.lp-stat { background:#fff; border:1px solid #e6eaf2; border-radius:18px; padding:20px 22px; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10); }
.lp-stat b { display:block; font-size:32px; font-weight:800; color:#0f172a; letter-spacing:-0.04em; line-height:1.1; }
.lp-stat span { font-size:13.5px; color:#475569; }
.lp-title { text-align:center; margin:54px 0 6px 0; font-size:32px; font-weight:800; letter-spacing:-0.03em; color:#0f172a; }
.lp-sub { text-align:center; font-size:16px; color:#475569; margin-bottom:26px; }
.lp-grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px; }
.lp-feature { background:#fff; border:1px solid #e6eaf2; border-radius:18px; padding:24px; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10); }
.lp-feature .ic { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; color:#4338ca; background:#eef2ff; margin-bottom:14px; }
.lp-feature h3 { font-size:17px; font-weight:700; margin:0 0 6px 0; padding:0; color:#0f172a; letter-spacing:-0.01em; }
.lp-feature p { font-size:14px; line-height:1.6; color:#475569; margin:0; }
.lp-steps { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px; }
.lp-step { border-radius:18px; padding:24px; background:linear-gradient(180deg,#eef2ff,#ffffff); border:1px solid #e0e7ff; }
.lp-step .n { width:32px; height:32px; border-radius:50%; background:#4f46e5; color:#fff; font-weight:700; display:flex; align-items:center; justify-content:center; font-size:14px; margin-bottom:12px; }
.lp-step h3 { font-size:16px; font-weight:700; margin:0 0 6px 0; padding:0; color:#0f172a; }
.lp-step p { font-size:14px; line-height:1.6; color:#475569; margin:0; }
.st-key-lp_cta { background:linear-gradient(135deg,#4f46e5,#0ea5e9); border-radius:24px; padding:36px 40px; margin-top:54px; }
.lp-cta-text { color:#fff; font-size:26px; font-weight:800; letter-spacing:-0.03em; line-height:1.2; }
.lp-cta-text small { display:block; font-size:15px; font-weight:500; color:#e0e7ff; margin-top:6px; letter-spacing:0; }
.st-key-lp_cta .stButton button { background:#fff; color:#312e81; border:0; }
.lp-foot { text-align:center; font-size:12.5px; color:#94a3b8; margin:34px 0 10px 0; line-height:1.7; }
.st-key-lp_hero [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 330px); }
.st-key-lp_hero [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 110px); }
.st-key-lp_cta [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 220px); }
.lp-h1 { font-size:clamp(32px, 4.2vw, 50px) !important; }
.lp-title { font-size:clamp(24px, 3vw, 32px) !important; }
.lp-stats { grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)) !important; }
.lp-grid, .lp-steps { grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)) !important; }
@media (max-width: 900px) { .lp-h1 { font-size:36px; } .lp-stats, .lp-grid, .lp-steps { grid-template-columns:1fr; } .st-key-lp_hero { padding:32px 22px; } }
</style>
"""

FEATURES = [
    ("trend", "Demand forecasts up to 72 hours ahead",
     "Saved Optivolt LSTM models generate 15-minute demand forecasts from historical depot readings."),
    ("alert", "Capacity risk warnings",
     "Assess when a historical-data-based LSTM forecast crosses a configurable planning threshold."),
    ("sliders", "Fleet growth scenarios",
     "Explore hypothetical fleet growth and capacity assumptions against the LSTM forecast."),
    ("layers", "Every depot in one view",
     "One status table for all sites, each with its own capacity and warning threshold."),
    ("upload", "Your own private workspace",
     "Each organisation has its own depots, users and history. Upload supported meter data for validation and Optivolt model training."),
    ("file", "Reports and audit trail",
     "Download Excel, PDF or CSV reports. Access reports and verified model evaluation where available."),
]
STEPS = [
    ("Create your workspace", "Sign up for your organisation, explore the sample depots, then upload your own meter data."),
    ("See the forecast", "The saved LSTM predicts intervals after the most recent historical observation."),
    ("Act with lead time", "Explore charging and fleet growth planning scenarios using historical forecasts."),
]


def go(view: str) -> None:
    target = st.session_state.get("_public_nav_pages", {}).get(view)
    if target is not None:
        st.switch_page(target)
    else:
        st.session_state["public_view"] = view



def _model_facts() -> dict:
    """Facts from bundled real Optivolt LSTM data, without loading the legacy model."""
    return {"depots": "9", "improvement": None, "interval": "15"}


def _preview_chart() -> None:
    """Show the saved Bexleyheath LSTM recursive forecast; not a live prediction."""
    from pathlib import Path
    import plotly.graph_objects as go
    root = Path(__file__).resolve().parents[2]
    saved = root / "lstm_24h_recursive_forecast.csv"
    source = root / "UKPN_OptimisePrime_Bexleyheath.csv"
    if not saved.is_file() or not source.is_file():
        st.caption("Historical LSTM forecast preview is unavailable.")
        return
    try:
        prediction = pd.read_csv(saved, parse_dates=["timestamp"])
        actual = pd.read_csv(source, usecols=["timestamp", "demand_kw"], parse_dates=["timestamp"])
        actual["demand_kw"] = pd.to_numeric(actual["demand_kw"], errors="coerce")
        first = prediction["timestamp"].min()
        window = actual[(actual["timestamp"] >= first - pd.Timedelta(hours=24)) &
                        (actual["timestamp"] < first)].dropna().sort_values("timestamp")
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=window["timestamp"], y=window["demand_kw"],
                                   mode="lines", name="Observed demand", line=dict(color="#94a3b8", width=2)))
        chart.add_trace(go.Scatter(x=prediction["timestamp"], y=prediction["lstm_forecast_kw"],
                                   mode="lines", name="Saved LSTM forecast", line=dict(color="#818cf8", width=3)))
        chart.add_hline(y=17.0, line_dash="dash", line_color="#f87171",
                        annotation_text="Planning capacity (17 kW)")
        chart.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0"), margin=dict(l=20,r=15,t=10,b=20),
                            xaxis=dict(showgrid=False),yaxis=dict(title="Demand (kW)",gridcolor="rgba(255,255,255,.1)"),
                            legend=dict(orientation="h", y=1.2))
        st.markdown('<div class="lp-preview-head"><span>Bexleyheath · saved 24-hour LSTM forecast (July 2022)</span>'
                    '<span class="live">Historical model output</span></div>',unsafe_allow_html=True)
        st.plotly_chart(chart, config={"displayModeBar":False}, key="landing_preview")
    except (ValueError, KeyError, OSError) as exc:
        st.caption(f"Historical LSTM chart unavailable: {exc}")


def render() -> None:
    st.markdown(PUBLIC_CSS, unsafe_allow_html=True)
    facts = _model_facts()
    signup = database.signup_allowed()

    brand, _, nav_a, nav_b = st.columns([5, 3, 1.1, 1.5], vertical_alignment="center")
    brand.markdown(f'<div class="lp-brand">{ui.LOGO_SVG.format(size=36)}{ui.esc(config.PRODUCT_NAME)}</div>',
                   unsafe_allow_html=True)
    nav_a.button("Sign in", key="nav_signin", on_click=go, args=("signin",), width="stretch")
    if signup:
        nav_b.button("Create account", key="nav_signup", type="primary", on_click=go, args=("signup",), width="stretch")

    with st.container(key="lp_hero"):
        left, right = st.columns([1, 1.08], gap="large", vertical_alignment="center")
        with left:
            st.markdown(
                '<span class="lp-eyebrow"><i></i>EV depot demand forecasting</span>'
                '<h1 class="lp-h1">Understand your fleet’s <span>capacity risk</span></h1>'
                '<p class="lp-lead">Optivolt uses saved LSTM forecasts from historical depot demand to compare demand '
                'against configurable planning capacities and support fleet decisions.</p>',
                unsafe_allow_html=True)
            a, b, _ = st.columns([1.25, 1, 0.6])
            a.button("Get started" if signup else "Sign in", key="hero_primary", type="primary", on_click=go,
                     args=("signup" if signup else "signin",), width="stretch", icon=":material/arrow_forward:")
            if signup:
                b.button("Sign in", key="hero_signin", on_click=go, args=("signin",), width="stretch")
            st.markdown(f'<div class="lp-trust">Trained on real metered data from {facts["depots"]} Royal Mail depots '
                        '(UK Power Networks, Optimise Prime).</div>', unsafe_allow_html=True)
        with right:
            _preview_chart()

    improvement = (f'<div class="lp-stat"><b>{facts["improvement"]}</b><span>lower forecast error (MAE) than the '
                   'previous-day baseline on held-out data</span></div>') if facts["improvement"] else ""
    st.markdown(
        '<div class="lp-stats">'
        f'<div class="lp-stat"><b>{facts["depots"]}</b><span>real depots in the training data</span></div>'
        f'<div class="lp-stat"><b>{facts["interval"]} min</b><span>forecast resolution</span></div>'
        f'<div class="lp-stat"><b>{max(config.HORIZON_CHOICES)} h</b><span>maximum forecast horizon</span></div>'
        f'{improvement}</div>', unsafe_allow_html=True)

    st.markdown('<div class="lp-title">Everything a depot needs to stay within capacity</div>'
                '<div class="lp-sub">Forecast, warn, plan and report from one place.</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-grid">' + "".join(
        f'<div class="lp-feature"><div class="ic">{ui.icon(name, 22)}</div><h3>{ui.esc(title)}</h3><p>{ui.esc(text)}</p></div>'
        for name, title, text in FEATURES) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="lp-title">How it works</div>'
                '<div class="lp-sub">From meter data to an early warning in three steps.</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-steps">' + "".join(
        f'<div class="lp-step"><div class="n">{i}</div><h3>{ui.esc(title)}</h3><p>{ui.esc(text)}</p></div>'
        for i, (title, text) in enumerate(STEPS, 1)) + "</div>", unsafe_allow_html=True)

    with st.container(key="lp_cta"):
        text, action = st.columns([3, 1], vertical_alignment="center")
        text.markdown('<div class="lp-cta-text">See your depots\' capacity outlook'
                      '<small>Sign in to open the dashboard for your organisation.</small></div>', unsafe_allow_html=True)
        action.button("Open the dashboard", key="cta_signin", on_click=go, args=("signin",), width="stretch",
                      icon=":material/arrow_forward:")

    st.markdown(f'<div class="lp-foot">{ui.esc(config.PRODUCT_NAME)} &middot; {ui.esc(config.PRODUCT_TAGLINE[0].upper() + config.PRODUCT_TAGLINE[1:])}<br>'
                'Depot data: UK Power Networks, Project Optimise Prime, licensed CC BY 4.0.</div>', unsafe_allow_html=True)
