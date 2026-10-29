"""Optivolt integrated EV depot forecasting and management.

Run with:  streamlit run app.py

Flow: public landing page -> sign in / create workspace -> dashboard.
Roles: platform owner (operates the service), organisation administrator, organisation viewer.
The application uses saved LSTM models and supports explicit Optivolt retraining.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

import config
config.PRODUCT_NAME = "Optivolt EV Depot Intelligence Portal"
from dashboard import database, services
from dashboard import theme as ui
from dashboard.pages import (alerts, depots, history, landing, live_forecast, login, overview, performance,
                             platform, reports, scenario, settings)

ICON = str(config.ASSETS_DIR / "icon.svg")
st.set_page_config(page_title=config.PRODUCT_NAME, page_icon=ICON, layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
st.markdown("""<style>
.ov-eyebrow{font-size:.78rem;font-weight:750;letter-spacing:.12em;color:#4f46e5;margin-bottom:.3rem}
[data-testid="stMetric"]{background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
  padding:14px 16px;box-shadow:0 4px 14px rgba(15,23,42,.035)}
[data-testid="stSidebarNav"] a{border-radius:9px}
/* Full width authenticated content: no 1400px cap and no large vacant right gutter. */
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] .block-container,
section.main .block-container {
  width: 100% !important;
  max-width: none !important;
  padding-left: clamp(1rem, 2.25vw, 2.5rem) !important;
  padding-right: clamp(1rem, 2.25vw, 2.5rem) !important;
}
[data-testid="stMainBlockContainer"] > div {max-width: none !important;}
[data-testid="stMetric"] {height:100%;}
@media(max-width:850px){[data-testid="stMain"] .block-container{padding-left:1rem!important;padding-right:1rem!important;}}

</style>""", unsafe_allow_html=True)


def public_site() -> None:
    """DepotWatt original public login appearance, Optivolt LSTM wording."""
    login.render(st.session_state.get("public_view", "signin"))


# Each original Optivolt screen now has a visible primary navigation entry.
# The underlying forecast, preprocessing, saved models and evaluation calculations
# remain in the original, unmodified (apart from navigation) LSTM module.
OPTIVOLT_SECTIONS = [
    ("Overview", "optivolt-overview", ":material/home:", "🏠 Home / Overview"),
    ("Depot Setup", "optivolt-depot-setup", ":material/warehouse:", "🏢 Depots Setup"),
    ("Upload Data", "optivolt-upload-data", ":material/upload_file:", "📊 Upload Depot Data"),
    ("Demand Forecast", "optivolt-demand-forecast", ":material/show_chart:", "📈 Demand Forecast"),
    ("Scenario Analysis", "optivolt-scenarios", ":material/tune:", "🚗 Scenario Analysis"),
    ("Model Performance", "optivolt-model-performance", ":material/analytics:", "⚙️ Model Performance"),
    ("System Alerts", "optivolt-system-alerts", ":material/notifications:", "🚨 System Alerts"),
    ("Reports", "optivolt-reports", ":material/description:", "📋 Reports"),
    ("Archive History", "optivolt-archive-history", ":material/history:", "📜 Archive History"),
    ("Admin Settings", "optivolt-admin-settings", ":material/settings:", "⚙️ Settings / Admin"),
]


# -----------------------------------------------------------------------------
# Optivolt LSTM dashboard surfaces. All numbers below are computed from the
# named depot's saved Keras model + original metered data. No XGBoost output is
# reused or relabelled. Legacy upload/retraining screens remain accessible.
# -----------------------------------------------------------------------------
import io
import zipfile
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

_PLANNING_CAPACITY = {
    'Bexleyheath':17., 'Dartford':45., 'Islington':13., 'Mount Pleasant':260.,
    'Orpington':14., 'Premier Park':70., 'Whitechapel':55., 'Camden':21., 'Victoria':29.
}


def _site_name():
    name = st.session_state.get('optivolt_portfolio_selection', 'All 9 Depots')
    return name if name != 'All 9 Depots' else None


def _profile(site, steps=96):
    """One site's historical observations and genuine LSTM multi-step forecast."""
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    folder = Path(__file__).parent
    slug = site.lower().replace(' ', '_')
    path = folder / f'UKPN_OptimisePrime_{site.replace(" ", "_")}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing UKPN CSV: {path.name}')
    source = pd.read_csv(path, usecols=['timestamp','demand_kw'], parse_dates=['timestamp'])
    hist = (source.dropna(subset=['timestamp','demand_kw'])
            .groupby('timestamp')['demand_kw'].mean().sort_index().clip(lower=0))
    hist = hist.resample('15min').mean().interpolate(limit=4).ffill().dropna()
    if len(hist) < 96:
        raise ValueError('At least 96 historical 15-minute intervals are required')
    model_file = folder / 'models' / (slug + '.keras')
    scaler_file = folder / 'models' / (slug + '_scalers.npz')
    if not model_file.exists() or not scaler_file.exists():
        raise FileNotFoundError('Saved LSTM model/scalers not found for ' + site)
    model = tf.keras.models.load_model(model_file, compile=False)
    with np.load(scaler_file) as bundle:
        xm, xs = float(bundle['x_mean'][0]), float(bundle['x_scale'][0])
        ym, ys = float(bundle['y_mean'][0]), float(bundle['y_scale'][0])
    xs = max(xs, 1e-12)
    values = list(hist.to_numpy(dtype=float))
    forecasts = []
    # Direct inference avoids spinning up a full Keras predict loop for every interval.
    for i in range(steps):  # 15-minute recursive intervals
        window = (np.asarray(values[-96:], dtype=np.float32) - xm) / xs
        predicted = float(model(window.reshape(1,96,1), training=False).numpy()[0,0]) * ys + ym
        predicted = max(0., predicted)
        forecasts.append(predicted)
        values.append(predicted)
    times = pd.date_range(hist.index[-1] + pd.Timedelta(minutes=15), periods=steps, freq='15min')
    return hist, pd.Series(forecasts,index=times,name='LSTM forecast')


@st.cache_resource(show_spinner=False)
def _cached_profile(site, steps=96):
    return _profile(site, steps)


def _capacity(site):
    # User-configured value, if available, supersedes project planning default.
    raw = _PLANNING_CAPACITY.get(site, 17.)
    if st.session_state.get('current_depot') == site:
        raw = st.session_state.get('grid_limit', raw)
    return float(raw)


def _status(values, capacity, warning_factor=.8):
    peak = float(np.max(values))
    return ('Breach' if peak >= capacity else
            'Warning' if peak >= capacity * warning_factor else 'Within capacity')


def _kpi(label, value, detail=''):
    st.metric(label, value, help=detail if detail else None)
    if detail:
        st.caption(detail)


def _chart(site, hist, forecast, capacity, scenario=None, history_hours=48):
    shown = hist.iloc[-max(1,int(history_hours*4)):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shown.index, y=shown.values, name='Observed demand', line=dict(color='#111827',width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='LSTM forecast',
                             line=dict(color='#5046EB',width=3),fill='tozeroy',fillcolor='rgba(80,70,235,.09)'))
    if scenario is not None:
        fig.add_trace(go.Scatter(x=forecast.index,y=scenario,name='Scenario-adjusted forecast (hypothetical)',
                                 line=dict(color='#089e90',width=3)))
    fig.add_hline(y=capacity*.8,line_dash='dash',line_color='#F59E0B',
                  annotation_text=f'Warning threshold ({capacity*.8:g} kW)')
    fig.add_hline(y=capacity,line_color='#EF4444',annotation_text=f'Planning capacity ({capacity:g} kW)')
    fig.add_vline(x=forecast.index[0].to_pydatetime().timestamp()*1000,
                  line_dash='dot',line_color='#94A3B8')
    fig.update_layout(height=420, margin=dict(l=16,r=20,t=30,b=25),
                      legend=dict(orientation='h',y=1.13),yaxis_title='Demand (kW)',
                      xaxis_title=None,hovermode='x unified')
    st.plotly_chart(fig,use_container_width=True)


def _lstm_snapshot(site, hist, pred, capacity, include_chart=True):
    peak = float(pred.max())
    status = _status(pred,capacity)
    br = pred[pred >= capacity]
    warn = pred[pred >= .8*capacity]
    expected = str(br.index[0].strftime('%d %b %Y %H:%M')) if not br.empty else 'None'
    lead = str(warn.index[0]-hist.index[-1]) if not warn.empty else 'N/A'
    st.subheader(f'{site} · LSTM capacity outlook')
    st.info(f'**{status}** · Predicted peak {peak:.1f} kW · '
            f'Planning headroom {capacity-peak:+.1f} kW. Forecast starts {pred.index[0]:%d %b %Y %H:%M}.')
    cols=st.columns(3)
    with cols[0]: _kpi('Last observed demand',f'{hist.iloc[-1]:.1f} kW',f'Data through {hist.index[-1]:%d %b %Y %H:%M}')
    with cols[1]: _kpi('Predicted LSTM peak',f'{peak:.1f} kW',f'At {pred.idxmax():%d %b %Y %H:%M}')
    with cols[2]: _kpi('Planning capacity',f'{capacity:.1f} kW','Application planning value, not a certified grid connection limit')
    cols=st.columns(3)
    with cols[0]: _kpi('Forecast status',status)
    with cols[1]: _kpi('Expected capacity breach',expected)
    with cols[2]: _kpi('Time until first warning',lead,'Forecast-relative interval, not measured warning performance')
    if include_chart:
        _chart(site,hist,pred,capacity)


def _portfolio_lstm():
    st.title('Optivolt Overview · All 9 Depots')
    st.caption('Historical UKPN data and LSTM forecasts. Computing nine models can take a while on the first visit.')
    rows=[]
    progress=st.progress(0)
    for i,(site,capacity) in enumerate(_PLANNING_CAPACITY.items()):
        try:
            hist,pred_all=_cached_profile(site)
            pred=pred_all.iloc[:96]
            rows.append({'Depot':site,'Last observed (kW)':round(float(hist.iloc[-1]),1),
                         'LSTM predicted peak (kW)':round(float(pred.max()),1),
                         'Planning capacity (kW)':capacity,
                         'Headroom (kW)':round(capacity-float(pred.max()),1),
                         'Status':_status(pred,capacity)})
        except Exception as exc:
            rows.append({'Depot':site,'Status':'Unavailable: '+str(exc)[:90]})
        progress.progress((i+1)/len(_PLANNING_CAPACITY))
    progress.empty()
    frame=pd.DataFrame(rows)
    if 'Status' in frame:
        a,b,c=st.columns(3)
        a.metric('Within capacity',int((frame.Status=='Within capacity').sum()))
        b.metric('Warnings',int((frame.Status=='Warning').sum()))
        c.metric('Breaches',int((frame.Status=='Breach').sum()))
    st.dataframe(frame,use_container_width=True,hide_index=True)
    st.caption('These forecasts extend the July 2022 historical dataset; they are not live telemetry.')


def optivolt_enhanced(section):
    if not st.session_state.get('user'):
        st.stop()
    titles = {'overview':'Overview', 'forecast':'Depot Demand Forecast',
              'scenario':'Scenario Analysis', 'alerts':'LSTM Capacity Alerts',
              'reports':'LSTM Reports and Exports', 'history':'LSTM History and Traceability'}
    if section != 'overview':
        st.title(titles.get(section, 'Optivolt LSTM'))
        st.caption('Historical UKPN Optimise Prime data · saved LSTM models')
    site=_site_name()
    if site is None:
        # Render immediately. Nine recursive neural forecasts are intentionally
        # NOT executed during navigation or when selecting the portfolio.
        optivolt_portfolio()
        return
    # Avoid invoking TensorFlow before any visible controls. Explicit inference
    # also avoids blank pages on resource-limited Streamlit Cloud instances.
    run_key = f'_lstm_generate_{site}'
    if not st.session_state.get(run_key, False):
        st.info(f'{site} is selected. Run the saved LSTM model to populate this page. Historical forecasts are not live telemetry.')
        if st.button('Generate LSTM forecast', type='primary', key=f'lstm_start_{section}_{site}'):
            st.session_state[run_key] = True
            st.rerun()
        if section == 'history':
            st.subheader('Observed historical demand')
            try:
                source = pd.read_csv(Path(__file__).parent / f'UKPN_OptimisePrime_{site.replace(" ", "_")}.csv',
                                     usecols=['timestamp','demand_kw'], parse_dates=['timestamp'])
                source = source.dropna().sort_values('timestamp').tail(96*7)
                st.line_chart(source.set_index('timestamp')['demand_kw'])
            except Exception as exc:
                st.warning(f'Historical demand unavailable: {exc}')
        return
    if st.button('Reset / regenerate forecast', key=f'lstm_reset_{section}_{site}'):
        _cached_profile.clear()
        st.session_state[run_key] = False
        st.rerun()
    try:
        with st.spinner(f'Loading saved LSTM model and forecasting {site} demand. The first request may take a little longer…'):
            hist, full=_cached_profile(site)
    except Exception as exc:
        st.error(f'LSTM forecast unavailable for {site}: {exc}')
        st.info('The original Optivolt Upload Data / Model Retraining pages remain available.')
        return
    capacity=_capacity(site)
    st.caption('OPTIVOLT SOLUTIONS LIMITED · UKPN Optimise Prime historical dataset · Saved LSTM model')
    if section=='overview':
        _lstm_snapshot(site,hist,full.iloc[:96],capacity,True)
    elif section=='forecast':
        col1,col2=st.columns(2)
        with col1: horizon=st.select_slider('Forecast horizon (hours)',options=[6,12,24,48,72],value=24,key='ov_new_horizon')
        with col2: history=st.select_slider('History shown (hours)',options=[24,48,72,168],value=48,key='ov_new_history')
        pred=(_cached_profile(site, int(horizon*4))[1] if horizon>24 else full.iloc[:int(horizon*4)])
        _lstm_snapshot(site,hist,pred,capacity,False)
        _chart(site,hist,pred,capacity,history_hours=history)
        st.caption('Forecasts are recursive LSTM predictions beyond the last historical record, not live grid readings.')
    elif section=='scenario':
        st.caption('Hypothetical demand adjustments applied to the saved LSTM base forecast.')
        c1,c2,c3,c4=st.columns(4)
        with c1: growth=st.slider('Fleet / demand growth (%)',-50,200,25,key='ov_new_growth')
        with c2: factor=st.slider('Utilisation factor',0.25,2.,1.,.05,key='ov_new_utilisation')
        with c3: cap=st.number_input('Planning capacity (kW)',min_value=0.1,value=float(capacity),step=1.,key='ov_new_capacity')
        with c4: horizon=st.selectbox('Forecast horizon (hours)',[6,12,24,48,72],index=2,key='ov_new_scenario_horizon')
        pred=(_cached_profile(site, horizon*4)[1] if horizon>24 else full.iloc[:horizon*4])
        scenario=np.maximum(0,pred.values*(1+growth/100)*factor)
        peak=float(scenario.max()); base=float(pred.max())
        threshold_time=pred.index[np.where(scenario>=cap)[0][0]].strftime('%d %b %H:%M') if np.any(scenario>=cap) else 'None'
        cols=st.columns(4)
        for col,label,value in zip(cols,['Base LSTM peak','Scenario peak','Planning capacity','Scenario status'],
                                   [f'{base:.1f} kW',f'{peak:.1f} kW',f'{cap:.1f} kW',_status(scenario,cap)]):
            col.metric(label,value)
        cols=st.columns(3)
        cols[0].metric('Headroom at peak',f'{cap-peak:+.1f} kW')
        cols[1].metric('Expected capacity breach',threshold_time)
        cols[2].metric('Growth before breach',f'{(cap/base-1)*100:.0f}%' if base>0 else 'N/A')
        _chart(site,hist,pred,float(cap),scenario=scenario)
        st.caption('Scenario values are hypothetical and are not new model predictions.')
    elif section=='alerts':
        horizon=st.selectbox('Alert horizon (hours)',[6,12,24,48,72],index=2,key='ov_new_alert_horizon')
        pred=(_cached_profile(site, horizon*4)[1] if horizon>24 else full.iloc[:horizon*4])
        status=_status(pred,capacity)
        warning=pred[pred>=.8*capacity]; breaches=pred[pred>=capacity]
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Capacity breaches',int(not breaches.empty));c2.metric('Warnings',int(status=='Warning'))
        marker=f'{site}|{pred.index[0].isoformat()}|{status}'
        ack=st.session_state.setdefault('optivolt_acknowledged',set())
        c3.metric('Awaiting acknowledgement',int(status!='Within capacity' and marker not in ack))
        c4.metric('Depots within capacity',int(status=='Within capacity'))
        alerts=[]
        if status!='Within capacity':
            hit=breaches if status=='Breach' else warning
            alerts.append({'Depot':site,'Level':status,'Issued (forecast origin)':str(hist.index[-1]),
                           'Expected at':str(hit.index[0]),'Predicted peak (kW)':round(float(pred.max()),2),
                           'Planning capacity (kW)':capacity,'Acknowledged':marker in ack})
        st.subheader('Active forecast alerts')
        st.dataframe(pd.DataFrame(alerts),use_container_width=True,hide_index=True)
        if alerts and st.button('Acknowledge selected forecast alert',key='ov_new_ack'):
            ack.add(marker);st.success('Acknowledged for this session.');st.rerun()
        st.caption('Alert acknowledgements on this page are session-only. For persisted organisation audit logs, use Workspace Alerts.')
    elif section=='reports':
        horizon=st.selectbox('Forecast horizon (hours)',[6,12,24,48,72],index=2,key='ov_report_horizon')
        pred=(_cached_profile(site, horizon*4)[1] if horizon>24 else full.iloc[:horizon*4])
        summary=pd.DataFrame([{'Depot':site,'Data through':str(hist.index[-1]),'Forecast from':str(pred.index[0]),
                               'Horizon (hours)':horizon,'Last observed (kW)':round(float(hist.iloc[-1]),2),
                               'LSTM peak (kW)':round(float(pred.max()),2),'Planning capacity (kW)':capacity,
                               'Headroom (kW)':round(capacity-float(pred.max()),2),'Risk status':_status(pred,capacity)}])
        intervals=pd.DataFrame({'Timestamp':pred.index,'LSTM forecast (kW)':pred.values,
                                 'Warning threshold (kW)':capacity*.8,'Planning capacity (kW)':capacity})
        st.subheader('Report preview')
        tab1,tab2=st.tabs(['Forecast and capacity-risk summary','Forecast detail (15-minute intervals)'])
        with tab1:st.dataframe(summary,use_container_width=True,hide_index=True)
        with tab2:st.dataframe(intervals,use_container_width=True,hide_index=True)
        csv=io.BytesIO()
        with zipfile.ZipFile(csv,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('lstm_forecast_summary.csv',summary.to_csv(index=False))
            z.writestr('lstm_forecast_intervals.csv',intervals.to_csv(index=False))
        c1,c2=st.columns(2)
        with c1:st.download_button('Download CSV files (.zip)',csv.getvalue(),f'optivolt_{site.lower().replace(" ","_")}_lstm_report.zip',mime='application/zip')
        xl=io.BytesIO()
        with pd.ExcelWriter(xl,engine='openpyxl') as writer:
            summary.to_excel(writer,sheet_name='Forecast summary',index=False)
            intervals.to_excel(writer,sheet_name='Forecast intervals',index=False)
        with c2:st.download_button('Download Excel report',xl.getvalue(),f'optivolt_{site.lower().replace(" ","_")}_lstm_report.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.caption('Metrics are produced from the saved LSTM. Past issued-forecast logs are not fabricated.')
    elif section=='history':
        tabs=st.tabs(['Observed demand','Saved model version','Held-out outcomes','Forecast log','Alert log','Scenario runs'])
        with tabs[0]:
            days=st.select_slider('Historical window (days)',options=[7,14,30,60],value=14)
            st.line_chart(hist.iloc[-days*96:],y_label='Observed demand (kW)')
        with tabs[1]:
            slug=site.lower().replace(' ','_'); p=Path(__file__).parent/'models'/f'{slug}_metadata.json'
            if p.exists():st.json(json.loads(p.read_text()))
            else:st.caption('Saved model artifact is present; no separate metadata file was provided.')
        with tabs[2]:
            source=Path(__file__).parent/'heldout_predictions_all_models.csv'
            if site=='Bexleyheath' and source.exists():
                df=pd.read_csv(source)
                if {'actual_kw','lstm_kw'}.issubset(df.columns):
                    st.metric('Held-out LSTM MAE (kW)',f'{np.mean(np.abs(df.actual_kw-df.lstm_kw)):.3f}')
                    st.dataframe(df.head(100),use_container_width=True,hide_index=True)
                else:st.info('The saved held-out results do not include expected actual_kw and lstm_kw fields.')
            else:st.info('No verified held-out LSTM predictions were bundled for this depot. No metrics are invented.')
        for tab,label in zip(tabs[3:],['forecast','alert','scenario']):
            with tab:st.info(f'Persistent {label} history requires integration with the organisation audit database. No historical entries are invented.')


def optivolt_portfolio() -> None:
    """Portfolio cards and one table of observed UKPN measurements only."""
    from optivolt_portfolio import portfolio_summary

    st.markdown('<div class="ov-eyebrow">OPTIVOLT SOLUTIONS LIMITED · FLEET INTELLIGENCE</div>',
                unsafe_allow_html=True)
    st.title("EV Depot Fleet Capacity & Risk Overview")
    st.caption("Historical UK Power Networks Optimise Prime observations · July 2021–July 2022")
    frame = portfolio_summary(Path(__file__).parent)
    if frame.empty:
        st.warning("The original nine depot CSV files could not be found.")
        return
    exceed = int((frame['Historical peak (kW)'] >= frame['Planning threshold (kW)']).sum())
    a, b, c, d = st.columns(4)
    a.metric("Depots", len(frame))
    b.metric("Metered intervals", f"{int(frame['Observed intervals'].sum()):,}")
    c.metric("Historical threshold exceedances", exceed)
    d.metric("Sampling interval", "15 min")
    st.markdown("#### Fleet Capacity & Demand Comparison")
    data = frame.rename(columns={
        'Mean observed (kW)':'Mean demand (kW)',
        'Planning threshold (kW)':'Planning capacity (kW)',
        'Historical peak / threshold (%)':'Historical utilisation (%)',
        'Last observed':'Data through',
    }).copy()
    data['Historical status'] = np.where(
        data['Historical peak (kW)'] >= data['Planning capacity (kW)'],
        'Exceeded threshold', 'Below threshold')
    st.dataframe(data[['Depot','Mean demand (kW)','Historical peak (kW)',
                       'Planning capacity (kW)','Historical utilisation (%)',
                       'Historical status','Data through']],
                 hide_index=True, use_container_width=True)
    with st.expander("About this portfolio", expanded=False):
        st.write("Values are measured historical demand, not live readings or LSTM predictions. "
                 "A historical threshold exceedance is not a forecast alert. "
                 "Planning capacities are configurable application assumptions, "
                 "not verified contracted grid connection capacities.")
    st.caption("For the original LSTM forecasting tools, select a depot using the sidebar dropdown.")


def optivolt_all_depots_section(section: str) -> None:
    """Meaningful All 9 Depots views; never misreport saved artifacts as test performance."""
    from optivolt_portfolio import portfolio_summary
    from pathlib import Path
    import plotly.graph_objects as go
    frame = portfolio_summary(Path(__file__).parent)
    if frame.empty:
        st.warning("The nine historical depot datasets are unavailable.")
        return
    st.caption("All 9 Depots · UKPN Optimise Prime observations, 2021–2022 · Not live telemetry")
    if section == "⚙️ Model Performance":
        st.title("LSTM Model Performance · All 9 Depots")
        st.write("Review the saved LSTM artifacts for each depot. Verified held-out performance is displayed only where supporting evaluation results exist.")
        rows=[]
        for depot in frame['Depot']:
            slug=str(depot).lower().replace(' ','_')
            model=Path(__file__).parent/'models'/f'{slug}.keras'
            scaler=Path(__file__).parent/'models'/f'{slug}_scalers.npz'
            rows.append({'Depot':depot,'Saved LSTM': 'Available' if model.is_file() else 'Missing',
                         'Input/target scalers':'Available' if scaler.is_file() else 'Missing',
                         'Held-out metrics':'Available (Bexleyheath evaluation)' if depot=='Bexleyheath' else 'Not verified in repository'})
        c1,c2,c3=st.columns(3)
        c1.metric('Depots',len(rows))
        c2.metric('Saved LSTM models',sum(r['Saved LSTM']=='Available' for r in rows))
        c3.metric('Models with verified held-out metrics',1 if (Path(__file__).parent/'model_comparison_metrics.csv').exists() else 0)
        st.subheader('Model readiness by depot')
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        metrics_file=Path(__file__).parent/'model_comparison_metrics.csv'
        if metrics_file.exists():
            metrics=pd.read_csv(metrics_file)
            lstm=metrics.loc[metrics['Model'].astype(str).str.casefold()=='lstm']
            if not lstm.empty:
                result=lstm.iloc[0]
                st.subheader('Bexleyheath · available held-out LSTM evaluation')
                cards=st.columns(4)
                for col,label,key,fmt in zip(cards,['MAE','RMSE','R²','Breach recall'],['MAE_kW','RMSE_kW','R2','Breach_Recall'],['{:.3f} kW','{:.3f} kW','{:.3f}','{:.1%}']):
                    if key in result and pd.notna(result[key]): col.metric(label,fmt.format(float(result[key])))
                st.caption('These are saved held-out evaluation statistics, not 24-hour recursive backtesting results. They must not be copied to other depots.')
    elif section == '🚨 System Alerts':
        st.title('Fleet Capacity Risk · Historical Screening')
        st.warning('This page shows historical threshold comparisons. It does not generate live or forward-looking LSTM alerts across nine depots.')
        st.dataframe(frame[['Depot','Historical peak (kW)','Planning threshold (kW)','Historical peak / threshold (%)']],hide_index=True,use_container_width=True)
    elif section == '🚗 Scenario Analysis':
        st.title('Fleet Planning · All 9 Depots')
        st.caption('Illustrative capacity screening based on measured historical peaks, not new LSTM forecast outputs.')
        growth=st.slider('Hypothetical fleet-demand growth (%)',0,200,20,5,key='all9_growth')
        working=frame[['Depot','Historical peak (kW)','Planning threshold (kW)']].copy()
        working['Illustrative demand (kW)']=working['Historical peak (kW)']*(1+growth/100)
        working['Headroom (kW)']=working['Planning threshold (kW)']-working['Illustrative demand (kW)']
        st.dataframe(working,hide_index=True,use_container_width=True)
        st.caption('Select an individual depot for the original LSTM scenario controls.')
    elif section == '📋 Reports':
        st.title('Fleet Report · All 9 Depots')
        st.caption('Historical UKPN depot demand and planning thresholds; no forecast results are implied.')
        st.dataframe(frame,hide_index=True,use_container_width=True)
        st.download_button('Download nine-depot report (CSV)',frame.to_csv(index=False).encode(),file_name='optivolt_nine_depot_historical_report.csv',mime='text/csv')
    elif section == '📜 Archive History':
        st.title('Fleet Historical Data · All 9 Depots')
        st.caption('Observed demand archive for nine depots, July 2021 to July 2022.')
        st.dataframe(frame,hide_index=True,use_container_width=True)
        st.info('Select an individual depot for model versions, forecast history and other original archive tools.')
    elif section == '⚙️ Settings / Admin':
        st.title('Fleet Settings · All 9 Depots')
        st.dataframe(frame[['Depot','Planning threshold (kW)']],hide_index=True,use_container_width=True)
        st.info('Use Depot Setup to change a specific depot, or Users & Settings for workspace accounts and access.')
    else:
        st.title('All 9 Depots')
        st.dataframe(frame,hide_index=True,use_container_width=True)


def optivolt_section(section: str) -> None:
    """Route portfolio selection or a single depot into the original LSTM screens."""
    user = st.session_state.get("user")
    if not user or (user.get("role") != "admin" and not user.get("platform_admin")):
        st.error("Optivolt LSTM screens are available to workspace administrators.")
        st.stop()
    choice = st.session_state.get("optivolt_portfolio_selection", "All 9 Depots")
    # Retain the enhanced overview only. All other operational pages use
    # their original Optivolt LSTM implementations and settings.
    updated = {"🏠 Home / Overview": "overview"}
    if section in updated:
        optivolt_enhanced(updated[section])
        return
    # Depot configuration and data upload must remain usable when the portfolio
    # selector is set to All 9 Depots. A depot is chosen locally for editing
    # settings; Upload Data opens directly so a new depot can be registered.
    if choice == "All 9 Depots":
        if section == "🏢 Depots Setup":
            from optivolt_portfolio import available_depots
            candidates = available_depots(Path(__file__).parent)
            candidates += [name for name in st.session_state.get("uploaded_depots_registry", {})
                           if name not in candidates]
            if not candidates:
                st.error("No depots are available for configuration.")
                return
            st.subheader("Depot setup")
            st.caption("Choose the depot to configure here. Your main portfolio selection can remain All 9 Depots.")
            choice = st.selectbox("Depot to configure", candidates, key="optivolt_setup_target")
        elif section == "📊 Upload Depot Data":
            # The legacy upload form requires an active depot for background
            # context but can create or register an entirely new depot itself.
            # Do not change the portfolio selector just to allow an upload.
            choice = st.session_state.get("current_depot", "Bexleyheath")
            if choice not in _PLANNING_CAPACITY and choice not in st.session_state.get("uploaded_depots_registry", {}):
                choice = "Bexleyheath"
        else:
            optivolt_all_depots_section(section)
            return
    # optivolt_lstm.py reads this exact key to choose the matching original
    # screen; without it, every legacy route defaults to Home / Overview.
    st.session_state["_optivolt_requested_section"] = section
    st.session_state["current_depot"] = choice
    st.session_state["authenticated"] = True
    st.session_state["_optivolt_integrated_navigation"] = True
    # The legacy screen renders its own title. A second title here caused
    # duplicated headings and contributed to the apparently empty top area.
    runpy.run_path(str(Path(__file__).with_name("optivolt_lstm.py")), run_name="__optivolt_lstm__")


def make_optivolt_page(section: str):
    def render() -> None:
        optivolt_section(section)
    return render


def sidebar() -> None:
    """Depot selector, signed-in user and sign out."""
    app_settings = services.get_settings()
    keys = list(app_settings["depots"])
    user = st.session_state["user"]
    with st.sidebar:
        st.markdown('<div class="dw-side-label">OPTIVOLT DEPOT SELECTION</div>', unsafe_allow_html=True)
        from optivolt_portfolio import available_depots
        optivolt_choices = ["All 9 Depots"] + available_depots(Path(__file__).parent)
        registry = st.session_state.get("uploaded_depots_registry", {})
        optivolt_choices.extend(sorted(name for name in registry if name not in optivolt_choices))
        previous = st.session_state.get("optivolt_portfolio_selection", "All 9 Depots")
        if previous not in optivolt_choices:
            st.session_state["optivolt_portfolio_selection"] = "All 9 Depots"
        chosen = st.selectbox("Depot selection", optivolt_choices,
                              key="optivolt_portfolio_selection", label_visibility="collapsed")
        st.caption("Portfolio overview" if chosen == "All 9 Depots" else f"LSTM depot: {chosen}")
        # Keep the authenticated workspace setting in sync when a named built-in depot is chosen.
        if chosen != "All 9 Depots" and keys:
            matching = [k for k in keys if app_settings["depots"][k]["name"] == chosen]
            if matching and matching[0] != app_settings.get("selected_depot"):
                app_settings["selected_depot"] = matching[0]
                services.save_settings()
        role = "Platform owner" if user.get("platform_admin") else "Administrator" if user["role"] == "admin" else "Viewer"
        initials = "".join(part[0] for part in user["name"].split()[:2]).upper() or "U"
        st.markdown(
            f'<div class="dw-user"><div class="avatar">{ui.esc(initials)}</div><div style="min-width:0">'
            f'<div class="who">{ui.esc(user["name"])}</div>'
            f'<div class="role">{role}</div></div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            database.drop_session(st.session_state.get("session_token"))
            database.audit(user["org_id"], user["email"], "Signed out")
            for key in ("user", "session_token", "settings", "depot_selector", "forecast_stamps"):
                st.session_state.pop(key, None)
            st.query_params.clear()
            st.rerun()


# ------------------------------------------------------------------ sign-in state
if "user" not in st.session_state:
    restored = database.resolve_session(st.query_params.get("session"))
    if restored:
        st.session_state["user"] = restored
        st.session_state["session_token"] = st.query_params.get("session")

def login_or_landing(view: str) -> None:
    """Render the requested public page without session-state view collisions."""
    if view == "landing":
        landing.render()
    else:
        login.render(view)


if not st.session_state.get("user"):
    # Registered pages are the single source of truth for all public buttons.
    # Changing only ``public_view`` is insufficient: Streamlit reruns the
    # selected page and previously reset it to its original value.
    public_pages = [
        st.Page(lambda: login_or_landing("signin"), title="Sign in", icon=":material/login:", url_path="sign-in", default=True),
        st.Page(lambda: login_or_landing("landing"), title="Home", icon=":material/home:", url_path="welcome"),
        st.Page(lambda: login_or_landing("signup"), title="Create account", icon=":material/person_add:", url_path="create-account"),
    ]
    st.session_state["_public_nav_pages"] = dict(zip(("landing", "signin", "signup"), public_pages))
    st.navigation(public_pages, position="sidebar", expanded=True).run()
    st.stop()

if st.session_state.get("session_token"):
    st.query_params["session"] = st.session_state["session_token"]  # keep the session across reloads

# ---------------------------------------------------------------------- dashboard
pages = []
if services.is_admin() or services.is_owner():
    for idx, (label, path, icon, section) in enumerate(OPTIVOLT_SECTIONS):
        pages.append(st.Page(make_optivolt_page(section), title=label, icon=icon,
                             url_path=path, default=(idx == 0)))

# Unified Settings/Users and depot management; no XGBoost forecast pages exposed.
pages.extend([
    st.Page(settings.render, title="Users & Settings", icon=":material/manage_accounts:", url_path="workspace-settings"),
])
if services.is_owner():
    pages.append(st.Page(platform.render, title="Platform Administration", icon=":material/admin_panel_settings:", url_path="platform"))
# Full-width workspace styling is deliberately applied ONLY after login.
# The original DepotWatt-style public welcome and sign-in layouts keep their
# own narrower, intentional max-width rules.
st.markdown("""<style>
/* Use the available width to the right of the permanent blue sidebar. */
[data-testid="stMainBlockContainer"],
[data-testid="stMainBlockContainer"] > div,
section[data-testid="stMain"] .block-container {
    max-width: none !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
section[data-testid="stMain"] .block-container {
    padding-left: clamp(1rem, 2vw, 2.25rem) !important;
    padding-right: clamp(1rem, 2vw, 2.25rem) !important;
}
/* Never give grid columns more width than their shared container. */
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
    width: 100%;
}
[data-testid="stMainBlockContainer"] [data-testid="stPlotlyChart"],
[data-testid="stMainBlockContainer"] [data-testid="stDataFrame"] {
    max-width: 100%;
}
@media (max-width: 768px) {
  section[data-testid="stMain"] .block-container {
      padding-left: 1rem !important;
      padding-right: 1rem !important;
  }
}
</style>""", unsafe_allow_html=True)
st.logo(str(config.ASSETS_DIR / "logo.svg"), size="large", icon_image=ICON)
current = st.navigation(pages, position="sidebar", expanded=True)
sidebar()
current.run()
