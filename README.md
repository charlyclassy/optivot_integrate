# DepotWatt — reconstructed functional Streamlit dashboard

This package provides the original supplied DepotWatt forecasting engine, trained XGBoost model and nine UKPN historical depot datasets, plus **newly written dashboard screens**. The twelve original dashboard page sources were never supplied, so the new pages are functional implementations derived from the project documentation, **not exact copies of the original design**.

## Included pages

- Landing and account sign-in / workspace registration
- Fleet overview with capacity-risk table and model-generated forecast
- Depot management with capacity and threshold editing and measured-demand CSV/Excel ingestion (regular 15-minute data)
- Multi-horizon XGBoost forecast with risk thresholds and CSV export
- Hypothetical growth/capacity scenarios with saved records
- Held-out test metrics, performance plots and model feature importance
- Live capacity-risk summary and acknowledgement of recorded alerts
- Historical forecasts, alerts, scenario runs and replay charts
- Excel / CSV ZIP / PDF reports
- Organisation settings, password change, account creation, roles and audit log
- Platform owner administration of organisations and registration policy

## Deployment

1. Extract the ZIP onto your computer. Upload the **contents** into the GitHub repository root, preserving subfolders such as `dashboard/pages/` and `models/global_model/`.
2. Deploy on Streamlit Community Cloud with `app.py` as main file and `requirements.txt` for dependencies.
3. Verify the pages with a real browser and Streamlit Cloud logs.

## Notes and limitations

- `data/processed/depot_demand.csv` contains historical measurements. A forecast starts **after the end of those historical observations**; it is not a forecast of today's electricity demand unless updated readings are uploaded.
- Model file validation and numerical forecast checks passed in the earlier audit. These newly written UI pages have passed syntax checks, but **a complete Streamlit runtime / browser test has not been completed in this environment**.
- Account credentials for demonstration are defined in `config.py`. A public Streamlit deployment must not be operated as a multi-tenant production service using those default accounts. State stored in the filesystem on Community Cloud is ephemeral, so customer accounts/uploads will not have durable storage.
- Some advanced original features (full model package replacement/rollback interface, original branded screens, automated irregular-data harmonisation and production account recovery) are **not implemented to the original specification**. The new upload screen validates contiguous 15-minute readings rather than silently estimating values.
