# Deployment readiness — DepotWatt

## Now included
- Full trained XGBoost package under `models/global_model/`
- Consolidated real metered demand from all nine supplied UKPN depot CSVs at `data/processed/depot_demand.csv`
- Core dashboard Python modules and application entrypoint
- Basic temporary replacements for all twelve page modules; **these are not the original full product screens**
- SVG assets and Streamlit configuration

## Limitations
- Original full-feature dashboard page source files have not been provided. The temporary pages provide only limited functionality.
- This package is intended for testing/demonstration and has not undergone full end-to-end Streamlit Cloud deployment testing.
- For a public production multi-user service, deploy with appropriate persistent storage, authentication hardening, and secrets management. Streamlit Community Cloud storage is ephemeral.
- Demo accounts with published passwords should never be used for production.

## GitHub upload
1. Extract this ZIP onto your computer.
2. In GitHub, upload the *contents* of the extracted folder, including its subfolders; do not upload the ZIP itself.
3. The root should contain `app.py` next to `dashboard/`, `models/`, and `data/`.
4. In Streamlit Cloud select `app.py` and redeploy/reboot.

## Source validation
Each depot CSV was checked for matching depot labels, required columns, ordered timestamps, nonnegative numeric demand, and valid observed flags.
