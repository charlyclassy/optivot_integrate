# Full Optivolt repository replacement

This archive contains the full Optivolt project from `Optivolt_Original_DepotWatt_SignIn_Ready.zip`, with the newer `app.py` and `dashboard/pages/login.py` from `Optivolt_Create_Workspace_Button_Fix.zip` applied.

It contains 9 saved `.keras` models and 9 original UKPN depot CSVs.

## Install

Extract the ZIP. The files inside (including `app.py`, `dashboard/`, `models/`, and all other files) belong directly in the **root** of your GitHub repository. Do not upload the ZIP itself as a single repo file and do not create an extra enclosing folder.

For a full replacement, back up the repository first. A GitHub web upload can add or overwrite files but will not automatically delete old files missing from this archive. To completely mirror the archive, delete obsolete tracked files in a local clone or via GitHub, then copy the extracted contents, commit, and push. Do **not** upload credentials, `.env`, user databases, or private secrets.

## Validation scope

All Python files passed AST syntax parsing and the ZIP was verified for integrity. This does not establish runtime functionality on Streamlit Cloud; sign-up, login, forecasting, and TensorFlow training must be tested after deployment. Legacy XGBoost modules may remain in the repository and do not imply that every backend workflow has been converted to LSTM.
