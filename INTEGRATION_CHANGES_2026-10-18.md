# Optivolt integrated repair notes

- Restored `app.py` from the latest uploaded standalone app because the GitHub ZIP does **not** contain `app.py`.
- Public sidebar CSS no longer hides Streamlit navigation. Public Home, Sign in and Create account routes are visible.
- Landing chart reads `lstm_24h_recursive_forecast.csv` (historical July 2022 saved LSTM output), not `services.depot_risk()` or an XGBoost model. No live-data claim.
- Landing/login statements use LSTM wording; planning thresholds explicitly differentiated from grid contracts.
- Retraining reports real Keras epoch completion at 40–75% with training and validation loss, then runs held-out evaluation. Actual training time remains resource-dependent.
- Sidebar page labels and portfolio headings in the restored app remain as previously customised.

## Important remaining architecture caveat

Legacy `dashboard/services.py`, `model_loader.py` and the unused global_model artifact still contain XGBoost functionality. They are not a valid LSTM backend and must not be substituted for the original Optivolt LSTM pipeline. They remain packaged because account administration and historical DepotWatt service dependencies may require those modules. This package replaces the public model preview, not every historical internal implementation.

## Deploy

Commit all files, not only app.py. Use Python 3.12 for a compatible TensorFlow wheel, and check the Streamlit logs for training memory errors.
