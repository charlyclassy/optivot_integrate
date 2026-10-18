# Optivolt integrated deployment — visible LSTM pages

The **ten original Optivolt sections** are now separate, visible sidebar pages for workspace administrators: Overview, Depot Setup, Upload Data, Demand Forecast, Scenario Analysis, Model Performance, System Alerts, Reports, Archive History, and Admin Settings. This corrects the earlier packaging error that hid all ten behind a single LSTM Portal link.

**Workspace Users & Settings** contains DepotWatt account creation, role management, password changes and organisation settings. Its original XGBoost features remain accessible as separately **XGBoost-labelled** supplementary pages, not substituted for LSTM output.

The nine original Keras models, UKPN CSVs, forecasting implementations and data files remain in the distribution. Optivolt screens use the original LSTM code. The organisation administration backend remains DepotWatt's SQLite implementation.

For Streamlit Cloud use Python **3.12**, configure production mode as appropriate, and provide persistent storage for account databases and uploaded files; the default local filesystem is ephemeral. End-to-end browser and model-inference validation is still required.
