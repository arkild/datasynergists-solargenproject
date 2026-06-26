# SPICE Solar Generation Dashboard

An interactive dashboard analyzing solar power generation at KKP1 kisikāw pīsim 
in Edmonton, Alberta — with a focus on the impact of wildfire smoke events.

## Pages

- **🏠 Home** — Overview of the dashboard, the wildfire/AOD detection methodology, and links to each page
- **🗺️ Map** — NASA GIBS satellite imagery of Edmonton during smoke events, with date scrubbing around detected events
- **⚡ The Paradox** — Wildfire smoke vs solar generation — the counterintuitive finding
- **🕒 Hourly Smoke Analysis** — Hourly AOD and generation across ±3 day daylight windows
- **🔮 Prediction Check** — Historical generation lookup using a Random Forest model (R²=0.8839, blind tested on 2025)
- **🔬 XAI** — Feature importance, Actual vs Predicted, and Partial Dependence Plots
- **🧠 SHAP** — SHAP beeswarm analysis showing how each feature influences individual predictions
- **💡 Future Work** — Next steps for model improvement and data expansion

## Data Sources

- AESO solar generation data (public)
- Edmonton Blatchford weather station (public)
- NASA POWER shortwave radiation (public)
- Copernicus CAMS aerosol optical depth (AOD) data (public)

## Model

Random Forest trained on 2022–2024, blind tested on 2025. R²=0.8839.

## Live Dashboard

[View Dashboard](https://datasynergists-solargenproject-nx3thggv35tww9o9q9swfn.streamlit.app/)