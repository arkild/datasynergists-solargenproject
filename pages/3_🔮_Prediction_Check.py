import streamlit as st
import pandas as pd
from datetime import date
from app import load_model, load_data

df = load_data()
model, feature_names = load_model()

st.title("🔮 Generation Prediction — Historical Lookup")
st.markdown(
    "Select any date and hour from our dataset to see what the model predicted "
    "versus what actually occurred. The model was trained on 2022–2024 data "
    "and tested **blind** on 2025 — achieving R² = **0.86** with no prior "
    "knowledge of that year."
)

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select a date",
        value=date(2025, 7, 15),
        min_value=df["dt"].min().date(),
        max_value=df["dt"].max().date()
    )

with col2:
    daylight_hours = sorted(df[df["solar_elevation"] > 0]["dt"].dt.hour.unique())
    min_hour = int(daylight_hours[0])
    max_hour = int(daylight_hours[-1])
    selected_hour = st.slider("Select hour of day", min_hour, max_hour, 12)

# Filter to selected datetime
target_dt = pd.Timestamp(selected_date).replace(hour=selected_hour)
row = df[df["dt"] == target_dt]

if row.empty:
    st.warning("No data available for this exact date and hour. Try another.")
else:
    row = row.iloc[0]
    X = row[feature_names].values.reshape(1, -1)
    predicted = model.predict(X)[0]
    actual = row["Volume"]
    diff = predicted - actual
    pct_err = abs(diff / actual * 100) if actual != 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("🔮 Model Predicted", f"{predicted:.3f} MW")
    col2.metric("⚡ Actual Generation", f"{actual:.3f} MW")
    col3.metric("📊 Difference", f"{diff:+.3f} MW", f"{pct_err:.1f}% error")

    st.subheader("Conditions on this day")
    condition_cols = ["shortwave", "cloud_pct", "solar_elevation", "attenuation_ratio",
                    "Temperature (degrees C)", "Relative Humidity"]
    available = [c for c in condition_cols if c in row.index]
    st.dataframe(pd.DataFrame(row[available]).T, use_container_width=True)