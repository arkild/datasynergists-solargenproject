import streamlit as st
import pandas as pd
from datetime import date
from app import load_model, load_data

df = load_data()
model, feature_names = load_model()

st.title("🔮 Generation Prediction — Historical Lookup")
st.markdown(
    "Select any date and hour to see what the model predicted versus what "
    "actually occurred. The dataset spans three distinct ranges, each with "
    "a different relationship between the model and the data."
)

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select a date",
        value=date(2025, 7, 15),
        min_value=date(2015, 1, 1),
        max_value=date(2025, 12, 31)
    )

with col2:
    daylight_hours = sorted(df[df["solar_elevation"] > 0]["dt"].dt.hour.unique())
    min_hour = int(daylight_hours[0])
    max_hour = int(daylight_hours[-1])
    selected_hour = st.slider("Select hour of day", min_hour, max_hour, 12)

# ── Determine which range the selected date falls into ──
if selected_date < date(2022, 9, 1):
    range_label = "🔮 Prediction Window"
    range_msg = (
        "This date falls in the **backcast period (2015 – Aug 2022)**, before "
        "real generation measurement began. There is no ground truth here — "
        "'Actual Generation' shown below is the model's own prediction, used "
        "as a stand-in to reconstruct historical generation for analysis."
    )
elif selected_date <= date(2024, 12, 31):
    range_label = "🧪 Model Tested On Its Own Training Data"
    range_msg = (
        "This date falls within the **model's training window (Sep 2022 – Dec 2024)**. "
        "The model has already seen this data during training, so its prediction "
        "here benefits from prior exposure — this comparison is informative but "
        "**not a fair test of real-world accuracy**."
    )
else:
    range_label = "✅ Blind Test on Ground Truth"
    range_msg = (
        "This date falls in **2025**, which was held out of training entirely. "
        "The model has never seen this data — this is the **true blind test**, "
        "achieving R² = **0.8839** across the full year with no prior knowledge."
    )

st.info(f"**{range_label}**\n\n{range_msg}")

target_dt = pd.Timestamp(selected_date).replace(hour=selected_hour)
row = df[df["dt"] == target_dt]

if row.empty:
    st.warning("No data available for this exact date and hour. Try another.")
else:
    row = row.iloc[0]
    X = row[feature_names].to_frame().T
    predicted = model.predict(X)[0]

    # Use ground truth if available, otherwise fall back to stored prediction
    if pd.notna(row["ground_truth"]):
        actual = row["ground_truth"]
        actual_label = "⚡ Actual Generation (Ground Truth)"
    else:
        actual = row["predicted_volume"]
        actual_label = "🔮 Reference Generation (Pre-KKP1 Estimate)"

    diff = predicted - actual
    pct_err = abs(diff / actual * 100) if actual != 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("🔮 Model Predicted (Live)", f"{predicted:.3f} MW")
    col2.metric(actual_label, f"{actual:.3f} MW")
    col3.metric("📊 Difference", f"{diff:+.3f} MW", f"{pct_err:.1f}% error")

    # ── Feature breakdown ─────────────────────────────────────────────────
    st.subheader("🧩 Feature Breakdown — This Prediction")
    st.markdown("Key conditions the model used to produce the prediction above.")

    # (column name, display label, unit suffix, format spec)
    core_display_spec = [
        ("shortwave", "Solar Irradiance", " W/m²", "{:.0f}"),
        ("cloud_pct", "Cloud Cover", "%", "{:.1f}"),
        ("cloudbase_m", "Cloud Base Height", " m", "{:,.0f}"),
        ("aod_smoke", "Smoke (AOD)", "", "{:.3f}"),
        ("solar_elevation", "Solar Elevation", "°", "{:.1f}"),
        ("attenuation_ratio", "Attenuation Ratio", "", "{:.2f}"),
        ("Temperature (degrees C)", "Temperature", "°C", "{:.1f}"),
        ("Relative Humidity", "Relative Humidity", "%", "{:.0f}"),
    ]
    # Only show ones that actually exist in this dataframe, in case naming
    # ever drifts between this list and the underlying data.
    core_display_spec = [spec for spec in core_display_spec if spec[0] in row.index]

    core_table = pd.DataFrame({
        "Condition": [label for _, label, _, _ in core_display_spec],
        "Value": [
            fmt.format(row[col]) + unit
            for col, label, unit, fmt in core_display_spec
        ]
    })
    st.dataframe(core_table, width='stretch', hide_index=True)

    # Full feature list + SHAP explanation both live behind this button —
    # avoids cluttering the default view and avoids triggering SHAP on
    # every date/hour change.
    if st.button("Show full feature list & SHAP explanation"):
        st.markdown("**Full feature set used by the model:**")
        feat_table = pd.DataFrame({
            "Feature": feature_names,
            "Value": row[feature_names].values
        })
        st.dataframe(feat_table, width='stretch', hide_index=True)

        import shap

        with st.spinner("Calculating SHAP values for this prediction..."):
            explainer = shap.TreeExplainer(model)
            X_row = row[feature_names].values.reshape(1, -1)
            shap_values = explainer(X_row, check_additivity=False)

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP value": shap_values.values[0]
        }).sort_values("SHAP value", key=abs, ascending=False)

        st.markdown(
            "How each feature pushed **this specific prediction** above or "
            "below the model's baseline output. Positive values push the "
            "prediction up, negative values pull it down."
        )
        st.dataframe(
            shap_df.style.background_gradient(
                cmap="RdBu_r", subset=["SHAP value"], vmin=-shap_df["SHAP value"].abs().max(),
                vmax=shap_df["SHAP value"].abs().max()
            ),
            width='stretch',
            hide_index=True
        )