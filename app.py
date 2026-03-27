import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests
from datetime import datetime, date

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPICE Solar Dashboard",
    page_icon="☀️",
    layout="wide"
)

# ── Load model and features ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/solar_cleaned.csv")
    df["dt"] = pd.to_datetime(df["dt"])
    return df

model, feature_names = load_model()
df = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("☀️ SPICE Solar")
st.sidebar.markdown("Solar Power Generation Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Map", "📊 Client Nulls", "🔮 Free Prediction", "⚡ The Paradox"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️ Map":
    st.title("🗺️ Sky Conditions During Wildfire Events")
    st.markdown(
        "Visualize what the sky looked like over Edmonton during key wildfire "
        "smoke events using NASA GIBS satellite imagery."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_date = st.date_input(
            "Select a date",
            value=date(2023, 6, 6),
            min_value=date(2022, 1, 1),
            max_value=date(2025, 12, 31)
        )

        st.markdown("**Edmonton coordinates**")
        st.write("Lat: 53.5461° N | Lon: -113.4938° W")

        layer = st.selectbox(
            "Satellite layer",
            [
                "MODIS_Terra_CorrectedReflectance_TrueColor",
                "MODIS_Aqua_CorrectedReflectance_TrueColor",
                "VIIRS_SNPP_CorrectedReflectance_TrueColor",
            ]
        )

    with col2:
        date_str = selected_date.strftime("%Y-%m-%d")
        wms_url = (
            f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
            f"SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
            f"&LAYERS={layer}"
            f"&CRS=EPSG:4326"
            f"&BBOX=50,-120,58,-105"
            f"&WIDTH=800&HEIGHT=600"
            f"&FORMAT=image/png"
            f"&TIME={date_str}"
        )

        try:
            response = requests.get(wms_url, timeout=10)
            if response.status_code == 200:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(response.content))
                st.image(img, caption=f"Edmonton region — {date_str}", use_column_width=True)
            else:
                st.warning("Could not load satellite image for this date.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLIENT NULLS (Bissell Gap Fill)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Client Nulls":
    st.title("📊 Bissell Thrift — Gap Fill via KKP Correlation")
    st.markdown(
        "Bissell Thrift had missing generation readings during certain periods. "
        "Using KKP1 as a proxy (r = **0.916**), we estimated what Bissell was "
        "likely generating during those gaps."
    )

    # Correlation parameters — update these from your actual data
    r = 0.916
    kkp_mean = df["Volume"].mean()  # replace with KKP mean if separate
    bissell_mean = kkp_mean * 0.85  # approximate — update with real Bissell mean

    # Generate predicted Bissell values from KKP
    df_gap = df.copy()
    df_gap["bissell_predicted"] = r * (df_gap["Volume"] - kkp_mean) + bissell_mean

    st.subheader("Predicted Bissell Generation vs KKP Actual")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_gap["dt"], df_gap["Volume"], alpha=0.6, label="KKP1 Actual", color="#f4a261")
    ax.plot(df_gap["dt"], df_gap["bissell_predicted"], alpha=0.6,
            label="Bissell Predicted (r=0.916)", color="#2a9d8f", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Generation (MW)")
    ax.legend()
    ax.set_title("KKP1 Actual vs Bissell Predicted Generation")
    st.pyplot(fig)

    st.metric("Pearson Correlation (r)", "0.916")
    st.metric("KKP Mean Generation", f"{kkp_mean:.3f} MW")
    st.metric("Estimated Bissell Mean", f"{bissell_mean:.3f} MW")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FREE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Free Prediction":
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
            value=date(2025, 6, 15),
            min_value=df["dt"].min().date(),
            max_value=df["dt"].max().date()
        )

    with col2:
        selected_hour = st.slider("Select hour of day", 0, 23, 12)

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
        condition_cols = ["Temperature (degrees C)", "Relative Humidity",
                          "cloud_pct", "shortwave", "pm25_mean"]
        available = [c for c in condition_cols if c in row.index]
        st.dataframe(pd.DataFrame(row[available]).T, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — THE PARADOX
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ The Paradox":
    st.title("⚡ The Wildfire Paradox")
    st.markdown(
        "Wildfires reduce solar generation **on the day** due to smoke — "
        "but the conditions that cause wildfires (hot, clear, dry weather) "
        "mean the **surrounding week** often shows **higher than average** generation "
        "compared to equivalent weeks in non-wildfire years."
    )

    # Identify wildfire events — update with your actual known event dates
    wildfire_events = {
        "June 2023 Smoke Event": ("2023-06-01", "2023-06-15"),
        "May 2023 Event": ("2023-05-10", "2023-05-20"),
    }

    event = st.selectbox("Select a wildfire event", list(wildfire_events.keys()))
    start, end = wildfire_events[event]

    df_event = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()

    if df_event.empty:
        st.warning("No data for this event range.")
    else:
        # Daily averages during event
        df_daily = df_event.groupby(df_event["dt"].dt.date)["Volume"].mean().reset_index()
        df_daily.columns = ["date", "avg_generation"]

        # Get same week from previous years for comparison
        event_start = pd.Timestamp(start)
        week_num = event_start.isocalendar().week

        df_compare = df[df["dt"].dt.isocalendar().week == week_num].copy()
        df_compare = df_compare[df_compare["dt"].dt.year != event_start.year]
        df_compare_daily = df_compare.groupby(
            df_compare["dt"].dt.date)["Volume"].mean()

        baseline_mean = df_compare_daily.mean()
        event_mean = df_daily["avg_generation"].mean()
        event_min = df_daily["avg_generation"].min()

        col1, col2, col3 = st.columns(3)
        col1.metric("Event Period Average", f"{event_mean:.3f} MW")
        col2.metric("Baseline (Same Week, Other Years)", f"{baseline_mean:.3f} MW")
        col3.metric(
            "Day-of Minimum",
            f"{event_min:.3f} MW",
            f"{event_min - baseline_mean:+.3f} MW vs baseline"
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_daily["date"], df_daily["avg_generation"],
                marker="o", label="During Event", color="#e76f51")
        ax.axhline(baseline_mean, linestyle="--", color="#2a9d8f",
                   label=f"Baseline avg ({baseline_mean:.2f} MW)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Avg Generation (MW)")
        ax.set_title(f"Daily Generation During {event} vs Baseline")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.info(
            "🔍 **The Paradox:** While smoke directly suppresses generation on "
            "peak wildfire days, the hot and dry pre-conditions associated with "
            "wildfire season can elevate weekly averages above non-wildfire baselines."
        )