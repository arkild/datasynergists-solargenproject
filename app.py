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


# ── Wildfire event detection ──────────────────────────────────────────────────
def detect_smoke_events(df, threshold=50, min_duration=1):
    """Auto-detect smoke events based on PM2.5 threshold."""
    daily_pm25 = df.groupby(df['dt'].dt.date)['pm25_mean'].mean().reset_index()
    daily_pm25.columns = ['date', 'pm25_mean']
    daily_pm25['date'] = pd.to_datetime(daily_pm25['date'])
    
    # Flag smoky days
    daily_pm25['is_smoke'] = daily_pm25['pm25_mean'] >= threshold
    
    # Group consecutive smoky days into events
    events = {}
    in_event = False
    start = None
    
    for _, row in daily_pm25.iterrows():
        if row['is_smoke'] and not in_event:
            in_event = True
            start = row['date']
        elif not row['is_smoke'] and in_event:
            in_event = False
            peak = daily_pm25[
                (daily_pm25['date'] >= start) & 
                (daily_pm25['date'] < row['date'])
            ]['pm25_mean'].max()
            label = f"{start.strftime('%b %Y')} — Peak {peak:.0f} µg/m³"
            events[label] = (start.strftime('%Y-%m-%d'), 
                           row['date'].strftime('%Y-%m-%d'))
    
    return events

wildfire_events = detect_smoke_events(df, threshold=50)

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
        event = st.selectbox(
            "Select a wildfire event (auto-fills date)",
            ["Custom date"] + list(wildfire_events.keys())
        )

        if event == "Custom date":
            selected_date = st.date_input(
                "Date",
                value=date(2023, 5, 19),
                min_value=date(2022, 9, 1),
                max_value=date(2025, 12, 31)
            )
        else:
            event_center = pd.Timestamp(wildfire_events[event][0]).date()
            day_offset = st.slider(
                "Days around event start",
                min_value=-5,
                max_value=5,
                value=0,
                format="%d days"
            )
            selected_date = event_center + pd.Timedelta(days=day_offset)
            st.caption(f"Viewing: {selected_date}")

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
                st.image(img, caption=f"Edmonton region — {date_str}", use_container_width=True)
            else:
                st.warning("Could not load satellite image for this date.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLIENT NULLS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Client Nulls":
    st.title("📊 Client Site — KKP Correlation & Gap Fill")
    st.markdown(
        "KKP1 and the client site are strongly correlated (r = **0.916**). "
        "Insights drawn from KKP generation patterns apply to the client site — and where "
        "the client site has missing readings, KKP can fill the gaps."
    )

    uploaded_file = st.file_uploader("Upload client site generation CSV", type="csv")

    if uploaded_file is not None:
        # Load client data
        df_client = pd.read_csv(uploaded_file)
        df_client = df_client.rename(columns={
            "Date and time": "dt",
            "Total system": "kwh"
        })
        df_client["dt"] = pd.to_datetime(df_client["dt"], errors="coerce")
        df_client["kwh"] = pd.to_numeric(df_client["kwh"], errors="coerce")
        df_client["date"] = df_client["dt"].dt.date

        # Client site capacity factor (30.7 kW capacity, daily kWh)
        df_client["cf_client"] = df_client["kwh"] / (30.7 * 24)

        # KKP — aggregate hourly to daily sum, then capacity factor (7000 kW)
        df_kkp = df.copy()
        df_kkp["date"] = df_kkp["dt"].dt.date
        df_kkp_daily = df_kkp.groupby("date")["Volume"].sum().reset_index()
        df_kkp_daily.columns = ["date", "kkp_kwh_sum"]
        df_kkp_daily["cf_kkp"] = (df_kkp_daily["kkp_kwh_sum"] * 1000) / (7000 * 24)

        # Merge on date
        df_merged = df_client.merge(df_kkp_daily, on="date", how="inner")

        # Start from first valid client date
        first_date = df_merged["dt"].min()
        df_merged = df_merged[df_merged["dt"] >= first_date]

        # Identify nulls — zero generation during solar months
        df_merged["is_null"] = (
            (df_merged["cf_client"] == 0) &
            (df_merged["dt"].dt.month.between(4, 10))
        )

        # Correlation parameters in CF space
        r = 0.916
        kkp_cf_mean = df_merged["cf_kkp"].mean()
        client_cf_mean = df_merged[~df_merged["is_null"]]["cf_client"].mean()

        # Gap fill
        df_merged["cf_predicted"] = r * (df_merged["cf_kkp"] - kkp_cf_mean) + client_cf_mean
        df_merged["cf_filled"] = df_merged["cf_client"].copy()
        df_merged.loc[df_merged["is_null"], "cf_filled"] = df_merged.loc[df_merged["is_null"], "cf_predicted"]

        null_count = df_merged["is_null"].sum()

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Pearson r", "0.916")
        col2.metric("Missing Days Filled", f"{null_count}")
        col3.metric("Data Start", str(first_date.date()))

        # Plot 1 — Scatter showing correlation
        st.subheader("Capacity Factor Correlation — KKP vs Client Site")
        df_clean = df_merged[~df_merged["is_null"]]

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(df_clean["cf_kkp"], df_clean["cf_client"],
                    alpha=0.4, s=10, color="#f4a261")
        m, b = np.polyfit(df_clean["cf_kkp"], df_clean["cf_client"], 1)
        x_line = np.linspace(df_clean["cf_kkp"].min(), df_clean["cf_kkp"].max(), 100)
        ax1.plot(x_line, m * x_line + b, color="#2a9d8f", linewidth=1.5,
                 label=f"r = 0.916")
        ax1.set_xlabel("KKP Capacity Factor")
        ax1.set_ylabel("Client Site Capacity Factor")
        ax1.set_title("KKP1 vs Client Site — Daily Capacity Factor")
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1)

        # Plot 2 — Time series with gap fill highlighted
        st.subheader("Generation Over Time — Actual & Gap-Filled")
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        ax2.plot(df_merged["date"], df_merged["cf_kkp"],
                 color="#a8dadc", linewidth=0.8, alpha=0.6, label="KKP1 CF")
        ax2.plot(df_merged["date"], df_merged["cf_client"],
                 color="#f4a261", linewidth=1.2, alpha=0.8, label="Client Site Actual CF")

        if null_count > 0:
            df_nulls = df_merged[df_merged["is_null"]]
            ax2.scatter(df_nulls["date"], df_nulls["cf_predicted"],
                        color="#e76f51", s=40, zorder=5,
                        label=f"Gap-Filled ({null_count} days)")

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Capacity Factor")
        ax2.set_title("Client Site vs KKP1 — Capacity Factor Over Time")
        ax2.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

        if null_count > 0:
            st.subheader("Gap-Filled Days")
            st.dataframe(
                df_merged[df_merged["is_null"]][["date", "cf_kkp", "cf_predicted"]]
                .rename(columns={
                    "date": "Date",
                    "cf_kkp": "KKP CF (Reference)",
                    "cf_predicted": "Client Site CF (Predicted)"
                }).reset_index(drop=True),
                use_container_width=True
            )

    else:
        st.info("Upload the client site generation CSV to begin analysis.")
        st.markdown("**Expected columns:** `Date and time`, `Total system`")

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
        condition_cols = ["Temperature (degrees C)", "Relative Humidity",
                          "cloud_pct", "shortwave", "pm25_mean"]
        available = [c for c in condition_cols if c in row.index]
        st.dataframe(pd.DataFrame(row[available]).T, use_container_width=True)

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)

    with st.expander("🔬 View Feature Importance"):
        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        feat_imp.plot(kind="barh", ax=ax_imp, color="#f4a261")
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Top 15 Feature Importances — Random Forest")
        plt.tight_layout()
        st.pyplot(fig_imp)

        st.info(
            "💡 **Future improvement:** Cloud type data (cirrus vs cumulus) could "
            "significantly improve prediction accuracy. Cirrus clouds filter far less "
            "sunlight than cumulus, but current data only captures total cloud coverage percentage."
        )

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
    df_daylight = df[df["solar_elevation"] > 0]

    event = st.selectbox("Select a wildfire event", list(wildfire_events.keys()))
    start, end = wildfire_events[event]

    event_start = pd.Timestamp(start)
    event_end = pd.Timestamp(end)

    # Expand window ±5 days
    window_start = event_start - pd.Timedelta(days=5)
    window_end = event_end + pd.Timedelta(days=5)

    # Daily generation during event window
    df_window = df_daylight[(df_daylight["dt"] >= window_start) & (df_daylight["dt"] <= window_end)].copy()
    df_daily = df_window.groupby(df_window["dt"].dt.date)["Volume"].mean().reset_index()
    df_daily.columns = ["date", "avg_generation"]
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    if df_daily.empty:
        st.warning("No data for this event range.")
    else:
        # Get same day-of-year range from other years
        event_year = event_start.year
        doy_start = window_start.dayofyear
        doy_end = window_end.dayofyear

        df_other = df_daylight[df_daylight["dt"].dt.year != event_year].copy()
        df_other = df_other[
            (df_other["dt"].dt.dayofyear >= doy_start) &
            (df_other["dt"].dt.dayofyear <= doy_end)
        ]

        # Per-year daily averages for comparison
        df_other["year"] = df_other["dt"].dt.year
        df_other["doy"] = df_other["dt"].dt.dayofyear
        df_other_daily = df_other.groupby(["year", "doy"])["Volume"].mean().reset_index()

        # Baseline mean across all other years
        baseline_mean = df_other_daily["Volume"].mean()
        event_mean = df_daily["avg_generation"].mean()
        event_min = df_daily["avg_generation"].min()

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Event Window Average", f"{event_mean:.3f} MW", 
                    f"{event_mean - baseline_mean:+.3f} MW vs baseline")
        col2.metric("Baseline (Same Period, Other Years)", f"{baseline_mean:.3f} MW")
        col3.metric(
            "Worst Day During Event",
            f"{event_min:.3f} MW",
            f"{event_min - baseline_mean:+.3f} MW vs baseline"
        )

        # Main comparison plot
        fig, ax = plt.subplots(figsize=(14, 5))

        # Plot each comparison year as a faint line
        colors = ["#a8dadc", "#457b9d", "#1d3557", "#e9c46a"]
        years = df_other_daily["year"].unique()
        for i, yr in enumerate(sorted(years)):
            yr_data = df_other_daily[df_other_daily["year"] == yr].sort_values("doy")
            # Map doy back to actual dates using event year for x-axis alignment
            yr_data = yr_data.copy()
            yr_data["date"] = pd.to_datetime(
                yr_data["doy"].apply(
                    lambda d: pd.Timestamp(f"{event_year}-01-01") + pd.Timedelta(days=int(d)-1)
                )
            )
            ax.plot(
                yr_data["date"], yr_data["Volume"],
                alpha=0.4, linewidth=1.2,
                color=colors[i % len(colors)],
                label=str(yr)
            )

        # Plot event window on top
        ax.plot(
            df_daily["date"], df_daily["avg_generation"],
            marker="o", linewidth=2.5, color="#e76f51",
            label=f"{event_year} (Event)", zorder=5
        )

        # Shade the actual smoke event period
        ax.axvspan(event_start, event_end, alpha=0.15, color="gray", label="Smoke Event")

        # Baseline
        ax.axhline(
            baseline_mean, linestyle="--", color="#2a9d8f", linewidth=1.5,
            label=f"Baseline avg ({baseline_mean:.2f} MW)"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Avg Generation (MW)")
        ax.set_title(f"Solar Generation: {event} (±5 days) vs Same Period Other Years")
        ax.legend(loc="upper left", fontsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "🔍 **The Paradox:** While smoke directly suppresses generation on "
            "peak wildfire days, the hot and dry pre-conditions associated with "
            "wildfire season can elevate weekly averages above non-wildfire baselines."
        )

# Bar charts — yearly comparison
        st.subheader("Year-over-Year Comparison — Same Window")

        # Build per-year aggregates including event year
        all_years_data = []

        # Event year
        all_years_data.append({
            "year": str(event_year),
            "avg_generation": event_mean,
            "avg_pm25": df_window["pm25_mean"].mean(),
            "avg_attenuation": df_window["attenuation_ratio"].mean()
        })

        # Other years
        for yr in sorted(df_other["year"].unique()):
            yr_df = df_other[df_other["year"] == yr]
            all_years_data.append({
                "year": str(yr),
                "avg_generation": yr_df["Volume"].mean(),
                "avg_pm25": yr_df["pm25_mean"].mean(),
                "avg_attenuation": yr_df["attenuation_ratio"].mean()
            })

        df_bars = pd.DataFrame(all_years_data).sort_values("year")
        bar_colors = ["#e76f51" if y == str(event_year) else "#a8dadc" 
                      for y in df_bars["year"]]

        fig2, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].bar(df_bars["year"], df_bars["avg_generation"], color=bar_colors)
        axes[0].set_title("Avg Generation (MW)")
        axes[0].set_ylabel("MW")

        axes[1].bar(df_bars["year"], df_bars["avg_pm25"], color=bar_colors)
        axes[1].set_title("Avg PM2.5 (µg/m³)")
        axes[1].set_ylabel("µg/m³")

        axes[2].bar(df_bars["year"], df_bars["avg_attenuation"], color=bar_colors)
        axes[2].set_title("Avg Attenuation Ratio")
        axes[2].set_ylabel("Ratio")

        plt.suptitle(f"Smoke Event Window — {event}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)