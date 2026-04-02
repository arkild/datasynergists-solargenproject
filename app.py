import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests
from datetime import datetime, date
import shap


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPICE Generation Dashboard",
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
    #Stick to wildfire season range
    daily_pm25 = daily_pm25[daily_pm25['date'].dt.month.between(4, 11)]
    
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
                event_window = daily_pm25[
                    (daily_pm25['date'] >= start) &
                    (daily_pm25['date'] < row['date'])
                ]
                peak = event_window['pm25_mean'].max()
                peak_date = event_window.loc[event_window['pm25_mean'].idxmax(), 'date']
                label = f"{peak_date.strftime('%b %d, %Y')} — Peak {peak:.0f} µg/m³"
                events[label] = (start.strftime('%Y-%m-%d'),
                                row['date'].strftime('%Y-%m-%d'),
                                peak_date.strftime('%Y-%m-%d'))
    
    return events

wildfire_events = detect_smoke_events(df, threshold=50)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("☀️ SPICE Generation")
st.sidebar.markdown("Solar Power Generation Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Map", "📊 Compare to Client", "🔮 Prediction Check", "⚡ The Paradox", "🕒 Hourly Smoke Analysis", "🔬 XAI", "💡 Future Work", "🤖 RAG Chatbot"]
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
                from PIL import ImageDraw, ImageFont
                # This draws a dot on the map where Edmonton is based on pixel calculations
                # BBOX: lat 50-58, lon -120 to -105
                # Image: 800x600
                img_width, img_height = 800, 600
                lat_min, lat_max = 50, 58
                lon_min, lon_max = -120, -105

                edmonton_lat = 53.5461
                edmonton_lon = -113.4938

                x = int((edmonton_lon - lon_min) / (lon_max - lon_min) * img_width)
                y = int((lat_max - edmonton_lat) / (lat_max - lat_min) * img_height)

                draw = ImageDraw.Draw(img)
                r = 6
                draw.ellipse([x-r, y-r, x+r, y+r], outline="red", width=3)
                draw.text((x+10, y-10), "Edmonton", fill="red")
                st.image(img, caption=f"Edmonton region — {date_str}", use_container_width=True)
            else:
                st.warning("Could not load satellite image for this date.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Compare to Client
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Compare to Client":
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

        fig1, ax1 = plt.subplots(figsize=(5, 4))
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
        col_scatter, col_empty = st.columns([1, 1])
        with col_scatter:
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
# PAGE 3 — Prediction Check
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction Check":
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
    start, end, peak_date_str = wildfire_events[event]
    peak_date = pd.Timestamp(peak_date_str)

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
        # Highlight peak smoke day
        peak_row = df_daily[df_daily["date"] == peak_date]
        if not peak_row.empty:
            ax.scatter(peak_row["date"], peak_row["avg_generation"],
                       s=200, zorder=6, facecolors='none',
                       edgecolors='red', linewidth=2.5,
                       label="Peak smoke day")

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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 - HOURLY SMOKE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🕒 Hourly Smoke Analysis":
    st.title("🕒 Hourly Smoke Analysis")
    st.markdown(
        "Explore hourly solar generation and PM2.5 across a ±3 day window around "
        "a smoke event. Nighttime gaps are shaded — PM2.5 readings across those "
        "gaps are not directly comparable to adjacent daylight hours."
    )

    df_hourly = df.copy()
    df_hourly["date"] = df_hourly["dt"].dt.date
    df_hourly["hour"] = df_hourly["dt"].dt.hour

    event_choice = st.selectbox(
        "Select a wildfire event (auto-fills date)",
        ["Custom date"] + list(wildfire_events.keys())
    )

    if event_choice == "Custom date":
        selected_date = st.date_input(
            "Choose a date",
            value=df_hourly["date"].min(),
            min_value=df_hourly["date"].min(),
            max_value=df_hourly["date"].max()
        )
        center_dt = pd.Timestamp(selected_date)
    else:
        start, end, peak_date_str = wildfire_events[event_choice]
        center_dt = pd.Timestamp(peak_date_str)
        st.caption(f"Peak smoke day: {center_dt.date()}")

    # ±3 days window, daylight only
    window_start = center_dt - pd.Timedelta(days=3)
    window_end = center_dt + pd.Timedelta(days=3)

    day_df = df_hourly[
        (df_hourly["dt"] >= window_start) &
        (df_hourly["dt"] <= window_end) &
        (df_hourly["solar_elevation"] > 0)
    ].copy().sort_values("dt").reset_index(drop=True)

    if day_df.empty:
        st.warning("No daylight data available for this window.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak Hourly PM2.5", f"{day_df['pm25_mean'].max():.1f} µg/m³")
        col2.metric("Average PM2.5", f"{day_df['pm25_mean'].mean():.1f} µg/m³")
        col3.metric("Peak Generation", f"{day_df['Volume'].max():.3f} MW")

        fig, ax1 = plt.subplots(figsize=(14, 5))

        # Find night gaps — consecutive rows more than 1 hour apart
        day_df["time_gap"] = day_df["dt"].diff().dt.total_seconds() / 3600
        night_gaps = day_df[day_df["time_gap"] > 1]

        # Mark night gaps with vertical dashed lines
        for _, gap_row in night_gaps.iterrows():
            gap_x = day_df.loc[gap_row.name - 1, "dt"] + (gap_row["dt"] - day_df.loc[gap_row.name - 1, "dt"]) / 2
            ax1.axvline(gap_x, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
            ax1.annotate("🌙", xy=(gap_x, ax1.get_ylim()[1]),
                        ha="center", fontsize=10, color="gray")

        from matplotlib.patches import Patch
        night_patch = Patch(facecolor='navy', alpha=0.15, label='Nighttime gap')

        # Generation line
        ax1.plot(
            day_df["dt"], day_df["Volume"],
            marker="o", linewidth=2, color="#e76f51", markersize=3,
            label="Solar Generation (MW)"
        )
        ax1.set_xlabel("Date & Hour")
        ax1.set_ylabel("Solar Generation (MW)")

        # PM2.5 line
        ax2 = ax1.twinx()
        ax2.plot(
            day_df["dt"], day_df["pm25_mean"],
            marker="s", linestyle="--", linewidth=2,
            color="#2a9d8f", markersize=3, label="PM2.5 (µg/m³)"
        )
        ax2.set_ylabel("PM2.5 (µg/m³)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + [night_patch], 
            labels1 + labels2 + ['Nighttime gap'], 
            loc="upper left", fontsize=8)

        plt.title(f"Daylight Hours ±3 Days — {event_choice}")
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

        show_cols = ["dt", "hour", "Volume", "pm25_mean"]
        extra_cols = ["shortwave", "cloud_pct", "solar_elevation", "attenuation_ratio",
                    "Temperature (degrees C)", "Relative Humidity"]
        for col in extra_cols:
            if col in day_df.columns:
                show_cols.append(col)

        with st.expander("View hourly data table"):
            st.dataframe(day_df[show_cols].reset_index(drop=True),
                         use_container_width=True)
            
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Future Work":
    st.title("💡 What's Next")
    st.markdown(
        "The current model achieves R² = 0.86 tested blind on 2025. "
        "Here are the most promising directions for improvement."
    )

    st.subheader("🌥️ Cloud Type Data")
    st.markdown(
        "Cloud coverage percentage alone was not in the model's top 15 features — "
        "the model gravitated toward shortwave radiation and attenuation ratio instead. "
        "This makes sense: a sky that is 80% covered by cirrus clouds behaves very "
        "differently from one covered by cumulus. Cirrus is largely transparent to "
        "shortwave radiation while cumulus blocks it almost entirely. "
        "Adding cloud fraction by altitude level — low, mid, and high clouds separately "
        "— would give the model an explicit cloud type signal. "
        "**ERA5 reanalysis data from Copernicus** provides exactly this at no cost "
        "and would integrate naturally into the existing data pipeline."
    )

    st.subheader("🌫️ Aerosol Optical Depth")
    st.markdown(
        "PM2.5 is a ground-level measurement and a reasonable smoke proxy, but aerosol "
        "optical depth (AOD) measures the actual column of particulates between the "
        "solar panel and the sun — which is what directly affects generation. "
        "The **Copernicus Atmosphere Monitoring Service (CAMS)** provides AOD data "
        "publicly and would complement the existing attenuation ratio feature."
    )

    st.subheader("📡 Expanding to More Sites")
    st.markdown(
        "The correlation framework used to fill client site gaps (r = 0.916) suggests "
        "KKP1 is a strong regional proxy. Using KKP1, we could use the same approach "
        "with solar data from other SPICE projects as a means of null handling "
        "and predictions, so long as the other sites have a similar correlation. "
    )

# ══════════════════════════════════════════════════════════════════════════════
## ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — XAI
# As we know Random forest is a blackbox model, to understand how it makes predictions, we need to use XAI techniques.
# So techniques like feature importance, Shap, and partial dependence plots can help us understand which features are driving the model's predictions.

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 XAI":
    st.title("🔬 Model Explainability")
    st.markdown(
        "This page explains what drives the Random Forest model's solar generation predictions "
        "using feature importance, SHAP summary plots, and partial dependence plots."
    )

    # Build the test dataset in the same way as the evaluation period.
    # Here, we are only using rows after 2024-12-31 because that is our test period.
    # We also keep only the exact features used by the trained model.
    test = df[df["dt"] > "2024-12-31"].reset_index(drop=True)
    X_test = test[feature_names].dropna().copy()

    # If there is no test data available, show a warning instead of breaking the app.
    if X_test.empty:
        st.warning("No test data available for XAI.")
    else:
       
        # Feature importance tells us which variables the Random Forest used the most
        # across all trees in the model.
        st.subheader("1. Feature Importance — Top 15")
        importances = model.feature_importances_

        # Convert feature importances into a pandas Series so we can sort them easily.
        # We keep only the top 15 most important features for a cleaner plot.
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)

        # Create a horizontal bar plot because it is easier to read feature names this way.
        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        feat_imp.plot(kind="barh", ax=ax_imp, color="#f4a261")
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Top 15 Feature Importances — Random Forest")
        plt.tight_layout()
        st.pyplot(fig_imp)

        # This short interpretation helps the client understand the main takeaway.
        st.info(
            "Shortwave radiation and related lag features dominate the model. "
            "This means the model is primarily learning from incoming solar energy and time-based patterns."
        )

        # -----------------------------
        # 2. SHAP summary plot
        # -----------------------------
        # SHAP gives a more detailed explanation than feature importance.
        # shap work by taking mean value of the generation prediction across the dataset as a baseline,
        #  then calculating how much each feature pushes individual predictions above or below that baseline.
        st.subheader("2. SHAP Summary Plot")
        st.markdown(
            "This plot shows which features have the strongest overall influence on predictions, "
            "and whether high or low values push generation upward or downward."
        )

        # SHAP is slow if we use for whole datasets, so we use a sample of up to 300 rows
        # to make the app faster while still giving a useful explanation.
        # we used random_state=42 to ensure the same sample is used each time for consistency in explanations.
        max_rows = min(300, len(X_test))
        X_shap = X_test.sample(max_rows, random_state=42) if len(X_test) > max_rows else X_test

        # Create a TreeExplainer because Random Forest is a tree-based model.
        # check_additivity=False helps avoid small numerical mismatch errors.
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_shap, check_additivity=False)

        # The beeswarm plot shows all features separately.
        # max_display=len(feature_names) is important because it avoids grouping
        # many features into one misleading "sum of other features" row.
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(
            shap_values,
            max_display=len(feature_names),
            show=False
        )
        st.pyplot(plt.gcf(), clear_figure=True)

        # Simple interpretation for the user/client.
        st.info(
            "Features with wider horizontal spread have greater influence. "
            "Points to the right increase predicted generation, while points to the left decrease it."
        )

        # -----------------------------
        # 3. Actual vs Predicted Plot
        # -----------------------------
        # This plot checks model performance visually.
        # If the predictions are close to the diagonal dashed line,
        # it means predicted values are close to actual values.
        st.subheader("3. Actual vs Predicted")
        st.markdown(
            "This plot compares the model's predicted solar generation with the actual observed generation."
        )

        # Get the true target values (actual generation) for the same rows used in X_test.
        y_test = test.loc[X_test.index, "Volume"]

        # Predict generation using the trained Random Forest model.
        y_pred = model.predict(X_test)

        # Scatter plot of actual vs predicted values.
        fig_ap, ax_ap = plt.subplots(figsize=(7, 6))
        ax_ap.scatter(y_test, y_pred, alpha=0.4, s=12, color="#2a9d8f")

        # Add a dashed 45-degree line.
        # Perfect predictions would fall exactly on this line.
        line_min = min(y_test.min(), y_pred.min())
        line_max = max(y_test.max(), y_pred.max())
        ax_ap.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=1)

        ax_ap.set_xlabel("Actual Generation (MW)")
        ax_ap.set_ylabel("Predicted Generation (MW)")
        ax_ap.set_title("Actual vs Predicted — Random Forest")

        plt.tight_layout()
        st.pyplot(fig_ap)

        # Help the reader understand what the plot means.
        st.info(
            "Points closer to the dashed diagonal line indicate better predictions. "
            "Large deviations from the line represent prediction error."
        )

        # -----------------------------
        # 4. PDP
        # -----------------------------
        # Partial Dependence Plot (PDP) shows the average effect of one feature
        # on the model prediction, while averaging out the influence of other features.
        st.subheader("4. Partial Dependence Plot")
        st.markdown(
            "This plot shows the average effect of one feature on the model prediction while averaging over the others."
        )

        # We remove lag features here to keep the dropdown simpler and easier to interpret.
        core_features = [f for f in feature_names if "lag" not in f]
        feature_to_plot = st.selectbox("Select feature to explore", core_features)

        from sklearn.inspection import PartialDependenceDisplay

        feature_idx = feature_names.index(feature_to_plot)

        with st.spinner("Calculating..."):
            fig_pdp, ax_pdp = plt.subplots(figsize=(8, 4))
            PartialDependenceDisplay.from_estimator(
                model,
                df[feature_names].dropna(),
                [feature_idx],
                ax=ax_pdp
            )
            ax_pdp.set_title(f"Partial Dependence — {feature_to_plot}")
            st.pyplot(fig_pdp)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — RAG CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 RAG Chatbot":
    st.title("🤖 Ask the Data (beta)")
    st.markdown(
        "Ask questions about solar generation, smoke events, and weather conditions. "
        "The chatbot retrieves relevant data from the KKP1 dataset to answer your questions."
    )
    st.info(
        "⚠️ **Note:** This chatbot has no memory between questions — each question is answered independently. "
        "For best results, include all relevant context in your question (e.g. 'What were the PM2.5 readings during smoke events in May 2023?'). "
        "Responses may also be cut off due to character limits."
    )

    from sentence_transformers import SentenceTransformer, util
    from huggingface_hub import InferenceClient

    HF_TOKEN = st.secrets["HF_TOKEN"]

    @st.cache_resource
    def load_embedder():
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_resource
    def load_hf_client():
        return InferenceClient(api_key=HF_TOKEN)

    @st.cache_data
    def build_documents(_df):
        df_day = _df.copy()
        df_day["date"] = df_day["dt"].dt.date
        daily = df_day.groupby("date").agg(
            avg_generation=("Volume", "mean"),
            max_generation=("Volume", "max"),
            avg_pm25=("pm25_mean", "mean"),
            avg_shortwave=("shortwave", "mean"),
            avg_cloud=("cloud_pct", "mean"),
        ).reset_index()

        docs = {}
        for _, row in daily.iterrows():
            smoke = "a smoke event was occurring" if row["avg_pm25"] > 50 else "no significant smoke"
            text = (
                f"On {row['date']}, average solar generation was {row['avg_generation']:.3f} MW "
                f"with a peak of {row['max_generation']:.3f} MW. "
                f"Average PM2.5 was {row['avg_pm25']:.1f} µg/m³ — {smoke}. "
                f"Average shortwave radiation was {row['avg_shortwave']:.1f} W/m² "
                f"and cloud coverage was {row['avg_cloud']:.1f}%."
            )
            docs[str(row["date"])] = text
        return docs

    @st.cache_data
    def build_embeddings(docs):
        embedder = load_embedder()
        return {k: embedder.encode(v, convert_to_tensor=True) for k, v in docs.items()}

    def retrieve_context(query, doc_embeddings, documents, top_k=5):
        embedder = load_embedder()
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        
        import re
        months = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12"
        }
        
        year_match = re.search(r'\b(20\d{2})\b', query.lower())
        month_match = next((m for m in months if m in query.lower()), None)
        
        # If asking about worst/highest/most smoke, sort by PM2.5 directly
        superlative_terms = ["worst", "highest", "most", "peak", "maximum", "max", "bad"]
        smoke_terms = ["smoke", "pm2.5", "pm25", "air quality"]
        
        if any(t in query.lower() for t in superlative_terms) and \
        any(t in query.lower() for t in smoke_terms):
            
            pm25_scores = {}
            for k, v in documents.items():
                if k == "app_usage":
                    continue
                if year_match and year_match.group(1) not in k:
                    continue
                if month_match and f"-{months[month_match]}-" not in k:
                    continue
                match = re.search(r'PM2\.5 was ([\d.]+)', v)
                if match:
                    pm25_scores[k] = float(match.group(1))
            
            if pm25_scores:
                top_keys = sorted(pm25_scores, key=pm25_scores.get, reverse=True)[:top_k]
                return "\n\n".join(documents[k] for k in top_keys)
        
        # Date filter
        if year_match or month_match:
            year = year_match.group(1) if year_match else None
            month = months[month_match] if month_match else None
            
            filtered_keys = []
            for k in doc_embeddings:
                if k == "app_usage":
                    continue
                if year and year not in k:
                    continue
                if month and f"-{month}-" not in k:
                    continue
                filtered_keys.append(k)
            
            if filtered_keys:
                scores = {k: util.pytorch_cos_sim(query_embedding, doc_embeddings[k]).item()
                        for k in filtered_keys}
                top_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
                return "\n\n".join(documents[k] for k in top_keys)
        
        # Fall back to full semantic search
        scores = {k: util.pytorch_cos_sim(query_embedding, doc_embeddings[k]).item()
                for k in doc_embeddings}
        top_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return "\n\n".join(documents[k] for k in top_keys)

    def query_hf(prompt):
        client = load_hf_client()
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a solar energy analyst. Answer questions clearly and concisely based on the data provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    # Build docs and embeddings
    with st.spinner("Loading data..."):
        documents = build_documents(df)
        doc_embeddings = build_embeddings(documents)

    # App usage guide
    app_guide = {
        "app_usage": (
            "This app has 8 pages. The Map page shows NASA satellite imagery of smoke events. "
            "The Compare to Client page fills missing generation data using KKP1 as a proxy. "
            "The Prediction Check page lets you look up model predictions vs actual generation for any date. "
            "The Paradox page shows how smoke events reduce daily generation but surrounding weeks "
            "often have higher generation due to dry sunny conditions. "
            "The Hourly Smoke Analysis page shows PM2.5 and generation hour by hour around smoke events. "
            "The XAI page shows feature importance and partial dependence plots. "
            "The Future Work page describes planned improvements using Copernicus data."
        )
    }
    documents.update(app_guide)
    app_embedder = load_embedder()
    doc_embeddings["app_usage"] = app_embedder.encode(
        app_guide["app_usage"], convert_to_tensor=True
    )

    query = st.text_input(
        "Ask a question about the solar data or how to use this app",
        placeholder="e.g. What was generation like during smoke events in 2023?"
    )

    if query:
        with st.spinner("Retrieving relevant data and generating answer..."):
            context = retrieve_context(query, doc_embeddings, documents, top_k=10)
            prompt = (
                f"Use the data below to answer the question clearly and concisely.\n\n"
                f"Data:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
            answer = query_hf(prompt)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("View retrieved data chunks"):
            st.text(context)