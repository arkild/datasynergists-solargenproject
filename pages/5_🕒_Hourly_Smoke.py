import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from app import load_data, detect_smoke_events

df = load_data()
wildfire_events = detect_smoke_events(df)

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