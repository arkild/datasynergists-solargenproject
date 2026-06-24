import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from app import load_data, detect_smoke_events, get_event_window_data

df = load_data()
wildfire_events = detect_smoke_events(df)

st.title("🕒 Hourly Smoke Analysis")
st.markdown(
    "Explore hourly solar generation and AOD across a ±3 day window around "
    "a smoke event. Nighttime gaps are shaded — AOD readings across those "
    "gaps are not directly comparable to adjacent daylight hours."
)

df_hourly = df.copy()
df_hourly["date"] = df_hourly["dt"].dt.date
df_hourly["hour"] = df_hourly["dt"].dt.hour

# Use ground truth where available, fall back to backcast prediction
df_hourly["generation"] = df_hourly["ground_truth"].combine_first(df_hourly["predicted_volume"])

event_keys = list(wildfire_events.keys())
gt_start = pd.Timestamp('2022-09-01')

default_index = 0
for i, key in enumerate(event_keys):
    event_start = pd.Timestamp(wildfire_events[key][0])
    if event_start >= gt_start:
        default_index = i
        break

mode = st.radio("Choose mode", ["Wildfire Event", "Custom Date"], horizontal=True)

if mode == "Custom Date":
    selected_date = st.date_input(
        "Choose a date",
        value=df_hourly["date"].min(),
        min_value=df_hourly["date"].min(),
        max_value=df_hourly["date"].max()
    )
    center_dt = pd.Timestamp(selected_date)
    event_choice = "Custom date"
    date_str = selected_date.strftime("%Y-%m-%d")
    # Single-day "event" — padding 3 days each side gives the same ±3 day
    # window, but routed through the same shared function as every other page.
    _, _, window_start, window_end = get_event_window_data(df, date_str, date_str, padding_days=3)
else:
    event_choice = st.selectbox(
        "Select a wildfire event (auto-fills date)",
        event_keys,
        index=default_index
    )
    start_str, end_str, peak_str = wildfire_events[event_choice]
    center_dt = pd.Timestamp(peak_str)
    st.caption(f"Peak smoke day: {center_dt.date()}")
    _, _, window_start, window_end = get_event_window_data(df, start_str, end_str, padding_days=3)

day_df = df_hourly[
    (df_hourly["dt"] >= window_start) &
    (df_hourly["dt"] <= window_end) &
    (df_hourly["solar_elevation"] > 0)
].copy().sort_values("dt").reset_index(drop=True)

if day_df.empty:
    st.warning("No daylight data available for this window.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak Hourly AOD", f"{day_df['aod_smoke'].max():.2f}")
    col2.metric("Average AOD", f"{day_df['aod_smoke'].mean():.2f}")
    col3.metric("Peak Generation", f"{day_df['generation'].max():.3f} MW")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Find night gaps — consecutive rows more than 1 hour apart
    day_df["time_gap"] = day_df["dt"].diff().dt.total_seconds() / 3600
    night_gaps = day_df[day_df["time_gap"] > 1]

    # Mark night gaps with vertical dashed lines
    for _, gap_row in night_gaps.iterrows():
        gap_x = day_df.loc[gap_row.name - 1, "dt"] + (gap_row["dt"] - day_df.loc[gap_row.name - 1, "dt"]) / 2
        ax1.axvline(gap_x, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    # Generation line — split by source (measured vs backcast)
    day_df["is_measured"] = day_df["ground_truth"].notna()

    measured_seg = day_df[day_df["is_measured"]]
    backcast_seg = day_df[~day_df["is_measured"]]

    ax1.plot(
        measured_seg["dt"], measured_seg["generation"],
        marker="o", linewidth=2, color="#e76f51", markersize=3,
        label="Solar Generation — Measured (MW)"
    )
    ax1.plot(
        backcast_seg["dt"], backcast_seg["generation"],
        marker="o", linewidth=2, color="#e76f51", markersize=3,
        linestyle="--", alpha=0.6,
        label="Solar Generation — Backcast Estimate (MW)"
    )
    ax1.set_xlabel("Date & Hour")
    ax1.set_ylabel("Solar Generation (MW)")

    # AOD line
    ax2 = ax1.twinx()
    ax2.plot(
        day_df["dt"], day_df["aod_smoke"],
        marker="s", linestyle="--", linewidth=2,
        color="#2a9d8f", markersize=3, label="AOD"
    )
    ax2.set_ylabel("Aerosol Optical Depth (AOD)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.title(f"Daylight Hours ±3 Days — {event_choice}")
    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    show_cols = ["dt", "hour", "generation", "aod_smoke"]
    extra_cols = ["shortwave", "cloud_pct", "solar_elevation", "attenuation_ratio",
                "Temperature (degrees C)", "Relative Humidity"]
    for col in extra_cols:
        if col in day_df.columns:
            show_cols.append(col)

    with st.expander("View hourly data table"):
        st.dataframe(day_df[show_cols].reset_index(drop=True),
                        width='stretch')