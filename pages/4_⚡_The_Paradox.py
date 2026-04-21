import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app import load_data, detect_smoke_events

df = load_data()
wildfire_events = detect_smoke_events(df)

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