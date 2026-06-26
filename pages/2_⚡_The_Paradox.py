import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app import load_data, detect_smoke_events, get_event_window_data, render_sidebar

render_sidebar(current="paradox")

df = load_data()
wildfire_events = detect_smoke_events(df)

st.title("⚡ The Wildfire Paradox")
st.markdown(
    "Wildfires reduce solar generation **on the day** due to smoke — "
    "but the conditions that cause wildfires (hot, clear, dry weather) "
    "mean the **surrounding week** often shows **higher than average** generation "
    "compared to equivalent weeks in non-wildfire years."
)

event_keys = list(wildfire_events.keys())
gt_start = pd.Timestamp('2022-09-01')

default_index = 0
for i, key in enumerate(event_keys):
    event_start_check = pd.Timestamp(wildfire_events[key][0])
    if event_start_check >= gt_start:
        default_index = i
        break

event = st.selectbox("Select a wildfire event", event_keys, index=default_index)
start_str, end_str, peak_str = wildfire_events[event]
peak_date = pd.Timestamp(peak_str)
event_start = pd.Timestamp(start_str)
event_end = pd.Timestamp(end_str)
event_year = event_start.year

if event_start < gt_start:
    st.warning(
        "⚠️ This event predates KKP1's commissioning (September 1, 2022). "
        "All generation values shown for this period are model-based backcast "
        "estimates, not measured ground truth."
    )

window_data, baseline_data, window_start, window_end = get_event_window_data(
    df, start_str, end_str, padding_days=5
)

if window_data.empty:
    st.warning("No data for this event range.")
else:
    baseline_mean = baseline_data["daily_generation"].mean()
    event_mean = window_data["daily_generation"].mean()

    # Worst day specifically within the smoke event, not the padded window
    smoke_only = window_data[
        (window_data["date"] >= event_start) & (window_data["date"] <= event_end)
    ]
    event_min = smoke_only["daily_generation"].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("Event Window Total (avg/day)", f"{event_mean:.2f} MWh",
                f"{event_mean - baseline_mean:+.2f} MWh vs baseline")
    col2.metric("Baseline (Same Period, Other Years)", f"{baseline_mean:.2f} MWh")
    col3.metric(
        "Worst Day During Smoke Event",
        f"{event_min:.2f} MWh",
        f"{event_min - baseline_mean:+.2f} MWh vs baseline"
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    # colors = ["#a8dadc", "#457b9d", "#1d3557", "#e9c46a", "#2a9d8f", "#264653"]
    # baseline_data = baseline_data.copy()
    # baseline_data["year"] = baseline_data["date"].dt.year
    # baseline_data["doy"] = baseline_data["date"].dt.dayofyear

    # for i, yr in enumerate(sorted(baseline_data["year"].unique())):
    #     yr_data = baseline_data[baseline_data["year"] == yr].sort_values("doy").copy()
    #     yr_data["plot_date"] = pd.to_datetime(
    #         yr_data["doy"].apply(
    #             lambda d: pd.Timestamp(f"{event_year}-01-01") + pd.Timedelta(days=int(d) - 1)
    #         )
    #     )
    #     ax.plot(
    #         yr_data["plot_date"], yr_data["daily_generation"],
    #         alpha=0.4, linewidth=1.2,
    #         color=colors[i % len(colors)],
    #         label=str(yr)
    #     )

    ax.plot(
        window_data["date"], window_data["daily_generation"],
        marker="o", linewidth=2.5, color="#e76f51",
        label=f"{event_year} (Event)", zorder=5
    )

    ax.axvspan(event_start, event_end, alpha=0.15, color="gray", label="Smoke Event")

    ax.axhline(
        baseline_mean, linestyle="--", color="#2a9d8f", linewidth=1.5,
        label=f"Baseline avg ({baseline_mean:.2f} MWh)"
    )

    peak_row = window_data[window_data["date"] == peak_date]
    if not peak_row.empty:
        ax.scatter(peak_row["date"], peak_row["daily_generation"],
                    s=200, zorder=6, facecolors='none',
                    edgecolors='red', linewidth=2.5,
                    label="Peak smoke day")
        
    # Flag days within the window meeting the same low-cloud criteria as the notebook
    window_full = df[
        (df["dt"] >= window_start) & (df["dt"] <= window_end)
    ].copy()
    daily_cloud_flag = window_full.groupby(window_full["dt"].dt.date).agg(
        cloud_pct=("cloud_pct", "mean"),
        cloudbase_m=("cloudbase_m", "mean")
    ).reset_index()
    daily_cloud_flag.columns = ["date", "cloud_pct", "cloudbase_m"]
    daily_cloud_flag["date"] = pd.to_datetime(daily_cloud_flag["date"])
    daily_cloud_flag["is_low_cloud"] = (
        (daily_cloud_flag["cloud_pct"] >= 75) &
        (daily_cloud_flag["cloudbase_m"] <= 2000)
    )

    low_cloud_days = daily_cloud_flag[daily_cloud_flag["is_low_cloud"]]

    # Shade low-cloud days with diagonal hatching so they're visually distinct from smoke shading
    for idx, row in low_cloud_days.iterrows():
        ax.axvspan(
            row["date"], row["date"] + pd.Timedelta(days=1),
            alpha=0.25, facecolor='none', edgecolor='#264653',
            hatch='///', linewidth=0,
            label='Low Cloud Day' if idx == low_cloud_days.index[0] else None
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Daily Generation (MWh)")
    ax.set_title(f"Solar Generation: {event} (±5 days) vs Same Period Other Years")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
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

    daily_full = df.groupby(df["dt"].dt.date).agg(
        aod_smoke = ("aod_smoke", "mean"),
        attenuation_ratio = ("attenuation_ratio", "mean"),
        daily_generation = ("generation", "sum")
    ).reset_index()
    daily_full.columns = ["date", "aod_smoke", "attenuation_ratio", "daily_generation"]
    daily_full["date"] = pd.to_datetime(daily_full["date"])
    daily_full["year"] = daily_full["date"].dt.year
    daily_full["doy"] = daily_full["date"].dt.dayofyear

    doy_start = window_start.dayofyear
    doy_end = window_end.dayofyear

    all_years_data = []

    # Event year — from window_data already built above
    event_window_full = daily_full[
        (daily_full["date"] >= window_start) & (daily_full["date"] <= window_end)
    ]
    all_years_data.append({
        "year": str(event_year),
        "avg_generation": event_window_full["daily_generation"].mean(),
        "avg_aod": event_window_full["aod_smoke"].mean(),
        "avg_attenuation": event_window_full["attenuation_ratio"].mean()
    })

    # Other years — same DOY range
    other_years_full = daily_full[
        (daily_full["year"] != event_year) &
        (daily_full["doy"] >= doy_start) &
        (daily_full["doy"] <= doy_end)
    ]
    for yr in sorted(other_years_full["year"].unique()):
        yr_df = other_years_full[other_years_full["year"] == yr]
        all_years_data.append({
            "year": str(yr),
            "avg_generation": yr_df["daily_generation"].mean(),
            "avg_aod": yr_df["aod_smoke"].mean(),
            "avg_attenuation": yr_df["attenuation_ratio"].mean()
        })

    df_bars = pd.DataFrame(all_years_data).sort_values("year")
    df_bars["year_label"] = df_bars["year"].apply(lambda y: f"'{y[2:]}")
    bar_colors = ["#e76f51" if y == str(event_year) else "#a8dadc"
                    for y in df_bars["year"]]

    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(df_bars["year_label"], df_bars["avg_generation"], color=bar_colors)
    axes[0].set_title("Avg Daily Generation (MWh)")
    axes[0].set_ylabel("MWh")

    axes[1].bar(df_bars["year_label"], df_bars["avg_aod"], color=bar_colors)
    axes[1].set_title("Avg AOD")
    axes[1].set_ylabel("AOD")

    axes[2].bar(df_bars["year_label"], df_bars["avg_attenuation"], color=bar_colors)
    axes[2].set_title("Avg Attenuation Ratio")
    axes[2].set_ylabel("Ratio")

    plt.suptitle(f"Smoke Event Window — {event}", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig2)