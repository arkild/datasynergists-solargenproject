import streamlit as st
import pandas as pd
import pickle
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPICE Generation Dashboard",
    page_icon="☀️",
    layout="wide"
)

# ── Load model and features ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/rf_best_kkp1.pkl")
    with open("models/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/spice_full_backcast.csv")
    df["dt"] = pd.to_datetime(df["dt"])
    df["generation"] = df["ground_truth"].combine_first(df["predicted_volume"])
    df["is_measured"] = df["ground_truth"].notna()
    return df

# ── Wildfire event detection ──────────────────────────────────────────────────
def detect_smoke_events(df, threshold=0.2, min_duration=1):
    daily_aod = df.groupby(df['dt'].dt.date)['aod_smoke'].mean().reset_index()
    daily_aod.columns = ['date', 'aod_smoke']
    daily_aod['date'] = pd.to_datetime(daily_aod['date'])
    daily_aod['is_smoke'] = daily_aod['aod_smoke'] >= threshold
    daily_aod = daily_aod[daily_aod['date'].dt.month.between(4, 11)]

    events = {}
    in_event = False
    start = None

    for _, row in daily_aod.iterrows():
        if row['is_smoke'] and not in_event:
            in_event = True
            start = row['date']
        elif not row['is_smoke'] and in_event:
            in_event = False
            event_window = daily_aod[
                (daily_aod['date'] >= start) &
                (daily_aod['date'] < row['date'])
            ]
            peak = event_window['aod_smoke'].max()
            peak_date = event_window.loc[event_window['aod_smoke'].idxmax(), 'date']
            duration = (row['date'] - start).days

            if duration == 1:
                label = f"{peak_date.strftime('%b %d, %Y')} — Peak AOD {peak:.2f}"
            else:
                label = (
                    f"{duration}d | "
                    f"{start.strftime('%b %d')}–{(row['date'] - pd.Timedelta(days=1)).strftime('%b %d, %Y')} — "
                    f"Peak AOD {peak:.2f}"
                )

            events[label] = (start.strftime('%Y-%m-%d'),
                             (row['date'] - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                             peak_date.strftime('%Y-%m-%d'))
    return events

# ── Event window + same-DOY baseline (matches methodology used in training/testing notebooks) ──
@st.cache_data
def get_event_window_data(df, event_start_str, event_end_str, padding_days=5):
    event_start = pd.Timestamp(event_start_str)
    event_end   = pd.Timestamp(event_end_str)
    window_start = event_start - pd.Timedelta(days=padding_days)
    window_end   = event_end + pd.Timedelta(days=padding_days)
    doy_start = window_start.dayofyear
    doy_end   = window_end.dayofyear
    event_year = event_start.year

    daily = df.groupby(df["dt"].dt.date).agg(
        aod_smoke = ("aod_smoke", "mean"),
        daily_generation = ("generation", "sum"),
        is_measured = ("is_measured", "first")
    ).reset_index()
    daily.columns = ["date", "aod_smoke", "daily_generation", "is_measured"]
    daily["date"] = pd.to_datetime(daily["date"])

    window_data = daily[
        (daily["date"] >= window_start) & (daily["date"] <= window_end)
    ].copy()

    baseline_data = daily[
        (daily["date"].dt.year != event_year) &
        (daily["date"].dt.dayofyear >= doy_start) &
        (daily["date"].dt.dayofyear <= doy_end)
    ].copy()

    return window_data, baseline_data, window_start, window_end

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("☀️ SPICE Generation")
st.sidebar.markdown("Solar Power Generation Dashboard")