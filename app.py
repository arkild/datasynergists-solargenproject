import streamlit as st
import pandas as pd
import pickle
import joblib
#for smoke event measurements
from datetime import timedelta

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
                    f"{start.strftime('%b %d')}–{(row['date'] - timedelta(days=1)).strftime('%b %d, %Y')} — "
                    f"Peak AOD {peak:.2f}"
                )

            events[label] = (start.strftime('%Y-%m-%d'),
                            row['date'].strftime('%Y-%m-%d'),
                            peak_date.strftime('%Y-%m-%d'))
    return events

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("☀️ SPICE Generation")
st.sidebar.markdown("Solar Power Generation Dashboard")