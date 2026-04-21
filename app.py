import streamlit as st
import pandas as pd
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
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

# ── Wildfire event detection ──────────────────────────────────────────────────
def detect_smoke_events(df, threshold=50, min_duration=1):
    daily_pm25 = df.groupby(df['dt'].dt.date)['pm25_mean'].mean().reset_index()
    daily_pm25.columns = ['date', 'pm25_mean']
    daily_pm25['date'] = pd.to_datetime(daily_pm25['date'])
    daily_pm25['is_smoke'] = daily_pm25['pm25_mean'] >= threshold
    daily_pm25 = daily_pm25[daily_pm25['date'].dt.month.between(4, 11)]

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("☀️ SPICE Generation")
st.sidebar.markdown("Solar Power Generation Dashboard")