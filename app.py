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

# ── Shared sidebar nav ──
def render_sidebar(current=None):
    st.sidebar.title("☀️ SPICE Generation")
    st.sidebar.markdown("Solar Power Generation Dashboard")
    st.sidebar.divider()

    pages = [
        ("app.py", "🏠 Home"),
        ("pages/1_🗺️_Map.py", "🗺️ Map"),
        ("pages/2_⚡_The_Paradox.py", "⚡ The Paradox"),
        ("pages/3_🕒_Hourly_Smoke.py", "🕒 Hourly Smoke"),
        ("pages/4_🔮_Prediction_Check.py", "🔮 Prediction Check"),
        ("pages/5_🔬_XAI.py", "🔬 XAI"),
        ("pages/6_🧠_SHAP.py", "🧠 SHAP"),
        ("pages/7_💡_Future_Work.py", "💡 Future Work"),
    ]
    for path, label in pages:
        st.sidebar.page_link(path, label=label)

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar(current="home")

# ── Home page content ──────────────────────────────────────────────────────────
st.title("☀️ SPICE Generation Dashboard", anchor=False)

st.markdown(
    "This dashboard, originally created as a NorQuest College project for SPICE "
    "(Solar Power Investment Cooperative of Edmonton), is a multi-page analysis of "
    "solar generation data for EPCOR's kīsikāw pīsim solar plant. Using data from "
    "2022–2024 and testing blind on 2025, we analyzed the potential impact of smoke "
    "from wildfires on solar energy production. Our findings show smoke has less "
    "impact on solar generation than we expected."
)

st.markdown(
    "Wildfire events are tracked using Copernicus CAMS's `aod_smoke` measure — "
    "Aerosol Optical Depth, a dimensionless measure of how much light transmission "
    "through the atmosphere is reduced. Any day averaging 0.2 AOD or higher is "
    "flagged as a smoke day; consecutive flagged days are grouped into a single "
    "event, shown in the dropdown as a date range with its peak AOD value rather "
    "than separate single-day entries."
)

st.markdown("Explore each part of the analysis below.")
st.divider()

# ── Section: Map ────────────────────────────────────────────────────────────────
st.subheader("🗺️ Map", anchor=False)
st.markdown(
    "The map page gives a snapshot of sky conditions using NASA's satellite "
    "imagery. Select a wildfire event from the dropdown to automatically jump to "
    "that date, and see what the sky looked like during the event and in the days "
    "surrounding the peak smoke day."
)
st.divider()

# ── Section: The Paradox ────────────────────────────────────────────────────────
st.subheader("⚡ The Paradox", anchor=False)
st.markdown(
    "Our key finding: while smoky skies tend to reduce power generation on the "
    "day, the days surrounding a wildfire event often generate more power than "
    "the same period across the prior ~10 years. This page graphs daily generation "
    "through and around each event, compared against that same-time-of-year "
    "baseline — with a separate marker for \"Low Cloud Days,\" when cloud cover "
    "was both heavy (75%+) and low-lying (below 2,000m), since those conditions "
    "block more sunlight than smoke does."
)
st.divider()

# ── Section: Hourly Smoke ───────────────────────────────────────────────────────
st.subheader("🕒 Hourly Smoke", anchor=False)
st.markdown(
    "This page breaks down KKP1's power production hour by hour — using actual "
    "measured generation from September 2022 onward, and backcast model "
    "predictions for the period before that (2015 through August 2022). Plotted "
    "alongside hourly `aod_smoke` values, this view lets you spot potential "
    "correlations between smoke levels and generation during specific wildfire "
    "events that the daily/weekly view can smooth over."
)
st.divider()

# ── Section: Prediction Check ───────────────────────────────────────────────────
st.subheader("🔮 Prediction Check", anchor=False)
st.markdown(
    "Pick any date and hour to compare the model's prediction against what "
    "actually happened. The page automatically tells you which of three regimes "
    "you're in, since each means something different: dates before September "
    "2022 are pure backcast, with no ground truth to check against — \"actual\" "
    "is really just the model's own historical estimate. Dates from September "
    "2022 through 2024 fall inside the model's training data, so a close match "
    "here is expected, not proof of accuracy. Only 2025 is a true blind test: "
    "the model never saw this data during training, and that's where its real "
    "performance — R² = 0.8839 — comes from."
)
st.markdown(
    "Every prediction also comes with the weather conditions behind it — "
    "irradiance, cloud cover, cloud base height, smoke level, and more — plus "
    "an optional SHAP breakdown showing exactly how each factor pushed that "
    "specific prediction up or down."
)
st.divider()

# ── Section: XAI ─────────────────────────────────────────────────────────────────
st.subheader("🔬 XAI", anchor=False)
st.markdown(
    "Our prediction model is a Random Forest — a collection of decision trees "
    "that each split predictions based on yes/no answers about specific feature "
    "values. This page has three parts: first, which features the model relies "
    "on most to make its predictions; second, a scatter plot comparing predicted "
    "generation against actual generation, so you can see how closely the model "
    "tracks reality; and third, a \"partial dependence plot\" showing how a "
    "single feature — say, cloud cover — moves the prediction up or down on "
    "average, with all other factors held steady."
)
st.divider()

# ── Section: SHAP ────────────────────────────────────────────────────────────────
st.subheader("🧠 SHAP", anchor=False)
st.markdown(
    "This page shows the same kind of relationship as the partial dependence "
    "plot — how features push predictions up or down — but as a \"beeswarm\" "
    "plot instead. Where the PDP shows one averaged effect per feature, this "
    "plot shows every individual prediction as its own dot: features are ranked "
    "top-to-bottom by overall importance, and each dot's position left or right "
    "shows whether that specific prediction was pushed up or down, and by how "
    "much. It's effectively feature importance and partial dependence combined, "
    "but at the resolution of individual predictions rather than averages."
)