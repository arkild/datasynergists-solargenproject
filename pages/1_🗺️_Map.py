import streamlit as st
import pandas as pd
import requests
from datetime import date
from app import load_data, detect_smoke_events

st.markdown("""
    <style>
    .block-container {
        max-width: 1100px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

df = load_data()
wildfire_events = detect_smoke_events(df)

st.title("🗺️ Sky Conditions During Wildfire Events")
st.markdown(
    "Visualize what the sky looked like over Edmonton during key wildfire "
    "smoke events using NASA GIBS satellite imagery."
)

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    event = st.selectbox(
        "Select a wildfire event (auto-fills date)",
        ["Custom date"] + list(wildfire_events.keys())
    )

with col2:
    layer = st.selectbox(
        "Satellite layer",
        [
            "MODIS_Terra_CorrectedReflectance_TrueColor",
            "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        ]
    )

with col3:
    if event == "Custom date":
        selected_date = st.date_input(
            "Date",
            value=date(2023, 5, 19),
            min_value=date(2015, 1, 1),
            max_value=date(2025, 12, 31)
        )
    else:
        event_center = pd.Timestamp(wildfire_events[event][2]).date()
        day_offset = st.slider(
            "Days around peak",
            min_value=-5,
            max_value=5,
            value=0,
            format="%d days"
        )
        selected_date = event_center + pd.Timedelta(days=day_offset)

# ── Event label ───────────────────────────────────────────────────────────────
if event != "Custom date":
    st.info(f"📍 {event} | Viewing: {selected_date}")

# ── Map ───────────────────────────────────────────────────────────────────────
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
        from PIL import Image, ImageDraw
        import io

        img_width, img_height = 800, 600
        lat_min, lat_max = 50, 58
        lon_min, lon_max = -120, -105
        edmonton_lat = 53.5461
        edmonton_lon = -113.4938

        img = Image.open(io.BytesIO(response.content))
        x = int((edmonton_lon - lon_min) / (lon_max - lon_min) * img_width)
        y = int((lat_max - edmonton_lat) / (lat_max - lat_min) * img_height)

        draw = ImageDraw.Draw(img)
        r = 6
        draw.ellipse([x-r, y-r, x+r, y+r], outline="red", width=3)
        draw.text((x+10, y-10), "Edmonton", fill="red")
        st.image(img, caption=f"Edmonton region — {date_str} | Lat: 53.5461° N | Lon: -113.4938° W", width='stretch')
    else:
        st.warning("Could not load satellite image for this date.")
except Exception as e:
    st.error(f"Error loading image: {e}")