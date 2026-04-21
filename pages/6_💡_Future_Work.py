import streamlit as st

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