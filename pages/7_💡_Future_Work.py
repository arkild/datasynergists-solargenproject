import streamlit as st
from app import render_sidebar

render_sidebar(current="future_work")

st.title("💡 What's Next")
st.markdown(
    "The current model achieves R² = 0.8839 tested blind on 2025. "
    "Here are the most promising directions for improvement."
)

st.subheader("⏳ Update with 2026 Data Later")
st.markdown(
    "KKP1 started generating power on September 1, 2022. Our training data for this " 
    "model only covers from 2022 to 2024 and tested on 2025. Once the weather and " 
    "power generation are up on AESO and NASA's APIs, we can train the model from " 
    "2022 to 2025 and test on 2026 to assess its robustness.")

st.subheader("📡 Expanding to More Sites")
st.markdown(
    "The correlation framework used to fill client site gaps (r = 0.916) suggests "
    "KKP1 is a strong regional proxy. Using KKP1, we could use the same approach "
    "with solar data from other SPICE projects as a means of null handling "
    "and predictions, so long as the other sites have a similar correlation. "
)