import streamlit as st
import matplotlib.pyplot as plt
import shap
from app import load_model, load_data, render_sidebar

render_sidebar(current="shap")

df = load_data()
model, feature_names = load_model()

st.title("🧠 SHAP Analysis")
st.markdown(
    "SHAP explains how each feature pushes individual predictions above or below "
    "the baseline average prediction. This may take 30-60 seconds to compute."
)

test = df[df["dt"] > "2024-12-31"].reset_index(drop=True)
X_test = test[feature_names].dropna().copy()

if X_test.empty:
    st.warning("No test data available for SHAP analysis.")
else:
    max_rows = min(300, len(X_test))
    X_shap = X_test.sample(max_rows, random_state=42) if len(X_test) > max_rows else X_test

    with st.spinner("Calculating SHAP values — this may take a moment..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_shap, check_additivity=False)

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(
        shap_values,
        max_display=len(feature_names),
        show=False
    )
    st.pyplot(plt.gcf(), clear_figure=True)

    st.info(
        "Features with wider horizontal spread have greater influence. "
        "Points to the right increase predicted generation, while points to the left decrease it."
    )