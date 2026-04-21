import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from app import load_model, load_data

df = load_data()
model, feature_names = load_model()

st.title("🔬 Model Explainability")
st.markdown(
    "This page explains what drives the Random Forest model's solar generation predictions "
    "using feature importance, SHAP summary plots, and partial dependence plots."
)

# Build the test dataset in the same way as the evaluation period.
# Here, we are only using rows after 2024-12-31 because that is our test period.
# We also keep only the exact features used by the trained model.
test = df[df["dt"] > "2024-12-31"].reset_index(drop=True)
X_test = test[feature_names].dropna().copy()

# If there is no test data available, show a warning instead of breaking the app.
if X_test.empty:
    st.warning("No test data available for XAI.")
else:
    
    # Feature importance tells us which variables the Random Forest used the most
    # across all trees in the model.
    st.subheader("1. Feature Importance — Top 15")
    importances = model.feature_importances_

    # Convert feature importances into a pandas Series so we can sort them easily.
    # We keep only the top 15 most important features for a cleaner plot.
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)

    # Create a horizontal bar plot because it is easier to read feature names this way.
    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    feat_imp.plot(kind="barh", ax=ax_imp, color="#f4a261")
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("Top 15 Feature Importances — Random Forest")
    plt.tight_layout()
    st.pyplot(fig_imp)

    # This short interpretation helps the client understand the main takeaway.
    st.info(
        "Shortwave radiation and related lag features dominate the model. "
        "This means the model is primarily learning from incoming solar energy and time-based patterns."
    )

    # -----------------------------
    # 3. Actual vs Predicted Plot
    # -----------------------------
    # This plot checks model performance visually.
    # If the predictions are close to the diagonal dashed line,
    # it means predicted values are close to actual values.
    st.subheader("3. Actual vs Predicted")
    st.markdown(
        "This plot compares the model's predicted solar generation with the actual observed generation."
    )

    # Get the true target values (actual generation) for the same rows used in X_test.
    y_test = test.loc[X_test.index, "Volume"]

    # Predict generation using the trained Random Forest model.
    y_pred = model.predict(X_test)

    # Scatter plot of actual vs predicted values.
    fig_ap, ax_ap = plt.subplots(figsize=(7, 6))
    ax_ap.scatter(y_test, y_pred, alpha=0.4, s=12, color="#2a9d8f")

    # Add a dashed 45-degree line.
    # Perfect predictions would fall exactly on this line.
    line_min = min(y_test.min(), y_pred.min())
    line_max = max(y_test.max(), y_pred.max())
    ax_ap.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=1)

    ax_ap.set_xlabel("Actual Generation (MW)")
    ax_ap.set_ylabel("Predicted Generation (MW)")
    ax_ap.set_title("Actual vs Predicted — Random Forest")

    plt.tight_layout()
    st.pyplot(fig_ap)

    # Help the reader understand what the plot means.
    st.info(
        "Points closer to the dashed diagonal line indicate better predictions. "
        "Large deviations from the line represent prediction error."
    )

    # -----------------------------
    # 4. PDP
    # -----------------------------
    # Partial Dependence Plot (PDP) shows the average effect of one feature
    # on the model prediction, while averaging out the influence of other features.
    st.subheader("4. Partial Dependence Plot")
    st.markdown(
        "This plot shows the average effect of one feature on the model prediction while averaging over the others."
    )

    # We remove lag features here to keep the dropdown simpler and easier to interpret.
    core_features = [f for f in feature_names if "lag" not in f]
    feature_to_plot = st.selectbox("Select feature to explore", core_features)

    from sklearn.inspection import PartialDependenceDisplay

    feature_idx = feature_names.index(feature_to_plot)

    with st.spinner("Calculating..."):
        fig_pdp, ax_pdp = plt.subplots(figsize=(8, 4))
        PartialDependenceDisplay.from_estimator(
            model,
            df[feature_names].dropna(),
            [feature_idx],
            ax=ax_pdp
        )
        ax_pdp.set_title(f"Partial Dependence — {feature_to_plot}")
        st.pyplot(fig_pdp)