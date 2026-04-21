import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app import load_data

df = load_data()

st.title("📊 Client Site — KKP Correlation & Gap Fill")
st.markdown(
    "KKP1 and the client site are strongly correlated (r = **0.916**). "
    "Insights drawn from KKP generation patterns apply to the client site — and where "
    "the client site has missing readings, KKP can fill the gaps."
)

uploaded_file = st.file_uploader("Upload client site generation CSV", type="csv")

if uploaded_file is not None:
    # Load client data
    df_client = pd.read_csv(uploaded_file)
    df_client = df_client.rename(columns={
        "Date and time": "dt",
        "Total system": "kwh"
    })
    df_client["dt"] = pd.to_datetime(df_client["dt"], errors="coerce")
    df_client["kwh"] = pd.to_numeric(df_client["kwh"], errors="coerce")
    df_client["date"] = df_client["dt"].dt.date

    # Client site capacity factor (30.7 kW capacity, daily kWh)
    df_client["cf_client"] = df_client["kwh"] / (30.7 * 24)

    # KKP — aggregate hourly to daily sum, then capacity factor (7000 kW)
    df_kkp = df.copy()
    df_kkp["date"] = df_kkp["dt"].dt.date
    df_kkp_daily = df_kkp.groupby("date")["Volume"].sum().reset_index()
    df_kkp_daily.columns = ["date", "kkp_kwh_sum"]
    df_kkp_daily["cf_kkp"] = (df_kkp_daily["kkp_kwh_sum"] * 1000) / (7000 * 24)

    # Merge on date
    df_merged = df_client.merge(df_kkp_daily, on="date", how="inner")

    # Start from first valid client date
    first_date = df_merged["dt"].min()
    df_merged = df_merged[df_merged["dt"] >= first_date]

    # Identify nulls — zero generation during solar months
    df_merged["is_null"] = (
        (df_merged["cf_client"] == 0) &
        (df_merged["dt"].dt.month.between(4, 10))
    )

    # Correlation parameters in CF space
    r = 0.916
    kkp_cf_mean = df_merged["cf_kkp"].mean()
    client_cf_mean = df_merged[~df_merged["is_null"]]["cf_client"].mean()

    # Gap fill
    df_merged["cf_predicted"] = r * (df_merged["cf_kkp"] - kkp_cf_mean) + client_cf_mean
    df_merged["cf_filled"] = df_merged["cf_client"].copy()
    df_merged.loc[df_merged["is_null"], "cf_filled"] = df_merged.loc[df_merged["is_null"], "cf_predicted"]

    null_count = df_merged["is_null"].sum()

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Pearson r", "0.916")
    col2.metric("Missing Days Filled", f"{null_count}")
    col3.metric("Data Start", str(first_date.date()))

    # Plot 1 — Scatter showing correlation
    st.subheader("Capacity Factor Correlation — KKP vs Client Site")
    df_clean = df_merged[~df_merged["is_null"]]

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.scatter(df_clean["cf_kkp"], df_clean["cf_client"],
                alpha=0.4, s=10, color="#f4a261")
    m, b = np.polyfit(df_clean["cf_kkp"], df_clean["cf_client"], 1)
    x_line = np.linspace(df_clean["cf_kkp"].min(), df_clean["cf_kkp"].max(), 100)
    ax1.plot(x_line, m * x_line + b, color="#2a9d8f", linewidth=1.5,
                label=f"r = 0.916")
    ax1.set_xlabel("KKP Capacity Factor")
    ax1.set_ylabel("Client Site Capacity Factor")
    ax1.set_title("KKP1 vs Client Site — Daily Capacity Factor")
    ax1.legend()
    plt.tight_layout()
    col_scatter, col_empty = st.columns([1, 1])
    with col_scatter:
        st.pyplot(fig1)

    # Plot 2 — Time series with gap fill highlighted
    st.subheader("Generation Over Time — Actual & Gap-Filled")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(df_merged["date"], df_merged["cf_kkp"],
                color="#a8dadc", linewidth=0.8, alpha=0.6, label="KKP1 CF")
    ax2.plot(df_merged["date"], df_merged["cf_client"],
                color="#f4a261", linewidth=1.2, alpha=0.8, label="Client Site Actual CF")

    if null_count > 0:
        df_nulls = df_merged[df_merged["is_null"]]
        ax2.scatter(df_nulls["date"], df_nulls["cf_predicted"],
                    color="#e76f51", s=40, zorder=5,
                    label=f"Gap-Filled ({null_count} days)")

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Capacity Factor")
    ax2.set_title("Client Site vs KKP1 — Capacity Factor Over Time")
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    if null_count > 0:
        st.subheader("Gap-Filled Days")
        st.dataframe(
            df_merged[df_merged["is_null"]][["date", "cf_kkp", "cf_predicted"]]
            .rename(columns={
                "date": "Date",
                "cf_kkp": "KKP CF (Reference)",
                "cf_predicted": "Client Site CF (Predicted)"
            }).reset_index(drop=True),
            use_container_width=True
        )

else:
    st.info("Upload the client site generation CSV to begin analysis.")
    st.markdown("**Expected columns:** `Date and time`, `Total system`")