
import os
import streamlit as st
import requests

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="FairCredit Simulator",
    page_icon="💳",
    layout="centered",
)

# -----------------------------
# API configuration
# -----------------------------
API_BASE = os.getenv(
    "API_URL",
    "https://faircredit-ai.onrender.com"  # FastAPI service URL
)

API_ENDPOINT = f"{API_BASE}/simulate"

# -----------------------------
# UI
# -----------------------------
st.title("💳 FairCredit – Credit Eligibility Simulator")
st.caption("This demo uses behavioral signals only. No personal data is collected.")

monthly_income = st.slider("Monthly Income", 0.0, 200000.0, 30000.0)
savings_rate = st.slider("Savings Rate", 0.0, 1.0, 0.2)
payment_regularity = st.slider("Payment Regularity", 0.0, 1.0, 0.8)
transaction_consistency = st.slider("Transaction Consistency", 0.0, 1.0, 0.7)
employment_stability = st.slider("Employment Stability (Years)", 0, 10, 3)
network_diversity = st.slider("Network Diversity", 0.0, 1.0, 0.2)

# -----------------------------
# Action
# -----------------------------
if st.button("🔍 Check Credit Eligibility"):
    payload = {
        "monthly_income": monthly_income,
        "savings_rate": savings_rate,
        "payment_regularity": payment_regularity,
        "transaction_consistency": transaction_consistency,
        "employment_stability": employment_stability,
        "network_diversity": network_diversity,
    }

    with st.spinner("Sending data to FairCredit API..."):
        try:
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                timeout=15,
            )

            if response.status_code != 200:
                st.error(f"API Error: {response.status_code}")
                st.text(response.text)
            else:
                result = response.json()

                st.success(f"Decision: **{result['decision']}**")
                st.metric("Risk Score", result["risk_score"])

                st.subheader("🔎 Explanation")
                st.json(result["explanations"])

                st.caption(result["fairness_note"])

        except Exception as e:
            st.error("Failed to connect to FairCredit API")
            st.exception(e)
