
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

API_ENDPOINT = f"{API_BASE}/predict"

# -----------------------------
# UI
# -----------------------------
st.title("💳 FairCredit – Credit Eligibility Simulator")
st.caption("This demo uses behavioral signals only. No personal data is collected.")

savings_rate = st.slider("Savings Rate", 0.0, 1.0, 0.2)
payment_regularity = st.slider("Payment Regularity", 0.0, 1.0, 0.8)
transaction_consistency = st.slider("Transaction Consistency", 0.0, 1.0, 0.7)
network_diversity = st.slider("Network Diversity", 0.0, 1.0, 0.2)

# -----------------------------
# Action
# -----------------------------
if st.button("🔍 Check Credit Eligibility"):
    payload = {
        "payment_regularity": payment_regularity,
        "income_consistency": transaction_consistency,
        "emergency_fund_ratio": savings_rate,
        "network_creditworthiness": network_diversity,
        "community_involvement": payment_regularity,
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

                approved = result["approved"]
                probability = result["probability"]

                st.success(
                    f"Decision: **{'Approved ✅' if approved else 'Declined ❌'}**"
                )
                st.metric("Approval Probability", round(probability, 3))

                st.caption(
                    "⚖️ This decision was generated without using protected attributes."
                )

        except Exception as e:
            st.error("Failed to connect to FairCredit API")
            st.exception(e)
