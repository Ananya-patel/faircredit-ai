import streamlit as st
import requests
import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

API_URL = API_URL = st.secrets.get(
    "API_URL",
    "https://faircredit-ai.onrender.com"
)


st.set_page_config(
    page_title="FairCredit Simulator",
    page_icon="⚖️",
    layout="centered"
)

st.title(" FairCredit — Credit Eligibility Simulator")

st.markdown(
    """
    ###  Simulation Mode (No Real Personal Data)

    This is a **simulated credit eligibility check** based on
    behavioral financial inputs.  
    **No real personal or financial data is collected or stored.**
    """
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📥 Enter Simulated Financial Behavior")

monthly_income = st.slider(
    "Monthly Income (₹)",
    min_value=10000,
    max_value=200000,
    step=5000,
    value=50000
)

savings_rate = st.slider(
    "Savings Rate (%)",
    min_value=0,
    max_value=60,
    step=5,
    value=20
) / 100

payment_regularity = st.slider(
    "Payment Regularity",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=0.85
)

transaction_consistency = st.slider(
    "Transaction Consistency",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=0.80
)

employment_stability = st.slider(
    "Employment Stability (Years)",
    min_value=0,
    max_value=10,
    step=1,
    value=3
)

network_diversity = st.slider(
    "Network Diversity",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=0.6
)

st.divider()

# -----------------------------
# Submit
# -----------------------------
if st.button("🔍 Check Credit Eligibility"):

    payload = {
        "monthly_income": monthly_income,
        "savings_rate": savings_rate,
        "payment_regularity": payment_regularity,
        "transaction_consistency": transaction_consistency,
        "employment_stability": employment_stability,
        "network_diversity": network_diversity
    }

    with st.spinner("Evaluating credit eligibility..."):
        try:
            response = requests.post(
                f"{API_URL}/predict_explain",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                st.success(" Evaluation Complete")

                st.subheader(" Decision")
                st.metric(
                    label="Eligibility Decision",
                    value=result.get("decision", "Unknown")
                )

                st.subheader(" Risk Score")
                st.metric(
                    label="Estimated Risk",
                    value=f"{round(result.get('risk_score', 0) * 100, 2)} %"
                )

                st.subheader(" Key Contributing Factors")
                for reason in result.get("explanations", []):
                    st.write(f"- {reason}")

                st.info(
                    " Fairness Note: No protected attributes were used in this evaluation."
                )

            else:
                st.error(" API Error. Please try again later.")

        except Exception as e:
            st.error(f"Connection failed: {e}")

st.divider()

st.caption(
    "Disclaimer: This simulator uses synthetic inputs for demonstration purposes only. "
    "It does not represent real credit decisions."
)
