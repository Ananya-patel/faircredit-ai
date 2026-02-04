"""
Monitoring dashboard for FairCredit AI.

Provides live model monitoring, fairness drift indicators,
and data drift visualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FairCredit Monitoring Dashboard",
    layout="wide",
)

st.title(" FairCredit — Model Monitoring Dashboard")

# ---------------------------------------------------------------------
# Simulated metrics (placeholder for production hooks)
# ---------------------------------------------------------------------
st.header(" Model Performance")

auc_roc = np.random.uniform(0.80, 0.90)
f1_score = np.random.uniform(0.75, 0.88)
latency_ms = np.random.uniform(40, 120)

col1, col2, col3 = st.columns(3)
col1.metric("AUC-ROC", f"{auc_roc:.3f}")
col2.metric("F1-Score", f"{f1_score:.3f}")
col3.metric("Avg Latency (ms)", f"{latency_ms:.1f}")

# ---------------------------------------------------------------------
# Fairness Monitoring
# ---------------------------------------------------------------------
st.header("⚖️ Fairness Metrics")

dp_diff = np.random.uniform(0.00, 0.08)
eo_diff = np.random.uniform(0.00, 0.07)

fcol1, fcol2 = st.columns(2)
fcol1.metric(
    "Demographic Parity Diff",
    f"{dp_diff:.3f}",
    delta="OK" if dp_diff < 0.05 else "ALERT",
)
fcol2.metric(
    "Equal Opportunity Diff",
    f"{eo_diff:.3f}",
    delta="OK" if eo_diff < 0.05 else "ALERT",
)

if dp_diff >= 0.05 or eo_diff >= 0.05:
    st.error(" Fairness threshold violated — retraining recommended")
else:
    st.success(" Fairness within acceptable limits")

# ---------------------------------------------------------------------
# Data Drift (Synthetic)
# ---------------------------------------------------------------------
st.header(" Data Drift Detection")

drift_score = np.random.uniform(0.0, 1.0)

st.progress(drift_score)
st.caption("Drift score (0 = stable, 1 = severe drift)")

if drift_score > 0.7:
    st.warning(" Significant data drift detected")
else:
    st.success(" No significant data drift")

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.caption("FairCredit AI • Ethical ML Monitoring • Demo Mode")

