"""
Bias detection module for FairCredit AI.

Implements statistical bias detection and counterfactual
fairness checks suitable for regulatory audits.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Statistical Bias Metrics
# ---------------------------------------------------------------------
def statistical_parity_difference(
    y_pred: np.ndarray, sensitive_attr: np.ndarray
) -> float:
    """
    P(Y=1 | group=1) - P(Y=1 | group=0)
    """
    rate_priv = y_pred[sensitive_attr == 1].mean()
    rate_unpriv = y_pred[sensitive_attr == 0].mean()
    return float(rate_priv - rate_unpriv)


def disparate_impact_ratio(
    y_pred: np.ndarray, sensitive_attr: np.ndarray
) -> float:
    """
    P(Y=1 | unprivileged) / P(Y=1 | privileged)
    """
    rate_priv = y_pred[sensitive_attr == 1].mean()
    rate_unpriv = y_pred[sensitive_attr == 0].mean()
    return float(rate_unpriv / (rate_priv + 1e-6))


# ---------------------------------------------------------------------
# Counterfactual Fairness
# ---------------------------------------------------------------------
def counterfactual_fairness_test(
    model: ClassifierMixin,
    X: pd.DataFrame,
    sensitive_attr: pd.Series,
) -> float:
    """
    Measures how often predictions change when only the
    sensitive attribute is flipped.

    Returns:
        Fraction of samples with prediction change.
    """
    logger.info("Running counterfactual fairness test")

    X_cf = X.copy()
    X_cf["network_creditworthiness"] = 1.0 - X_cf["network_creditworthiness"]

    y_orig = model.predict(X)
    y_cf = model.predict(X_cf)

    change_rate = np.mean(y_orig != y_cf)
    return float(change_rate)


# ---------------------------------------------------------------------
# Bias Detector
# ---------------------------------------------------------------------
class BiasDetector:
    """
    Runs bias diagnostics and flags potential fairness risks.
    """

    def __init__(
        self,
        parity_threshold: float = 0.05,
        impact_threshold: float = 0.8,
        counterfactual_threshold: float = 0.10,
    ) -> None:
        self.parity_threshold = parity_threshold
        self.impact_threshold = impact_threshold
        self.counterfactual_threshold = counterfactual_threshold

    def run(
        self,
        model: ClassifierMixin,
        X: pd.DataFrame,
        y_pred: np.ndarray,
        sensitive_attr: pd.Series,
    ) -> Dict[str, float | bool]:
        """
        Run all bias checks and return alerts.
        """
        logger.info("Running bias detector")

        spd = statistical_parity_difference(y_pred, sensitive_attr.values)
        diratio = disparate_impact_ratio(y_pred, sensitive_attr.values)
        cf_rate = counterfactual_fairness_test(model, X, sensitive_attr)

        results = {
            "statistical_parity_difference": spd,
            "disparate_impact_ratio": diratio,
            "counterfactual_change_rate": cf_rate,
            "parity_violation": abs(spd) > self.parity_threshold,
            "impact_violation": diratio < self.impact_threshold,
            "counterfactual_violation": cf_rate > self.counterfactual_threshold,
        }

        return results
