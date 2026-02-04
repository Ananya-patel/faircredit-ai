"""
Fairness metrics and auditing framework for FairCredit AI.

Provides group-wise fairness metrics with statistical robustness.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Core Fairness Metrics
# ---------------------------------------------------------------------
def demographic_parity_difference(
    y_pred: np.ndarray, sensitive_attr: np.ndarray
) -> float:
    """
    Difference in positive prediction rates between groups.
    """
    rates = []
    for group in np.unique(sensitive_attr):
        rates.append(y_pred[sensitive_attr == group].mean())
    return float(np.max(rates) - np.min(rates))


def disparate_impact(
    y_pred: np.ndarray, sensitive_attr: np.ndarray
) -> float:
    """
    Ratio of positive prediction rates (min / max).
    """
    rates = []
    for group in np.unique(sensitive_attr):
        rates.append(y_pred[sensitive_attr == group].mean())
    return float(np.min(rates) / (np.max(rates) + 1e-6))


def equal_opportunity_difference(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray
) -> float:
    """
    Difference in true positive rates across groups.
    """
    tprs = []

    for group in np.unique(sensitive_attr):
        mask = sensitive_attr == group
        tn, fp, fn, tp = confusion_matrix(
            y_true[mask], y_pred[mask], labels=[0, 1]
        ).ravel()
        tpr = tp / (tp + fn + 1e-6)
        tprs.append(tpr)

    return float(np.max(tprs) - np.min(tprs))


# ---------------------------------------------------------------------
# Consistency Metric
# ---------------------------------------------------------------------
def consistency_score(
    X: pd.DataFrame, y_pred: np.ndarray, k: int = 5
) -> float:
    """
    Measures prediction consistency among nearest neighbors.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    inconsistencies = []
    for i in range(len(X)):
        neighbor_preds = y_pred[indices[i][1:]]
        inconsistencies.append(np.abs(y_pred[i] - neighbor_preds).mean())

    return 1.0 - float(np.mean(inconsistencies))


# ---------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------
def bootstrap_metric(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a fairness metric.
    """
    values = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        values.append(metric_fn(y_true[idx], y_pred[idx], sensitive_attr[idx]))

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ---------------------------------------------------------------------
# Comprehensive Audit
# ---------------------------------------------------------------------
class ComprehensiveFairnessAudit:
    """
    Runs a full fairness audit and returns interpretable results.
    """

    def run(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive_attr: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute all fairness metrics.

        Args:
            X: Feature matrix.
            y_true: True labels.
            y_pred: Predicted labels.
            sensitive_attr: Sensitive group labels.

        Returns:
            Dictionary of fairness metrics.
        """
        logger.info("Running comprehensive fairness audit")

        results = {
            "demographic_parity_difference": demographic_parity_difference(
                y_pred, sensitive_attr.values
            ),
            "disparate_impact": disparate_impact(
                y_pred, sensitive_attr.values
            ),
            "equal_opportunity_difference": equal_opportunity_difference(
                y_true.values, y_pred, sensitive_attr.values
            ),
            "consistency_score": consistency_score(X, y_pred),
        }

        logger.info("Fairness audit complete")
        return results
