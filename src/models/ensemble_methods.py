"""
Fair ensemble and calibration methods for FairCredit AI.

Combines multiple models and applies group-wise probability calibration
to balance performance and fairness.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Fair Ensemble
# ---------------------------------------------------------------------
class FairEnsemble:
    """
    Ensemble of baseline and debiased models with fairness-aware weighting.
    """

    def __init__(self, weight_baseline: float = 0.5) -> None:
        if not 0.0 <= weight_baseline <= 1.0:
            raise ValueError("weight_baseline must be in [0, 1]")
        self.weight_baseline = weight_baseline
        self.weight_debiased = 1.0 - weight_baseline

        self.baseline_model = None
        self.debiased_model = None

    def fit(
        self,
        baseline_model,
        debiased_model,
    ) -> None:
        """
        Register trained models.
        """
        self.baseline_model = baseline_model
        self.debiased_model = debiased_model
        logger.info("FairEnsemble initialized with trained models")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Weighted ensemble probability.
        """
        if self.baseline_model is None or self.debiased_model is None:
            raise RuntimeError("Models not fitted to ensemble")

        p_base = self.baseline_model.predict_proba(X)
        p_debias = self.debiased_model.predict_proba(X.values)

        return (
            self.weight_baseline * p_base
            + self.weight_debiased * p_debias
        )

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------
# Group-wise Calibration
# ---------------------------------------------------------------------
class GroupCalibrator:
    """
    Calibrates probabilities separately for each group.
    """

    def __init__(self) -> None:
        self.calibrators: Dict[int, CalibratedClassifierCV] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attr: pd.Series,
    ) -> None:
        """
        Train per-group calibrators.
        """
        logger.info("Fitting group-wise calibrators")

        for group in sensitive_attr.unique():
            mask = sensitive_attr == group
            calibrator = CalibratedClassifierCV(
                base_estimator=LogisticRegression(max_iter=1000),
                method="sigmoid",
                cv=3,
            )
            calibrator.fit(X[mask], y[mask])
            self.calibrators[int(group)] = calibrator

    def predict_proba(
        self, X: pd.DataFrame, sensitive_attr: pd.Series
    ) -> np.ndarray:
        """
        Calibrated probability prediction.
        """
        probs = np.zeros(len(X))

        for group, calibrator in self.calibrators.items():
            mask = sensitive_attr == group
            probs[mask] = calibrator.predict_proba(X[mask])[:, 1]

        return probs
