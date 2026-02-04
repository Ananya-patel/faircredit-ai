"""
Fairness-aware baseline classifiers for FairCredit AI.

Implements interpretable baseline models with optional
fairness reweighting.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Fairness-Constrained Classifier
# ---------------------------------------------------------------------
class FairnessConstrainedClassifier:
    """
    Logistic Regression with optional fairness-aware reweighting.
    """

    def __init__(
        self,
        fairness_constraint: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            fairness_constraint:
                None | "demographic_parity" | "equal_opportunity"
        """
        self.fairness_constraint = fairness_constraint
        self.random_state = random_state

        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state,
        )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attr: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the classifier.

        Args:
            X: Feature matrix.
            y: Binary labels.
            sensitive_attr: Optional sensitive attribute for reweighting.
        """
        logger.info("Training FairnessConstrainedClassifier")

        sample_weight = None

        if self.fairness_constraint and sensitive_attr is not None:
            logger.info(
                "Applying fairness reweighting: %s",
                self.fairness_constraint,
            )
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y if self.fairness_constraint == "demographic_parity" else y & sensitive_attr,
            )

        self.model.fit(X, y, sample_weight=sample_weight)

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute standard performance metrics.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        metrics = {
            "auc_roc": roc_auc_score(y, y_prob),
            "f1_score": f1_score(y, y_pred),
        }

        logger.info(
            "Evaluation metrics | AUC=%.3f, F1=%.3f",
            metrics["auc_roc"],
            metrics["f1_score"],
        )

        return metrics
