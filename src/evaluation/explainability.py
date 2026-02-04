"""
Explainability utilities for FairCredit AI.

Provides SHAP-based local explanations suitable for
regulatory review and audit trails.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ShapExplainer:
    """
    SHAP explainer wrapper for linear models.
    """

    def __init__(self, model: LogisticRegression) -> None:
        self.model = model
        self.explainer = shap.LinearExplainer(
            model,
            masker=shap.maskers.Independent(np.zeros((1, model.coef_.shape[1]))),
        )

    def explain(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Return feature attributions for a single sample.
        """
        if len(X) != 1:
            raise ValueError("Explain expects exactly one sample")

        shap_values = self.explainer.shap_values(X.values)[0]
        return dict(zip(X.columns, shap_values.tolist()))
