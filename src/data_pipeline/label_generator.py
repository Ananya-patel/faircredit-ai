"""
Label generation module for FairCredit AI.

Creates a realistic proxy target for creditworthiness
based on engineered financial and behavioral features.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Label Generator
# ---------------------------------------------------------------------
class CreditLabelGenerator:
    """
    Generates binary creditworthiness labels.

    Label = 1 → creditworthy
    Label = 0 → non-creditworthy
    """

    def __init__(
        self,
        approval_threshold: float = 0.6,
        noise_std: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self.approval_threshold = approval_threshold
        self.noise_std = noise_std
        np.random.seed(random_seed)
        logger.info("CreditLabelGenerator initialized")

    # -----------------------------------------------------------------
    # Internal score
    # -----------------------------------------------------------------
    def _credit_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute a continuous latent credit score.
        """
        score = (
            0.30 * df["payment_regularity"]
            + 0.25 * df["income_consistency"]
            + 0.20 * df["emergency_fund_ratio"]
            + 0.15 * df["network_creditworthiness"]
            + 0.10 * df["community_involvement"]
        )

        noise = np.random.normal(0.0, self.noise_std, size=len(df))
        return np.clip(score + noise, 0.0, 1.0)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def generate(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate binary creditworthiness labels.

        Args:
            df: Cleaned feature DataFrame.

        Returns:
            (X, y) where:
              X → features
              y → binary target
        """
        logger.info("Generating creditworthiness labels")

        credit_score = self._credit_score(df)
        labels = (credit_score >= self.approval_threshold).astype(int)

        logger.info(
            "Label distribution | approved=%.2f%%",
            labels.mean() * 100,
        )

        return df.copy(), pd.Series(labels, name="creditworthy")
