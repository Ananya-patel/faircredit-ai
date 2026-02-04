"""
Feature engineering module for FairCredit AI.

Transforms raw alternative data into interpretable,
privacy-preserving creditworthiness indicators.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Differential Privacy
# ---------------------------------------------------------------------
def add_gaussian_noise(
    data: np.ndarray,
    epsilon: float,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian noise for differential privacy.

    Args:
        data: Numeric array.
        epsilon: Privacy budget.
        sensitivity: Sensitivity of the query.

    Returns:
        Noisy data.
    """
    sigma = sensitivity / epsilon
    noise = np.random.normal(0.0, sigma, size=data.shape)
    return data + noise


# ---------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------
class AlternativeFeatureEngineer:
    """
    Generates higher-level creditworthiness features from alternative data.
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        logger.info("AlternativeFeatureEngineer initialized")

    # -----------------------------------------------------------------
    # Financial discipline features
    # -----------------------------------------------------------------
    def _payment_regularity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Proxy for bill payment regularity.
        """
        return (
            df["transaction_consistency"] * 0.6
            + df["transaction_frequency"] / (df["transaction_frequency"].max() + 1e-6) * 0.4
        )

    def _income_consistency(self, df: pd.DataFrame) -> np.ndarray:
        """
        Stability of incoming funds.
        """
        return np.sqrt(df["transaction_consistency"] * df["savings_rate"])

    def _emergency_fund_ratio(self, df: pd.DataFrame) -> np.ndarray:
        """
        Proxy for financial resilience.
        """
        return df["savings_rate"] * df["transaction_consistency"]

    # -----------------------------------------------------------------
    # Social & behavioral features
    # -----------------------------------------------------------------
    def _network_creditworthiness(self, df: pd.DataFrame) -> np.ndarray:
        """
        PageRank-inspired proxy for social financial reliability.
        """
        base_score = (
            df["network_diversity"] * 0.7
            + df["transaction_consistency"] * 0.3
        )
        return base_score / (base_score.max() + 1e-6)

    def _community_involvement(self, df: pd.DataFrame) -> np.ndarray:
        """
        Community engagement proxy.
        """
        return (
            df["network_diversity"] * 0.5
            + df["geographic_stability"] * 0.5
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered features with privacy guarantees.

        Args:
            df: Raw alternative data.

        Returns:
            DataFrame with engineered features.
        """
        logger.info("Starting feature engineering")

        features = pd.DataFrame(
            {
                "payment_regularity": self._payment_regularity(df),
                "income_consistency": self._income_consistency(df),
                "emergency_fund_ratio": self._emergency_fund_ratio(df),
                "network_creditworthiness": self._network_creditworthiness(df),
                "community_involvement": self._community_involvement(df),
            }
        )

        logger.info("Applying differential privacy to engineered features")
        for col in features.columns:
            features[col] = add_gaussian_noise(
                features[col].values,
                epsilon=self.epsilon,
                sensitivity=1.0,
            )

        features = features.clip(lower=0.0, upper=1.0)

        logger.info(
            "Feature engineering complete | rows=%d, cols=%d",
            features.shape[0],
            features.shape[1],
        )

        return features
