"""
Alternative data collection module for FairCredit AI.

This module generates privacy-preserving synthetic alternative credit data
based on published financial inclusion research.

Author: FairCredit AI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Differential Privacy utilities
# ---------------------------------------------------------------------
def add_laplace_noise(
    data: np.ndarray,
    epsilon: float,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """
    Apply Laplace noise for differential privacy.

    Args:
        data: Original numeric data.
        epsilon: Privacy budget.
        sensitivity: Sensitivity of the query.

    Returns:
        Noisy data array.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=data.shape)
    return data + noise


# ---------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------
@dataclass
class DataGenerationConfig:
    n_customers: int = 10_000
    epsilon: float = 0.1
    random_seed: int = 42


# ---------------------------------------------------------------------
# Main Collector
# ---------------------------------------------------------------------
class AlternativeDataCollector:
    """
    Generates synthetic alternative credit data with privacy guarantees.

    Features are inspired by:
    - Mobile money transaction behavior
    - Utility payment consistency
    - Behavioral and social stability indicators
    """

    def __init__(self, config: DataGenerationConfig) -> None:
        self.config = config
        np.random.seed(self.config.random_seed)
        logger.info("AlternativeDataCollector initialized")

    # -----------------------------------------------------------------
    # Core generators
    # -----------------------------------------------------------------
    def _generate_transaction_behavior(self) -> Dict[str, np.ndarray]:
        """
        Simulate mobile money transaction behavior.
        """
        n = self.config.n_customers

        transaction_frequency = np.random.gamma(shape=2.0, scale=3.0, size=n)
        transaction_consistency = np.clip(
            np.random.normal(loc=0.7, scale=0.15, size=n), 0, 1
        )
        savings_rate = np.clip(
            np.random.beta(a=2.0, b=5.0, size=n), 0, 1
        )

        return {
            "transaction_frequency": transaction_frequency,
            "transaction_consistency": transaction_consistency,
            "savings_rate": savings_rate,
        }

    def _generate_social_behavior(self) -> Dict[str, np.ndarray]:
        """
        Simulate social and behavioral stability indicators.
        """
        n = self.config.n_customers

        network_diversity = np.clip(
            np.random.normal(loc=0.5, scale=0.2, size=n), 0, 1
        )
        geographic_stability = np.clip(
            np.random.beta(a=5.0, b=2.0, size=n), 0, 1
        )

        return {
            "network_diversity": network_diversity,
            "geographic_stability": geographic_stability,
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def collect(self) -> pd.DataFrame:
        """
        Generate the full alternative credit dataset with differential privacy.

        Returns:
            Pandas DataFrame with synthetic, privacy-preserving features.
        """
        logger.info("Starting synthetic data generation")

        transaction_data = self._generate_transaction_behavior()
        social_data = self._generate_social_behavior()

        data = {**transaction_data, **social_data}
        df = pd.DataFrame(data)

        logger.info("Applying differential privacy noise")
        for col in df.columns:
            df[col] = add_laplace_noise(
                df[col].values,
                epsilon=self.config.epsilon,
                sensitivity=1.0,
            )

        df = df.clip(lower=0)

        logger.info(
            "Synthetic data generation complete | rows=%d, cols=%d",
            df.shape[0],
            df.shape[1],
        )

        return df
    
    print("AlternativeDataCollector class defined successfully")
    
