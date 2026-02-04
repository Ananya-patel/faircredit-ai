"""
Data cleaning and validation module for FairCredit AI.

Ensures data quality, handles outliers, imputes missing values,
and enforces regulatory-style constraints.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def iqr_bounds(series: pd.Series, factor: float = 1.5) -> Tuple[float, float]:
    """
    Compute IQR-based lower and upper bounds.

    Args:
        series: Numeric pandas Series.
        factor: IQR multiplier.

    Returns:
        (lower_bound, upper_bound)
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


# ---------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------
class DataCleaner:
    """
    Cleans and validates engineered credit features.
    """

    def __init__(
        self,
        max_missing_ratio: float = 0.05,
        max_data_age_days: int = 30,
    ) -> None:
        self.max_missing_ratio = max_missing_ratio
        self.max_data_age_days = max_data_age_days
        logger.info("DataCleaner initialized")

    # -----------------------------------------------------------------
    # Validation checks
    # -----------------------------------------------------------------
    def _check_missingness(self, df: pd.DataFrame) -> None:
        missing_ratio = df.isna().mean().max()
        logger.info("Max missing ratio: %.4f", missing_ratio)

        if missing_ratio > self.max_missing_ratio:
            raise ValueError(
                f"Missing data ratio {missing_ratio:.2%} exceeds "
                f"allowed threshold {self.max_missing_ratio:.2%}"
            )

    def _check_freshness(self, df: pd.DataFrame) -> None:
        """
        Simulate freshness check using a synthetic timestamp column.
        """
        if "data_timestamp" not in df.columns:
            logger.warning("No data_timestamp column found; skipping freshness check")
            return

        max_age = datetime.utcnow() - df["data_timestamp"].max()
        logger.info("Data age: %s", max_age)

        if max_age > timedelta(days=self.max_data_age_days):
            raise ValueError("Data freshness constraint violated")

    # -----------------------------------------------------------------
    # Cleaning steps
    # -----------------------------------------------------------------
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap outliers using IQR bounds.
        """
        cleaned = df.copy()

        for col in cleaned.select_dtypes(include=[np.number]).columns:
            lower, upper = iqr_bounds(cleaned[col])
            cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

        return cleaned

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using median (privacy-safe).
        """
        imputed = df.copy()

        for col in imputed.select_dtypes(include=[np.number]).columns:
            median_value = imputed[col].median()
            imputed[col] = imputed[col].fillna(median_value)

        return imputed

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate feature data.

        Args:
            df: Engineered feature DataFrame.

        Returns:
            Cleaned and validated DataFrame.
        """
        logger.info("Starting data cleaning")

        self._check_missingness(df)
        self._check_freshness(df)

        df_clean = self._handle_outliers(df)
        df_clean = self._impute_missing(df_clean)

        logger.info(
            "Data cleaning complete | rows=%d, cols=%d",
            df_clean.shape[0],
            df_clean.shape[1],
        )

        return df_clean
