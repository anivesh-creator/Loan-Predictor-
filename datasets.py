"""
datasets.py

Generate or load a synthetic loan dataset with additional derived features
and a small dataset helper API.

Exports:
- DataConfig: dataclass containing dataset file paths and generation settings
- DatasetBuilder: class responsible for generating and saving datasets
- load_dataset: convenience function to load a CSV dataset into a DataFrame
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


dataclass
class DataConfig:
    """Configuration for dataset generation and storage."""
    output_csv: str = "loan_data.csv"
    random_seed: int = 42
    default_count: int = 500


class DatasetBuilder:
    """
    Build realistic synthetic loan application datasets.

    The builder will create base features and derived features such as:
    - monthly_salary
    - fico_score
    - requested_loan_amount
    - employment_years
    - existing_debt
    - monthly_payment_est (derived)
    - debt_to_income_ratio (derived)
    - loan_to_income_ratio (derived)
    - fico_band (categorical derived column for analysis)
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        np.random.seed(self.config.random_seed)
        logger.info("DatasetBuilder initialized with seed=%s", self.config.random_seed)

    def _generate_base(self, count: int) -> pd.DataFrame:
        """Generate base numeric features using realistic ranges."""
        monthly_salary = np.random.randint(20_000, 300_000, size=count)
        fico_score = np.random.randint(300, 850, size=count)
        requested_loan_amount = np.random.randint(50_000, 5_000_000, size=count)
        employment_years = np.random.randint(0, 40, size=count)
        existing_debt = np.random.randint(0, 2_000_000, size=count)

        df = pd.DataFrame({
            "monthly_salary": monthly_salary,
            "fico_score": fico_score,
            "requested_loan_amount": requested_loan_amount,
            "employment_years": employment_years,
            "existing_debt": existing_debt,
        })
        logger.debug("Base features generated: %s rows", len(df))
        return df

    def _derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe."""
        # Estimate a monthly payment assuming 20-year term (240 months)
        df["monthly_payment_est"] = df["requested_loan_amount"] / 240.0

        # Debt-to-income ratio: monthly debt obligations vs monthly salary
        # add a small epsilon to avoid division by zero
        df["debt_to_income_ratio"] = df["existing_debt"] / (df["monthly_salary"] + 1e-6)

        # Loan-to-income ratio (total requested / annual salary)
        df["loan_to_annual_income_ratio"] = df["requested_loan_amount"] / ((df["monthly_salary"] * 12) + 1e-6)

        # Credit bands
        def _fico_band(score: int) -> str:
            if score >= 800:
                return "excellent"
            if score >= 740:
                return "very_good"
            if score >= 670:
                return "good"
            if score >= 580:
                return "fair"
            return "poor"

        df["fico_band"] = df["fico_score"].map(_fico_band)

        # Simple affordability index (used only to compute target)
        df["affordability_index"] = (
            (df["monthly_salary"] * 3) - (df["monthly_payment_est"] * 100)
        )

        logger.debug("Derived features added")
        return df

    def _generate_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create a semi-realistic target using a rule-based approach with some noise.
        Returns an integer Series with values 0 (rejected) or 1 (approved).
        """
        # Base rules:
        rules = (
            (df["fico_score"] >= 650).astype(int) +
            (df["monthly_salary"] * 3 > df["requested_loan_amount"] / 100).astype(int) +
            (df["employment_years"] >= 1).astype(int) +
            (df["existing_debt"] < df["monthly_salary"] * 0.4).astype(int)
        )

        # Convert rules sum to probability with some randomness
        prob = rules / 4.0
        rng = np.random.default_rng(self.config.random_seed)
        noise = rng.normal(loc=0.0, scale=0.1, size=len(df))
        prob = np.clip(prob + noise, 0.0, 1.0)

        target = (prob > 0.5).astype(int)
        logger.debug("Target generated; approved=%s", int(target.sum()))
        return target

    def build(self, count: Optional[int] = None, output_csv: Optional[str] = None) -> pd.DataFrame:
        """Generate a dataset, compute derived features and save as CSV."""
        count = count or self.config.default_count
        output_csv = output_csv or self.config.output_csv

        logger.info("Generating dataset with %s rows", count)
        df = self._generate_base(count)
        df = self._derive_features(df)
        df["loan_approved"] = self._generate_target(df)

        # Save a subset of columns that the model/pipeline will consume.
        # Keep derived features for model training, but also save original columns.
        df.to_csv(output_csv, index=False)
        logger.info("Saved generated dataset to %s", output_csv)
        return df


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV path, with defensive checks."""
    logger.info("Loading dataset from %s", path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        logger.exception("Dataset file not found: %s", path)
        raise
    required_cols = {
        "monthly_salary", "fico_score", "requested_loan_amount",
        "employment_years", "existing_debt", "loan_approved"
    }
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Dataset missing required columns: %s", missing)
        raise ValueError(f"Dataset missing required columns: {missing}")
    logger.info("Loaded dataset with %s rows", len(df))
    return df


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build(count=builder.config.default_count)
