"""
prediction.py

LoanPredictor encapsulates the trained pipeline loading and prediction/explanation logic.

Usage:
  predictor = LoanPredictor(model_dir="model")
  result = predictor.predict(
      monthly_salary=100000,
      fico_score=750,
      requested_loan_amount=2_000_000,
      employment_years=5,
      existing_debt=20000
  )
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


dataclass
class PredictorConfig:
    model_dir: str = "model"
    pipeline_filename: str = "loan_pipeline.pkl"
    metadata_filename: str = "loan_metadata.json"


class LoanPredictor:
    """
    Load a trained pipeline and provide predict() and explain() methods.
    """

    def __init__(self, config: PredictorConfig = None):
        self.config = config or PredictorConfig()
        self.pipeline_path = f"{self.config.model_dir}/{self.config.pipeline_filename}"
        self.metadata_path = f"{self.config.model_dir}/{self.config.metadata_filename}"
        logger.info("Initializing LoanPredictor. Loading pipeline from %s", self.pipeline_path)

        try:
            self.pipeline = joblib.load(self.pipeline_path)
        except Exception as exc:
            logger.exception("Failed to load pipeline from %s", self.pipeline_path)
            raise

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)
        except Exception:
            logger.warning("Metadata not found or invalid at %s; proceeding with best-effort defaults", self.metadata_path)
            self.metadata = {"feature_columns": [
                "monthly_salary",
                "fico_score",
                "requested_loan_amount",
                "employment_years",
                "existing_debt",
                "monthly_payment_est",
                "debt_to_income_ratio",
                "loan_to_annual_income_ratio"
            ]}

        self.feature_columns = self.metadata["feature_columns"]
        logger.info("Predictor ready. Using feature columns: %s", self.feature_columns)

    def _validate_inputs(self, payload: Dict[str, Any]) -> None:
        """Defensive validation for input payload."""
        for key in ("monthly_salary", "fico_score", "requested_loan_amount", "employment_years", "existing_debt"):
            if key not in payload:
                raise ValueError(f"Missing required input: {key}")
            if not isinstance(payload[key], (int, float, np.integer, np.floating)):
                raise ValueError(f"Invalid type for {key}: expected numeric, got {type(payload[key])}")

    def _build_feature_row(self, monthly_salary: float, fico_score: float, requested_loan_amount: float,
                           employment_years: float, existing_debt: float) -> pd.DataFrame:
        """Create a single-row DataFrame consistent with pipeline's expected features."""
        monthly_payment_est = requested_loan_amount / 240.0
        debt_to_income_ratio = existing_debt / (monthly_salary + 1e-6)
        loan_to_annual_income_ratio = requested_loan_amount / ((monthly_salary * 12) + 1e-6)

        row = {
            "monthly_salary": float(monthly_salary),
            "fico_score": float(fico_score),
            "requested_loan_amount": float(requested_loan_amount),
            "employment_years": float(employment_years),
            "existing_debt": float(existing_debt),
            "monthly_payment_est": float(monthly_payment_est),
            "debt_to_income_ratio": float(debt_to_income_ratio),
            "loan_to_annual_income_ratio": float(loan_to_annual_income_ratio),
        }
        df = pd.DataFrame([row], columns=self.feature_columns)
        return df

    def predict(self, monthly_salary: float, fico_score: float, requested_loan_amount: float,
                employment_years: float, existing_debt: float) -> Dict[str, Any]:
        """
        Return a dictionary with:
         - approved: bool
         - confidence: float (percentage)
         - reasons: List[str]
         - raw_probs: List[float]
        """
        payload = {
            "monthly_salary": monthly_salary,
            "fico_score": fico_score,
            "requested_loan_amount": requested_loan_amount,
            "employment_years": employment_years,
            "existing_debt": existing_debt,
        }
        self._validate_inputs(payload)

        X = self._build_feature_row(**payload)
        probs = None
        try:
            probs = self.pipeline.predict_proba(X)[0]
            prediction = int(self.pipeline.predict(X)[0])
        except AttributeError:
            # If the pipeline's final estimator doesn't support predict_proba
            logger.warning("Pipeline estimator does not support predict_proba; using deterministic prediction")
            prediction = int(self.pipeline.predict(X)[0])
            probs = [float(prediction), 1.0 - float(prediction)]

        confidence = float(probs[prediction]) * 100.0
        approved = prediction == 1

        reasons = self.explain(payload, approved, confidence)

        return {
            "approved": approved,
            "confidence": confidence,
            "reasons": reasons,
            "raw_probs": list(map(float, probs))
        }

    def explain(self, payload: Dict[str, float], approved: bool, confidence: float) -> List[str]:
        """
        Produce human-readable reasons, combining simple rule-based checks.
        These explanations are intentionally conservative and deterministic.
        """
        reasons: List[str] = []

        fico = float(payload["fico_score"])
        salary = float(payload["monthly_salary"])
        requested = float(payload["requested_loan_amount"])
        years = float(payload["employment_years"])
        debt = float(payload["existing_debt"])

        # Credit score
        if fico >= 800:
            reasons.append("✅ Excellent credit score")
        elif fico >= 740:
            reasons.append("✅ Very good credit score")
        elif fico >= 670:
            reasons.append("✅ Good credit score")
        elif fico >= 580:
            reasons.append("⚠️ Fair credit score")
        else:
            reasons.append("❌ Poor credit score (usually <580)")

        # Affordability
        monthly_payment = requested / 240.0
        if monthly_payment < salary * 0.3:
            reasons.append("✅ Monthly payment estimate looks affordable")
        elif monthly_payment < salary * 0.45:
            reasons.append("⚠️ Monthly payment borderline vs salary")
        else:
            reasons.append("❌ Monthly payment likely too high for salary")

        # Employment
        if years >= 3:
            reasons.append("✅ Stable employment history")
        elif years >= 1:
            reasons.append("⚠️ Short employment history")
        else:
            reasons.append("❌ Very limited or no employment history")

        # Existing debt
        if debt < salary * 0.3:
            reasons.append("✅ Existing debt is low relative to salary")
        elif debt < salary * 0.4:
            reasons.append("⚠️ Moderate existing debt")
        else:
            reasons.append("❌ High existing debt relative to salary")

        # Model confidence note
        reasons.append(f"Model confidence: {confidence:.1f}%")
        reasons.append("Decision produced by an automated model — treat as indicative, not definitive.")

        return reasons


if __name__ == "__main__":
    predictor = LoanPredictor()
    sample = predictor.predict(
        monthly_salary=100_000,
        fico_score=750,
        requested_loan_amount=2_000_000,
        employment_years=5,
        existing_debt=20_000
    )
    print(sample)
