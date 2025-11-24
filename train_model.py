"""
train_model.py

Train a scikit-learn pipeline for loan approval prediction. The pipeline:
- selects features
- applies scaling
- trains a RandomForestClassifier inside a Pipeline object

Saves:
- A serialized pipeline (joblib) at model/loan_pipeline.pkl
- Metadata (json) at model/loan_metadata.json
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from datasets import load_dataset, DataConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


dataclass
class TrainConfig:
    dataset_path: str = DataConfig().output_csv
    model_dir: str = "model"
    pipeline_filename: str = "loan_pipeline.pkl"
    metadata_filename: str = "loan_metadata.json"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5


class LoanModelTrainer:
    """
    Encapsulates training logic for the loan approval predictor.
    """

    def __init__(self, config: TrainConfig = None):
        self.config = config or TrainConfig()
        os.makedirs(self.config.model_dir, exist_ok=True)
        logger.info("LoanModelTrainer initialized. Models will be saved to %s", self.config.model_dir)

    def _get_feature_columns(self) -> List[str]:
        """
        Return the feature columns used by the model. Keep order deterministic.
        """
        return [
            "monthly_salary",
            "fico_score",
            "requested_loan_amount",
            "employment_years",
            "existing_debt",
            # derived features
            "monthly_payment_est",
            "debt_to_income_ratio",
            "loan_to_annual_income_ratio",
        ]

    def train(self) -> dict:
        """Train the model pipeline and save artifacts. Returns training metadata."""
        logger.info("Loading data from %s", self.config.dataset_path)
        df = load_dataset(self.config.dataset_path)

        features = self._get_feature_columns()
        if not set(features).issubset(df.columns):
            missing = set(features) - set(df.columns)
            logger.error("Missing required training columns: %s", missing)
            raise ValueError(f"Missing required columns: {missing}")

        X = df[features]
        y = df["loan_approved"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )

        # Column transformer could be expanded later for categorical features
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, features),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", RandomForestClassifier(random_state=self.config.random_state))
            ]
        )

        # Simple grid for n_estimators and max_depth
        param_grid = {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [None, 10, 20],
        }

        logger.info("Starting GridSearchCV with cv=%s", self.config.cv_folds)
        gs = GridSearchCV(pipeline, param_grid, cv=self.config.cv_folds, n_jobs=-1, scoring="accuracy", verbose=1)
        gs.fit(X_train, y_train)

        best_pipeline = gs.best_estimator_
        logger.info("Best params: %s", gs.best_params_)

        train_score = best_pipeline.score(X_train, y_train)
        test_score = best_pipeline.score(X_test, y_test)
        cv_scores = cross_val_score(best_pipeline, X, y, cv=self.config.cv_folds, scoring="accuracy", n_jobs=-1)

        logger.info("Training complete. Train: %.3f Test: %.3f CV-mean: %.3f", train_score, test_score, cv_scores.mean())

        pipeline_path = os.path.join(self.config.model_dir, self.config.pipeline_filename)
        metadata_path = os.path.join(self.config.model_dir, self.config.metadata_filename)

        # Save pipeline and metadata
        joblib.dump(best_pipeline, pipeline_path)
        metadata = {
            "feature_columns": features,
            "pipeline_path": pipeline_path,
            "training_scores": {
                "train_score": float(train_score),
                "test_score": float(test_score),
                "cv_mean_score": float(cv_scores.mean())
            }
        }
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("Saved pipeline to %s and metadata to %s", pipeline_path, metadata_path)
        return metadata


if __name__ == "__main__":
    trainer = LoanModelTrainer()
    meta = trainer.train()
    print("Training metadata:", json.dumps(meta, indent=2))
