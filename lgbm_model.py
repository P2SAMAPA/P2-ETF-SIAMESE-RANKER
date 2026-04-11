"""
core/lgbm_model.py
LightGBM pairwise ranker — CPU-friendly fallback with identical interface.
"""

import numpy as np
import pickle
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Run: pip install lightgbm")


# ---------------------------------------------------------------------------
# LightGBM Ranker (pairwise, identical interface to Siamese)
# ---------------------------------------------------------------------------

class LGBMRanker:
    """
    LightGBM-based pairwise ranker.
    Input: (Xi, Xj) concatenated → [Xi | Xj | Xi-Xj | |Xi-Xj|]
    Output: P(i > j) ∈ [0, 1]
    """

    def __init__(
        self,
        n_estimators: int = 200,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_jobs: int = -1,
    ):
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm is required. pip install lightgbm")

        self.params = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "objective": "binary",  # BCE on pairwise labels
            "metric": "binary_logloss",
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.is_fitted = False

    def _make_features(self, Xi: np.ndarray, Xj: np.ndarray) -> np.ndarray:
        """Concat [Xi, Xj, Xi-Xj, |Xi-Xj|] — mirrors Siamese comparator."""
        delta = Xi - Xj
        return np.concatenate([Xi, Xj, delta, np.abs(delta)], axis=1)

    def fit(
        self,
        Xi_train: np.ndarray,
        Xj_train: np.ndarray,
        y_train: np.ndarray,
        Xi_val: np.ndarray,
        Xj_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        X_train = self._make_features(Xi_train, Xj_train)
        X_val = self._make_features(Xi_val, Xj_val)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        self.is_fitted = True

        train_acc = np.mean((self.model.predict(X_train) > 0.5) == y_train)
        val_acc = np.mean((self.model.predict(X_val) > 0.5) == y_val)
        logger.info(f"LGBM trained | Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}")

        return {"train_acc": train_acc, "val_acc": val_acc}

    def predict_proba(self, xi: np.ndarray, xj: np.ndarray, **kwargs) -> np.ndarray:
        """Predict P(i > j). xi/xj can be 1D (single pair) or 2D (batch)."""
        if xi.ndim == 1:
            xi = xi.reshape(1, -1)
            xj = xj.reshape(1, -1)
        X = self._make_features(xi, xj)
        proba = self.model.predict_proba(X)[:, 1]
        return proba if proba.shape[0] > 1 else float(proba[0])


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_lgbm(model: LGBMRanker, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"LGBM model saved to {path}")


def load_lgbm(path: str) -> LGBMRanker:
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"LGBM model loaded from {path}")
    return model
