"""
lgbm_model.py
LightGBM pairwise ranker — CPU-friendly fallback, identical interface to Siamese.
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
    logger.warning("LightGBM not installed. pip install lightgbm")


class LGBMRanker:
    """Pairwise LightGBM ranker. Input: (Xi, Xj) → P(i > j)."""

    def __init__(self, n_estimators=200, num_leaves=31, learning_rate=0.05, n_jobs=-1):
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm is required. pip install lightgbm")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            objective="binary",
            metric="binary_logloss",
            n_jobs=n_jobs,
            verbose=-1,
        )
        self.is_fitted = False

    def _featurise(self, Xi, Xj):
        delta = Xi - Xj
        return np.concatenate([Xi, Xj, delta, np.abs(delta)], axis=1)

    def fit(self, Xi_tr, Xj_tr, y_tr, Xi_vl, Xj_vl, y_vl) -> dict:
        self.model.fit(
            self._featurise(Xi_tr, Xj_tr), y_tr,
            eval_set=[(self._featurise(Xi_vl, Xj_vl), y_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        self.is_fitted = True
        X_tr = self._featurise(Xi_tr, Xj_tr)
        X_vl = self._featurise(Xi_vl, Xj_vl)
        train_acc = np.mean((self.model.predict(X_tr) > 0.5) == y_tr)
        val_acc   = np.mean((self.model.predict(X_vl) > 0.5) == y_vl)
        logger.info(f"LGBM | Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}")
        return {"train_acc": train_acc, "val_acc": val_acc}

    def predict_proba(self, xi: np.ndarray, xj: np.ndarray, **kwargs) -> np.ndarray:
        if xi.ndim == 1:
            xi, xj = xi.reshape(1, -1), xj.reshape(1, -1)
        proba = self.model.predict_proba(self._featurise(xi, xj))[:, 1]
        return proba if proba.shape[0] > 1 else float(proba[0])


def save_lgbm(model: LGBMRanker, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"LGBM saved → {path}")


def load_lgbm(path: str) -> LGBMRanker:
    with open(path, "rb") as f:
        return pickle.load(f)
