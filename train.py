"""
train.py
Training orchestrator — Siamese + LightGBM fallback, all horizons.
All .py files are flat at repo root; YAML configs are flat at root too
(or optionally in a configs/ subfolder — both are searched).
"""

import numpy as np
import logging
import time
import os
import yaml
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_yaml(filename: str) -> str:
    """
    Find a YAML config file. Search order:
      1. Flat at repo root          e.g. global_config.yaml
      2. In configs/ subfolder      e.g. configs/global_config.yaml
    Both _ROOT-relative and cwd-relative are tried for each location.
    """
    candidates = [
        os.path.join(_ROOT, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(_ROOT, "configs", filename),
        os.path.join(os.getcwd(), "configs", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find '{filename}'. Tried:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_global_config() -> dict:
    return load_config(_find_yaml("global_config.yaml"))


def load_module_config(module: str) -> dict:
    return load_config(_find_yaml(f"{module}_config.yaml"))


# ---------------------------------------------------------------------------
# Single-Horizon Training
# ---------------------------------------------------------------------------

def train_one_horizon(
    Xi_train, Xj_train, y_train,
    Xi_val,   Xj_val,   y_val,
    input_dim:           int,
    cfg:                 dict,
    horizon:             int,
    time_limit:          Optional[float] = None,
    force_lgbm:          bool = False,
) -> Tuple[object, str, dict]:
    """
    Train model for a single horizon.
    Falls back to LightGBM if Siamese exceeds time limit or fails.
    Returns: model, backend_name ("siamese" or "lgbm"), history dict
    """
    global_cfg  = load_global_config()
    siamese_cfg = global_cfg.get("siamese", {})
    lgbm_cfg    = global_cfg.get("lgbm", {})
    threshold   = time_limit or global_cfg.get("training", {}).get("time_threshold_seconds", 600)

    backend = "lgbm" if force_lgbm else "siamese"

    if backend == "siamese":
        from siamese_model import train_siamese
        start = time.time()
        try:
            model, history = train_siamese(
                Xi_train, Xj_train, y_train,
                Xi_val,   Xj_val,   y_val,
                input_dim=input_dim,
                hidden_dims=siamese_cfg.get("hidden_dims", [64, 32]),
                head_dims=siamese_cfg.get("head_dims", [32, 16]),
                dropout=siamese_cfg.get("dropout", 0.1),
                epochs=cfg.get("training", {}).get("epochs", siamese_cfg.get("epochs", 20)),
                batch_size=cfg.get("training", {}).get("batch_size", siamese_cfg.get("batch_size", 256)),
                lr=siamese_cfg.get("learning_rate", 0.001),
                early_stopping_patience=cfg.get("training", {}).get("early_stopping_patience", 5),
                time_limit_seconds=threshold,
            )
            elapsed = time.time() - start
            if elapsed > threshold:
                logger.warning(f"H={horizon}: Siamese exceeded {elapsed:.0f}s. Falling back to LightGBM.")
                backend = "lgbm"
            else:
                logger.info(f"H={horizon}: Siamese trained in {elapsed:.1f}s.")
                return model, "siamese", history
        except Exception as e:
            logger.error(f"H={horizon}: Siamese failed: {e}. Falling back to LightGBM.")
            backend = "lgbm"

    # LightGBM fallback
    from lgbm_model import LGBMRanker
    lgbm = LGBMRanker(
        n_estimators=lgbm_cfg.get("n_estimators", 200),
        num_leaves=lgbm_cfg.get("num_leaves", 31),
        learning_rate=lgbm_cfg.get("learning_rate", 0.05),
    )
    history = lgbm.fit(Xi_train, Xj_train, y_train, Xi_val, Xj_val, y_val)
    logger.info(f"H={horizon}: LightGBM trained.")
    return lgbm, "lgbm", history


# ---------------------------------------------------------------------------
# Multi-Horizon Training
# ---------------------------------------------------------------------------

def train_all_horizons(
    df,
    universe:    List[str],
    module_cfg:  dict,
    global_cfg:  dict,
    start_date:  Optional[str] = None,
    end_date:    Optional[str] = None,
    force_lgbm:  bool = False,
) -> Dict[int, Tuple[object, str]]:
    """Train models for H = 1, 3, 5 days. Returns {horizon: (model, backend)}."""
    from dataset import build_pairwise_dataset, temporal_split, date_range_split

    horizons    = global_cfg.get("holding_periods", [1, 3, 5])
    train_ratio = global_cfg.get("training", {}).get("train_ratio", 0.80)
    val_ratio   = global_cfg.get("training", {}).get("val_ratio",   0.10)
    models      = {}

    for h in horizons:
        logger.info(f"=== Pairwise dataset H={h} ===")
        Xi, Xj, y, dates = build_pairwise_dataset(df, universe, horizon=h)

        if start_date and end_date:
            splits = date_range_split(Xi, Xj, y, dates, start_date, end_date, train_ratio, val_ratio)
        else:
            splits = temporal_split(Xi, Xj, y, dates, train_ratio, val_ratio)

        Xi_tr, Xj_tr, y_tr, _ = splits["train"]
        Xi_vl, Xj_vl, y_vl, _ = splits["val"]

        if len(y_tr) == 0:
            logger.warning(f"H={h}: empty training set, skipping.")
            continue

        input_dim = Xi_tr.shape[1]
        logger.info(f"H={h}: Train={len(y_tr)}, Val={len(y_vl)}, Dim={input_dim}")

        model, backend, history = train_one_horizon(
            Xi_tr, Xj_tr, y_tr, Xi_vl, Xj_vl, y_vl,
            input_dim=input_dim, cfg=module_cfg,
            horizon=h, force_lgbm=force_lgbm,
        )
        models[h] = (model, backend)

    return models


# ---------------------------------------------------------------------------
# Save All Models
# ---------------------------------------------------------------------------

def save_all_models(models: Dict[int, Tuple[object, str]], module: str, base_path: str):
    from siamese_model import save_model
    from lgbm_model    import save_lgbm

    os.makedirs(base_path, exist_ok=True)
    for h, (model, backend) in models.items():
        if backend == "siamese":
            path = os.path.join(base_path, f"{module}_siamese_h{h}.pt")
            save_model(model, path, meta={"module": module, "horizon": h})
        else:
            path = os.path.join(base_path, f"{module}_lgbm_h{h}.pkl")
            save_lgbm(model, path)
        logger.info(f"Saved H={h} {backend} → {path}")
