"""
inference.py
Ranking inference: conviction scores, multi-horizon selection, output formatting.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional
import logging
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def compute_conviction_scores(
    model,
    feature_dict: Dict[str, np.ndarray],
    universe: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Score_i = Σ_{j≠i} P(i>j)
    Conviction_i = Score_i / (N-1)
    """
    valid = [e for e in universe if e in feature_dict]
    N = len(valid)
    if N < 2:
        return {}

    scores = {e: 0.0 for e in valid}
    for ei, ej in combinations(valid, 2):
        xi = feature_dict[ei].reshape(1, -1)
        xj = feature_dict[ej].reshape(1, -1)
        _raw = model.predict_proba(xi, xj, device=device)
        p_ij = float(np.asarray(_raw).flat[0])  # safely extract scalar from any array shape
        scores[ei] += p_ij
        scores[ej] += (1.0 - p_ij)

    return {e: scores[e] / (N - 1) for e in valid}


def rank_etfs(conviction: Dict[str, float]) -> List[Dict]:
    ranked = sorted(conviction.items(), key=lambda x: x[1], reverse=True)
    return [{"etf": e, "score": round(s, 4)} for e, s in ranked]


def infer_best_horizon(
    model_by_horizon: Dict[int, object],
    feature_dict: Dict[str, np.ndarray],
    universe: List[str],
    device: str = "cpu",
) -> Dict:
    """Run inference for H=1,3,5 and pick the horizon with the highest top-ETF conviction."""
    results = {}
    for h, model in model_by_horizon.items():
        conviction = compute_conviction_scores(model, feature_dict, universe, device)
        ranking    = rank_etfs(conviction)
        results[h] = {"ranking": ranking, "conviction": conviction}

    best_h = max(results.keys(), key=lambda h: results[h]["ranking"][0]["score"] if results[h]["ranking"] else 0)
    logger.info(f"Best horizon: {best_h}d  top conviction: {results[best_h]['ranking'][0]['score']:.4f}")

    return {
        "best_horizon":  best_h,
        "ranking":       results[best_h]["ranking"],
        "conviction":    results[best_h]["conviction"],
        "all_horizons":  results,
    }


def format_output(
    module:        str,
    ranking:       List[Dict],
    conviction:    Dict[str, float],
    best_horizon:  int,
    signal_date:   str,
    model_backend: str = "siamese",
    source:        str = "fixed_split",
) -> Dict:
    top    = ranking[0]
    second = ranking[1] if len(ranking) > 1 else None
    third  = ranking[2] if len(ranking) > 2 else None

    return {
        "module":            module,
        "signal_date":       signal_date,
        "generated_utc":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_backend":     model_backend,
        "best_horizon_days": best_horizon,
        "source":            source,
        "top_pick":          top["etf"],
        "top_conviction":    top["score"],
        "second":            {"etf": second["etf"], "score": second["score"]} if second else None,
        "third":             {"etf": third["etf"],  "score": third["score"]}  if third  else None,
        "ranking":           ranking,
        "conviction_scores": conviction,
    }


def save_output(output: Dict, path: str):
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Output saved → {path}")


def load_output(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
