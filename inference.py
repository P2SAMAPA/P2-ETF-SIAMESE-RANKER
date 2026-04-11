"""
core/inference.py
Ranking inference: generate conviction scores for all ETFs in a module.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional, Union
import logging
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conviction Score Aggregation
# ---------------------------------------------------------------------------

def compute_conviction_scores(
    model,
    feature_dict: Dict[str, np.ndarray],
    universe: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    For each ETF i, compute:
        Score_i = sum_{j != i} P(i > j)
        Conviction_i = Score_i / (N - 1)

    Args:
        model: SiameseRanker or LGBMRanker with .predict_proba(xi, xj)
        feature_dict: {etf: np.ndarray of features}
        universe: ordered list of ETF tickers
        device: torch device

    Returns:
        dict: {etf: conviction_score}
    """
    valid_etfs = [e for e in universe if e in feature_dict]
    N = len(valid_etfs)
    if N < 2:
        logger.warning("Need at least 2 ETFs to rank.")
        return {}

    scores = {etf: 0.0 for etf in valid_etfs}

    for etf_i, etf_j in combinations(valid_etfs, 2):
        xi = feature_dict[etf_i].reshape(1, -1)
        xj = feature_dict[etf_j].reshape(1, -1)

        p_ij = float(model.predict_proba(xi, xj, device=device))  # P(i > j)
        p_ji = 1.0 - p_ij                                          # P(j > i)

        scores[etf_i] += p_ij
        scores[etf_j] += p_ji

    # Normalise to [0, 1]
    conviction = {etf: scores[etf] / (N - 1) for etf in valid_etfs}
    return conviction


def rank_etfs(conviction: Dict[str, float]) -> List[Dict]:
    """Sort ETFs by conviction score descending."""
    ranked = sorted(conviction.items(), key=lambda x: x[1], reverse=True)
    return [{"etf": etf, "score": round(score, 4)} for etf, score in ranked]


# ---------------------------------------------------------------------------
# Multi-Horizon Inference
# ---------------------------------------------------------------------------

def infer_best_horizon(
    model_by_horizon: Dict[int, object],
    feature_dict: Dict[str, np.ndarray],
    universe: List[str],
    device: str = "cpu",
) -> Dict:
    """
    Run inference for each horizon and return the one with highest top-ETF conviction.

    Args:
        model_by_horizon: {horizon: model}
        feature_dict: {etf: features}
        universe: ETF list

    Returns:
        dict with best horizon, ranking, and all horizon results
    """
    results = {}
    for h, model in model_by_horizon.items():
        conviction = compute_conviction_scores(model, feature_dict, universe, device)
        ranking = rank_etfs(conviction)
        results[h] = {"ranking": ranking, "conviction": conviction}

    # Pick horizon with highest top-ETF conviction
    best_horizon = max(results.keys(), key=lambda h: results[h]["ranking"][0]["score"])
    logger.info(f"Best horizon: {best_horizon}d (top ETF conviction: {results[best_horizon]['ranking'][0]['score']:.4f})")

    return {
        "best_horizon": best_horizon,
        "ranking": results[best_horizon]["ranking"],
        "conviction": results[best_horizon]["conviction"],
        "all_horizons": results,
    }


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def format_output(
    module: str,
    ranking: List[Dict],
    conviction: Dict[str, float],
    best_horizon: int,
    signal_date: str,
    model_backend: str = "siamese",
    source: str = "fixed_split",
) -> Dict:
    """Format final output JSON for a module."""
    top = ranking[0]
    second = ranking[1] if len(ranking) > 1 else None
    third = ranking[2] if len(ranking) > 2 else None

    return {
        "module": module,
        "signal_date": signal_date,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_backend": model_backend,
        "best_horizon_days": best_horizon,
        "source": source,
        "top_pick": top["etf"],
        "top_conviction": top["score"],
        "second": {"etf": second["etf"], "score": second["score"]} if second else None,
        "third": {"etf": third["etf"], "score": third["score"]} if third else None,
        "ranking": ranking,
        "conviction_scores": conviction,
    }


def save_output(output: Dict, path: str):
    """Save output JSON to disk."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Output saved to {path}")


def load_output(path: str) -> Optional[Dict]:
    """Load output JSON from disk."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Output file not found: {path}")
        return None


# ---------------------------------------------------------------------------
# Historical Conviction Tracker (for Streamlit stacked chart)
# ---------------------------------------------------------------------------

def compute_historical_convictions(
    model,
    df: pd.DataFrame,
    universe: List[str],
    dates: pd.DatetimeIndex,
    device: str = "cpu",
    stride: int = 5,  # compute every N days for efficiency
) -> pd.DataFrame:
    """
    Compute conviction scores over time for all ETFs (for stacked chart).

    Returns:
        DataFrame with columns = ETF tickers, index = dates
    """
    from core.dataset import build_feature_matrix

    feature_dict_all = build_feature_matrix(df, universe)
    results = []

    for date in dates[::stride]:
        day_features = {}
        for etf in universe:
            if etf not in feature_dict_all:
                continue
            row = feature_dict_all[etf]
            available = row[row.index <= date].dropna()
            if len(available) == 0:
                continue
            day_features[etf] = available.iloc[-1].values.astype(np.float32)

        if len(day_features) < 2:
            continue

        conviction = compute_conviction_scores(model, day_features, universe, device)
        conviction["date"] = date
        results.append(conviction)

    if not results:
        return pd.DataFrame()

    conv_df = pd.DataFrame(results).set_index("date")
    return conv_df
