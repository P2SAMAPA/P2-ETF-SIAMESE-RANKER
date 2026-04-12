"""
dataset.py
Feature engineering pipeline and pairwise dataset builder.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

MACRO_FEATURES = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_source_data(hf_token: Optional[str] = None) -> pd.DataFrame:
    """Load master ETF + macro dataset from HuggingFace."""
    from datasets import load_dataset
    token = hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(
        "P2SAMAPA/fi-etf-macro-signal-master-data",
        split="train",
        token=token,
    )
    df = ds.to_pandas()
    date_col = "__index_level_0__"
    if date_col in df.columns:
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.info(f"Loaded: {df.shape[0]} rows  {df.index.min().date()} → {df.index.max().date()}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_etf_features(
    prices: pd.Series,
    macro: pd.DataFrame,
    return_windows: List[int] = [1, 5, 21],
    vol_windows: List[int] = [5, 21],
    momentum_windows: List[int] = [5, 21],
) -> pd.DataFrame:
    """Per-ETF feature vector at each date t."""
    feat = pd.DataFrame(index=prices.index)
    log_ret = np.log(prices / prices.shift(1))

    for w in return_windows:
        feat[f"r_{w}d"] = log_ret.rolling(w).sum()
    for w in vol_windows:
        feat[f"vol_{w}d"] = log_ret.rolling(w).std()
    for w in momentum_windows:
        feat[f"mom_{w}d"] = prices / prices.shift(w) - 1

    feat["price_vs_ma21"] = prices / prices.rolling(21).mean() - 1
    feat["range_5d"]      = log_ret.rolling(5).max() - log_ret.rolling(5).min()

    for col in MACRO_FEATURES:
        if col in macro.columns:
            feat[f"macro_{col}"] = macro[col].ffill().reindex(feat.index).ffill()

    return feat


def build_feature_matrix(
    df: pd.DataFrame,
    universe: List[str],
    return_windows: List[int] = [1, 5, 21],
    vol_windows: List[int] = [5, 21],
    momentum_windows: List[int] = [5, 21],
) -> dict:
    """Build {etf: feature_df} for all ETFs in universe."""
    macro = df[MACRO_FEATURES]
    out = {}
    for etf in universe:
        if etf not in df.columns:
            logger.warning(f"{etf} not in dataset — skipping.")
            continue
        prices = df[etf].dropna()
        out[etf] = compute_etf_features(prices, macro, return_windows, vol_windows, momentum_windows)
    return out


# ---------------------------------------------------------------------------
# Pairwise Dataset Builder
# ---------------------------------------------------------------------------

def build_pairwise_dataset(
    df: pd.DataFrame,
    universe: List[str],
    horizon: int = 1,
    return_windows: List[int] = [1, 5, 21],
    vol_windows: List[int] = [5, 21],
    momentum_windows: List[int] = [5, 21],
    sample_pairs_per_day: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pairwise training data.
    For each day t and each pair (i,j):
        y_ij = 1 if r_i(t+H) > r_j(t+H) else 0
    """
    feature_dict = build_feature_matrix(df, universe, return_windows, vol_windows, momentum_windows)
    valid_etfs   = list(feature_dict.keys())

    # Common valid dates
    common_idx = feature_dict[valid_etfs[0]].dropna().index
    for etf in valid_etfs[1:]:
        common_idx = common_idx.intersection(feature_dict[etf].dropna().index)

    # Forward returns
    fwd = {}
    for etf in valid_etfs:
        p = df[etf].reindex(common_idx).ffill()
        fwd[etf] = p.shift(-horizon) / p - 1

    pairs     = list(combinations(valid_etfs, 2))
    all_pairs = pairs + [(j, i) for i, j in pairs]
    feat_cols = feature_dict[valid_etfs[0]].columns.tolist()

    Xi_list, Xj_list, y_list, date_list = [], [], [], []

    for date in common_idx[:-horizon]:
        day_pairs = all_pairs
        if sample_pairs_per_day is not None:
            idx = np.random.choice(len(all_pairs), min(sample_pairs_per_day, len(all_pairs)), replace=False)
            day_pairs = [all_pairs[k] for k in idx]

        for ei, ej in day_pairs:
            xi = feature_dict[ei].loc[date, feat_cols].values
            xj = feature_dict[ej].loc[date, feat_cols].values
            if np.any(np.isnan(xi)) or np.any(np.isnan(xj)):
                continue
            ri, rj = fwd[ei].loc[date], fwd[ej].loc[date]
            if pd.isna(ri) or pd.isna(rj):
                continue
            Xi_list.append(xi)
            Xj_list.append(xj)
            y_list.append(1.0 if ri > rj else 0.0)
            date_list.append(date)

    Xi    = np.array(Xi_list,  dtype=np.float32)
    Xj    = np.array(Xj_list,  dtype=np.float32)
    y     = np.array(y_list,   dtype=np.float32)
    dates = np.array(date_list)
    logger.info(f"Pairwise dataset: {len(y)} pairs, H={horizon}d, ETFs={valid_etfs}")
    return Xi, Xj, y, dates


# ---------------------------------------------------------------------------
# Temporal Splits
# ---------------------------------------------------------------------------

def temporal_split(
    Xi, Xj, y, dates,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
) -> dict:
    """Chronological 80/10/10 split."""
    unique = np.sort(np.unique(dates))
    n = len(unique)
    tr_end  = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_set = set(unique[:tr_end])
    val_set   = set(unique[tr_end:val_end])
    test_set  = set(unique[val_end:])

    def mask(s):
        return np.array([d in s for d in dates])

    return {
        "train": (Xi[mask(train_set)], Xj[mask(train_set)], y[mask(train_set)], dates[mask(train_set)]),
        "val":   (Xi[mask(val_set)],   Xj[mask(val_set)],   y[mask(val_set)],   dates[mask(val_set)]),
        "test":  (Xi[mask(test_set)],  Xj[mask(test_set)],  y[mask(test_set)],  dates[mask(test_set)]),
        "date_splits": {
            "train_end": unique[tr_end - 1],
            "val_end":   unique[val_end - 1],
            "test_end":  unique[-1],
        },
    }


def date_range_split(
    Xi, Xj, y, dates,
    start_date: str, end_date: str,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
) -> dict:
    """Split within a specific date range (used for shrinking window)."""
    # Convert to pd.Timestamp for safe comparison regardless of date type
    ts_start = pd.Timestamp(start_date)
    ts_end   = pd.Timestamp(end_date)
    mask = np.array([pd.Timestamp(d) >= ts_start and pd.Timestamp(d) <= ts_end for d in dates])
    return temporal_split(Xi[mask], Xj[mask], y[mask], dates[mask], train_ratio, val_ratio)


# ---------------------------------------------------------------------------
# Inference Feature Builder
# ---------------------------------------------------------------------------

def build_inference_features(
    df: pd.DataFrame,
    universe: List[str],
    as_of_date: Optional[pd.Timestamp] = None,
    return_windows: List[int] = [1, 5, 21],
    vol_windows: List[int] = [5, 21],
    momentum_windows: List[int] = [5, 21],
) -> dict:
    """Build latest feature vector for each ETF (for live inference)."""
    feature_dict = build_feature_matrix(df, universe, return_windows, vol_windows, momentum_windows)
    if as_of_date is None:
        as_of_date = df.index.max()
    result = {}
    for etf, feat_df in feature_dict.items():
        available = feat_df[feat_df.index <= as_of_date].dropna()
        if len(available) == 0:
            logger.warning(f"No valid features for {etf} as of {as_of_date}")
            continue
        result[etf] = available.iloc[-1].values.astype(np.float32)
    return result
