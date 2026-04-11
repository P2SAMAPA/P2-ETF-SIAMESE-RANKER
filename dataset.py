"""
core/dataset.py
Feature engineering pipeline and pairwise dataset builder.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from itertools import combinations
from typing import List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_source_data(hf_token: Optional[str] = None) -> pd.DataFrame:
    """Load master ETF + macro dataset from HuggingFace."""
    token = hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(
        "P2SAMAPA/fi-etf-macro-signal-master-data",
        split="train",
        token=token,
    )
    df = ds.to_pandas()

    # Set date index
    date_col = "__index_level_0__"
    if date_col in df.columns:
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    logger.info(f"Loaded data: {df.shape[0]} rows, {df.index.min()} → {df.index.max()}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

MACRO_FEATURES = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]


def compute_etf_features(
    prices: pd.Series,
    macro: pd.DataFrame,
    return_windows: List[int] = [1, 5, 21],
    vol_windows: List[int] = [5, 21],
    momentum_windows: List[int] = [5, 21],
) -> pd.DataFrame:
    """
    Compute per-ETF feature vector at each date t.

    Features:
      r_1d, r_5d, r_21d         — log returns
      vol_5d, vol_21d           — rolling volatility
      momentum_5d, momentum_21d — price momentum (price / lagged price - 1)
      price_vs_ma_21            — price vs 21d MA
      volume_z                  — not available (placeholder 0)
      range_5d                  — rolling high-low range proxy (using return range)
      + macro features (VIX, DXY, etc.)
    """
    feat = pd.DataFrame(index=prices.index)

    # Log returns
    log_ret = np.log(prices / prices.shift(1))
    for w in return_windows:
        feat[f"r_{w}d"] = log_ret.rolling(w).sum()

    # Volatility
    for w in vol_windows:
        feat[f"vol_{w}d"] = log_ret.rolling(w).std()

    # Momentum
    for w in momentum_windows:
        feat[f"momentum_{w}d"] = prices / prices.shift(w) - 1

    # Price vs MA
    feat["price_vs_ma_21"] = prices / prices.rolling(21).mean() - 1

    # Rolling range (volatility proxy using 5d returns range)
    feat["range_5d"] = log_ret.rolling(5).max() - log_ret.rolling(5).min()

    # Macro features — forward-fill, then merge
    for col in MACRO_FEATURES:
        if col in macro.columns:
            macro_series = macro[col].ffill()
            feat[f"macro_{col}"] = macro_series.reindex(feat.index).ffill()

    return feat


def build_feature_matrix(
    df: pd.DataFrame,
    universe: List[str],
    return_windows: List[int] = [1, 5, 21],
    vol_windows: List[int] = [5, 21],
    momentum_windows: List[int] = [5, 21],
) -> dict:
    """
    Build feature matrices for all ETFs in the universe.

    Returns:
        dict: {etf: pd.DataFrame of features}
    """
    macro = df[MACRO_FEATURES]
    feature_dict = {}

    for etf in universe:
        if etf not in df.columns:
            logger.warning(f"ETF {etf} not found in dataset, skipping.")
            continue
        prices = df[etf].dropna()
        feat = compute_etf_features(
            prices, macro, return_windows, vol_windows, momentum_windows
        )
        feature_dict[etf] = feat
        logger.debug(f"Features built for {etf}: {feat.shape}")

    return feature_dict


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build pairwise training dataset.

    For each day t and each pair (i, j):
        Xi(t), Xj(t), y_ij = 1 if r_i(t+H) > r_j(t+H) else 0

    Args:
        df: Source dataframe (prices + macro)
        universe: List of ETF tickers
        horizon: Forward return horizon in days
        sample_pairs_per_day: If set, randomly sample this many pairs per day (CPU efficiency)

    Returns:
        Xi, Xj: feature arrays of shape (N, F)
        y: label array of shape (N,)
        dates: date array of shape (N,)
    """
    feature_dict = build_feature_matrix(
        df, universe, return_windows, vol_windows, momentum_windows
    )

    # Align all ETFs to common dates (drop NaN rows)
    valid_etfs = list(feature_dict.keys())
    all_features = {etf: feature_dict[etf] for etf in valid_etfs}

    # Common valid dates across all ETFs
    common_idx = all_features[valid_etfs[0]].dropna().index
    for etf in valid_etfs[1:]:
        common_idx = common_idx.intersection(all_features[etf].dropna().index)

    # Forward returns for labelling
    fwd_returns = {}
    for etf in valid_etfs:
        prices = df[etf].reindex(common_idx).ffill()
        fwd_returns[etf] = prices.shift(-horizon) / prices - 1

    pairs = list(combinations(valid_etfs, 2))
    all_pairs = pairs + [(j, i) for i, j in pairs]  # symmetric pairs

    Xi_list, Xj_list, y_list, date_list = [], [], [], []

    feature_cols = all_features[valid_etfs[0]].columns.tolist()

    for date in common_idx[:-horizon]:  # exclude last horizon days (no labels)
        if sample_pairs_per_day is not None:
            day_pairs = [all_pairs[k] for k in np.random.choice(
                len(all_pairs), min(sample_pairs_per_day, len(all_pairs)), replace=False
            )]
        else:
            day_pairs = all_pairs

        for etf_i, etf_j in day_pairs:
            xi = all_features[etf_i].loc[date, feature_cols].values
            xj = all_features[etf_j].loc[date, feature_cols].values

            if np.any(np.isnan(xi)) or np.any(np.isnan(xj)):
                continue

            ri = fwd_returns[etf_i].loc[date]
            rj = fwd_returns[etf_j].loc[date]

            if pd.isna(ri) or pd.isna(rj):
                continue

            y = 1.0 if ri > rj else 0.0

            Xi_list.append(xi)
            Xj_list.append(xj)
            y_list.append(y)
            date_list.append(date)

    Xi = np.array(Xi_list, dtype=np.float32)
    Xj = np.array(Xj_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates = np.array(date_list)

    logger.info(f"Pairwise dataset: {len(y)} pairs, horizon={horizon}d, ETFs={valid_etfs}")
    return Xi, Xj, y, dates


# ---------------------------------------------------------------------------
# Train / Val / Test Split
# ---------------------------------------------------------------------------

def temporal_split(
    Xi: np.ndarray,
    Xj: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict:
    """
    Temporal (chronological) 80/10/10 split.
    """
    unique_dates = np.sort(np.unique(dates))
    n = len(unique_dates)

    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_dates = set(unique_dates[:train_end_idx])
    val_dates = set(unique_dates[train_end_idx:val_end_idx])
    test_dates = set(unique_dates[val_end_idx:])

    def mask(date_set):
        return np.array([d in date_set for d in dates])

    train_mask = mask(train_dates)
    val_mask = mask(val_dates)
    test_mask = mask(test_dates)

    return {
        "train": (Xi[train_mask], Xj[train_mask], y[train_mask], dates[train_mask]),
        "val":   (Xi[val_mask],   Xj[val_mask],   y[val_mask],   dates[val_mask]),
        "test":  (Xi[test_mask],  Xj[test_mask],  y[test_mask],  dates[test_mask]),
        "date_splits": {
            "train_end": unique_dates[train_end_idx - 1],
            "val_end": unique_dates[val_end_idx - 1],
            "test_end": unique_dates[-1],
        }
    }


def date_range_split(
    Xi: np.ndarray,
    Xj: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    start_date: str,
    end_date: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict:
    """
    Split within a specific date range (used for shrinking window backtests).
    """
    mask = (dates >= np.datetime64(start_date)) & (dates <= np.datetime64(end_date))
    return temporal_split(
        Xi[mask], Xj[mask], y[mask], dates[mask], train_ratio, val_ratio
    )


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
    """
    Build latest feature vector for each ETF (for live inference).

    Returns:
        dict: {etf: np.ndarray of shape (F,)}
    """
    feature_dict = build_feature_matrix(
        df, universe, return_windows, vol_windows, momentum_windows
    )

    if as_of_date is None:
        as_of_date = df.index.max()

    result = {}
    for etf, feat_df in feature_dict.items():
        # Get last available row on or before as_of_date
        available = feat_df[feat_df.index <= as_of_date].dropna()
        if len(available) == 0:
            logger.warning(f"No valid features for {etf} as of {as_of_date}")
            continue
        result[etf] = available.iloc[-1].values.astype(np.float32)

    return result
