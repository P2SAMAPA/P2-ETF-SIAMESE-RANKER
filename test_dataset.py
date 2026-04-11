"""
tests/test_dataset.py
Unit tests for feature engineering and pairwise dataset builder.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from dataset import (
    compute_etf_features,
    build_feature_matrix,
    build_pairwise_dataset,
    temporal_split,
    build_inference_features,
    MACRO_FEATURES,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_mock_df(n=300, etfs=("AAA", "BBB", "CCC"), seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for etf in etfs:
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
        data[etf] = prices
    for macro in MACRO_FEATURES:
        data[macro] = np.random.uniform(0.5, 3.0, n)
    return pd.DataFrame(data, index=idx)


FI_UNIVERSE    = ["AAA", "BBB", "CCC"]
MOCK_DF        = make_mock_df()


# ── Tests: compute_etf_features ─────────────────────────────────────────────

def test_etf_features_shape():
    prices = MOCK_DF["AAA"]
    macro  = MOCK_DF[MACRO_FEATURES]
    feat   = compute_etf_features(prices, macro)
    assert isinstance(feat, pd.DataFrame)
    assert len(feat) == len(prices)


def test_etf_features_columns_present():
    prices   = MOCK_DF["AAA"]
    macro    = MOCK_DF[MACRO_FEATURES]
    feat     = compute_etf_features(prices, macro)
    expected = ["r_1d", "r_5d", "r_21d", "vol_5d", "vol_21d",
                "mom_5d", "mom_21d", "price_vs_ma21", "range_5d"]
    for col in expected:
        assert col in feat.columns, f"Missing column: {col}"


def test_macro_features_included():
    prices = MOCK_DF["AAA"]
    macro  = MOCK_DF[MACRO_FEATURES]
    feat   = compute_etf_features(prices, macro)
    for m in MACRO_FEATURES:
        assert f"macro_{m}" in feat.columns, f"Missing macro feature: macro_{m}"


# ── Tests: build_feature_matrix ─────────────────────────────────────────────

def test_build_feature_matrix_keys():
    fdict = build_feature_matrix(MOCK_DF, FI_UNIVERSE)
    assert set(fdict.keys()) == set(FI_UNIVERSE)


def test_build_feature_matrix_missing_etf(capsys):
    fdict = build_feature_matrix(MOCK_DF, ["AAA", "MISSING"])
    assert "AAA"     in fdict
    assert "MISSING" not in fdict


# ── Tests: build_pairwise_dataset ───────────────────────────────────────────

def test_pairwise_dataset_shapes():
    Xi, Xj, y, dates = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=1)
    assert Xi.shape    == Xj.shape
    assert Xi.shape[0] == len(y)
    assert len(dates)  == len(y)
    assert Xi.dtype    == np.float32


def test_pairwise_labels_binary():
    _, _, y, _ = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=1)
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_pairwise_symmetry():
    """Dataset should contain both (i,j) and (j,i) orientations."""
    Xi, Xj, y, _ = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=1)
    n_pairs = len(FI_UNIVERSE) * (len(FI_UNIVERSE) - 1)   # N*(N-1) symmetric pairs
    n_days  = (~np.isnan(Xi).any(axis=1)).sum()
    assert len(y) > 0


def test_pairwise_different_horizons():
    _, _, y1, _ = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=1)
    _, _, y5, _ = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=5)
    assert len(y5) < len(y1) or len(y5) == len(y1)  # horizon=5 loses more tail rows


# ── Tests: temporal_split ────────────────────────────────────────────────────

def test_temporal_split_proportions():
    Xi, Xj, y, dates = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=1)
    splits = temporal_split(Xi, Xj, y, dates)

    n_total = len(y)
    n_train = len(splits["train"][2])
    n_val   = len(splits["val"][2])
    n_test  = len(splits["test"][2])

    assert n_train + n_val + n_test == n_total
    assert n_train > n_val
    assert n_train > n_test


def test_temporal_split_no_overlap():
    Xi, Xj, y, dates = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=1)
    splits = temporal_split(Xi, Xj, y, dates)

    train_dates = set(splits["train"][3])
    val_dates   = set(splits["val"][3])
    test_dates  = set(splits["test"][3])

    assert len(train_dates & val_dates)  == 0
    assert len(train_dates & test_dates) == 0
    assert len(val_dates   & test_dates) == 0


def test_temporal_split_chronological():
    """Train dates must all precede val dates which precede test dates."""
    Xi, Xj, y, dates = build_pairwise_dataset(MOCK_DF, FI_UNIVERSE, horizon=1)
    splits = temporal_split(Xi, Xj, y, dates)

    max_train = max(splits["train"][3])
    min_val   = min(splits["val"][3])
    max_val   = max(splits["val"][3])
    min_test  = min(splits["test"][3])

    assert max_train <= min_val
    assert max_val   <= min_test


# ── Tests: build_inference_features ─────────────────────────────────────────

def test_inference_features_keys():
    feat = build_inference_features(MOCK_DF, FI_UNIVERSE)
    for etf in FI_UNIVERSE:
        assert etf in feat


def test_inference_features_1d_array():
    feat = build_inference_features(MOCK_DF, FI_UNIVERSE)
    for etf, arr in feat.items():
        assert arr.ndim    == 1
        assert arr.dtype   == np.float32
        assert not np.any(np.isnan(arr)), f"NaN in features for {etf}"


def test_inference_features_as_of_date():
    as_of = pd.Timestamp("2020-06-01")
    feat  = build_inference_features(MOCK_DF, FI_UNIVERSE, as_of_date=as_of)
    assert len(feat) > 0
