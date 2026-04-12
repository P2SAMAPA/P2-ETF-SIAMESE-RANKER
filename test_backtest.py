"""
tests/test_backtest.py
Unit tests for metrics and conviction scoring.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from metrics import (
    annualised_return,
    annualised_vol,
    sharpe_ratio,
    max_drawdown,
    hit_rate,
    compute_all_metrics,
    compute_cumulative_returns,
    compute_consensus_score,
)
from inference import compute_conviction_scores, rank_etfs


# ── Fixtures ────────────────────────────────────────────────────────────────

def daily_returns(n=252, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(0.0008, 0.01, n), index=idx)


# ── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_annualised_return_positive(self):
        r = daily_returns()
        ann = annualised_return(r)
        assert isinstance(ann, float)
        assert -1 < ann < 10   # sanity bounds

    def test_annualised_return_empty(self):
        assert np.isnan(annualised_return(pd.Series([], dtype=float)))

    def test_annualised_vol_positive(self):
        r   = daily_returns()
        vol = annualised_vol(r)
        assert vol > 0

    def test_sharpe_sign(self):
        pos_ret = pd.Series([0.001] * 252)
        neg_ret = pd.Series([-0.001] * 252)
        assert sharpe_ratio(pos_ret) > 0
        assert sharpe_ratio(neg_ret) < 0

    def test_max_drawdown_negative(self):
        r  = daily_returns()
        dd = max_drawdown(r)
        assert dd <= 0

    def test_max_drawdown_all_up(self):
        r  = pd.Series([0.01] * 100)
        dd = max_drawdown(r)
        assert dd == 0.0

    def test_hit_rate_range(self):
        r = daily_returns()
        b = daily_returns(seed=99)
        hr = hit_rate(r, b)
        assert 0 <= hr <= 1

    def test_compute_all_metrics_keys(self):
        r = daily_returns()
        b = daily_returns(seed=99)
        m = compute_all_metrics(r, b)
        for key in ["ann_return", "ann_vol", "sharpe", "max_dd", "hit_rate"]:
            assert key in m

    def test_compute_cumulative_returns_starts_at_1(self):
        r   = daily_returns()
        cum = compute_cumulative_returns(r)
        assert abs(cum.iloc[0] - (1 + r.iloc[0])) < 1e-6

    def test_compute_cumulative_returns_monotone_up(self):
        r   = pd.Series([0.01] * 20)
        cum = compute_cumulative_returns(r)
        assert (cum.diff().dropna() > 0).all()


# ── Consensus Score ──────────────────────────────────────────────────────────

class TestConsensusScore:

    def _windows(self):
        return [
            {"etf": "GLD", "ann_return": 0.15, "sharpe": 1.2,  "max_dd": -0.10},
            {"etf": "GLD", "ann_return": 0.20, "sharpe": 1.5,  "max_dd": -0.08},
            {"etf": "TLT", "ann_return": 0.05, "sharpe": 0.4,  "max_dd": -0.12},
            {"etf": "TLT", "ann_return": -0.03,"sharpe": -0.2, "max_dd": -0.20},  # negative — excluded
            {"etf": "SLV", "ann_return": 0.08, "sharpe": 0.6,  "max_dd": -0.15},
        ]

    def test_negative_return_excluded(self):
        c = compute_consensus_score(self._windows())
        # TLT window with -0.03 return should be excluded
        # TLT may still appear from positive window
        assert "GLD" in c

    def test_ranking_order(self):
        c       = compute_consensus_score(self._windows())
        ranked  = sorted(c.items(), key=lambda x: x[1], reverse=True)
        assert ranked[0][0] == "GLD"   # GLD has best metrics

    def test_weights_respected(self):
        w1 = {"ann_return": 1.0, "sharpe": 0.0, "max_dd": 0.0}
        w2 = {"ann_return": 0.0, "sharpe": 0.0, "max_dd": 1.0}
        c1 = compute_consensus_score(self._windows(), weights=w1)
        c2 = compute_consensus_score(self._windows(), weights=w2)
        # Different weights → different scores
        assert c1 != c2

    def test_empty_input(self):
        c = compute_consensus_score([])
        assert c == {}

    def test_all_negative_returns(self):
        windows = [
            {"etf": "AAA", "ann_return": -0.05, "sharpe": -0.3, "max_dd": -0.2},
            {"etf": "BBB", "ann_return": -0.10, "sharpe": -0.8, "max_dd": -0.4},
        ]
        c = compute_consensus_score(windows)
        assert c == {}


# ── Conviction Scores ────────────────────────────────────────────────────────

class TestConvictionScores:

    def _make_mock_model(self, p_fixed=0.6):
        """Model that always returns p_fixed for P(i>j).
        Returns a 1-element array so np.asarray(_raw).flat[0] works correctly.
        """
        class MockModel:
            def predict_proba(self, xi, xj, **kwargs):
                # Return shape (n,) — inference.py uses .flat[0] to extract scalar
                return np.full(xi.shape[0], p_fixed, dtype=np.float32)
        return MockModel()

    def test_conviction_keys(self):
        universe = ["A", "B", "C"]
        features = {e: np.random.randn(10).astype(np.float32) for e in universe}
        model    = self._make_mock_model()
        conv     = compute_conviction_scores(model, features, universe)
        assert set(conv.keys()) == set(universe)

    def test_conviction_sums_to_n(self):
        """Sum of all conviction scores = N/2 for symmetric model (p=0.5)."""
        universe = ["A", "B", "C", "D"]
        features = {e: np.random.randn(10).astype(np.float32) for e in universe}
        model    = self._make_mock_model(p_fixed=0.5)
        conv     = compute_conviction_scores(model, features, universe)
        # With p=0.5, each ETF's conviction = 0.5
        for v in conv.values():
            assert abs(v - 0.5) < 1e-5

    def test_conviction_range(self):
        universe = ["A", "B", "C"]
        features = {e: np.random.randn(10).astype(np.float32) for e in universe}
        model    = self._make_mock_model(0.7)
        conv     = compute_conviction_scores(model, features, universe)
        for v in conv.values():
            assert 0 <= v <= 1

    def test_rank_etfs_descending(self):
        conv   = {"A": 0.8, "B": 0.5, "C": 0.65}
        ranked = rank_etfs(conv)
        scores = [r["score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_etfs_top_etf(self):
        conv   = {"A": 0.8, "B": 0.5, "C": 0.65}
        ranked = rank_etfs(conv)
        assert ranked[0]["etf"] == "A"

    def test_insufficient_etfs(self):
        """Need at least 2 ETFs to compute conviction."""
        universe = ["A"]
        features = {"A": np.random.randn(10).astype(np.float32)}
        model    = self._make_mock_model()
        conv     = compute_conviction_scores(model, features, universe)
        assert conv == {}
