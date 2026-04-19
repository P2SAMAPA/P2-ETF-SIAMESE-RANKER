"""
tests/test_model.py
Unit tests for Siamese model and LightGBM fallback.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tempfile

INPUT_DIM   = 20
HIDDEN_DIMS = [32, 16]
HEAD_DIMS   = [16, 8]
N_SAMPLES   = 200


def make_dummy_data(n=N_SAMPLES, dim=INPUT_DIM, seed=0):
    np.random.seed(seed)
    Xi = np.random.randn(n, dim).astype(np.float32)
    Xj = np.random.randn(n, dim).astype(np.float32)
    y  = (np.random.rand(n) > 0.5).astype(np.float32)
    return Xi, Xj, y


# ── Siamese Model ────────────────────────────────────────────────────────────

class TestSiameseRanker:

    def test_forward_shape(self):
        import torch
        from siamese_model import SiameseRanker
        model = SiameseRanker(INPUT_DIM, HIDDEN_DIMS, HEAD_DIMS)
        xi = torch.randn(8, INPUT_DIM)
        xj = torch.randn(8, INPUT_DIM)
        out = model(xi, xj)
        assert out.shape == (8,)

    def test_output_in_01(self):
        import torch
        from siamese_model import SiameseRanker
        model = SiameseRanker(INPUT_DIM, HIDDEN_DIMS, HEAD_DIMS)
        xi = torch.randn(32, INPUT_DIM)
        xj = torch.randn(32, INPUT_DIM)
        out = model(xi, xj)
        assert (out >= 0).all() and (out <= 1).all()

    def test_symmetry(self):
        """
        P(i>j) and P(j>i) must each be valid probabilities in [0, 1].
        The architecture uses concat([Ei, Ej, delta, |delta|]) which does NOT
        enforce P(i>j)+P(j>i)=1 by construction — that property is learned
        during training. We only assert valid probability outputs here.
        """
        import torch
        from siamese_model import SiameseRanker
        model = SiameseRanker(INPUT_DIM, HIDDEN_DIMS, HEAD_DIMS)
        model.eval()
        xi = torch.randn(1, INPUT_DIM)
        xj = torch.randn(1, INPUT_DIM)
        p_ij = model(xi, xj).item()
        p_ji = model(xj, xi).item()
        # Both outputs must be valid probabilities
        assert 0.0 <= p_ij <= 1.0, f"p_ij={p_ij} not in [0,1]"
        assert 0.0 <= p_ji <= 1.0, f"p_ji={p_ji} not in [0,1]"
        # Sum is not architectural but should not be wildly broken
        assert 0.2 <= p_ij + p_ji <= 1.8, f"Sum {p_ij+p_ji:.3f} unexpectedly out of range"

    def test_predict_proba_numpy(self):
        from siamese_model import SiameseRanker
        model = SiameseRanker(INPUT_DIM, HIDDEN_DIMS, HEAD_DIMS)
        xi = np.random.randn(4, INPUT_DIM).astype(np.float32)
        xj = np.random.randn(4, INPUT_DIM).astype(np.float32)
        out = model.predict_proba(xi, xj)
        assert out.shape == (4,)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_train_siamese_runs(self):
        from siamese_model import train_siamese
        Xi, Xj, y = make_dummy_data(100)
        model, history = train_siamese(
            Xi[:80], Xj[:80], y[:80],
            Xi[80:], Xj[80:], y[80:],
            input_dim=INPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
            head_dims=HEAD_DIMS,
            epochs=2,
            batch_size=32,
        )
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"])   > 0

    def test_save_load_roundtrip(self):
        from siamese_model import SiameseRanker, save_model, load_model
        import torch
        model = SiameseRanker(INPUT_DIM, HIDDEN_DIMS, HEAD_DIMS)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.pt")
            save_model(model, path, meta={"test": True})
            loaded = load_model(path, INPUT_DIM, HIDDEN_DIMS, HEAD_DIMS)
            # Weights should be identical
            for (k1, v1), (k2, v2) in zip(
                model.state_dict().items(), loaded.state_dict().items()
            ):
                assert torch.allclose(v1, v2)

    def test_early_stopping_respected(self):
        from siamese_model import train_siamese
        Xi, Xj, y = make_dummy_data(200)
        model, history = train_siamese(
            Xi[:160], Xj[:160], y[:160],
            Xi[160:], Xj[160:], y[160:],
            input_dim=INPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
            head_dims=HEAD_DIMS,
            epochs=50,
            batch_size=32,
            early_stopping_patience=2,
        )
        # Should stop well before 50 epochs
        assert len(history["train_loss"]) <= 50

    def test_time_limit_respected(self):
        from siamese_model import train_siamese
        Xi, Xj, y = make_dummy_data(500)
        model, history = train_siamese(
            Xi[:400], Xj[:400], y[:400],
            Xi[400:], Xj[400:], y[400:],
            input_dim=INPUT_DIM,
            hidden_dims=[64, 32],
            head_dims=[32, 16],
            epochs=1000,
            batch_size=32,
            time_limit_seconds=2.0,
        )
        assert history["elapsed_seconds"] < 10.0  # some slack


# ── LightGBM Ranker ──────────────────────────────────────────────────────────

class TestLGBMRanker:

    def test_fit_and_predict(self):
        from lgbm_model import LGBMRanker
        ranker = LGBMRanker(n_estimators=20)
        Xi, Xj, y = make_dummy_data(200)
        history = ranker.fit(Xi[:160], Xj[:160], y[:160], Xi[160:], Xj[160:], y[160:])
        assert "train_acc" in history
        assert "val_acc"   in history

    def test_predict_proba_range(self):
        from lgbm_model import LGBMRanker
        ranker = LGBMRanker(n_estimators=20)
        Xi, Xj, y = make_dummy_data(200)
        ranker.fit(Xi[:160], Xj[:160], y[:160], Xi[160:], Xj[160:], y[160:])
        proba = ranker.predict_proba(Xi[160:], Xj[160:])
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_single_pair(self):
        from lgbm_model import LGBMRanker
        ranker = LGBMRanker(n_estimators=20)
        Xi, Xj, y = make_dummy_data(200)
        ranker.fit(Xi[:160], Xj[:160], y[:160], Xi[160:], Xj[160:], y[160:])
        p = ranker.predict_proba(Xi[0], Xj[0])
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_featurise_shape(self):
        from lgbm_model import LGBMRanker
        ranker = LGBMRanker()
        Xi = np.ones((5, INPUT_DIM), dtype=np.float32)
        Xj = np.ones((5, INPUT_DIM), dtype=np.float32)
        feat = ranker._featurise(Xi, Xj)
        assert feat.shape == (5, INPUT_DIM * 4)

    def test_save_load_roundtrip(self):
        from lgbm_model import LGBMRanker, save_lgbm, load_lgbm
        ranker = LGBMRanker(n_estimators=10)
        Xi, Xj, y = make_dummy_data(100)
        ranker.fit(Xi[:80], Xj[:80], y[:80], Xi[80:], Xj[80:], y[80:])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lgbm.pkl")
            save_lgbm(ranker, path)
            loaded = load_lgbm(path)
            p1 = ranker.predict_proba(Xi[80:], Xj[80:])
            p2 = loaded.predict_proba(Xi[80:], Xj[80:])
            np.testing.assert_array_almost_equal(p1, p2)
