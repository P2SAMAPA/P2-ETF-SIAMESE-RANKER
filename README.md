# P2-ETF-SIAMESE-RANKER

**Cross-Sectional Siamese Ranking Engine** — Dual-module relative value ranker for ETFs.

> Learns *who beats whom* rather than forecasting absolute returns.

---

## Architecture

```
f(ETF_i, ETF_j, X_t) → P(i outperforms j)
```

Two independent modules share the same Siamese core:

| Module | Universe | Benchmark |
|--------|----------|-----------|
| FI / Commodities | TLT, LQD, HYG, VNQ, GLD, SLV, VCIT | AGG |
| Equity Sectors | QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XME, GDX, IWM, XLB, XLRE | SPY |

---

## Repo Structure

```
P2-ETF-SIAMESE-RANKER/
├── core/
│   ├── dataset.py            # Feature engineering + pairwise builder
│   ├── siamese_model.py      # Siamese neural network (PyTorch)
│   ├── lgbm_model.py         # LightGBM ranker fallback
│   ├── train.py              # Training loop (80/10/10 split)
│   ├── inference.py          # Ranking inference + conviction scores
│   ├── backtest.py           # Backtest engine (fixed split + shrinking window)
│   ├── metrics.py            # Ann return, Sharpe, Max DD, Hit rate
│   └── hf_storage.py         # HuggingFace dataset I/O
├── fi_module/
│   ├── config.yaml           # FI universe + hyperparameters
│   ├── train.py              # FI training entrypoint
│   ├── predict.py            # FI inference entrypoint
│   └── output.json           # Latest FI ranking output
├── equity_module/
│   ├── config.yaml           # Equity universe + hyperparameters
│   ├── train.py              # Equity training entrypoint
│   ├── predict.py            # Equity inference entrypoint
│   └── output.json           # Latest equity ranking output
├── configs/
│   └── global_config.yaml    # Global settings (HF repo, paths, cron schedule)
├── scripts/
│   ├── run_all.sh            # Full pipeline: data → train → predict → push
│   ├── run_fi.sh             # FI module only
│   ├── run_equity.sh         # Equity module only
│   └── cron_setup.sh         # Sets up 22:00 cron job
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_backtest.py
├── notebooks/
│   └── exploration.ipynb     # EDA notebook
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/P2SAMAPA/P2-ETF-SIAMESE-RANKER.git
cd P2-ETF-SIAMESE-RANKER
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add your HF_TOKEN to .env

# 3. Run full pipeline (both modules)
bash scripts/run_all.sh

# 4. Run a single module
bash scripts/run_fi.sh
bash scripts/run_equity.sh

# 5. Launch Streamlit dashboard
streamlit run app.py
```

---

## Pipeline Flow

```
22:00  Pull latest data from HuggingFace source dataset
22:10  Build features (returns, vol, momentum, macro)
22:15  Train FI Siamese model   → fallback to LightGBM if > time threshold
22:25  Predict FI ranking       → save output.json
22:30  Train Equity model       → fallback to LightGBM if > time threshold
22:45  Predict Equity ranking   → save output.json
22:50  Push results to HF results dataset
```

---

## Model: Siamese Network

```
INPUT: (X_i, X_j)  ← per-ETF feature vectors
         ↓
SHARED ENCODER (same weights for both)
  Dense(64) + ReLU
  Dense(32) + ReLU
  → Embedding E_i, E_j
         ↓
COMPARATOR
  ΔE = E_i - E_j
  concat([E_i, E_j, ΔE, |ΔE|])
         ↓
HEAD
  Dense(32) + ReLU → Dense(16) + ReLU → Dense(1) + Sigmoid
         ↓
OUTPUT: P(i > j)  ∈ [0, 1]
```

**Fallback:** LightGBM RankNet — wired from day 1, same interface.

---

## Backtests

Two backtest modes per module:

**1. Fixed Split (80/10/10 temporal)**
- Train: first 80% of dates
- Val: next 10%
- Test: last 10%

**2. Shrinking Window**
- Window 1: 2008→2026 YTD (full)
- Window 2: 2009→2026 YTD
- ...
- Window 17: 2024→2026 YTD
- Each window: train on data, OOS = last year of window
- Consensus ETF = weighted avg score across windows (excluding negative-return windows):
  - Ann Return: 60%
  - Sharpe Ratio: 20%
  - Max DD (negated): 20%

---

## Holding Period Optimisation

Model trains on H = 1, 3, 5 days simultaneously. At inference, the H yielding the highest return is selected per module.

---

## HuggingFace

- **Source data:** `P2SAMAPA/fi-etf-macro-signal-master-data`
- **Results:** `P2SAMAPA/p2-etf-siamese-ranker-results`

Stored artifacts:
- `rankings/` — daily ranking JSONs
- `backtest_metrics/` — full + shrinking window metrics
- `model_weights/` — trained model state dicts
- `features/` — precomputed feature parquets
- `signal_history/` — date, pick, conviction, actual return

---

## Streamlit Dashboard

Two tabs: **FI / Alts** | **Equity Sectors**

Each tab shows:
- Hero card: top ETF pick, conviction %, signal date, 2nd/3rd ranked ETFs
- Fixed Split metrics: Ann Return, Ann Vol, Sharpe, Max DD, Hit Rate + cumulative return chart vs benchmark
- Shrinking Window metrics: same + active window info
- Stacked conviction score chart (all ETFs over backtest period)
- Signal history table: Date | Pick | Conviction | Actual Return | Hit

---

## Environment Variables

```
HF_TOKEN=your_huggingface_token
HF_SOURCE_DATASET=P2SAMAPA/fi-etf-macro-signal-master-data
HF_RESULTS_DATASET=P2SAMAPA/p2-etf-siamese-ranker-results
MODEL_BACKEND=siamese   # or lgbm
TRAINING_TIME_THRESHOLD=600   # seconds; if exceeded, fall back to LightGBM
```
