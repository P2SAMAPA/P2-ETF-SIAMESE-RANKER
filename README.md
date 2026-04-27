# P2-ETF-SIAMESE-RANKER

**Pairwise Deep Ranking – Conviction‑Based ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-SIAMESE-RANKER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-SIAMESE-RANKER/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--siamese--ranker--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-siamese-ranker-results)

## Overview

`P2-ETF-SIAMESE-RANKER` uses a **Siamese Neural Network** to learn pairwise ETF rankings directly. For every pair of ETFs `(i, j)`, the model predicts the probability `P(i > j)` — that ETF `i` will outperform ETF `j`. These pairwise probabilities are averaged to produce a **conviction score** per ETF, which is used for final ranking. The model is trained on the full 2008‑2026 dataset with three output modes.

## Methodology

1. **Feature Engineering** – Lagged returns (1, 5, 21, 63 days) + macro features (VIX, DXY, T10Y2Y, TBILL_3M).
2. **Siamese Encoder** – Shared neural network (Dense 64→32 + ReLU) that embeds each ETF's features.
3. **Comparator Head** – Predicts `P(i > j)` from `[E_i, E_j, ΔE, |ΔE|]`.
4. **Conviction Score** – `Score_i = Σ_{j≠i} P(i > j) / (N-1)` — the average probability of outperforming every other ETF.
5. **Three Training Modes** – Daily, Global, Shrinking Windows Consensus.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
Dashboard
Three sub‑tabs per universe: Daily (504d), Global (2008‑YTD), Shrinking Consensus.

Hero card with conviction score (Strong/Moderate/Weak).

Top 3 & Full ETF ranking tables.
