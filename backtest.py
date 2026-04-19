"""
backtest.py
Backtest engine — Fixed Split (80/10/10) and Shrinking Window modes.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy Return Series
# ---------------------------------------------------------------------------

def build_strategy_returns(
    df: pd.DataFrame,
    universe: List[str],
    model_by_horizon: Dict[int, object],
    test_dates: pd.DatetimeIndex,
    benchmark: str,
    device: str = "cpu",
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Each test day: rank ETFs → pick top → record next-day return.
    Returns strategy returns, benchmark returns, picks DataFrame.
    """
    from dataset  import build_inference_features
    from inference import compute_conviction_scores, rank_etfs

    price_cols  = [e for e in universe + [benchmark] if e in df.columns]
    daily_ret   = df[price_cols].ffill().pct_change()

    strat_rows, bench_rows, pick_rows = [], [], []

    for date in test_dates:
        if date not in daily_ret.index:
            continue

        feat = build_inference_features(df, universe, as_of_date=date)
        if len(feat) < 2:
            continue

        best_score, best_h, best_ranking = -1, 1, None
        for h, model in model_by_horizon.items():
            conv    = compute_conviction_scores(model, feat, universe, device)
            ranking = rank_etfs(conv)
            if ranking and ranking[0]["score"] > best_score:
                best_score   = ranking[0]["score"]
                best_h       = h
                best_ranking = ranking

        if best_ranking is None:
            continue

        top_etf    = best_ranking[0]["etf"]
        conviction = best_ranking[0]["score"]

        future = daily_ret.index[daily_ret.index > date]
        if len(future) == 0:
            continue
        nxt = future[0]

        etf_ret   = daily_ret.loc[nxt, top_etf]   if top_etf   in daily_ret.columns else np.nan
        bench_ret = daily_ret.loc[nxt, benchmark]  if benchmark in daily_ret.columns else np.nan

        strat_rows.append({"date": nxt, "return": etf_ret})
        bench_rows.append({"date": nxt, "return": bench_ret})
        pick_rows.append({
            "date":          str(date.date()),
            "pick":          top_etf,
            "conviction":    round(conviction, 4),
            "horizon":       best_h,
            "actual_return": round(float(etf_ret), 6) if not np.isnan(etf_ret) else None,
        })

    strat = pd.Series(
        [r["return"] for r in strat_rows],
        index=pd.DatetimeIndex([r["date"] for r in strat_rows]),
        name="strategy",
    ).dropna()

    bench = pd.Series(
        [r["return"] for r in bench_rows],
        index=pd.DatetimeIndex([r["date"] for r in bench_rows]),
        name="benchmark",
    ).dropna()

    return strat, bench, pd.DataFrame(pick_rows)


# ---------------------------------------------------------------------------
# Fixed Split Backtest
# ---------------------------------------------------------------------------

def run_fixed_split_backtest(
    df: pd.DataFrame,
    universe: List[str],
    models: Dict[int, Tuple[object, str]],
    benchmark: str,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    device:      str   = "cpu",
) -> Dict:
    from metrics import compute_all_metrics, compute_cumulative_returns

    valid_dates     = df[[e for e in universe if e in df.columns]].dropna(how="all").index
    n               = len(valid_dates)
    test_start_idx  = int(n * (train_ratio + val_ratio))
    test_dates      = valid_dates[test_start_idx:]

    model_by_horizon = {h: m for h, (m, _) in models.items()}
    logger.info(f"Fixed split backtest | {test_dates[0].date()} → {test_dates[-1].date()}")

    strat, bench, picks = build_strategy_returns(
        df, universe, model_by_horizon, test_dates, benchmark, device
    )

    cum_strat = compute_cumulative_returns(strat)
    cum_bench = compute_cumulative_returns(bench)

    return {
        "mode":       "fixed_split",
        "test_start": str(test_dates[0].date()),
        "test_end":   str(test_dates[-1].date()),
        "metrics":             compute_all_metrics(strat, bench),
        "benchmark_metrics":   compute_all_metrics(bench),
        "cumulative_returns": {
            "strategy":  {str(k.date()): v for k, v in cum_strat.items()},
            "benchmark": {str(k.date()): v for k, v in cum_bench.items()},
        },
        "picks":   picks.to_dict(orient="records") if len(picks) > 0 else [],
        "n_picks": len(picks),
    }


# ---------------------------------------------------------------------------
# Shrinking Window Backtest
# ---------------------------------------------------------------------------

def run_shrinking_window_backtest(
    df: pd.DataFrame,
    universe: List[str],
    module_cfg: dict,
    global_cfg: dict,
    benchmark: str,
    device:       str  = "cpu",
    force_lgbm:   bool = False,
    best_horizon: int  = 1,
) -> Dict:
    """
    Shrinking window backtest.

    Speed optimisations vs naive approach:
      - Always uses LightGBM (fast CPU ranker) regardless of force_lgbm flag
      - Only trains on the single best horizon (determined by fixed-split run)
      - Prebuilds the full pairwise dataset once per horizon and slices by date
    """
    from train   import train_one_horizon, load_global_config
    from dataset import build_pairwise_dataset, date_range_split
    from metrics import compute_all_metrics, compute_cumulative_returns, compute_consensus_score

    bt_cfg      = global_cfg.get("backtest", {})
    start_years = range(
        bt_cfg.get("shrinking_start_year", 2008),
        bt_cfg.get("shrinking_end_year",   2024) + 1,
    )
    end_date   = df.index.max().strftime("%Y-%m-%d")
    train_ratio = global_cfg.get("training", {}).get("train_ratio", 0.80)
    val_ratio   = global_cfg.get("training", {}).get("val_ratio",   0.10)

    # ── Pre-build full pairwise dataset for best_horizon once ────────────────
    logger.info(f"Shrinking window: pre-building pairwise dataset H={best_horizon}...")
    Xi_all, Xj_all, y_all, dates_all = build_pairwise_dataset(
        df, universe, horizon=best_horizon
    )
    logger.info(f"Full pairwise dataset: {len(y_all)} pairs")

    window_results, consensus_inputs = [], []

    for sy in start_years:
        win_start = f"{sy}-01-01"
        window_df = df[df.index >= win_start]

        if len(window_df) < 252:
            logger.warning(f"Window {sy}: only {len(window_df)} rows — skipping.")
            continue

        last_year = window_df.index.max().year
        oos_start = f"{last_year - 1}-01-01"
        train_end = f"{last_year - 1}-12-31"
        train_df  = window_df[window_df.index <= pd.Timestamp(train_end)]

        logger.info(f"=== Window {sy}→{end_date} | OOS: {oos_start}→{end_date} ===")

        try:
            # Slice pre-built pairwise dataset to this window's train range
            splits = date_range_split(
                Xi_all, Xj_all, y_all, dates_all,
                win_start, train_end, train_ratio, val_ratio
            )
            Xi_tr, Xj_tr, y_tr, _ = splits["train"]
            Xi_vl, Xj_vl, y_vl, _ = splits["val"]

            if len(y_tr) < 100:
                logger.warning(f"Window {sy}: too few training pairs ({len(y_tr)}), skipping.")
                continue

            # Always use LightGBM for speed in shrinking window
            model, backend, _ = train_one_horizon(
                Xi_tr, Xj_tr, y_tr, Xi_vl, Xj_vl, y_vl,
                input_dim=Xi_tr.shape[1],
                cfg=module_cfg,
                horizon=best_horizon,
                force_lgbm=True,   # always LGBM for shrinking window speed
            )
            models = {best_horizon: (model, backend)}
        except Exception as e:
            logger.error(f"Window {sy}: training failed: {e}")
            continue

        if not models:
            continue

        oos_df          = window_df[window_df.index >= pd.Timestamp(oos_start)]
        model_by_h      = {h: m for h, (m, _) in models.items()}

        strat, bench, picks = build_strategy_returns(
            oos_df, universe, model_by_h, oos_df.index, benchmark, device
        )

        metrics   = compute_all_metrics(strat, bench)
        cum_strat = compute_cumulative_returns(strat)
        cum_bench = compute_cumulative_returns(bench)
        top_etf   = picks["pick"].mode()[0] if len(picks) > 0 else None

        wr = {
            "window":     f"{sy}→{end_date}",
            "win_start":  win_start,
            "oos_start":  oos_start,
            "oos_end":    end_date,
            "metrics":    metrics,
            "benchmark_metrics": compute_all_metrics(bench),
            "cumulative_returns": {
                "strategy":  {str(k.date()): v for k, v in cum_strat.items()},
                "benchmark": {str(k.date()): v for k, v in cum_bench.items()},
            },
            "picks":    picks.to_dict(orient="records") if len(picks) > 0 else [],
            "top_etf":  top_etf,
        }
        window_results.append(wr)

        if top_etf and metrics.get("ann_return") is not None:
            consensus_inputs.append({
                "etf":        top_etf,
                "ann_return": metrics["ann_return"],
                "sharpe":     metrics["sharpe"],
                "max_dd":     metrics["max_dd"],
                "window":     wr["window"],
            })

    consensus_weights  = bt_cfg.get("consensus_weights", {"ann_return": 0.60, "sharpe": 0.20, "max_dd": 0.20})
    consensus          = compute_consensus_score(consensus_inputs, consensus_weights)
    consensus_ranking  = sorted(consensus.items(), key=lambda x: x[1], reverse=True)
    consensus_etf      = consensus_ranking[0][0] if consensus_ranking else None

    # Normalise consensus scores to conviction-style [0,1] probabilities.
    # Use sigmoid-style rescaling so a single ETF or equal scores don't collapse to 0.
    # Formula: p_i = (rank_i_score - 0.5) clamped into [0,1] offset around 0.5,
    # but simpler: map via min-max then shift into [0.4, 0.6] band to stay readable.
    if consensus_ranking:
        raw_scores  = [s for _, s in consensus_ranking]
        score_min   = min(raw_scores)
        score_max   = max(raw_scores)
        score_range = score_max - score_min

        if score_range == 0 or len(raw_scores) == 1:
            # All ETFs tied or single ETF — assign uniform 0.5 conviction
            consensus_ranking_normalised = [
                {"etf": e, "score": 0.5000}
                for e, _ in consensus_ranking
            ]
        else:
            # Min-max into [0.40, 0.60] so it reads as conviction near 50%
            consensus_ranking_normalised = [
                {"etf": e, "score": round(0.40 + 0.20 * (s - score_min) / score_range, 4)}
                for e, s in consensus_ranking
            ]
    else:
        consensus_ranking_normalised = []

    logger.info(f"Consensus ETF: {consensus_etf}")

    return {
        "mode":              "shrinking_window",
        "n_windows":         len(window_results),
        "consensus_etf":     consensus_etf,
        "consensus_ranking": consensus_ranking_normalised,
        "window_results":    window_results,
        "consensus_inputs":  consensus_inputs,
    }
