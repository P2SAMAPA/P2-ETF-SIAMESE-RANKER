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
    device:      str  = "cpu",
    force_lgbm:  bool = False,
) -> Dict:
    from train   import train_all_horizons
    from metrics import compute_all_metrics, compute_cumulative_returns, compute_consensus_score

    bt_cfg      = global_cfg.get("backtest", {})
    start_years = range(
        bt_cfg.get("shrinking_start_year", 2008),
        bt_cfg.get("shrinking_end_year",   2024) + 1,
    )
    end_date = df.index.max().strftime("%Y-%m-%d")

    window_results, consensus_inputs = [], []

    for sy in start_years:
        win_start  = f"{sy}-01-01"
        window_df  = df[df.index >= win_start]

        if len(window_df) < 252:
            logger.warning(f"Window {sy}: only {len(window_df)} rows — skipping.")
            continue

        last_year  = window_df.index.max().year
        oos_start  = f"{last_year - 1}-01-01"
        train_end  = f"{last_year - 1}-12-31"
        train_df   = window_df[window_df.index <= train_end]

        logger.info(f"=== Window {sy}→{end_date} | OOS: {oos_start}→{end_date} ===")

        try:
            models = train_all_horizons(
                train_df, universe, module_cfg, global_cfg,
                start_date=win_start, end_date=train_end, force_lgbm=force_lgbm,
            )
        except Exception as e:
            logger.error(f"Window {sy}: training failed: {e}")
            continue

        if not models:
            continue

        oos_df          = window_df[window_df.index >= oos_start]
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

    logger.info(f"Consensus ETF: {consensus_etf}")

    return {
        "mode":              "shrinking_window",
        "n_windows":         len(window_results),
        "consensus_etf":     consensus_etf,
        "consensus_ranking": [{"etf": e, "score": round(s, 4)} for e, s in consensus_ranking],
        "window_results":    window_results,
        "consensus_inputs":  consensus_inputs,
    }
