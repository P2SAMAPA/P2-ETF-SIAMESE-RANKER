"""
core/backtest.py
Backtest engine — Fixed Split and Shrinking Window modes.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy Return Series Builder
# ---------------------------------------------------------------------------

def build_strategy_returns(
    df: pd.DataFrame,
    universe: List[str],
    model_by_horizon: Dict[int, object],
    test_dates: pd.DatetimeIndex,
    benchmark: str,
    device: str = "cpu",
    retrain_daily: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Simulate strategy: each day, pick the top-ranked ETF and hold for best horizon.

    Returns:
        strategy_returns: daily return series
        benchmark_returns: benchmark daily return series
        picks_df: DataFrame with date, pick, conviction, horizon, actual_return
    """
    from core.dataset import build_inference_features
    from core.inference import compute_conviction_scores, rank_etfs

    # Pre-compute daily returns for all ETFs
    price_cols = [e for e in universe + [benchmark] if e in df.columns]
    prices = df[price_cols].ffill()
    daily_returns = prices.pct_change()

    strategy_ret = []
    benchmark_ret = []
    picks = []

    for date in test_dates:
        if date not in daily_returns.index:
            continue

        # Get features as of this date
        feat = build_inference_features(df, universe, as_of_date=date)
        if len(feat) < 2:
            continue

        # Pick best horizon
        best_score = -1
        best_h = 1
        best_ranking = None

        for h, model in model_by_horizon.items():
            conviction = compute_conviction_scores(model, feat, universe, device)
            ranking = rank_etfs(conviction)
            if ranking and ranking[0]["score"] > best_score:
                best_score = ranking[0]["score"]
                best_h = h
                best_ranking = ranking

        if best_ranking is None:
            continue

        top_etf = best_ranking[0]["etf"]
        top_conviction = best_ranking[0]["score"]

        # Get actual return for the next day
        future_dates = daily_returns.index[daily_returns.index > date]
        if len(future_dates) == 0:
            continue
        next_date = future_dates[0]

        if top_etf not in daily_returns.columns or next_date not in daily_returns.index:
            continue

        etf_return = daily_returns.loc[next_date, top_etf]
        bench_return = daily_returns.loc[next_date, benchmark] if benchmark in daily_returns.columns else np.nan

        strategy_ret.append({"date": next_date, "return": etf_return})
        benchmark_ret.append({"date": next_date, "return": bench_return})
        picks.append({
            "date": date,
            "pick": top_etf,
            "conviction": round(top_conviction, 4),
            "horizon": best_h,
            "actual_return": round(float(etf_return), 6) if not np.isnan(etf_return) else None,
        })

    strat_series = pd.Series(
        [r["return"] for r in strategy_ret],
        index=pd.DatetimeIndex([r["date"] for r in strategy_ret]),
        name="strategy"
    ).dropna()

    bench_series = pd.Series(
        [r["return"] for r in benchmark_ret],
        index=pd.DatetimeIndex([r["date"] for r in benchmark_ret]),
        name="benchmark"
    ).dropna()

    picks_df = pd.DataFrame(picks)

    return strat_series, bench_series, picks_df


# ---------------------------------------------------------------------------
# Fixed Split Backtest
# ---------------------------------------------------------------------------

def run_fixed_split_backtest(
    df: pd.DataFrame,
    universe: List[str],
    models: Dict[int, Tuple[object, str]],
    benchmark: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    device: str = "cpu",
) -> Dict:
    """
    Run fixed-split backtest on the test period (last 10% of dates).
    """
    from core.dataset import build_pairwise_dataset, temporal_split
    from core.metrics import compute_all_metrics, compute_cumulative_returns

    # Get test dates
    valid_dates = df.dropna(subset=[universe[0]]).index
    n = len(valid_dates)
    test_start_idx = int(n * (train_ratio + val_ratio))
    test_dates = valid_dates[test_start_idx:]

    model_by_horizon = {h: m for h, (m, _) in models.items()}

    logger.info(f"Fixed split backtest | Test period: {test_dates[0].date()} → {test_dates[-1].date()}")

    strat_ret, bench_ret, picks_df = build_strategy_returns(
        df, universe, model_by_horizon, test_dates, benchmark, device
    )

    metrics = compute_all_metrics(strat_ret, bench_ret)
    bench_metrics = compute_all_metrics(bench_ret)

    cum_strat = compute_cumulative_returns(strat_ret)
    cum_bench = compute_cumulative_returns(bench_ret)

    return {
        "mode": "fixed_split",
        "test_start": str(test_dates[0].date()),
        "test_end": str(test_dates[-1].date()),
        "metrics": metrics,
        "benchmark_metrics": bench_metrics,
        "cumulative_returns": {
            "strategy": cum_strat.to_dict(),
            "benchmark": cum_bench.to_dict(),
        },
        "picks": picks_df.to_dict(orient="records") if len(picks_df) > 0 else [],
        "n_picks": len(picks_df),
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
    device: str = "cpu",
    force_lgbm: bool = False,
) -> Dict:
    """
    Shrinking window backtest:
    Window 1: 2008→2026 YTD
    Window 2: 2009→2026 YTD
    ...
    Window 17: 2024→2026 YTD

    For each window: train on data, evaluate OOS (last year of window).
    """
    from core.train import train_all_horizons
    from core.metrics import compute_all_metrics, compute_cumulative_returns, compute_consensus_score

    backtest_cfg = global_cfg.get("backtest", {})
    start_years = range(
        backtest_cfg.get("shrinking_start_year", 2008),
        backtest_cfg.get("shrinking_end_year", 2024) + 1,
    )
    end_date = df.index.max().strftime("%Y-%m-%d")

    window_results = []
    consensus_inputs = []

    for start_year in start_years:
        window_start = f"{start_year}-01-01"
        window_df = df[df.index >= window_start]

        if len(window_df) < 252:  # skip windows with insufficient data
            logger.warning(f"Window {start_year}: insufficient data ({len(window_df)} rows), skipping.")
            continue

        # OOS = last year of the window
        oos_start = f"{window_df.index.max().year - 1}-01-01"
        train_end = f"{window_df.index.max().year - 1}-12-31"
        train_df = window_df[window_df.index <= train_end]

        logger.info(f"=== Window {start_year}→{end_date} | OOS: {oos_start}→{end_date} ===")

        try:
            models = train_all_horizons(
                train_df, universe, module_cfg, global_cfg,
                start_date=window_start, end_date=train_end,
                force_lgbm=force_lgbm,
            )
        except Exception as e:
            logger.error(f"Window {start_year}: Training failed: {e}")
            continue

        if not models:
            continue

        oos_df = window_df[window_df.index >= oos_start]
        oos_dates = oos_df.index
        model_by_horizon = {h: m for h, (m, _) in models.items()}

        strat_ret, bench_ret, picks_df = build_strategy_returns(
            oos_df, universe, model_by_horizon, oos_dates, benchmark, device
        )

        metrics = compute_all_metrics(strat_ret, bench_ret)
        cum_strat = compute_cumulative_returns(strat_ret)
        cum_bench = compute_cumulative_returns(bench_ret)

        # Top ETF for this window
        top_etf = picks_df["pick"].mode()[0] if len(picks_df) > 0 else None

        window_result = {
            "window": f"{start_year}→{end_date}",
            "window_start": window_start,
            "oos_start": oos_start,
            "oos_end": end_date,
            "metrics": metrics,
            "benchmark_metrics": compute_all_metrics(bench_ret),
            "cumulative_returns": {
                "strategy": cum_strat.to_dict(),
                "benchmark": cum_bench.to_dict(),
            },
            "picks": picks_df.to_dict(orient="records") if len(picks_df) > 0 else [],
            "top_etf": top_etf,
        }
        window_results.append(window_result)

        # For consensus: add best ETF metrics
        if top_etf and metrics.get("ann_return") is not None:
            consensus_inputs.append({
                "etf": top_etf,
                "ann_return": metrics["ann_return"],
                "sharpe": metrics["sharpe"],
                "max_dd": metrics["max_dd"],
                "window": window_result["window"],
            })

    # Compute consensus
    consensus_weights = backtest_cfg.get("consensus_weights", {
        "ann_return": 0.60, "sharpe": 0.20, "max_dd": 0.20
    })
    consensus = compute_consensus_score(consensus_inputs, consensus_weights)
    consensus_ranking = sorted(consensus.items(), key=lambda x: x[1], reverse=True)
    consensus_etf = consensus_ranking[0][0] if consensus_ranking else None

    logger.info(f"Consensus ETF: {consensus_etf} | Score: {consensus.get(consensus_etf, 0):.4f}")

    return {
        "mode": "shrinking_window",
        "n_windows": len(window_results),
        "consensus_etf": consensus_etf,
        "consensus_ranking": [{"etf": e, "score": round(s, 4)} for e, s in consensus_ranking],
        "window_results": window_results,
        "consensus_inputs": consensus_inputs,
    }
