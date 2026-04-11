"""
core/metrics.py
Performance metrics: Ann Return, Ann Vol, Sharpe, Max DD, Hit Rate.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


TRADING_DAYS = 252


def annualised_return(returns: pd.Series) -> float:
    """Annualised return from daily return series."""
    if len(returns) == 0:
        return np.nan
    total = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    if n_years <= 0:
        return np.nan
    return float(total ** (1 / n_years) - 1)


def annualised_vol(returns: pd.Series) -> float:
    """Annualised volatility."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    ann_ret = annualised_return(returns)
    ann_vol = annualised_vol(returns)
    if ann_vol == 0:
        return np.nan
    return float((ann_ret - risk_free) / ann_vol)


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown (peak-to-trough). Returns negative float."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())


def hit_rate(picks: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Fraction of days where the picked ETF outperformed the benchmark.
    """
    aligned = picks.align(benchmark_returns, join="inner")
    picked_ret, bench_ret = aligned
    hits = (picked_ret > bench_ret).sum()
    total = len(picked_ret.dropna())
    if total == 0:
        return np.nan
    return float(hits / total)


def compute_all_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute full metric suite for a return series.

    Returns dict with:
        ann_return, ann_vol, sharpe, max_dd, hit_rate (vs benchmark if provided)
    """
    clean = strategy_returns.dropna()
    metrics = {
        "ann_return": annualised_return(clean),
        "ann_vol": annualised_vol(clean),
        "sharpe": sharpe_ratio(clean),
        "max_dd": max_drawdown(clean),
    }
    if benchmark_returns is not None:
        metrics["hit_rate"] = hit_rate(clean, benchmark_returns.reindex(clean.index))
    else:
        metrics["hit_rate"] = np.nan

    return {k: round(float(v), 4) if not np.isnan(v) else None for k, v in metrics.items()}


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative return series starting at 1.0."""
    return (1 + returns.fillna(0)).cumprod()


# ---------------------------------------------------------------------------
# Consensus Score (Shrinking Window)
# ---------------------------------------------------------------------------

def compute_consensus_score(
    window_metrics: list,
    weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Compute weighted consensus score across shrinking windows.

    Args:
        window_metrics: list of dicts, each with {etf, ann_return, sharpe, max_dd}
        weights: {ann_return: 0.6, sharpe: 0.2, max_dd: 0.2}

    Returns:
        dict: {etf: consensus_score}
    """
    if weights is None:
        weights = {"ann_return": 0.60, "sharpe": 0.20, "max_dd": 0.20}

    # Filter out negative-return windows
    positive_windows = [m for m in window_metrics if m.get("ann_return", -1) > 0]

    if not positive_windows:
        return {}

    # Collect scores per ETF
    etf_scores = {}
    etf_counts = {}

    for window in positive_windows:
        etf = window["etf"]
        ann_ret = window.get("ann_return", 0) or 0
        sharpe = window.get("sharpe", 0) or 0
        max_dd = window.get("max_dd", 0) or 0  # negative value

        # Negate max_dd so higher = better
        score = (
            weights["ann_return"] * ann_ret
            + weights["sharpe"] * sharpe
            + weights["max_dd"] * (-max_dd)
        )

        if etf not in etf_scores:
            etf_scores[etf] = 0.0
            etf_counts[etf] = 0
        etf_scores[etf] += score
        etf_counts[etf] += 1

    # Average across windows
    consensus = {
        etf: etf_scores[etf] / etf_counts[etf]
        for etf in etf_scores
        if etf_counts[etf] > 0
    }

    return consensus
