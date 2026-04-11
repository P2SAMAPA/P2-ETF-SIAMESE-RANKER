"""
metrics.py
Performance metrics: Ann Return, Ann Vol, Sharpe, Max DD, Hit Rate, Consensus.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

TRADING_DAYS = 252


def annualised_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    n_years = len(clean) / TRADING_DAYS
    if n_years <= 0:
        return np.nan
    return float((1 + clean).prod() ** (1 / n_years) - 1)


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.dropna().std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    ann_ret = annualised_return(returns)
    ann_vol = annualised_vol(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float((ann_ret - risk_free) / ann_vol)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns.fillna(0)).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())


def hit_rate(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    s, b = strategy_returns.align(benchmark_returns, join="inner")
    valid = s.notna() & b.notna()
    if valid.sum() == 0:
        return np.nan
    return float((s[valid] > b[valid]).sum() / valid.sum())


def compute_all_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict:
    clean = strategy_returns.dropna()
    m = {
        "ann_return": annualised_return(clean),
        "ann_vol":    annualised_vol(clean),
        "sharpe":     sharpe_ratio(clean),
        "max_dd":     max_drawdown(clean),
        "hit_rate":   hit_rate(clean, benchmark_returns.reindex(clean.index))
                      if benchmark_returns is not None else np.nan,
    }
    return {k: round(float(v), 4) if (v is not None and not np.isnan(v)) else None
            for k, v in m.items()}


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


# ---------------------------------------------------------------------------
# Consensus Score (Shrinking Window)
# ---------------------------------------------------------------------------

def compute_consensus_score(
    window_metrics: List[Dict],
    weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Weighted consensus across shrinking windows.
    Excludes windows where the picked ETF had negative ann_return.
    Weights: ann_return 60%, sharpe 20%, -max_dd 20%.
    """
    if weights is None:
        weights = {"ann_return": 0.60, "sharpe": 0.20, "max_dd": 0.20}

    positive = [m for m in window_metrics if (m.get("ann_return") or 0) > 0]
    if not positive:
        return {}

    scores, counts = {}, {}
    for w in positive:
        etf      = w["etf"]
        ann_ret  = w.get("ann_return") or 0
        sharpe   = w.get("sharpe")     or 0
        max_dd   = w.get("max_dd")     or 0   # negative value
        score    = (
            weights["ann_return"] * ann_ret
            + weights["sharpe"]   * sharpe
            + weights["max_dd"]   * (-max_dd)   # negate so higher = better
        )
        scores[etf]  = scores.get(etf, 0.0) + score
        counts[etf]  = counts.get(etf, 0)   + 1

    return {e: scores[e] / counts[e] for e in scores if counts[e] > 0}
