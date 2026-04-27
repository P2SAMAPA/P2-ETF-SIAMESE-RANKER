"""
Main training script – Daily, Global, and Shrinking‑Windows modes.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from siamese_model import SiameseRanker
import push_results


def run_siamese_mode(returns, macro, tickers, mode_name, epochs):
    """Run Siamese ranking on a data slice and return top picks."""
    if len(returns) < config.MIN_OBSERVATIONS:
        return None

    X1, X2, labels = data_manager.build_pairwise_dataset(returns, macro, tickers)
    if len(X1) < 100:
        return None

    input_dim = X1.shape[1]
    model = SiameseRanker(
        input_dim=input_dim,
        hidden_dims=config.HIDDEN_LAYERS,
        lr=config.LEARNING_RATE,
        seed=config.RANDOM_SEED
    )

    print(f"  Training Siamese Ranker on {len(X1)} pairs...")
    model.fit(X1, X2, labels, epochs=epochs, batch_size=config.BATCH_SIZE)

    latest_idx = len(returns) - 1
    features_dict = {}
    for ticker in tickers:
        features_dict[ticker] = data_manager.build_feature_vector(returns, macro, ticker, latest_idx)

    conviction_scores = model.compute_conviction_scores(features_dict)

    sorted_tickers = sorted(conviction_scores.items(), key=lambda x: x[1], reverse=True)
    top3 = [{'ticker': t, 'conviction': float(s)} for t, s in sorted_tickers[:3]]
    all_scores = [{'ticker': t, 'conviction': float(s)} for t, s in sorted_tickers]

    return {
        'top_picks': top3,
        'all_scores': all_scores,
        'training_start': str(returns.index[0].date()),
        'training_end': str(returns.index[-1].date()),
        'n_observations': len(returns)
    }


def run_shrinking_windows(df_master, macro, tickers):
    """Fixed shrinking windows with consensus on top ETF."""
    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{start_year}-01-01")
        ed = pd.Timestamp(f"{start_year+2}-12-31")
        mask = (df_master['Date'] >= sd) & (df_master['Date'] <= ed)
        window_df = df_master[mask].copy()
        if len(window_df) < config.MIN_OBSERVATIONS:
            continue

        returns = data_manager.prepare_returns_matrix(window_df, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        m = macro.loc[returns.index]
        mode_out = run_siamese_mode(returns, m, tickers, f"Shrinking {start_year}",
                                    epochs=config.SHRINKING_EPOCHS)
        if mode_out:
            top_ticker = mode_out['top_picks'][0]['ticker']
            top_conviction = mode_out['top_picks'][0]['conviction']
            windows.append({
                'window_start': start_year,
                'window_end': start_year + 2,
                'ticker': top_ticker,
                'conviction': top_conviction
            })

    if not windows:
        return None

    vote = {}
    for w in windows:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(windows) * 100
    return {'ticker': pick, 'conviction': conviction, 'num_windows': len(windows), 'windows': windows}


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            continue

        m = macro.loc[returns_all.index].dropna()
        returns_all = returns_all.loc[m.index]
        m = m.loc[returns_all.index]

        universe_out = {}

        # Daily
        daily_ret = returns_all.iloc[-config.DAILY_LOOKBACK:]
        daily_macro = m.iloc[-config.DAILY_LOOKBACK:]
        daily_out = run_siamese_mode(daily_ret, daily_macro, tickers, "Daily",
                                     epochs=config.DAILY_EPOCHS)
        if daily_out:
            universe_out['daily'] = daily_out
            print(f"  Daily top: {daily_out['top_picks'][0]['ticker']}")

        # Global
        global_out = run_siamese_mode(returns_all, m, tickers, "Global",
                                      epochs=config.GLOBAL_EPOCHS)
        if global_out:
            universe_out['global'] = global_out
            print(f"  Global top: {global_out['top_picks'][0]['ticker']}")

        # Shrinking Windows
        shrinking = run_shrinking_windows(df_master, macro, tickers)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
