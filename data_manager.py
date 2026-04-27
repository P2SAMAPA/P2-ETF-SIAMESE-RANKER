"""
Data loading and feature engineering for Siamese Ranker engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare wide‑format log returns."""
    available_tickers = [t for t in tickers if t in df_wide.columns]
    df_long = pd.melt(
        df_wide, id_vars=['Date'], value_vars=available_tickers,
        var_name='ticker', value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers].dropna()

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Extract macro columns and return as DataFrame with Date index."""
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_feature_vector(returns: pd.DataFrame, macro: pd.DataFrame, ticker: str, idx: int) -> np.ndarray:
    """Build a feature vector for a single ETF at a given time index."""
    features = []
    for w in config.FEATURE_WINDOWS:
        if idx >= w:
            features.append(float(returns[ticker].iloc[idx-w:idx].mean()))
        else:
            features.append(0.0)
    
    for col in macro.columns:
        features.append(float(macro[col].iloc[idx]))
    
    return np.array(features, dtype=np.float32)

def build_pairwise_dataset(returns: pd.DataFrame, macro: pd.DataFrame, tickers: list) -> tuple:
    """
    Build pairwise ranking dataset for Siamese network.
    Subsamples pairs per day to config.PAIR_SAMPLE_FRAC.
    """
    X1_list, X2_list, labels_list = [], [], []
    n_tickers = len(tickers)
    rng = np.random.default_rng(config.RANDOM_SEED)
    
    for idx in range(63, len(returns) - 1):
        future_returns = returns.iloc[idx + 1]
        # Enumerate all pairs for this day, then subsample
        pairs = [(i, j) for i in range(n_tickers) for j in range(n_tickers) if i != j]
        n_pairs = len(pairs)
        n_sample = max(2, int(n_pairs * config.PAIR_SAMPLE_FRAC))
        sampled = rng.choice(n_pairs, size=n_sample, replace=False)
        
        for pair_idx in sampled:
            i, j = pairs[pair_idx]
            t1, t2 = tickers[i], tickers[j]
            feat1 = build_feature_vector(returns, macro, t1, idx)
            feat2 = build_feature_vector(returns, macro, t2, idx)
            label = 1.0 if future_returns[t1] > future_returns[t2] else 0.0
            
            X1_list.append(feat1)
            X2_list.append(feat2)
            labels_list.append(label)
    
    return (
        np.array(X1_list, dtype=np.float32),
        np.array(X2_list, dtype=np.float32),
        np.array(labels_list, dtype=np.float32)
    )
