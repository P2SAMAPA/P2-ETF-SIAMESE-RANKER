"""
Configuration for P2-ETF-SIAMESE-RANKER engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-siamese-ranker-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Columns ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Siamese Network Parameters ---
HIDDEN_LAYERS = [64, 32]
LEARNING_RATE = 0.001
BATCH_SIZE = 128
CONVICTION_THRESHOLD = 0.55
RANDOM_SEED = 42

# --- Feature Engineering ---
FEATURE_WINDOWS = [1, 5, 21, 63]
MIN_OBSERVATIONS = 252

# --- Pairwise dataset subsampling ---
PAIR_SAMPLE_FRAC = 0.10           # use 10% of possible pairs to stay fast

# --- Training Epochs ---
DAILY_EPOCHS = 30
GLOBAL_EPOCHS = 30
SHRINKING_EPOCHS = 15

# --- Training Modes ---
DAILY_LOOKBACK = 504
GLOBAL_TRAIN_START = "2008-01-01"
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
