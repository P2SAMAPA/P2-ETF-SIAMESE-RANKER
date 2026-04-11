"""
core/hf_storage.py
HuggingFace dataset I/O for storing and retrieving results.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

HF_RESULTS_DATASET = "P2SAMAPA/p2-etf-siamese-ranker-results"


def get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Push Results
# ---------------------------------------------------------------------------

def push_ranking_output(output: Dict, module: str):
    """Push latest ranking output JSON to HF dataset."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=get_hf_token())

        content = json.dumps(output, indent=2, default=str)
        path_in_repo = f"rankings/{module}/latest.json"

        # Also archive with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        archive_path = f"rankings/{module}/{date_str}.json"

        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=path_in_repo,
            repo_id=HF_RESULTS_DATASET,
            repo_type="dataset",
            commit_message=f"Update {module} ranking {date_str}",
        )
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=archive_path,
            repo_id=HF_RESULTS_DATASET,
            repo_type="dataset",
            commit_message=f"Archive {module} ranking {date_str}",
        )
        logger.info(f"Pushed {module} ranking to HF: {path_in_repo}")
    except Exception as e:
        logger.error(f"Failed to push ranking to HF: {e}")


def push_backtest_results(backtest: Dict, module: str, mode: str):
    """Push backtest results JSON to HF dataset."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=get_hf_token())

        content = json.dumps(backtest, indent=2, default=str)
        date_str = datetime.now().strftime("%Y-%m-%d")
        path_in_repo = f"backtest_metrics/{module}/{mode}_{date_str}.json"
        latest_path = f"backtest_metrics/{module}/{mode}_latest.json"

        for p in [path_in_repo, latest_path]:
            api.upload_file(
                path_or_fileobj=content.encode(),
                path_in_repo=p,
                repo_id=HF_RESULTS_DATASET,
                repo_type="dataset",
                commit_message=f"Backtest {module} {mode} {date_str}",
            )
        logger.info(f"Pushed {module} {mode} backtest to HF")
    except Exception as e:
        logger.error(f"Failed to push backtest to HF: {e}")


def push_model_weights(local_path: str, module: str, horizon: int, backend: str):
    """Push model weights to HF dataset."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=get_hf_token())
        ext = ".pt" if backend == "siamese" else ".pkl"
        path_in_repo = f"model_weights/{module}/{backend}_h{horizon}{ext}"
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=HF_RESULTS_DATASET,
            repo_type="dataset",
            commit_message=f"Model weights {module} H={horizon} {backend}",
        )
        logger.info(f"Pushed {backend} H={horizon} weights for {module}")
    except Exception as e:
        logger.error(f"Failed to push model weights: {e}")


def push_features(parquet_path: str, module: str):
    """Push precomputed feature parquet to HF dataset."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=get_hf_token())
        path_in_repo = f"features/{module}/features_latest.parquet"
        api.upload_file(
            path_or_fileobj=parquet_path,
            path_in_repo=path_in_repo,
            repo_id=HF_RESULTS_DATASET,
            repo_type="dataset",
            commit_message=f"Features {module}",
        )
        logger.info(f"Pushed {module} features to HF")
    except Exception as e:
        logger.error(f"Failed to push features: {e}")


def push_signal_history(history: List[Dict], module: str):
    """Push/update signal history CSV to HF dataset."""
    try:
        from huggingface_hub import HfApi
        import io
        api = HfApi(token=get_hf_token())

        df = pd.DataFrame(history)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        content = buf.getvalue().encode()

        path_in_repo = f"signal_history/{module}/history.csv"
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=path_in_repo,
            repo_id=HF_RESULTS_DATASET,
            repo_type="dataset",
            commit_message=f"Signal history {module}",
        )
        logger.info(f"Pushed {module} signal history to HF ({len(history)} records)")
    except Exception as e:
        logger.error(f"Failed to push signal history: {e}")


# ---------------------------------------------------------------------------
# Pull Results
# ---------------------------------------------------------------------------

def pull_latest_ranking(module: str) -> Optional[Dict]:
    """Pull latest ranking JSON from HF dataset."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=f"rankings/{module}/latest.json",
            repo_type="dataset",
            token=get_hf_token(),
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not pull {module} ranking from HF: {e}")
        return None


def pull_backtest_results(module: str, mode: str) -> Optional[Dict]:
    """Pull latest backtest results from HF dataset."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=f"backtest_metrics/{module}/{mode}_latest.json",
            repo_type="dataset",
            token=get_hf_token(),
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not pull {module} {mode} backtest from HF: {e}")
        return None


def pull_signal_history(module: str) -> Optional[pd.DataFrame]:
    """Pull signal history CSV from HF dataset."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=f"signal_history/{module}/history.csv",
            repo_type="dataset",
            token=get_hf_token(),
        )
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Could not pull {module} signal history from HF: {e}")
        return None


# ---------------------------------------------------------------------------
# Local cache fallback
# ---------------------------------------------------------------------------

def load_local_or_hf(local_path: str, hf_loader, *args) -> Optional[Dict]:
    """Try local cache first, fall back to HF pull."""
    if os.path.exists(local_path):
        try:
            with open(local_path) as f:
                return json.load(f)
        except Exception:
            pass
    return hf_loader(*args)
