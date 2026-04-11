"""
hf_storage.py
HuggingFace dataset I/O — push and pull all result artifacts.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

HF_RESULTS_DATASET = "P2SAMAPA/p2-etf-siamese-ranker-results"


def _token():
    return os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def _upload(content_bytes: bytes, path_in_repo: str, msg: str):
    from huggingface_hub import HfApi
    HfApi(token=_token()).upload_file(
        path_or_fileobj=content_bytes,
        path_in_repo=path_in_repo,
        repo_id=HF_RESULTS_DATASET,
        repo_type="dataset",
        commit_message=msg,
    )


def push_ranking_output(output: Dict, module: str):
    try:
        content   = json.dumps(output, indent=2, default=str).encode()
        date_str  = datetime.now().strftime("%Y-%m-%d")
        _upload(content, f"rankings/{module}/latest.json",       f"Ranking {module} {date_str}")
        _upload(content, f"rankings/{module}/{date_str}.json",   f"Archive {module} {date_str}")
        logger.info(f"Pushed {module} ranking → HF")
    except Exception as e:
        logger.error(f"push_ranking_output: {e}")


def push_backtest_results(backtest: Dict, module: str, mode: str):
    try:
        content  = json.dumps(backtest, indent=2, default=str).encode()
        date_str = datetime.now().strftime("%Y-%m-%d")
        _upload(content, f"backtest_metrics/{module}/{mode}_{date_str}.json", f"Backtest {module} {mode} {date_str}")
        _upload(content, f"backtest_metrics/{module}/{mode}_latest.json",     f"Backtest latest {module} {mode}")
        logger.info(f"Pushed {module} {mode} backtest → HF")
    except Exception as e:
        logger.error(f"push_backtest_results: {e}")


def push_model_weights(local_path: str, module: str, horizon: int, backend: str):
    try:
        ext = ".pt" if backend == "siamese" else ".pkl"
        _upload(
            open(local_path, "rb").read(),
            f"model_weights/{module}/{backend}_h{horizon}{ext}",
            f"Weights {module} H={horizon} {backend}",
        )
        logger.info(f"Pushed {backend} H={horizon} weights for {module}")
    except Exception as e:
        logger.error(f"push_model_weights: {e}")


def push_signal_history(history: List[Dict], module: str):
    try:
        import io
        buf = io.StringIO()
        pd.DataFrame(history).to_csv(buf, index=False)
        _upload(
            buf.getvalue().encode(),
            f"signal_history/{module}/history.csv",
            f"Signal history {module}",
        )
        logger.info(f"Pushed {module} signal history ({len(history)} records)")
    except Exception as e:
        logger.error(f"push_signal_history: {e}")


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def pull_latest_ranking(module: str) -> Optional[Dict]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=f"rankings/{module}/latest.json",
            repo_type="dataset", token=_token(),
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"pull_latest_ranking({module}): {e}")
        return None


def pull_backtest_results(module: str, mode: str) -> Optional[Dict]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=f"backtest_metrics/{module}/{mode}_latest.json",
            repo_type="dataset", token=_token(),
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"pull_backtest_results({module},{mode}): {e}")
        return None


def pull_signal_history(module: str) -> Optional[pd.DataFrame]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=f"signal_history/{module}/history.csv",
            repo_type="dataset", token=_token(),
        )
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"pull_signal_history({module}): {e}")
        return None
