"""
equity_module/train.py
Equity module training entrypoint.
"""

import logging
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(force_lgbm: bool = False):
    from core.dataset import load_source_data
    from core.train import train_all_horizons, save_all_models, load_config, load_global_config
    from core.hf_storage import push_model_weights

    module_cfg = load_config("equity_module/config.yaml")
    global_cfg = load_global_config()

    universe = module_cfg["universe"]
    benchmark = module_cfg["benchmark"]
    model_base_path = global_cfg["paths"]["models_dir"]

    logger.info(f"=== Equity Module Training ===")
    logger.info(f"Universe: {universe}")
    logger.info(f"Benchmark: {benchmark}")

    df = load_source_data()

    models = train_all_horizons(
        df, universe, module_cfg, global_cfg, force_lgbm=force_lgbm
    )

    if not models:
        logger.error("No models trained successfully!")
        return

    save_all_models(models, "equity", model_base_path)

    for h, (model, backend) in models.items():
        ext = ".pt" if backend == "siamese" else ".pkl"
        local_path = os.path.join(model_base_path, f"equity_{backend}_h{h}{ext}")
        if os.path.exists(local_path):
            push_model_weights(local_path, "equity", h, backend)

    logger.info("=== Equity Training Complete ===")
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm", action="store_true", help="Force LightGBM backend")
    args = parser.parse_args()
    main(force_lgbm=args.lgbm)
