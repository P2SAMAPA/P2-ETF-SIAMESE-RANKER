"""
fi_module/train.py
FI module training entrypoint.
"""

import logging
import yaml
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(force_lgbm: bool = False):
    from core.dataset import load_source_data
    from core.train import train_all_horizons, save_all_models, load_config, load_global_config
    from core.hf_storage import push_model_weights, push_features

    module_cfg = load_config("fi_module/config.yaml")
    global_cfg = load_global_config()

    universe = module_cfg["universe"]
    benchmark = module_cfg["benchmark"]
    model_base_path = global_cfg["paths"]["models_dir"]

    logger.info(f"=== FI Module Training ===")
    logger.info(f"Universe: {universe}")
    logger.info(f"Benchmark: {benchmark}")

    # Load data
    df = load_source_data()

    # Train all horizons
    models = train_all_horizons(
        df, universe, module_cfg, global_cfg, force_lgbm=force_lgbm
    )

    if not models:
        logger.error("No models trained successfully!")
        return

    # Save locally
    save_all_models(models, "fi", model_base_path)

    # Push weights to HF
    for h, (model, backend) in models.items():
        ext = ".pt" if backend == "siamese" else ".pkl"
        local_path = os.path.join(model_base_path, f"fi_{backend}_h{h}{ext}")
        if os.path.exists(local_path):
            push_model_weights(local_path, "fi", h, backend)

    logger.info("=== FI Training Complete ===")
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm", action="store_true", help="Force LightGBM backend")
    args = parser.parse_args()
    main(force_lgbm=args.lgbm)
