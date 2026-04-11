"""
fi_train.py
FI / Commodities module — training entrypoint.
Run: python fi_train.py [--lgbm]
"""

import logging
import os
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(force_lgbm: bool = False):
    from dataset     import load_source_data
    from train       import train_all_horizons, save_all_models, load_module_config, load_global_config
    from hf_storage  import push_model_weights

    module_cfg      = load_module_config("fi")
    global_cfg      = load_global_config()
    universe        = module_cfg["universe"]
    model_base_path = global_cfg["paths"]["models_dir"]

    logger.info(f"=== FI Training | Universe: {universe} ===")

    df     = load_source_data()
    models = train_all_horizons(df, universe, module_cfg, global_cfg, force_lgbm=force_lgbm)

    if not models:
        logger.error("No models trained.")
        return None

    save_all_models(models, "fi", model_base_path)

    for h, (_, backend) in models.items():
        ext  = ".pt" if backend == "siamese" else ".pkl"
        path = os.path.join(model_base_path, f"fi_{backend}_h{h}{ext}")
        if os.path.exists(path):
            push_model_weights(path, "fi", h, backend)

    logger.info("=== FI Training Complete ===")
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FI module")
    parser.add_argument("--lgbm", action="store_true", help="Force LightGBM backend")
    args = parser.parse_args()
    main(force_lgbm=args.lgbm)
