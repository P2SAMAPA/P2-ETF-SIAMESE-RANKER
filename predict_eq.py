"""
equity_module/predict.py
Equity module inference + backtest entrypoint.
"""

import logging
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(run_backtest: bool = True, force_lgbm: bool = False):
    from core.dataset import load_source_data, build_inference_features
    from core.train import load_config, load_global_config, train_all_horizons
    from core.inference import infer_best_horizon, format_output, save_output
    from core.backtest import run_fixed_split_backtest, run_shrinking_window_backtest
    from core.hf_storage import (
        push_ranking_output, push_backtest_results, push_signal_history,
        pull_signal_history
    )

    module_cfg = load_config("equity_module/config.yaml")
    global_cfg = load_global_config()

    universe = module_cfg["universe"]
    benchmark = module_cfg["benchmark"]

    logger.info("=== Equity Module Prediction ===")

    df = load_source_data()
    signal_date = df.index.max().strftime("%Y-%m-%d")

    models = train_all_horizons(df, universe, module_cfg, global_cfg, force_lgbm=force_lgbm)

    if not models:
        logger.error("No models available!")
        return

    feat = build_inference_features(df, universe)
    model_by_horizon = {h: m for h, (m, _) in models.items()}
    backends = {h: b for h, (m, b) in models.items()}

    result = infer_best_horizon(model_by_horizon, feat, universe)
    best_backend = backends.get(result["best_horizon"], "siamese")

    output_fixed = format_output(
        module="equity",
        ranking=result["ranking"],
        conviction=result["conviction"],
        best_horizon=result["best_horizon"],
        signal_date=signal_date,
        model_backend=best_backend,
        source="fixed_split",
    )

    if run_backtest:
        sw_backtest = run_shrinking_window_backtest(
            df, universe, module_cfg, global_cfg, benchmark, force_lgbm=force_lgbm
        )
        consensus_ranking = sw_backtest.get("consensus_ranking", result["ranking"])
        output_sw = format_output(
            module="equity",
            ranking=consensus_ranking,
            conviction={r["etf"]: r["score"] for r in consensus_ranking},
            best_horizon=result["best_horizon"],
            signal_date=signal_date,
            model_backend=best_backend,
            source="shrinking_window",
        )
    else:
        output_sw = output_fixed.copy()
        output_sw["source"] = "shrinking_window"
        sw_backtest = {}

    combined_output = {
        "fixed_split": output_fixed,
        "shrinking_window": output_sw,
    }

    save_output(combined_output, "equity_module/output.json")

    if run_backtest:
        fs_backtest = run_fixed_split_backtest(df, universe, models, benchmark)
        os.makedirs("outputs/backtest", exist_ok=True)
        with open("outputs/backtest/equity_fixed_split.json", "w") as f:
            json.dump(fs_backtest, f, indent=2, default=str)

        signal_history = []
        existing = pull_signal_history("equity")
        if existing is not None:
            signal_history = existing.to_dict(orient="records")

        signal_history.append({
            "date": signal_date,
            "pick": output_fixed["top_pick"],
            "conviction": output_fixed["top_conviction"],
            "source": "fixed_split",
            "actual_return": None,
        })

        push_ranking_output(combined_output, "equity")
        push_backtest_results(fs_backtest, "equity", "fixed_split")
        if sw_backtest:
            push_backtest_results(sw_backtest, "equity", "shrinking_window")
        push_signal_history(signal_history, "equity")

    logger.info(f"=== Equity Prediction Complete ===")
    logger.info(f"Top pick: {output_fixed['top_pick']} | Conviction: {output_fixed['top_conviction']:.1%}")
    logger.info(f"Shrinking window consensus: {output_sw['top_pick']}")

    return combined_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-backtest", action="store_true")
    parser.add_argument("--lgbm", action="store_true")
    args = parser.parse_args()
    main(run_backtest=not args.no_backtest, force_lgbm=args.lgbm)
