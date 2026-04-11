"""
equity_predict.py
Equity Sectors module — inference + backtest entrypoint.
Run: python equity_predict.py [--no-backtest] [--lgbm]
"""

import logging
import os
import json
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(run_backtest: bool = True, force_lgbm: bool = False):
    from dataset    import load_source_data, build_inference_features
    from train      import train_all_horizons, load_module_config, load_global_config
    from inference  import infer_best_horizon, format_output, save_output
    from backtest   import run_fixed_split_backtest, run_shrinking_window_backtest
    from hf_storage import (push_ranking_output, push_backtest_results,
                             push_signal_history, pull_signal_history)

    module_cfg = load_module_config("equity")
    global_cfg = load_global_config()
    universe   = module_cfg["universe"]
    benchmark  = module_cfg["benchmark"]

    logger.info("=== Equity Prediction ===")

    df          = load_source_data()
    signal_date = df.index.max().strftime("%Y-%m-%d")

    models = train_all_horizons(df, universe, module_cfg, global_cfg, force_lgbm=force_lgbm)
    if not models:
        logger.error("No models available.")
        return None

    feat             = build_inference_features(df, universe)
    model_by_horizon = {h: m for h, (m, _) in models.items()}
    backends         = {h: b for h, (_, b) in models.items()}

    result       = infer_best_horizon(model_by_horizon, feat, universe)
    best_backend = backends.get(result["best_horizon"], "siamese")

    output_fixed = format_output(
        module="equity", ranking=result["ranking"], conviction=result["conviction"],
        best_horizon=result["best_horizon"], signal_date=signal_date,
        model_backend=best_backend, source="fixed_split",
    )

    if run_backtest:
        sw_bt = run_shrinking_window_backtest(
            df, universe, module_cfg, global_cfg, benchmark, force_lgbm=force_lgbm
        )
        sw_ranking = sw_bt.get("consensus_ranking", result["ranking"])
        output_sw  = format_output(
            module="equity", ranking=sw_ranking,
            conviction={r["etf"]: r["score"] for r in sw_ranking},
            best_horizon=result["best_horizon"], signal_date=signal_date,
            model_backend=best_backend, source="shrinking_window",
        )
    else:
        output_sw = {**output_fixed, "source": "shrinking_window"}
        sw_bt     = {}

    combined = {"fixed_split": output_fixed, "shrinking_window": output_sw}
    save_output(combined, "outputs/equity_output.json")

    if run_backtest:
        fs_bt = run_fixed_split_backtest(df, universe, models, benchmark)
        os.makedirs("outputs/backtest", exist_ok=True)
        with open("outputs/backtest/equity_fixed_split.json", "w") as f:
            json.dump(fs_bt, f, indent=2, default=str)

        existing = pull_signal_history("equity")
        history  = existing.to_dict(orient="records") if existing is not None else []
        history.append({
            "date":          signal_date,
            "pick":          output_fixed["top_pick"],
            "conviction":    output_fixed["top_conviction"],
            "source":        "fixed_split",
            "actual_return": None,
        })

        push_ranking_output(combined, "equity")
        push_backtest_results(fs_bt, "equity", "fixed_split")
        if sw_bt:
            push_backtest_results(sw_bt, "equity", "shrinking_window")
        push_signal_history(history, "equity")

    logger.info(f"=== Equity Done | Top: {output_fixed['top_pick']}  "
                f"Conviction: {output_fixed['top_conviction']:.1%} ===")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equity module prediction")
    parser.add_argument("--no-backtest", action="store_true")
    parser.add_argument("--lgbm",        action="store_true")
    args = parser.parse_args()
    main(run_backtest=not args.no_backtest, force_lgbm=args.lgbm)
