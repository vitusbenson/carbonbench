"""Analyze SwinTransformer Optuna study results and compare with UNet baseline.

Usage:
    python analyze_results.py --study-db optuna_study.db
    python analyze_results.py --study-db optuna_study.db --include-best-model
"""

import argparse
import logging
from pathlib import Path

import optuna
import pandas as pd

from neural_transport.training.study_analysis import analyze_study
from neural_transport.training.tuning import get_best_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
UNET_BASELINE_DIR = EXP_DIR.parent / "11_fm_unet_final"


def main():
    parser = argparse.ArgumentParser(description="Analyze SwinTransformer Optuna results")
    parser.add_argument("--study-db", type=str, default="optuna_study.db")
    parser.add_argument("--study-name", type=str, default="fm_swin_tuning")
    parser.add_argument("--include-best-model", action="store_true")
    args = parser.parse_args()

    storage = f"sqlite:///{Path(args.study_db).resolve()}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    # Clean up zombie trials
    zombies = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if zombies:
        logger.info("Cleaning up %d zombie RUNNING trials", len(zombies))
        from optuna.storages import RDBStorage
        st = RDBStorage(storage)
        for t in zombies:
            st.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)
        study = optuna.load_study(study_name=args.study_name, storage=storage)

    # Run Optuna analysis (plots, importance, summary)
    analysis_dir = EXP_DIR / "analysis"
    run_dir = EXP_DIR / "optuna_runs"
    analyze_study(study, out_dir=analysis_dir, run_dir=run_dir)

    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])
    n_failed = len([t for t in study.trials if t.state.name == "FAIL"])

    logger.info("Study: %d complete, %d pruned, %d failed", n_complete, n_pruned, n_failed)
    logger.info("Best value: %.6f", study.best_value)

    best = get_best_config(study)
    logger.info("Best params: %s", {k: v for k, v in best.items()
                                     if k not in ("best_value", "best_trial_number")})

    # Compare with UNet baseline (exp 11)
    comparison_dir = EXP_DIR / "comparison"
    comparison_dir.mkdir(exist_ok=True)

    rows = []

    # UNet baseline metrics
    unet_csv = (
        UNET_BASELINE_DIR / "singlestep" / "preds" / "ckpt=best_massfixer=default"
        / "distributional_eval" / "scores" / "distributional_metrics.csv"
    )
    if unet_csv.exists():
        unet = pd.read_csv(unet_csv)
        unet["experiment"] = "UNet (exp 11)"
        rows.append(unet)
        logger.info("Loaded UNet baseline metrics from %s", unet_csv)

    # SwinTransformer best model metrics (if trained)
    if args.include_best_model:
        swin_csv = (
            EXP_DIR / "singlestep" / "preds" / "ckpt=best_massfixer=default"
            / "distributional_eval" / "scores" / "distributional_metrics.csv"
        )
        if swin_csv.exists():
            swin_df = pd.read_csv(swin_csv)
            swin_df["experiment"] = "SwinTransformer (exp 12)"
            rows.append(swin_df)
            logger.info("Loaded SwinTransformer metrics from %s", swin_csv)

    if rows:
        comparison = pd.concat(rows, ignore_index=True)
        out_path = comparison_dir / "comparison.csv"
        comparison.to_csv(out_path, index=False)
        logger.info("Comparison saved to %s", out_path)
        logger.info("\n%s", comparison.to_string())


if __name__ == "__main__":
    main()
