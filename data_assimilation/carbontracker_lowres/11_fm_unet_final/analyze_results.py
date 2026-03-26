"""Analyze Optuna study results and compare with baseline.

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
BASELINE_DIR = EXP_DIR.parent / "10_fm_unet_tuning"


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna results")
    parser.add_argument("--study-db", type=str, default="optuna_study.db")
    parser.add_argument("--study-name", type=str, default="fm_tuning")
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

    # Compare with baseline (exp 10)
    comparison_dir = EXP_DIR / "comparison"
    comparison_dir.mkdir(exist_ok=True)

    rows = []

    # Baseline metrics
    baseline_csv = (
        BASELINE_DIR / "singlestep" / "preds" / "ckpt=best_massfixer=default"
        / "distributional_eval" / "scores" / "distributional_metrics.csv"
    )
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv)
        baseline["experiment"] = "baseline (exp 10)"
        rows.append(baseline)
        logger.info("Loaded baseline metrics from %s", baseline_csv)

    # Best model metrics (if trained)
    if args.include_best_model:
        best_csv = (
            EXP_DIR / "singlestep" / "preds" / "ckpt=best_massfixer=default"
            / "distributional_eval" / "scores" / "distributional_metrics.csv"
        )
        if best_csv.exists():
            best_df = pd.read_csv(best_csv)
            best_df["experiment"] = "best (exp 11)"
            rows.append(best_df)
            logger.info("Loaded best model metrics from %s", best_csv)

    if rows:
        comparison = pd.concat(rows, ignore_index=True)
        out_path = comparison_dir / "comparison.csv"
        comparison.to_csv(out_path, index=False)
        logger.info("Comparison saved to %s", out_path)


if __name__ == "__main__":
    main()
