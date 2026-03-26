"""Post-hoc analysis of Optuna study and best model results.

Generates publication-quality visualizations of the tuning process,
compares with previous baselines (10_fm_unet_tuning, 11_fm_unet_final).

Usage:
    python analyze_results.py
    python analyze_results.py --include-best-model
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
CARBONBENCH_ROOT = EXP_DIR.parent


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna study results")
    parser.add_argument("--study-db", type=str, default=str(EXP_DIR / "optuna_fm_study.db"))
    parser.add_argument("--study-name", type=str, default="fm_tuning")
    parser.add_argument("--include-best-model", action="store_true",
                        help="Include best model's distributional eval in comparison")
    args = parser.parse_args()

    import optuna

    from neural_transport.training.study_analysis import analyze_study
    from neural_transport.training.tuning import get_best_config

    # 1. Analyze Optuna study
    storage = f"sqlite:///{Path(args.study_db).resolve()}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    analysis_dir = EXP_DIR / "analysis"
    analyze_study(study, out_dir=analysis_dir, run_dir=EXP_DIR / "optuna_runs")

    best = get_best_config(study)
    logger.info("Best config: %s", best)

    # 2. Compare with previous baselines
    comparison_dir = EXP_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Load previous tuning results if available
    prev_tuning_dir = CARBONBENCH_ROOT / "10_fm_unet_tuning" / "comparison"
    prev_summary = prev_tuning_dir / "tuning_summary.csv"
    if prev_summary.exists():
        prev_df = pd.read_csv(prev_summary)
        logger.info("Loaded previous tuning results: %d configs", len(prev_df))

        # Get best from previous tuning
        if "energy_distance" in prev_df.columns:
            best_prev = prev_df.loc[prev_df["energy_distance"].idxmin()]
            all_results["prev_best"] = best_prev.to_dict()
            logger.info("Previous best: %s (energy_distance=%.4f)",
                        best_prev.get("config", "?"), best_prev.get("energy_distance", float("inf")))

    # Load 11_fm_unet_final results if available
    final_model_dir = CARBONBENCH_ROOT / "11_fm_unet_final"
    final_dist_dirs = list(final_model_dir.rglob("distributional_eval/scores/distributional_metrics.csv"))
    if final_dist_dirs:
        final_df = pd.read_csv(final_dist_dirs[0])
        if len(final_df) > 0:
            all_results["11_final"] = final_df.iloc[0].to_dict()
            logger.info("Final model metrics loaded")

    # Load best model results if requested
    if args.include_best_model:
        best_dist_dirs = list(EXP_DIR.rglob("distributional_eval/scores/distributional_metrics.csv"))
        if best_dist_dirs:
            best_df = pd.read_csv(best_dist_dirs[0])
            if len(best_df) > 0:
                all_results["optuna_best"] = best_df.iloc[0].to_dict()
                logger.info("Optuna best model metrics loaded")

    # Create comparison table
    if all_results:
        comp_df = pd.DataFrame(all_results).T
        comp_df.to_csv(comparison_dir / "baseline_comparison.csv")
        logger.info("Comparison table saved to %s", comparison_dir / "baseline_comparison.csv")
        print("\n=== Baseline Comparison ===")
        print(comp_df.to_string())

        # Create comparison plots
        try:
            from neural_transport.plots.distributional_plots import plot_tuning_comparison
            plot_tuning_comparison(all_results, comparison_dir, imgformats=("pdf", "png"))
            logger.info("Comparison plots saved")
        except Exception as e:
            logger.warning("Could not create comparison plots: %s", e)

    logger.info("Analysis complete. Results in %s and %s", analysis_dir, comparison_dir)


if __name__ == "__main__":
    main()
