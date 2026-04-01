"""Optuna tuning for all posterior sampling methods (Phase 23).

Runs one Optuna study per method (DPS, FlowDPS, SDE, FIG, ICTM) using
satellite mask patterns on CarbonTracker test data. Each trial evaluates
pressure-weighted RMSE of the ensemble mean vs ground truth.

Usage:
    python run_optuna.py                          # all methods
    python run_optuna.py --method flowdps         # single method
    python run_optuna.py --method sde --n-trials 100
    python run_optuna.py --device cpu             # for testing
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CARBONBENCH_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent
MODEL_DIRS = [
    CARBONBENCH_ROOT / "11_fm_unet_final",
    CARBONBENCH_ROOT / "09_fm_unet_ot_training",
    CARBONBENCH_ROOT / "01_fm_unet_training_baseline",
]

# Data paths
LOCAL_DATA = Path("/scratch/vbenson") / "Carbontracker"
REMOTE_DATA = Path("/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker")


def get_data_path():
    """Find data path, preferring local scratch."""
    for base in [LOCAL_DATA, REMOTE_DATA]:
        test_path = base / "test"
        if test_path.exists():
            return str(test_path)
    raise FileNotFoundError(f"No test data found at {LOCAL_DATA} or {REMOTE_DATA}")


def main():
    parser = argparse.ArgumentParser(description="Posterior sampler Optuna tuning")
    parser.add_argument("--method", type=str, default=None,
                        choices=["dps", "flowdps", "sde", "fig", "ictm"],
                        help="Method to tune (default: all)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from neural_transport.configs import DataConfig, GenerateConfig
    from neural_transport.data.inference_loader import GridInfo
    from neural_transport.inference.tuning import METHODS, run_posterior_study
    from neural_transport.training.train import load_model

    # Load model
    model = None
    for model_dir in MODEL_DIRS:
        try:
            model = load_model(model_dir, device=args.device)
            logger.info("Loaded model from %s", model_dir)
            break
        except Exception as e:
            logger.warning("Failed to load from %s: %s", model_dir, e)
    if model is None:
        raise RuntimeError(f"Could not load model from any of: {MODEL_DIRS}")

    # Load dataset
    data_config = DataConfig(
        dataset="carbontracker", grid="latlon5.625", vertical_levels="l10",
        freq="6h", target_vars=["co2massmix", "p_bottom", "p_top"], forcing_vars=[],
    )
    grid_info = GridInfo.from_config(data_config)
    data_path = get_data_path()

    from neural_transport.data.inference_loader import InferenceDataLoader
    loader = InferenceDataLoader(data_config, data_path=data_path)
    dataset = loader.load_dataset()

    # Base config: satellite mask, column conditioning
    base_config = GenerateConfig(
        n_samples=args.n_samples,
        steps=21,
        refine_start=1.0,
        avg_over_levels=False,
        conditioning={
            "masking": True,
            "mask_source": "column",
            "mask_pattern": "satellite",
            "masking_time": None,
            "t_threshold": 0.9,
            "masking_method": "total_column_average_simple",
            "obs_fraction": 0.3,
            "guidance_scale": 1.0,
            "condition_one_timestep": True,
            "conditioning_mode": "correction",
        },
        sampler_params={
            "sigma_obs": 0.1,
            "spatial_smoothing_sigma": 0.0,
            "fresh_noise": True,
        },
        analyze_masking=False,
        noise_pattern=None,
        analyze_noise=False,
    )

    methods = [args.method] if args.method else METHODS

    for method in methods:
        logger.info("=" * 60)
        logger.info("Starting Optuna study for method: %s", method)
        logger.info("=" * 60)

        run_dir = EXP_DIR / "optuna_runs" / method
        storage = f"sqlite:///{EXP_DIR / 'optuna_runs' / f'{method}_study.db'}"

        study = run_posterior_study(
            method=method,
            model=model,
            dataset=dataset,
            grid_info=grid_info,
            base_config=base_config,
            study_name=f"posterior_{method}",
            storage=storage,
            run_dir=run_dir,
            target_vars=["co2massmix"],
            n_trials=args.n_trials,
            n_targets=args.n_targets,
            n_samples=args.n_samples,
            nan_penalty=100.0,
            device=args.device,
            seed=args.seed,
        )

        # Run analysis
        try:
            from neural_transport.training.study_analysis import analyze_study
            analyze_study(
                study,
                out_dir=EXP_DIR / "analysis" / method,
                run_dir=run_dir,
            )
        except Exception as e:
            logger.warning("Study analysis failed for %s: %s", method, e)


if __name__ == "__main__":
    main()
