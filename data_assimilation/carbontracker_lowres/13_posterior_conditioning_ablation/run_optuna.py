"""Unified Optuna tuning for all posterior conditioning methods.

Tunes one method per invocation. Multiple workers can share the same
Optuna study via SQLite storage for parallelism.

Methods: dps, flowdps, sde, fig, ictm, mcg, pcfm, fmps

Usage:
    python run_optuna.py --method fmps --n-trials 5 --worker-id 0
    python run_optuna.py --method dps --n-trials 50   # single worker, all trials
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
CARBONBENCH_ROOT = EXP_DIR.parent
MODEL_DIRS = [
    CARBONBENCH_ROOT / "11_fm_unet_final",
    CARBONBENCH_ROOT / "09_fm_unet_ot_training",
    CARBONBENCH_ROOT / "01_fm_unet_training_baseline",
]

LOCAL_DATA = Path("/scratch/vbenson") / "Carbontracker"
REMOTE_DATA = Path("/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker")

ALL_METHODS = ["dps", "flowdps", "sde", "fig", "ictm", "mcg", "pcfm", "fmps", "dflow"]


def get_data_path():
    for base in [LOCAL_DATA, REMOTE_DATA]:
        if (base / "test").exists():
            return str(base / "test")
    raise FileNotFoundError(f"No test data found at {LOCAL_DATA} or {REMOTE_DATA}")


def main():
    parser = argparse.ArgumentParser(description="Unified Optuna tuning for posterior conditioning")
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS)
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Trials per worker (total = n_workers * n_trials)")
    parser.add_argument("--n-targets", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker ID for seed offset (use SLURM array task modulo)")
    args = parser.parse_args()

    # Each worker gets a unique seed for TPE exploration diversity
    args.seed = args.seed + args.worker_id

    from neural_transport.configs import DataConfig, GenerateConfig
    from neural_transport.data.inference_loader import GridInfo, InferenceDataLoader
    from neural_transport.inference.tuning import run_posterior_study
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
            "soft_boundary_sigma": 0.0,
            "fresh_noise": True,
        },
        analyze_masking=False,
        noise_pattern=None,
        analyze_noise=False,
    )

    method = args.method
    run_dir = EXP_DIR / "optuna_runs" / method
    storage = f"sqlite:///{EXP_DIR / 'optuna_runs' / f'{method}_study.db'}"
    study_name = f"posterior_{method}"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Worker 0 creates the study; others wait for the DB file
    import optuna

    if args.worker_id == 0:
        optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )
        logger.info("Worker 0: study '%s' created/loaded", study_name)
    else:
        db_path = EXP_DIR / "optuna_runs" / f"{method}_study.db"
        for _ in range(120):
            if db_path.exists():
                break
            time.sleep(1)
        logger.info("Worker %d: study DB found, joining", args.worker_id)

    logger.info("=" * 60)
    logger.info("Optuna worker %d for %s: %d trials, %d targets x %d samples",
                args.worker_id, method.upper(), args.n_trials, args.n_targets, args.n_samples)
    logger.info("=" * 60)

    study = run_posterior_study(
        method=method,
        model=model,
        dataset=dataset,
        grid_info=grid_info,
        base_config=base_config,
        study_name=study_name,
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

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info("Worker %d done. Study '%s' has %d complete trials.", args.worker_id, study_name, n_complete)

    if n_complete > 0:
        logger.info("Best trial: #%d, value=%.4f", study.best_trial.number, study.best_value)
        logger.info("Best params: %s", study.best_params)

    # Only worker 0 runs analysis
    if args.worker_id == 0:
        try:
            from neural_transport.training.study_analysis import analyze_study
            analyze_study(study, out_dir=EXP_DIR / "analysis" / method, run_dir=run_dir)
        except Exception as e:
            logger.warning("Study analysis failed: %s", e)


if __name__ == "__main__":
    main()
