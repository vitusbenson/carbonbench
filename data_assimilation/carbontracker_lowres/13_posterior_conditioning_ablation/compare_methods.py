"""Multi-target comparison of best posterior sampling methods (Phase 23).

For each method: generates 10 ensemble members for each of 20 target
samples, using GPU-efficient batched generation with streaming zarr writes.

Usage:
    python compare_methods.py                     # all methods
    python compare_methods.py --method sde        # single method
    python compare_methods.py --n-targets 5 --n-samples 3  # quick test
    python compare_methods.py --plot-only         # regenerate plots from existing zarr
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

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
    for base in [LOCAL_DATA, REMOTE_DATA]:
        if (base / "test").exists():
            return str(base / "test")
    raise FileNotFoundError(f"No test data found at {LOCAL_DATA} or {REMOTE_DATA}")


def get_base_config():
    from neural_transport.configs import GenerateConfig

    return GenerateConfig(
        n_samples=10,
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


def load_best_method_configs():
    """Load best config per method from Optuna study summaries."""
    from neural_transport.configs import compat_to_generate_kwargs
    from neural_transport.inference.tuning import METHODS, _build_generate_config, _reconstruct_method_params

    base_config = get_base_config()
    method_configs = {}

    for method in METHODS:
        summary_path = EXP_DIR / "optuna_runs" / method / "study_summary.json"
        if not summary_path.exists():
            logger.warning("No study summary for %s at %s", method, summary_path)
            continue

        summary = json.loads(summary_path.read_text())
        best_params = summary["best_params"]
        method_params = _reconstruct_method_params(method, best_params)
        config = _build_generate_config(method_params, base_config)
        method_configs[method] = config
        logger.info("Best %s: objective=%.4f, params=%s", method, summary["best_objective"], best_params)

    return method_configs


def run_method(method_name, model, dataset, target_indices, config, args):
    """Run multi-target generation for one method, return zarr path."""
    from neural_transport.configs import compat_to_generate_kwargs
    from neural_transport.inference.generation import generate_multi_target

    generate_kwargs = compat_to_generate_kwargs(config)

    method_dir = EXP_DIR / "results" / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    zarr_path = generate_multi_target(
        model=model,
        dataset=dataset,
        target_indices=target_indices,
        n_samples_per_target=args.n_samples,
        batch_size=args.batch_size,
        generate_kwargs=generate_kwargs,
        out_dir=method_dir,
        device=args.device,
        target_var="co2massmix",
        verbose=True,
        seed=args.seed,
    )
    wall_time = time.perf_counter() - t0

    # Save timing and config
    info = {
        "method": method_name,
        "wall_time_sec": wall_time,
        "n_targets": len(target_indices),
        "n_samples_per_target": args.n_samples,
        "config": config.to_dict(),
    }
    with open(method_dir / "method_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    logger.info("%s: completed in %.1fs → %s", method_name, wall_time, zarr_path)
    return zarr_path


def main():
    parser = argparse.ArgumentParser(description="Multi-target posterior method comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-targets", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default=None, help="Run single method")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        from plot_results import plot_all
        plot_all(EXP_DIR)
        return

    from neural_transport.configs import DataConfig, compat_to_generate_kwargs
    from neural_transport.data.inference_loader import InferenceDataLoader
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
    data_path = get_data_path()
    loader = InferenceDataLoader(data_config, data_path=data_path)
    dataset = loader.load_dataset()

    # Random target indices (reproducible)
    rng = np.random.RandomState(args.seed)
    n_available = len(dataset)
    target_indices = sorted(rng.choice(n_available, size=min(args.n_targets, n_available), replace=False).tolist())
    logger.info("Selected %d target indices: %s", len(target_indices), target_indices)

    # Save target indices for reproducibility
    results_dir = EXP_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "target_indices.json", "w") as f:
        json.dump({"target_indices": target_indices, "seed": args.seed}, f, indent=2)

    # Load best configs
    method_configs = load_best_method_configs()

    # Add unconditional baseline
    base_config = get_base_config()
    uncond_config = base_config.merge(**{"conditioning.masking": False})
    method_configs = {"unconditional": uncond_config, **method_configs}

    # Run selected methods
    methods_to_run = [args.method] if args.method else list(method_configs.keys())

    for method_name in methods_to_run:
        if method_name not in method_configs:
            logger.warning("No config for method %s, skipping", method_name)
            continue
        logger.info("=" * 60)
        logger.info("Running method: %s", method_name)
        logger.info("=" * 60)
        run_method(method_name, model, dataset, target_indices, method_configs[method_name], args)

    # Generate plots
    from plot_results import plot_all
    plot_all(EXP_DIR)


if __name__ == "__main__":
    main()
