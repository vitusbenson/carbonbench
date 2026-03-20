"""ICTM (Iterative Corrupted Trajectory Matching) Ablation Experiment (Phase 9).

Sweeps r_max, r_schedule, n_inner_steps, inner_lr, sigma_obs, step count.

Usage:
    python run_ablation.py
    python run_ablation.py --device cuda
    python run_ablation.py --filter r_max
"""

from pathlib import Path

from neural_transport.configs import DataConfig, GenerateConfig
from neural_transport.experiments.ablation_runner import AblationRunner

CARBONBENCH_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent
MODEL_DIRS = [
    CARBONBENCH_ROOT / "11_fm_unet_final",
    CARBONBENCH_ROOT / "09_fm_unet_ot_training",
    CARBONBENCH_ROOT / "01_fm_unet_training_baseline",
]

DATA_CONFIG = DataConfig(
    dataset="carbontracker", grid="latlon5.625", vertical_levels="l10",
    freq="6h", target_vars=["co2massmix"], forcing_vars=[],
)

BASE_CONFIG = GenerateConfig(
    n_samples=100, steps=21, refine_start=1.0, avg_over_levels=False,
    sampler="ictm",
    conditioning={"masking": True, "mask_source": "test", "mask_pattern": "vertical",
                  "masking_time": None, "t_threshold": 0.9,
                  "masking_method": "total_column_average_simple",
                  "obs_fraction": 0.3, "guidance_scale": 1.0,
                  "condition_one_timestep": True},
    sampler_params={"sigma_obs": 0.1, "spatial_smoothing_sigma": 0.0, "fresh_noise": True,
                    "r_max": 1.0, "r_schedule": "decreasing",
                    "n_inner_steps": 1, "inner_lr": 0.1},
    analyze_masking=False, noise_pattern=None, analyze_noise=False,
)

ABLATION_CONFIGS = {}

ABLATION_CONFIGS["unconditional"] = {"conditioning.masking": False, "sampler": None}
ABLATION_CONFIGS["best_flowdps"] = {"sampler": "flowdps", "sampler_params.sigma_obs": 0.1}
ABLATION_CONFIGS["best_sde"] = {
    "sampler": "sde", "sampler_params.sigma_obs": 0.1, "sampler_params.sigma_max": 0.3,
    "sampler_params.fresh_noise": True, "sampler_params.noise_schedule": "annealed",
    "sampler_params.n_corrector_steps": 0, "sampler_params.corrector_step_size": 0.01,
    "sampler_params.use_projection": True,
}
ABLATION_CONFIGS["best_fig"] = {
    "sampler": "fig", "sampler_params.sigma_obs": 0.1, "sampler_params.step_size_c": 10.0,
    "sampler_params.k_steps": 1, "sampler_params.noise_scale_w": 0.0,
    "sampler_params.skip_first_last": True,
}

for r in [0.1, 0.5, 1.0, 2.0, 5.0]:
    ABLATION_CONFIGS[f"ictm_rmax{r}"] = {"sampler_params.r_max": r}

for sched in ["constant", "decreasing", "increasing", "cosine"]:
    ABLATION_CONFIGS[f"ictm_sched_{sched}"] = {"sampler_params.r_schedule": sched}

for n in [1, 2, 3, 5, 10]:
    ABLATION_CONFIGS[f"ictm_inner{n}"] = {"sampler_params.n_inner_steps": n}

for s in [0.01, 0.1, 0.5, 1.0]:
    ABLATION_CONFIGS[f"ictm_sobs{s}"] = {"sampler_params.sigma_obs": s}

for st in [21, 51, 101]:
    ABLATION_CONFIGS[f"ictm_steps{st}"] = {"steps": st}

for lr in [0.01, 0.05, 0.1, 0.5]:
    ABLATION_CONFIGS[f"ictm_lr{lr}_inner5"] = {
        "sampler_params.n_inner_steps": 5, "sampler_params.inner_lr": lr,
    }

if __name__ == "__main__":
    AblationRunner(EXP_DIR, DATA_CONFIG, MODEL_DIRS).main_cli(BASE_CONFIG, ABLATION_CONFIGS)
