"""DPS Guidance Ablation Experiment (Phase 5c).

Sweeps guidance_scale x sigma_obs x spatial_smoothing_sigma x timing.

Usage:
    python run_ablation.py
    python run_ablation.py --device cuda
    python run_ablation.py --filter sigma
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
    conditioning={"masking": True, "mask_source": "test", "mask_pattern": "vertical",
                  "masking_time": None, "t_threshold": 0.9,
                  "masking_method": "total_column_average_simple",
                  "conditioning_mode": "guidance", "obs_fraction": 0.3,
                  "guidance_scale": 1.0, "condition_one_timestep": True},
    sampler_params={"sigma_obs": 1.0, "spatial_smoothing_sigma": 0.0},
    analyze_masking=False, noise_pattern=None, analyze_noise=False,
)

# --- Ablation configurations (only ~50 unique lines) ---
ABLATION_CONFIGS = {}

ABLATION_CONFIGS["unconditional"] = {"conditioning.masking": False}

for s in [2.0, 1.0, 0.5, 0.1, 0.05]:
    ABLATION_CONFIGS[f"sigma_{s}"] = {"sampler_params.sigma_obs": s}

for g in [0.1, 0.5, 1.0, 2.0, 5.0]:
    ABLATION_CONFIGS[f"scale_{g}_s0.5"] = {
        "conditioning.guidance_scale": g, "sampler_params.sigma_obs": 0.5,
    }

for sm in [0.0, 1.0, 2.0, 4.0]:
    ABLATION_CONFIGS[f"smooth_{sm}_s0.5"] = {
        "sampler_params.sigma_obs": 0.5, "sampler_params.spatial_smoothing_sigma": sm,
    }

ABLATION_CONFIGS["s0.5_sm2_full"] = {
    "sampler_params.sigma_obs": 0.5, "sampler_params.spatial_smoothing_sigma": 2.0,
    "conditioning.masking_time": None,
}
ABLATION_CONFIGS["s0.5_sm2_late08"] = {
    "sampler_params.sigma_obs": 0.5, "sampler_params.spatial_smoothing_sigma": 2.0,
    "conditioning.masking_time": "smooth_late_masking", "conditioning.t_threshold": 0.8,
}
ABLATION_CONFIGS["s0.5_sm2_late05"] = {
    "sampler_params.sigma_obs": 0.5, "sampler_params.spatial_smoothing_sigma": 2.0,
    "conditioning.masking_time": "smooth_late_masking", "conditioning.t_threshold": 0.5,
}
ABLATION_CONFIGS["s0.5_sm2_early08"] = {
    "sampler_params.sigma_obs": 0.5, "sampler_params.spatial_smoothing_sigma": 2.0,
    "conditioning.masking_time": "smooth_early_masking", "conditioning.t_threshold": 0.8,
}

if __name__ == "__main__":
    AblationRunner(EXP_DIR, DATA_CONFIG, MODEL_DIRS).main_cli(BASE_CONFIG, ABLATION_CONFIGS)
