"""FlowDPS Ablation Experiment (Phase 6).

Sweeps sigma_obs, spatial_smoothing, step count, and fresh_noise.

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
    sampler="flowdps",
    conditioning={"masking": True, "mask_source": "test", "mask_pattern": "vertical",
                  "masking_time": None, "t_threshold": 0.9,
                  "masking_method": "total_column_average_simple",
                  "obs_fraction": 0.3, "guidance_scale": 1.0,
                  "condition_one_timestep": True},
    sampler_params={"sigma_obs": 0.1, "spatial_smoothing_sigma": 0.0, "fresh_noise": True},
    analyze_masking=False, noise_pattern=None, analyze_noise=False,
)

ABLATION_CONFIGS = {}

ABLATION_CONFIGS["unconditional"] = {"conditioning.masking": False, "sampler": None}
ABLATION_CONFIGS["best_dps"] = {
    "sampler": None, "conditioning.conditioning_mode": "guidance",
    "sampler_params.sigma_obs": 0.5, "sampler_params.spatial_smoothing_sigma": 2.0,
}

for s in [0.01, 0.1, 0.5, 1.0]:
    ABLATION_CONFIGS[f"flowdps_s{s}"] = {"sampler_params.sigma_obs": s}

for sm in [1.0, 2.0, 4.0]:
    ABLATION_CONFIGS[f"flowdps_s0.1_smooth{int(sm)}"] = {
        "sampler_params.sigma_obs": 0.1, "sampler_params.spatial_smoothing_sigma": sm,
    }

for st in [11, 21, 51]:
    ABLATION_CONFIGS[f"flowdps_s0.1_steps{st}"] = {"sampler_params.sigma_obs": 0.1, "steps": st}

ABLATION_CONFIGS["flowdps_s0.1_fixednoise"] = {
    "sampler_params.sigma_obs": 0.1, "sampler_params.fresh_noise": False,
}

if __name__ == "__main__":
    AblationRunner(EXP_DIR, DATA_CONFIG, MODEL_DIRS).main_cli(BASE_CONFIG, ABLATION_CONFIGS)
