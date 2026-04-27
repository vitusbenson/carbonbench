"""Load Phase 23f best-Optuna configs for FMPS and D-Flow and adapt them
for the Phase 24 transport-prior FM model.

Most sampler hyperparameters carry over since the posterior samplers use the
velocity model as a black box. We keep the Phase 23f tuned values by default
but expose ``n_opt_steps`` / ``n_samples`` / ``use_checkpointing`` for
memory-bounded long trajectory rollouts.
"""

from __future__ import annotations

import json
from pathlib import Path

from neural_transport.configs import GenerateConfig, compat_to_generate_kwargs
from neural_transport.inference.tuning import (
    _build_generate_config,
    _reconstruct_method_params,
)

PHASE23_ROOT = Path(__file__).resolve().parent.parent / "13_posterior_conditioning_ablation"


def base_generate_config(n_samples=20, steps=21) -> GenerateConfig:
    return GenerateConfig(
        n_samples=n_samples,
        steps=steps,
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


def load_method_config(method: str, *, n_samples=20, steps=21) -> GenerateConfig:
    """Load Phase 23f best Optuna config and splice onto a Phase 24-ready base."""
    summary_path = PHASE23_ROOT / "optuna_runs" / method / "study_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No phase-23 study summary at {summary_path}")
    summary = json.loads(summary_path.read_text())
    params = _reconstruct_method_params(method, summary["best_params"])
    base = base_generate_config(n_samples=n_samples, steps=steps)
    return _build_generate_config(params, base)


def adapt_for_trajectory(
    cfg: GenerateConfig,
    *,
    n_samples: int,
    method: str,
    n_opt_steps: int | None = None,
) -> dict:
    """Return a generate_kwargs dict trimmed for trajectory-level posterior use."""
    cfg = cfg.merge(**{"n_samples": n_samples})
    if method == "dflow":
        if n_opt_steps is not None:
            cfg = cfg.merge(**{"sampler_params.n_opt_steps": n_opt_steps})
        cfg = cfg.merge(**{"sampler_params.use_checkpointing": True})
    return compat_to_generate_kwargs(cfg)


def free_kwargs_from_obs_kwargs(obs_kwargs: dict) -> dict:
    """Strip sampler/masking fields so the unconditional path is clean."""
    out = dict(obs_kwargs)
    for k in (
        "sampler",
        "masking",
        "mask_source",
        "mask_pattern",
        "obs_fraction",
        "ak_10",
        "soft_boundary_sigma",
    ):
        out.pop(k, None)
    out["masking"] = False
    return out
