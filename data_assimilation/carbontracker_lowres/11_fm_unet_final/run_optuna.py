"""Optuna hyperparameter sweep for Flow Matching models.

Searches over optimizer, scheduler, architecture size, normalization,
OT coupling, and FM-specific training parameters. Uses a composite
objective (energy_distance + stability_penalty) with median pruning.

Usage:
    python run_optuna.py                                    # Full sweep (7 trials per worker)
    python run_optuna.py --smoke                            # Smoke test (2 trials, 50 steps)
    python run_optuna.py --n-trials 10 --max-steps 5000
    python run_optuna.py --analyze-only                     # Post-hoc analysis only
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.training.tuning import MODEL_SIZES, run_optuna_study

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"

TARGET_VARS = ["co2massmix"]
GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"

EXP_DIR = Path(__file__).resolve().parent


def build_base_configs(data_root, max_steps=3000, batch_size=512):
    """Build base configurations. Optuna overrides model_size, lr, etc."""
    nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[VERTICAL_LEVELS]["level"])
    lat = LATLON_PROTOTYPE_COORDS[GRID]["lat"]
    lon = LATLON_PROTOTYPE_COORDS[GRID]["lon"]

    LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)

    cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
    cos_lat = cos_lat / np.mean(cos_lat)
    METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}

    # Default model size (S) — Optuna will override
    default_size = MODEL_SIZES["S"]

    regulargrid_kwargs = dict(
        input_vars=TARGET_VARS,
        target_vars=TARGET_VARS,
        nlat=len(lat),
        nlon=len(lon),
        predict_delta=False,
        add_surfflux=False,
        dt=60 * 60 * 6,
        massfixer="",
        targshift=True,
    )

    wrapper_kwargs = dict(
        **regulargrid_kwargs,
        model_kwargs=dict(
            submodel="unet",
            model_kwargs=dict(
                **regulargrid_kwargs,
                model_kwargs=dict(
                    in_chans=LEN_ALL_TARGET_VARS + 1,
                    out_chans=LEN_ALL_TARGET_VARS,
                    embed_dim=default_size["embed_dim"],
                    act="leakyrelu",
                    norm="group",
                    enc_filters=default_size["enc_filters"],
                    dec_filters=default_size["dec_filters"],
                    in_interpolation="bilinear",
                    out_interpolation="nearest-exact",
                    out_clip=None,
                ),
            ),
            generating=True,
            return_intermediates=True,
            method="midpoint",
            nlev=nlev,
            step_size=0.2,
            use_ot_coupling=False,
            time_sampling="uniform",
            time_sampling_kwargs=None,
            time_loss_weight=None,
        ),
    )

    lit_module_kwargs = dict(
        model="flowmatching",
        model_kwargs=wrapper_kwargs,
        loss="flowmatching_mse",
        loss_kwargs=dict(),
        metrics=[
            dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
            for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
        ],
        no_grad_step_shedule=None,
        lr=3e-3,
        weight_decay=0.0,
        lr_shedule_kwargs=dict(
            warmup_steps=500,
            halfcosine_steps=10000,
            min_lr=1e-7,
            max_lr=0.8,
        ),
        val_dataloader_names=["singlestep"],
        plot_kwargs=dict(
            variables=["co2molemix"],
            layer_idxs=[0, 3, 5, 8],
            n_samples=2,
            dataset="carbontracker",
            grid=GRID,
            vertical_levels=VERTICAL_LEVELS,
            max_workers=32,
        ),
    )

    data_kwargs = dict(
        data_path=data_root,
        dataset="carbontracker",
        grid=GRID,
        vertical_levels=VERTICAL_LEVELS,
        freq=FREQ,
        n_timesteps=1,
        batch_size_train=batch_size,
        batch_size_pred=32,
        num_workers=0,
        val_rollout_n_timesteps=None,
        target_vars=["co2massmix"],
        forcing_vars=[],
        compute=True,
    )

    trainer_kwargs = dict(
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=100,
        gradient_clip_val=32,
        precision="bf16-mixed",
        strategy="auto",
    )

    return lit_module_kwargs, data_kwargs, trainer_kwargs


def main():
    parser = argparse.ArgumentParser(description="Optuna HP sweep for Flow Matching")
    parser.add_argument("--n-trials", type=int, default=7,
                        help="Trials per worker (default 7; 32 workers × 7 = 224 total)")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--study-name", type=str, default="fm_tuning")
    parser.add_argument("--training-data-root", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="2 trials, 50 steps")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip trials, just run analysis on existing study")
    args = parser.parse_args()

    if args.smoke:
        args.n_trials = 2
        args.max_steps = 50

    run_dir = EXP_DIR / "optuna_runs"
    storage = f"sqlite:///{EXP_DIR / 'optuna_study.db'}"

    if args.analyze_only:
        import optuna
        from neural_transport.training.study_analysis import analyze_study

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

        analysis_dir = EXP_DIR / "analysis"
        analyze_study(study, out_dir=analysis_dir, run_dir=run_dir)
        logger.info("Analysis complete. Best value: %.6f", study.best_value)
        logger.info("Best params: %s", study.best_params)
        return

    data_root = args.training_data_root or DEFAULT_DATA_ROOT

    lit_module_kwargs, data_kwargs, trainer_kwargs = build_base_configs(
        data_root, max_steps=args.max_steps, batch_size=args.batch_size,
    )

    import os
    worker_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

    logger.info(
        "Starting Optuna worker %s: %d trials, %d steps, batch=%d",
        worker_id, args.n_trials, args.max_steps, args.batch_size,
    )

    # Pre-load data once — shared across all trials
    from neural_transport.datamodule import CarbonDataModule
    dm = CarbonDataModule(**data_kwargs)
    dm.setup("fit")
    val_dataset = dm.val_dataset if hasattr(dm, "val_dataset") else None

    study = run_optuna_study(
        study_name=args.study_name,
        storage=storage,
        base_data_kwargs=data_kwargs,
        base_lit_module_kwargs=lit_module_kwargs,
        base_trainer_kwargs=trainer_kwargs,
        run_dir=run_dir,
        n_trials=args.n_trials,
        val_dataset=val_dataset,
        gen_eval_kwargs=dict(eval_every_n_epochs=1, n_gt_samples=5, n_gen_samples=20),
        ema_kwargs=dict(decay=0.9999, ema_start_step=1000),
        shared_datamodule=dm,
        model_sizes=["XS", "S", "M"],  # L is too large for A40 with batch_size=512
    )

    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    logger.info("Worker %s done. %d complete trials. Best: %.6f",
                worker_id, n_complete, study.best_value)


if __name__ == "__main__":
    main()
