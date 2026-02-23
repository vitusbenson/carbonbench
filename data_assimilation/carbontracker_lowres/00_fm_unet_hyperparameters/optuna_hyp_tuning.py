"""Tune hyperparameters using Optuna."""

import copy
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis
import optuna.visualization.matplotlib as vis_mat
import pandas as pd
from train import MODEL_DIMS, data_kwargs, lit_module_kwargs, main, trainer_kwargs, wrapper_kwargs

# ---------------------------
# Get loss from CSV logs
# ---------------------------

def read_crps_from_scores(trial_logdir: Path) -> float:
    """Read CRPS_ensemble_mean from CSV log file."""
    scalar = trial_logdir.glob("**/scores/**/metrics_global_scalars.csv")
    scalar_files = list(scalar)
    df = pd.read_csv(scalar_files[-1])
    if "CRPS_ensemble_mean" in df.columns:
        return float(df["CRPS_ensemble_mean"].iloc[-1])
    raise ValueError(f"CRPS_ensemble_mean not found in {scalar_files[-1]}")

# ---------------------------
# Objective function
# ---------------------------
def objective(trial: optuna.trial.Trial) -> float:
    """Hyperparameter optimization objective function."""
    # ---------------------------
    # Per-trial copies of kwargs
    # ---------------------------
    trial_wrapper_kwargs = copy.deepcopy(wrapper_kwargs)
    trial_lit_module_kwargs = copy.deepcopy(lit_module_kwargs)
    trial_trainer_kwargs = copy.deepcopy(trainer_kwargs)
    trial_data_kwargs = copy.deepcopy(data_kwargs)

    # ---------------------------
    # Sample hyperparameters
    # ---------------------------
    model_size = trial.suggest_categorical("MODEL_SIZE", ["S", "M", "L"])
    step_size = trial.suggest_categorical("step_size", [0.05, 0.1, 0.2])
    method = trial.suggest_categorical("method", ["midpoint", "euler"])
    batch_size_train = trial.suggest_categorical("batch_size_train", [32, 64, 128])

    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1])
    warmup_steps = trial.suggest_int("warmup_steps", 500, 5000, step=500)
    halfcosine_steps = trial.suggest_int("halfcosine_steps", 10000, 100000, step=10000)
    min_lr = trial.suggest_float("min_lr", 1e-7, 1e-5, log=True)
    max_lr = trial.suggest_float("max_lr", 0.1, 1.0, log=True)

    # ---------------------------
    # Apply hyperparameters
    # ---------------------------

    unet_config = trial_wrapper_kwargs["model_kwargs"]["model_kwargs"]["model_kwargs"]
    unet_config["embed_dim"] = MODEL_DIMS[model_size]["embed_dim"]
    trial_wrapper_kwargs["model_kwargs"]["step_size"] = step_size
    trial_wrapper_kwargs["model_kwargs"]["method"] = method

    trial_lit_module_kwargs["lr"] = lr
    trial_lit_module_kwargs["weight_decay"] = weight_decay
    trial_lit_module_kwargs["lr_shedule_kwargs"] = dict(
        warmup_steps=warmup_steps,
        halfcosine_steps=halfcosine_steps,
        min_lr=min_lr,
        max_lr=max_lr,
    )

    trial_trainer_kwargs["max_steps"] = 2000  # 2000 shorter for tuning
    trial_data_kwargs["batch_size_train"] = batch_size_train // 1  # N_GPUS = 1

    # ---------------------------
    # Per-trial logdir
    # ---------------------------
    run_dir = Path(__file__).resolve().parent
    trial_logdir = run_dir / f"trial_{trial.number}"
    if trial_logdir.exists():
        shutil.rmtree(trial_logdir)
    trial_logdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Run training
    # ---------------------------
    try:
        main(
            rollout=False,
            train=True,
            training_data_root=None,
            masking_data_root=None,
            forecast_data_root=None,
            data_kwargs=trial_data_kwargs,
            lit_module_kwargs=trial_lit_module_kwargs,
            trainer_kwargs=trial_trainer_kwargs,
            wrapper_kwargs=trial_wrapper_kwargs,
            run_dir=trial_logdir,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")

    # ---------------------------
    # Extract validation loss from TensorBoard
    # ---------------------------
    return read_crps_from_scores(trial_logdir)

# ---------------------------
# Run Optuna study
# ---------------------------
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)  # track progress
    out_dir = Path(__file__).resolve().parent
    db_path = out_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="minimize",
        study_name="flowmatching_20251029_2_hyperparameters_dev",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best CRPS:", study.best_value)

    # ---------------------------
    # Visualize results
    # ---------------------------
    fig1 = vis.plot_param_importances(study)
    fig2 = vis.plot_optimization_history(study)
    try:
        fig1.write_image(out_dir / "optuna_param_importances.png")
        fig2.write_image(out_dir / "optuna_optimization_history.png")
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"Could not write static images ({e}), falling back to HTML.")
        fig1.write_html(out_dir / "optuna_param_importances.html")
        fig2.write_html(out_dir / "optuna_optimization_history.html")
    fig3 = vis_mat.plot_param_importances(study).figure
    fig4 = vis_mat.plot_optimization_history(study).figure
    fig3.savefig(out_dir / "optuna_param_importances_matplotlib.png")
    fig4.savefig(out_dir / "optuna_optimization_history_matplotlib.png")
    plt.close(fig3)
    plt.close(fig4)
