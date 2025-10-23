import copy
import optuna
from pathlib import Path
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import optuna.visualization as vis
import optuna.visualization.matplotlib as vis_mat
import matplotlib.pyplot as plt

from train import main, wrapper_kwargs, lit_module_kwargs, trainer_kwargs, data_kwargs, MODEL_DIMS

# ---------------------------
# TensorBoard helper
# ---------------------------
def get_last_val_loss(logdir: str, metric_name="Loss/Val_singlestep"):
    logdir = Path(logdir)
    event_files = list(logdir.glob("**/events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {logdir}")

    ea = EventAccumulator(str(event_files[-1]))
    ea.Reload()

    if metric_name not in ea.Tags().get("scalars", []):
        raise ValueError(f"Metric {metric_name} not found in TensorBoard logs")

    vals = ea.Scalars(metric_name)
    if not vals:
        raise ValueError(f"No values found for {metric_name} in {logdir}")

    val_str = vals[-1].value
    try:
        return float(val_str)
    except ValueError:
        return float(str(val_str).split("-")[0])

# ---------------------------
# Objective function
# ---------------------------
def objective(trial):
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
    trial_wrapper_kwargs["model_kwargs"]["model_kwargs"]["model_kwargs"]["embed_dim"] = MODEL_DIMS[model_size]["embed_dim"]
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

    trial_trainer_kwargs["max_steps"] = 2000  # shorter for tuning
    trial_data_kwargs["batch_size_train"] = batch_size_train // 1  # N_GPUS = 1

    # ---------------------------
    # Per-trial logdir
    # ---------------------------
    trial_logdir = Path(f"logs/trial_{trial.number}")
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
            data_path=None,
            wrapper_kwargs=trial_wrapper_kwargs,
            lit_module_kwargs=trial_lit_module_kwargs,
            trainer_kwargs=trial_trainer_kwargs,
            data_kwargs=trial_data_kwargs,
            run_dir=trial_logdir,
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")

    # ---------------------------
    # Extract validation loss from TensorBoard
    # ---------------------------
    return get_last_val_loss(trial_logdir)

# ---------------------------
# Run Optuna study
# ---------------------------
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)  # track progress

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)

    # ---------------------------
    # Visualize results
    # ---------------------------
    fig1 = vis.plot_param_importances(study)
    fig2 = vis.plot_optimization_history(study)
    try:
        fig1.write_image("optuna_param_importances.png")
        fig2.write_image("optuna_optimization_history.png")
    except Exception:
        fig1.write_html("optuna_param_importances.html")
        fig2.write_html("optuna_optimization_history.html")
    fig3 = vis_mat.plot_param_importances(study).figure
    fig4 = vis_mat.plot_optimization_history(study).figure
    fig3.savefig("optuna_param_importances_matplotlib.png")
    fig4.savefig("optuna_optimization_history_matplotlib.png")
    plt.close(fig3)
    plt.close(fig4)
