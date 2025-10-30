# %%
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import xarray as xr

from neural_transport.litmodule import NeuralTransport
from neural_transport.training.train import load_dataset

# %%
CKPT_PATH = "/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20250916_dev/singlestep/checkpoints/last.ckpt"
DATA_PATH = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
N_GPUS = 1
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_PRED = 16

# %%
model = NeuralTransport.load_from_checkpoint(
    CKPT_PATH,
    map_location="cpu",
)

# %%
# Setting Data Arguments
grid = "latlon5.625"
vertical_levels = "l10"
freq = "6h"
nlev = 10

data_kwargs = dict(
    data_path=DATA_PATH,
    dataset="carbontracker",
    grid=grid,
    vertical_levels=vertical_levels,
    freq=freq,
    n_timesteps=1,
    batch_size_train=BATCH_SIZE_TRAIN // N_GPUS,
    batch_size_pred=BATCH_SIZE_PRED,
    num_workers=32 * N_GPUS,
    val_rollout_n_timesteps=31,
    target_vars=[
        "co2massmix",
    ],
    forcing_vars=[],
    compute=False,
)
dset = load_dataset(Path(DATA_PATH) / "test", data_kwargs)

# %%
t = 0
batch = {k: v.unsqueeze(0).to("cuda") for k, v in dset[t].items()}

# %%
model.train()
model.to("cuda")
with torch.no_grad():
    preds = model(batch)

# %%
batch_normalized = {}
for v in ["co2massmix"]:
    for suffix in ["", "_next"]:
        key = f"{v}{suffix}"
        if key in batch:
            mean = batch[f"{v}_offset"]
            std = batch[f"{v}_scale"]
            x_in_curr = (batch[key] - mean) / std
            batch_normalized[key] = x_in_curr

# %%
# Get tensors and move to cpu
pred = preds["co2massmix"].squeeze().cpu().numpy()  # shape: nloc x nlev
truth = (
    preds["dx_t"].squeeze().cpu().numpy()
)  # shape: nloc x nlev

# Get grid shape from dset
nlat = 32
nlon = 64

# Select 3 levels to plot (e.g., bottom, middle, top)
levels = [0, nlev // 2, nlev - 1]
level_names = [f"Level {i}" for i in levels]

fig, axes = plt.subplots(len(levels), 2, figsize=(10, 3 * len(levels)))
for i, lev in enumerate(levels):
    pred_map = pred[:, lev].reshape(nlat, nlon)
    truth_map = truth[:, lev].reshape(nlat, nlon)
    # Prediction
    ax = axes[i, 0]
    im = ax.imshow(pred_map, origin="lower")
    ax.set_title(f"Prediction {level_names[i]}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Ground Truth
    ax = axes[i, 1]
    im = ax.imshow(truth_map, origin="lower")
    ax.set_title(f"Ground Truth {level_names[i]}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
# %%
model.eval()
model.to("cuda")
with torch.no_grad():
    preds = model(batch)
# %%
trajectory = (
    preds["trajectory"].squeeze().permute(3,1,2,0).cpu().numpy()
)  # shape: Nlev x Nlat x Nlon x Nsteps


# Plot a timeseries of maps of the trajectory at for a given level
level_to_plot = 0  # Choose a level to plot
nsteps = trajectory.shape[0]

# Create a clean flow visualization with two rows
ncols = nsteps // 2 if nsteps % 2 == 0 else (nsteps + 1) // 2
nrows = 2 if nsteps > ncols else 1

fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

# Handle different cases for axes indexing
if nrows == 1:
    if ncols == 1:
        axes = [axes]
    else:
        axes = axes.reshape(1, -1)
elif ncols == 1:
    axes = axes.reshape(-1, 1)

# Find global min/max for consistent colormap
vmin = trajectory[:, :, :, level_to_plot].min()
vmax = trajectory[:, :, :, level_to_plot].max()

for i in range(nsteps):
    row = i // ncols
    col = i % ncols

    if nrows == 1:
        ax = axes[col] if ncols > 1 else axes[0]
    else:
        ax = axes[row, col]

    im = ax.imshow(
        trajectory[i, :, :, level_to_plot],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )

    # Remove all axis elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Optional: add step number as text overlay
    ax.text(
        0.05,
        0.95,
        f"t={i}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

# Hide any unused subplots
if nsteps < nrows * ncols:
    for i in range(nsteps, nrows * ncols):
        row = i // ncols
        col = i % ncols
        if nrows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)

# Add horizontal colorbar below
plt.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

plt.tight_layout()
plt.show()

# %%
