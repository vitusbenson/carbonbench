"""Auto-regressive evaluation of the Phase 24 FM transport-prior model.

Rolls out the model over the test period by feeding predicted CO2 back as
conditioning at each timestep; wind fields come from CarbonTracker (GT external
forcing). Optionally re-initialises from GT every N steps (sliding-window mode).

Usage:
    python eval_autoregressive.py
    python eval_autoregressive.py --reinit-every 120   # ~1 month at 6h freq
    python eval_autoregressive.py --n-steps 200 --init-idx 0
"""

import argparse
import logging
from pathlib import Path

import xarray as xr

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.analyse import compute_score_df
from neural_transport.inference.generation import generate_autoregressive
from neural_transport.training.train import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"

TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]


def main():
    parser = argparse.ArgumentParser(description="Phase 24 auto-regressive eval")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Number of autoregressive steps (default: full split)")
    parser.add_argument("--init-idx", type=int, default=0)
    parser.add_argument("--reinit-every", type=int, default=None,
                        help="Sliding-window re-init frequency (in steps). Default: none → pure AR")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    mode_name = f"slidingwindow_{args.reinit_every}" if args.reinit_every else "autoregressive"
    out_dir = EXP_DIR / "singlestep" / "preds" / f"eval_{mode_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(EXP_DIR, ckpt=args.ckpt, device=args.device)
    logger.info("Loaded model from %s (ckpt=%s)", EXP_DIR, args.ckpt)

    data_cfg = DataConfig(
        target_vars=TARGET_VARS,
        forcing_vars=FORCING_VARS,
    )
    loader = InferenceDataLoader(data_cfg, Path(args.data_root) / args.split)
    n_steps = args.n_steps if args.n_steps is not None else len(loader) - args.init_idx - 1
    logger.info("Rolling out %d steps from init_idx=%d (reinit_every=%s)",
                n_steps, args.init_idx, args.reinit_every)

    preds = generate_autoregressive(
        model,
        loader,
        n_steps=n_steps,
        target_var=TARGET_VARS[0],
        init_idx=args.init_idx,
        reinit_every=args.reinit_every,
        device=args.device,
        verbose=True,
    )
    preds_path = out_dir / "preds.zarr"
    preds.to_zarr(preds_path, mode="w")
    logger.info("Saved predictions → %s", preds_path)

    # Ground truth over the same time window. compute_score_df needs airmass
    # from the GT side (predictions only carry co2massmix).
    gt_path = Path(args.data_root) / args.split / "carbontracker_latlon5.625_l10_6h.zarr"
    gt = xr.open_zarr(gt_path)[TARGET_VARS + ["airmass"]].sel(time=preds.time)

    # Score: dataset-level metrics as a pandas Series (RMSE / R² / mass error).
    score_dir = out_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    metrics = compute_score_df(gt, preds)
    metrics.to_csv(score_dir / "metrics.csv")
    logger.info("Scores → %s", score_dir / "metrics.csv")


if __name__ == "__main__":
    main()
