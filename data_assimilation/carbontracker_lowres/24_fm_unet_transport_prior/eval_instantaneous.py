"""One-step distributional eval for the Phase 24 FM transport-prior model.

Thin wrapper around ``GenerationPipeline.run_distributional``: draws 20 random
test timesteps and 10 independent conditional samples per timestep, then writes
gt/gen pools and runs the standard distributional metrics downstream.
"""

import argparse
import logging
from pathlib import Path

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.generation import GenerationPipeline
from neural_transport.training.train import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"

TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]


def main():
    parser = argparse.ArgumentParser(description="Phase 24 one-step distributional eval")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--n-ref-timesteps", type=int, default=20)
    parser.add_argument("--n-gen-per-ref", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = EXP_DIR / "singlestep" / "preds" / "eval_instantaneous"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(EXP_DIR, ckpt=args.ckpt, device=args.device)
    data_cfg = DataConfig(target_vars=TARGET_VARS, forcing_vars=FORCING_VARS)
    loader = InferenceDataLoader(data_cfg, Path(args.data_root) / args.split)

    pipeline = GenerationPipeline(model, loader, target_vars_3d=TARGET_VARS, device=args.device)
    gt_ds, gen_ds = pipeline.run_distributional(
        out_dir,
        n_ref_timesteps=args.n_ref_timesteps,
        n_gen_per_ref=args.n_gen_per_ref,
        seed=args.seed,
    )
    logger.info("gt_pool=%s, gen_pool=%s → %s",
                dict(gt_ds.sizes), dict(gen_ds.sizes), out_dir)


if __name__ == "__main__":
    main()
