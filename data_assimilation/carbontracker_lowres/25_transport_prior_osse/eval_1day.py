"""Phase 25 Eval 1: instantaneous (1-step) posterior conditioning.

Uses the Phase 24 transport-prior FM model with FMPS / D-Flow posterior
samplers. Mirrors Phase 23f's comparison protocol so results are directly
comparable.

Usage:
    python eval_1day.py --method fmps
    python eval_1day.py --method dflow
    python eval_1day.py --method fmps --n-targets 4 --n-samples 2  # smoke
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.generation import generate_multi_target
from neural_transport.training.train import load_model

from configs import load_method_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
PHASE24_DIR = EXP_DIR.parent / "24_fm_unet_transport_prior"
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["fmps", "dflow", "unconditional"], default="fmps")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--n-targets", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--out-tag", default=None,
                   help="Optional suffix for out dir")
    args = p.parse_args()

    model = load_model(PHASE24_DIR, ckpt=args.ckpt, device=args.device)
    logger.info("Loaded Phase 24 model from %s", PHASE24_DIR)

    data_cfg = DataConfig(
        dataset="carbontracker",
        grid="latlon5.625",
        vertical_levels="l10",
        freq="6h",
        target_vars=["co2massmix", "p_bottom", "p_top"],
        forcing_vars=["co2massmix", "u", "v"],
    )
    loader = InferenceDataLoader(data_cfg, data_path=f"{args.data_root}/{args.split}")
    dataset = loader.load_dataset()

    rng = np.random.RandomState(args.seed)
    target_indices = sorted(rng.choice(len(dataset),
                                       size=min(args.n_targets, len(dataset)),
                                       replace=False).tolist())

    if args.method == "unconditional":
        from neural_transport.configs import compat_to_generate_kwargs
        from configs import base_generate_config
        cfg = base_generate_config(n_samples=args.n_samples).merge(**{"conditioning.masking": False})
        generate_kwargs = compat_to_generate_kwargs(cfg)
    else:
        cfg = load_method_config(args.method, n_samples=args.n_samples)
        from neural_transport.configs import compat_to_generate_kwargs
        generate_kwargs = compat_to_generate_kwargs(cfg)

    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_dir = EXP_DIR / "results" / f"{args.method}_1day{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    zarr_path = generate_multi_target(
        model=model,
        dataset=dataset,
        target_indices=target_indices,
        n_samples_per_target=args.n_samples,
        batch_size=args.batch_size,
        generate_kwargs=generate_kwargs,
        out_dir=out_dir,
        device=args.device,
        target_var="co2massmix",
        verbose=True,
        seed=args.seed,
    )
    wall = time.perf_counter() - t0

    info = {
        "eval": "1day",
        "phase": 25,
        "method": args.method,
        "wall_time_sec": wall,
        "n_targets": len(target_indices),
        "n_samples_per_target": args.n_samples,
        "ckpt": args.ckpt,
        "target_indices": target_indices,
    }
    with open(out_dir / "method_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    logger.info("Done (%.1fs) → %s", wall, zarr_path)


if __name__ == "__main__":
    main()
