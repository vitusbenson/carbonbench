"""Surgically expand the phase-2p residual-FM checkpoint for the amortized model.

The amortized UNet has 2 extra input channels (obs_value, obs_mask) inserted
between det_pred and the time channel, so in_chans 51 -> 53 and the time channel
moves from index 50 to 52. We remap the first conv accordingly (copy the 50
dynamics/conditioning channels, move the time channel, zero-init the 2 new obs
channels) so fine-tuning starts from the trained residual-FM instead of scratch.
"""

import argparse
import glob
from pathlib import Path

import torch

FIRST_CONV = "model.submodel.enc_stages.0.0.conv.weight"
PHASE2P = (
    "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/"
    "carbontracker_lowres/25c_v4_residual_fm/phase2p_residual_fm_stable_leakfree/"
    "singlestep/checkpoints"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="source .ckpt (default: phase2p best Epoch*)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "init_amortized.ckpt"))
    args = ap.parse_args()

    src = args.src or sorted(glob.glob(f"{PHASE2P}/Epoch*.ckpt"))[0]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    w = sd[FIRST_CONV]  # [out, 51, k, k]
    out_c, in_c, kh, kw = w.shape
    assert in_c == 51, f"expected 51 in-chans, got {in_c}"

    new = torch.zeros(out_c, 53, kh, kw, dtype=w.dtype)
    new[:, 0:50] = w[:, 0:50]   # x_t(10)+co2(10)+u(10)+v(10)+det(10)
    new[:, 52:53] = w[:, 50:51]  # time channel moves 50 -> 52
    # channels 50,51 (obs_value, obs_mask) left at zero -> no effect at init.
    sd[FIRST_CONV] = new

    # Keep only model.* and strip the lightning wrapper bits the loader ignores.
    torch.save({"state_dict": sd}, args.out)
    print(f"src={src}")
    print(f"wrote {args.out}: {FIRST_CONV} {tuple(w.shape)} -> {tuple(new.shape)}")


if __name__ == "__main__":
    main()
