"""Freeze EMA shadow weights into a standalone checkpoint.

Rollout-FT (phase1p) trains with an EMACallback that stores `ema_state_dict`
in the checkpoint. The residual-FM stage (phase2p) loads its `f_det` via
`NeuralTransport.load_from_checkpoint`, which reads the plain `state_dict`.
This writes a new checkpoint whose `state_dict` IS the EMA-averaged weights,
so phase2p picks up the EMA backbone. (Recreates the archived
tools/freeze_ema_ckpt.py described in Phase 25p.A2.)

Usage:
    python freeze_ema_ckpt.py --in <rollout_ft last.ckpt> --out <ema_frozen.ckpt>
"""

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    blob = torch.load(args.inp, map_location="cpu", weights_only=False)
    ema_sd = blob.get("ema_state_dict")
    if ema_sd is None:
        raise SystemExit(f"No ema_state_dict in {args.inp} — cannot freeze EMA weights.")
    blob["state_dict"] = ema_sd
    torch.save(blob, args.out)
    print(f"Wrote EMA-frozen checkpoint: {args.out}")


if __name__ == "__main__":
    main()
