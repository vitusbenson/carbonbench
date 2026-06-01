"""P2: prepare a LEAK-FREE CarbonTracker split for the FM-DA-vs-OCO2-MIP study.

The SOTA model was trained on the default split (train 2000-2016, val 2017,
test 2018-2020), which overlaps the OCO-2 MIP period (2015-2020) — leakage that
breaks the independent-method claim. This script re-slices the *existing* full
regridded zarr (no raw regridding) into a leak-free split:

    train = 2000-2013   val = 2014   test = 2015-2020 (the full MIP period)

Normalization stats are recomputed on the <=2013 train ONLY (stats leakage
matters too). Output layout matches the loader's expectation:

    <out_root>/{train,val,test}/carbontracker_<grid>_<vlev>_<freq>.zarr
    <out_root>/{train,val,test}/carbontracker_<grid>_<vlev>_<freq>_stats.zarr

Usage:
    python prepare_leakfree_data.py \
        --regrid-zarr /Net/.../Carbontracker/CT2022_regrid/CT2022_regrid_latlon5.625_l10_6h.zarr \
        --out-root    /Net/.../Carbontracker_leakfree
"""

import argparse
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from neural_transport.datasets.common import compute_stats, optimize_zarr

SPLITS = {
    "train": slice(None, "2013-12-31"),
    "val": slice("2014-01-01", "2014-12-31"),
    "test": slice("2015-01-01", "2020-12-31"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regrid-zarr", required=True, type=str)
    p.add_argument("--out-root", required=True, type=str)
    p.add_argument("--grid", default="latlon5.625")
    p.add_argument("--vertical-levels", default="l10")
    p.add_argument("--freq", default="6h")
    args = p.parse_args()

    out_root = Path(args.out_root)
    name = f"carbontracker_{args.grid}_{args.vertical_levels}_{args.freq}"
    ds_full = xr.open_zarr(args.regrid_zarr)
    print(f"Full regrid: {str(ds_full.time.values.min())[:10]} -> {str(ds_full.time.values.max())[:10]} "
          f"(n={ds_full.sizes['time']})")

    # --- write the three splits ---
    for split, tslice in SPLITS.items():
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_split = ds_full.sel(time=tslice)
        t = ds_split.time.values
        print(f"\n[{split}] {str(t.min())[:10]} -> {str(t.max())[:10]} (n={len(t)})  -> {out_dir / (name + '.zarr')}")
        ds_opt = optimize_zarr(ds_split)
        with ProgressBar():
            ds_opt.to_zarr(out_dir / f"{name}.zarr", mode="w")

    # --- stats from the LEAK-FREE TRAIN ONLY, written to all three splits ---
    print("\nComputing stats on the <=2013 train split only (no leakage)...")
    ds_train = xr.open_zarr(out_root / "train" / f"{name}.zarr")
    ds_stats = compute_stats(ds_train)
    for split in SPLITS:
        ds_stats.to_zarr(out_root / split / f"{name}_stats.zarr", mode="w")
        print(f"  wrote {split}/{name}_stats.zarr")
    print("\nLeak-free CarbonTracker preparation complete.")


if __name__ == "__main__":
    main()
