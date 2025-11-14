"""Create a MIP OCO-2 dataset by downloading, filtering, regridding, writing and computing statistics."""

from neural_transport.datasets.mip_oco2 import (
    download_data,
    filter_mip_oco2,
    regrid_mip_oco2,
    stats_mip_oco2,
    write_mip_oco2,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gridname", type=str, default="latlon1x1")
    parser.add_argument("--vertical_levels", type=str, default="l34")
    parser.add_argument("--freq", type=str, default="3h")
    args = parser.parse_args()

    download_data(args.save_dir)


    filter_mip_oco2(args.save_dir)


    regrid_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )


    write_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )


    stats_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )