from neural_transport.datasets.carbontracker import (
    download_data,
    obspack_carbontracker,
    regrid_carbontracker,
    resample_carbontracker,
    stats_carbontracker,
    write_carbontracker,
)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gridname", type=str, default="latlon2x3")
    parser.add_argument("--vertical_levels", type=str, default="l34")
    parser.add_argument("--freq", type=str, default="3h")
    args = parser.parse_args()

    download_data(args.save_dir)

    regrid_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
    )

    resample_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )

    write_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )

    stats_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )

    obspack_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )
