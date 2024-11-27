from neural_transport.datasets.obspack import download_obspack, prepare_obspack_for_carboscope

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--freq", type=str, default="3h")
    args = parser.parse_args()

    FREQ = args.freq

    download_obspack(args.save_dir)

    prepare_obspack_for_carboscope(
        args.save_dir
    )
