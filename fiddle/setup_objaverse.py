import argparse

import objaverse

def download_small_set(num_objects: int) -> None:
    uids = objaverse.load_uids()
    objaverse.load_objects(uids[:num_objects])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Flag used to download a set of random objects from Objaverse",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Size of the random sample of objects to be downloaded",
    )

    args = parser.parse_args()

    if args.download:
        download_small_set(args.num)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
