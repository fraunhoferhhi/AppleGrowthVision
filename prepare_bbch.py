from pathlib import Path
import argparse

BBCH_MAPPING = {
    1: [
        "brandenburg/data/2023-03-22",
    ],
    3: [
        "brandenburg/data/2022-03-11",
        "brandenburg/data/2022-03-25",
    ],
    5: [
        "brandenburg/data/2022-04-01",
        "brandenburg/data/2022-04-08",
        "brandenburg/data/2023-04-21",
        "brandenburg/data/2022-04-29",
    ],
    6: [
        "brandenburg/data/2023-05-04",
        "saxony/data/Pillnitz2023/20230504_*.jpg",
        "brandenburg/data/2022-05-06",
        "brandenburg/data/2022-05-13",
        "brandenburg/data/2023-05-16",
    ],
    7: [
        "saxony/data/Pillnitz2023/20230524_*.jpg",
        "brandenburg/data/2022-05-30",
        "brandenburg/data/2022-06-13",
        "brandenburg/data/2022-06-27",
        "brandenburg/data/2022-08-03",
    ],
    8: [
        "saxony/data/Pillnitz2023/20230823_*.jpg",
        "saxony/data/Pillnitz2024/20240826_*.jpg",
        "brandenburg/data/2022-09-06",
        "saxony/data/Pillnitz2024/20230919_*.jpg",
    ],
    9: [
        "branderburg/data/2022-11-02",
    ]
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare BBCH data.")
    parser.add_argument(
        "-p",
        type=Path,
        help="Path to the top directory of AppleGrowthVision."
    )
    parser.add_argument(
        "-o",
        type=Path,
        help="Path to the output directory where processed data will be saved."
    )
    args = parser.parse_args()

    assert args.p is not None, "Please provide the path to the top directory of AppleGrowthVision."
    assert args.o is not None, "Please provide the output directory path."
    assert args.p.exists(), f"The provided path {args.p} does not exist."

    output_dir = args.o / "bbch"
    output_dir.mkdir(parents=True, exist_ok=True)

    for bbch, paths in BBCH_MAPPING.items():
        output_dir_bbch = output_dir / str(bbch)

        for f in Path(paths):
            f = args.p / f
            if f.exists():
                if f.is_dir():
                    for sub_f in f.glob("*.JPG"):
                        sub_f.rename(output_dir_bbch / sub_f.name)
                else:
                    for sub_f in f.parent.glob(f.name):
                        sub_f.rename(output_dir_bbch / sub_f.name)
            else:
                print(f"Warning: {f} does not exist.")