import argparse

from lidar_catenary.pipeline import run_pipeline


def main() -> None:
    """Command-line entry point for the LiDAR catenary pipeline."""
    parser = argparse.ArgumentParser(
        description="Detect wires in a LiDAR point cloud and fit catenary models."
    )
    parser.add_argument(
        "file_path",
        help="Path to the input parquet file.",
    )

    args = parser.parse_args()
    results = run_pipeline(args.file_path)

    if results.empty:
        print("No valid wire models were detected.")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
