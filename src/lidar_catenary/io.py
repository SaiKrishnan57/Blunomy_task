from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["x", "y", "z"]


def load_point_cloud(file_path: str | Path) -> pd.DataFrame:
    """Load a parquet point cloud and return only the x, y, z columns."""
    path = Path(file_path)
    point_cloud = pd.read_parquet(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in point_cloud.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return point_cloud[REQUIRED_COLUMNS].copy()
