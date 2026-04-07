import pandas as pd


def clean_point_cloud(point_cloud: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning to the LiDAR point cloud."""
    cleaned = point_cloud.dropna(subset=["x", "y", "z"])
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.reset_index(drop=True)
    return cleaned
