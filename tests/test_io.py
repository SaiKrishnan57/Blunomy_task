from pathlib import Path

import pandas as pd
import pytest

from lidar_catenary.io import load_point_cloud


def test_load_point_cloud_keeps_only_required_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
            "extra": [7, 8],
        }
    )
    path = tmp_path / "sample.parquet"
    frame.to_parquet(path)

    loaded = load_point_cloud(path)

    assert list(loaded.columns) == ["x", "y", "z"]
    assert loaded.shape == (2, 3)


def test_load_point_cloud_raises_when_required_columns_missing(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.0], "y": [2.0]})
    path = tmp_path / "invalid.parquet"
    frame.to_parquet(path)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_point_cloud(path)

