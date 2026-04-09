import pandas as pd

from lidar_catenary.preprocess import clean_point_cloud


def test_clean_point_cloud_removes_nulls_and_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 4.0, "y": None, "z": 6.0},
            {"x": 7.0, "y": 8.0, "z": 9.0},
        ]
    )

    cleaned = clean_point_cloud(frame)

    assert len(cleaned) == 2
    assert cleaned.to_dict(orient="records") == [
        {"x": 1.0, "y": 2.0, "z": 3.0},
        {"x": 7.0, "y": 8.0, "z": 9.0},
    ]

