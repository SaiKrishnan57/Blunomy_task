import pandas as pd

from lidar_catenary.pipeline import assign_wire_groups


def test_assign_wire_groups_merges_only_when_vertical_separation_is_clear() -> None:
    medium_like = pd.DataFrame(
        [
            {"cluster": 0, "y0": -0.72},
            {"cluster": 1, "y0": -0.70},
            {"cluster": 2, "y0": -0.49},
            {"cluster": 3, "y0": -0.51},
        ]
    )

    grouped = assign_wire_groups(medium_like)

    assert grouped["wire_group"].tolist() == ["-0.7", "-0.7", "-0.5", "-0.5"]


def test_assign_wire_groups_keeps_clusters_separate_when_y0_spread_is_small() -> None:
    hard_like = pd.DataFrame(
        [
            {"cluster": 0, "y0": -0.52},
            {"cluster": 1, "y0": -0.54},
            {"cluster": 2, "y0": -0.50},
        ]
    )

    grouped = assign_wire_groups(hard_like)

    assert grouped["wire_group"].tolist() == ["0", "1", "2"]

