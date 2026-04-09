import numpy as np

from lidar_catenary.geometry import estimate_local_frame, project_to_local_2d


def test_project_to_local_2d_returns_expected_shape() -> None:
    points = np.array(
        [
            [-2.0, 0.0, 10.0],
            [-1.0, 0.0, 9.8],
            [0.0, 0.0, 9.7],
            [1.0, 0.0, 9.8],
            [2.0, 0.0, 10.0],
        ]
    )

    center, wire_direction, local_vertical = estimate_local_frame(points)
    local_points = project_to_local_2d(points, center, wire_direction, local_vertical)

    assert local_points.shape == (5, 2)
    assert np.isfinite(local_points).all()


def test_estimate_local_frame_fails_for_vertical_cluster() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )

    try:
        estimate_local_frame(points)
    except ValueError as error:
        assert "stable plane" in str(error)
    else:
        raise AssertionError("Expected estimate_local_frame to fail for a vertical cluster.")

