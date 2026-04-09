import numpy as np

from lidar_catenary.catenary import catenary_curve, fit_catenary


def test_fit_catenary_recovers_parameters_from_synthetic_data() -> None:
    u = np.linspace(-5.0, 5.0, 50)
    expected_x0 = 0.5
    expected_y0 = -1.2
    expected_c = 8.0
    v = catenary_curve(u, expected_x0, expected_y0, expected_c)
    local_points = np.column_stack([u, v])

    x0, y0, c = fit_catenary(local_points)

    assert abs(x0 - expected_x0) < 1e-2
    assert abs(y0 - expected_y0) < 1e-2
    assert abs(c - expected_c) < 1e-2

