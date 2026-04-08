import numpy as np


def estimate_local_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a local coordinate frame for one wire cluster."""
    center = points.mean(axis=0)
    centered_points = points - center

    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)

    wire_direction = vh[0]
    vertical_direction = np.array([0.0, 0.0, 1.0])

    normal_direction = np.cross(wire_direction, vertical_direction)
    normal_norm = np.linalg.norm(normal_direction)

    if normal_norm < 1e-8:
        raise ValueError("Could not estimate a stable plane for this wire cluster.")

    normal_direction = normal_direction / normal_norm
    local_vertical = np.cross(normal_direction, wire_direction)
    local_vertical = local_vertical / np.linalg.norm(local_vertical)

    return center, wire_direction, local_vertical
