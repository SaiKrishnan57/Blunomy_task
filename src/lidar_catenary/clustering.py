import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_wires(point_cloud: pd.DataFrame, eps: float = 0.6, min_samples: int = 10) -> pd.DataFrame:
    """Cluster LiDAR points into candidate wire groups using a cross-sectional view."""
    points = point_cloud[["x", "y", "z"]].to_numpy()

    center = points.mean(axis=0)
    centered_points = points - center

    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    main_direction = vh[0]

    cross_section_axes = vh[1:]
    cross_section_points = centered_points @ cross_section_axes.T

    scaler = StandardScaler()
    scaled_cross_section_points = scaler.fit_transform(cross_section_points)

    print(f"Using DBSCAN with eps={eps}, min_samples={min_samples}")

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled_cross_section_points)

    clustered = point_cloud.copy()
    clustered["cluster"] = labels
    return clustered
