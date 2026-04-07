import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_wires(point_cloud: pd.DataFrame, eps: float = 0.3, min_samples: int = 10) -> pd.DataFrame:
    """Cluster LiDAR points into candidate wire groups using DBSCAN."""
    features = point_cloud[["x", "y", "z"]].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled_features)

    clustered = point_cloud.copy()
    clustered["cluster"] = labels
    return clustered
