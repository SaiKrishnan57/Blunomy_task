import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lidar_catenary.io import load_point_cloud
from lidar_catenary.preprocess import clean_point_cloud
from lidar_catenary.clustering import cluster_wires


file_path = r"C:\Users\saikr\OneDrive\Desktop\Projects\Blunomy\Data Science - LiDAR Technical Test\lidar_cable_points_medium.parquet"

point_cloud = load_point_cloud(file_path)
cleaned_point_cloud = clean_point_cloud(point_cloud)
clustered_point_cloud = cluster_wires(cleaned_point_cloud)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for cluster_label in sorted(clustered_point_cloud["cluster"].unique()):
    cluster_df = clustered_point_cloud[clustered_point_cloud["cluster"] == cluster_label]

    if cluster_label == -1:
        ax.scatter(
            cluster_df["x"],
            cluster_df["y"],
            cluster_df["z"],
            s=10,
            c="gray",
            alpha=0.4,
            label="noise",
        )
    else:
        ax.scatter(
            cluster_df["x"],
            cluster_df["y"],
            cluster_df["z"],
            s=12,
            alpha=0.8,
            label=f"cluster {cluster_label}",
        )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Clustered LiDAR Points")
ax.legend()
plt.tight_layout()
plt.show()
