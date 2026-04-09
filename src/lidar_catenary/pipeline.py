import pandas as pd

from lidar_catenary.io import load_point_cloud
from lidar_catenary.preprocess import clean_point_cloud
from lidar_catenary.clustering import cluster_wires
from lidar_catenary.geometry import estimate_local_frame, project_to_local_2d
from lidar_catenary.catenary import fit_catenary


def assign_wire_groups(results_df: pd.DataFrame, y0_threshold: float = 0.1) -> pd.DataFrame:
    """Assign wire groups only when fitted vertical positions show clear separation."""
    if results_df.empty:
        return results_df

    y0_spread = results_df["y0"].max() - results_df["y0"].min()
    grouped = results_df.copy()

    if y0_spread < y0_threshold:
        grouped["wire_group"] = grouped["cluster"].astype(str)
        return grouped

    grouped["wire_group"] = grouped["y0"].round(1).astype(str)
    return grouped


def run_pipeline(file_path: str) -> pd.DataFrame:
    """Run the full wire detection and catenary fitting pipeline on one parquet file."""
    point_cloud = load_point_cloud(file_path)
    cleaned_point_cloud = clean_point_cloud(point_cloud)
    clustered_point_cloud = cluster_wires(cleaned_point_cloud)
    results = []

    for cluster_label in sorted(clustered_point_cloud["cluster"].unique()):
        if cluster_label == -1:
            continue

        cluster_points_df = clustered_point_cloud[clustered_point_cloud["cluster"] == cluster_label]
        if len(cluster_points_df) < 5:
            continue

        cluster_points = cluster_points_df[["x", "y", "z"]].to_numpy()

        try:
            center, wire_direction, local_vertical = estimate_local_frame(cluster_points)
            local_points = project_to_local_2d(
                cluster_points,
                center,
                wire_direction,
                local_vertical,
            )
            x0, y0, c = fit_catenary(local_points)

            results.append(
                {
                    "cluster": int(cluster_label),
                    "num_points": len(cluster_points_df),
                    "x0": float(x0),
                    "y0": float(y0),
                    "c": float(c),
                }
            )
        except Exception as error:
            results.append(
                {
                    "cluster": int(cluster_label),
                    "num_points": len(cluster_points_df),
                    "error": str(error),
                }
            )

    results_df = pd.DataFrame(results)
    if {"cluster", "y0"}.issubset(results_df.columns):
        results_df = assign_wire_groups(results_df)

    return results_df
