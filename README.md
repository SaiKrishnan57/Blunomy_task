# Blunomy LiDAR Catenary Detection

This project is a Python solution for Blunomy's LiDAR technical test.

The goal is to process LiDAR point cloud data stored in `.parquet` files, identify how many electrical wires are present in each dataset, and fit a 3D catenary model to each detected wire.

## Problem Summary

Each input file contains a 3D point cloud with `x`, `y`, and `z` coordinates.  
Each point represents a LiDAR return in space.  
A single file may contain points from multiple wires mixed together.

The objective is to:

- detect the number of wires in a point cloud
- group points belonging to the same wire
- fit a catenary model to each wire
- provide a reusable and maintainable Python implementation

## Approach

The solution is designed as a pipeline with the following stages:

1. Load the LiDAR point cloud from a parquet file.
2. Perform lightweight preprocessing and validation.
3. Cluster the point cloud into candidate wire groups.
4. Estimate a local plane or frame for each wire.
5. Project the 3D wire points into a 2D local coordinate system.
6. Fit a catenary curve in that local frame.
7. Reconstruct the fitted curve back into 3D.

This approach allows the original 2D catenary equation to be adapted to 3D wire geometry.

## Repository Structure

