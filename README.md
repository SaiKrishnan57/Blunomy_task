# Blunomy LiDAR Catenary Detection

Python package for Blunomy's LiDAR technical test.

The goal of this project is to process LiDAR point cloud data stored in `.parquet` files, identify wire structures in each scene, and fit catenary models to the detected wires.

## Problem Overview

Each input parquet file contains a 3D point cloud with:

- x: horizontal coordinate
- y: horizontal coordinate
- z: height

Each row is one LiDAR point, not one full wire.

A single file may contain points from multiple wires mixed together, so the task is not only curve fitting. It also requires segmentation: separating which points belong to which wire before fitting a catenary model.

The test brief provides the standard 2D catenary equation:

`y(x) = y0 + c * cosh((x - x0) / c) - c`

Because the LiDAR data is 3D, the solution first estimates a local frame for each wire and then fits the catenary in that local 2D frame.

## Datasets

The provided datasets are treated as independent LiDAR scenes:

- `lidar_cable_points_easy.parquet`
- `lidar_cable_points_medium.parquet`
- `lidar_cable_points_hard.parquet`
- `lidar_cable_points_extrahard.parquet`

## Approach

The current implementation follows a staged geometric pipeline:

1. Load a parquet point cloud.
2. Apply lightweight preprocessing.
3. Cluster points into candidate wire segments.
4. Estimate a local coordinate frame for each cluster.
5. Project each cluster into local 2D coordinates.
6. Fit a catenary curve in that local frame.
7. Group similar fitted clusters into final wire groups when needed.

### Clustering strategy

The project began with DBSCAN directly on raw `x`, `y`, and `z` coordinates as a baseline.

That worked on simpler scenes, but it tended to merge nearby parallel wires in harder datasets. To improve this, clustering was changed to work in the 2D cross-section perpendicular to the dominant wire direction of the scene.

This made wire separation more meaningful, but it also introduced over-segmentation in some cases. To handle that, the pipeline adds a conservative post-fit grouping step based on fitted wire characteristics. Clusters are only merged into broader wire groups when the fitted vertical positions show clear separation; otherwise they are kept separate to avoid over-merging.

## Project Structure

```text
src/
  lidar_catenary/
    io.py
    preprocess.py
    clustering.py
    geometry.py
    catenary.py
    pipeline.py
    cli.py
tests/
  test_io.py
  test_preprocess.py
  test_geometry.py
  test_catenary.py
  test_pipeline.py
```

### Module Summary

- `io.py`: loads parquet point-cloud data and validates required columns
- `preprocess.py`: performs lightweight cleaning
- `clustering.py`: segments candidate wire clusters using DBSCAN in cross-sectional space
- `geometry.py`: estimates local wire frames and projects 3D points to local 2D coordinates
- `catenary.py`: defines and fits the catenary equation
- `pipeline.py`: runs the full workflow on one input file
- `cli.py`: provides a command-line entry point

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -e .
```

For development, testing, and coverage:

```powershell
pip install -e .[dev]
```

## Usage

Run the pipeline on one parquet file:

```powershell
python -m lidar_catenary.cli "C:\path\to\lidar_cable_points_easy.parquet"
```

Example with the provided data:

```powershell
python -m lidar_catenary.cli "C:\Users\saikr\OneDrive\Desktop\Projects\Blunomy\Data Science - LiDAR Technical Test\lidar_cable_points_easy.parquet"
```

## Tests

Run the full test suite:

```powershell
pytest
```

If `pytest` is not available on your shell path, use:

```powershell
python -m pytest
```

## Coverage

To view terminal coverage with missing lines:

```powershell
pytest --cov=src/lidar_catenary --cov-report=term-missing
```

To generate an HTML coverage report:

```powershell
pytest --cov=src/lidar_catenary --cov-report=html
```

This creates an `htmlcov/` directory. Open `htmlcov/index.html` in a browser to inspect line-by-line coverage.

## Output

The CLI currently prints a table with one row per fitted cluster, including:

- `cluster`: raw DBSCAN cluster label
- `num_points`: number of points in that cluster
- `x0`: fitted trough location in local horizontal coordinates
- `y0`: fitted minimum height in local coordinates
- `c`: fitted catenary curvature parameter
- `wire_group`: conservative post-fit interpretation of which clusters likely belong to the same final wire

During development, the pipeline also prints temporary diagnostics such as coordinate ranges, DBSCAN settings, and cluster sizes. These were useful for understanding clustering behavior and can be cleaned up later if a quieter final output is preferred.

## Design Notes

- The solution is implemented as a Python package rather than a notebook-first analysis.
- The pipeline is modular so each stage can be reasoned about independently.
- The implementation favors clarity and explainability over heavy optimization.
- The current grouping logic is intentionally lightweight, heuristic, and conservative.

## Current Limitations

The current version works as an interview-ready prototype, but it is not a final production system.

Known limitations include:

- DBSCAN parameters are still hand-tuned.
- Cross-sectional clustering assumes a meaningful dominant wire direction in the scene.
- Post-fit wire grouping is heuristic and relies mainly on fitted vertical separation. When the fitted clusters do not show clear separation, the pipeline keeps them distinct rather than forcing a merge.
- The pipeline does not yet compute explicit fit-quality metrics such as RMSE.
- Visual inspection/export of fitted curves is not yet included.

## Possible Next Improvements

If extended further, likely next steps would be:

- add fit-quality metrics
- improve cluster merging using multiple fitted parameters
- expose parameters through CLI options
- add plots or exports of fitted 3D curves
- expand automated tests and add stronger integration-style validation
