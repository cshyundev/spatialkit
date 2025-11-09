# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.2] - 2025-11-09

### Added
- **Point Cloud Processing**: A new `geom.pointcloud` module provides functionalities for creating, manipulating, and transforming 3D point clouds. Includes `PointCloud` class and supporting functions.
- **3D Registration**: A new `geom.registration` module implements ICP (Iterative Closest Point) for point cloud alignment.
- **Image Transformation**: The `imgproc.transform` module has been renamed to `imgproc.img_tf` and expanded with new functions for image transformations, including `extract_patches`.
- **Dependencies**: Added `open3d` for 3D data processing and visualization.
- **Error Handling**: Introduced more specific exceptions like `NotArrayLikeError` and `InvalidArgumentError`.
- **Tests**: Added comprehensive tests for the new point cloud, registration, and patch extraction functionalities.

### Changed
- **Documentation**: Updated documentation to reflect the new modules and API changes.

## [0.3.1] - 2025-11-03

### Fixed
- **Dependencies**: Moved all optional dependencies (markers, vis3d) to required to fix `ModuleNotFoundError` on import. Added `absl-py`, `rich`, `dt-apriltags`, `stag-python`, and `open3d` to required dependencies.

## [0.3.0] - 2025-11-03

### Initial Public Release

First public release of spatialkit - a computer vision and robotics library for 3D vision tasks.

Key features include unified NumPy/PyTorch operations, 3D geometry classes, multiple camera models, and ROS2 compatibility.
