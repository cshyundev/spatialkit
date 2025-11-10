"""
Geometry package for 3D computer vision primitives.

This package provides core geometric classes and utilities for 3D vision tasks,
including rotations, poses, transformations, and geometric algorithms.

Modules:
    rotation: 3D rotation representations (SO3, so3, quaternion, RPY)
    pose: 6-DOF pose (rotation + translation)
    tf: 6-DOF transformation class
    epipolar: Epipolar geometry (essential/fundamental matrices)
    multiview: Multi-view geometry primitives (PnP, triangulation, pose estimation)
    pointcloud: Point cloud and depth map conversions
    geom_utils: [Deprecated] Legacy module - use specific modules above

Note:
    Camera models have been moved to spatialkit.camera
    2D image transformations have been moved to spatialkit.imgproc
    View synthesis has been moved to spatialkit.imgproc
"""

from . import rotation
from . import pose
from . import tf
from . import epipolar
from . import multiview
from . import pointcloud

# High-level classes for convenience
from .rotation import Rotation, RotType
from .pose import Pose, interpolate_pose
from .tf import Transform, interpolate_transform

# Commonly used functions - imported from new modular structure
from .epipolar import (
    compute_essential_matrix_from_pose,
    compute_essential_matrix_from_fundamental,
    compute_fundamental_matrix_from_points,
    compute_fundamental_matrix_from_essential,
    compute_fundamental_matrix_using_ransac,
    decompose_essential_matrix,
    compute_points_for_epipolar_curve,
)

from .multiview import (
    solve_pnp,
    triangulate_points,
    compute_relative_transform,
    compute_relative_transform_from_points,
    find_corresponding_points,
)

from .pointcloud import (
    convert_point_cloud_to_depth,
    convert_depth_to_point_cloud,
    down_sample_point_cloud,
    compute_point_cloud_normals,
    depth_to_normal_map,
    depth_to_normal_map_fast
)

from .registration import (
    icp,
    icp_point_to_point,
    icp_point_to_plane,
    icp_generalized,
)

__all__ = [
    # Modules
    "rotation",
    "pose",
    "tf",
    "epipolar",
    "multiview",
    "pointcloud",

    # High-level classes
    "Rotation",
    "RotType",
    "Pose",
    "Transform",

    # Interpolation functions
    "interpolate_pose",
    "interpolate_transform",

    # Epipolar geometry functions
    "compute_essential_matrix_from_pose",
    "compute_essential_matrix_from_fundamental",
    "compute_fundamental_matrix_from_points",
    "compute_fundamental_matrix_from_essential",
    "compute_fundamental_matrix_using_ransac",
    "decompose_essential_matrix",
    "compute_points_for_epipolar_curve",

    # Multi-view geometry functions
    "solve_pnp",
    "triangulate_points",
    "compute_relative_transform",
    "compute_relative_transform_from_points",
    "find_corresponding_points",

    # Point cloud functions
    "convert_point_cloud_to_depth",
    "convert_depth_to_point_cloud",
    "down_sample_point_cloud",
    "compute_point_cloud_normals",
    "depth_to_normal_map",
    "depth_to_normal_map_fast",

    # Registration functions
    "icp",
    "icp_point_to_point",
    "icp_point_to_plane",
    "icp_generalized",
]

