"""
Geometry package for 3D computer vision primitives.

This package provides core geometric classes and utilities for 3D vision tasks,
including rotations, poses, transformations, camera models, and geometric algorithms.

Modules:
    camera: Camera models (Perspective, Fisheye, Equirectangular, etc.)
    rotation: 3D rotation representations (SO3, so3, quaternion, RPY)
    pose: 6-DOF pose (rotation + translation)
    tf: 6-DOF transformation class
    img_tf: 2D image transformation utilities
    geom_utils: Geometric algorithms (PnP, triangulation, essential matrix, etc.)
"""

from . import camera
from . import rotation
from . import pose
from . import tf
from . import img_tf
from . import geom_utils

# High-level classes for convenience
from .rotation import Rotation, RotType
from .pose import Pose
from .tf import Transform
from .camera import (
    Camera,
    CamType,
    PerspectiveCamera,
    OpenCVFisheyeCamera,
    ThinPrismFisheyeCamera,
    OmnidirectionalCamera,
    DoubleSphereCamera,
    EquirectangularCamera,
)

# Commonly used functions from geom_utils
from .geom_utils import (
    solve_pnp,
    triangulate_points,
    compute_essential_matrix_from_pose,
    compute_fundamental_matrix_from_points,
    compute_relative_transform,
    compute_relative_transform_from_points,
    convert_depth_to_point_cloud,
    convert_point_cloud_to_depth,
)

__all__ = [
    # Modules
    "camera",
    "rotation",
    "pose",
    "tf",
    "img_tf",
    "geom_utils",

    # High-level classes
    "Rotation",
    "RotType",
    "Pose",
    "Transform",
    "Camera",
    "CamType",
    "PerspectiveCamera",
    "OpenCVFisheyeCamera",
    "ThinPrismFisheyeCamera",
    "OmnidirectionalCamera",
    "DoubleSphereCamera",
    "EquirectangularCamera",

    # Common functions
    "solve_pnp",
    "triangulate_points",
    "compute_essential_matrix_from_pose",
    "compute_fundamental_matrix_from_points",
    "compute_relative_transform",
    "compute_relative_transform_from_points",
    "convert_depth_to_point_cloud",
    "convert_point_cloud_to_depth",
]

