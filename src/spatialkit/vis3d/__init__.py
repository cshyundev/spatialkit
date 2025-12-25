"""
3D visualization utilities using Open3D.

This package provides utilities for creating and visualizing 3D geometries.
"""

# Type aliases
from .common import O3dGeometry, O3dLineSet, O3dPCD, O3dTriMesh

# Circle primitives
from .circles import (
    create_thick_circle_mesh,
)

# Line and arrow primitives
from .lines import (
    create_arrow,
    create_arrow_from_points,
    create_arrow_mesh,
    create_arrow_mesh_from_points,
    create_line_from_points,
    create_line_mesh_from_points,
)

# Box primitives
from .boxes import (
    create_box,
    create_box_from_bounds,
)

# Plane primitives
from .planes import (
    create_axis_aligned_plane,
    create_plane_from_corners,
    create_plane_from_normal,
)

# Visualization indicators
from .indicators import (
    create_camera_indicator_frame,
    create_coordinate,
    create_spherical_camera_indicator_frame,
)

# Image-textured geometries
from .textures import (
    create_image_plane,
    create_image_sphere,
)

# Point cloud and mesh utilities
from .o3dutils import (
    create_mesh,
    create_mesh_from_pcd,
    create_point_cloud,
    load_mesh,
    load_pcd,
    save_mesh,
    save_pcd,
    visualize_geometries,
)

# High-level API
from .api import (
    create_camera_vis,
    create_pose_vis,
    create_trajectory_vis,
)

__all__ = [
    # Types
    "O3dGeometry",
    "O3dLineSet",
    "O3dPCD",
    "O3dTriMesh",
    # Circles
    "create_thick_circle_mesh",
    # Lines
    "create_arrow",
    "create_arrow_from_points",
    "create_arrow_mesh",
    "create_arrow_mesh_from_points",
    "create_line_from_points",
    "create_line_mesh_from_points",
    # Boxes
    "create_box",
    "create_box_from_bounds",
    # Planes
    "create_axis_aligned_plane",
    "create_plane_from_corners",
    "create_plane_from_normal",
    # Indicators
    "create_camera_indicator_frame",
    "create_coordinate",
    "create_spherical_camera_indicator_frame",
    # Textures
    "create_image_plane",
    "create_image_sphere",
    # Utils
    "create_mesh",
    "create_mesh_from_pcd",
    "create_point_cloud",
    "load_mesh",
    "load_pcd",
    "save_mesh",
    "save_pcd",
    "visualize_geometries",
    # API
    "create_camera_vis",
    "create_pose_vis",
    "create_trajectory_vis",
]
