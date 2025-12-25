"""
3D box primitives for visualization.

This module provides functions to create box geometries:
- Box from size_xyz and pose (for object pose estimation visualization)
- Box from min/max bounds (AABB)
"""

from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d

from ..common.exceptions import InvalidShapeError, RenderingError
from .common import O3dLineSet, O3dTriMesh


# Box vertices indices for 8 corners
# Bottom face (z=min): 0,1,2,3  Top face (z=max): 4,5,6,7
#     7 -------- 6
#    /|         /|
#   4 -------- 5 |
#   | |        | |
#   | 3 -------| 2
#   |/         |/
#   0 -------- 1

_BOX_EDGES = np.array(
    [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

_BOX_TRIANGLES = np.array(
    [
        # Bottom face (z=min, normal -z)
        [0, 2, 1], [0, 3, 2],
        # Top face (z=max, normal +z)
        [4, 5, 6], [4, 6, 7],
        # Front face (y=min, normal -y)
        [0, 1, 5], [0, 5, 4],
        # Back face (y=max, normal +y)
        [2, 3, 7], [2, 7, 6],
        # Left face (x=min, normal -x)
        [0, 4, 7], [0, 7, 3],
        # Right face (x=max, normal +x)
        [1, 2, 6], [1, 6, 5],
    ],
    dtype=np.int32,
)


def _compute_box_vertices(size_xyz: Tuple[float, float, float]) -> np.ndarray:
    """
    Compute 8 box vertices centered at origin.

    Args:
        size_xyz (Tuple[float, float, float]): Box size (x, y, z).

    Returns:
        np.ndarray: 8 vertices with shape (8, 3).
    """
    hx, hy, hz = size_xyz[0] / 2.0, size_xyz[1] / 2.0, size_xyz[2] / 2.0

    vertices = np.array(
        [
            [-hx, -hy, -hz],  # 0: bottom-left-front
            [hx, -hy, -hz],   # 1: bottom-right-front
            [hx, hy, -hz],    # 2: bottom-right-back
            [-hx, hy, -hz],   # 3: bottom-left-back
            [-hx, -hy, hz],   # 4: top-left-front
            [hx, -hy, hz],    # 5: top-right-front
            [hx, hy, hz],     # 6: top-right-back
            [-hx, hy, hz],    # 7: top-left-back
        ],
        dtype=np.float64,
    )
    return vertices


def create_box(
    size_xyz: Tuple[float, float, float],
    pose: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    wireframe: bool = True,
) -> Union[O3dLineSet, O3dTriMesh]:
    """
    Create a 3D box (wireframe or solid mesh) centered at origin or pose.

    Args:
        size_xyz (Tuple[float, float, float]): Box size as (x, y, z) dimensions.
        pose (Optional[np.ndarray], [4, 4]): 4x4 transformation matrix for position and orientation.
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        wireframe (bool): If True, create wireframe (LineSet). If False, create solid mesh.

    Returns:
        Union[O3dLineSet, O3dTriMesh]: Box geometry (LineSet for wireframe, TriangleMesh for solid).

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If box creation fails.

    Example:
        >>> # Simple box at origin
        >>> box = create_box(size_xyz=(1.0, 2.0, 1.5))
        >>>
        >>> # Box with position and orientation
        >>> pose = np.eye(4)
        >>> pose[:3, 3] = [1, 0, 0]  # position
        >>> box = create_box(size_xyz=(1, 1, 2), pose=pose)
        >>>
        >>> # Solid box
        >>> box = create_box(size_xyz=(1, 1, 1), color=(255, 0, 0), wireframe=False)
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}."
            )

    try:
        vertices = _compute_box_vertices(size_xyz)
        color_normalized = np.array(color, dtype=np.float64) / 255.0

        if wireframe:
            line_set = O3dLineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(_BOX_EDGES)
            line_set.colors = o3d.utility.Vector3dVector(
                [color_normalized for _ in range(len(_BOX_EDGES))]
            )
            if pose is not None:
                line_set.transform(pose)
            return line_set
        else:
            mesh = O3dTriMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(_BOX_TRIANGLES)
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                [color_normalized for _ in range(8)]
            )
            mesh.compute_vertex_normals()
            if pose is not None:
                mesh.transform(pose)
            return mesh

    except InvalidShapeError:
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create box: {e}") from e


def create_box_from_bounds(
    min_pt: np.ndarray,
    max_pt: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    wireframe: bool = True,
) -> Union[O3dLineSet, O3dTriMesh]:
    """
    Create an axis-aligned bounding box (AABB) from min/max points.

    Args:
        min_pt (np.ndarray, [3,]): Minimum corner point (x_min, y_min, z_min).
        max_pt (np.ndarray, [3,]): Maximum corner point (x_max, y_max, z_max).
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        wireframe (bool): If True, create wireframe (LineSet). If False, create solid mesh.

    Returns:
        Union[O3dLineSet, O3dTriMesh]: AABB geometry.

    Raises:
        InvalidShapeError: If min_pt or max_pt is not a 3D vector, or if min_pt >= max_pt in any dimension.
        RenderingError: If box creation fails.

    Example:
        >>> min_pt = np.array([-1, -1, 0])
        >>> max_pt = np.array([1, 1, 2])
        >>> aabb = create_box_from_bounds(min_pt, max_pt)
    """
    if not isinstance(min_pt, np.ndarray) or min_pt.shape != (3,):
        raise InvalidShapeError(
            f"min_pt must be a 3D vector with shape (3,), got {type(min_pt)} "
            f"with shape {getattr(min_pt, 'shape', 'unknown')}."
        )

    if not isinstance(max_pt, np.ndarray) or max_pt.shape != (3,):
        raise InvalidShapeError(
            f"max_pt must be a 3D vector with shape (3,), got {type(max_pt)} "
            f"with shape {getattr(max_pt, 'shape', 'unknown')}."
        )

    if np.any(min_pt >= max_pt):
        raise InvalidShapeError(
            f"min_pt must be less than max_pt in all dimensions. "
            f"Got min_pt={min_pt}, max_pt={max_pt}."
        )

    try:
        # Compute center and size_xyz from bounds
        center = (min_pt + max_pt) / 2.0
        size_xyz = tuple(max_pt - min_pt)

        # Create pose from center
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = center

        return create_box(
            size_xyz=size_xyz,
            pose=pose,
            color=color,
            wireframe=wireframe,
        )

    except InvalidShapeError:
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create box from bounds: {e}") from e
