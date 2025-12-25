"""
3D plane primitives for visualization.

This module provides functions to create plane geometries:
- Axis-aligned planes (xy, xz, yz)
- Planes from corner points
- Planes from normal vector and distance
"""

from typing import Optional, Tuple

import numpy as np
import open3d as o3d

from ..common.exceptions import GeometryError, InvalidShapeError, RenderingError
from .common import O3dTriMesh


def _create_plane_mesh(
    vertices: np.ndarray,
    color: Tuple[int, int, int],
    pose: Optional[np.ndarray] = None,
) -> O3dTriMesh:
    """
    Internal helper to create a double-sided plane mesh from 4 vertices.

    The plane is double-sided (visible from both directions) by duplicating
    triangles with opposite winding order. This ensures the plane is visible
    regardless of the viewing direction.

    Args:
        vertices (np.ndarray, [4, 3]): Four corner vertices in order (BL, BR, TR, TL).
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        pose (Optional[np.ndarray], [4, 4]): 4x4 transformation matrix.

    Returns:
        O3dTriMesh: A double-sided TriangleMesh object representing the plane.
    """
    # Normalize color to 0-1 scale
    color_normalized = np.array(color, dtype=np.float64) / 255.0

    # Create double-sided triangles (4 triangles total)
    # Front face (CCW winding): normal points "up"
    # Back face (CW winding): normal points "down"
    # Vertices order: 0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left
    triangles = np.array([
        # Front face (CCW)
        [0, 1, 2], [0, 2, 3],
        # Back face (CW) - reversed winding
        [0, 2, 1], [0, 3, 2],
    ], dtype=np.int32)

    # Vertex colors (same color for all vertices)
    vertex_colors = np.tile(color_normalized, (4, 1))

    mesh = O3dTriMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Compute normals for proper lighting
    mesh.compute_vertex_normals()

    if pose is not None:
        mesh.transform(pose)

    return mesh


def create_axis_aligned_plane(
    plane: str,
    width: float,
    height: float,
    offset: float = 0.0,
    color: Tuple[int, int, int] = (200, 200, 200),
    pose: Optional[np.ndarray] = None,
) -> O3dTriMesh:
    """
    Create an axis-aligned plane (xy, xz, or yz).

    Args:
        plane (str): Plane specification - "xy", "xz", or "yz".
        width (float): Width of the plane (first axis in plane name).
        height (float): Height of the plane (second axis in plane name).
        offset (float): Offset along the perpendicular axis (default 0.0).
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        pose (Optional[np.ndarray], [4, 4]): Additional 4x4 transformation matrix.

    Returns:
        O3dTriMesh: A TriangleMesh object representing the plane.

    Raises:
        GeometryError: If plane is not one of "xy", "xz", "yz".
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If plane creation fails.

    Example:
        >>> plane = create_axis_aligned_plane("xy", width=2.0, height=2.0)
        >>> plane_offset = create_axis_aligned_plane("xz", width=1.0, height=1.5, offset=0.5)
    """
    valid_planes = ("xy", "xz", "yz")
    if plane not in valid_planes:
        raise GeometryError(
            f"Invalid plane '{plane}'. Supported planes are: {valid_planes}. "
            f"Please provide a valid plane specification."
        )

    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}."
            )

    try:
        half_w, half_h = width / 2.0, height / 2.0

        if plane == "xy":
            # XY plane: normal is Z, offset along Z
            vertices = np.array(
                [
                    [-half_w, -half_h, offset],  # bottom-left
                    [half_w, -half_h, offset],  # bottom-right
                    [half_w, half_h, offset],  # top-right
                    [-half_w, half_h, offset],  # top-left
                ],
                dtype=np.float64,
            )
        elif plane == "xz":
            # XZ plane: normal is Y, offset along Y
            vertices = np.array(
                [
                    [-half_w, offset, -half_h],  # bottom-left
                    [half_w, offset, -half_h],  # bottom-right
                    [half_w, offset, half_h],  # top-right
                    [-half_w, offset, half_h],  # top-left
                ],
                dtype=np.float64,
            )
        else:  # "yz"
            # YZ plane: normal is X, offset along X
            vertices = np.array(
                [
                    [offset, -half_w, -half_h],  # bottom-left
                    [offset, half_w, -half_h],  # bottom-right
                    [offset, half_w, half_h],  # top-right
                    [offset, -half_w, half_h],  # top-left
                ],
                dtype=np.float64,
            )

        return _create_plane_mesh(vertices, color, pose)

    except GeometryError:
        raise
    except InvalidShapeError:
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create axis-aligned plane: {e}") from e


def create_plane_from_corners(
    corners: np.ndarray,
    color: Tuple[int, int, int] = (200, 200, 200),
    pose: Optional[np.ndarray] = None,
) -> O3dTriMesh:
    """
    Create a plane from 4 corner points.

    Args:
        corners (np.ndarray, [4, 3]): Four 3D corner points in order
            (bottom-left, bottom-right, top-right, top-left).
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        pose (Optional[np.ndarray], [4, 4]): Additional 4x4 transformation matrix.

    Returns:
        O3dTriMesh: A TriangleMesh object representing the plane.

    Raises:
        InvalidShapeError: If corners is not shape (4, 3) or pose is not 4x4.
        GeometryError: If corners are colinear (degenerate plane).
        RenderingError: If plane creation fails.

    Example:
        >>> corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        >>> plane = create_plane_from_corners(corners)
    """
    if not isinstance(corners, np.ndarray) or corners.shape != (4, 3):
        raise InvalidShapeError(
            f"Corners must be a numpy array with shape (4, 3), got {type(corners)} "
            f"with shape {getattr(corners, 'shape', 'unknown')}. "
            f"Please provide 4 corner points as a 4x3 array."
        )

    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}."
            )

    # Check for degenerate plane (colinear points)
    edge1 = corners[1] - corners[0]
    edge2 = corners[3] - corners[0]
    normal = np.cross(edge1, edge2)

    if np.linalg.norm(normal) < 1e-10:
        raise GeometryError(
            "Corner points are colinear or nearly colinear, cannot form a valid plane. "
            "Please provide 4 non-colinear points."
        )

    try:
        return _create_plane_mesh(corners, color, pose)

    except GeometryError:
        raise
    except InvalidShapeError:
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create plane from corners: {e}") from e


def create_plane_from_normal(
    normal: np.ndarray,
    distance: float,
    width: float,
    height: float,
    color: Tuple[int, int, int] = (200, 200, 200),
    pose: Optional[np.ndarray] = None,
) -> O3dTriMesh:
    """
    Create a plane from a normal vector and distance from origin.

    Args:
        normal (np.ndarray, [3,]): Normal vector of the plane (will be normalized).
        distance (float): Distance from the origin to the plane along the normal.
        width (float): Width of the plane (local x-axis).
        height (float): Height of the plane (local y-axis).
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        pose (Optional[np.ndarray], [4, 4]): Additional 4x4 transformation matrix.

    Returns:
        O3dTriMesh: A TriangleMesh object representing the plane.

    Raises:
        InvalidShapeError: If normal is not a 3D vector or pose is not 4x4.
        GeometryError: If normal is a zero vector.
        RenderingError: If plane creation fails.

    Example:
        >>> normal = np.array([0, 0, 1])
        >>> plane = create_plane_from_normal(normal, distance=1.0, width=2.0, height=2.0)
    """
    if not isinstance(normal, np.ndarray) or normal.shape != (3,):
        raise InvalidShapeError(
            f"Normal must be a 3D vector with shape (3,), got {type(normal)} "
            f"with shape {getattr(normal, 'shape', 'unknown')}."
        )

    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}."
            )

    # Normalize the normal vector
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-10:
        raise GeometryError(
            f"Normal vector must be non-zero, got {normal}. "
            "Please provide a valid direction vector."
        )
    normal_unit = normal / normal_length

    try:
        # Compute orthonormal basis for the plane
        # Find a vector not parallel to normal
        if abs(normal_unit[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])

        # Local x-axis (tangent 1)
        local_x = np.cross(normal_unit, arbitrary)
        local_x = local_x / np.linalg.norm(local_x)

        # Local y-axis (tangent 2)
        local_y = np.cross(normal_unit, local_x)
        local_y = local_y / np.linalg.norm(local_y)

        # Compute center point
        center = normal_unit * distance

        # Compute 4 corner vertices
        half_w, half_h = width / 2.0, height / 2.0
        vertices = np.array(
            [
                center - half_w * local_x - half_h * local_y,  # bottom-left
                center + half_w * local_x - half_h * local_y,  # bottom-right
                center + half_w * local_x + half_h * local_y,  # top-right
                center - half_w * local_x + half_h * local_y,  # top-left
            ],
            dtype=np.float64,
        )

        return _create_plane_mesh(vertices, color, pose)

    except GeometryError:
        raise
    except InvalidShapeError:
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create plane from normal: {e}") from e
