"""
3D line and arrow primitives for visualization.

This module provides functions to create line-based geometries:
- Line from two points
- Arrow (vector visualization) - wireframe (LineSet) and solid mesh versions
"""

from typing import Optional, Tuple

import numpy as np
import open3d as o3d

from ..common.exceptions import GeometryError, InvalidShapeError, RenderingError
from .common import O3dLineSet, O3dTriMesh


def create_line_from_points(
    point1: np.ndarray,
    point2: np.ndarray,
    color: Optional[Tuple[int, int, int]] = None,
) -> O3dLineSet:
    """
    Create a line connecting two 3D points.

    Args:
        point1 (np.ndarray, [3,]): [x, y, z] coordinates of the first point.
        point2 (np.ndarray, [3,]): [x, y, z] coordinates of the second point.
        color (Optional[Tuple[int, int, int]]): RGB color (0-255 scale).

    Returns:
        O3dLineSet: A LineSet object representing the line between the two points.

    Raises:
        InvalidShapeError: If points don't have correct 3D coordinates.
        RenderingError: If line creation fails.

    Example:
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([1, 1, 1])
        >>> line = create_line_from_points(p1, p2, color=(255, 0, 0))
    """
    if not isinstance(point1, np.ndarray) or point1.shape != (3,):
        raise InvalidShapeError(
            f"point1 must be a 3D coordinate array with shape (3,), got {type(point1)} "
            f"with shape {getattr(point1, 'shape', 'unknown')}. "
            "Please provide a valid [x, y, z] coordinate array."
        )

    if not isinstance(point2, np.ndarray) or point2.shape != (3,):
        raise InvalidShapeError(
            f"point2 must be a 3D coordinate array with shape (3,), got {type(point2)} "
            f"with shape {getattr(point2, 'shape', 'unknown')}. "
            "Please provide a valid [x, y, z] coordinate array."
        )

    try:
        points = np.array([point1, point2], dtype=np.float64)
        lines = np.array([[0, 1]], dtype=np.int32)

        line_set = O3dLineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        if color is None:
            line_color = np.random.rand(3)
        else:
            line_color = np.array(color) / 255.0

        line_set.colors = o3d.utility.Vector3dVector([line_color])

        return line_set
    except Exception as e:
        raise RenderingError(f"Failed to create line from points: {e}") from e


def create_line_mesh_from_points(
    point1: np.ndarray,
    point2: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    radius: float = 0.01,
) -> O3dTriMesh:
    """
    Create a solid mesh line (cylinder) connecting two 3D points.

    Args:
        point1 (np.ndarray, [3,]): [x, y, z] coordinates of the first point.
        point2 (np.ndarray, [3,]): [x, y, z] coordinates of the second point.
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        radius (float): Radius of the cylinder.

    Returns:
        O3dTriMesh: A TriangleMesh cylinder representing the line.

    Raises:
        InvalidShapeError: If points don't have correct 3D coordinates.
        GeometryError: If the two points are the same.
        RenderingError: If line creation fails.

    Example:
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([1, 1, 1])
        >>> line = create_line_mesh_from_points(p1, p2, color=(255, 0, 0), radius=0.02)
    """
    if not isinstance(point1, np.ndarray) or point1.shape != (3,):
        raise InvalidShapeError(
            f"point1 must be a 3D coordinate array with shape (3,), got {type(point1)} "
            f"with shape {getattr(point1, 'shape', 'unknown')}. "
            "Please provide a valid [x, y, z] coordinate array."
        )

    if not isinstance(point2, np.ndarray) or point2.shape != (3,):
        raise InvalidShapeError(
            f"point2 must be a 3D coordinate array with shape (3,), got {type(point2)} "
            f"with shape {getattr(point2, 'shape', 'unknown')}. "
            "Please provide a valid [x, y, z] coordinate array."
        )

    direction = point2 - point1
    length = np.linalg.norm(direction)

    if length < 1e-10:
        raise GeometryError(
            f"point1 and point2 are the same or too close: point1={point1}, point2={point2}. "
            "Please provide two distinct points."
        )

    try:
        direction_unit = direction / length

        # Create cylinder along Z-axis
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=length,
            resolution=20,
            split=1,
        )
        # Move so bottom is at origin (default is centered)
        cylinder.translate([0, 0, length / 2])

        # Rotate to align with direction
        rotation_matrix = _compute_arrow_rotation(direction_unit)
        cylinder.rotate(rotation_matrix, center=[0, 0, 0])

        # Translate to point1
        cylinder.translate(point1)

        # Set color
        color_normalized = [c / 255.0 for c in color]
        cylinder.paint_uniform_color(color_normalized)

        # Compute normals
        cylinder.compute_vertex_normals()

        return cylinder

    except (GeometryError, InvalidShapeError):
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create line mesh from points: {e}") from e


def _compute_arrow_rotation(direction_unit: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix to align Z-axis with target direction.

    Args:
        direction_unit (np.ndarray, [3,]): Unit direction vector.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z_axis, direction_unit)

    if dot > 0.9999:
        # Already aligned, no rotation needed
        return np.eye(3)
    elif dot < -0.9999:
        # Opposite direction, rotate 180 degrees around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return o3d.geometry.get_rotation_matrix_from_axis_angle(perp * np.pi)
    else:
        # General case: compute rotation axis and angle
        rotation_axis = np.cross(z_axis, direction_unit)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)


def _validate_arrow_inputs(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    head_length_ratio: float,
) -> np.ndarray:
    """
    Validate arrow inputs and return normalized direction.

    Args:
        origin (np.ndarray, [3,]): Starting point of the arrow.
        direction (np.ndarray, [3,]): Direction vector.
        length (float): Total length of the arrow.
        head_length_ratio (float): Ratio of head length to total length.

    Returns:
        np.ndarray: Normalized direction vector.

    Raises:
        InvalidShapeError: If origin or direction is not a 3D vector.
        GeometryError: If direction is zero or parameters are invalid.
    """
    if not isinstance(origin, np.ndarray) or origin.shape != (3,):
        raise InvalidShapeError(
            f"origin must be a 3D vector with shape (3,), got {type(origin)} "
            f"with shape {getattr(origin, 'shape', 'unknown')}."
        )

    if not isinstance(direction, np.ndarray) or direction.shape != (3,):
        raise InvalidShapeError(
            f"direction must be a 3D vector with shape (3,), got {type(direction)} "
            f"with shape {getattr(direction, 'shape', 'unknown')}."
        )

    dir_length = np.linalg.norm(direction)
    if dir_length < 1e-10:
        raise GeometryError(
            f"direction must be a non-zero vector, got {direction}. "
            "Please provide a valid direction vector."
        )

    if length <= 0:
        raise GeometryError(f"length must be positive, got {length}.")

    if not (0.0 < head_length_ratio < 1.0):
        raise GeometryError(
            f"head_length_ratio must be between 0 and 1, got {head_length_ratio}."
        )

    return direction / dir_length


def _validate_arrow_points(start: np.ndarray, end: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Validate arrow points and return direction and length.

    Args:
        start (np.ndarray, [3,]): Starting point.
        end (np.ndarray, [3,]): End point.

    Returns:
        Tuple[np.ndarray, float]: Direction vector and length.

    Raises:
        InvalidShapeError: If start or end is not a 3D vector.
        GeometryError: If start and end are the same point.
    """
    if not isinstance(start, np.ndarray) or start.shape != (3,):
        raise InvalidShapeError(
            f"start must be a 3D vector with shape (3,), got {type(start)} "
            f"with shape {getattr(start, 'shape', 'unknown')}."
        )

    if not isinstance(end, np.ndarray) or end.shape != (3,):
        raise InvalidShapeError(
            f"end must be a 3D vector with shape (3,), got {type(end)} "
            f"with shape {getattr(end, 'shape', 'unknown')}."
        )

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-10:
        raise GeometryError(
            f"start and end points are the same or too close: start={start}, end={end}. "
            "Please provide two distinct points."
        )

    return direction, length


def create_arrow(
    origin: np.ndarray = None,
    direction: np.ndarray = None,
    length: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    head_length_ratio: float = 0.2,
    head_width_ratio: float = 0.5,
) -> O3dLineSet:
    """
    Create a 3D wireframe arrow (LineSet).

    The arrow points from origin in the given direction with specified length.

    Args:
        origin (np.ndarray, [3,]): Starting point of the arrow. Defaults to [0, 0, 0].
        direction (np.ndarray, [3,]): Direction vector (will be normalized). Defaults to [0, 0, 1].
        length (float): Total length of the arrow.
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        head_length_ratio (float): Ratio of head length to total length (0.0 to 1.0).
        head_width_ratio (float): Ratio of head width to head length.

    Returns:
        O3dLineSet: Wireframe arrow geometry.

    Raises:
        InvalidShapeError: If origin or direction is not a 3D vector.
        GeometryError: If direction is a zero vector or parameters are invalid.
        RenderingError: If arrow creation fails.

    Example:
        >>> arrow = create_arrow(
        ...     origin=np.array([0, 0, 0]),
        ...     direction=np.array([0, 0, 1]),
        ...     length=2.0,
        ...     color=(255, 0, 0),
        ... )
    """
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])

    direction_unit = _validate_arrow_inputs(origin, direction, length, head_length_ratio)

    try:
        # Compute arrow geometry
        head_length = length * head_length_ratio
        shaft_length = length - head_length
        head_width = head_length * head_width_ratio

        # Arrow tip point
        tip = origin + direction_unit * length

        # Shaft end point (where head starts)
        shaft_end = origin + direction_unit * shaft_length

        # Compute perpendicular vectors for arrow head
        if abs(direction_unit[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])

        perp1 = np.cross(direction_unit, arbitrary)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction_unit, perp1)

        # Arrow head base points (4 points for better visibility)
        head_base1 = shaft_end + perp1 * head_width
        head_base2 = shaft_end - perp1 * head_width
        head_base3 = shaft_end + perp2 * head_width
        head_base4 = shaft_end - perp2 * head_width

        # Create points array
        points = np.array([
            origin,       # 0
            shaft_end,    # 1
            tip,          # 2
            head_base1,   # 3
            head_base2,   # 4
            head_base3,   # 5
            head_base4,   # 6
        ], dtype=np.float64)

        # Create lines: shaft + 4 head lines
        lines = np.array([
            [0, 2],  # Full shaft from origin to tip
            [2, 3],  # Head line 1
            [2, 4],  # Head line 2
            [2, 5],  # Head line 3
            [2, 6],  # Head line 4
        ], dtype=np.int32)

        # Create LineSet
        line_set = O3dLineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set colors
        color_normalized = np.array(color, dtype=np.float64) / 255.0
        line_set.colors = o3d.utility.Vector3dVector([color_normalized] * len(lines))

        return line_set

    except (GeometryError, InvalidShapeError):
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create arrow: {e}") from e


def create_arrow_mesh(
    origin: np.ndarray = None,
    direction: np.ndarray = None,
    length: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    head_length_ratio: float = 0.2,
    shaft_radius: float = 0.02,
    head_radius: float = 0.05,
) -> O3dTriMesh:
    """
    Create a 3D solid mesh arrow (TriangleMesh).

    The arrow points from origin in the given direction with specified length.

    Args:
        origin (np.ndarray, [3,]): Starting point of the arrow. Defaults to [0, 0, 0].
        direction (np.ndarray, [3,]): Direction vector (will be normalized). Defaults to [0, 0, 1].
        length (float): Total length of the arrow.
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        head_length_ratio (float): Ratio of head length to total length (0.0 to 1.0).
        shaft_radius (float): Radius of the arrow shaft.
        head_radius (float): Radius of the arrow head base.

    Returns:
        O3dTriMesh: Solid mesh arrow geometry.

    Raises:
        InvalidShapeError: If origin or direction is not a 3D vector.
        GeometryError: If direction is a zero vector or parameters are invalid.
        RenderingError: If arrow creation fails.

    Example:
        >>> arrow = create_arrow_mesh(
        ...     origin=np.array([0, 0, 0]),
        ...     direction=np.array([0, 0, 1]),
        ...     length=2.0,
        ...     color=(255, 0, 0),
        ... )
    """
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])

    direction_unit = _validate_arrow_inputs(origin, direction, length, head_length_ratio)

    try:
        # Compute dimensions
        head_length = length * head_length_ratio
        shaft_length = length - head_length

        # Create cylinder (shaft) - default orientation is along Z-axis
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=shaft_radius,
            height=shaft_length,
            resolution=20,
            split=1,
        )
        # Move cylinder so its base is at origin (default is centered)
        cylinder.translate([0, 0, shaft_length / 2])

        # Create cone (head) - default orientation is along Z-axis
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=head_radius,
            height=head_length,
            resolution=20,
            split=1,
        )
        # Move cone to sit on top of the cylinder
        cone.translate([0, 0, shaft_length])

        # Combine shaft and head
        arrow_mesh = cylinder + cone

        # Compute and apply rotation
        rotation_matrix = _compute_arrow_rotation(direction_unit)
        arrow_mesh.rotate(rotation_matrix, center=[0, 0, 0])

        # Translate to origin
        arrow_mesh.translate(origin)

        # Set color
        color_normalized = [c / 255.0 for c in color]
        arrow_mesh.paint_uniform_color(color_normalized)

        # Compute normals for proper lighting
        arrow_mesh.compute_vertex_normals()

        return arrow_mesh

    except (GeometryError, InvalidShapeError):
        raise
    except Exception as e:
        raise RenderingError(f"Failed to create arrow mesh: {e}") from e


def create_arrow_from_points(
    start: np.ndarray,
    end: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    head_length_ratio: float = 0.2,
    head_width_ratio: float = 0.5,
) -> O3dLineSet:
    """
    Create a 3D wireframe arrow from start point to end point.

    Args:
        start (np.ndarray, [3,]): Starting point of the arrow.
        end (np.ndarray, [3,]): End point (tip) of the arrow.
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        head_length_ratio (float): Ratio of head length to total length (0.0 to 1.0).
        head_width_ratio (float): Ratio of head width to head length.

    Returns:
        O3dLineSet: Wireframe arrow geometry.

    Raises:
        InvalidShapeError: If start or end is not a 3D vector.
        GeometryError: If start and end are the same point.
        RenderingError: If arrow creation fails.

    Example:
        >>> arrow = create_arrow_from_points(
        ...     start=np.array([0, 0, 0]),
        ...     end=np.array([1, 1, 1]),
        ...     color=(255, 0, 0),
        ... )
    """
    direction, length = _validate_arrow_points(start, end)

    return create_arrow(
        origin=start,
        direction=direction,
        length=length,
        color=color,
        head_length_ratio=head_length_ratio,
        head_width_ratio=head_width_ratio,
    )


def create_arrow_mesh_from_points(
    start: np.ndarray,
    end: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    head_length_ratio: float = 0.2,
    shaft_radius: float = 0.02,
    head_radius: float = 0.05,
) -> O3dTriMesh:
    """
    Create a 3D solid mesh arrow from start point to end point.

    Args:
        start (np.ndarray, [3,]): Starting point of the arrow.
        end (np.ndarray, [3,]): End point (tip) of the arrow.
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        head_length_ratio (float): Ratio of head length to total length (0.0 to 1.0).
        shaft_radius (float): Radius of the arrow shaft.
        head_radius (float): Radius of the arrow head base.

    Returns:
        O3dTriMesh: Solid mesh arrow geometry.

    Raises:
        InvalidShapeError: If start or end is not a 3D vector.
        GeometryError: If start and end are the same point.
        RenderingError: If arrow creation fails.

    Example:
        >>> arrow = create_arrow_mesh_from_points(
        ...     start=np.array([0, 0, 0]),
        ...     end=np.array([1, 1, 1]),
        ...     color=(255, 0, 0),
        ... )
    """
    direction, length = _validate_arrow_points(start, end)

    return create_arrow_mesh(
        origin=start,
        direction=direction,
        length=length,
        color=color,
        head_length_ratio=head_length_ratio,
        shaft_radius=shaft_radius,
        head_radius=head_radius,
    )
