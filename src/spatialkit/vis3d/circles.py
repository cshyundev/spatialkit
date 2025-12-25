"""
3D circle primitives for visualization.

This module provides functions to create circle geometries:
- Thick circle mesh (using cylinders)
"""

from typing import List, Tuple

import numpy as np
import open3d as o3d

from ..common.exceptions import GeometryError, RenderingError
from .common import O3dTriMesh


def create_thick_circle_mesh(
    radius: float = 1.0,
    tube_radius: float = 0.05,
    segments: int = 64,
    plane: str = "xy",
    color: Tuple[int, int, int] = (255, 0, 0),
) -> List[O3dTriMesh]:
    """
    Create a thick circle mesh using multiple cylinders (to simulate thick lines).

    Args:
        radius (float): The radius of the circle.
        tube_radius (float): The radius (thickness) of the tubes forming the circle.
        segments (int): The number of segments to divide the circle.
        plane (str): The plane in which the circle is drawn. ["xy", "yz", "xz"]
        color (Tuple[int, int, int]): RGB color (0-255 scale).

    Returns:
        List[O3dTriMesh]: A list of cylinder meshes forming the thick circle.

    Raises:
        GeometryError: If plane parameter is invalid.
        RenderingError: If thick circle mesh creation fails.

    Example:
        >>> circle = create_thick_circle_mesh(radius=1.0, plane="xy", color=(255, 0, 0))
    """
    if plane not in ["xy", "yz", "xz"]:
        raise GeometryError(
            f"Invalid plane '{plane}'. Supported planes are 'xy', 'yz', 'xz'. "
            "Please provide a valid plane specification."
        )

    try:
        theta = np.linspace(0, 2 * np.pi, segments)

        if plane == "xy":
            points = np.array(
                [[radius * np.cos(t), radius * np.sin(t), 0] for t in theta]
            )
        elif plane == "yz":
            points = np.array(
                [[0, radius * np.cos(t), radius * np.sin(t)] for t in theta]
            )
        else:  # xz
            points = np.array(
                [[radius * np.cos(t), 0, radius * np.sin(t)] for t in theta]
            )

        line_set = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            cylinder = O3dTriMesh.create_cylinder(tube_radius, np.linalg.norm(p2 - p1))
            cylinder.translate((p1 + p2) / 2)
            direction = (p2 - p1) / np.linalg.norm(p2 - p1)
            axis = np.cross([0, 0, 1], direction)
            angle = np.arccos(np.dot([0, 0, 1], direction))
            cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle))

            # Set the color of the cylinder
            cylinder.paint_uniform_color([c / 255.0 for c in color])

            line_set.append(cylinder)

        return line_set
    except Exception as e:
        raise RenderingError(f"Failed to create thick circle mesh: {e}") from e
