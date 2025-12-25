"""
3D visualization indicators.

This module provides functions to create visualization indicators:
- Coordinate frames
- Camera indicators (perspective and spherical)
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import open3d as o3d

from ..common.constant import OPENCV_TO_OPENGL, OPENCV_TO_ROS
from ..common.exceptions import GeometryError, InvalidShapeError, RenderingError
from .common import O3dLineSet, O3dTriMesh
from .circles import create_thick_circle_mesh
from .textures import create_image_plane, create_image_sphere


def create_coordinate(
    scale: float = 1.0,
    radius: float = 0.02,
    pose: Optional[np.ndarray] = None,
) -> O3dTriMesh:
    """
    Create a coordinate frame with RGB colors for the axes.

    Args:
        scale (float): Length of each axis.
        radius (float): Radius of the cylinders representing the axes.
        pose (Optional[np.ndarray], [4,4]): 4x4 transformation matrix.

    Returns:
        O3dTriMesh: A TriangleMesh object representing the coordinate frame.

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If coordinate frame creation fails.
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}. "
                "Please provide a valid 4x4 numpy array."
            )

    try:
        mesh_frame = O3dTriMesh()

        # X axis (red)
        x_axis = O3dTriMesh.create_cylinder(radius, scale)
        x_axis.paint_uniform_color([1, 0, 0])
        x_axis.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]))
        x_axis.translate([scale / 2, 0, 0])
        mesh_frame += x_axis

        # Y axis (green)
        y_axis = O3dTriMesh.create_cylinder(radius, scale)
        y_axis.paint_uniform_color([0, 1, 0])
        y_axis.rotate(o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0]))
        y_axis.translate([0, scale / 2, 0])
        mesh_frame += y_axis

        # Z axis (blue)
        z_axis = O3dTriMesh.create_cylinder(radius, scale)
        z_axis.paint_uniform_color([0, 0, 1])
        z_axis.translate([0, 0, scale / 2])
        mesh_frame += z_axis

        # Apply the transformation to the entire frame
        if pose is not None:
            mesh_frame.transform(pose)

        return mesh_frame
    except Exception as e:
        raise RenderingError(f"Failed to create coordinate frame: {e}") from e


def create_camera_indicator_frame(
    cam_size: Tuple[int, int],
    focal_length: float,
    color: Tuple[int, int, int] = (255, 0, 0),
    scale: float = 0.5,
    pose: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    coordinate_frame: str = "opencv",
) -> Any:
    """
    Create a camera indicator.

    Args:
        cam_size (Tuple[int,int]): Camera size (width, height).
        focal_length (float): focal_length.
        color (Tuple[int,int,int]): RGB color (0-255 scale).
        scale (float): camera indicator scale.
        pose (Optional[np.ndarray], [4,4]): 4x4 transformation matrix.
        image (Optional[np.ndarray], [H,W,3]): RGB image.
        coordinate_frame (str): Coordinate frame convention. Supported values:
            - "opencv": Z-forward, Y-down (default)
            - "ros": X-forward, Z-up
            - "opengl": Z-backward, Y-up

    Returns:
        O3dLineSet: A LineSet object representing the camera indicator.
        If image is provided, returns tuple (O3dLineSet, O3dTriMesh).

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix or
                          if image dimensions don't match cam_size.
        GeometryError: If coordinate_frame is not a supported value.
        RenderingError: If camera indicator creation fails.

    Details:
        - cam_size and image's resolution should be the same.
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}. "
                "Please provide a valid 4x4 numpy array."
            )

    if image is not None:
        if not isinstance(image, np.ndarray) or image.shape[0:2] != cam_size[::-1]:
            raise InvalidShapeError(
                f"Image dimensions must match camera size. Expected image shape (H,W) = {cam_size[::-1]}, "
                f"got {getattr(image, 'shape', 'unknown')}. Please ensure image resolution matches cam_size."
            )

    # Validate coordinate_frame
    valid_frames = ("opencv", "ros", "opengl")
    if coordinate_frame not in valid_frames:
        raise GeometryError(
            f"Invalid coordinate_frame '{coordinate_frame}'. "
            f"Supported frames are: {valid_frames}."
        )

    # Get coordinate transformation matrix
    if coordinate_frame == "opencv":
        coord_transform = np.eye(4, dtype=np.float32)
    elif coordinate_frame == "ros":
        coord_transform = OPENCV_TO_ROS
    else:  # opengl
        coord_transform = OPENCV_TO_OPENGL

    # Compute final transformation
    if pose is not None:
        final_transform = pose @ coord_transform
    else:
        final_transform = coord_transform

    try:
        w = cam_size[0] / focal_length
        h = cam_size[1] / focal_length

        # Define the vertices of the pyramid with fixed base and height
        cam_vertices = (
            np.array(
                [
                    [-w / 2.0, -h / 2.0, 1],  # Bottom-left
                    [w / 2.0, -h / 2.0, 1],  # Bottom-right
                    [w / 2.0, h / 2.0, 1],  # Top-right
                    [-w / 2.0, h / 2.0, 1],  # Top-left
                    [0, 0, 0],  # Apex at (0, 0, 0)
                ],
                dtype=np.float32,
            )
            * scale
        )

        # Define the edges of the pyramid
        cam_edges = np.array(
            [
                [0, 1],  # Bottom edge
                [1, 2],  # Bottom edge
                [2, 3],  # Bottom edge
                [3, 0],  # Bottom edge
                [0, 4],  # Side edge
                [1, 4],  # Side edge
                [2, 4],  # Side edge
                [3, 4],  # Side edge
            ],
            dtype=np.int32,
        )

        # Indicator vertices
        indicator_vertices = (
            np.array(
                [
                    [-w / 8.0, -h / 2.0, 1],  # Indicator top-left
                    [w / 8, -h / 2.0, 1],  # Indicator top-right
                    [0, -h / 1.6, 1],  # Indicator top
                ],
                dtype=np.float32,
            )
            * scale
        )

        # Indicator edges
        indicator_edges = np.array(
            [[0, 1], [1, 2], [2, 0]],
            dtype=np.int32,
        )

        # Combine vertices and edges
        vertices = np.vstack((cam_vertices, indicator_vertices))
        edges = np.vstack((cam_edges, indicator_edges + len(cam_vertices)))

        # Create a LineSet object
        cam_indicator = O3dLineSet()
        cam_indicator.points = o3d.utility.Vector3dVector(vertices)
        cam_indicator.lines = o3d.utility.Vector2iVector(edges)

        # Convert color from 255 scale to 0-1 scale for Open3D
        color_normalized = np.array(color) / 255.0

        # Apply color to all edges
        colors = [color_normalized for _ in range(len(edges))]
        cam_indicator.colors = o3d.utility.Vector3dVector(colors)

        # Apply final transformation (coordinate frame + pose)
        cam_indicator.transform(final_transform)

        if image is not None:
            image_plane = create_image_plane(
                image, (w * scale, h * scale), scale, final_transform
            )
            return cam_indicator, image_plane
        return cam_indicator
    except Exception as e:
        raise RenderingError(f"Failed to create camera indicator frame: {e}") from e


def create_spherical_camera_indicator_frame(
    radius: float = 1.0,
    pose: Optional[np.ndarray] = None,
    tube_radius: float = 0.05,
    color: Tuple[int, int, int] = (255, 0, 0),
    image: Optional[np.ndarray] = None,
) -> List[Any]:
    """
    Create a spherical camera indicator frame with two perpendicular thick circles.

    Creates a frame with one circle in the XY plane and one in the YZ plane,
    centered at (0,0,0).

    Args:
        radius (float): The radius of the camera frame.
        pose (Optional[np.ndarray], [4,4]): Optional 4x4 transformation matrix.
        tube_radius (float): The thickness of the tubes representing the circles.
        color (Tuple[int, int, int]): RGB color (0-255 scale).
        image (Optional[np.ndarray]): Image to be used for the sphere texture.

    Returns:
        List[Any]: A list of geometry objects representing the spherical camera indicator.

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If spherical camera indicator creation fails.
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}. "
                "Please provide a valid 4x4 numpy array."
            )

    try:
        circle_xy = create_thick_circle_mesh(
            radius, tube_radius, segments=64, plane="xy", color=color
        )
        circle_yz = create_thick_circle_mesh(
            radius, tube_radius, segments=64, plane="yz", color=color
        )

        geometry_list = circle_xy + circle_yz

        if pose is not None:
            for geom in geometry_list:
                geom.transform(pose)

        if image is not None:
            geometry_list.append(create_image_sphere(image, radius, pose))

        return geometry_list
    except Exception as e:
        raise RenderingError(
            f"Failed to create spherical camera indicator frame: {e}"
        ) from e
