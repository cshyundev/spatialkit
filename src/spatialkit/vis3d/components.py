"""
Module Name: o3dcomp.py

Description:
This module provides utility functions for creating and manipulating 3D geometries using Open3D.
It includes functions to create coordinate frames, camera indicators, image planes, lines, and textured spheres.

Main Functions:
- create_coordinate: Creates a coordinate frame with RGB colored axes.
- create_camera_indicator_frame: Creates a camera indicator with optional image plane.
- create_image_plane: Creates an image plane from a given image.
- create_line_from_points: Creates a line connecting two 3D points.
- create_image_sphere: Creates a textured sphere from an equirectangular image.
- create_thick_circle_mesh: Creates a thick circle mesh using cylinders.
- create_spherical_camera_indicator_frame: Creates a spherical camera indicator frame.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT License
"""

from typing import Optional, Tuple, List, Any
import numpy as np
import open3d as o3d
from .common import O3dLineSet, O3dTriMesh
from ..common.exceptions import InvalidShapeError, GeometryError, RenderingError


def create_coordinate(
    scale: float = 1.0, radius: float = 0.02, pose: Optional[np.ndarray] = None
) -> Any:
    """
    Create a coordinate frame with RGB colors for the axes.

    Args:
        size (float): Length of each axis.
        radius (float): Radius of the cylinders representing the axes.
        pose (Optional[np.ndarray], [4,4]): 4x4 transformation matrix.

    Return:
        O3dTriMesh: A TriangleMesh object representing the coordinate frame.
        
    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If coordinate frame creation fails.
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
            )

    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
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
    color: Optional[Tuple[int, int, int]] = (255, 0, 0),
    scale: float = 0.5,
    pose: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
) -> Any:
    """
    Create a camera indicator.

    Args:
        cam_size (Tuple[int,int]): Camera size (width, height).
        focal_length (float): focal_length.
        color (Optional[Tuple[int,int,int]]): RGB color.
        scale (float): camera indicator scale.
        pose (Optional[np.ndarray], [4,4]): 4x4 transformation matrix.
        image (Optional[np.ndarray], [H,W,3]): RGB image.

    Return:
        O3dLineSet: A TriangleMesh object representing the coordinate frame.

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix or 
                          if image dimensions don't match cam_size.
        RenderingError: If camera indicator creation fails.

    Details:
    - cam_size and image's resolution are same.
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
            )
    
    if image is not None:
        if not isinstance(image, np.ndarray) or image.shape[0:2] != cam_size[::-1]:
            raise InvalidShapeError(
                f"Image dimensions must match camera size. Expected image shape (H,W) = {cam_size[::-1]}, "
                f"got {getattr(image, 'shape', 'unknown')}. Please ensure image resolution matches cam_size."
            )

    if image is not None:
        if not isinstance(image, np.ndarray) or image.shape[0:2] != cam_size[::-1]:
            raise InvalidShapeError(
                f"Image dimensions must match camera size. Expected image shape (H,W) = {cam_size[::-1]}, "
                f"got {getattr(image, 'shape', 'unknown')}. Please ensure image resolution matches cam_size."
            )

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
            [[0, 1], [1, 2], [2, 0]],  # Indicator base  # Indicator right  # Indicator left
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
        color = np.array(color) / 255.0

        # Apply color to all edges
        colors = [color for _ in range(len(edges))]
        cam_indicator.colors = o3d.utility.Vector3dVector(colors)

        if pose is not None:
            cam_indicator.transform(pose)

        if image is not None:
            image_plane = create_image_plane(image, (w * scale, h * scale), scale, pose)
            return cam_indicator, image_plane
        return cam_indicator
    except Exception as e:
        raise RenderingError(f"Failed to create camera indicator frame: {e}") from e


def create_image_plane(
    image: np.ndarray,
    plane_size: Tuple[float, float],
    z: float = 1.0,
    pose: Optional[np.ndarray] = None,
) -> Any:
    """
    Create an image plane from Image

    Args:
        image (np.ndarray, [H,W,3]): Image.
        plane_size (Tuple[float,float]): Image plane size (width,height).
        z (float): z-value of image plane
        pose (Optional[np.ndarray], [4,4]): 4x4 transformation matrix.

    Return:
        O3dTriMesh: A TriangleMesh object representing the coordinate frame.

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If image plane creation fails.

    Details:
    - The center of plane is (0,0) before transformed.
      i.e. palne size is [2,3], then x ~ [-1,1] and y ~ [-1.5,1.5] and z = 0
    - # of vertices is H*W
    - # of faces is H*W*2 (two faces per pixel)
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
            )

    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
            )

    try:
        height, width = image.shape[:2]

        x, y = np.meshgrid(
            np.linspace(-plane_size[0] / 2.0, plane_size[0] / 2.0, width),
            np.linspace(-plane_size[1] / 2.0, plane_size[1] / 2.0, height),
        )
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        vertices = np.concatenate([x, y, np.ones_like(x) * z], axis=1)  # N * 3
        if image.dtype == np.uint8:
            colors = image.astype(np.float64).reshape(-1, 3) / 255.0
        else:  # floating points
            colors = image.reshape(-1, 3)

        indices = np.arange(height * width).reshape(height, width)
        v1 = indices[:-1, :-1].reshape(-1, 1)
        v2 = indices[:-1, 1:].reshape(-1, 1)
        v3 = indices[1:, :-1].reshape(-1, 1)
        v4 = indices[1:, 1:].reshape(-1, 1)

        faces = np.vstack(
            [
                np.concatenate([v1, v3, v4], axis=1),
                np.concatenate([v1, v4, v2], axis=1),
            ]
        )

        mesh = O3dTriMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        if pose is not None:
            mesh.transform(pose)  # Transform

        return mesh
    except Exception as e:
        raise RenderingError(f"Failed to create image plane: {e}") from e


def create_line_from_points(
    point1: np.ndarray, point2: np.ndarray, color: Optional[Tuple[int]] = None
) -> Any:
    """
    Create a line connecting two 3D points.

    Args:
        point1 (np.ndarray, [3,]): [x, y, z] coordinates of the first point.
        point2 (np.ndarray, [3,]): [x, y, z] coordinates of the second point.
        color (Optional[Tuple[int]]): RGB color.

    Return:
        O3dLineSet: A LineSet object representing the line between the two points.
        
    Raises:
        InvalidShapeError: If points don't have correct 3D coordinates.
        RenderingError: If line creation fails.
    """
    if not isinstance(point1, np.ndarray) or point1.shape != (3,):
        raise InvalidShapeError(
            f"point1 must be a 3D coordinate array with shape (3,), got {type(point1)} with shape {getattr(point1, 'shape', 'unknown')}. "
            f"Please provide a valid [x, y, z] coordinate array."
        )
    
    if not isinstance(point2, np.ndarray) or point2.shape != (3,):
        raise InvalidShapeError(
            f"point2 must be a 3D coordinate array with shape (3,), got {type(point2)} with shape {getattr(point2, 'shape', 'unknown')}. "
            f"Please provide a valid [x, y, z] coordinate array."
        )

    try:
        points = np.array([point1, point2], dtype=np.float64)
        lines = np.array([[0, 1]], dtype=np.int32)

        line_set = O3dLineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        if color is None:
            color = np.random.rand(3)
        else:
            color = np.array(color) / 255.0

        line_set.colors = o3d.utility.Vector3dVector([color])

        return line_set
    except Exception as e:
        raise RenderingError(f"Failed to create line from points: {e}") from e


def create_image_sphere(
    image: np.ndarray, radius: float = 1.0, pose: Optional[np.ndarray] = None
) -> Any:
    """
    Create a textured sphere from an equirectangular image.

    Args:
        image (np.ndarray, [H,W,3]): Equirectangular image.
        radius (float): Radius of the sphere.
        pose (Optional[np.ndarray], [4,4]): 4x4 transformation matrix.

    Returns:
        O3dTriMesh: A TriangleMesh object representing the textured sphere.
        
    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If image sphere creation fails.
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
            )

    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
            )

    try:
        height, width = image.shape[:2]

        u, v = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

        theta = (u - 0.5) * np.pi * 2.0
        phi = (v - 0.5) * np.pi

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(phi)
        z = radius * np.cos(theta) * np.cos(phi)

        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        indices = np.arange(height * width).reshape(height, width)
        v1 = indices[:-1, :-1].reshape(-1, 1)
        v2 = indices[:-1, 1:].reshape(-1, 1)
        v3 = indices[1:, :-1].reshape(-1, 1)
        v4 = indices[1:, 1:].reshape(-1, 1)

        faces = np.vstack(
            [np.concatenate([v1, v3, v4], axis=1), np.concatenate([v1, v4, v2], axis=1)]
        )

        if image.dtype == np.uint8:
            colors = image.astype(np.float64).reshape(-1, 3) / 255.0
        else:
            colors = image.reshape(-1, 3)

        mesh = O3dTriMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        if pose is not None:
            mesh.transform(pose)

        return mesh
    except Exception as e:
        raise RenderingError(f"Failed to create image sphere: {e}") from e


def create_thick_circle_mesh(
    radius: float = 1.0,
    tube_radius: float = 0.05,
    segments: int = 64,
    plane: str = "xy",
    color: Optional[Tuple[int, int, int]] = (255, 0, 0),
) -> List[Any]:
    """
    Create a thick circle mesh using multiple cylinders (to simulate thick lines) and apply color.

    Args:
        radius (float): The radius of the circle.
        tube_radius (float): The radius (thickness) of the tubes forming the circle.
        segments (int): The number of segments to divide the circle.
        plane (str): The plane in which the circle is drawn. [xy, yz ,xz]
        color (Optional[Tuple[int, int, int]]): The RGB color of the circle, with values in the range [0, 255].

    Returns:
        list: A list of cylinder meshes forming the thick circle, with the specified color.
        
    Raises:
        GeometryError: If plane parameter is invalid.
        RenderingError: If thick circle mesh creation fails.
    """
    if plane not in ["xy", "yz", "xz"]:
        raise GeometryError(
            f"Invalid plane '{plane}'. Supported planes are 'xy', 'yz', 'xz'. "
            f"Please provide a valid plane specification."
        )

    try:
        theta = np.linspace(0, 2 * np.pi, segments)

        if plane == "xy":
            points = np.array([[radius * np.cos(t), radius * np.sin(t), 0] for t in theta])
        elif plane == "yz":
            points = np.array([[0, radius * np.cos(t), radius * np.sin(t)] for t in theta])
        else:  # xz
            points = np.array([[radius * np.cos(t), 0, radius * np.sin(t)] for t in theta])

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


def create_spherical_camera_indicator_frame(
    radius: float = 1.0,
    pose: Optional[np.ndarray] = None,
    tube_radius: float = 0.05,
    color: Optional[Tuple[int, int, int]] = (255, 0, 0),
    image: Optional[np.ndarray] = None,
) -> List[Any]:
    """
    Create a spherical camera indicator frame with two perpendicular thick circles (one in XY and one in YZ planes),
    with specified color.

    Args:
        radius (float): The radius of the camera frame.
        pose (Optional[np.ndarray], [4,4]): Optional 4x4 transformation matrix to position the frame.
        tube_radius (float): The thickness of the tubes representing the circles.
        color (Optional[Tuple[int, int, int]]): The RGB color of the circles, with values in the range [0, 255].
        image (Optional[np.ndarray]): Image to be used for the sphere.

    Returns:
        o3d.geometry.TriangleMesh: A combined TriangleMesh object representing the spherical camera indicator.

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If spherical camera indicator creation fails.

    Details:
    - The frame consists of two thick circles, one in the XY plane and one in the YZ plane, centered at (0,0,0).
    - The frame is transformed if a pose matrix is provided.
    - The circles are colored using the input `color`.
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} with shape {getattr(pose, 'shape', 'unknown')}. "
                f"Please provide a valid 4x4 numpy array."
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
        raise RenderingError(f"Failed to create spherical camera indicator frame: {e}") from e
