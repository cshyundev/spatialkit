"""
Module Name: api.py

Description:
High-level API for 3D visualization that accepts Camera, Pose, and Transform instances directly.
This module wraps the low-level functions in indicators.py to provide a more convenient interface.

Main Functions:
- create_camera_vis: Create camera indicator from Camera and Pose/Transform instances.
- create_pose_vis: Create coordinate frame from Pose/Transform instance.
- create_trajectory_vis: Create trajectory line from a list of Pose/Transform instances.

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT License
"""

from typing import Union, Optional, Tuple, List, Any
import numpy as np
import open3d as o3d

from ..camera.radial_base import RadialCamera
from ..geom.pose import Pose
from ..geom.tf import Transform
from ..common.exceptions import GeometryError, RenderingError
from ..vis2d.convert import generate_rainbow_colors
from .indicators import (
    create_camera_indicator_frame,
    create_coordinate,
)
from .common import O3dLineSet


def create_camera_vis(
    camera: RadialCamera,
    pose: Optional[Union[Pose, Transform]] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    scale: float = 0.5,
    image: Optional[np.ndarray] = None,
    coordinate_frame: str = "opencv",
) -> Any:
    """
    Create a camera indicator from Camera and Pose/Transform instances.

    Args:
        camera (RadialCamera): Camera instance (PerspectiveCamera, FisheyeCamera, etc.).
        pose (Optional[Union[Pose, Transform]]): Pose or Transform instance for camera position.
        color (Tuple[int, int, int]): RGB color for the camera indicator.
        scale (float): Scale factor for the camera indicator.
        image (Optional[np.ndarray], [H,W,3]): RGB image to display on the image plane.
        coordinate_frame (str): Coordinate frame convention ("opencv", "ros", "opengl").

    Returns:
        O3dLineSet or tuple: Camera indicator geometry, or (indicator, image_plane) if image provided.

    Raises:
        GeometryError: If camera type is not supported or coordinate_frame is invalid.
        RenderingError: If camera indicator creation fails.

    Example:
        >>> from spatialkit.camera import PerspectiveCamera
        >>> from spatialkit.geom import Pose
        >>> cam = PerspectiveCamera.from_fov([640, 480], 60.0)
        >>> pose = Pose.from_identity()
        >>> indicator = create_camera_vis(cam, pose)
    """
    if not isinstance(camera, RadialCamera):
        raise GeometryError(
            f"Unsupported camera type: {type(camera).__name__}. "
            f"Only RadialCamera subclasses (PerspectiveCamera, FisheyeCamera, etc.) are supported."
        )

    cam_size = (camera.width, camera.height)
    focal_length = (camera.fx + camera.fy) / 2.0

    pose_mat = None
    if pose is not None:
        pose_mat = pose.mat44()

    return create_camera_indicator_frame(
        cam_size=cam_size,
        focal_length=focal_length,
        color=color,
        scale=scale,
        pose=pose_mat,
        image=image,
        coordinate_frame=coordinate_frame,
    )


def create_pose_vis(
    pose: Union[Pose, Transform],
    scale: float = 1.0,
    radius: float = 0.02,
) -> Any:
    """
    Create a coordinate frame from Pose or Transform instance.

    Args:
        pose (Union[Pose, Transform]): Pose or Transform instance.
        scale (float): Length of each axis.
        radius (float): Radius of the cylinders representing the axes.

    Returns:
        O3dTriMesh: A TriangleMesh object representing the coordinate frame.

    Raises:
        RenderingError: If coordinate frame creation fails.

    Example:
        >>> from spatialkit.geom import Pose
        >>> pose = Pose.from_identity()
        >>> coord = create_pose_vis(pose, scale=0.5)
    """
    pose_mat = pose.mat44()
    return create_coordinate(scale=scale, radius=radius, pose=pose_mat)


def create_trajectory_vis(
    poses: List[Union[Pose, Transform]],
    color: Union[Tuple[int, int, int], str] = "rainbow",
) -> Any:
    """
    Create a trajectory line from a list of Pose/Transform instances.

    Args:
        poses (List[Union[Pose, Transform]]): List of Pose or Transform instances.
        color: Color specification for the trajectory.
            - (R, G, B) tuple: Single color for the entire trajectory.
            - "rainbow": Gradient from red (start) to violet (end).

    Returns:
        O3dLineSet: A LineSet object representing the trajectory.

    Raises:
        GeometryError: If color specification is invalid or poses list is too short.
        RenderingError: If trajectory creation fails.

    Example:
        >>> from spatialkit.geom import Pose
        >>> poses = [Pose.from_identity() for _ in range(10)]
        >>> trajectory = create_trajectory_vis(poses, color="rainbow")
        >>> trajectory_single = create_trajectory_vis(poses, color=(255, 255, 0))
    """
    if len(poses) < 2:
        raise GeometryError(
            f"At least 2 poses are required to create a trajectory, got {len(poses)}."
        )

    # Validate color parameter
    if isinstance(color, str):
        if color != "rainbow":
            raise GeometryError(
                f"Invalid color string '{color}'. Only 'rainbow' is supported."
            )
        use_rainbow = True
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        use_rainbow = False
    else:
        raise GeometryError(
            f"Invalid color specification: {color}. "
            f"Use (R, G, B) tuple or 'rainbow' string."
        )

    try:
        # Extract translation vectors from poses
        points = np.array([pose.t.flatten() for pose in poses], dtype=np.float64)

        # Create line indices (connect consecutive points)
        num_lines = len(poses) - 1
        lines = np.array([[i, i + 1] for i in range(num_lines)], dtype=np.int32)

        # Generate colors
        if use_rainbow:
            rainbow = generate_rainbow_colors(num_lines)
            colors = np.array(rainbow, dtype=np.float64) / 255.0
        else:
            single_color = np.array(color, dtype=np.float64) / 255.0
            colors = np.tile(single_color, (num_lines, 1))

        # Create LineSet
        line_set = O3dLineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set
    except Exception as e:
        raise RenderingError(f"Failed to create trajectory visualization: {e}") from e
