"""
Module Name: pointcloud.py

Description:
This module provides functions for converting between 3D point clouds and depth maps.
Supports multiple depth representation types (MPI, MSI, MCI) for various camera models.

Supported Functions:
    - Convert 3D point cloud to depth map
    - Convert depth map to 3D point cloud

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.0

License: MIT LICENSE
"""

from typing import Union, Optional, Tuple
import numpy as np
from ..ops.uops import *
from ..ops.umath import norm, sqrt
from ..common.constant import EPSILON
from ..common.logger import LOG_CRITICAL, LOG_WARN
from .pose import Pose
from .tf import Transform
from ..camera import Camera, CamType


def convert_point_cloud_to_depth(pcd: np.ndarray, cam: Camera, map_type: str = "MPI") -> np.ndarray:
    """
    Convert 3D point to depth map of given camera.

    Args:
        pcd (np.ndarray, [N,3] or [3,N]): 3D Point Cloud
        cam: (Camera): Camera Instance with [H,W] resolution.
        map_type:(str): Depth map represntation type (see Details).

    Returns:
        depth_map (np.ndarray, [H,W]): depth map converted from point cloud

    Raises:
        ValueError: If unsupported map_type is provided.

    Details:
    - Available map_type: MPI, MSI, MCI
    - Multi-Plane Image (MPI): Depth = Z
    - Multi-Spherical Image (MSI): Depth = sqrt(X^2 + Y^2 + Z^2)
    - Multi-Cylinder Image (MCI): Depth = sqrt(X^2 + Z^2)
    - The depth map stores the smallest depth value for each converted pixel coordinate.
    """

    if map_type.lower() not in ["mpi", "msi", "mci"]:
        LOG_CRITICAL(f"Unsupported Depth Map Type, {map_type}.")
        raise ValueError(f"Unsupported Depth Map Type, {map_type}.")

    if pcd.shape[0] != 3:  # pcd's shape = [N,3]
        pcd = swapaxes(pcd, 0, 1)  # convert pcd's shape as [3,N]

    if map_type.lower() == "mpi":  # Depth = Z
        depth = pcd[2, :]
    elif map_type.lower() == "msi":  # Depth = sqrt(X^2 + Y^2 + Z^2)
        depth = norm(pcd, dim=0)
    else:  # Depth = sqrt(X^2 + Z^2)
        depth = sqrt(pcd[0, :] ** 2 + pcd[2, :] ** 2)

    uv, mask = cam.convert_to_pixels(pcd)  # [2,N], [N,]

    # remain valid pixel coords and these depth
    uv = uv[:, mask]
    depth = depth[mask]

    depth_map = np.full((cam.width * cam.height), np.inf)
    indices = uv[1, :] * cam.width + uv[0, :]
    np.minimum.at(depth_map, indices, depth)
    depth_map[depth_map == np.inf] = 0.0
    depth_map = depth_map.reshape((cam.height, cam.width))
    return depth_map


def convert_depth_to_point_cloud(
    depth: np.ndarray,
    cam: Camera,
    image: Optional[np.ndarray] = None,
    map_type: str = "MPI",
    pose: Optional[Union[Pose, Transform]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert depth map to point cloud.

    Args:
        depth (np.ndarray, [H,W]): depth map.
        cam (Camera): Camera Instance with [H,W] resolution.
        image (np.ndarray, [H,W,3], optional): color image.
        map_type (str): Depth map representation type (see Details).
        pose (Pose or Transform, optional): Transform instance.

    Returns:
        pcd (np.ndarray, [N,3]): 3D Point Cloud
        colors (np.ndarray, [N,3]): Point Cloud's color if image was given

    Raises:
        ValueError: If depth/image resolution doesn't match camera or unsupported map_type.

    Details:
    - Available map_type: MPI, MSI, MCI
    - Multi-Plane Image (MPI): Depth = Z
    - Multi-Spherical Image (MSI): Depth = sqrt(X^2 + Y^2 + Z^2)
    - Multi-Cylinder Image (MCI): Depth = sqrt(X^2 + Z^2)
    - Return only valid point cloud (i.e. N <= H*W).
    """
    if depth.shape != cam.hw:
        LOG_CRITICAL(
            f"Depth map's resolution must be same as camera image size, but got depth's shape={depth.shape}."
        )
        raise ValueError(
            f"Depth map's resolution must be same as camera image size, but got depth's shape={depth.shape}."
        )
    if image is not None and image.shape[0:2] != cam.hw:
        LOG_CRITICAL(
            f"Image's resolution must be same as camera image size, but got image's shape={image.shape}."
        )
        raise ValueError(
            f"Image's resolution must be same as camera image size, but got image's shape={image.shape}."
        )

    if (
        cam.cam_type in [CamType.PERSPECTIVE, CamType.OPENCVFISHEYE, CamType.THINPRISM]
        and map_type != "MPI"
    ):
        LOG_WARN(
            f"Camera type {cam.cam_type} typically expects MPI depth map, but got {map_type}. "
            f"Results may be less accurate."
        )

    rays, mask = cam.convert_to_rays()
    depth = depth.reshape(
        -1,
    )

    if map_type == "MPI":
        Z = rays[2:3, :]
        mask = logical_and(
            (Z != 0.0).reshape(
                -1,
            ),
            mask,
        )
        Z[Z == 0.0] = EPSILON
        rays = rays / Z  # set Z = 1
        pts3d = rays * depth
    elif map_type == "MSI":
        pts3d = rays * depth
    elif map_type == "MCI":
        r = sqrt(rays[0, :] ** 2 + rays[2, :] ** 2).reshape(1, -1)
        mask = logical_and(
            mask,
            (r != 0.0).reshape(
                -1,
            ),
        )
        r[r == 0.0] = EPSILON
        pts3d = rays * depth / r
    else:
        LOG_CRITICAL(f"Unsupported map_type {map_type}.")
        raise ValueError(f"Unsupported map_type {map_type}.")

    if pose is not None:
        if isinstance(pose, Pose):
            pose = Transform.from_pose(pose)
        pts3d = pose * pts3d

    pts3d = swapaxes(pts3d, 0, 1)

    valid_depth_mask = depth > 0.0

    mask = logical_and(mask, valid_depth_mask)
    pts3d = pts3d[mask, :]
    if image is not None:
        colors = image.reshape(-1, 3)[mask, :]
        return pts3d, colors
    return pts3d
