"""
Image-textured 3D geometry creation.

This module provides functions to create 3D geometries with image textures:
- Image planes
- Image spheres (for equirectangular images)
"""

from typing import Optional

import numpy as np
import open3d as o3d

from ..common.exceptions import InvalidShapeError, RenderingError
from .common import O3dTriMesh


def create_image_plane(
    image: np.ndarray,
    plane_size: tuple[float, float],
    z: float = 1.0,
    pose: Optional[np.ndarray] = None,
) -> O3dTriMesh:
    """
    Create an image plane from an image.

    Args:
        image (np.ndarray, [H,W,3]): Image.
        plane_size (Tuple[float,float]): Image plane size (width, height).
        z (float): z-value of image plane.
        pose (Optional[np.ndarray], [4,4]): 4x4 transformation matrix.

    Returns:
        O3dTriMesh: A TriangleMesh object representing the image plane.

    Raises:
        InvalidShapeError: If pose is not a 4x4 transformation matrix.
        RenderingError: If image plane creation fails.

    Details:
        - The center of plane is (0,0) before transformed.
          i.e. plane size is [2,3], then x ~ [-1,1] and y ~ [-1.5,1.5] and z = z
        - # of vertices is H*W
        - # of faces is H*W*2 (two faces per pixel)
    """
    if pose is not None:
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise InvalidShapeError(
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}. "
                "Please provide a valid 4x4 numpy array."
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
            mesh.transform(pose)

        return mesh
    except Exception as e:
        raise RenderingError(f"Failed to create image plane: {e}") from e


def create_image_sphere(
    image: np.ndarray,
    radius: float = 1.0,
    pose: Optional[np.ndarray] = None,
) -> O3dTriMesh:
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
                f"Pose must be a 4x4 transformation matrix, got {type(pose)} "
                f"with shape {getattr(pose, 'shape', 'unknown')}. "
                "Please provide a valid 4x4 numpy array."
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
            [
                np.concatenate([v1, v3, v4], axis=1),
                np.concatenate([v1, v4, v2], axis=1),
            ]
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
