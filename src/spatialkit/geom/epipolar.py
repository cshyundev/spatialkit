"""
Module Name: epipolar.py

Description:
This module provides functions for epipolar geometry, including computation and
manipulation of essential and fundamental matrices. These matrices represent the
geometric relationship between two camera views.

Supported Functions:
    - Compute essential matrix from pose or fundamental matrix
    - Compute fundamental matrix from points or essential matrix
    - Decompose essential matrix into possible poses
    - Compute points along epipolar curves

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.0

License: MIT LICENSE
"""

from typing import Union, List, Tuple
import numpy as np
from ..ops.uops import *
from ..ops.umath import inv, svd, dehomo, determinant
from ..common.exceptions import InvalidShapeError, InvalidDimensionError, ProjectionError
from .pose import Pose
from .rotation import Rotation
from .tf import Transform
from ..camera import Camera


def compute_essential_matrix_from_pose(rel_p: Union[Pose, Transform]) -> np.ndarray:
    """
    Compute the essential matrix from the relative camera pose.

    Args:
        rel_p (Pose or Transform): Relative pose object containing rotation and translation between cameras.

    Returns:
        E (np.ndarray, [3,3]): The computed essential matrix (3x3).
    """
    skew_t = rel_p.skew_t()
    r_mat = rel_p.rot_mat()
    return convert_numpy(skew_t @ r_mat)


def compute_essential_matrix_from_fundamental(
    K1: np.ndarray, K2: np.ndarray, F: np.ndarray
) -> np.ndarray:
    """
    Compute the essential matrix from the fundamental matrix and intrinsic camera matrices.

    Args:
        K1 (np.ndarray, [3,3]): Intrinsic camera matrix for the first camera.
        K2 (np.ndarray, [3,3]): Intrinsic camera matrix for the second camera.
        F  (np.ndarray, [3,3]): Fundamental matrix.

    Returns:
        E (np.ndarray, [3,3]): The computed essential matrix (3x3).
    """
    return K2.T @ F @ K1


def compute_fundamental_matrix_from_points(
    pts1: Union[np.ndarray, List[Tuple[int]]], pts2: Union[np.ndarray, List[Tuple[int]]]
) -> np.ndarray:
    """
    Compute the fundamental matrix given point correspondences.

    Args:
        pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], [2,N] or [N,2]): Corresponding points in each image.
            Supports both float32 and float64. If dtypes differ, they are promoted to higher precision.

    Returns:
        F (np.ndarray, [3,3]): The computed fundamental matrix with promoted dtype.

    Raises:
        InvalidShapeError: If point arrays have mismatched shapes.
        InvalidDimensionError: If insufficient points or incorrect dimensions.
    """
    # Ensure pts1 and pts2 are numpy arrays
    if isinstance(pts1, list):
        pts1 = np.array(pts1).T
    if isinstance(pts2, list):
        pts2 = np.array(pts2).T

    # Promote dtypes if different (auto dtype promotion)
    if pts1.dtype != pts2.dtype:
        target_dtype = promote_types(pts1, pts2)
        pts1 = pts1.astype(target_dtype)
        pts2 = pts2.astype(target_dtype)

    if pts1.shape != pts2.shape:
        raise InvalidShapeError(
            f"Point arrays must have the same shape, got {pts1.shape} and {pts2.shape}. "
            f"Please ensure both point sets have matching dimensions."
        )
    if pts1.shape[0] != 2:
        raise InvalidDimensionError(
            f"Point arrays must have 2 coordinates (x,y), got {pts1.shape[0]}. "
            f"Expected shape: (2, N) where N is the number of points."
        )
    if pts1.shape[1] < 8:
        raise InvalidDimensionError(
            f"Fundamental matrix computation requires at least 8 point correspondences, got {pts1.shape[1]}. "
            f"Please provide at least 8 matching point pairs."
        )

    # Construct matrix A for linear equation system
    x1 = pts1[0, :].reshape(-1, 1)
    y1 = pts1[1, :].reshape(-1, 1)
    x2 = pts2[0, :].reshape(-1, 1)
    y2 = pts2[1, :].reshape(-1, 1)

    A = np.concatenate(
        [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones_like(x1)], 1
    )
    # Solve using SVD
    _, _, Vt = svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce the rank-2 constraint
    U, S, Vt = svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    return F


def compute_fundamental_matrix_from_essential(
    K1: np.ndarray, K2: np.ndarray, E: np.ndarray
) -> np.ndarray:
    """
    Compute the fundamental matrix from the essential matrix and intrinsic camera matrices.

    Args:
        K1, K2 (np.ndarray, [3,3]): Intrinsic matrices of the two cameras.
        E (np.ndarray, [3,3]): Essential matrix.

    Returns:
        F (np.ndarray, [3,3]): The computed fundamental matrix.

    Raises:
        InvalidShapeError: If insufficient or incorrect data is provided.
    """
    if not (K1.shape == (3, 3) and K2.shape == (3, 3) and E.shape == (3, 3)):
        raise InvalidShapeError(
            f"Intrinsic and Essential matrices must be 3x3, got K1: {K1.shape}, K2: {K2.shape}, E: {E.shape}. "
            f"Please provide valid 3x3 camera matrices."
        )
    # Compute fundamental matrix from essential matrix
    F = inv(K2).T @ E @ inv(K1)
    return F


def compute_fundamental_matrix_using_ransac(
    pts1: Union[np.ndarray, List[Tuple[int]]],
    pts2: Union[np.ndarray, List[Tuple[int]]],
    threshold: float = 1e-2,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, int]:
    """
    Compute the fundamental matrix given point correspondences using the RANSAC algorithm.

    Args:
        pts1,pts2 (Union[np.ndarray, List[Tuple[int, int]]], [2,N] or [N,2]): Corresponding points in the images respectively.
        threshold (float, optional): Distance threshold to determine inliers. Defaults to 1e-2.
        max_iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.

    Returns:
        best_F (np.ndarray, [3, 3]): The computed fundamental matrix.
        best_inliers (int): Number of inliers for the best fundamental matrix.

    Raises:
        InvalidShapeError: If point arrays have mismatched shapes.
        InvalidDimensionError: If insufficient points or incorrect dimensions.
    """
    # Ensure pts1 and pts2 are numpy arrays
    if isinstance(pts1, list):
        pts1 = np.array(pts1).T
    if isinstance(pts2, list):
        pts2 = np.array(pts2).T

    if pts1.shape != pts2.shape:
        raise InvalidShapeError(
            f"Point arrays must have the same shape, got {pts1.shape} and {pts2.shape}. "
            f"Please ensure both point sets have matching dimensions."
        )
    if pts1.shape[0] != 2:
        raise InvalidDimensionError(
            f"Point arrays must have 2 coordinates (x,y), got {pts1.shape[0]}. "
            f"Expected shape: (2, N) where N is the number of points."
        )
    if pts1.shape[1] < 8:
        raise InvalidDimensionError(
            f"Fundamental matrix computation requires at least 8 point correspondences, got {pts1.shape[1]}. "
            f"Please provide at least 8 matching point pairs."
        )

    best_F = None
    best_inliers = 0

    num_pts = pts1.shape[1]
    for _ in range(max_iterations):
        # Randomly select 8 points for the minimal sample set
        indices = np.random.choice(num_pts, 8, replace=False)
        sample_pts1 = pts1[:, indices]
        sample_pts2 = pts2[:, indices]

        # Compute the fundamental matrix
        F = compute_fundamental_matrix_from_points(sample_pts1, sample_pts2)

        # Calculate the number of inliers
        inliers = 0
        for i in range(num_pts):
            pt1 = np.append(pts1[:, i], 0)
            pt2 = np.append(pts2[:, i], 0)
            error = abs(pt2.T @ F @ pt1)
            if error < threshold:
                inliers += 1

        # Update the best model if current model has more inliers
        if inliers > best_inliers:
            best_F = F
            best_inliers = inliers

    return best_F, best_inliers


def decompose_essential_matrix(E: np.ndarray) -> Tuple[Transform, Transform, Transform, Transform]:
    """
    Decompose the essential matrix into possible rotations and translations.

    Args:
        E (np.ndarray, [3, 3]): Essential matrix.

    Returns:
        transform1 (Transform): First possible pose (R1, t).
        transform2 (Transform): Second possible pose (R1, -t).
        transform3 (Transform): Third possible pose (R2, t).
        transform4 (Transform): Fourth possible pose (R2, -t).
    """
    U, _, Vt = svd(E)
    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    if determinant(U) < 0:
        U = -U
    if determinant(Vt) < 0:
        Vt = -Vt

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    tf1 = Transform(t, Rotation.from_mat3(R1))
    tf2 = Transform(-t, Rotation.from_mat3(R1))
    tf3 = Transform(t, Rotation.from_mat3(R2))
    tf4 = Transform(-t, Rotation.from_mat3(R2))

    return tf1, tf2, tf3, tf4


def compute_points_for_epipolar_curve(
    pt_cam1: np.ndarray,
    cam1: Camera,
    cam2: Camera,
    rel_tf: Transform,
    depth_rng: Tuple[float, float],
    max_pts: int,
) -> np.ndarray:
    """
    Compute points for the epipolar curve from cam1 to cam2.

    Args:
        pt_cam1 (np.ndarray, [2,1]): The 2D point in cam1 image space.
        cam1 (Camera): The first camera object.
        cam2 (Camera): The second camera object.
        rel_tf (Transform): The relative transform between cam1 and cam2.
        depth_rng (Tuple[float, float]): The range of depths to consider.
        max_pts (int): The maximum number of points to compute.

    Returns:
        points (np.ndarray, [2,N]): List of valid 2D points in cam2 image space.

    Raises:
        InvalidDimensionError: If depth_rng or max_pts are invalid.
        InvalidShapeError: If pt_cam1 has incorrect shape.
        ProjectionError: If no valid ray found from the given point.
    """
    if not (isinstance(depth_rng, tuple) and len(depth_rng) == 2):
        raise InvalidDimensionError(
            f"depth_rng must be a tuple with two elements, got {type(depth_rng)} with length {len(depth_rng) if hasattr(depth_rng, '__len__') else 'unknown'}. "
            f"Please provide a tuple like (min_depth, max_depth)."
        )
    if pt_cam1.shape != (2, 1):
        raise InvalidShapeError(
            f"pt_cam1 must be a 2x1 numpy array, got shape {pt_cam1.shape}. "
            f"Please provide a 2D point in format [[x], [y]]."
        )
    if max_pts <= 0:
        raise InvalidDimensionError(
            f"max_pts must be greater than 0, got {max_pts}. "
            f"Please provide a positive integer for maximum points."
        )

    ray, mask = cam1.convert_to_rays(pt_cam1)
    if mask.sum() == 0:
        raise ProjectionError(
            "No valid ray found from the given point in cam1. "
            "The point may be outside the camera's field of view or invalid."
        )

    def compute_pixel_in_cam2(
        ray: np.ndarray, cam2: Camera, rel_tf: Transform, depth: np.ndarray
    ):
        pt3d = ray * depth
        pt3d = rel_tf * pt3d
        return cam2.convert_to_pixels(pt3d, out_subpixel=False)

    def unique_and_sort_pts2d(pts2d_cam2: np.ndarray, distance: np.ndarray):
        pts2d_cam2 = np.unique(pts2d_cam2.T, axis=0).T
        indices = np.lexsort((pts2d_cam2[0], pts2d_cam2[1]))[::-1]
        distance = distance[indices]
        pts2d_cam2 = pts2d_cam2[:, indices]
        return pts2d_cam2, distance

    distance = np.linspace(depth_rng[0], depth_rng[1], max_pts + 1)
    pts2d_cam2, mask = compute_pixel_in_cam2(ray, cam2, rel_tf, distance)

    pts2d_cam2 = pts2d_cam2[:, mask]
    distance = distance[mask]

    pts2d_cam2, distance = unique_and_sort_pts2d(pts2d_cam2, distance)

    return pts2d_cam2
