"""
Module Name: multiview.py

Description:
This module provides multi-view geometry primitives for practical 3D vision tasks,
including pose estimation, 3D reconstruction, and feature-based correspondence.
These algorithms are essential for structure-from-motion and multi-view stereo applications.

Supported Functions:
    - Solve Perspective-n-Point (PnP) for pose estimation
    - Triangulate 3D points from multi-view correspondences
    - Compute relative transforms between camera views
    - Find feature correspondences between images

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.0

License: MIT LICENSE
"""

from typing import Union, List, Tuple, Any, Optional
import cv2 as cv
import numpy as np
from ..ops.uops import *
from ..ops.umath import dehomo
from ..common.exceptions import (
    InvalidShapeError,
    InvalidDimensionError,
    CalibrationError,
    UnsupportedCameraTypeError,
)
from .pose import Pose
from .rotation import Rotation
from .tf import Transform
from ..camera import Camera
from .epipolar import compute_fundamental_matrix_from_points, compute_fundamental_matrix_using_ransac


def solve_pnp(
    pts2d: np.ndarray, pts3d: np.ndarray, cam: Camera, cv_flags: Any = None
) -> Transform:
    """
    Computes the camera pose using the Perspective-n-Point (PnP) problem solution.

    Args:
        pts2d (np.ndarray, [2,N]): 2D image points. Supports both float32 and float64.
        pts3d (np.ndarray, [3,N]): Corresponding 3D scene points. Supports both float32 and float64.
        cam (Camera): Camera instance which can be one of available models.
        cv_flags (Any, optional): Additional options for solving PnP.

    Returns:
        Transform: A Transform Instance from object coordinates to camera coordinates.

    Raises:
        UnsupportedCameraTypeError: If the camera type is not supported for PnP.
        CalibrationError: If insufficient points provided or PnP computation fails.

    Note:
        Internally converts to float64 for OpenCV compatibility.
    """
    cam_type = cam.cam_type
    unavailable_cam_types = []
    if cam_type in unavailable_cam_types:
        raise UnsupportedCameraTypeError(
            f"Camera type {cam_type.name} is not supported for PnP operation. "
            f"Please use a different camera model."
        )

    rays, mask = cam.convert_to_rays(pts2d)  # 2 * N
    rays = rays[:, mask]  # delete unavailable rays

    if rays.shape[1] < 4:
        raise CalibrationError(
            f"PnP algorithm requires at least 4 points, got {rays.shape[1]}. "
            f"Please provide at least 4 valid 3D-2D point correspondences."
        )

    pts2d = dehomo(rays)  # [xd, yd, 1]

    # OpenCV solvePnP requires float64
    pts3d_f64 = as_float(pts3d, 64)
    pts2d_f64 = as_float(pts2d, 64)
    K_f64 = np.eye(3, dtype=np.float64)
    dist_f64 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    ret, rvec, tvec = cv.solvePnP(
        pts3d_f64.T, pts2d_f64.T, K_f64, dist_f64, flags=cv_flags
    )
    if ret is True:
        return Transform.from_rot_vec_t(rvec, tvec)
    else:
        raise CalibrationError(
            "PnP algorithm failed to find a solution. "
            "Please check the quality of your 3D-2D point correspondences and camera parameters."
        )


def triangulate_points(
    pts1: Union[np.ndarray, List[Tuple[int]]],
    pts2: Union[np.ndarray, List[Tuple[int]]],
    cam1: Camera,
    cam2: Camera,
    w2c1: Union[Pose, Transform],
    w2c2: Union[Pose, Transform],
) -> np.ndarray:
    """
    Triangulate points from corresponding points between two cameras.

    Args:
        pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], [2,N] or [N,2]): Corresponding points.
            Supports both float32 and float64. If dtypes differ, they are promoted to higher precision.
        cam1, cam2 (Camera): Camera Instance.
        w2c1 (Union[Pose, Transform]): World to first camera Transform or Pose.
        w2c2 (Union[Pose, Transform]): World to first camera Transform or Pose.

    Returns:
        (np.ndarray, [3,N]) : Array containing the 3D coordinates of the triangulated points.

    Note:
        Internally converts to float64 for OpenCV compatibility.
    """
    # Convert points to numpy array if they are given as lists
    if isinstance(pts1, list):
        pts1 = np.array(pts1).T  # [2,N]
    if isinstance(pts2, list):
        pts2 = np.array(pts2).T  # [2,N]

    # Promote dtypes if different (auto dtype promotion)
    if isinstance(pts1, np.ndarray) and isinstance(pts2, np.ndarray):
        if pts1.dtype != pts2.dtype:
            target_dtype = promote_types(pts1, pts2)
            pts1 = pts1.astype(target_dtype)
            pts2 = pts2.astype(target_dtype)

    rays1, _ = cam1.convert_to_rays(pts1)
    rays2, _ = cam2.convert_to_rays(pts2)

    pts1_norm = dehomo(rays1)
    pts2_norm = dehomo(rays2)

    P1 = w2c1.mat34()
    P2 = w2c2.mat34()

    # OpenCV triangulatePoints requires float64
    P1_f64 = as_float(P1, 64)
    P2_f64 = as_float(P2, 64)
    pts1_norm_f64 = as_float(pts1_norm, 64)
    pts2_norm_f64 = as_float(pts2_norm, 64)

    points_4d_hom = cv.triangulatePoints(P1_f64, P2_f64, pts1_norm_f64, pts2_norm_f64)

    points_3d = dehomo(points_4d_hom)
    return points_3d


def compute_relative_transform_from_points(
    pts1: Union[np.ndarray, List[Tuple[int]]],
    pts2: Union[np.ndarray, List[Tuple[int]]],
    cam1: Camera,
    cam2: Camera,
    use_ransac: bool = False,
    threshold: float = 1e-2,
    max_iterations: int = 1000,
) -> Transform:
    """
    Computes the relative transform between two sets of points observed by two cameras.

    Args:
        pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], [2,N] or [N,2]): Corresponding points.
            Supports both float32 and float64. If dtypes differ, they are promoted to higher precision.
        cam1, cam2 (Camera): Camera Instance.
        use_ransac (bool, optional): Flag to use RANSAC for fundamental matrix computation. Defaults to False.
        threshold (float, optional): RANSAC reprojection threshold. Defaults to 1e-2.
        max_iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.

    Returns:
        Transform: The computed relative transform between the two camera views.

    Details:
    - Converts input points to normalized image coordinates to support various camera models.
    - Uses only the points with positive z components of the rays for the computation.

    Note:
        Internally converts to float64 for OpenCV compatibility.
    """
    # Convert points to numpy array if they are given as lists
    if isinstance(pts1, list):
        pts1 = np.array(pts1).T  # [2,N]
    if isinstance(pts2, list):
        pts2 = np.array(pts2).T  # [2,N]

    # Promote dtypes if different (auto dtype promotion)
    if isinstance(pts1, np.ndarray) and isinstance(pts2, np.ndarray):
        if pts1.dtype != pts2.dtype:
            target_dtype = promote_types(pts1, pts2)
            pts1 = pts1.astype(target_dtype)
            pts2 = pts2.astype(target_dtype)

    rays1, mask1 = cam1.convert_to_rays(pts1)
    rays2, mask2 = cam2.convert_to_rays(pts2)

    forward_ray_mask = logical_and(
        rays1[2, :] > 0, rays2[2, :] > 0
    )  # for verifiying relative pose
    mask = logical_and(mask1, mask2, forward_ray_mask)

    pts1_norm = dehomo(rays1[:, mask])
    pts2_norm = dehomo(rays2[:, mask])

    # Since pts1_norm,pts2_norm are normalized points, fundamental matrix(F) is same as essential_matrix
    if use_ransac:
        E, _ = compute_fundamental_matrix_using_ransac(
            pts1_norm, pts2_norm, threshold, max_iterations
        )
    else:
        E = compute_fundamental_matrix_from_points(pts1_norm, pts2_norm)

    # OpenCV recoverPose requires float64
    E_f64 = as_float(E, 64)
    pts1_norm_f64 = as_float(pts1_norm.T, 64)
    pts2_norm_f64 = as_float(pts2_norm.T, 64)
    K_f64 = np.eye(3, dtype=np.float64)

    _, R, t, _ = cv.recoverPose(E_f64, pts1_norm_f64, pts2_norm_f64, K_f64)

    return Transform(t, Rotation.from_mat3(R))


def compute_relative_transform(
    image1: np.ndarray,
    image2: np.ndarray,
    cam1: Camera,
    cam2: Camera,
    feature_type: str = "ORB",
    max_matches: int = 50,
    use_ransac: bool = True,
    threshold: float = 1e-3,
    max_iterations: int = 1000,
) -> Transform:
    """
    Computes the relative transform between two images.

    Args:
        image1, image2 (np.ndarray, [H,W] or [H,W,3]): Images
        cam1, cam2 (Camera): Camera Instance.
        feature_type (str, optional): Type of feature extractor to use ('SIFT', 'ORB', etc.). Defaults to 'ORB'.
        max_matches (int, optional): Maximum number of matching points to use. Defaults to 50.
        use_ransac (bool, optional): Flag to use RANSAC for fundamental matrix computation. Defaults to True.
        threshold (float, optional): RANSAC reprojection threshold. Defaults to 1e-3.
        max_iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.

    Returns:
        Transform: The computed relative transform between the two camera views.

    Details:
    - Converts input points to normalized image coordinates to support various camera models.
    - Uses only the points with positive z components of the rays for the computation.
    """

    pts1, pts2 = find_corresponding_points(image1, image2, feature_type, max_matches)

    return compute_relative_transform_from_points(
        pts1, pts2, cam1, cam2, use_ransac, threshold, max_iterations
    )


def find_corresponding_points(
    image1: np.ndarray,
    image2: np.ndarray,
    feature_type: str = "SIFT",
    max_matches: int = 50,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Extracts features from two images and finds matching points between them.

    Args:
        image1 (np.ndarray, [H,W] or [H,W,3]): First input image.
        image2 (np.ndarray, [H,W] or [H,W,3]): Second input image.
        feature_type (str): Type of feature extractor to use ('SIFT', 'ORB', etc.).
        max_matches (int): Maximum number of matching points to return.

    Returns:
        pts1,pts2 (List[Tuple[float], [N,2]): Lists of matching points in the first and second images.

    Raises:
        ValueError: If unsupported feature type is provided.

    Details:
    - Initialize the feature detector and descriptor based on user input.
    - Detect and compute keypoints and descriptors for both images.
    - Match descriptors between images using the appropriate matcher.
    - Sort matches by their distance (quality).
    - Extract the locations of the best matches based on the specified maximum number of matches.
    """
    # Initialize the feature detector and descriptor based on user input
    if feature_type == "SIFT":
        detector = cv.SIFT_create()
    elif feature_type == "ORB":
        detector = cv.ORB_create(nfeatures=max_matches)  # Limit the number of features
    else:
        raise ValueError("Unsupported feature type. Choose 'SIFT' or 'ORB'.")

    # Find keypoints and descriptors with chosen detector
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    # Match descriptors between images
    if feature_type == "SIFT":
        # Use FLANN based matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    elif feature_type == "ORB":
        # Use BFMatcher (Brute Force Matcher) for ORB
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError("UnSupported Feature Type.")

    # Perform matching
    matches = matcher.match(descriptors1, descriptors2)
    # Sort matches by their distance (quality)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = [keypoints1[m.queryIdx].pt for m in matches[:max_matches]]
    pts2 = [keypoints2[m.trainIdx].pt for m in matches[:max_matches]]

    return pts1, pts2
