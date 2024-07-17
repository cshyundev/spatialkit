"""
Module Name: img_tf.py

Description:
This module provides various geometric transformation functions for images. 

Supported Transformations:
    - Translation
    - Rotation
    - Shearing
    - Scaling
    - Similarity transformation (combining rotation, translation, and scaling)
    - Affine transformation (combining translation, rotation, scaling, and shearing)
    - Homography (perspective transformation)

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""
import numpy as np
from typing import *
from ..ops.uops import *
from ..ops.umath import *
import cv2 as cv
import random

def translation(tx: int=0, ty: int=0) -> np.ndarray:
    """
        Translation transformation matrix

        Args:
            tx (int, optional): Translation in x-direction. Defaults to 0.
            ty (int, optional): Translation in y-direction. Defaults to 0.
        
        Return:
            mat (np.ndarray, [3,3]): Translation transformation matrix
    """
    mat33 = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]], np.float32)
    return mat33

def rotation(angle: float, center: tuple=(0, 0)) -> np.ndarray:
    """
        Rotation transformation matrix around a given center point.

        Args:
            angle (float): Rotation angle in degrees.
            center (Tuple, float): The center point (cx, cy) to rotate around.
        
        Return:
            mat (np.ndarray, [3,3]): Rotation transformation matrix.
    """
    cx, cy = center
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    # Translation to origin
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]], np.float32)
    
    # Rotation
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]], np.float32)
    
    # Translation back to the center
    T2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]], np.float32)
    
    # Combined transformation
    mat33 = T2 @ R @ T1
    return mat33

def shear(shx: float = 0, shy: float=0) -> np.ndarray:
    """
        Shear transformation matrix

        Args:
            shx (float, optional): Shear in x-direction. Defaults to 0.
            shy (float, optional): Shear in y-direction. Defaults to 0.
        
        Return:
            mat (np.ndarray, [3,3]): Shear transformation matrix.
    """
    mat33 = np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]], np.float32)
    return mat33

def scaling(sx: float = 1, sy: float=1) -> np.ndarray:
    """
        Scaling transformation matrix

        Args:
            sx (float, optional): Scaling factor in x-direction. Defaults to 1.
            sy (float, optional): Scaling factor in y-direction. Defaults to 1.

        Return:
            mat (np.ndarray, [3,3]): Scaling transformation matrix.
    """
    mat33 = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]], np.float32)
    return mat33

def similarity(angle: float, tx: int=0, ty: int=0, scale: float=1.0) -> np.ndarray:
    """
        Similarity transformation matrix (combining rotation, translation, and scaling)

        Args:
            angle (float): Rotation angle in degrees.
            tx (int, optional): Translation in x-direction. Defaults to 0.
            ty (int, optional): Translation in y-direction. Defaults to 0.
            scale (float, optional): Scaling factor. Defaults to 1.0.
        
        Return:
            mat (np.ndarray, [3,3]): Similarity transformation matrix.
    """
    rad = np.deg2rad(angle)
    s_cos_a = np.cos(rad) * scale
    s_sin_a = np.sin(rad) * scale
    mat33 = np.array([
        [s_cos_a, -s_sin_a, tx],
        [s_sin_a, s_cos_a, ty],
        [0, 0, 1]], np.float32)
    return mat33

def affine(tx: int=0, ty: int=0, angle: float=0, center: tuple=(0, 0), 
                     sx: float=1, sy: float=1, shx: float=0, shy: float=0) -> np.ndarray:
    """
    Create an affine transformation matrix that combines translation, rotation, scaling, and shearing.

    Args:
        tx (int): Translation along the x-axis.
        ty (int): Translation along the y-axis.
        angle (float): Rotation angle in degrees.
        center (tuple): Center of rotation (cx, cy).
        sx (float): Scaling factor along the x-axis.
        sy (float): Scaling factor along the y-axis.
        shx (float): Shear factor along the x-axis.
        shy (float): Shear factor along the y-axis.

    Return:
        mat (np.ndarray, [3,3]): Affine transformation matrix.
    """
    # Calculate radians for rotation
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    cx, cy = center

    # Translation matrix to center
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0, 1]], dtype=np.float32)

    # Rotation matrix
    R = np.array([[cos_a, -sin_a, 0],
                  [sin_a, cos_a, 0],
                  [0, 0, 1]], dtype=np.float32)

    # Scaling matrix
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]], dtype=np.float32)

    # Shear matrix
    Sh = np.array([[1, shx, 0],
                   [shy, 1, 0],
                   [0, 0, 1]], dtype=np.float32)

    # Translation back to the original position
    T2 = np.array([[1, 0, cx+tx],
                   [0, 1, cy+ty],
                   [0, 0, 1]], dtype=np.float32)

    # Combined transformation
    transform = T2 @ S @ Sh @ R @ T1
    return transform

def compute_homography(pts1: Union[np.ndarray,List[Tuple[int,int]]], pts2: Union[np.ndarray,List[Tuple[int,int]]],\
                        use_ransac: bool=False, ransac_threshold: float=5.0, ransac_iterations: int=1000) -> np.ndarray:
    """
        Compute the homography matrix from two sets of points.

        Args:
            pts1 (Union[np.ndarray, List[Tuple[int, int]]], [N,]): First set of points.
            pts2 (Union[np.ndarray, List[Tuple[int, int]]], [N,]): Second set of points.
            use_ransac (bool, optional): Whether to use RANSAC for robust estimation. Defaults to False.
            ransac_threshold (float, optional): RANSAC threshold. Defaults to 5.0.
            ransac_iterations (int, optional): Number of RANSAC iterations. Defaults to 1000.

        Return:
            mat (np.ndarray, [3,3]): The computed homography matrix.
    """
    if isinstance(pts1, List): pts1 = np.array(pts1, dtype=float)
    if isinstance(pts2, List): pts2 = np.array(pts2, dtype=float)
    assert(type(pts1)==type(pts2)), "Two pts1 and pts2 must be same type."
    assert(is_array(pts1) and is_array(pts2)), "pts must be ArrayLike type."
    assert(pts1.shape[0] >= 4), f"To compute homograpy, correspondence pairs must be larger than 4, but got {pts1.shape[0]}"

    def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the points so that the mean is 0 and the average distance is sqrt(2).
        """

        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid
        scale = np.sqrt(2) / np.mean(np.linalg.norm(centered_pts, axis=1))
        transform = np.array([[scale, 0, -scale * centroid[0]],
                              [0, scale, -scale * centroid[1]],
                              [0, 0, 1]])
        normalized_points = np.dot(transform, np.concatenate((pts.T, np.ones((1, pts.shape[0])))))
        
        return normalized_points.T, transform
    
    def _compute_homography_from_points(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        A = []
        for i in range(pts1_norm.shape[0]):
            x1, y1 = pts1[i,0],pts1[i,1]
            x2, y2 = pts2[i,0],pts2[i,1]
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
        A = np.array(A)
        _, _, Vt = svd(A)
        H = Vt[-1].reshape((3, 3))
        return H

    pts1_norm, T1 = _normalize_points(pts1)
    pts2_norm, T2 = _normalize_points(pts2)

    if pts1.shape[0] > 4 and use_ransac:
        best_inliers = 0
        best_homography = None
        for _ in range(ransac_iterations):
            indices = random.sample(range(pts1.shape[0]), 4)
            H_candidate = _compute_homography_from_points(pts1_norm[indices], pts2_norm[indices])
            H_candidate_denorm = dot(inv(T2), dot(H_candidate, T1))
            
            inliers = 0
            for i in range(pts1.shape[0]):
                pt1_homog = np.append(pts1[i], 1)
                estimate_pt2 = np.dot(H_candidate_denorm, pt1_homog)
                estimate_pt2 /= estimate_pt2[2]
                error = np.linalg.norm(estimate_pt2[:2] - pts2[i])
                if error < ransac_threshold:
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_homography = H_candidate_denorm
        
        if best_homography is not None:
            best_homography /= best_homography[2, 2]
            return best_homography

    # Compute Homography using SVD decomposition without RANSAC
    H = _compute_homography_from_points(pts1_norm, pts2_norm)
    H = dot(inv(T2), dot(H, T1))
    H /= H[2, 2]
    
    # convert numpy to tensor
    return H

def apply_transform(image: np.ndarray, transform: np.ndarray, output_size: Tuple[int,int], inverse: bool=True) -> np.ndarray:
    """
        Apply a perspective transformation to the given image.

        Args:
            image (np.ndarray, [H,W] or [H,W,3]): input image array.
            transform (np.ndarray, [3,3]):  transformation matrix applied in image coordinates.
            output_size (Tuple[int, int]): (out_w, out_h) of the output image.
            inverse (bool): Boolean flag to indicate whether to perform inverse warping (default) or forward warping.

        Return:
        output (np.ndarray, [out_h,out_w] or [out_h, out_w, 3]): The transformed image 
    """
    assert transform.shape == (3, 3), "Transformation matrix must be a 3x3 matrix."
    transform = transform if inverse else inv(transform)

    return cv.warpPerspective(image, transform, output_size)
