import numpy as np
from typing import *
from src.hybrid_operations import *
from src.hybrid_math import *
from src.camera import Camera
from scipy.ndimage import map_coordinates
import cv2 as cv
import random


def make_pixel_grid(width:int, height:int) -> np.ndarray:
    u,v = np.meshgrid(range(width), range(height))
    uv = concat([u.reshape((1,-1)), v.reshape((1,-1))], 0)
    return uv

def translation(tx: int = 0, ty: int = 0) -> np.ndarray:
    """
    Translation transformation matrix

    Args:
        tx (int, optional): Translation in x-direction. Defaults to 0.
        ty (int, optional): Translation in y-direction. Defaults to 0.
    Returns:
         Translation transformation matrix
    """
    mat33 = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]], np.float32)
    return mat33

def rotation(angle: float) -> np.ndarray:
    """
    Rotation transformation matrix

    Args:
        angle (float): Rotation angle in degrees.
    Returns:
        Rotation transformation matrix
    """
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    mat33 = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]], np.float32)
    return mat33

def shear(shx: float = 0, shy: float = 0) -> np.ndarray:
    """
    Shear transformation matrix

    Args:
        shx (float, optional): Shear in x-direction. Defaults to 0.
        shy (float, optional): Shear in y-direction. Defaults to 0.
    Returns:
        Shear transformation matrix
    """
    mat33 = np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]], np.float32)
    return mat33

def scaling(sx: float = 1, sy: float = 1) -> np.ndarray:
    """
    Scaling transformation matrix

    Args:
        sx (float, optional): Scaling factor in x-direction. Defaults to 1.
        sy (float, optional): Scaling factor in y-direction. Defaults to 1.
    Returns:
        Scaling transformation matrix
    """
    mat33 = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]], np.float32)
    return mat33

def similarity(angle: float, tx: int = 0, ty: int = 0, scale: float = 1.0) -> np.ndarray:
    """
    Similarity transformation matrix (combining rotation, translation, and scaling)

    Args:
        angle (float): Rotation angle in degrees.
        tx (int, optional): Translation in x-direction. Defaults to 0.
        ty (int, optional): Translation in y-direction. Defaults to 0.
        scale (float, optional): Scaling factor. Defaults to 1.0.
    Returns:
        Similarity transformation matrix
    """
    rad = np.deg2rad(angle)
    s_cos_a = np.cos(rad) * scale
    s_sin_a = np.sin(rad) * scale
    mat33 = np.array([
        [s_cos_a, -s_sin_a, tx],
        [s_sin_a, s_cos_a, ty],
        [0, 0, 1]], np.float32)
    return mat33

def compute_homography(pts1: Union[Array, List[Tuple[int, int]]], pts2: Union[Array, List[Tuple[int, int]]],\
                        use_ransac: bool = False, ransac_threshold: float = 5.0, ransac_iterations: int = 1000) -> Array:
    if isinstance(pts1, List): pts1 = np.array(pts1, dtype=float)
    if isinstance(pts2, List): pts2 = np.array(pts2, dtype=float)
    assert(type(pts1)==type(pts2)), "Two pts1 and pts2 must be same type."
    assert(is_array(pts1) and is_array(pts2)), "pts must be Array type."
    assert(pts1.shape[0] >= 4), f"To compute homograpy, correspondence pairs must be larger than 4, but got {pts1.shape[0]}"

    if is_tensor(pts1):
        assert(pts1.device == pts2.device), "Two tensor must be same on device."

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

    # check whether pts1 and pts2 are tensor     
    device = pts1.device if is_tensor(pts1) else None

    if device is not None:
        pts1 = convert_numpy(pts1)
        pts2 = convert_numpy(pts2)

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
    if device is not None: H = torch.tensor(H, device=device)
    return H

def apply_transform(image: np.ndarray, transform: np.ndarray, output_size: Tuple[int, int], inverse: bool = True) -> np.ndarray:
    """
    Apply a perspective transformation to the given image.

    Parameters:
    image (np.ndarray): input image array.
    transform (np.ndarray): 3x3 transformation matrix applied in image coordinates.
    output_size (Tuple[int, int]): (width, height) of the output image.
    inverse (bool): Boolean flag to indicate whether to perform inverse warping (default) or forward warping.

    Returns:
    np.ndarray: The transformed image 
    """
    assert transform.shape == (3, 3), "Transformation matrix must be a 3x3 matrix."
    transform = transform if inverse else inv(transform)

    return cv.warpPerspective(image, transform, output_size)

def transition_camera_view(image: np.ndarray, src_cam: Camera, dst_cam: Camera, transform: np.ndarray) -> np.ndarray:
    """
    Transition the view from one camera to another with a specified transformation matrix.
    
    Parameters:
    image (np.ndarray): The input image array from the source camera.
    src_cam (Camera): Source camera object with get_rays method.
    dst_cam (Camera): Destination camera object with project_pixel method.
    transform (np.ndarray): A 3x3 transformation matrix applied in normalized coordinates.

    Returns:
    np.ndarray: The output image transformed and projected onto the destination camera's resolution.
    """
    
    out_width,out_height = dst_cam.resolution
    # Prepare the output image array
    if len(image) == 3:
        output_image = np.zeros((out_height, out_width, image.shape[2]), dtype=image.dtype)
    else:
        output_image = np.zeros((out_height, out_width), dtype=image.dtype)

    output_rays = dst_cam.get_rays() # 3 * N

    # Apply inverse transform
    transformed_rays = matmul(inv(transform), output_rays) # 3 * N

    # project ray onto source camera
    input_coords = src_cam.project_rays(transformed_rays,out_subpixel=True) # 2 * N
    
    # Split coordinates
    input_x, input_y = input_coords[0, :], input_coords[1, :]
    
    if image.ndim == 3:
        # For multi-channel images, handle each channel separately
        for c in range(image.shape[2]):
            output_image[..., c] = map_coordinates(image[..., c], [input_y, input_x], order=1, mode='reflect').reshape((out_height, out_width))
    else:
        # For single-channel images
        output_image = map_coordinates(image, [input_y, input_x], order=1, mode='reflect').reshape((out_height, out_width))

    return output_image