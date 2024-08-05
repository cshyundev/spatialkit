"""
Module Name: geom_utils.py

Description:
This module provides various geometric transformation functions for 3D geometry. 
These functions include solving Perspective-n-Point (PnP) problems, computing essential 
matrices, and other operations related to 3D geometric transformations and computations.

Supported Functions:
    - Solve PnP (Perspective-n-Point)
    - Compute essential matrix
    - Decompose essential matrix
    - Compute fundamental matrix
    - Find Corresponding points
    - Transition Camera View (Image Warping)
    
Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.0-alpha

License: MIT LICENSE
"""
import numpy as np
from ..ops.uops import *
from .pose import Pose
from .rotation import Rotation 
from .tf import Transform
from .camera import *
from typing import *
import cv2 as cv
from scipy.ndimage import map_coordinates
from ..common.logger import *

def compute_essential_matrix_from_pose(rel_p: Union[Pose,Transform]) -> np.ndarray:
    """
        Compute the essential matrix from the relative camera pose.

        Args:
            rel_p (Pose or Transform): Relative pose object containing rotation and translation between cameras.

        Return:
            E (np.ndarray, [3,3]): The computed essential matrix (3x3).
    """
    skew_t = rel_p.skew_t()
    r_mat = rel_p.rot_mat()
    return convert_numpy(skew_t @ r_mat)

def compute_essential_matrix_from_fundamental(K1: np.ndarray, K2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
        Compute the essential matrix from the fundamental matrix and intrinsic camera matrices.

        Args:
            K1 (np.ndarray, [3,3]): Intrinsic camera matrix for the first camera.
            K2 (np.ndarray, [3,3]): Intrinsic camera matrix for the second camera.
            F  (np.ndarray, [3,3]): Fundamental matrix.

        Return:
            E (np.ndarray, [3,3]): The computed essential matrix (3x3).
    """
    return K2.T @ F @ K1

def compute_fundamental_matrix_from_points(pts1: Union[np.ndarray, List[Tuple[int]]], 
                                           pts2: Union[np.ndarray, List[Tuple[int]]]) -> np.ndarray:
    """
    Compute the fundamental matrix given point correspondences.

    Args:
        pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], [2,N] or [N,2]): Corresponding points in each image.

    Return:
        F (np.ndarray, [3,3]): The computed fundamental matrix.

    Raises:
        ValueError: If insufficient or incorrect data is provided.
    """
    # Ensure pts1 and pts2 are numpy arrays
    if isinstance(pts1, list):
        pts1 = np.array(pts1).T
    if isinstance(pts2, list):
        pts2 = np.array(pts2).T

    assert pts1.shape == pts2.shape, "Point arrays must have the same shape."
    assert pts1.shape[0] == 2, "Point arrays must have two coordinates."
    assert pts1.shape[1] >= 8, f"The number of corresponding points must be larger than 8, but got {pts1.shape[0]}"

    # Construct matrix A for linear equation system
    x1 = pts1[0,:].reshape(-1,1)
    y1 = pts1[1,:].reshape(-1,1)
    x2 = pts2[0,:].reshape(-1,1)
    y2 = pts2[1,:].reshape(-1,1)
    
    A = np.concatenate([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2,ones_like(x1)],1)
    # Solve using SVD
    _, _, Vt = svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce the rank-2 constraint
    U, S, Vt = svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    return F

def compute_fundamental_matrix_from_essential(K1: np.ndarray, K2: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
        Compute the fundamental matrix from the essential matrix and intrinsic camera matrices.

        Args:
            K1, K2 (np.ndarray, [3,3]): Intrinsic matrices of the two cameras.
            E (np.ndarray, [3,3]): Essential matrix.

        Return:
            F (np.ndarray, [3,3]): The computed fundamental matrix.

        Raises:
            ValueError: If insufficient or incorrect data is provided.
    """
    assert K1.shape == (3, 3) and K2.shape == (3, 3) and E.shape == (3, 3), "Intrinsic and Essential matrices must be 3x3."
    # Compute fundamental matrix from essential matrix
    F = inv(K2).T @ E @ inv(K1)
    return F

def compute_fundamental_matrix_using_ransac(pts1: Optional[Union[np.ndarray, List[Tuple[int]]]],
                                            pts2: Optional[Union[np.ndarray, List[Tuple[int]]]],
                                            threshold:float=1e-2,
                                            max_iterations:int=1000):
    """
        Compute the fundamental matrix given point correspondences using the RANSAC algorithm.

        Args:
            pts1,pts2 (Union[np.ndarray, List[Tuple[int, int]]], [N, 2]): Corresponding points in the images respectively.
            threshold (float, optional): Distance threshold to determine inliers. Defaults to 1e-3.
            max_iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.

        Returns:
            best_F (np.ndarray, [3, 3]): The computed fundamental matrix.
            best_inliers (int): Number of inliers for the best fundamental matrix.
    """
    # Ensure pts1 and pts2 are numpy arrays
    if isinstance(pts1, list): pts1 = np.array(pts1).T
    if isinstance(pts2, list): pts2 = np.array(pts2).T
    assert pts1.shape == pts2.shape, "Point arrays must have the same shape."
    assert pts1.shape[0] == 2, "Point arrays must have two coordinates."
    assert pts1.shape[1] >= 8, f"The number of corresponding points must be larger than 8, but got {pts1.shape[0]}"

    best_F = None
    best_inliers = 0

    num_pts = pts1.shape[1]
    for _ in range(max_iterations):
        # Randomly select 8 points for the minimal sample set
        indices = np.random.choice(num_pts, 8, replace=False)
        sample_pts1 = pts1[:,indices]
        sample_pts2 = pts2[:,indices]

        # Compute the fundamental matrix
        F = compute_fundamental_matrix_from_points(sample_pts1, sample_pts2)

        # Calculate the number of inliers
        inliers = 0
        for i in range(num_pts):
            pt1 = np.append(pts1[:,i], 0)
            pt2 = np.append(pts2[:,i], 0)
            error = abs(pt2.T @ F @ pt1)
            if error < threshold:
                inliers += 1

        # Update the best model if current model has more inliers
        if inliers > best_inliers:
            best_F = F
            best_inliers = inliers

    return best_F, best_inliers

def decompose_essential_matrix(E: np.ndarray) -> Tuple[Transform]:
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
    if determinant(U) < 0: U = -U
    if determinant(Vt) < 0: Vt = -Vt

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:,2]

    tf1 = Transform(t, Rotation.from_mat3(R1))
    tf2 = Transform(-t, Rotation.from_mat3(R1))
    tf3 = Transform(t, Rotation.from_mat3(R2))
    tf4 = Transform(-t, Rotation.from_mat3(R2))

    return tf1,tf2,tf3,tf4

def solve_pnp(pts2d: np.ndarray, pts3d: np.ndarray, cam:Camera, cv_flags:Any=None) -> Transform:
    """
       Computes the camera pose using the Perspective-n-Point (PnP) problem solution.
       Args:
            pts2d (np.ndarray, [N,2]): 2D image points.
            pts3d (np.ndarray, [N,3]): Corresponding 3D scene points.
            cam (Camera): Camera instance which can be one of available models.
            cv_flags (Any, optional): Additional options for solving PnP.

        Return:
            Transform: A Transform Instance from object coordinates to camera coordinates.

        Raises:
            AssertionError: If the camera type is not supported.
    """
    cam_type = cam.cam_type
    unavailable_cam_types = [ ]
    assert(cam_type  not in unavailable_cam_types), f"Unavailable Camera Type: {cam_type.name}"

    rays, mask = cam.convert_to_rays(pts2d) # 2 * N
    rays = rays[:,mask] # delete unavailable rays
    
    if rays.shape[1] < 4:
        LOG_ERROR(f"Not enough points to solvePnP. available points must be 4 at least, but got {rays.shape[1]}.")
        return None

    pts2d =  dehomo(rays) # [xd, yd, 1]

    ret, rvec, tvec = cv.solvePnP(pts3d.T,pts2d.T,np.eye(3),np.array([0., 0., 0., 0., 0.]))
    if ret:
        return Transform.from_rot_vec_t(rvec,tvec)
    else:
        LOG_ERROR("Failed to SolvePnP Algorithm.")
        return None

def transition_camera_view(src_image: np.ndarray, src_cam: Camera, dst_cam: Camera, img_tf:Optional[np.ndarray]=None) -> np.ndarray:
    """
        Transition the view from one camera to another with a specified transformation.

        Args:
            image (np.ndarray, [H,W] or [H,W,3]): The input image array from the source camera.
            src_cam (Camera): Source camera instance, camera size = [W,H]
            dst_cam (Camera): Destination camera instance, camera size = [out_W,out_H]
            transform (Transform): Transform instance applied in normalized coordinates.

        Return:
            output_image (np.ndarray, [out_H,out_W] or [out_H,out_W,3]): The output image transformed and projected onto the destination camera's resolution.
    """
    out_height,out_width  = dst_cam.hw
    # Prepare the output image array
    if src_image.ndim == 3:
        output_image = np.zeros((out_height, out_width, src_image.shape[2]), dtype=src_image.dtype)
    else:
        output_image = np.zeros((out_height, out_width), dtype=src_image.dtype)

    output_rays, dst_valid_mask = dst_cam.convert_to_rays()  # 3 * N
    # Apply inverse transform
    if img_tf is not None:
        inverse_img_tf = inv(img_tf)
        output_rays = inverse_img_tf @ output_rays  # 3 * N

    # Project ray onto source camera
    input_coords, src_valid_mask  = src_cam.convert_to_pixels(output_rays, out_subpixel=True)  # 2 * N
    # input_coords = input_coords[:,src_valid_mask]
    # Split coordinates
    input_x, input_y = input_coords[0, :], input_coords[1, :]
    if src_image.ndim == 3:
        # For multi-channel images, handle each channel separately
        for c in range(src_image.shape[2]):
            output_image[..., c] = map_coordinates(src_image[..., c], [input_y, input_x], order=1, mode='constant').reshape((out_height, out_width))
    else:
        # For single-channel images
        output_image = map_coordinates(src_image, [input_y, input_x], order=1, mode='constant').reshape((out_height, out_width))
    mask = logical_and(src_valid_mask,dst_valid_mask).reshape((out_height,out_width))
    output_image[~mask,0] = 0    

    return output_image

def find_corresponding_points(image1: np.ndarray, image2: np.ndarray, 
                              feature_type: str='SIFT', max_matches: int=50) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
        Extracts features from two images and finds matching points between them.

        Args:
            image1 (np.ndarray, [H,W] or [H,W,3]): First input image.
            image2 (np.ndarray, [H,W] or [H,W,3]): Second input image.
            feature_type (str): Type of feature extractor to use ('SIFT', 'ORB', etc.).
            max_matches (int): Maximum number of matching points to return.

        Returns:
            pts1,pts2 (List[Tuple[float], [N,2]): Lists of matching points in the first and second images.
        
        Details:
        - Initialize the feature detector and descriptor based on user input.
        - Detect and compute keypoints and descriptors for both images.
        - Match descriptors between images using the appropriate matcher.
        - Sort matches by their distance (quality).
        - Extract the locations of the best matches based on the specified maximum number of matches.
    """
    # Initialize the feature detector and descriptor based on user input
    if feature_type == 'SIFT':
        detector = cv.SIFT_create()
    elif feature_type == 'ORB':
        detector = cv.ORB_create(nfeatures=max_matches)  # Limit the number of features
    else:
        raise ValueError("Unsupported feature type. Choose 'SIFT' or 'ORB'.")

    # Find keypoints and descriptors with chosen detector
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    # Match descriptors between images
    if feature_type == 'SIFT':
        # Use FLANN based matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    elif feature_type == 'ORB':
        # Use BFMatcher (Brute Force Matcher) for ORB
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Perform matching
    matches = matcher.match(descriptors1, descriptors2)
    # Sort matches by their distance (quality)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = [keypoints1[m.queryIdx].pt for m in matches[:max_matches]]
    pts2 = [keypoints2[m.trainIdx].pt for m in matches[:max_matches]]

    return pts1, pts2

def triangulate_points(pts1: Union[np.ndarray, List[Tuple[int]]],
                       pts2: Union[np.ndarray, List[Tuple[int]]],
                       cam1:Camera, cam2:Camera,
                       w2c1:Union[Pose,Transform],
                       w2c2:Union[Pose,Transform]) -> np.ndarray:
    """
        Triangulate points from corresponding points between two cameras.

        Args:
            pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], [2,N] or [N,2]): Corresponding points
            cam1, cam2 (Camera): Camera Instance.
            w2c1 (Union[Pose, Transform]): World to first camera Transform or Pose.
            w2c2 (Union[Pose, Transform]): World to first camera Transform or Pose.
        
        Return:
            (np.ndarray, [3,N]) : Array containing the 3D coordinates of the triangulated points.
    """
    # Convert points to numpy array if they are given as lists
    if isinstance(pts1, list):
        pts1 = np.array(pts1).T # [2,N]
    if isinstance(pts2, list):
        pts2 = np.array(pts2).T # [2,N]
    
    rays1, _ = cam1.convert_to_rays(pts1)
    rays2, _ = cam2.convert_to_rays(pts2)

    pts1_norm = dehomo(rays1)
    pts2_norm = dehomo(rays2) 

    P1 = w2c1.mat34()
    P2 = w2c2.mat34()

    points_4d_hom = cv.triangulatePoints(P1,P2,pts1_norm,pts2_norm)

    points_3d = dehomo(points_4d_hom)
    return points_3d

def compute_relative_transform_from_points(pts1: Union[np.ndarray, List[Tuple[int]]],
                               pts2: Union[np.ndarray, List[Tuple[int]]],
                               cam1: Camera,
                               cam2: Camera,
                               use_ransac: bool=False,
                               threshold:float=1e-2,
                               max_iterations:int=1000) -> Transform:
    """
        Computes the relative transform between two sets of points observed by two cameras.

        Args:
            pts1, pts2 (Union[np.ndarray, List[Tuple[int]]], [2,N] or [N,2]): Corresponding points
            cam1, cam2 (Camera): Camera Instance.
            use_ransac (bool, optional): Flag to use RANSAC for fundamental matrix computation. Defaults to False.
            threshold (float, optional): RANSAC reprojection threshold. Defaults to 1e-3.
            max_iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.

        Return:
            Transform: The computed relative transform between the two camera views.

        Details:
        - Converts input points to normalized image coordinates to support various camera models.
        - Uses only the points with positive z components of the rays for the computation.
    """
    
    # Convert points to numpy array if they are given as lists
    if isinstance(pts1, list):
        pts1 = np.array(pts1).T # [2,N]
    if isinstance(pts2, list):
        pts2 = np.array(pts2).T # [2,N]

    rays1, mask1 = cam1.convert_to_rays(pts1)
    rays2, mask2 = cam2.convert_to_rays(pts2)

    forward_ray_mask = logical_and(rays1[2,:] > 0, rays2[2,:] > 0) # for verifiying relative pose
    mask = logical_and(mask1,mask2,forward_ray_mask)

    pts1_norm = dehomo(rays1[:,mask])
    pts2_norm = dehomo(rays2[:,mask])

    # Since pts1_norm,pts2_norm are normalized points, fundamental matrix(F) is same as essential_matrix  
    if use_ransac:
        E,_ = compute_fundamental_matrix_using_ransac(pts1_norm,pts2_norm,threshold,max_iterations)
    else:
        E = compute_fundamental_matrix_from_points(pts1_norm, pts2_norm)

    _,R,t,_ = cv.recoverPose(E,pts1_norm.T,pts2_norm.T,np.eye(3))

    return Transform(t,Rotation.from_mat3(R))

def compute_relative_transform(image1: np.ndarray,
                               image2: np.ndarray,
                               cam1: Camera,
                               cam2: Camera,
                               feature_type: str='ORB',
                               max_matches: int = 50,
                               use_ransac: bool=True,
                               threshold:float=1e-3,
                               max_iterations:int=1000) -> Transform:
    """
        Computes the relative transform between two images.

        Args:
            image1, image2 (np.ndarray, [H,W] or [H,W,3]): Images
            cam1, cam2 (Camera): Camera Instance.
            feature_type (str, optional): Type of feature extractor to use ('SIFT', 'ORB', etc.). Defaults to 'SIFT'.
            max_matches (int, optional): Maximum number of matching points to use. Defaults to 50.
            use_ransac (bool, optional): Flag to use RANSAC for fundamental matrix computation. Defaults to False.
            threshold (float, optional): RANSAC reprojection threshold. Defaults to 1e-3.
            max_iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.

        Return:
            Transform: The computed relative transform between the two camera views.

        Details:
        - Converts input points to normalized image coordinates to support various camera models.
        - Uses only the points with positive z components of the rays for the computation.
    """
    
    pts1,pts2 = find_corresponding_points(image1,image2,feature_type,max_matches)
    
    return compute_relative_transform_from_points(pts1,pts2,cam1,cam2,use_ransac,threshold,max_iterations)

def convert_point_cloud_to_depth(pcd: np.ndarray, cam:Camera, map_type:str='MPI'):
    """
        Convert 3D point to depth map of given camera.

        Args:
            pcd (np.ndarray, [N,3] or [3,N]): 3D Point Cloud
            cam: (Camera): Camera Instance with [H,W] resolution.
            map_type:(str): Depth map represntation type (see Details).

        Return:
            depth_map (np.ndarray, [H,W]): depth map converted from point cloud
        
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

    if pcd.shape[0] != 3: # pcd's shape = [N,3]
        pcd = swapaxes(pcd,0,1) # convert pcd's shape as [3,N]

    if map_type.lower() == "mpi": # Depth = Z
        depth = pcd[2,:] 
    elif map_type.lower() == "msi": # Depth = sqrt(X^2 + Y^2 + Z^2)
        depth = norm(pcd,dim=0)
    else: # Depth = sqrt(X^2 + Z^2)
        depth = sqrt(pcd[0,:]**2 + pcd[2,:]**2)
    
    uv, mask = cam.convert_to_pixels(pcd) # [2,N], [N,]
    
    # remain valid pixel coords and these depth
    uv = uv[:,mask]
    depth = depth[mask]

    
    depth_map = np.full((cam.width*cam.height), np.inf)
    indices = uv[1,:] * cam.width + uv[0,:]
    np.minimum.at(depth_map, indices, depth)
    depth_map[depth_map == np.inf] = 0.
    depth_map = depth_map.reshape((cam.height,cam.width))
    return depth_map

def convert_depth_to_point_cloud(depth: np.ndarray, cam: Camera, image: Optional[np.ndarray]=None, map_type: str='MPI',pose:Optional[Union[Pose,Transform]]=None):
    """
        Convert depth map to point cloud.

        Args:
            depth_map (np.ndarray, [H,W]): depth map.
            cam (Camera): Camera Instance with [H,W] resolution.
            image (np.ndarray, [H,W,3]): color image. 
            map_type (str): Depth map representation type (see Details).
            pose (Pose or Transform): Transform instance.

        Returns:
            pcd (np.ndarray, [N,3]): 3D Point Cloud
            colors (np.ndarray, [N,3]): Point Cloud's color if image was given
    
        Details:
        - Available map_type: MPI, MSI, MCI
        - Multi-Plane Image (MPI): Depth = Z
        - Multi-Spherical Image (MSI): Depth = sqrt(X^2 + Y^2 + Z^2)
        - Multi-Cylinder Image (MCI): Depth = sqrt(X^2 + Z^2)
        - Return only valid point cloud (i.e. N <= H*W).
    """
    if depth.shape != cam.hw:
        LOG_CRITICAL(f"Depth map's resolution must be same as camera image size, but got depth's shape={depth.shape}.")
        raise ValueError(f"Depth map's resolution must be same as camera image size, but got depth's shape={depth.shape}.")
    if image is not None and image.shape[0:2] != cam.hw:
        LOG_CRITICAL(f"Image's resolution must be same as camera image size, but got image's shape={image.shape}.")
        raise ValueError(f"Image's resolution must be same as camera image size, but got image's shape={image.shape}.")

    if cam.cam_type in [CamType.PERSPECTIVE, CamType.OPENCVFISHEYE, CamType.THINPRISM] and map_type != 'MPI':
        LOG_WARN(f"Camera type {cam.cam_type} typically expects MPI depth map, but got {map_type}.")

    rays, mask = cam.convert_to_rays()
    depth = depth.reshape(-1,)

    if map_type == 'MPI':
        Z = rays[2:3,:]
        mask = logical_and((Z != 0.).reshape(-1,), mask)
        Z[Z == 0.] = EPSILON
        rays = rays / Z # set Z = 1
        pts3d = rays * depth
    elif map_type == 'MSI':
        pts3d = rays * depth
    elif map_type == 'MCI':
        r = sqrt(rays[0, :]**2 + rays[2, :]**2).reshape(-1, 1)
        mask = logical_and(mask, (r != 0.).reshape(-1,) )
        r[r == 0.] = EPSILON
        pts3d = rays * depth / r
    else:
        LOG_CRITICAL(f"Unsupported map_type {map_type}.")
        raise ValueError(f"Unsupported map_type {map_type}.")

    if pose is not None:
        if isinstance(pose,Pose): pose = Transform.from_pose(pose)
        pts3d = pose * pts3d

    pts3d = swapaxes(pts3d,0,1)

    valid_depth_mask = depth > 0.

    mask = logical_and(mask,valid_depth_mask)
    pts3d = pts3d[mask,:]
    if image is not None:
        colors = image.reshape(-1,3)[mask,:]
        return pts3d, colors
    return pts3d

def compute_points_for_epipolar_curve(pt_cam1: np.ndarray, cam1: Camera, cam2: Camera, rel_tf: Transform, depth_rng: Tuple[float, float], max_pts: int):
    """
        Compute points for the epipolar curve from cam1 to cam2.

        Args:
            pt_cam1 (np.ndarray, [2,1]): The 2D point in cam1 image space.
            cam1 (Camera): The first camera object.
            cam2 (Camera): The second camera object.
            rel_tf (Transform): The relative transform between cam1 and cam2.
            depth_rng (Tuple[float, float]): The range of depths to consider.
            max_pts (int): The maximum number of points to compute.

        Return:
            points (np.ndarray, [2,N]): List of valid 2D points in cam2 image space.
    """
    assert isinstance(depth_rng, tuple) and len(depth_rng) == 2, "depth_rng must be a tuple with two elements."
    assert pt_cam1.shape == (2, 1), "pt_cam1 must be a 2x1 numpy array."
    assert max_pts > 0, "max_pts must be greater than 0."

    ray, mask = cam1.convert_to_rays(pt_cam1)
    if mask.sum() == 0:
        LOG_ERROR("No valid ray found from the given point in cam1.")
        return None

    def compute_pixel_in_cam2(ray: np.ndarray, cam2: Camera, rel_tf: Transform, depth: np.ndarray):
        pt3d = ray * depth
        pt3d = rel_tf * pt3d
        return cam2.convert_to_pixels(pt3d, out_subpixel=False)
        
    def unique_and_sort_pts2d(pts2d_cam2:np.ndarray,distance:np.ndarray):
        pts2d_cam2 = np.unique(pts2d_cam2.T,axis=0).T
        indices = np.lexsort((pts2d_cam2[0],pts2d_cam2[1]))[::-1]
        distance = distance[indices]
        pts2d_cam2 = pts2d_cam2[:,indices]
        return pts2d_cam2,distance

    distance = np.linspace(depth_rng[0],depth_rng[1],max_pts+1)
    pts2d_cam2, mask = compute_pixel_in_cam2(ray,cam2,rel_tf,distance)

    pts2d_cam2 = pts2d_cam2[:,mask]
    distance = distance[mask]
    
    pts2d_cam2,distance = unique_and_sort_pts2d(pts2d_cam2,distance)


    return pts2d_cam2
