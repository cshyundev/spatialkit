"""
Module Name: pointcloud.py

Description:
This module provides functions for converting between 3D point clouds and depth maps,
normal estimation from depth maps, and point cloud processing utilities.

Supported Functions:
    - Convert 3D point cloud to depth map
    - Convert depth map to 3D point cloud
    - Estimate surface normals from depth maps (SVD-based and gradient-based)
    - Estimate surface normals from point clouds
    - Down-sample point clouds using voxel grid

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.1

License: MIT LICENSE
"""

from typing import Union, Optional, Tuple
import numpy as np
from ..ops.uops import *
from ..ops.umath import norm, sqrt, min as umin, max as umax, floor as ufloor, normalize,mean,svd
from ..common.constant import EPSILON
from ..common.logger import LOG_CRITICAL, LOG_WARN
from ..common.exceptions import (
    InvalidDimensionError,
    InvalidArgumentError,
    InvalidShapeError
)
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
        raise InvalidArgumentError(f"Unsupported Depth Map Type, {map_type}.")

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


def _unproject_depth_to_points(
    depth: np.ndarray,
    cam: Camera,
    map_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth map to 3D points (shared logic).

    Args:
        depth (np.ndarray, [H,W]): Depth map.
        cam (Camera): Camera instance.
        map_type (str): Depth map type ("MPI", "MSI", or "MCI").

    Returns:
        pts3d (np.ndarray, [3,H*W]): 3D points.
        mask (np.ndarray, [H*W]): Valid pixel mask.
    """
    rays, mask = cam.convert_to_rays()
    # Convert rays dtype to match depth dtype (important for numpy 2.0+)
    rays = astype_like(rays, depth)
    depth_flat = depth.reshape(-1)

    if map_type.lower() == "mpi":
        Z = rays[2:3, :]
        mask = logical_and(
            (Z != 0.0).reshape(-1),
            mask,
        )
        Z[Z == 0.0] = EPSILON
        rays = rays / Z  # set Z = 1
        pts3d = rays * depth_flat
    elif map_type.lower() == "msi":
        pts3d = rays * depth_flat
    elif map_type.lower() == "mci":
        r = sqrt(rays[0, :] ** 2 + rays[2, :] ** 2).reshape(1, -1)
        mask = logical_and(
            mask,
            (r != 0.0).reshape(-1),
        )
        r[r == 0.0] = EPSILON
        pts3d = rays * depth_flat / r
    else:
        LOG_CRITICAL(f"Unsupported map_type {map_type}.")
        raise InvalidArgumentError(f"Unsupported map_type {map_type}.")

    valid_depth_mask = depth_flat > 0.0
    mask = logical_and(mask, valid_depth_mask)

    return pts3d, mask


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
        InvalidDimensionError: If depth/image resolution doesn't match camera or unsupported map_type.

    Details:
    - Available map_type: MPI, MSI, MCI
    - Multi-Plane Image (MPI): Depth = Z
    - Multi-Spherical Image (MSI): Depth = sqrt(X^2 + Y^2 + Z^2)
    - Multi-Cylinder Image (MCI): Depth = sqrt(X^2 + Z^2)
    - Return only valid point cloud (i.e. N <= H*W).
    """
    if depth.shape != cam.hw:
        raise InvalidDimensionError(
            f"Depth map's resolution must be same as camera image size, but got depth's shape={depth.shape}."
        )
    if image is not None and image.shape[0:2] != cam.hw:
        raise InvalidDimensionError(
            f"Image's resolution must be same as camera image size, but got image's shape={image.shape}."
        )
    
    if map_type not in ["MPI", "MSI", "MCI"]:
        raise InvalidArgumentError(
            f"Unsupported map_type: {map_type}. Choose 'MPI', 'MSI', or 'MCI'."
        )

    if (
        cam.cam_type in [CamType.PERSPECTIVE, CamType.OPENCVFISHEYE, CamType.THINPRISM]
        and map_type != "MPI"
    ):
        LOG_WARN(
            f"Camera type {cam.cam_type} typically expects MPI depth map, but got {map_type}. "
            f"Results may be less accurate."
        )

    pts3d, mask = _unproject_depth_to_points(depth, cam, map_type)

    if pose is not None:
        if isinstance(pose, Pose):
            pose = Transform.from_pose(pose)
        pts3d = pose * pts3d

    pts3d = swapaxes(pts3d, 0, 1)
    pts3d = pts3d[mask, :]

    if image is not None:
        colors = image.reshape(-1, 3)[mask, :]
        return pts3d, colors
    return pts3d


def down_sample_point_cloud(
    pcd: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """
    Downsample point cloud using voxel grid filtering with centroid averaging.

    This function uses a unified approach based on unique() for efficient
    vectorized computation. Points within the same voxel are aggregated and their
    centroid is computed as the representative point.

    Args:
        pcd (np.ndarray, [N,3]): Input point cloud.
        voxel_size (float): Size of each voxel grid.

    Returns:
        down_pcd (np.ndarray, [M,3]): Downsampled point cloud where M <= N.

    Raises:
        TypeError: If pcd is not a NumPy array.
        InvalidArgumentError: If voxel_size is non-positive.
        InvalidDimensionError: If pcd is not 2D array or doesn't have shape [N, 3].

    Example:
        >>> import numpy as np
        >>> pcd = np.random.rand(1000, 3)
        >>> downsampled = down_sample_point_cloud(pcd, voxel_size=0.1)
        >>> downsampled.shape[0] <= pcd.shape[0]
        True
    """
    if not isinstance(pcd, np.ndarray):
        raise TypeError(
            f"pcd must be numpy.ndarray, got {type(pcd).__name__}. "
            f"Convert PyTorch tensors using tensor.detach().cpu().numpy()"
        )

    if voxel_size <= 0.0:
        raise InvalidArgumentError(f"voxel_size must be positive, but got {voxel_size}.")

    if pcd.ndim != 2:
        raise InvalidDimensionError(
            f"Point cloud must be 2D array with shape [N, 3], but got {pcd.ndim}D array."
        )

    if pcd.shape[1] != 3:
        raise InvalidDimensionError(
            f"Point cloud must have 3 coordinates (x, y, z), but got shape {pcd.shape}."
        )

    # Handle empty point cloud
    if pcd.shape[0] == 0:
        return deep_copy(pcd)

    # Compute voxel indices
    min_bound = umin(pcd, dim=0)  # [3,]
    voxel_indices = ufloor((pcd - min_bound) / voxel_size)  # [N, 3]
    voxel_indices = as_int(voxel_indices, n=64)  # Convert to int64

    # Convert 3D voxel indices to 1D hash
    # Use grid_size-based mapping to guarantee no hash collisions
    grid_min = umin(voxel_indices, dim=0)  # [3,]
    grid_max = umax(voxel_indices, dim=0)  # [3,]
    grid_size = grid_max - grid_min + 1    # [3,] - size in each dimension

    # Offset indices to start from 0 (handles negative indices)
    voxel_indices_offset = voxel_indices - grid_min  # [N, 3]

    # Compute 1D hash (collision-free: mathematically guaranteed unique mapping)
    voxel_hash = (voxel_indices_offset[:, 0] * grid_size[1] * grid_size[2] +
                 voxel_indices_offset[:, 1] * grid_size[2] +
                 voxel_indices_offset[:, 2])  # [N,]

    # Find unique voxels
    unique_hash, inverse_indices, counts = unique(
        voxel_hash, return_inverse=True, return_counts=True
    )
    num_voxels = len(unique_hash)

    # Compute centroid for each voxel (vectorized)
    downsampled = zeros((num_voxels, 3), dtype=pcd.dtype, like=pcd)  # [M, 3]

    # Accumulate points into their corresponding voxels (unified!)
    scatter_add(downsampled, inverse_indices, pcd, dim=0)

    # Compute average: reshape counts to [M, 1] for broadcasting
    counts = as_float(counts)  # Convert to float for division
    counts_expanded = expand_dim(counts, dim=1)  # [M, 1]
    downsampled /= counts_expanded

    return downsampled


def compute_point_cloud_normals(
    points: np.ndarray,
    search_param: str = "knn",
    k: int = 30,
    radius: Optional[float] = None,
    max_nn: Optional[int] = None,
    fast_normal_computation: bool = True,
    orient_method: Optional[str] = None,
    orient_direction: Optional[np.ndarray] = None,
    orient_k: int = 10,
) -> np.ndarray:
    """
    Estimate point cloud normals using Open3D's optimized normal estimation.

    This function provides a convenient interface to Open3D's normal estimation
    with support for multiple search methods and orientation strategies.

    Args:
        points (np.ndarray, [N,3]): Point cloud.
        search_param (str): Search method for neighbors. Options:
            - "knn": K-nearest neighbors (default)
            - "radius": Radius search
            - "hybrid": Hybrid (radius + max neighbors)
        k (int): Number of nearest neighbors (for KNN). Default is 30.
        radius (Optional[float]): Search radius (for radius/hybrid). Default is None.
        max_nn (Optional[int]): Maximum neighbors (for hybrid). Default is None.
        fast_normal_computation (bool): If True, uses faster non-iterative method.
            Default is True. Set to False for better numerical stability.
        orient_method (Optional[str]): Normal orientation method. Options:
            - None: No orientation (raw normals with ±180° ambiguity)
            - "camera": Orient towards camera location (requires orient_direction)
            - "direction": Align with given direction (requires orient_direction)
            - "tangent": Consistent tangent plane (Hoppe et al. 1992, most robust)
        orient_direction (Optional[np.ndarray]): Direction for camera/direction methods.
            - For "camera": 3D camera location [x, y, z]
            - For "direction": 3D direction vector [x, y, z]
        orient_k (int): Number of neighbors for tangent plane method. Default is 10.

    Returns:
        normals (np.ndarray, [N,3]): Estimated normals for each point.

    Raises:
        ImportError: If Open3D is not installed.
        TypeError: If points is not a NumPy array.
        InvalidDimensionError: If points is not 2D array.
        InvalidShapeError: If points doesn't have shape [N, 3].
        InvalidArgumentError: If search parameters are invalid.
        ValueError: If orient_method is unknown or missing required parameters.

    Example:
        >>> import numpy as np
        >>> points = np.random.rand(1000, 3)
        >>>
        >>> # Basic usage (KNN with k=30)
        >>> normals = compute_point_cloud_normals(points)
        >>>
        >>> # With radius search
        >>> normals = compute_point_cloud_normals(points, search_param="radius", radius=0.1)
        >>>
        >>> # With hybrid search
        >>> normals = compute_point_cloud_normals(
        ...     points, search_param="hybrid", radius=0.1, max_nn=30
        ... )
        >>>
        >>> # With orientation towards camera
        >>> normals = compute_point_cloud_normals(
        ...     points, orient_method="camera", orient_direction=np.array([0., 0., 0.])
        ... )
        >>>
        >>> # With consistent tangent plane orientation
        >>> normals = compute_point_cloud_normals(points, orient_method="tangent", orient_k=15)

    Note:
        - Uses Open3D's C++ optimized implementation (3-6x faster than pure NumPy)
        - Conversion overhead is minimal (~1% of total time)
        - Results are identical to Open3D's native implementation
        - For large point clouds (>100K), consider using "hybrid" search for robustness
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Open3D is required for normal estimation. "
            "Install it with: pip install open3d"
        )

    # Input validation
    if not isinstance(points, np.ndarray):
        raise TypeError(
            f"points must be numpy.ndarray, got {type(points).__name__}. "
            f"Convert PyTorch tensors using tensor.detach().cpu().numpy()"
        )

    if points.ndim != 2:
        raise InvalidDimensionError(
            f"Point cloud must be 2D array with shape [N, 3], but got {points.ndim}D array."
        )

    if points.shape[1] != 3:
        raise InvalidShapeError(
            f"Point cloud must have 3 coordinates (x, y, z), but got shape {points.shape}."
        )

    n_points = points.shape[0]

    # Handle empty point cloud
    if n_points == 0:
        return np.zeros((0, 3), dtype=points.dtype)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Configure search parameter
    if search_param == "knn":
        if k <= 0:
            raise InvalidArgumentError(f"k must be positive, got {k}.")
        # Auto-adjust k if larger than n_points
        if k > n_points:
            k = n_points
        search = o3d.geometry.KDTreeSearchParamKNN(knn=k)

    elif search_param == "radius":
        if radius is None or radius <= 0:
            raise InvalidArgumentError(
                f"radius must be positive for 'radius' search, got {radius}."
            )
        search = o3d.geometry.KDTreeSearchParamRadius(radius=radius)

    elif search_param == "hybrid":
        if radius is None or radius <= 0:
            raise InvalidArgumentError(
                f"radius must be positive for 'hybrid' search, got {radius}."
            )
        if max_nn is None or max_nn <= 0:
            raise InvalidArgumentError(
                f"max_nn must be positive for 'hybrid' search, got {max_nn}."
            )
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)

    else:
        raise ValueError(
            f"Unknown search_param: {search_param}. "
            f"Choose from: 'knn', 'radius', 'hybrid'"
        )

    # Estimate normals
    pcd.estimate_normals(
        search_param=search,
        fast_normal_computation=fast_normal_computation
    )

    # Orient normals if requested
    if orient_method is not None:
        if orient_method == "camera":
            if orient_direction is None:
                raise ValueError(
                    "orient_direction (camera location) is required for 'camera' method"
                )
            pcd.orient_normals_towards_camera_location(
                camera_location=orient_direction.astype(np.float64)
            )

        elif orient_method == "direction":
            if orient_direction is None:
                raise ValueError(
                    "orient_direction (reference vector) is required for 'direction' method"
                )
            pcd.orient_normals_to_align_with_direction(
                orientation_reference=orient_direction.astype(np.float64)
            )

        elif orient_method == "tangent":
            if orient_k <= 0:
                raise InvalidArgumentError(f"orient_k must be positive, got {orient_k}.")
            pcd.orient_normals_consistent_tangent_plane(k=orient_k)

        else:
            raise ValueError(
                f"Unknown orient_method: {orient_method}. "
                f"Choose from: 'camera', 'direction', 'tangent', or None"
            )

    # Extract normals
    normals = np.asarray(pcd.normals).astype(points.dtype)

    return normals


def depth_to_normal_map(
    depth: np.ndarray,
    cam: Camera,
    map_type: str = "MPI",
    patch_size: int = 3,
    curvature_threshold: float = 0.1,
) -> np.ndarray:
    """
    Convert depth map to surface normal map using SVD-based PCA.

    This function estimates surface normals using PCA and validates planarity
    using curvature measure from SVD singular values.

    Args:
        depth (np.ndarray, [H,W]): Depth map.
        cam (Camera): Camera instance with [H,W] resolution.
        map_type (str): Depth map representation ("MPI", "MSI", "MCI"). Default "MPI".
        patch_size (int): Patch size (3 or 5). Larger = more robust, slower. Default 3.
        curvature_threshold (float): Max curvature for planar patches (0-1). Default 0.1.
            Curvature = σ₃/(σ₁+σ₂+σ₃) where sigmas are singular values from SVD.
            - 0 = perfect plane
            - >threshold = non-planar (corner/edge/noise/discontinuity)

    Returns:
        normals (np.ndarray, [H,W,3]): Unit normals pointing towards camera.
            Non-planar regions have zero normals.

    Raises:
        InvalidDimensionError: If depth resolution doesn't match camera.
        InvalidArgumentError: If map_type or patch_size is invalid.

    Example:
        >>> from spatialkit.camera import PerspectiveCamera
        >>> import numpy as np
        >>> cam_dict = {
        ...     "image_size": (640, 480),
        ...     "focal_length": (500, 500),
        ...     "principal_point": (320, 240),
        ... }
        >>> cam = PerspectiveCamera(cam_dict)
        >>> depth = np.random.rand(480, 640).astype(np.float32) * 5.0
        >>> normals = depth_to_normal_map(depth, cam)
        >>> # Stricter planarity
        >>> normals = depth_to_normal_map(depth, cam, curvature_threshold=0.05)
    """
    if depth.shape != cam.hw:
        raise InvalidDimensionError(
            f"Depth resolution {depth.shape} must match camera {cam.hw}."
        )

    if map_type not in ["MPI", "MSI", "MCI"]:
        raise InvalidArgumentError(
            f"Unsupported map_type: {map_type}. Choose 'MPI', 'MSI', or 'MCI'."
        )

    if not isinstance(patch_size, int) or patch_size % 2 != 1 or patch_size < 3 or patch_size > min(cam.hw):
        raise InvalidArgumentError(f"patch_size must be odd integer in [3,{min(cam.hw)}], got {patch_size}.")

    H, W = depth.shape

    pts3d, valid_mask = _unproject_depth_to_points(depth, cam, map_type)
    pts3d = swapaxes(pts3d, 0, 1).reshape(H, W, 3)
    valid_mask = valid_mask.reshape(H, W)

    patches_pts = extract_patches(pts3d, patch_size=patch_size, padding='same').reshape(H, W, -1, 3)

    # Compute normals using SVD
    centroids = mean(patches_pts, dim=2, keepdims=True) # (H,W,1,3)
    centered = patches_pts - centroids  # (H,W,P²,3)

    _, S, Vt = svd(centered.reshape(H * W, patch_size**2, 3))

    normals = Vt[:, -1, :].reshape(H, W, 3) # Last right singular vector as normal
    normals = normalize(normals, dim=-1, eps=EPSILON)

    # Orient normals to point towards camera
    dot_product = sum(normals * pts3d, dim=-1, keepdims=True)
    normals = where(dot_product > 0, -normals, normals)

    sigma1, sigma2, sigma3 = S[:, 0], S[:, 1], S[:, 2]
    sigma_sum = sigma1 + sigma2 + sigma3
    curvature = (sigma3 / (sigma_sum + EPSILON)).reshape(H, W)

    invalid = logical_or(curvature > curvature_threshold, logical_not(valid_mask))
    normals = where(expand_dim(invalid, dim=-1), zeros_like(normals), normals)

    return normals


def depth_to_normal_map_fast(
    depth: np.ndarray,
    depth_threshold: float = 0.1,
) -> np.ndarray:
    """
    Fast normal estimation from depth map using depth gradient cross product.

    This function computes surface normals directly from depth gradients:
    - Tangent u: [1, 0, ∂D/∂u]
    - Tangent v: [0, 1, ∂D/∂v]
    - Normal = normalize(tangent_u x tangent_v)

    Assumptions:
    - MPI depth map (depth = Z coordinate)
    - Depth discontinuities indicate surface boundaries

    Args:
        depth (np.ndarray, [H,W]): Depth map where values represent Z coordinate.
        depth_threshold (float): Maximum depth gradient magnitude for valid normals.
            Larger gradients are considered discontinuities. Default is 0.1.

    Returns:
        normals (np.ndarray, [H,W,3]): Normal map with unit vectors pointing towards camera.
            Invalid regions (zero depth, discontinuities) have zero normals.

    Example:
        >>> import numpy as np
        >>> depth = np.random.rand(480, 640).astype(np.float32) * 5.0
        >>> normals = depth_to_normal_map_fast(depth)
        >>> normals.shape
        (480, 640, 3)

    Note:
        - For more robust estimation with fisheye/wide FOV cameras, use depth_to_normal_map()
        - Normals point TOWARDS camera (positive Z is into the scene, normals have negative Z)
    """
    # Pad depth for computing differences using edge replication
    depth_padded = pad(depth, pad_width=((1, 1), (1, 1)), mode='edge')

    # Compute depth gradients using central differences
    # dD/du (horizontal gradient)
    dD_du = (depth_padded[:, 2:] - depth_padded[:, :-2]) / 2.0  # (H+2, W)
    dD_du = dD_du[1:-1, :]  # (H, W)

    # dD/dv (vertical gradient)
    dD_dv = (depth_padded[2:, :] - depth_padded[:-2, :]) / 2.0  # (H, W+2)
    dD_dv = dD_dv[:, 1:-1]  # (H, W)

    # Surface tangent vectors
    # Tangent in u direction: [1, 0, dD/du]
    # Tangent in v direction: [0, 1, dD/dv]
    # Cross product: [1,0,dD/du] × [0,1,dD/dv] = [-dD/du, -dD/dv, 1]
    # Negate Z to point towards camera (camera looks down +Z axis)

    normals = stack([-dD_du, -dD_dv, -ones_like(dD_du)], dim=-1)

    # Normalize to unit vectors
    normals = normalize(normals, dim=-1, eps=EPSILON)

    # Identify discontinuities using gradient magnitude
    gradient_mag = sqrt(dD_du**2 + dD_dv**2)

    # Combined validity mask
    valid = depth > 0.0
    valid = logical_and(valid, gradient_mag <= depth_threshold)

    # Zero out invalid normals
    normals[~valid] = 0.0

    return normals
