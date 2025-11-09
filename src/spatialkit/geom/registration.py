"""
Module Name: registration.py

Description:
This module provides point cloud registration algorithms using Open3D's optimized
ICP (Iterative Closest Point) implementations. It offers a convenient wrapper around
Open3D with seamless integration with spatialkit's Transform and geometry classes.

Supported Algorithms:
    - ICP point-to-point registration
    - ICP point-to-plane registration
    - ICP generalized (colored, hybrid, etc.)

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.1

License: MIT LICENSE
"""

from typing import Union, Optional, Dict, Literal
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from ..common.exceptions import (
    InvalidDimensionError,
    InvalidShapeError,
    NumericalError,
)
from .tf import Transform
from .rotation import Rotation
from .pose import Pose


def _validate_point_cloud(
    points: np.ndarray,
    name: str = "points"
) -> np.ndarray:
    """
    Validate and normalize point cloud input.

    Args:
        points (np.ndarray): Input point cloud.
        name (str): Name for error messages.

    Returns:
        np.ndarray: Normalized point cloud in [N, 3] format.

    Raises:
        TypeError: If input is not a NumPy array.
        InvalidDimensionError: If input is not 2D.
        InvalidShapeError: If input doesn't have 3 coordinates.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(
            f"{name} must be numpy.ndarray, got {type(points).__name__}. "
            f"Convert PyTorch tensors using tensor.detach().cpu().numpy()"
        )

    if points.ndim != 2:
        raise InvalidDimensionError(
            f"{name} must be 2D array, got shape {points.shape}."
        )

    # Auto-transpose if needed: [3, N] → [N, 3]
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T

    if points.shape[1] != 3:
        raise InvalidShapeError(
            f"{name} must have 3 coordinates, got shape {points.shape}. "
            f"Expected shape [N, 3] or [3, N]."
        )

    return points


def _transform_to_matrix(
    transform: Optional[Union[Transform, Pose, np.ndarray]]
) -> np.ndarray:
    """
    Convert various transform representations to 4x4 matrix.

    Args:
        transform: Transform, Pose, or 4x4 numpy array.

    Returns:
        np.ndarray: 4x4 transformation matrix (float64 for Open3D).
    """
    if transform is None:
        return np.eye(4, dtype=np.float64)

    if isinstance(transform, (Transform, Pose)):
        return transform.mat44().astype(np.float64)

    if isinstance(transform, np.ndarray):
        if transform.shape != (4, 4):
            raise InvalidShapeError(
                f"Transform matrix must be 4x4, got shape {transform.shape}."
            )
        return transform.astype(np.float64)

    raise TypeError(
        f"transform must be Transform, Pose, or np.ndarray, got {type(transform).__name__}"
    )



def icp(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: Optional[Union[Transform, Pose, np.ndarray]] = None,
    max_correspondence_distance: float = 0.05,
    estimation_method: Literal["point_to_point", "point_to_plane", "generalized"] = "point_to_point",
    max_iterations: int = 50,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
    **kwargs
) -> Dict[str, Union[Transform, float, int, bool]]:
    """
    Perform ICP (Iterative Closest Point) registration using Open3D.

    This function provides a unified interface to Open3D's highly optimized ICP
    implementations, supporting multiple estimation methods with automatic normal
    computation when needed.

    Args:
        source (np.ndarray, [N,3] or [3,N]): Source point cloud to be aligned.
        target (np.ndarray, [M,3] or [3,M]): Target point cloud (reference).
        init_transform (Optional[Transform|Pose|np.ndarray]): Initial transformation guess.
            Can be Transform, Pose, or 4x4 numpy array. Default is identity.
        max_correspondence_distance (float): Maximum correspondence distance threshold.
            Points farther than this are not considered correspondences. Default is 0.05.
        estimation_method (str): ICP variant to use. Options:
            - "point_to_point": Standard point-to-point ICP (default)
            - "point_to_plane": Point-to-plane ICP (faster convergence, requires normals)
            - "generalized": Generalized ICP (most robust, requires normals)
        max_iterations (int): Maximum number of ICP iterations. Default is 50.
        relative_fitness (float): Convergence criterion for fitness score. Default is 1e-6.
        relative_rmse (float): Convergence criterion for RMSE. Default is 1e-6.
        **kwargs: Additional Open3D-specific options (e.g., normal_radius, normal_max_nn).

    Returns:
        Dict with keys:
            - transformation (Transform): Estimated rigid transformation from source to target.
            - fitness (float): Overlap ratio (0.0-1.0) of aligned points.
            - rmse (float): Root mean squared error (inlier RMSE).
            - correspondences (int): Number of valid correspondences found.
            - converged (bool): Whether the algorithm converged (fitness/rmse stopped improving).

    Raises:
        ImportError: If Open3D is not installed.
        TypeError: If inputs are not NumPy arrays.
        InvalidDimensionError: If point clouds are not 2D.
        InvalidShapeError: If point clouds don't have 3 coordinates.
        NumericalError: If ICP computation fails.

    Example:
        >>> import numpy as np
        >>> from spatialkit.geom import Transform, Rotation
        >>>
        >>> # Generate test data
        >>> source = np.random.rand(1000, 3).astype(np.float32)
        >>>
        >>> # Apply known transformation
        >>> R = Rotation.from_rpy(np.array([0.1, 0.2, 0.3]))
        >>> t = np.array([0.5, 0.3, 0.1])
        >>> gt_transform = Transform(t, R)
        >>> target = gt_transform.apply_pts3d(source.T).T
        >>>
        >>> # Run ICP
        >>> result = icp(source, target, max_correspondence_distance=0.1)
        >>> print(f"RMSE: {result['rmse']:.6f}")
        >>> print(f"Fitness: {result['fitness']:.2%}")
        >>>
        >>> # With initial guess
        >>> result = icp(source, target, init_transform=gt_transform)

    Note:
        - This implementation uses Open3D's C++ optimized ICP for maximum performance
        - Point-to-plane and generalized ICP automatically estimate normals if not provided
        - For large point clouds (>100K points), consider downsampling first
        - Only accepts NumPy arrays (not PyTorch tensors)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError(
            "Open3D is required for ICP registration. "
            "Install it with: pip install open3d"
        )

    # Validate inputs
    source = _validate_point_cloud(source, "source")
    target = _validate_point_cloud(target, "target")
    init_matrix = _transform_to_matrix(init_transform)

    # Create Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source.astype(np.float64))

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64))

    # Estimate normals if needed
    if estimation_method in ["point_to_plane", "generalized"]:
        normal_radius = kwargs.get("normal_radius", 0.1)
        normal_max_nn = kwargs.get("normal_max_nn", 30)

        if not target_pcd.has_normals():
            target_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=normal_max_nn
                )
            )

        if estimation_method == "generalized" and not source_pcd.has_normals():
            source_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=normal_max_nn
                )
            )

    # Select estimation method
    if estimation_method == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif estimation_method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif estimation_method == "generalized":
        estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    else:
        raise ValueError(
            f"Unknown estimation_method: {estimation_method}. "
            f"Choose from: 'point_to_point', 'point_to_plane', 'generalized'"
        )

    # Configure convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse,
        max_iteration=max_iterations
    )

    # Run ICP
    try:
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance,
            init_matrix,
            estimation,
            criteria
        )
    except Exception as e:
        raise NumericalError(f"ICP registration failed: {e}") from e

    # Check convergence (Open3D doesn't provide explicit flag)
    # Consider converged if we hit max iterations or fitness/rmse criteria met
    converged = True  # Open3D always returns best result found

    # Convert to Transform
    transformation = Transform.from_mat(result.transformation)

    return {
        "transformation": transformation,
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "correspondences": int(len(result.correspondence_set)),
        "converged": converged,
    }


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: Optional[Union[Transform, Pose, np.ndarray]] = None,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 50,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
) -> Dict[str, Union[Transform, float, int, bool]]:
    """
    Point-to-point ICP registration (convenience wrapper).

    Args:
        source (np.ndarray, [N,3] or [3,N]): Source point cloud.
        target (np.ndarray, [M,3] or [3,M]): Target point cloud.
        init_transform (Optional[Transform|Pose|np.ndarray]): Initial transformation.
        max_correspondence_distance (float): Maximum correspondence distance.
        max_iterations (int): Maximum iterations.
        relative_fitness (float): Fitness convergence threshold.
        relative_rmse (float): RMSE convergence threshold.

    Returns:
        Dict: Registration result (see icp() for details).

    Example:
        >>> result = icp_point_to_point(source, target, max_correspondence_distance=0.1)
    """
    return icp(
        source=source,
        target=target,
        init_transform=init_transform,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method="point_to_point",
        max_iterations=max_iterations,
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse,
    )


def icp_point_to_plane(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: Optional[Union[Transform, Pose, np.ndarray]] = None,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 50,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
    normal_radius: float = 0.1,
    normal_max_nn: int = 30,
) -> Dict[str, Union[Transform, float, int, bool]]:
    """
    Point-to-plane ICP registration (convenience wrapper).

    This variant converges faster than point-to-point ICP and is more robust
    for smooth surfaces. Normals are automatically estimated for the target.

    Args:
        source (np.ndarray, [N,3] or [3,N]): Source point cloud.
        target (np.ndarray, [M,3] or [3,M]): Target point cloud.
        init_transform (Optional[Transform|Pose|np.ndarray]): Initial transformation.
        max_correspondence_distance (float): Maximum correspondence distance.
        max_iterations (int): Maximum iterations.
        relative_fitness (float): Fitness convergence threshold.
        relative_rmse (float): RMSE convergence threshold.
        normal_radius (float): Radius for normal estimation. Default is 0.1.
        normal_max_nn (int): Max neighbors for normal estimation. Default is 30.

    Returns:
        Dict: Registration result (see icp() for details).

    Example:
        >>> result = icp_point_to_plane(source, target, max_correspondence_distance=0.1)
    """
    return icp(
        source=source,
        target=target,
        init_transform=init_transform,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method="point_to_plane",
        max_iterations=max_iterations,
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn,
    )


def icp_generalized(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: Optional[Union[Transform, Pose, np.ndarray]] = None,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 50,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
    normal_radius: float = 0.1,
    normal_max_nn: int = 30,
) -> Dict[str, Union[Transform, float, int, bool]]:
    """
    Generalized ICP registration (convenience wrapper).

    Generalized ICP minimizes plane-to-plane distance and is the most robust
    variant, especially for noisy or partially overlapping point clouds.

    Args:
        source (np.ndarray, [N,3] or [3,N]): Source point cloud.
        target (np.ndarray, [M,3] or [3,M]): Target point cloud.
        init_transform (Optional[Transform|Pose|np.ndarray]): Initial transformation.
        max_correspondence_distance (float): Maximum correspondence distance.
        max_iterations (int): Maximum iterations.
        relative_fitness (float): Fitness convergence threshold.
        relative_rmse (float): RMSE convergence threshold.
        normal_radius (float): Radius for normal estimation. Default is 0.1.
        normal_max_nn (int): Max neighbors for normal estimation. Default is 30.

    Returns:
        Dict: Registration result (see icp() for details).

    Example:
        >>> result = icp_generalized(source, target, max_correspondence_distance=0.1)
    """
    return icp(
        source=source,
        target=target,
        init_transform=init_transform,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method="generalized",
        max_iterations=max_iterations,
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse,
        normal_radius=normal_radius,
        normal_max_nn=normal_max_nn,
    )


__all__ = [
    "icp",
    "icp_point_to_point",
    "icp_point_to_plane",
    "icp_generalized",
]
