"""
Module Name: o3dutils.py

Description:
This module provides utility functions for creating, saving, loading, and visualizing
3D geometries using Open3D. It includes functions to create point clouds and meshes
from arrays, save and load these geometries from files, and visualize them in a window.

Main Functions:
- create_point_cloud: Create a point cloud from 3D points and optional colors.
- create_mesh: Create a mesh from vertices, triangles, and optional vertex colors.
- create_mesh_from_pcd: Generate a mesh from a point cloud using the Ball Pivoting Algorithm.
- save_mesh: Save a mesh to a specified file path.
- save_pcd: Save a point cloud to a specified file path.
- load_mesh: Load a mesh from a specified file path.
- load_pcd: Load a point cloud from a specified file path.
- visualize_geometries: Visualize a list of geometries in a single window.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT License
"""

from typing import Optional, List, Any
import numpy as np
import open3d as o3d
from .common import O3dPCD, O3dTriMesh
from ..common.exceptions import InvalidShapeError, RenderingError, IncompatibleTypeError, IOError, FileNotFoundError, FileFormatError


def create_point_cloud(pt3d: np.ndarray, colors: Optional[np.ndarray] = None) -> Any:
    """
    Create a point cloud from 3D points.

    Args:
        pt3d (np.ndarray, [N,3]): Array of 3D points (N, 3).
        colors (Optional[np.ndarray], [N,3]): Array of colors for the points (N, 3). Defaults to None.

    Returns:
        O3dPCD: The created point cloud.
        
    Raises:
        InvalidShapeError: If pt3d is not a 2D array with shape (N, 3) or colors have incompatible shape.
        RenderingError: If point cloud creation fails.
    """
    if not isinstance(pt3d, np.ndarray) or pt3d.ndim != 2 or pt3d.shape[1] != 3:
        raise InvalidShapeError(
            f"pt3d must be a 2D array with shape (N, 3), got {type(pt3d)} with shape {getattr(pt3d, 'shape', 'unknown')}. "
            f"Please provide a valid array of 3D points."
        )
    
    if colors is not None:
        if not isinstance(colors, np.ndarray) or colors.ndim != 2 or colors.shape[1] != 3:
            raise InvalidShapeError(
                f"colors must be a 2D array with shape (N, 3), got {type(colors)} with shape {getattr(colors, 'shape', 'unknown')}. "
                f"Please provide a valid array of RGB colors."
            )
        if colors.shape[0] != pt3d.shape[0]:
            raise InvalidShapeError(
                f"Number of colors ({colors.shape[0]}) must match number of points ({pt3d.shape[0]}). "
                f"Please ensure colors and points have the same count."
            )

    try:
        pcd = O3dPCD()
        pcd.points = o3d.utility.Vector3dVector(pt3d)
        if colors is not None:
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    except Exception as e:
        raise RenderingError(f"Failed to create point cloud: {e}") from e


def create_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None,
) -> Any:
    """
    Create a mesh from vertices and triangles.

    Args:
        vertices (np.ndarray): Array of vertex positions (N, 3).
        triangles (np.ndarray): Array of triangle indices (N, 3).
        vertex_colors (Optional[np.ndarray]): Array of colors for the vertices (N, 3). Defaults to None.

    Returns:
        o3d.geometry.TriangleMesh: The created mesh.
        
    Raises:
        InvalidShapeError: If vertices, triangles, or vertex_colors have incorrect shapes.
        RenderingError: If mesh creation fails.
    """
    if not isinstance(vertices, np.ndarray) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise InvalidShapeError(
            f"vertices must be a 2D array with shape (N, 3), got {type(vertices)} with shape {getattr(vertices, 'shape', 'unknown')}. "
            f"Please provide a valid array of vertex positions."
        )
    
    if not isinstance(triangles, np.ndarray) or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise InvalidShapeError(
            f"triangles must be a 2D array with shape (M, 3), got {type(triangles)} with shape {getattr(triangles, 'shape', 'unknown')}. "
            f"Please provide a valid array of triangle indices."
        )
    
    if vertex_colors is not None:
        if not isinstance(vertex_colors, np.ndarray) or vertex_colors.ndim != 2 or vertex_colors.shape[1] != 3:
            raise InvalidShapeError(
                f"vertex_colors must be a 2D array with shape (N, 3), got {type(vertex_colors)} with shape {getattr(vertex_colors, 'shape', 'unknown')}. "
                f"Please provide a valid array of vertex colors."
            )
        if vertex_colors.shape[0] != vertices.shape[0]:
            raise InvalidShapeError(
                f"Number of vertex colors ({vertex_colors.shape[0]}) must match number of vertices ({vertices.shape[0]}). "
                f"Please ensure colors and vertices have the same count."
            )

    try:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        if vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        return mesh
    except Exception as e:
        raise RenderingError(f"Failed to create mesh: {e}") from e


def create_mesh_from_pcd(
    pcd: Any,
    method: str = "BPA",
    radius_multiplier: float = 3.0,
    normal_radius: float = 1.0,
    normal_max_nn: int = 30,
) -> Any:
    """
    Make a mesh from a Point Cloud.

    Args:
        pcd (o3d.geometry.PointCloud): PointCloud instance.
        method (str): An algorithm name to make mesh. Currently 'BPA' (Ball Pivoting Algorithm) is only available.
        radius_multiplier (float): Multiplier for the radius used in BPA.
        normal_radius (float): Radius used for normal estimation.
        normal_max_nn (int): Maximum number of nearest neighbors used for normal estimation.

    Returns:
        mesh (o3d.geometry.TriangleMesh): The mesh generated from the point cloud.

    Raises:
        IncompatibleTypeError: If the input pcd is not an instance of o3d.geometry.PointCloud.
        IncompatibleTypeError: If an unsupported method is specified.
        RenderingError: If mesh generation fails.
    """
    if not isinstance(pcd, O3dPCD):
        raise IncompatibleTypeError(
            f"pcd must be an instance of o3d.geometry.PointCloud, got {type(pcd)}. "
            f"Please provide a valid Open3D PointCloud object."
        )

    if not isinstance(method, str) or method.lower() != "bpa":
        raise IncompatibleTypeError(
            f"Unsupported method '{method}'. Currently only 'BPA' (Ball Pivoting Algorithm) is supported. "
            f"Please use method='BPA'."
        )

    try:
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )

        # Compute mesh using Ball Pivoting Algorithm
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = radius_multiplier * avg_dist

        mesh = O3dTriMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2, radius * 4])
        )
        
        return mesh
    except Exception as e:
        raise RenderingError(f"Failed to create mesh from point cloud: {e}") from e


def save_mesh(mesh: Any, path: str) -> None:
    """
    Save a mesh to a file.

    Args:
        mesh (O3dTriMesh): The mesh to save.
        path (str): The file path to save the mesh.
        
    Raises:
        IncompatibleTypeError: If mesh is not a valid Open3D TriangleMesh.
        IOError: If file saving fails.
    """
    if not isinstance(mesh, O3dTriMesh):
        raise IncompatibleTypeError(
            f"mesh must be an instance of o3d.geometry.TriangleMesh, got {type(mesh)}. "
            f"Please provide a valid Open3D TriangleMesh object."
        )
    
    try:
        success = o3d.io.write_triangle_mesh(path, mesh)
        if not success:
            raise IOError(f"Failed to save mesh to '{path}'. Please check the file path and permissions.")
    except Exception as e:
        raise IOError(f"Failed to save mesh to '{path}': {e}") from e


def save_pcd(pcd: Any, path: str) -> None:
    """
    Save a point cloud to a file.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to save.
        path (str): The file path to save the point cloud.
        
    Raises:
        IncompatibleTypeError: If pcd is not a valid Open3D PointCloud.
        IOError: If file saving fails.
    """
    if not isinstance(pcd, O3dPCD):
        raise IncompatibleTypeError(
            f"pcd must be an instance of o3d.geometry.PointCloud, got {type(pcd)}. "
            f"Please provide a valid Open3D PointCloud object."
        )
    
    try:
        success = o3d.io.write_point_cloud(path, pcd)
        if not success:
            raise IOError(f"Failed to save point cloud to '{path}'. Please check the file path and permissions.")
    except Exception as e:
        raise IOError(f"Failed to save point cloud to '{path}': {e}") from e


def load_mesh(path: str) -> Any:
    """
    Load a mesh from a file.

    Args:
        path (str): The file path to load the mesh from.

    Returns:
        o3d.geometry.TriangleMesh: The loaded mesh.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        FileFormatError: If the file format is not supported or the file is corrupted.
        IOError: If file loading fails.
    """
    import os
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mesh file not found: '{path}'. Please check the file path and ensure the file exists."
        )
    
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) == 0:
            raise FileFormatError(
                f"Failed to load mesh from '{path}': empty or invalid mesh data. "
                f"Please ensure the file contains valid mesh data and is in a supported format."
            )
        return mesh
    except Exception as e:
        raise IOError(f"Failed to load mesh from '{path}': {e}") from e


def load_pcd(path: str) -> Any:
    """
    Load a point cloud from a file.

    Args:
        path (str): The file path to load the point cloud from.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        FileFormatError: If the file format is not supported or the file is corrupted.
        IOError: If file loading fails.
    """
    import os
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Point cloud file not found: '{path}'. Please check the file path and ensure the file exists."
        )
    
    try:
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise FileFormatError(
                f"Failed to load point cloud from '{path}': empty or invalid point cloud data. "
                f"Please ensure the file contains valid point cloud data and is in a supported format."
            )
        return pcd
    except Exception as e:
        raise IOError(f"Failed to load point cloud from '{path}': {e}") from e


def visualize_geometries(geometries: List[Any], window_name: str = "Open3D") -> None:
    """
    Visualize multiple geometries in a single window.

    Args:
        geometries (List[o3d.geometry.Geometry]): List of geometries to visualize.
        window_name (str): The window name for visualization. Defaults to "Open3D".
        
    Raises:
        IncompatibleTypeError: If geometries is not a list or contains invalid geometry objects.
        RenderingError: If visualization fails.
    """
    if not isinstance(geometries, list):
        geometries = [geometries]
    
    if not geometries:
        raise IncompatibleTypeError(
            "geometries list cannot be empty. Please provide at least one geometry object to visualize."
        )
    
    # Validate all geometries are valid Open3D objects
    for i, geom in enumerate(geometries):
        if not hasattr(geom, 'get_geometry_type'):
            raise IncompatibleTypeError(
                f"geometries[{i}] is not a valid Open3D geometry object: {type(geom)}. "
                f"Please ensure all objects in the list are valid Open3D geometry instances."
            )
    
    try:
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
    except Exception as e:
        raise RenderingError(f"Failed to visualize geometries: {e}") from e
