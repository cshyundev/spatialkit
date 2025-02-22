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


def create_point_cloud(pt3d: np.ndarray, colors: Optional[np.ndarray] = None) -> Any:
    """
    Create a point cloud from 3D points.

    Args:
        pt3d (np.ndarray, [N,3]): Array of 3D points (N, 3).
        colors (Optional[np.ndarray], [N,3]): Array of colors for the points (N, 3). Defaults to None.

    Returns:
        O3dPCD: The created point cloud.
    """
    pcd = O3dPCD()
    pcd.points = o3d.utility.Vector3dVector(pt3d)
    if colors is not None:
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


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
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if vertex_colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


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
        AssertionError: If the input pcd is not an instance of o3d.geometry.PointCloud.
        TypeError: If an unsupported method is specified.
    """
    assert isinstance(pcd, O3dPCD)

    if isinstance(method, str) and method.lower() == "BPA":
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
    else:
        raise TypeError(f"Unsupported method: {method}")

    return mesh


def save_mesh(mesh: Any, path: str) -> None:
    """
    Save a mesh to a file.

    Args:
        mesh (O3dTriMesh): The mesh to save.
        path (str): The file path to save the mesh.
    """
    o3d.io.write_triangle_mesh(path, mesh)


def save_pcd(pcd: Any, path: str) -> None:
    """
    Save a point cloud to a file.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to save.
        path (str): The file path to save the point cloud.
    """
    o3d.io.write_point_cloud(path, pcd)


def load_mesh(path: str) -> Any:
    """
    Load a mesh from a file.

    Args:
        path (str): The file path to load the mesh from.

    Returns:
        o3d.geometry.TriangleMesh: The loaded mesh.
    """
    return o3d.io.read_triangle_mesh(path)


def load_pcd(path: str) -> Any:
    """
    Load a point cloud from a file.

    Args:
        path (str): The file path to load the point cloud from.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    return o3d.io.read_point_cloud(path)


def visualize_geometries(geometries: List[Any], window_name: str = "Open3D") -> None:
    """
    Visualize multiple geometries in a single window.

    Args:
        geometries (List[o3d.geometry.Geometry]): List of geometries to visualize.
        window_name (str): The window name for visualization. Defaults to "Open3D".
    """
    if isinstance(geometries, list) is False:
        geometries = [geometries]
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
