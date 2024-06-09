import numpy as np
from typing import Optional, List
import open3d as o3d
import math

def make_point_cloud(pt3d: np.ndarray, colors: Optional[np.ndarray] = None):
    """
    Create a point cloud from 3D points.

    Args:
        pt3d (np.ndarray): Array of 3D points (N, 3).
        colors (Optional[np.ndarray]): Array of colors for the points (N, 3). Defaults to None.

    Returns:
        o3d.geometry.PointCloud: The created point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt3d)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def make_mesh(vertices: np.ndarray, triangles: np.ndarray, vertex_colors: Optional[np.ndarray] = None):
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

def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str):
    """
    Save a mesh to a file.

    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to save.
        path (str): The file path to save the mesh.
    """
    o3d.io.write_triangle_mesh(path, mesh)
    
def save_pcd(pcd: o3d.geometry.PointCloud, path: str):
    """
    Save a point cloud to a file.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to save.
        path (str): The file path to save the point cloud.
    """
    o3d.io.write_point_cloud(path, pcd)

def create_camera_model(fov: float, resolution: tuple, camera_pose: np.ndarray):
    """
    Create a camera model visualization.

    Args:
        fov (float): Field of view in degrees.
        resolution (tuple): Resolution of the camera as (width, height).
        camera_pose (np.ndarray): 4x4 transformation matrix representing the camera pose.

    Returns:
        None
    """
    # Calculate camera parameters
    fov_rad = math.radians(fov)
    height = resolution[1]
    width = resolution[0]
    focal_length = height / (2 * math.tan(fov_rad / 2))

    # Apply camera pose
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic.set_intrinsics(width, height, focal_length, focal_length, width / 2, height / 2)
    camera.extrinsic = np.linalg.inv(camera_pose)

    # Create coordinate frame
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # Create frustum
    vertices = [
        [0, 0, 0],  # Camera origin
        [-width / 2, -height / 2, focal_length],  # Four corners of the image plane
        [width / 2, -height / 2, focal_length],
        [width / 2, height / 2, focal_length],
        [-width / 2, height / 2, focal_length],
    ]
    triangles = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1]
    ]
    frustum = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(triangles))
    frustum.paint_uniform_color([1, 0, 0])  # Set color to red

    # Visualize
    o3d.visualization.draw_geometries([mesh_frame, frustum])

def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """
    Load a mesh from a file.

    Args:
        path (str): The file path to load the mesh from.

    Returns:
        o3d.geometry.TriangleMesh: The loaded mesh.
    """
    return o3d.io.read_triangle_mesh(path)

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a file.

    Args:
        path (str): The file path to load the point cloud from.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    return o3d.io.read_point_cloud(path)

def visualize_geometries(geometries: List[o3d.geometry.Geometry], window_name: str = "Open3D"):
    """
    Visualize multiple geometries in a single window.

    Args:
        geometries (List[o3d.geometry.Geometry]): List of geometries to visualize.
        window_name (str): The window name for visualization. Defaults to "Open3D".
    """
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
