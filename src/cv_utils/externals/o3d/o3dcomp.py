import numpy as np
import open3d as o3d
from typing import *
from .common import *

def create_coordinate(scale: float=1.0, radius: float=0.02, pose:Optional[np.ndarray]=None) \
    -> O3dTriMesh:
    """
        Create a coordinate frame with RGB colors for the axes.

        Args:
            size (float): Length of each axis.
            radius (float): Radius of the cylinders representing the axes.
            pose (np.ndarray, [4,4]): 4x4 transformation matrix.

        Return:
            O3dTriMesh: A TriangleMesh object representing the coordinate frame.
    """
    assert(pose is None or pose.shape == (4,4))

    mesh_frame = O3dTriMesh()

    # X axis (red)
    x_axis = O3dTriMesh.create_cylinder(radius, scale)
    x_axis.paint_uniform_color([1, 0, 0])
    x_axis.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]))
    x_axis.translate([scale / 2, 0, 0])
    mesh_frame += x_axis

    # Y axis (green)
    y_axis = O3dTriMesh.create_cylinder(radius, scale)
    y_axis.paint_uniform_color([0, 1, 0])
    y_axis.rotate(o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0]))
    y_axis.translate([0, scale / 2, 0])
    mesh_frame += y_axis

    # Z axis (blue)
    z_axis = O3dTriMesh.create_cylinder(radius, scale)
    z_axis.paint_uniform_color([0, 0, 1])
    z_axis.translate([0, 0, scale / 2])
    mesh_frame += z_axis

    # Apply the transformation to the entire frame
    if pose is not None: mesh_frame.transform(pose)

    return mesh_frame

def create_camera_indicator_frame(cam_size: Tuple[int, int], focal_length:float, color:Optional[Tuple[int]]=[255, 0, 0],
                        scale:float=0.5, pose:Optional[np.ndarray]=None, image:Optional[np.ndarray]=None) -> O3dLineSet:
    """
        Create a camera indicator.

        Args:
            cam_size (Tuple[int,int]): Camera size (width, height).
            focal_length (float): focal_length.
            color (Tuple[int,int,int]): RGB color.
            scale (float): camera indicator scale.
            pose (np.ndarray, [4,4]): 4x4 transformation matrix.
            image (np.ndarray, [H,W,3], optional): RGB image.

        Return:
            O3dLineSet: A TriangleMesh object representing the coordinate frame.

        Details:
        - cam_size and image's resolution are same.
    """
    assert(pose is None or pose.shape == (4,4))
    assert(image is None or image.shape[0:2] == cam_size[::-1])

    w = cam_size[0] / focal_length
    h = cam_size[1] / focal_length

    # Define the vertices of the pyramid with fixed base and height
    cam_vertices = np.array([
        [-w / 2., -h / 2., 1],  # Bottom-left
        [w / 2., -h / 2., 1],  # Bottom-right
        [w / 2., h / 2., 1],  # Top-right
        [-w / 2., h / 2., 1],  # Top-left
        [0, 0, 0]  # Apex at (0, 0, 0)
    ], dtype=np.float32) * scale

    # Define the edges of the pyramid
    cam_edges = np.array([
        [0, 1],  # Bottom edge
        [1, 2],  # Bottom edge
        [2, 3],  # Bottom edge
        [3, 0],  # Bottom edge
        [0, 4],  # Side edge
        [1, 4],  # Side edge
        [2, 4],  # Side edge
        [3, 4],  # Side edge
    ], dtype=np.int32)

    # Indicator vertices
    indicator_vertices = np.array([
        [-w / 8., -h / 2., 1],  # Indicator top-left
        [w / 8, -h / 2., 1],  # Indicator top-right
        [0, -h / 1.6, 1]  # Indicator top
    ], dtype=np.float32) * scale

    # Indicator edges
    indicator_edges = np.array([
        [0, 1],  # Indicator base
        [1, 2],  # Indicator right
        [2, 0]   # Indicator left
    ], dtype=np.int32)

    # Combine vertices and edges
    vertices = np.vstack((cam_vertices, indicator_vertices))
    edges = np.vstack((cam_edges, indicator_edges + len(cam_vertices)))


    # Create a LineSet object
    cam_indicator = O3dLineSet()
    cam_indicator.points = o3d.utility.Vector3dVector(vertices)
    cam_indicator.lines = o3d.utility.Vector2iVector(edges)

    # Convert color from 255 scale to 0-1 scale for Open3D
    color = np.array(color) / 255.0

    # Apply color to all edges
    colors = [color for _ in range(len(edges))]
    cam_indicator.colors = o3d.utility.Vector3dVector(colors)

    if pose is not None:
        cam_indicator.transform(pose)

    if image is not None:
        image_plane = create_image_plane(image,(w*scale,h*scale),scale,pose)
        return cam_indicator, image_plane
    return cam_indicator

def create_image_plane(image:np.ndarray,plane_size:Tuple[float,float], z:float=1.,pose:Optional[np.ndarray]=None) \
    -> O3dTriMesh:
    """
        Create an image plane from Image
        

        Args:
            image (np.ndarray, [H,W,3]): Image.
            plane_size (Tuple[float,float]): Image plane size (width,height).
            z (float): z-value of image plane 
            pose (np.ndarray, [4,4]): 4x4 transformation matrix.

        Return:
            O3dTriMesh: A TriangleMesh object representing the coordinate frame.

        Details:
        - The center of plane is (0,0) before transformed.
          i.e. palne size is [2,3], then x ~ [-1,1] and y ~ [-1.5,1.5] and z = 0  
        - # of vertices is H*W
        - # of faces is H*W*2 (two faces per pixel)
    """
    assert(pose is None or pose.shape == (4,4))

    height,width = image.shape[:2]

    x,y=np.meshgrid(np.linspace(-plane_size[0]/2.,plane_size[0]/2.,width),
                    np.linspace(-plane_size[1]/2.,plane_size[1]/2.,height))
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    vertices = np.concatenate([x,y,np.ones_like(x)*z],axis=1) # N * 3
    if image.dtype == np.uint8:
        colors = image.astype(np.float64).reshape(-1,3) / 255.
    else: # floating points
        colors = image.reshape(-1,3)

    indices = np.arange(height*width).reshape(height,width)
    v1 = indices[:-1,:-1].reshape(-1,1)
    v2 = indices[:-1,1:].reshape(-1,1)
    v3 = indices[1:,:-1].reshape(-1,1)
    v4 = indices[1:,1:].reshape(-1,1)

    faces = np.vstack([
        np.concatenate([v1,v3,v4], axis=1),
        np.concatenate([v1,v4,v2], axis=1),
        ]) 
    
    mesh = O3dTriMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    if pose is not None: mesh.transform(pose) # Transform

    return mesh

def create_line_from_points(point1: np.ndarray, point2: np.ndarray, color:Optional[Tuple[int]]=None) -> O3dLineSet:
    """
        Create a line connecting two 3D points.

        Args:
            point1 (np.ndarray, [3,]): [x, y, z] coordinates of the first point.
            point2 (np.ndarray, [3,]): [x, y, z] coordinates of the second point.

        Return:
            O3dLineSet: A LineSet object representing the line between the two points.
    """
    
    points = np.array([point1, point2], dtype=np.float64)
    lines = np.array([[0, 1]], dtype=np.int32)

    line_set = O3dLineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    if color is None: color = np.random.rand(3)
    else: color = np.array(color) / 255.

    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set