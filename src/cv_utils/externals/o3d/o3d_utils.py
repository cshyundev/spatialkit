import numpy as np
from typing import * 
import o3d



def make_point_cloud(pt3d: np.ndarray, colors: Optional[np.ndarray]=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt3d)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def make_mesh(vertices:np.ndarray, triangles:np.ndarray, vertex_colors:Optional[np.ndarray]=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if vertex_colors is not None:
        mesh.vertex_colors = vertex_colors
    return mesh

def save_mesh(mesh, path:str):
    o3d.io.write_triangle_mesh(path, mesh)
    

def save_pcd(pcd, path:str):
    o3d.io.write_point_cloud(path, pcd)
