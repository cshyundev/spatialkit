from data import *
import os.path as osp
import open3d as o3d 


def make_point_cloud(pt3d: Array, colors: Optional[Array]=None, save_path:str=None):
    
    pt3d = convert_numpy(pt3d)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt3d)
    
    if colors is not None:
        colors = convert_numpy(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if save_path: o3d.io.write_point_cloud(save_path, pcd)
    
    return pcd