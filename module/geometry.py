import numpy as np
from module.data import *
from module.pose import Pose
from module.camera import *
from module.io import * 
import open3d as o3d 
from typing import *
import os.path as osp 
import cv2 as cv
from matplotlib import pyplot as plt



def make_point_cloud(pt3d: Array, colors: Optional[Array]=None, save_path:str=None):
    pt3d = convert_numpy(pt3d)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt3d)
    if colors is not None:
        colors = convert_numpy(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if save_path: o3d.io.write_point_cloud(save_path, pcd)
    return pcd

def make_mesh(vertices:Array, triangles:Array, vertex_colors:Optional[Array]=None, save_path:str=None):
    vertices, triangles = convert_numpy(vertices),convert_numpy(triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if vertex_colors is not None:
        mesh.vertex_colors = convert_numpy(vertex_colors)
    if save_path: o3d.io.write_triangle_mesh(save_path, mesh)
    return mesh

class MultiView:
    """
    Multiview Geometry
    
    Available Contents
    Epipolar Line: TODO
    make point clouds: TODO
    """
    def __init__(self,
                image_path: List[str],
                cameras: List[Camera],
                poses: List[Pose],
                depth_path: Optional[List[str]]=None,
                normal_path: Optional[List[str]]=None
                ) -> None:
        
        self.image_path = image_path
        self.cameras = cameras
        self.poses: poses
        self.depth_path = depth_path
        self.normal_path = normal_path
        
        self.n_views = len(cameras)      
                
    @staticmethod
    def from_meta_data(
        root_path: str,
        meta_data:Dict[str,Any]) -> 'MultiView':
        image_path = []
        cameras = []
        poses = []
        depth_path = []
        normal_path = []
        
        height = meta_data["height"]
        width = meta_data["width"]
        
        frames = meta_data["frames"]
        for frame in frames:
            image_path.append(osp.join(root_path,frame["rgb_path"]))
            depth_path.append(osp.join(root_path,frame["mono_depth_path"]))
            normal_path.append(osp.join(root_path,frame["mono_normal_path"])) 
            poses.append(Pose.from_mat4(np.array(frame["camtoworld"]))) 
            # create camera
            K = frame["intrinsics"]
            cam_dict = {}
            cam_dict["fx"] = K[0,0]
            cam_dict["fy"] = K[1,1]
            cam_dict["cx"] = K[0,2]
            cam_dict["cy"] = K[1,2]
            cam_dict["skew"] = K[0,1]
            cam_dict["height"] = height
            cam_dict["width"] = width
            cameras.append(Camera.create_cam(cam_dict))
            
        return MultiView(image_path,cameras,poses,
                         depth_path,normal_path)   
    
    def fundamental_matrix(self, idx1:int, idx2:int):
        ## F = K_1^-1*(t×R)*K_2^-1
        ## [R,t]: relative pose from frame1 to frame2
        
        inv_K1 = self.cameras[idx1].inv_K()
        inv_K2 = self.cameras[idx2].inv_K()
        E = self.essential_matrix(idx1,idx2)
        F = matmul(inv_K1,E)
        F = matmul(F,inv_K2)
        return F
    
    def essential_matrix(self, idx1:int, idx2:int):
        ## E = t×R,  
        w1tow2 = self.relative_pose(idx1,idx2)
        skew_t = w1tow2.skew_t()
        r_mat = w1tow2.rot_mat()
        return matmul(skew_t,r_mat)        
    
    def relative_pose(self, idx1:int, idx2:int) -> Pose:
        ## [R,t]: relative pose from frame1 to frame2
        ## Relative Pose = merge(world1tocam1)Pose2 - Pose1
        w1toc = self.poses[idx1].inverse()
        ctow2 = self.poses[idx2]
        return w1toc.merge(ctow2)
    
    def draw_epipolar_line(self, idx1:int, idx2:int, save_path: str,
                           left_pt2d:List[Tuple[int]]=None):
        
        cam1 = self.cameras[idx1]
        cam2 = self.cameras[idx2]
        
        if (cam1.cam_type == cam2.cam_type) and cam1.cam_type == CamType.PINHOLE:
            self.__draw_epipolar_line_between_pinholes(idx1,idx2, save_path, left_pt2d)
        
        ## 
        
        raise NotImplementedError
    
    def __draw_lines(img1:np.ndarray,img2:np.ndarray, lines, left_pt2d):
        return img1,img2
    
    def __draw_epipolar_line_between_pinholes(self, idx1:int, idx2:int,
                                              left_pt2d:List[Tuple[int,int]]
                                              ,save_path:str):
        """
        Draw Epipolar Line between Pinhole Cameras
        Args:
            idx1,idx2: left and right camera inices respectively
            left_pt2d: [n,2], 2D points in Left Image
            save_path: path to save the result
        """
        assert(len(left_pt2d) > 0), ("To draw epipolar line, 2D points must be needed.")
        
        F = self.fundamental_matrix(idx1,idx2)
        F = convert_numpy(F)
        lines=[]
        pts_homo=[]
        for pt in left_pt2d:
            pt_homo = np.array([pt[0], pt[1], 1.]).reshape(3,1)
            pts_homo.append(pt_homo)
            lines.append(F@pt_homo) 
        img1, img2 = read_image(self.image_path[idx1]),read_image(self.image_path[idx2]) 
        img1,img2 = self.__draw_lines(img1, img2, lines, left_pt2d)
        
        concated_image = concat_image([img1,img2], vertical=False)
        
        
        return        
    
    def save_point_cloud(self, save_path: str):
        
        pts3d = []
        colors = []
        
        for i in range(self.n_views):
            rays = self.cameras[i].get_rays() # [n,3]
            origin, direction = self.poses[i].get_origin_direction(rays) # [n,3], [n,3]
            depth = read_float(self.depth_path[i]).reshape(-1,1)
            pts3d_w = origin + depth * direction            
            pts3d.append(pts3d_w)
            
            color = read_image(self.image_path[i])            
            colors.append(color)
        
        pt3d = concat(pt3d, 0)
        colors = concat(colors, 0)            
        
        make_point_cloud(pt3d, colors, save_path)
        
        
        raise NotImplementedError

if __name__ == '__main__':
    arr = np.ones(4)
    norm_arr = normalize(arr,0)
    norm = norm_l2(arr,0,False)
    
    print(arr)
    print(norm_arr)
    print(norm)