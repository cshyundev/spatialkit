import numpy as np
from data import *
from module.pose import Pose
from module.camera import *
from typing import *
import os.path as osp 

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
        raise NotImplementedError        
    
    def essential_matrix(self, idx1:int, idx2:int):
        ## E = t×R,  
        ## [R,t]: relative pose from frame1 to frame2
        raise NotImplementedError
    
    def draw_epipolar_line(self, idx1:int, idx2:int, save_path: str,
                           left_pt2d:List[Tuple[int]]=None):
        raise NotImplementedError
    
    def save_point_cloud(self, save_path: str):
        raise NotImplementedError