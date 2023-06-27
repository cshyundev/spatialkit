import numpy as np
from data import *
from module.pose import Pose
from module.camera import *
from io import *
from o3d import *
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
        ## [R,t]: relative pose from frame1 to frame2
        ## Relative Pose = merge(world1tocam1)Pose2 - Pose1
        relative_pose = merge_transform(self.poses[idx1].inverse_pose(), self.poses[idx2])
        skew_t = relative_pose.skew_t()
        r_mat = relative_pose.rot_mat()
        E = skew_t@r_mat
        return E        
    
    def draw_epipolar_line(self, idx1:int, idx2:int, save_path: str,
                           left_pt2d:List[Tuple[int]]=None):
        
        cam1 = self.cameras[idx1]
        cam2 = self.cameras[idx2]
        
        if (cam1.cam_type == cam2.cam_type) and cam1.cam_type == CamType.PINHOLE:
            self.__draw_epipolar_line_between_pinholes(idx1,idx2, save_path, left_pt2d)
        
        ## 
        
        raise NotImplementedError
    
    def __draw_epipolar_line_between_pinholes(self, idx1:int, idx2:int, save_path:str,
                                              left_pt2d:List[Tuple[int]]=None):
        """
        Draw Epipolar Line between Pinhole Cameras
        l: 
        """
        return None
    
    
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