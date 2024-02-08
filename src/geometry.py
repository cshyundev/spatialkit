import numpy as np

from src.hybrid_operations import *
from src.pose import Pose
from src.camera import *
from src.file_utils import *
from src.plot import * 
import open3d as o3d 
from typing import *
import os.path as osp
import matplotlib.pyplot as plt 

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
        self.poses = poses
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
            poses.append(Pose.from_mat(np.array(frame["camtoworld"]))) 
            # create camera
            K = frame["intrinsics"]
            cam_dict = {}
            cam_dict["fx"] = K[0][0]
            cam_dict["fy"] = K[1][1]
            cam_dict["cx"] = K[0][2]
            cam_dict["cy"] = K[1][2]
            cam_dict["skew"] = K[0][1]
            cam_dict["height"] = height
            cam_dict["width"] = width
            cam_dict["cam_type"] = meta_data["camera_model"]
            cameras.append(Camera.create_cam(cam_dict))
            
        return MultiView(image_path,cameras,poses,
                         depth_path,normal_path)   
    
    @staticmethod
    def from_dict(frame_dict:Dict[str,Any]) -> 'MultiView':
        root = frame_dict["root"]
        frames = frame_dict["frames"]
        image_path = []
        depth_path = []
        normal_path = []
        cameras = []
        poses = []
        for frame in frames:
            cameras.append(Camera.create_cam(frame["cam"]))
            poses.append(Pose.from_mat(np.array(frame["camtoworld"])))
            image_path.append(osp.join(root,frame["image"]))
            if "depth" in frame:
                depth_path.append(osp.join(root,frame["depth"]))
        
        return MultiView(image_path,cameras,poses,depth_path,normal_path)

    def fundamental_matrix(self, idx1:int, idx2:int):
        ## F = K_1^-1*(t×R)*K_2^-1
        ## [R,t]: relative pose from frame1 to frame2
        
        inv_K1 = self.cameras[idx1].inv_K()
        K2 = self.cameras[idx2].K()
        E = self.essential_matrix(idx1,idx2)
        F = matmul(np.linalg.inv(K2.T) ,E)
        F = matmul(F,inv_K1)
        return F
    
    def essential_matrix(self, idx1:int, idx2:int):
        ## E = [t]×R,  
        c1_to_c2 = self.relative_pose(idx1,idx2)
        skew_t = c1_to_c2.skew_t()
        r_mat = c1_to_c2.rot_mat()
        return matmul(skew_t,r_mat)        
    
    def relative_pose(self, idx1:int, idx2:int) -> Pose:
        ## [R,t]: relative pose from frame1 to frame2
        ## Relative Pose = Pose2 - Pose1
        c1tow = self.poses[idx1]
        wtoc2 = self.poses[idx2].inverse()
        return wtoc2.merge(c1tow)
    
    def choice_points(self, idx1:int, idx2:int, n_points:int) -> List[Tuple[int,int]]:
        left, right = read_image(self.image_path[idx1]),read_image(self.image_path[idx2])
        plt.subplot(1,2,1),plt.imshow(left)
        plt.subplot(1,2,2),plt.imshow(right)
        pts = plt.ginput(n_points)
        pts = np.int32(pts)
        plt.close()
        return pts.tolist()
    
    def draw_epipolar_line(self, idx1:int, idx2:int,
                           left_pt2d:List[Tuple[int]]=None):
        cam1 = self.cameras[idx1]
        cam2 = self.cameras[idx2]
        if (cam1.cam_type == cam2.cam_type) and cam1.cam_type == CamType.PINHOLE:
            return self._draw_epipolar_line_between_pinholes(idx1,idx2, left_pt2d)
    
    def _draw_epipolar_line_between_pinholes(self, idx1:int, idx2:int,
                                              left_pt2d:List[Tuple[int,int]]) -> np.ndarray:
        """
        Draw Epipolar Line between Pinhole Cameras
        Args:
            idx1,idx2: left and right camera inices respectively
            left_pt2d: (n,2), 2D points in left image
            save_path: path to save the result
        Return:
            Image: (H,2W,3) or (H,2W), float, 
        """
        assert(len(left_pt2d) > 0), ("To draw epipolar line, 2D points must be needed.")
        left, right = read_image(self.image_path[idx1]),read_image(self.image_path[idx2])
        
        F = self.fundamental_matrix(idx1,idx2)
        F = convert_numpy(F)
        for pt in left_pt2d:
            pt_homo = np.array([pt[0], pt[1], 1.])
            color = tuple(np.random.randint(0,255,3).tolist())
            left = draw_circle(left, pt, 1,color,2)
            right = draw_line_by_line(right,tuple((F@pt_homo).tolist()), color, 2)            
                     
        image = concat_images([left,right], vertical=False)
        return image        
    
    def save_point_cloud(self, save_path: str):
        
        pts3d = []
        colors = []
        
        for i in range(self.n_views):
            rays = self.cameras[i].get_rays(norm=False) # [n,3]
            origin, direction = self.poses[i].get_origin_direction(rays) # [n,3], [n,3]
            depth = read_float(self.depth_path[i]).reshape(-1,1)
            if depth is not None:
                pts3d_w = origin + depth * direction            
                pts3d.append(pts3d_w)
                color = read_image(self.image_path[i],as_float=True).reshape(-1,3) # as_float:0~255 -> 0~1.          
                colors.append(color)
        pts3d = concat(pts3d, 0)
        colors = concat(colors, 0)            
        
        make_point_cloud(pts3d, colors, save_path)
    
    def get_image(self, idx:int) -> np.ndarray:
        return read_image(self.image_path[idx])
    
# if __name__ == '__main__':
    
    
    
    