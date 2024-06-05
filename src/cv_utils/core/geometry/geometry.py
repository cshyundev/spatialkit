import numpy as np

from ..operations.hybrid_operations import *
from pose import Pose
from tf import Transform
from camera import *
from typing import *
import os.path as osp
import matplotlib.pyplot as plt 


"""
TODO
1. essential matrix
2. fundamental matrix
3. relative transform(?) 
4. solve pnp 2D-3D Matching
6. recover transform (두 이미지 매칭쌍으로 부터 상대 포즈 구하기)
7. triangulation
8. ICP (3D-3D Matching, or 2D-2D Matching) 이건 후순위
"""


class MultiView:
    """
    Multiview Geometry
    
    Available Contents
    Epipolar Line: epipolar_line_test.py
    make point clouds: TODO
    """
    def __init__(self,
                image_path: List[str],
                cameras: List[Camera],
                poses: List[Pose],
                depth_path: Optional[List[str]]=None,
                normal_path: Optional[List[str]]=None
                ) -> None:
        assert(len(image_path) == len(cameras) and len(image_path) == len(poses)), "Number of images, camera, and poses must be same."
        if depth_path is not None:
            assert(len(image_path) == len(depth_path)), "Number of images and these depth must be same."
        if normal_path is not None:
            assert(len(image_path) == len(normal_path)), "Number of images and these normal must be same."
            

        self.image_path = image_path
        self.cameras = cameras
        self.poses = poses
        self.depth_path = depth_path
        self.normal_path = normal_path
        self.n_views = len(cameras)

        self.has_depth = True if len(self.depth_path) != 0 else False
                
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
        
        inv_K1 = self.cameras[idx1].inv_K
        K2 = self.cameras[idx2].K
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
        left_image, right_image = read_image(self.image_path[idx1]),read_image(self.image_path[idx2])  
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(left_image)
        axs[1].imshow(right_image)
        axs[0].set_title(f"Choose {n_points} points on left image")
        pts = plt.ginput(n_points)
        pts = np.int32(pts)
        plt.close(fig)
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
    
    def get_camera(self, idx:int) -> Camera:
        return self.cameras[idx]

    def __normalize_points(self, pts: Array) -> Tuple[np.ndarray,np.ndarray]:
        """
        Normalize the points so that the mean is 0 and the average distance is sqrt(2).
        """
            
        pts = convert_numpy(pts)
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid
        scale = np.sqrt(2) / np.mean(np.linalg.norm(centered_pts, axis=1))
        transform = np.array([[scale, 0, -scale * centroid[0]],
                              [0, scale, -scale * centroid[1]],
                              [0, 0, 1]])
        normalized_points = np.dot(transform, np.concatenate((pts.T, np.ones((1, pts.shape[0])))))
        return normalized_points.T, transform
    
    def recover_pose(self, left_pts: List[Tuple[int,int]],right_pts: List[Tuple[int,int]], K:Array=None) -> Pose:
        assert(len(left_pts) >= 3), f"To recover the relative pose, correspondence pairs must be larger than 3, but got {len(left_pts)}"
        assert(len(right_pts) >= 3), f"To recover the relative pose, correspondence pairs must be larger than 3, but got {len(right_pts)}"
        assert(len(left_pts) == len(right_pts)), "Correspondence pairs must be same."
        
        left_pts = np.array(left_pts)
        right_pts = np.array(right_pts)
        
        left_pts_norm, T1 = self.__normalize_points(left_pts)
        right_pts_norm, T2 = self.__normalize_points(right_pts)
        
        A = []
        for l,r in zip(left_pts_norm,right_pts_norm):
            x1, y1 = l 
            x2, y2 = r
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

        A = np.array(A)
        # Compute Homography using SVD decomposition 
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape((3, 3))

        # denormalize
        H = np.dot(np.linalg.inv(T2), np.dot(H, T1))

        # Make the last element 1
        H /= H[-1, -1]
        
        return H


    
# if __name__ == '__main__':
    
    
    
    