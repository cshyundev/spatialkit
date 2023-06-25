import numpy as np
import torch
from data import *
from enum import Enum
from typing import *



class CamType(Enum):
    PINHOLE = ("PINHOLE", "Pinhole Camera Type")
    OPENCV_FISHEYE = ("OPENCV_FISHEYE", "OpenCV Fisheye Camera with Distortion Parameters")
    NONE = ("NONE", "No Camera Type")
    
    @staticmethod
    def from_string(type_str: str)-> 'CamType':
        if type_str == 'PINHOLE':
            return CamType.PINHOLE
        elif type_str == 'OPENCV_FISHEYE':
            return CamType.OPENCV_FISHEYE
        else:
            return CamType.NONE
class Camera:
    def __init__(self, cam_dict: Dict[str, Any]):
        self.cam_type = CamType.NONE
        self.width = cam_dict['width']
        self.height = cam_dict['height']
        
        # if hasattr(cam_dict, 'c2w'):
        #     self.c2w = np.array(cam_dict['c2w']) 
        # else:
        #     self.c2w = Pose(None,None)
    
    def make_pixel_grid(self) -> np.ndarray:
        u, v = np.meshgrid(range(self.width), range(self.height))
        uv = concat([u.reshape((-1, 1)), v.reshape((-1, 1))], 1)
        return uv
    
    def get_rays(self) -> Array:
        raise NotImplementedError
    
    @staticmethod
    def create_cam(cam_dict: Dict[str, Any]) -> 'Camera':
        cam_type = CamType.from_string(cam_dict['cam_type']) 
        
        if cam_type is CamType.PINHOLE:
            return PinholeCamera(cam_dict)
        elif cam_type is CamType.OPENCV_FISHEYE:
            raise NotImplementedError
        else:
            raise Exception("Unkwon Camera Type")

class PinholeCamera(Camera):
    """
    Pinhole Camera Model without distortion Parameters.
    
    """
    def __init__(self, cam_dict: Dict[str, Any]):
        super(PinholeCamera, self).__init__(cam_dict)
        
        self.cam_type = CamType.PINHOLE
        
        self.fx = cam_dict['fx']
        self.fy = cam_dict['fy']        
        self.cx = cam_dict['cx'] 
        self.cy = cam_dict['cy']
        self.skew = cam_dict['skew']
    
    @property
    def K(self):
        K = np.eye(3)
        K[0,0] = self.fx
        K[1,1] = self.fy
        K[0,1] = self.skew
        K[0,2] = self.cx
        K[1,2] = self.cy
        return K
    
    @property
    def inv_K(self):
        return np.linalg.inv(self.K)
    
    def get_rays(self, normalize=False):
        uv = self.make_pixel_grid() # (HW,2)
        x = (uv[:,0:1] - self.cx) / self.fx
        y = (uv[:,1:2] - self.cy) / self.fy
        z = ones_like(x)
        rays = concat([x,y,z], -1)
        if normalize: rays = normalize(rays, -1)
        return rays # (HW,3)
    
    
if __name__ == '__main__':
    
    cam_dict = {
        'cam_type': 'PINHOLE',
        'width': 640,
        'height': 480,
        'fx': 160.,
        'fy': 240.,
        'cx': 320.,
        'cy': 240.,
        'skew': 1.4
    }    
    cam = Camera.create_cam(cam_dict)
    print(cam.K)
    
    print(cam.inv_K)
    rays =cam.get_rays(False) 
    print(rays.shape)
    
    