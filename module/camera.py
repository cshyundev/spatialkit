import numpy as np
from module.hybrid_operations import *
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

        self.radial_params = cam_dict['k'] # radial distortion parameters 
        self.tangential_params = cam_dict['p'] # tangential distortion parameters, [p1,p2]
    
    def K(self):
        K = np.eye(3)
        K[0,0] = self.fx
        K[1,1] = self.fy
        K[0,1] = self.skew
        K[0,2] = self.cx
        K[1,2] = self.cy
        return K
    
    def inv_K(self):
        return np.linalg.inv(self.K())
    
    def __distort(self, x: Array, y: Array) -> Tuple[Array,Array]:
        x2, y2 = x**2, y**2
        r2 = x2 + y2
        xy = x*y
        radial_distortion = 1.
        r_poly = r2
        for k in self.radial_params:
            radial_distortion += r_poly * k
            r_poly = r_poly * r2
        p1, p2 = self.tangential_params[0], self.tangential_params[1]
        x_tangential_distortion = 2*p1*xy + p2*(r2 + 2*x2)
        y_tangential_distortion = p1*(r2 + 2*y2) + 2*p2*xy
        x_dist = x * radial_distortion + x_tangential_distortion
        y_dist = y * radial_distortion + y_tangential_distortion
        return x_dist, y_dist

    def get_rays(self,
                 uv: np.ndarray=None, 
                 norm: bool=False,
                 err_thr: float = 1e-6,
                 max_iter:int = 10
                 ) -> np.ndarray:
        if uv is None: uv = self.make_pixel_grid() # (HW,2)
        x = (uv[:,0:1] - self.cx - self.skew / self.fy*(uv[:,1:2] -self.cy)) / self.fx
        y = (uv[:,1:2] - self.cy) / self.fy
        if len(self.radial_params) > 0:
            for _ in range(max_iter):
                x_d, y_d = self.__distort(x,y)
                err_x, err_y = x_d - x, y_d - y
                x, y = x - err_x, y - err_y
                err = sqrt(err_x**2 + err_y**2).max()
                if err < err_thr: break
        z = ones_like(x)
        rays = concat([x,y,z], -1)
        if norm: rays = normalize(rays, -1)
        return rays # (HW,3)

class OpenCVFIsheyeCamera(Camera):
    def __init__(self, cam_dict: Dict[str, Any]):
        super(OpenCVFIsheyeCamera, self).__init__(cam_dict)
        
        self.cam_type = CamType.OPENCV_FISHEYE
        
        self.fx = cam_dict['fx']
        self.fy = cam_dict['fy']        
        self.cx = cam_dict['cx'] 
        self.cy = cam_dict['cy']
        self.skew = cam_dict['skew']
         
        self.k = cam_dict['k'] # radial distortion parameters 
        self.p = cam_dict['p'] # tangential distortion parameters

    def K(self):
        K = np.eye(3)
        K[0,0] = self.fx
        K[1,1] = self.fy
        K[0,1] = self.skew
        K[0,2] = self.cx
        K[1,2] = self.cy
        return K
    
    def inv_K(self):
        return np.linalg.inv(self.K())

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
    rays =cam.get_rays() 
    print(rays.shape)
    
    