import numpy as np
from .hybrid_operations import *
from .hybrid_math import *
from enum import Enum
from typing import *

class CamType(Enum):
    PINHOLE = ("PINHOLE", "Pinhole Camera Type")
    OPENCV_FISHEYE = ("OPENCV_FISHEYE", "OpenCV Fisheye Camera with Distortion Parameters")
    EQUIRECT = ("EQUIRECT", "Equirectangular Camera Type")
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
        self.height, self.width = cam_dict['image_size']
    
    def make_pixel_grid(self) -> np.ndarray:
        u,v = np.meshgrid(range(self.width), range(self.height))
        uv = concat([u.reshape((1,-1)), v.reshape((1,-1))], 0)
        return uv 
    
    def get_rays(self) -> Array:
        raise NotImplementedError
    
    def project(self, rays:Array) -> Array:
        raise NotImplementedError
    
    def depth_scale(self) -> Array:
        return 1.
    
    def get_radii(self) -> Array:
        raise NotImplementedError
    
    @staticmethod
    def create_cam(cam_dict: Dict[str, Any]) -> 'Camera':
        cam_type = CamType.from_string(cam_dict['cam_type']) 
        
        if cam_type is CamType.PINHOLE:
            return PinholeCamera(cam_dict)
        elif cam_type is CamType.OPENCV_FISHEYE:
            raise NotImplementedError
        elif cam_type is CamType.EQUIRECT:
            return EquirectangularCamera(cam_dict)
        else:
            raise Exception("Unkwon Camera Type")

class PinholeCamera(Camera):
    """
    Pinhole Camera Model without distortion Parameters.
    """
    def __init__(self, cam_dict: Dict[str, Any]):
        super(PinholeCamera, self).__init__(cam_dict)
        self.cam_type = CamType.PINHOLE
        self.fx,self.fy = cam_dict['focal_length']
        self.cx,self.cy = cam_dict['principal_point'] 
        self.skew = cam_dict['skew']
        self.radial_params = cam_dict['radial'] # radial distortion parameters [k1,k2,k3]
        self.tangential_params = cam_dict['tangential'] # tangential distortion parameters, [p1,p2]
    
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

    def _distort(self, x: Array, y: Array, extra: bool=False) \
            -> Union[Tuple[Array,Array],Tuple[Array,Array,Array,Array]] :
        x2, y2 = x**2, y**2
        r = x2 + y2
        xy = x*y
        k1,k2,k3 = self.radial_params[0], self.radial_params[1], self.radial_params[2] 
        radial_distortion = 1. + r * (k1 + r * (k2 + k3 *r))
        p1, p2 = self.tangential_params[0], self.tangential_params[1]
        x_tangential_distortion = 2*p1*xy + p2*(r + 2*x2)
        y_tangential_distortion = p1*(r + 2*y2) + 2*p2*xy
        x_dist = x * radial_distortion + x_tangential_distortion
        y_dist = y * radial_distortion + y_tangential_distortion
        
        if extra:
            # compute derivative of radial distortion, x, y
            radial_distortion_r = k1 + r * (2. * k2 + 3. * k3 * r)
            radial_distortion_x = 2. * x * radial_distortion_r
            radial_distortion_y = 2. * y * radial_distortion_r
            return x_dist, y_dist, radial_distortion, \
                radial_distortion_x, radial_distortion_y
        return x_dist, y_dist
        
    def _compute_residual_jacobian(self,x:Array,y:Array, xd:Array, yd:Array) \
            -> Tuple[Array,Array,Array,Array,Array,Array]:
        # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
        x_dist,y_dist, dist, dist_x, dist_y = self._distort(x,y,True)
        x_res,y_res = x_dist - xd, y_dist - yd
        
        p1, p2 = self.tangential_params[0], self.tangential_params[1]
        # compute derivative of x_dist over x and y 
        x_res_x = dist + dist_x * x + 2. * p1 * y + 6. * p2 * x
        x_res_y = dist_y * x + 2. * p1 * x + 2. * p2 * y
        # compute derivative of y_dist over x and y        
        y_res_x = dist_x * y + 2.0 * p2 * y + 2.0 * p1 * x
        y_res_y = dist + dist_y * y + 2.0 * p2 * x + 6.0 * p1 * y
        
        return x_res, y_res, x_res_x, x_res_y, y_res_x, y_res_y
        
    def _undistort(self, xd: Array, yd: Array, err_thr:float, max_iter:int) -> Tuple[Array,Array]:
        """
        Compute undistorted coords
        Adapted from MultiNeRF
        https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509
            
        Args:
            xd: (1,N), float, The distorted coordinates x.
            yd: (1,N), float, The distorted coordinates y.
            err_thr: scalar, float, error threshold.
            max_iter: scalar, int, maximum iteration times.        
        Returns:
            xu: (1,N), float, The undistorted coordinates x.
            yu: (1,N), float, The undistorted coordinates y.
            
        We want to undistortion point like distortion == distort(undistortion)
        x,y := undistorted coords. x,y
        r(x,y):= (rx,ry) = distort(x,y) - (xd,yd), residual of undistorted coords.
        J(x,y) := |rx_x rx_y|  , Jacobian of residual
                  |ry_x ry_y|
        J^-1 = 1/D | ry_y -rx_y|
                   |-ry_x  rx_x| 
        D := rx_x * ry_y - rx_y * ry_x, Determinant of Jacobian
        
        Initialization:
            x,y := xd,yd
        Using Newton's Method:
            Iteratively
             x,y <- x,y - J ^-1 * (rx ry)^T  = (x,y) - 1/D * (ry_y*rx-rx_y*ry,rx_x*ry-ry_x*rx)   
        """
        xu,yu = deep_copy(xd), deep_copy(yd)
        
        for _ in range(max_iter):
            rx, ry, rx_x, rx_y, ry_x, ry_y \
                = self._compute_residual_jacobian(xu, yu, xd, yd)
            Det = rx_x * ry_y - rx_y * ry_x
            x_numerator,y_numerator = ry_y*rx-rx_y*ry, rx_x*ry-ry_x*rx
            step_x = where(abs(Det) > err_thr, x_numerator / Det, zeros_like(Det))
            step_y = where(abs(Det) > err_thr, y_numerator / Det, zeros_like(Det))
            xu = xu - step_x
            yu = yu - step_y
        return xu,yu
            
    def get_rays(self,
                 uv:np.ndarray = None, 
                 norm:bool = True,
                 out_scale:bool = False,
                 err_thr:float = 1e-6,
                 max_iter:int = 5
                 ) -> np.ndarray:
        if uv is None: uv = self.make_pixel_grid() # (2,HW)
        x = (uv[0:1,:] - self.cx - self.skew / self.fy*(uv[1:2,:] -self.cy)) / self.fx # (1,HW)
        y = (uv[1:2,:] - self.cy) / self.fy #(1,HW)
        if self.radial_params[0] != 0.:
            x,y = self._undistort(x,y,err_thr, max_iter)
        z = ones_like(x)
        rays = concat([x,y,z], 0) # (3,HW)
        if norm: rays = normalize(rays, 0)
        if out_scale:
            depth_scale = rays[2:3,:]
            return rays, depth_scale
        return rays # (3,HW)
    
    def project(self, rays: Array) -> Array:
        X = rays[0:1,:]
        Y = rays[1:2,:]
        Z = rays[2:3,:]
        x,y = X / Z, Y / Z
        if self.radial_params[0] != 0.:
            x,y = self._distort(x,y)
        u = self.fx * x + self.skew * y + self.cx 
        v = self.fy * y + self.cy
        return concat([u,v], dim=0) # (2,HW)

    def get_radii(self, err_thr:float = 1e-6, max_iter:int = 5):
        uv = self.make_pixel_grid()
        x = (uv[0:1,:] - self.cx - self.skew / self.fy*(uv[1:2,:] -self.cy)) / self.fx # (1,HW)
        y = (uv[1:2,:] - self.cy) / self.fy # (1,HW)
        if self.radial_params[0] != 0.:
            x,y = self._undistort(x,y,err_thr, max_iter)
        dx = x[:,:-1] - x[:,1:]
        dx = concat([dx, dx[:,-2:-1]], dim=1)
        dy = y[:,-1:] - y[:,1:]
        dy = concat([dy, dy[:,-2:-1]],dim=1)
        radii = sqrt(dx**2 + dy**2) * 2 / np.sqrt(12) # (1,HW)
        return radii.reshape(-1,1) # (HW,1)

# TODO
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

class EquirectangularCamera(Camera):
    def __init__(self, cam_dict: Dict[str, Any]):
        super(EquirectangularCamera, self).__init__(cam_dict)
        self.cam_type = CamType.EQUIRECT
        
        self.min_phi_deg:float = cam_dict["min_phi_deg"]
        self.max_phi_deg:float = cam_dict["max_phi_deg"]
        self.cx = self.width / 2.0
        self.cy = (self.height - 1.) / 2.0
        
    def get_rays(self, uv:np.ndarray = None, out_scale:bool=False):
        if uv is None: uv = self.make_pixel_grid()
        theta = (uv[0:1,:]-self.cx) / self.cx * np.pi
        phi_scale = np.deg2rad((self.max_phi_deg - self.min_phi_deg)*0.5)
        phi_offset = np.deg2rad((self.max_phi_deg + self.min_phi_deg)*0.5)
        phi = (uv[1:2,:] - self.cy) / self.cy * phi_scale + phi_offset
        x = sin(theta) * cos(phi)
        y = sin(phi)
        z = cos(theta) * cos(phi)
        ray = concat([x,y,z],0)
        invalid_ray = np.logical_or(phi < self.min_phi_deg, phi > self.max_phi_deg).reshape(-1,)
        ray[:,invalid_ray] = np.nan
        if out_scale:
            depth_scale = ones_like(z)
            depth_scale[:,invalid_ray] = np.nan
            return ray, depth_scale
        return ray # (3,HW)
    
    def project(self, rays:Array) -> Array:
        return None

    def get_radii(self) -> Tensor:
        uv = self.make_pixel_grid()
        dt_theta = np.pi / self.width
        phi_scale = np.deg2rad((self.max_phi_deg - self.min_phi_deg)*0.5)
        phi_offset = np.deg2rad((self.max_phi_deg + self.min_phi_deg)*0.5)
        phi = (uv[1:2,:] -self.cy) / self.cy * phi_scale - phi_offset
        radii = 2 * sin(dt_theta) * cos(phi)
        return radii.reshape(-1,1) * 2 / np.sqrt(12)


if __name__ == '__main__':
    
    cam_dict = {
    "cam_type": "PINHOLE",
    "image_size": [
     384,
     384
    ],
    "focal_length": [
     600.0,
     599.9999389648438
    ],
    "principal_point": [
     599.5,
     339.4999694824219
    ],
    "skew": 1.7359692719765007e-05,
    "radial": [
     0.0,
     0.0,
     0.0
    ],
    "tangential": [
     0.0,
     0.0
    ]
   }
    cam = Camera.create_cam(cam_dict)
    pt = np.array([[233.,253.]])
    rays =cam.get_rays(pt)
    uv = cam.project(rays)
    print(uv)
    
    