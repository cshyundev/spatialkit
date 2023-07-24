import numpy as np
from module.hybrid_operations import *
from enum import Enum
from typing import *

from utils.cv_utils.module.hybrid_operations import Array


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
    
    def project(self, rays:Array) -> Array:
        raise NotImplementedError
    
    def depth_scale(self) -> Array:
        return 1.
    
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
    """
    def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
) -> torch.Tensor:
    Computes undistorted coords given opencv distortion parameters.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=coords[..., 0], yd=coords[..., 1], distortion_params=distortion_params
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(torch.abs(denominator) > eps, x_numerator / denominator, torch.zeros_like(denominator))
        step_y = torch.where(torch.abs(denominator) > eps, y_numerator / denominator, torch.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)
    
    def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).


    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y
    """
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
                 uv:np.ndarray = None, 
                 norm:bool = False,
                 err_thr:float = 1e-6,
                 max_iter:int = 5
                 ) -> np.ndarray:
        if uv is None: uv = self.make_pixel_grid() # (HW,2)
        x = (uv[:,0:1] - self.cx - self.skew / self.fy*(uv[:,1:2] -self.cy)) / self.fx
        y = (uv[:,1:2] - self.cy) / self.fy
        # if len(self.radial_params) > 0:
        #     for _ in range(max_iter):
        #         x_d,y_d = self.__distort(x,y)
        #         err_x = x_d - x
        #         err_y = y_d - y
        #         x = x - err_x
        z = ones_like(x)
        rays = concat([x,y,z], -1)
        if norm: rays = normalize(rays, -1)
        return rays # (HW,3)
    
    def depth_scale(self) -> np.ndarray:
        return super().depth_scale()

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
    print(cam.K)
    
    print(cam.inv_K)
    rays =cam.get_rays() 
    print(rays.shape)
    
    