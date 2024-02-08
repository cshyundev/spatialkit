import numpy as np

from src.hybrid_math import Array
from src.hybrid_operations import Array
from .hybrid_operations import *
from .hybrid_math import *
from enum import Enum
from typing import *
import cv2 as cv


class CamType(Enum):
    PINHOLE = ("PINHOLE", "Pinhole Camera Type")
    EQUIDISTANT = ("EQUIDISTANT", "Equidistant Camera Type")
    EQUIRECT = ("EQUIRECT", "Equirectangular Camera Type")
    NONE = ("NONE", "No Camera Type")
    
    @staticmethod
    def from_string(type_str: str)-> 'CamType':
        if type_str == 'PINHOLE':
            return CamType.PINHOLE
        elif type_str == 'OPENCV_FISHEYE':
            return CamType.EQUIDISTANT
        elif type_str == 'EQUIRECT':
            return CamType.EQUIRECT
        else:
            return CamType.NONE
        
class Camera:
    def __init__(self, cam_dict: Dict[str, Any]):
        self.cam_type = CamType.NONE
        self.width,self.height = cam_dict['image_size']
    
    def make_pixel_grid(self) -> np.ndarray:
        u,v = np.meshgrid(range(self.width), range(self.height))
        uv = concat([u.reshape((1,-1)), v.reshape((1,-1))], 0)
        return uv 
    
    def get_rays(self) -> Array:
        raise NotImplementedError
    
    def project_rays(self, rays:Array) -> Array:
        raise NotImplementedError
    
    def depth_scale(self) -> Array:
        raise NotImplementedError
    
    def get_radii(self) -> Array:
        raise NotImplementedError
    
    @staticmethod
    def create_cam(cam_dict: Dict[str, Any]) -> 'Camera':
        cam_type = CamType.from_string(cam_dict['cam_type']) 
        
        if cam_type is CamType.PINHOLE:
            return PinholeCamera(cam_dict)
        elif cam_type is CamType.EQUIDISTANT:
            raise NotImplementedError
        elif cam_type is CamType.EQUIRECT:
            return EquirectangularCamera(cam_dict)
        else:
            raise Exception("Unkwon Camera Type")

class PinholeCamera(Camera):
    """
    Pinhole Camera Model without distortion Parameters.

    Attributes:
        cam_type (CamType): Camera type, set to PINHOLE.
        fx, fy (float): Focal length in x and y directions.
        cx, cy (float): Principal point coordinates (center of the image).
        skew (float): Skew coefficient between x and y axis.
        radial_params (List[float]): Radial distortion parameters [k1, k2, k3].
        tangential_params (List[float]): Tangential distortion parameters [p1, p2].
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super(PinholeCamera, self).__init__(cam_dict)
        self.cam_type = CamType.PINHOLE
        self.fx, self.fy = cam_dict['focal_length']
        self.cx, self.cy = cam_dict['principal_point']
        self.skew = cam_dict['skew']
        self.radial_params = cam_dict['radial']
        self.tangential_params = cam_dict['tangential']

    @staticmethod
    def from_K(K: List[float], image_size:List[int], dist: List[float] = None) -> 'PinholeCamera':
        """
        Static method to create a PinholeCamera instance from intrinsic matrix and distortion coefficients.

        Args:
            K (List[float]): Intrinsic matrix parameters as a list [fx, skew, cx, fy, cy].
            dist (List[float]): Distortion coefficients as a list [k1, k2, p1, p2, k3].

        Returns:
            PinholeCamera: An instance of PinholeCamera with given parameters.
        """

        cam_dict = {
            'image_size': image_size,
            'focal_length': (K[0][0], K[1][1]), # fx and fy
            'principal_point': (K[0][2], K[1][2]), # cx and cy
            'skew': K[0][1],
        }
        
        if dist is not None:
            cam_dict["radial"] = (dist[0],dist[1],dist[4]) # k1, k2, k3
            cam_dict["tangential"] = (dist[2],dist[3]) # p1, p2
        else:
            cam_dict["radial"] = [0.,0.,0.]
            cam_dict["tangential"] = [0.,0.]

        return PinholeCamera(cam_dict)
    
    @staticmethod
    def from_fov(image_size:List[int], fov:List[float]):
        """
        Static method to create a PinholeCamera instance from image resolution and field of view.

        Args:
            image_size (List[int]): Image resolution as a list [width, height].
            fov(List[float]): Field of view in degrees as a list [fov_x, fov_y].

        Returns:
            PinholeCamera: An instance of PinholeCamera with calculated parameters.
        """
        width,height = image_size
        fov_x, fov_y = fov
        # Calculate focal length based on FOV and image resolution
        fx = width / (2 * tan(deg2rad(fov_x) / 2))
        fy = height / (2 * tan(deg2rad(fov_y) / 2))
        # Assuming principal point is at the center of the image
        cx = width / 2.
        cy = height / 2.
        # Assuming no skew and no distortion
        skew = 0.
        radial_params = [0, 0, 0]  # No radial distortion
        tangential_params = [0, 0]  # No tangential distortion

        cam_dict = {
            'image_size': image_size,
            'focal_length': (fx, fy),
            'principal_point': (cx, cy),
            'skew': skew,
            'radial': radial_params,
            'tangential': tangential_params
        }
        return PinholeCamera(cam_dict)

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
        return np.linalg.inv(self.K())
    
    @property
    def fov(self):
        fov_x = 2 * rad2deg(arctan(self.width / (2 * self.fx))) 
        fov_y = 2 * rad2deg(arctan(self.height / (2 * self.fy))) 
        return fov_x, fov_y

    @property
    def dist_coeffs(self):
        _dist_coeffs = self.radial_params[0:2] + self.tangential_params + self.radial_params[2:3]
        return np.array(_dist_coeffs)

    def _is_distorted(self):
        cnt = np.count_nonzero(self.radial_params) + np.count_nonzero(self.tangential_params)
        return cnt != 0

    def _distort(self, x: Array, y: Array, extra: bool=False) \
            -> Union[Tuple[Array,Array],Tuple[Array,Array,Array,Array]] :
        x2, y2 = x**2, y**2
        r2 = x2 + y2
        xy = x*y
        k1,k2,k3 = self.radial_params[0], self.radial_params[1], self.radial_params[2] 
        radial_distortion = 1. + r2 * (k1 + r2 * (k2 + k3 *r2))
        p1, p2 = self.tangential_params[0], self.tangential_params[1]
        x_tangential_distortion = 2*p1*xy + p2*(r2 + 2*x2)
        y_tangential_distortion = p1*(r2 + 2*y2) + 2*p2*xy
        x_dist = x * radial_distortion + x_tangential_distortion
        y_dist = y * radial_distortion + y_tangential_distortion
        
        if extra:
            # compute derivative of radial distortion, x, y
            radial_distortion_r = k1 + r2 * (2. * k2 + 3. * k3 * r2)
            radial_distortion_x = 2. * x * radial_distortion_r
            radial_distortion_y = 2. * y * radial_distortion_r
            return x_dist, y_dist, radial_distortion, \
                radial_distortion_x, radial_distortion_y
        return x_dist, y_dist
        
    def _compute_residual_jacobian(self,xu:Array,yu:Array, xd:Array, yd:Array) \
            -> Tuple[Array,Array,Array,Array,Array,Array]:
        
        """
        Computes the Jacobian matrix of the residual between undistorted (ideal) points and distorted (actual) points
        caused by camera lens distortion. Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py    
        Parameters:
        - xu, yu: Arrays of x and y coordinates of the undistorted points.
        - xd, yd: Arrays of x and y coordinates of the distorted points.
    
        Returns:
        - Tuple containing six arrays:
        - x_res, y_res: The residuals in x and y directions.
        - x_res_x, x_res_y: Partial derivatives of the x residual with respect to x and y.
        - y_res_x, y_res_y: Partial derivatives of the y residual with respect to x and y.

        The derivatives are calculated based on the distortion model which includes radial and tangential components.
        """
        # Compute distorted coordinates (_xd, _yd) and the distortion terms
        # (dist, dist_x, dist_y) based on the undistorted coordinates (xu, yu)
        _xd,_yd, dist, dist_x, dist_y = self._distort(xu,yu,True)
        x_res,y_res = _xd - xd, _yd - yd
        
        p1, p2 = self.tangential_params[0], self.tangential_params[1]
        # Compute derivative of x_dist over x and y 
        x_res_x = dist + dist_x * xu + 2. * p1 * yu + 6. * p2 * xu
        x_res_y = dist_y * xu + 2. * p1 * xu + 2. * p2 * yu
        # Compute derivative of y_dist over x and y        
        y_res_x = dist_x * yu + 2.0 * p2 * yu + 2.0 * p1 * xu
        y_res_y = dist + dist_y * yu + 2.0 * p2 * xu + 6.0 * p1 * yu
        
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
            if sqrt(rx**2 + ry**2).max() < 1e-4: break
            Det = ry_x * rx_y - rx_x * ry_y
            x_numerator,y_numerator = ry_y*rx-rx_y*ry, rx_x*ry-ry_x*rx
            step_x = where(abs(Det) > err_thr, x_numerator / Det, zeros_like(Det))
            step_y = where(abs(Det) > err_thr, y_numerator / Det, zeros_like(Det))
            xu = xu + clip(step_x,-0.5, 0.5)
            yu = yu + clip(step_y,-0.5, 0.5)

        return xu,yu

    def _to_image_plane(self, x:Array, y:Array, use_clip:bool=False) -> Tuple[Array,Array]:
        u = self.fx * x + self.skew * y + self.cx
        v = self.fy * y + self.cy
        if use_clip:
            u = clip(u,0.,float(self.width))
            v = clip(v,0.,float(self.height))
        return u,v
    
    def _to_normalized_plane(self, uv:Array) -> Tuple[Array,Array]:
        x = (uv[0:1,:] - self.cx - self.skew / self.fy*(uv[1:2,:] -self.cy)) / self.fx # (1,HW)
        y = (uv[1:2,:] - self.cy) / self.fy #(1,HW)
        return x,y

    def get_rays(self,
                 uv:Array = None, 
                 norm:bool = True,
                 out_scale:bool = False,
                 err_thr:float = 1e-4,
                 max_iter:int = 5
                 ) -> Array:
        if uv is None: uv = self.make_pixel_grid() # (2,HW)
        x,y = self._to_normalized_plane(uv) 
        if self._is_distorted():
            x,y = self._undistort(x,y,err_thr, max_iter)
        z = ones_like(x)
        rays = concat([x,y,z], 0) # (3,HW)
        if norm: rays = normalize(rays, 0)
        if out_scale:
            depth_scale = rays[2:3,:]
            return rays, depth_scale
        return rays # (3,HW)
    
    def project_rays(self, rays: Array) -> Array:
        X = rays[0:1,:]
        Y = rays[1:2,:]
        Z = rays[2:3,:]
        x,y = X / Z, Y / Z
        if self._is_distorted(): x,y = self._distort(x,y)
        u,v = self._to_image_plane(u,v)
        return as_int(concat([u,v], dim=0),n=32)   # (2,HW)

    def reprojection_mask(self, rays: Array, uv: Array, reprojection_err_thr: int = 1) -> Array:
        """
        Generate a mask based on the reprojection error of rays projected onto the image plane.
        
        Args:
        - rays: (3,N), float, The rays to project, assumed to be normalized.
        - uv: (2,N), int, The actual image coordinates corresponding to the rays.
        - reprojection_err_thr: scalar, int, The threshold for reprojection error to decide mask values.

        Returns:
        - Array, (1,N): A mask array where True represents a reprojection error below the threshold, and False otherwise.
        """
        # Project rays onto the image plane to get the reprojected coordinates
        reprojected_uv = self.project_rays(rays)
        # Calculate the reprojection error between actual and reprojected coordinates
        reprojection_err = normalize(reprojected_uv - uv, dim=0)
        # Generate mask based on the reprojection error threshold
        return reprojection_err <= reprojection_err_thr

    def get_radii(self, uv:np.ndarray=None, err_thr:float = 1e-4, max_iter:int = 5):
        if uv is None: uv = self.make_pixel_grid()
        u,v = uv[0:1,:],uv[1:2,:] # (1,N)
        num_pixels = u.shape[1]
        uu = concat([u, u+1], dim=1) # (1,2N)
        vv = concat([v, v+1], dim=1) # (1,2N)
        xx = (uu - self.cx - self.skew / self.fy*(vv -self.cy)) / self.fx # (1,2N)
        yy = (vv - self.cy) / self.fy # (1,2N)
        if self._is_distorted():
            xx,yy = self._undistort(xx,yy,err_thr, max_iter)
        # dx = x[:,:-1] - x[:,1:]
        # dx = concat([dx, dx[:,-2:-1]], dim=1)
        # dy = y[:,-1:] - y[:,1:]
        # dy = concat([dy, dy[:,-2:-1]],dim=1)
        dx = xx[:,num_pixels:] - xx[:,:num_pixels] 
        dy = yy[:,num_pixels:] - yy[:,:num_pixels] 
        radii = sqrt(dx**2 + dy**2) * 2 / np.sqrt(12) # (1,HW)
        return radii.reshape(-1,1) # (HW,1)

    def distort_pixel(self, uv:Array, use_clip:bool=False) -> Array:
        if self._is_distorted() is False: return uv
        
        x,y = self._to_normalized_plane(uv)
        xd,yd = self._distort(x,y)
        u,v = self._to_image_plane(xd,yd,use_clip)
        return as_int(concat([u,v], dim=0),n=32)   # (2,HW)

    def undistort_pixel(self, uv:Array, use_clip:bool=False, err_thr:float=1e-4, max_iter:int=5) -> Array:
        if self._is_distorted() is False: return uv
        x,y = self._to_normalized_plane(uv)        
        xu,yu = self._undistort(x,y,err_thr,max_iter)
        u,v = self._to_image_plane(xu,yu,use_clip)
        return as_int(concat([u,v], dim=0),n=32)   # (2,HW)

# TODO
class EquidistantCamera(Camera):
    def __init__(self, cam_dict: Dict[str, Any]):
        super(EquidistantCamera, self).__init__(cam_dict)
        
        self.cam_type = CamType.EQUIDISTANT
        
        self.fx, self.fy = cam_dict['focal_length']
        self.cx, self.cy = cam_dict['principal_point']
        self.skew = cam_dict['skew']
        self.radial_params = cam_dict['radial'] # k1,k2,k3,k4,k5
        self.tangential_params = cam_dict['tangential'] # p1,p2
      
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
        return np.linalg.inv(self.K())
    
    def get_rays(uv:np.ndarray = None, out_scale:bool=False):
        raise NotImplementedError
    
    def project_rays(self, rays: ndarray) -> ndarray:
        return super().project_rays(rays)

class EquirectangularCamera(Camera):
    """
    Equirectangular Camera Model.
    
    This camera model is used for representing 360-degree panoramic images
    using equirectangular projection. It maps spherical coordinates to a 2D
    equirectangular plane.

    Attributes:
        cam_type (CamType): Camera type, set to EQUIRECT (Equirectangular).
        min_phi_deg (float): Minimum vertical field of view angle (phi) in degrees.
        max_phi_deg (float): Maximum vertical field of view angle (phi) in degrees.
        cx (float): The x-coordinate of the central point of the equirectangular image.
        cy (float): The y-coordinate of the central point of the equirectangular image.
    """
    def __init__(self, cam_dict: Dict[str, Any]):
        """
        Initializes the EquirectangularCamera with camera parameters.

        Args:
            cam_dict (Dict[str, Any]): A dictionary containing camera parameters.
                Expected to have 'min_phi_deg' and 'max_phi_deg' as keys with corresponding values.
        """
        super(EquirectangularCamera, self).__init__(cam_dict)
        self.cam_type = CamType.EQUIRECT
        
        self.min_phi_deg:float = cam_dict["min_phi_deg"]
        self.max_phi_deg:float = cam_dict["max_phi_deg"]
        self.cx = self.width / 2.0
        self.cy = (self.height - 1.) / 2.0

        self.phi_scale = deg2rad((self.max_phi_deg - self.min_phi_deg)*0.5)
        self.phi_offset = deg2rad((self.max_phi_deg + self.min_phi_deg)*0.5)

    def get_rays(self, uv:np.ndarray = None, out_scale:bool=False):
        if uv is None: uv = self.make_pixel_grid()
        theta = (uv[0:1,:]-self.cx) / self.cx * np.pi
        phi = (uv[1:2,:] - self.cy) / self.cy * self.phi_scale + self.phi_offset
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
    
    def project_rays(self, rays:Array) -> Array:
        """
        Projects a 3D ray back to the equirectangular image plane.

        Args:
            ray (Array Type): A 3D vector array representing the ray directions.
        Returns:
            Array Type: The x and y coordinates on the equirectangular image for each ray.
        """
        # Normalize the ray vector
        norm = normalize(rays, dim=0)
        rays = rays / norm

        # Convert Cartesian coordinates to spherical coordinates
        phi = arctan2(rays[1], sqrt(rays[0]**2 + rays[2]**2))
        theta = arctan2(rays[0], rays[2])
        # Convert spherical coordinates to pixel coordinates
        x = (rad2deg(theta) + 180.0) * self.cx / np.pi
        y = ((rad2deg(phi) - rad2deg(self.phi_offset)) / rad2deg(self.phi_scale)) + self.cy
        return as_int(concat([x, y], dim=0),n=32) 

    def get_radii(self, uv:np.ndarray=None) -> Tensor:
        if uv is None: uv = self.make_pixel_grid()
        dt_theta = np.pi / self.width
        phi = (uv[1:2,:] -self.cy) / self.cy * self.phi_scale - self.phi_offset
        radii = 2 * sin(dt_theta) * cos(phi)
        return radii.reshape(-1,1) * 2 / sqrt(12)
    