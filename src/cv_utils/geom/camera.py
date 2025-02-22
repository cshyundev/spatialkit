"""
Module Name: camera.py

Description:
The Camera module provides implementations of various camera models. 
It includes the Camera abstract class, which serves as a base class for specific camera types, 
and defines essential methods like projection and unprojection.

Camera Types:
- PERSPECTIVE: Perspective Camera Type
- OPENCV: OpenCV Fisheye Camera Type
- THINPRISM: Thin Prism Fisheye Camera Type
- OMNIDIRECT: Omnidirectional FIsheye Camera Type
- DOUBLESPHERE: Double Sphere Camera Type
- EQUIRECT: Equirectangular Camera Type
- NONE: No Camera Type

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT LICENSE
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import cv2 as cv

from ..ops.uops import *
from ..ops.umath import *

from ..common.constant import PI, NORM_PIXEL_THRESHOLD, EPSILON
from ..common.logger import LOG_CRITICAL, LOG_ERROR, LOG_WARN


class CamType(Enum):
    """
    Placeholder docstring for a function or method.

    This docstring should be updated with a detailed description of the
    function or method's purpose, parameters, return values, and any
    exceptions raised.
    """

    PERSPECTIVE = ("PERSPECTIVE", "Perspective Camera Type")
    OPENCVFISHEYE = ("OPENCVFISHEYE", "Open Fisheye Camera Type")
    THINPRISM = ("THINPRISIM", "Thin Prism Fisheye Camera Type")
    OMNIDIRECT = ("OMNIDIRECT", "Omnidirectional Camera Type")
    DOUBLESPHERE = ("DOUBLESPHERE", "Double Sphere Camera Type")
    EQUIRECT = ("EQUIRECT", "Equirectangular Camera Type")
    NONE = ("NONE", "No Camera Type")

    @staticmethod
    def from_string(type_str: str) -> "CamType":
        if type_str == "PERSPECTIVE":
            return CamType.PERSPECTIVE
        elif type_str == "OPENCV_FISHEYE":
            return CamType.OPENCVFISHEYE
        elif type_str == "EQUIRECT":
            return CamType.EQUIRECT
        elif type_str == "THINPRISIM":
            return CamType.THINPRISM
        elif type_str == "OMNIDIRECT":
            return CamType.OMNIDIRECT
        elif type_str == "DOUBLESPHERE":
            return CamType.DOUBLESPHERE
        else:
            return CamType.NONE


# Camera Abstract Class
class Camera:
    """
    Abstract base class for different camera models. This class provides basic functionality
    and outlines essential methods to be implemented by subclasses.

    Attributes:
        cam_type (CamType): Type of the camera. Defaults to CamType.NONE.
        width (int): Width of the camera image.
        height (int): Height of the camera image.
        _mask (np.ndarray): Mask of the camera image initialized to True for all pixels.
        _max_fov_deg: Maximum Field of View (FOV) in degrees

    Abstract Methods:
        convert_to_rays: Abstract method to convert pixel coordinates into camera rays.
        convert_to_pixels: Abstract method to project camera rays into pixel coordinates.
        export_cam_dict: A Method to export camera parameters as dict.
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        self.cam_type = CamType.NONE
        self.width, self.height = cam_dict["image_size"]
        self._mask = np.full((self.height, self.width), True, dtype=bool).reshape(
            -1,
        )
        self._max_fov_deg = cam_dict.get("fov_deg", -1.0)

    def make_pixel_grid(self) -> np.ndarray:
        """
        Creates a grid of pixel coordinates.

        Return:
            pixel_grid (np.ndarray, [2, height * width]): ArrayLike containing pixel coordinates.

        Details:
        - uv[:,i] means i-th (u,v) in 2D image pixel plane(i.e. i = v * width + u).
        """
        u, v = np.meshgrid(range(self.width), range(self.height))
        uv = concat([u.reshape((1, -1)), v.reshape((1, -1))], 0).astype(np.float32)
        return uv

    def set_mask(self, mask: ArrayLike):
        if mask.shape != self.hw:
            LOG_ERROR(
                f"Mask's shape must be same as image size{self.hw}, but Mask's shape = {mask.shape}."
            )
        mask = convert_numpy(mask).reshape(
            -1,
        )
        self._mask = logical_and(self._mask, mask)

    def convert_to_rays(
        self, uv: ArrayLike, z_fixed: Optional[bool] = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Converts pixel coordinates into unit vector camera rays.

        Args:
            uv (ArrayLike, [2,N]): 2D image pixel coordinates.
            z_fixed (bool): If True, the rays have their Z-component fixed at 1.

        Returns:
            rays (ArrayLike, [3, N]): unit vector camera rays.
            mask (ArrayLike, [N,]): Boolean mask indicating which rays are valid.

        Details:
        - This method must be implemented by subclasses.
        - Converts pixel coordinates (uv) to rays and checks their validity.
        """
        raise NotImplementedError

    def convert_to_pixels(
        self, rays: ArrayLike, out_subpixel: Optional[bool] = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Projects camera rays into pixel coordinates.

        Args:
            rays (ArrayLike, [3, N]): ArrayLike containing camera rays.
            out_subpixel (bool, optional): If True, returns subpixel coordinates as floats. Defaults to False.

        Returns:
            pixels (ArrayLike, [2, N]): ArrayLike containing pixel coordinates.
            mask (ArrayLike, [N,]): Boolean mask indicating which pixels are valid.

        Details
            - This method must be implemented by subclasses.
            - Converts camera rays to pixel coordinates and checks their validity.
            - If out_subpixel is False, rounds the pixel coordinates to integers.
        """
        raise NotImplementedError

    @property
    def hw(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def mask(self) -> np.ndarray:
        return self._mask.reshape(self.height, self.width)

    def export_cam_dict(self) -> Dict[str, Any]:
        cam_dict = {}
        cam_dict["cam_type"] = self.cam_type.name
        cam_dict["image_size"] = (self.width, self.height)
        if self._max_fov_deg > 0:
            cam_dict["fov_deg"] = self._max_fov_deg
        return cam_dict

    def _warp(
        self, image: ArrayLike, uv: ArrayLike, valid_mask: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Warp the given image according to the provided uv coordinates.

        Args:
            image (ArrayLike, [H,W] or [H,W,3]): Input image to be warped. Can be grayscale or color.
            uv (ArrayLike, [2,N]): 2D input coordinates for warping. N should be equal to self.height * self.width.
            valid_mask (ArrayLike, [2,H*W], optional): Valid mask to specify which coordinates are valid. Default is None.

        Returns:
            warped_image (np.ndarray, [H,W] or [H,W,3]): Warped output image.

        Example:
            - This function uses the cv2.remap function to warp the input image.
            - The uv coordinates are expected to be in the format [2, N], where N = self.height * self.width.
            - If the input image is grayscale, it is converted to a 3D array with a single channel for consistent processing.
            - Each channel of the image is warped separately, and the results are combined to form the output image.
            - If a valid_mask is provided, only valid coordinates will be considered during warping.
        """
        _uv = convert_numpy(uv)  # check uv type
        u = _uv[0, :].reshape(self.hw).astype(np.float32)
        v = _uv[1, :].reshape(self.hw).astype(np.float32)
        output_image = cv.remap(image, u, v, cv.INTER_LINEAR)
        if valid_mask:
            valid_mask = convert_numpy(valid_mask).reshape(self.hw)
            output_image[~valid_mask] = 0.0
        return convert_array(output_image, uv)

    def _extract_mask(self, uv: ArrayLike) -> ArrayLike:
        """
        Extract a mask indicating valid uv points within the image bounds and mask.

        Arg:
            uv (ArrayLike, [2,N]): ArrayLike of shape (2, N) containing uv points where
                                     uv[0, :] are x coordinates and uv[1, :] are y coordinates.
        Return:
            ArrayLike (ArrayLike, [N,]): Boolean array of shape (N,) indicating whether each uv point
                                           is valid (True) or not (False).
        Details:
            - A uv point is considered valid if it is within the image bounds
              (0 <= x < self.width and 0 <= y < self.height).
            - Additionally, the corresponding point in the predefined mask
              (self._mask) must be True for the uv point to be considered valid.
            - The function handles uv points with both integer and float values.
            - The mask (self._mask) is expected to be a flattened array of size (H * W,).
        Example:
            uv = np.array([[1, 3, 5, 11, 0],  # x coordinates
                           [1, 3, 5, 5, -1]]) # y coordinates
            valid_mask = self._extract_mask(uv)
        """
        num_points = uv.shape[1]
        valid_mask = full_like(uv[0], False, bool)  # (N,)

        within_image_bounds = logical_and(
            uv[0, :] < self.width, uv[1, :] < self.height, uv[0, :] >= 0, uv[1, :] >= 0
        )  # (N,)

        is_in_image_indices = arange(uv, 0, num_points)[within_image_bounds]

        valid_uv = uv[:, within_image_bounds]
        if valid_uv.size == 0:
            return valid_mask
        mask_indices = as_int(valid_uv[1]) * self.width + as_int(valid_uv[0])  # (N,)

        mask = convert_array(self._mask, uv)
        valid_mask[is_in_image_indices] = mask[mask_indices]

        return valid_mask

    @staticmethod
    def load_from_cam_dict(cam_dict: Dict[str, Any]):
        cam_type = CamType.from_string(cam_dict["cam_type"])

        if cam_type == CamType.PERSPECTIVE:
            return PerspectiveCamera(cam_dict)
        elif cam_type == CamType.OPENCVFISHEYE:
            return OpenCVFisheyeCamera(cam_dict)
        elif cam_type == CamType.THINPRISM:
            return ThinPrismFisheyeCamera(cam_dict)
        elif cam_type == CamType.OMNIDIRECT:
            return OmnidirectionalCamera(cam_dict)
        elif cam_type == CamType.DOUBLESPHERE:
            return DoubleSphereCamera(cam_dict)
        elif cam_type == CamType.EQUIRECT:
            return EquirectangularCamera(cam_dict)
        else:
            LOG_CRITICAL(f"{cam_type.name} camera type is not supported.")
            return None


# Abstract Class for Radial Camera Model
class RadialCamera(Camera):
    """
    Abstract base class for radial camera models.This class extends the Camera class
    and includes intrinsic parameters and radial distortion properties.

    Args:
        cam_dict (Dict[str, Any]): Dictionary containing camera parameters.

    Attributes (except Inherited Attributes):
        fx,fy (float): Focal length in x,y direction respectively.
        cx,cy (float): Principal point in x,y direction respectively.
        skew (float): Skew parameter. Defaults to 0.

    Abstract Methods:
        dist_coeffs: distortion parameters
        _distort: Internal Method to undistorted points to distorted points.
        _undistort: Internal Method to distorted points to undistorted points.
        _compute_residual_jacobian: Internal Method to compute jacobian matrix of residuals.
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super(RadialCamera, self).__init__(cam_dict)
        self.fx, self.fy = cam_dict["focal_length"]
        self.cx, self.cy = cam_dict["principal_point"]
        self.skew = cam_dict.get("skew", 0.0)
        # For computing undistort function
        self.max_iter = 0.0
        self.err_thr = 0.0

    @property
    def K(self) -> np.ndarray:
        K = np.eye(3)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 1] = self.skew
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    @property
    def inv_K(self) -> np.ndarray:
        return inv(self.K)

    @property
    def fov(self) -> Tuple[float, float]:
        fov_x = 2 * rad2deg(arctan(self.width / (2 * self.fx)))
        fov_y = 2 * rad2deg(arctan(self.height / (2 * self.fy)))
        return fov_x.item(), fov_y.item()

    @property
    def dist_coeffs(self) -> np.ndarray:
        raise NotImplementedError

    def export_cam_dict(self) -> Dict[str, Any]:
        cam_dict = super().export_cam_dict()
        cam_dict["focal_length"] = (self.fx, self.fy)
        cam_dict["principal_point"] = (self.cx, self.cy)
        cam_dict["skew"] = self.skew
        cam_dict["dist_coeffs"] = self.dist_coeffs.tolist()
        return cam_dict

    def _to_image_plane(
        self, x: ArrayLike, y: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        u = self.fx * x + self.skew * y + self.cx
        v = self.fy * y + self.cy
        return u, v

    def _to_normalized_plane(self, uv: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        x = (
            uv[0:1, :] - self.cx - self.skew / self.fy * (uv[1:2, :] - self.cy)
        ) / self.fx  # (1,N)
        y = (uv[1:2, :] - self.cy) / self.fy  # (1,N)
        return x, y

    def has_distortion(self) -> bool:
        cnt = np.count_nonzero(self.dist_coeffs)
        return cnt != 0

    def _distort(self, x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        raise NotImplementedError

    def _compute_residual_jacobian(
        self, xu: ArrayLike, yu: ArrayLike, xd: ArrayLike, yd: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        raise NotImplementedError

    def _undistort(self, xd: ArrayLike, yd: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute undistorted coords
        Adapted from MultiNeRF
        https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

        Args:
            xd (ArrayLike, [1,N]): The distorted coordinates x.
            yd (ArrayLike, [1,N]): The distorted coordinates y.

        Returns:
            xu (ArrayLike, [1,N]): The undistorted coordinates x.
            yu (ArrayLike, [1,N]): The undistorted coordinates y.

        Details
        We want to undistortion point like distortion == distort(undistortion)
        x,y := undistorted coords. x,y
        res(x,y):= (rx,ry) = distort(x,y) - (xd,yd), residual of undistorted coords.
        J(x,y) := |rx_x rx_y|  , Jacobian of residual
                  |ry_x ry_y|
        J^-1 = 1/D | ry_y -rx_y|
                   |-ry_x  rx_x|
        D := rx_x * ry_y - rx_y * ry_x, Determinant of Jacobian

        Initialization:
            x,y := xd,yd
        Using Newton's Method:
            Iteratively
             x,y <- [x,y]^T - J^-1 * [rx, ry]^T
             => [x,y]^T - 1/D * [ry_y*rx-rx_y*ry, rx_x*ry-ry_x*rx]^T
        """
        xu, yu = deep_copy(xd), deep_copy(yd)

        for _ in range(self.max_iter):
            rx, ry, rx_x, rx_y, ry_x, ry_y = self._compute_residual_jacobian(
                xu, yu, xd, yd
            )
            if sqrt(rx**2 + ry**2).max() < NORM_PIXEL_THRESHOLD:
                break
            Det = ry_x * rx_y - rx_x * ry_y
            x_numerator, y_numerator = ry_y * rx - rx_y * ry, rx_x * ry - ry_x * rx
            step_x = where(abs(Det) > self.err_thr, x_numerator / Det, zeros_like(Det))
            step_y = where(abs(Det) > self.err_thr, y_numerator / Det, zeros_like(Det))
            xu = xu + clip(step_x, -0.5, 0.5)
            yu = yu + clip(step_y, -0.5, 0.5)
        return xu, yu

    def _compute_fov_mask(self) -> ArrayLike:
        uv = self.make_pixel_grid()
        x, y = self._to_normalized_plane(uv)
        if self.has_distortion():
            x, y = self._undistort(x, y)
        r = sqrt(x**2 + y**2)
        theta = arctan2(r, ones_like(r))
        fovx, fovy = self.fov
        max_theta = deg2rad(np.max(fovx, fovy)) / 2.0
        fov_mask = theta <= max_theta
        return fov_mask

    def convert_to_rays(
        self, uv: Optional[ArrayLike] = None, z_fixed: Optional[bool] = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        if uv is None:
            uv = self.make_pixel_grid()  # (2,HW)
            mask = convert_array(self._mask, uv)
        else:
            mask = self._extract_mask(uv)

        x, y = self._to_normalized_plane(uv)
        if self.has_distortion():
            x, y = self._undistort(x, y)
        z = ones_like(x)
        rays = concat([x, y, z], 0)  # (3,HW)
        if z_fixed is False:
            rays = normalize(rays, dim=0)
        return rays, mask

    def convert_to_pixels(
        self, rays: ArrayLike, out_subpixel: Optional[bool] = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        X = rays[0:1, :]
        Y = rays[1:2, :]
        Z = rays[2:3, :]

        invalid_depth = (Z == 0.0).reshape(
            -1,
        )

        Z[:, invalid_depth] = EPSILON

        x, y = X / Z, Y / Z
        if self.has_distortion():
            x, y = self._distort(x, y)
        u, v = self._to_image_plane(x, y)
        uv = concat([u, v], dim=0)

        uv = uv if out_subpixel else as_int(uv, n=32)  # (2,N)
        mask = self._extract_mask(uv)
        mask = logical_and(mask, logical_not(invalid_depth))
        return uv, mask

    def distort_pixel(
        self, uv: ArrayLike, out_subpixel: Optional[bool] = False
    ) -> ArrayLike:
        if self.has_distortion() is False:
            return uv
        x, y = self._to_normalized_plane(uv)
        xd, yd = self._distort(x, y)
        u, v = self._to_image_plane(xd, yd)
        uv = concat([u, v], dim=0)
        return uv if out_subpixel else as_int(uv, n=32)  # (2,HW)

    def undistort_pixel(
        self, uv: ArrayLike, out_subpixel: Optional[bool] = False
    ) -> ArrayLike:
        if self.has_distortion() is False:
            return uv
        x, y = self._to_normalized_plane(uv)
        xu, yu = self._undistort(x, y)
        u, v = self._to_image_plane(xu, yu)
        uv = concat([u, v], dim=0)
        return uv if out_subpixel else as_int(uv, n=32)  # (2,HW)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self.hw != image.shape[0:2]:
            LOG_CRITICAL("Image's resolution must be same as camera's resolution.")
            raise ValueError

        if self.has_distortion() is False:
            LOG_WARN("There is no distortion, thus the output image unchanged.")
            return image

        uv = self.distort_pixel(self.make_pixel_grid(), out_subpixel=True)
        output_image = self._warp(image, uv)
        return output_image

    def distort_image(self, image: np.ndarray) -> np.ndarray:
        if self.hw != image.shape[0:2]:
            LOG_CRITICAL("Image's resolution must be same as camera's resolution.")
            raise ValueError

        if self.has_distortion() is False:
            LOG_WARN("There is no distortion, thus the output image unchanged.")
            return image

        uv = self.undistort_pixel(self.make_pixel_grid(), out_subpixel=True)
        output_image = self._warp(image, uv)
        return output_image


# Simple Radial Camera Model or small distortion
class PerspectiveCamera(RadialCamera):
    """
    Perpective Camera Model.

    Represents and Brown-Conrady Camera Model, which is suitable to a simple radial camera model suitable for small distortions.

    Attributes:
        cam_type (CamType): Camera type, set to PERESPECTIVE.
        fx, fy (float): Focal length in x and y directions.
        cx, cy (float): Principal point coordinates (center of the image).
        skew (float): Skew coefficient between x and y axis.
        radial_params (List[float]): Radial distortion parameters [k1, k2, k3].
        tangential_params (List[float]): Tangential distortion parameters [p1, p2].
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super(PerspectiveCamera, self).__init__(cam_dict)
        self.cam_type = CamType.PERSPECTIVE
        self.radial_params = cam_dict.get("radial", [0.0, 0.0, 0.0])
        self.tangential_params = cam_dict.get("tangential", [0.0, 0.0])
        self.err_thr: float = NORM_PIXEL_THRESHOLD
        self.max_iter: int = 5

    @property
    def dist_coeffs(self) -> np.ndarray:
        _dist_coeffs = (
            self.radial_params[0:2] + self.tangential_params + self.radial_params[2:3]
        )
        return np.array(_dist_coeffs)

    @staticmethod
    def from_K(
        K: List[List[float]],
        image_size: List[int],
        dist_coeffs: Optional[List[float]] = None,
    ) -> "PerspectiveCamera":
        """
        Static method to create a PerspectiveCamera instance from intrinsic matrix and distortion coefficients.

        Args:
            K (List[List[float]]): Intrinsic matrix parameters as a list 3*3 format.
            image_size (List[int]): Image resolution as a list [width, height].
            dist_coeffs (List[float]): Distortion coefficients as a list [k1, k2, p1, p2, k3].

        ReturnS:
            PerspectiveCamera: An instance of PerspectiveCamera with given parameters.
        """

        cam_dict = {
            "image_size": image_size,
            "focal_length": (K[0][0], K[1][1]),  # fx and fy
            "principal_point": (K[0][2], K[1][2]),  # cx and cy
            "skew": K[0][1],
        }

        if dist_coeffs is not None:
            if len(dist_coeffs) != 5:
                LOG_CRITICAL(
                    f"The distortion factor must be 5, but got {len(dist_coeffs)}"
                )
            cam_dict["radial"] = (
                dist_coeffs[0],
                dist_coeffs[1],
                dist_coeffs[4],
            )  # k1, k2, k3
            cam_dict["tangential"] = (dist_coeffs[2], dist_coeffs[3])  # p1, p2
        else:
            cam_dict["radial"] = [0.0, 0.0, 0.0]
            cam_dict["tangential"] = [0.0, 0.0]

        return PerspectiveCamera(cam_dict)

    @staticmethod
    def from_fov(image_size: List[int], fov: Union[List[float], float]):
        """
        Static method to create a PerspectiveCamera instance from image resolution and field of view.

        Args:
            image_size (List[int]): Image resolution as a list [width, height].
            fov(Union[List[float],float]): Field of view in degrees as a list [fov_x, fov_y] or width field of view.

        Returns:
            PerspectiveCamera: An instance of PerspectiveCamera with calculated parameters.
        """
        width, height = image_size
        if isinstance(fov, float):
            fov_x = fov
            fov_y = fov
        else:
            fov_x, fov_y = fov
        # Calculate focal length based on FOV and image resolution
        fx = width / (2 * tan(deg2rad(fov_x) / 2))
        fy = height / (2 * tan(deg2rad(fov_y) / 2))
        # Assuming principal point is at the center of the image
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        # Assuming no skew and no distortion
        skew = 0.0
        radial_params = [0, 0, 0]  # No radial distortion
        tangential_params = [0, 0]  # No tangential distortion

        cam_dict = {
            "image_size": image_size,
            "focal_length": (fx, fy),
            "principal_point": (cx, cy),
            "skew": skew,
            "radial": radial_params,
            "tangential": tangential_params,
        }
        return PerspectiveCamera(cam_dict)

    def _distort(self, x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Applies radial and tangential distortions to the given point(x,y).

        Args:
            x (ArrayLike, [1,N]): undistorted points.
            y (ArrayLike, [1,N]): undistorted points.

        Returns:
            xd (ArrayLike, [1,N]): distorted points.
            yd (ArrayLike, [1,N]): distorted points.

        Details
        - [xd,yd]^T = R(x,y)[x,y]^T + T(x,y)
        - where R(.,.) & T(.,.) are Radial and Tangential distortion terms.
        - R(x,y) = 1 + k1*r^2 + + k2*r^4 + + k3*r^6
        - T(x,y) = [2*p1*xy + p2*(r^2+2x^2), p1*(r^2+2y^2) + 2*p2*xy]^T
        - r^2 = x^2+y^2
        """
        x2, y2 = x**2, y**2
        r2 = x2 + y2
        xy = x * y
        k1, k2, k3 = self.radial_params
        R = 1.0 + r2 * (k1 + r2 * (k2 + k3 * r2))  # Radial distortion R(x,y)
        # Tangential distortion term T(x,y) = (Tx(x,y),Ty(x,y))
        p1, p2 = self.tangential_params
        Tx = 2 * p1 * xy + p2 * (r2 + 2 * x2)  # Tx(x,y)
        Ty = p1 * (r2 + 2 * y2) + 2 * p2 * xy  # Ty(x,y)
        xd = R * x + Tx  # R(x,y)*x + Tx(x,y)
        yd = R * y + Ty  # R(x,y)*y + Ty(x,y)
        return xd, yd

    def _compute_residual_jacobian(
        self, xu: ArrayLike, yu: ArrayLike, xd: ArrayLike, yd: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Calculates the Jacobian matrix of the residual between the given distorted points(xd,yd)
        and the distorted points calculated from the undistorted points(xu,yu).

        Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py

        Args:
            xu, yu (ArrayLike, [1,N]): the undistorted points.
            xd, yd (ArrayLike, [1,N]): the distorted points.

        Returns:
            res_x, res_y (ArrayLike, [1,N]): The residuals in x and y directions.
            res_x_x, res_x_y (ArrayLike, [1,N]): Partial derivatives of the x residual with respect to x and y.
            res_y_x, res_y_y (ArrayLike, [1,N]): Partial derivatives of the y residual with respect to x and y.

        Details
        - The derivatives are calculated based on the distortion model which includes radial and tangential components.
        - [xd,yd]^T = R(r)[x,y]^T + T(x,y) where T(x,y) = [Tx,Ty]^T and r = sqrt(x^2 + y^2)
        - xd = R*x +Tx
        - yd = R*y +Ty
        - res_x_x: ∂(xd)/∂x = R + ∂R/∂x * x + ∂(Tx)/∂x
        - res_x_y: ∂(xd)/∂y = ∂R/∂y * x + ∂(Tx)/∂y
        - res_y_x: ∂(yd)/∂x = ∂R/∂x * y + ∂(Ty)/∂x
        - res_y_y: ∂(yd)/∂y = R + ∂R/∂y * y + ∂(Ty)/∂y
        """
        # Compute distorted coordinates (_xd, _yd)
        _xd, _yd = self._distort(xu, yu)
        res_x, res_y = _xd - xd, _yd - yd  # residuals of

        # Radial distortion term and its partial derivative (R, dRdx, dRdy)
        k1, k2, k3 = self.radial_params
        r2 = xu**2 + yu**2
        R = 1.0 + r2 * (k1 + r2 * (k2 + k3 * r2))  # radial distortion term R(.,.)
        dRdr = 2 * (k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2))  # ∂R/∂r * 1/r
        dRdx = dRdr * xu  # ∂R/∂x = ∂R/∂r * ∂r/∂x = ∂R/∂r * x/r
        dRdy = dRdr * yu  # ∂R/∂y = ∂R/∂r * ∂r/∂y = ∂R/∂r * y/r

        p1, p2 = self.tangential_params
        # Compute derivative of distorted point(xd) over x and y
        res_x_x = R + dRdx * xu + 2.0 * p1 * yu + 6.0 * p2 * xu  # R + ∂r/∂x + ∂(Tx)/∂x
        res_x_y = dRdy * xu + 2.0 * p1 * xu + 2.0 * p2 * yu  # ∂R/∂y * x + ∂(Tx)/∂y
        # Compute derivative of distorted point(yd) over x and y
        res_y_x = dRdy * yu + 2.0 * p2 * yu + 2.0 * p1 * xu  # ∂R/∂x * y + ∂(Ty)/∂x
        res_y_y = (
            R + dRdy * yu + 2.0 * p2 * xu + 6.0 * p1 * yu
        )  # R + ∂R/∂y * y + ∂(Ty)/∂y

        return res_x, res_y, res_x_x, res_x_y, res_y_x, res_y_y


# OpenCV Fisheye Camera Model
class OpenCVFisheyeCamera(RadialCamera):
    """
    OpenCV Fisheye Camera Model.

    Represents an Kannala-Brandt(KB) Camera Model, typically used for wide-angle or fisheye lenses.

    Attributes:
        cam_type (CamType): Camera type, set to OpenCV, indicating the OpenCV projection model.
        fx, fy (float): Focal length in x and y directions. These parameters define the scale of the image on the sensor.
        cx, cy (float): Principal point coordinates (center of the image), indicating where the optical axis intersects the image sensor.
        skew (float): Skew coefficient between x and y axis, representing the non-orthogonality between these axes.
        radial (List[float]): Radial distortion parameters [k1, k2, k3, k4], specifying the lens's radial distortion

    https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#ga75d8877a98e38d0b29b6892c5f8d7765
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super().__init__(cam_dict)
        self.cam_type = CamType.OPENCVFISHEYE
        self.radial_params = cam_dict["radial"]  # k1,k2,k3,k4
        self.err_thr: float = NORM_PIXEL_THRESHOLD
        self.max_iter: int = 20

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.radial_params)

    @staticmethod
    def from_K_D(
        K: List[float], image_size: List[int], D: List[float]
    ) -> "OpenCVFisheyeCamera":
        """
        Static method to create an EquidistantCamera (fisheye) instance from intrinsic matrix and distortion coefficients.

        Args:
            K (List[float], (3,3)): Intrinsic matrix parameters as 3*3 size.
            image_size (List[int], (2,)): Image resolution as a list [width, height].
            D (List[float], (4,)): Distortion coefficients as a list [k1, k2, k3, k4].
        Returns:
            OpenCVFisheyeCamera: OpenCVFisheyeCamera instance.
        """
        cam_dict = {
            "image_size": image_size,
            "focal_length": (K[0][0], K[1][1]),  # fx and fy
            "principal_point": (K[0][2], K[1][2]),  # cx and cy
            "skew": K[0][1],
            "radial": D,
        }
        return OpenCVFisheyeCamera(cam_dict)

    def _distort(self, x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Applies radial distortions to the given point(x,y).

        Args:
            x (ArrayLike, [1,N]): undistorted points.
            y (ArrayLike, [1,N]): undistorted points.

        Returns:
            xd (ArrayLike, [1,N]): distorted points.
            yd (ArrayLike, [1,N]): distorted points.

        Details:
        - [xd,yd]^T = R(x,y)[x,y]^T
        - where R(.,.) is Radial distortion terms.
        - R(x,y) = 1 + k1*r^2 + + k2*r^4 + + k3*r^6 + k4*r^8
        - r^2 = x^2+y^2
        """
        r2 = x**2 + y**2

        k1, k2, k3, k4 = self.radial_params
        R = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))  # Radial distortion term
        xd = R * x
        yd = R * y

        return xd, yd

    def _compute_residual_jacobian(
        self, xu: ArrayLike, yu: ArrayLike, xd: ArrayLike, yd: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Calculates the Jacobian matrix of the residual between the given distorted points(xd,yd)
        and the distorted points calculated from the undistorted points(xu,yu).

        Args:
            xu, yu (ArrayLike, [1,N]): the undistorted points.
            xd, yd (ArrayLike, [1,N]): the distorted points.

        Returns:
            res_x, res_y (ArrayLike, [1,N]): The residuals in x and y directions.
            res_x_x, res_x_y (ArrayLike, [1,N]): Partial derivatives of the x residual with respect to x and y.
            res_y_x, res_y_y (ArrayLike, [1,N]): Partial derivatives of the y residual with respect to x and y.

        Details
        - The derivatives are calculated based on the distortion model which includes radial and tangential components.
        - [xd,yd]^T = R(r)[x,y]^T where r = sqrt(x^2 + y^2)
        - xd = R*x
        - yd = R*y
        - res_x_x: ∂(xd)/∂x = R + ∂R/∂x * x
        - res_x_y: ∂(xd)/∂y = ∂R/∂y * x
        - res_y_x: ∂(yd)/∂x = ∂R/∂x * y
        - res_y_y: ∂(yd)/∂y = R + ∂R/∂y * y
        """
        # Compute distorted coordinates (_xd, _yd) and the distortion terms
        _xd, _yd = self._distort(xu, yu)
        res_x, res_y = _xd - xd, _yd - yd

        k1, k2, k3, k4 = self.radial_params
        r2 = xu**2 + yu**2
        R = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))  # Radial distortion term
        dRdr = 2 * (k1 + r2 * (2.0 * k2 + r2 * (3 * k3 + 4.0 * k4 * r2)))  # ∂R/∂r * 1/r
        dRdx = 2.0 * dRdr * xu  # ∂R/∂x = ∂R/∂r * ∂r/∂x = ∂R/∂r * x/r
        dRdy = 2.0 * dRdr * yu  # ∂R/∂y = ∂R/∂r * ∂r/∂y = ∂R/∂r * y/r

        # Compute derivative of distorted point(xd) over x and y
        res_x_x = R + dRdx * xu  # R + ∂R/∂x * x
        res_x_y = dRdy * xu  # ∂R/∂y * x
        # Compute derivative of distorted point(yd) over x and y
        res_y_x = dRdx * yu  # ∂R/∂x * y
        res_y_y = R + dRdy * yu  # R + ∂R/∂y * y

        return res_x, res_y, res_x_x, res_x_y, res_y_x, res_y_y


# COLMAP Camera Model
class ThinPrismFisheyeCamera(RadialCamera):
    """
    Thin Prism Fisheye Camera Model.

    This camera model is used for fisheye lenses, characterized by its ability to correct for complex lens distortions
    using a combination of radial, tangential, and thin prism distortion parameters.

    Attributes:
        cam_type (CamType): Camera type, set to THINPRISM, indicating the thin prism fisheye model.
        radial_params (List[float]): Radial distortion parameters [k1, k2, k3, k4].
        tangential_params (List[float]): Tangential distortion parameters [p1, p2].
        prism_params (List[float]): Prism distortion parameters [sx1, sy1], addressing thin prism distortions.
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super().__init__(cam_dict)
        self.cam_type = CamType.THINPRISM
        self.radial_params = cam_dict["radial"]  # k1,k2,k3,k4
        self.tangential_params = cam_dict["tangential"]  # p1,p2
        self.prism_params = cam_dict["prism"]  # sx1,sy1
        self.err_thr: float = NORM_PIXEL_THRESHOLD
        self.max_iter: int = 20

    @property
    def dist_coeffs(self) -> np.ndarray:
        _dist_coeffs = (
            self.radial_params[0:2]
            + self.tangential_params
            + self.radial_params[2:]
            + self.prism_params
        )
        return np.array(_dist_coeffs)

    @staticmethod
    def from_K_D(
        K: List[float], image_size: List[int], D: List[float]
    ) -> "ThinPrismFisheyeCamera":
        """
        Static method to create an ThinPrismFisheyeCamera (fisheye) instance from intrinsic matrix and distortion coefficients.

        Args:
            K (List[float]): Intrinsic matrix parameters as a list of list.
            image_size (List[int]): Image resolution as a list [width, height].
            D (List[float]): Distortion coefficients as a list [k1, k2, p1, p2, k3, k4, sx1, sy1].

        Returns:
            ThinPrismFisheyeCamera: An instance of ThinPrismFisheyeCamera with given parameters.
        """
        assert len(D) == 8, "Distortion parameters must be consist of 8"

        cam_dict = {
            "image_size": image_size,  # width and height
            "focal_length": (K[0][0], K[1][1]),  # fx and fy
            "principal_point": (K[0][2], K[1][2]),  # cx and cy
            "radial": (D[0], D[1], D[4], D[5]),  # k1,k2,k3,k4
            "tangential": (D[2], D[3]),  # p1,p2
            "prism": (D[6], D[7]),  # sx1,sy1
        }
        return ThinPrismFisheyeCamera(cam_dict)

    @staticmethod
    def from_params(image_size: List[int], params: List[float]):
        """
        Static method to create an ThinPrismFisheyeCamera (fisheye) instance from intrinsic matrix and distortion coefficients.

        Args:
            image_size (List[int]): Image resolution as a list [width, height].
            params (List[float]): Distortion coefficients as a list [fx,fy,cx,cy, k1, k2, p1, p2, k3, k4, sx1, sy1].

        Returns:
            ThinPrismFisheyeCamera: An instance of ThinPrismFisheyeCamera with given parameters.
        """
        assert len(params) == 12, "params must be 12."

        cam_dict = {
            "image_size": image_size,  # width and height
            "focal_length": (params[0], params[1]),  # fx and fy
            "principal_point": (params[2], params[3]),  # cx and cy
            "radial": (params[4], params[5], params[8], params[9]),  # k1,k2,k3,k4
            "tangential": (params[6], params[7]),  # p1,p2
            "prism": (params[10], params[11]),  # sx1,sy1
        }
        return ThinPrismFisheyeCamera(cam_dict)

    def _distort(self, x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Applies radial, tangential, and thin prism distortions to the given point(x,y).

        Args:
            x (ArrayLike, [1,N]): undistorted points.
            y (ArrayLike, [1,N]): undistorted points.

        Returns:
            xd (ArrayLike, [1,N]): distorted points.
            yd (ArrayLike, [1,N]): distorted points.

        Details
        - [xd,yd]^T = R(r)[x,y]^T + T(x,y) + S(r)
        - where R(.) & T(.,.) & S(.) are Radial, Tangential, and Thin Prism distortion terms.
        - R(r) = 1 + k1*r^2 + + k2*r^4 + + k3*r^6
        - T(x,y) = [2*p1*xy + p2*(r^2+2x^2), p1*(r^2+2y^2) + 2*p2*xy]^T
        - S(r) = [sx1 * r^2, sy1 * r^2]^T
        - r^2 = x^2+y^2
        """
        x2, y2 = x**2, y**2
        xy = x * y
        r2 = x2 + y2

        k1, k2, k3, k4 = self.radial_params
        R = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))

        p1, p2 = self.tangential_params
        # Tangential distortion term (Tx,Ty)
        Tx = 2 * p1 * xy + p2 * (r2 + 2 * x2)
        Ty = p1 * (r2 + 2 * y2) + 2 * p2 * xy
        # Thin Prism distortion term (sx,sy)
        sx1, sy1 = self.prism_params
        xd = R * x + Tx + sx1 * r2
        yd = R * y + Ty + sy1 * r2
        return xd, yd

    def _compute_residual_jacobian(
        self, xu: ArrayLike, yu: ArrayLike, xd: ArrayLike, yd: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Calculates the Jacobian matrix of the residual between the given distorted points(xd,yd)
        and the distorted points calculated from the undistorted points(xu,yu).

        Args:
            xu, yu (ArrayLike, [1,N]): the undistorted points.
            xd, yd (ArrayLike, [1,N]): the distorted points.

        Returns:
            res_x, res_y (ArrayLike, [1,N]): The residuals in x and y directions.
            res_x_x, res_x_y (ArrayLike, [1,N]): Partial derivatives of the x residual with respect to x and y.
            res_y_x, res_y_y (ArrayLike, [1,N]): Partial derivatives of the y residual with respect to x and y.

        Details
        - The derivatives are calculated based on the distortion model which includes radial, tangential, and prism components.
        - [xd,yd]^T = R(r)[x,y]^T + T(x,y) + S(r)
            where T(x,y) = [Tx,Ty]^T, S(r) = [Sx,Sy]^T, and r = sqrt(x^2 + y^2)
        - xd = R*x + Tx + Sx
        - yd = R*y + Ty + Sy
        - res_x_x: ∂(xd)/∂x = R + ∂R/∂x * x + ∂(Tx)/∂x + ∂(Sx)/∂x
        - res_x_y: ∂(xd)/∂y = ∂R/∂y * x + ∂(Tx)/∂y + ∂(Sx)/∂y
        - res_y_x: ∂(yd)/∂x = ∂R/∂x * y + ∂(Ty)/∂x + ∂(Sy)/∂x
        - res_y_y: ∂(yd)/∂y = R + ∂R/∂y * y + ∂(Ty)/∂y + ∂(Sy)/∂y
        - ∂(Sx)/∂x = ∂(Sx)/∂r * ∂r/∂x = (sx1 * 2r) * x/r = 2 * sx1 * x
        - ∂(Sx)/∂y = ∂(Sx)/∂r * ∂r/∂y = (sx1 * 2r) * y/r = 2 * sx1 * y
        - ∂(Sy)/∂x = ∂(Sy)/∂r * ∂r/∂x = (sy1 * 2r) * x/r = 2 * sy1 * x
        - ∂(Sy)/∂y = ∂(Sy)/∂r * ∂r/∂y = (sy1 * 2r) * y/r = 2 * sy1 * y
        """
        # Compute distorted coordinates (_xd, _yd)
        _xd, _yd = self._distort(xu, yu)
        res_x, res_y = _xd - xd, _yd - yd  # residuals of

        # Radial distortion term and its partial derivative (R, dRdx, dRdy)
        k1, k2, k3, k4 = self.radial_params
        r2 = xu**2 + yu**2
        R = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))  # Radial distortion term
        dRdr = 2 * (k1 + r2 * (2.0 * k2 + r2 * (3 * k3 + 4.0 * k4 * r2)))  # ∂R/∂r * 1/r
        dRdx = dRdr * xu  # ∂R/∂x = ∂R/∂r * ∂r/∂x = ∂R/∂r * x/r
        dRdy = dRdr * yu  # ∂R/∂y = ∂R/∂r * ∂r/∂y = ∂R/∂r * y/r

        p1, p2 = self.tangential_params
        sx1, sy1 = self.prism_params
        # Compute derivative of distorted point(xd) over x and y
        res_x_x = (
            R + dRdx * xu + 2.0 * p1 * yu + 6.0 * p2 * xu + 2.0 * sx1 * xu
        )  # R + ∂r/∂x + ∂(Tx)/∂x + ∂(Sx)/∂x
        res_x_y = (
            dRdy * xu + 2.0 * p1 * xu + 2.0 * p2 * yu + 2.0 * sx1 * yu
        )  # ∂R/∂y * x + ∂(Tx)/∂y + ∂(Sx)/∂y
        # Compute derivative of distorted point(yd) over x and y
        res_y_x = (
            dRdy * yu + 2.0 * p2 * yu + 2.0 * p1 * xu + 2.0 * sy1 * xu
        )  # ∂R/∂x * y + ∂(Ty)/∂x + ∂(Sy)/∂x
        res_y_y = (
            R + dRdy * yu + 2.0 * p2 * xu + 6.0 * p1 * yu + 2.0 * sy1 * yu
        )  # R + ∂R/∂y * y + ∂(Ty)/∂y + + ∂(Sy)/∂y

        return res_x, res_y, res_x_x, res_x_y, res_y_x, res_y_y


# Scaramuzza Fisheye Model (Wide FOV Fisheye Model)
class OmnidirectionalCamera(Camera):
    """
    OmnidirectionalCamera Camera Model.

    Attributes:
        cam_type (CamType): Camera type, set to OMNIDIRECT (OmnidirectionalCamera).
        cx (float): The x-coordinate of the central point of the omnidirectional image.
        cy (float): The y-coordinate of the central point of the omnidirectional image.
        affine (List[float]): Affine Transform elements.
        poly_vals (List[float]): Polynomial coefficients for forward projection.
        inv_poly_vals (List[float]): Polynomial coefficients for inverse projection.
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super().__init__(cam_dict)
        self.cam_type = CamType.OMNIDIRECT
        self.cx, self.cy = cam_dict["distortion_center"]
        self.poly_coeffs = np.array(cam_dict["poly_coeffs"])
        self.inv_poly_coeffs = np.array(cam_dict["inv_poly_coeffs"])
        self._affine = cam_dict.get("affine", [1.0, 0.0, 0.0])  # c,d,e
        assert self._max_fov_deg > 0, "FOV must be positive."
        fov_mask = self._compute_fov_mask()  # maximum fov mask
        self._mask = logical_and(fov_mask, self._mask)

    def _compute_fov_mask(self):
        uv = self.make_pixel_grid()
        u, v = uv[0, :] - self.cx, uv[1, :] - self.cy
        c, d, e = self._affine
        inv_det = 1.0 / (c - d * e)
        x = inv_det * (u - d * v)
        y = inv_det * (-e * u + c * v)

        rho = sqrt(x**2 + y**2)
        z = polyval(self.poly_coeffs, rho)
        norm_scale = sqrt(x**2 + y**2 + z**2)
        theta = arccos(z / norm_scale)
        max_theta = deg2rad(self._max_fov_deg / 2.0)
        # compute max_r
        fov_mask = theta <= max_theta
        return fov_mask

    def convert_to_rays(
        self, uv: Optional[ArrayLike] = None, z_fixed: bool = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        if uv is None:
            uv = self.make_pixel_grid()  # (2, HW)
            mask = convert_array(self._mask, uv)
        else:
            mask = self._extract_mask(uv)

        u, v = uv[0:1, :] - self.cx, uv[1:2, :] - self.cy
        c, d, e = self._affine
        inv_det = 1.0 / (c - d * e)
        x = inv_det * (u - d * v)
        y = inv_det * (-e * u + c * v)

        rho = sqrt(x**2 + y**2)
        z = polyval(self.poly_coeffs, rho)
        rays = concat([x, y, z], 0)
        rays = normalize(rays, dim=0)

        if z_fixed:
            valid = z != 0
            mask = mask & valid

            rays[:, valid] = rays[:, valid] / z[valid]
            rays[:, ~valid] = 0.0
        else:
            rays = normalize(rays, dim=0)

        return rays, mask  # (3, N)

    def convert_to_pixels(
        self, rays: ArrayLike, out_subpixel: Optional[bool] = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        rays = normalize(rays, dim=0)
        X = rays[0:1, :]
        Y = rays[1:2, :]
        Z = rays[2:3, :]

        theta = arccos(Z)
        rho = polyval(self.inv_poly_coeffs, theta)

        r = sqrt(X**2 + Y**2)
        r_proj = rho / (r + EPSILON)
        # sensor coordinates
        x = r_proj * X
        y = r_proj * Y

        c, d, e = self._affine
        # image coordinates
        u = c * x + d * y + self.cx
        v = e * x + y + self.cy

        uv = concat([u, v], dim=0)
        mask = self._extract_mask(uv)
        uv = uv if out_subpixel else as_int(uv, n=32)
        return uv, mask  # (2,N)

    def export_cam_dict(self) -> Dict[str, Any]:
        cam_dict = super().export_cam_dict()
        cam_dict["distortion_center"] = (self.cx, self.cy)
        cam_dict["affine"] = self._affine
        cam_dict["poly_coeffs"] = self.inv_poly_coeffs.tolist()
        cam_dict["inv_poly_coeffs"] = self.inv_poly_coeffs.tolist()
        return cam_dict


# Double Sphere Model
class DoubleSphereCamera(Camera):
    """
    Double Sphere Camera Model. Adapted by https://github.com/matsuren/dscamera

    Attributes:
        cam_type (CamType): Camera type, set to DOUBLESPHERE (Double Sphere Camera).
        cx (float): The x-coordinate of the central point of the image.
        cy (float): The y-coordinate of the central point of the image.
        fx (float): The focal length in x direction.
        fy (float): The focal length in y direction.
        xi (float): The first parameter of the double sphere model.
        alpha (float): The second parameter of the double sphere model.
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super().__init__(cam_dict)
        self.cam_type = CamType.DOUBLESPHERE
        self.cx, self.cy = cam_dict["principal_point"]
        self.fx, self.fy = cam_dict["focal_length"]
        self.xi = cam_dict["xi"]
        self.alpha = cam_dict["alpha"]

        assert self._max_fov_deg > 0, "FOV must be positive."
        self.fov_cos = cos(deg2rad(self._max_fov_deg / 2.0))

    def export_cam_dict(self) -> Dict[str, Any]:
        cam_dict = super().export_cam_dict()
        cam_dict["principal_point"] = (self.cx, self.cy)
        cam_dict["focal_length"] = (self.fx, self.fy)
        cam_dict["xi"] = self.xi
        cam_dict["alpha"] = self.alpha
        return cam_dict

    def _compute_fov_mask(self, z: ArrayLike) -> ArrayLike:
        # z must be an element of unit vector. i.e. |(x,y,z)| = 1.
        return z >= convert_array(self.fov_cos, z)

    def convert_to_rays(
        self, uv: Optional[ArrayLike] = None, z_fixed: bool = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        if uv is None:
            uv = self.make_pixel_grid()  # (2, HW)
            mask = convert_array(self._mask, uv)
        else:
            mask = self._extract_mask(uv)

        mx = (uv[0:1, :] - self.cx) / self.fx
        my = (uv[1:2, :] - self.cy) / self.fy
        r2 = mx**2 + my**2

        s = 1.0 - (2 * self.alpha - 1.0) * r2
        valid_mask: ArrayLike = s >= 0.0
        s[logical_not(valid_mask)] = 0.0
        mz = (1 - self.alpha * self.alpha * r2) / (
            self.alpha * sqrt(s) + 1.0 - self.alpha
        )

        k = (mz * self.xi + sqrt(mz**2 + (1.0 - self.xi * self.xi) * r2)) / (mz**2 + r2)
        X = k * mx
        Y = k * my
        Z = k * mz - self.xi
        rays = concat([X, Y, Z], 0)

        # Compute FOV Mask
        fov_mask = self._compute_fov_mask(Z)

        if z_fixed:
            valid_z: ArrayLike = Z != 0
            mask = logical_and(
                mask,
                fov_mask.reshape(
                    -1,
                ),
                valid_mask.reshape(
                    -1,
                ),
                valid_z.reshape(
                    -1,
                ),
            )
            rays[:, valid_z] = rays[:, valid_z] / Z[valid_z]
            rays[:, logical_not(valid_z)] = 0.0
        else:
            mask = logical_and(
                mask,
                fov_mask.reshape(
                    -1,
                ),
                valid_mask.reshape(
                    -1,
                ),
            )

        return rays, mask  # (3, N)

    def convert_to_pixels(
        self, rays: ArrayLike, out_subpixel: Optional[bool] = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        rays = normalize(rays, dim=0)
        X = rays[0:1, :]
        Y = rays[1:2, :]
        Z = rays[2:3, :]

        X2, Y2, Z2 = X**2, Y**2, Z**2
        d1 = sqrt(X2 + Y2 + Z2)

        xidz = self.xi * d1 + Z
        d2 = sqrt(X2 + Y2 + xidz**2)

        denom = self.alpha * d2 + (1.0 - self.alpha) * xidz
        u = self.fx * X / denom + self.cx
        v = self.fy * Y / denom + self.cy
        uv = concat([u, v], dim=0)

        # compute valid area
        if self.alpha <= 0.5:
            w1 = self.alpha / (1.0 - self.alpha)
        else:
            w1 = (1.0 - self.alpha) / self.alpha
        w2 = w1 + self.xi / sqrt(2 * w1 * self.xi + self.xi**2 + 1.0)
        valid_mask: ArrayLike = Z > -w2 * d1

        fov_mask = self._compute_fov_mask(Z)
        mask = self._extract_mask(uv)
        mask = logical_and(
            fov_mask.reshape(
                -1,
            ),
            valid_mask.reshape(
                -1,
            ),
            mask,
        )
        uv = uv if out_subpixel else as_int(uv, n=32)

        return uv, mask  # (2,HW)


# Equirectangular Camera Model
class EquirectangularCamera(Camera):
    """
    Equirectangular Camera Model.

    This camera model is used for representing 360-degree panoramic images
    using equirectangular projection. It maps spherical coordinates to a 2D equirectangular plane.

    Attributes:
        cam_type (CamType): Camera type, set to EQUIRECT (Equirectangular).
        min_phi_deg (float): Minimum vertical field of view angle (phi) in degrees.
        max_phi_deg (float): Maximum vertical field of view angle (phi) in degrees.
        cx (float): The x-coordinate of the central point of the equirectangular image.
        cy (float): The y-coordinate of the central point of the equirectangular image.
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super().__init__(cam_dict)
        self.cam_type = CamType.EQUIRECT

        self.min_phi_deg: float = cam_dict.get("min_phi_deg", -90.0)
        self.max_phi_deg: float = cam_dict.get("max_phi_deg", 90.0)
        self.cx: float = (self.width - 1) / 2.0
        self.cy: float = (self.height - 1) / 2.0

        self.phi_scale = deg2rad(self.max_phi_deg - self.min_phi_deg)
        self.phi_offset = deg2rad((self.max_phi_deg + self.min_phi_deg) * 0.5)

    @staticmethod
    def from_image_size(image_size: Tuple[int, int]) -> "EquirectangularCamera":
        cam_dict = {"image_size": image_size, "min_phi_deg": -90.0, "max_phi_deg": 90.0}
        return EquirectangularCamera(cam_dict)

    def export_cam_dict(self) -> Dict[str, Any]:
        cam_dict = super().export_cam_dict()
        cam_dict["min_phi_deg"] = self.min_phi_deg
        cam_dict["max_phi_deg"] = self.max_phi_deg
        return cam_dict

    def convert_to_rays(
        self, uv: Optional[ArrayLike] = None, z_fixed: bool = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        if uv is None:
            uv = self.make_pixel_grid()
            mask = convert_array(self._mask, uv)
        else:
            mask = self._extract_mask(uv)

        theta = (uv[0:1, :] - self.cx) / self.width * PI * 2.0
        phi = (uv[1:2, :] - self.cy) / self.height * self.phi_scale + self.phi_offset
        x = sin(theta) * cos(phi)
        y = sin(phi)
        z = cos(theta) * cos(phi)
        rays = concat([x, y, z], 0)  # (3,N)
        valid_ray = logical_and(
            phi >= self.min_phi_deg, phi <= self.max_phi_deg
        ).reshape(
            -1,
        )
        mask = logical_and(valid_ray, mask)

        if z_fixed is True:
            rays = rays / z

        return rays, mask

    def convert_to_pixels(
        self, rays: ArrayLike, out_subpixel: Optional[bool] = False
    ) -> Tuple[ArrayLike, ArrayLike]:
        # Normalize the ray vector
        rays = normalize(rays, dim=0)
        X = rays[0:1, :]
        Y = rays[1:2, :]
        Z = rays[2:3, :]

        # Convert Cartesian coordinates to spherical coordinates
        theta = arctan2(X, Z)
        phi = arcsin(Y)
        # Convert spherical coordinates to pixel coordinates
        u = theta / (PI * 2) * self.width + self.cx
        v = (phi - self.phi_offset) * self.height / self.phi_scale + self.cy
        uv = concat([u, v], dim=0)
        mask = self._extract_mask(uv)
        uv = uv if out_subpixel else as_int(uv, n=32)
        return uv, mask
