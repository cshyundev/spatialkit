"""
Radial camera model base class.

This module provides the RadialCamera abstract base class for camera models
that use radial distortion (perspective and fisheye cameras).

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT LICENSE
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..ops.uops import *
from ..ops.umath import *

from ..common.constant import NORM_PIXEL_THRESHOLD, EPSILON
from ..common.logger import LOG_WARN
from ..common.exceptions import InvalidShapeError

from .base import Camera


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
        """
        Undistorts an input image using camera parameters.

        Args:
            image (np.ndarray): Input image to undistort.

        Returns:
            np.ndarray: Undistorted image.

        Raises:
            InvalidShapeError: If image resolution doesn't match camera resolution.
        """
        if self.hw != image.shape[0:2]:
            raise InvalidShapeError(
                f"Image resolution {image.shape[0:2]} must match camera resolution {self.hw}. "
                f"Please ensure the input image has the same dimensions as the camera."
            )

        if self.has_distortion() is False:
            LOG_WARN("No distortion parameters found, returning image unchanged.")
            return image

        uv = self.distort_pixel(self.make_pixel_grid(), out_subpixel=True)
        output_image = self._warp(image, uv)
        return output_image

    def distort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Distorts an input image using camera parameters.

        Args:
            image (np.ndarray): Input image to distort.

        Returns:
            np.ndarray: Distorted image.

        Raises:
            InvalidShapeError: If image resolution doesn't match camera resolution.
        """
        if self.hw != image.shape[0:2]:
            raise InvalidShapeError(
                f"Image resolution {image.shape[0:2]} must match camera resolution {self.hw}. "
                f"Please ensure the input image has the same dimensions as the camera."
            )

        if self.has_distortion() is False:
            LOG_WARN("No distortion parameters found, returning image unchanged.")
            return image

        uv = self.undistort_pixel(self.make_pixel_grid(), out_subpixel=True)
        output_image = self._warp(image, uv)
        return output_image

    def __repr__(self) -> str:
        """
        Return a verbose string representation of the RadialCamera.

        Returns:
            str: Multi-line string showing camera parameters.
        """
        fov_x, fov_y = self.fov
        has_dist = self.has_distortion()
        lines = [
            f"{self.__class__.__name__}(",
            f"  type={self.cam_type.value[0]}",
            f"  size=({self.width}, {self.height})",
            f"  f=({self.fx:.4f}, {self.fy:.4f})",
            f"  c=({self.cx:.4f}, {self.cy:.4f})",
            f"  fov=({fov_x:.1f}, {fov_y:.1f}) deg",
            f"  distortion={has_dist}",
            ")",
        ]
        return "\n".join(lines)


__all__ = ["RadialCamera"]
