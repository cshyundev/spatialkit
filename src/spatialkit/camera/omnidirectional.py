"""
Omnidirectional camera model (Scaramuzza fisheye model).

This module provides the OmnidirectionalCamera class implementing the Scaramuzza
omnidirectional fisheye model for wide FOV cameras.

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT LICENSE
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..ops.uops import *
from ..ops.umath import *

from ..common.constant import EPSILON
from ..common.exceptions import InvalidCameraParameterError

from .base import Camera, CamType


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
        """
        Initialize OmnidirectionalCamera.

        Args:
            cam_dict (Dict[str, Any]): Camera parameter dictionary.

        Raises:
            InvalidCameraParameterError: If FOV parameter is invalid.
        """
        super().__init__(cam_dict)
        self.cam_type = CamType.OMNIDIRECT
        self.cx, self.cy = cam_dict["distortion_center"]
        self.poly_coeffs = np.array(cam_dict["poly_coeffs"])
        self.inv_poly_coeffs = np.array(cam_dict["inv_poly_coeffs"])
        self._affine = cam_dict.get("affine", [1.0, 0.0, 0.0])  # c,d,e
        if self._max_fov_deg <= 0:
            raise InvalidCameraParameterError(
                f"Field of view must be positive, got {self._max_fov_deg} degrees. "
                f"Please provide a valid FOV value greater than 0."
            )
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
            valid = (z != 0).reshape(-1,)
            mask = mask & valid

            # Only divide where z is non-zero
            rays[:, valid] = rays[:, valid] / z[:, valid]
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

    def __repr__(self) -> str:
        """
        Return a verbose string representation of the OmnidirectionalCamera.

        Returns:
            str: Multi-line string showing camera parameters.
        """
        lines = [
            f"{self.__class__.__name__}(",
            f"  type={self.cam_type.value[0]}",
            f"  size=({self.width}, {self.height})",
            f"  center=({self.cx:.4f}, {self.cy:.4f})",
            f"  fov={self._max_fov_deg:.1f} deg",
            f"  poly_order={len(self.poly_coeffs)}",
            ")",
        ]
        return "\n".join(lines)


__all__ = ["OmnidirectionalCamera"]
