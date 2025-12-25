"""
Double sphere camera model.

This module provides the DoubleSphereCamera class implementing the double sphere
projection model for wide FOV cameras.

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT LICENSE
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..ops.uops import *
from ..ops.umath import *

from ..common.exceptions import InvalidCameraParameterError

from .base import Camera, CamType


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
        """
        Initialize DoubleSphereCamera.

        Args:
            cam_dict (Dict[str, Any]): Camera parameter dictionary.

        Raises:
            InvalidCameraParameterError: If FOV parameter is invalid.
        """
        super().__init__(cam_dict)
        self.cam_type = CamType.DOUBLESPHERE
        self.cx, self.cy = cam_dict["principal_point"]
        self.fx, self.fy = cam_dict["focal_length"]
        self.xi = cam_dict["xi"]
        self.alpha = cam_dict["alpha"]

        if self._max_fov_deg <= 0:
            raise InvalidCameraParameterError(
                f"Field of view must be positive, got {self._max_fov_deg} degrees. "
                f"Please provide a valid FOV value greater than 0."
            )
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
            valid_z: ArrayLike = (Z != 0).reshape(-1,)
            mask = logical_and(
                mask,
                fov_mask.reshape(
                    -1,
                ),
                valid_mask.reshape(
                    -1,
                ),
                valid_z,
            )
            # Only divide where Z is non-zero
            rays[:, valid_z] = rays[:, valid_z] / Z[:, valid_z]
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

    def __repr__(self) -> str:
        """
        Return a verbose string representation of the DoubleSphereCamera.

        Returns:
            str: Multi-line string showing camera parameters.
        """
        lines = [
            f"{self.__class__.__name__}(",
            f"  type={self.cam_type.value[0]}",
            f"  size=({self.width}, {self.height})",
            f"  f=({self.fx:.4f}, {self.fy:.4f})",
            f"  c=({self.cx:.4f}, {self.cy:.4f})",
            f"  xi={self.xi:.4f}, alpha={self.alpha:.4f}",
            f"  fov={self._max_fov_deg:.1f} deg",
            ")",
        ]
        return "\n".join(lines)


__all__ = ["DoubleSphereCamera"]
