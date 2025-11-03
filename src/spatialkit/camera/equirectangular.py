"""
Equirectangular camera model.

This module provides the EquirectangularCamera class for 360-degree panoramic
images using equirectangular projection.

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT LICENSE
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..ops.uops import *
from ..ops.umath import *

from ..common.constant import PI

from .base import Camera, CamType


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


__all__ = ["EquirectangularCamera"]
