"""
Perspective camera model.

This module provides the PerspectiveCamera class implementing the Brown-Conrady
camera model suitable for perspective projection with radial and tangential distortion.

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT LICENSE
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from ..ops.uops import *
from ..ops.umath import *

from ..common.constant import NORM_PIXEL_THRESHOLD
from ..common.logger import LOG_CRITICAL

from .base import CamType
from .radial_base import RadialCamera


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


__all__ = ["PerspectiveCamera"]
