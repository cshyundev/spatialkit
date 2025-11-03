"""
Fisheye camera models.

This module provides fisheye camera implementations including OpenCV fisheye
(Kannala-Brandt model) and ThinPrism fisheye model.

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT LICENSE
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from ..ops.uops import *
from ..ops.umath import *

from ..common.constant import NORM_PIXEL_THRESHOLD
from ..common.exceptions import InvalidCameraParameterError

from .base import CamType
from .radial_base import RadialCamera


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

        Raises:
            InvalidCameraParameterError: If distortion parameters are invalid.
        """
        if len(D) != 8:
            raise InvalidCameraParameterError(
                f"ThinPrism camera requires exactly 8 distortion parameters, got {len(D)}. "
                f"Expected parameters: [k1, k2, p1, p2, k3, k4, sx1, sy1]."
            )

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

        Raises:
            InvalidCameraParameterError: If parameter count is invalid.
        """
        if len(params) != 12:
            raise InvalidCameraParameterError(
                f"ThinPrism camera requires exactly 12 parameters, got {len(params)}. "
                f"Expected: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1]."
            )

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


__all__ = ["OpenCVFisheyeCamera", "ThinPrismFisheyeCamera"]
