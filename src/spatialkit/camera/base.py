"""
Base camera classes and types.

This module provides the base Camera class and CamType enumeration that serve
as foundations for all camera models in spatialkit.

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT LICENSE
"""

from typing import Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import cv2 as cv

from ..ops.uops import *
from ..ops.umath import *

from ..common.constant import EPSILON
from ..common.logger import LOG_CRITICAL
from ..common.exceptions import InvalidShapeError


class CamType(Enum):
    """
    Enumeration of camera types supported by spatialkit.

    Each camera type corresponds to a different projection model suitable
    for different lens characteristics and field of view requirements.
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
        """
        Sets a mask for the camera.

        Args:
            mask (ArrayLike): Boolean mask array.

        Raises:
            InvalidShapeError: If mask shape doesn't match image size.
        """
        if mask.shape != self.hw:
            raise InvalidShapeError(
                f"Mask shape {mask.shape} must match image size {self.hw}. "
                f"Please ensure the mask has the same dimensions as the camera image."
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
        mask_indices = as_int(valid_uv[1], n=32) * self.width + as_int(valid_uv[0], n=32)  # (N,)

        mask = convert_array(self._mask, uv)
        # Ensure mask values are properly cast to boolean type
        mask_values = mask[mask_indices]
        if is_tensor(mask_values):
            mask_values = mask_values.bool()
        valid_mask[is_in_image_indices] = mask_values

        return valid_mask

    @staticmethod
    def load_from_cam_dict(cam_dict: Dict[str, Any]):
        """
        Factory method to create camera instances from a parameter dictionary.

        Args:
            cam_dict (Dict[str, Any]): Camera parameter dictionary.

        Returns:
            Camera: Instance of the appropriate camera subclass.
        """
        # Import here to avoid circular imports
        from .perspective import PerspectiveCamera
        from .fisheye import OpenCVFisheyeCamera, ThinPrismFisheyeCamera
        from .omnidirectional import OmnidirectionalCamera
        from .doublesphere import DoubleSphereCamera
        from .equirectangular import EquirectangularCamera

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

    def __repr__(self) -> str:
        """
        Return a verbose string representation of the Camera.

        Returns:
            str: Multi-line string showing camera type and image size.
        """
        lines = [
            f"{self.__class__.__name__}(",
            f"  type={self.cam_type.value[0]}",
            f"  size=({self.width}, {self.height})",
            ")",
        ]
        return "\n".join(lines)


__all__ = ["CamType", "Camera"]
