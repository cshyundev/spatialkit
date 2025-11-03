"""
Module Name: base.py

Description:
    This module defines the abstract MarkerDetector base class that provides
    common functionality for all fiducial marker detectors.

Main Classes:
    - MarkerDetector: Abstract base class for detecting markers and estimating poses.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: v0.3.0
License: MIT License
"""

from typing import List
import numpy as np
import cv2 as cv

from ..camera import Camera
from ..ops.uops import ArrayLike, convert_numpy, is_array, swapaxes
from .marker import Marker, FiducialMarkerType
from ..vis2d import draw_polygon
from ..common.exceptions import (
    InvalidMarkerTypeError,
    MarkerDetectionError,
    IncompatibleTypeError
)


class MarkerDetector:
    """
    A base (abstract) class for detecting markers and estimating their pose.

    Attributes:
        _cam (Camera): The camera model containing intrinsics/extrinsics.
        _marker_size (float): The size (in meters or arbitrary units) of the square marker.
        _marker_type (FiducialMarkerType): The type of marker dictionary used.
        corner_3d (np.ndarray): An array of shape (4, 3) representing 3D coordinates
                                of the marker corners relative to the marker's center.

    Note:
        - By default, the marker is assumed to lie on the Z=0 plane, with its center
          at the origin (0, 0, 0). The 3D corners are ordered as follows:

            (1) Top-Left    : (-s, s, 0)
            (2) Top-Right   : ( s, s, 0)
            (3) Bottom-Right: ( s,-s, 0)
            (4) Bottom-Left : (-s,-s, 0)
        where s = marker_size / 2.
    """

    def __init__(
        self,
        cam: Camera,
        marker_size: float,
        marker_type: FiducialMarkerType = FiducialMarkerType.NONE,
    ):
        """
        Initializes the MarkerDetector with a camera, marker size, and marker dictionary type.

        Args:
            cam (Camera): A camera object containing intrinsic/extrinsic parameters.
            marker_size (float): The side length of the marker in meters.
            marker_type (FiducialMarkerType): The specific marker dictionary type to use.

        Raises:
            IncompatibleTypeError: If cam is not a Camera instance.
            InvalidMarkerTypeError: If marker_size is not positive or marker_type is invalid.
        """
        # Arguments Check
        if not isinstance(cam, Camera):
            raise IncompatibleTypeError(
                f"Camera must be a Camera instance, got {type(cam)}. "
                f"Please provide a valid Camera object."
            )

        if not isinstance(marker_size, (int, float)):
            raise InvalidMarkerTypeError(
                f"Marker size must be a number, got {type(marker_size)}. "
                f"Please provide a valid positive number."
            )

        if marker_size <= 0.0:
            raise InvalidMarkerTypeError(
                f"Marker size must be positive, got {marker_size}. "
                f"Please provide a positive marker size value."
            )

        if not isinstance(marker_type, FiducialMarkerType):
            raise InvalidMarkerTypeError(
                f"Marker type must be a FiducialMarkerType enum, got {type(marker_type)}. "
                f"Please provide a valid FiducialMarkerType."
            )

        self._cam: Camera = cam
        self._marker_size: float = float(marker_size)
        self._marker_type: FiducialMarkerType = marker_type

        # 3D corner
        s = self._marker_size / 2.0
        self.corner_3d = np.array(
            [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float32
        )

    @property
    def marker_size(self):
        return self._marker_size

    @property
    def marker_type(self):
        return self._marker_type

    def detect_marker(
        self, image: ArrayLike, estimate_pose: bool = True
    ) -> List[Marker]:
        """
        Detects markers in the input image and returns a list of Marker objects.

        Args:
            image (ArrayLike, [H,W,3] or [H,W]): An input image (RGB or grayscale)
                            where markers might be present.
            estimate_pose (bool): If True, estimate pose using a PnP algorithm.

        Returns:
            List[Marker]: A list of Marker objects containing IDs, corner points,
                          and camera-relative pose information.

        Raises:
            MarkerDetectionError: This is an abstract method that must be implemented in subclasses.

        Note:
            - This is an abstract method; it must be implemented in a subclass.
            - The typical flow involves:
                1) Marker detection (finding corner points and IDs).
                2) Pose estimation using solvePnP or a similar approach.
                3) Creating Marker instances with the results.
        """
        raise MarkerDetectionError(
            "detect_marker is an abstract method and must be implemented in a subclass. "
            "Please use OpenCVMarkerDetector, AprilTagMarkerDetector, or STagMarkerDetector."
        )

    def draw_axes(
        self,
        image: np.ndarray,
        marker: Marker,
        axis_length: float = 0.1,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draws the local X, Y, and Z axes of a marker onto the image.

        Depending on whether the image is grayscale or color:
          - Grayscale:
              Each axis is drawn with a different gray intensity.
          - Color (RGB):
              X-axis: Red, Y-axis: Green, Z-axis: Blue.
        Args:
            image (ArrayLike, [H,W,3] or [H,W]): An input image (RGB or grayscale)
                            where markers might be present.
            marker (Marker):
                The marker whose pose we want to visualize.
                marker.marker2cam is the 3D transform from marker-frame to camera-frame.
            axis_length (float):
                The length of each axis in marker-local units (e.g. meters).
            thickness (int):
                Line thickness in pixels.

        Returns:
            np.ndarray:
                The image with axes drawn. For grayscale images,
                axes are drawn with different intensities; for color images,
                they are drawn in RGB colors.

        Raises:
            InvalidShapeError: If image dimensions are invalid.
            MarkerDetectionError: If marker visualization fails.

        Notes:
            - Four points are defined in marker-local space:
                * [0, 0, 0]             : Origin
                * [axis_length, 0, 0]    : End of X-axis
                * [0, axis_length, 0]    : End of Y-axis
                * [0, 0, axis_length]    : End of Z-axis
            - These points are transformed into the camera frame and then
              projected to 2D pixel coordinates via self._cam.convert_to_pixels.
        """
        # Determine if the image is grayscale or color
        is_grayscale = len(image.shape) == 2

        # For safe drawing, create a copy of the input
        out_img = image.copy()

        # 1) Define 3D axis points in the marker's local frame
        axes_3d_marker = np.array(
            [
                [0, 0, 0],  # Origin
                [axis_length, 0, 0],  # X-axis end
                [0, axis_length, 0],  # Y-axis end
                [0, 0, axis_length],  # Z-axis end
            ],
            dtype=np.float32,
        )

        # 2) Transform these points to camera frame
        axes_3d_cam = marker.marker2cam * swapaxes(axes_3d_marker, 0, 1)

        # 3) Project points to 2D
        axes_2d, _ = self._cam.convert_to_pixels(axes_3d_cam)
        # axes_2d[0] -> origin
        # axes_2d[1] -> X-axis end
        # axes_2d[2] -> Y-axis end
        # axes_2d[3] -> Z-axis end

        # Convert to (x,y) integer coordinates for line drawing
        origin = tuple(axes_2d[:, 0].astype(np.int32))
        x_axis = tuple(axes_2d[:, 1].astype(np.int32))
        y_axis = tuple(axes_2d[:, 2].astype(np.int32))
        z_axis = tuple(axes_2d[:, 3].astype(np.int32))

        if is_grayscale:
            # Single-channel drawing: use different intensities
            # X -> 255, Y -> 170, Z -> 85 (example intensities)
            cv.line(out_img, origin, x_axis, 255, thickness)  # X-axis
            cv.line(out_img, origin, y_axis, 170, thickness)  # Y-axis
            cv.line(out_img, origin, z_axis, 85, thickness)  # Z-axis
        else:
            # Multi-channel (assume BGR)
            cv.line(out_img, origin, x_axis, (255, 0, 0), thickness)  # X-axis: Blue
            cv.line(out_img, origin, y_axis, (0, 255, 0), thickness)  # Y-axis: Green
            cv.line(out_img, origin, z_axis, (0, 0, 255), thickness)  # Z-axis: Red

        return out_img

    def draw_markers(
        self,
        image: ArrayLike,
        markers: List[Marker],
        draw_axes: bool = True,
        thickness: int = 2,
        axis_length: float = 0.05,
    ) -> np.ndarray:
        """
        Draws detected markers (ID text, axes, etc.) on the input image.

        Args:
            image (ArrayLike, [H,W,3] or [H,W]): An input image (RGB or grayscale)
                            where markers might be present.
            markers (List[Marker]):
                A list of Marker objects to be visualized.
            draw_axes (bool): If True, the marker's axes are drawn.
            thickness (int): Line thickness in pixels.
            axis_length (float):
                The length of the 3D axes to draw for each marker (in marker-local units).

        Returns:
            np.ndarray:
                A copy of the input image with the markers' IDs and axes drawn.
                The output remains single-channel for grayscale images and multi-channel for color images.

        Raises:
            IncompatibleTypeError: If image is not an array-like object.
        """
        if not is_array(image):
            raise IncompatibleTypeError(
                f"Image must be ArrayLike (numpy or torch), got {type(image)}. "
                f"Please provide a valid image array."
            )

        image = convert_numpy(image)
        is_grayscale = len(image.shape) == 2
        out_img = image.copy()

        for marker in markers:
            # Draw marker ID near the first corner

            # Draw Rectangle
            corners = marker.corners.astype(np.int32)
            text_pos = tuple(corners[0])

            # Put ID on the top-left corner
            if is_grayscale:
                cv.putText(
                    out_img,
                    f"{marker.id}",
                    text_pos,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    170,
                    thickness,
                )
                out_img = draw_polygon(out_img, corners, 170, 2)
            else:
                cv.putText(
                    out_img,
                    f"{marker.id}",
                    text_pos,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    thickness,
                )
                out_img = draw_polygon(out_img, corners, (255, 0, 0), thickness)

            # Draw Axes
            if draw_axes:
                # Draw the 3D axes for this marker
                out_img = self.draw_axes(
                    out_img, marker, axis_length=axis_length, thickness=thickness
                )
        return out_img
