"""
Module Name: opencv_detector.py

Description:
    This module defines a detector for ArUco and other markers supported by OpenCV.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: v0.3.0
License: MIT License
"""

from typing import List
import numpy as np
import cv2 as cv

from ..camera import Camera
from ..geom import solve_pnp
from ..ops.uops import ArrayLike, convert_numpy, is_array, swapaxes
from .marker import Marker, FiducialMarkerType
from ..common.exceptions import (
    InvalidMarkerTypeError,
    MarkerDetectionError,
    IncompatibleTypeError
)
from .base import MarkerDetector


class OpenCVMarkerDetector(MarkerDetector):
    """
    A marker detector that utilizes OpenCV's ArUco module to detect fiducial markers.

    This class implements marker detection by using predefined ArUco dictionaries from OpenCV.
    It supports various ArUco marker types defined in FiducialMarkerType.
    """

    # Predefine which FiducialMarkerTypes are valid ArUco dictionaries
    _VALID_OPENCV_TYPES_MAP = {
        # ArUco
        FiducialMarkerType.ARUCO_4X4_50: cv.aruco.DICT_4X4_50,
        FiducialMarkerType.ARUCO_4X4_100: cv.aruco.DICT_4X4_100,
        FiducialMarkerType.ARUCO_4X4_250: cv.aruco.DICT_4X4_250,
        FiducialMarkerType.ARUCO_4X4_1000: cv.aruco.DICT_4X4_1000,
        FiducialMarkerType.ARUCO_5X5_50: cv.aruco.DICT_5X5_50,
        FiducialMarkerType.ARUCO_5X5_100: cv.aruco.DICT_5X5_100,
        FiducialMarkerType.ARUCO_5X5_250: cv.aruco.DICT_5X5_250,
        FiducialMarkerType.ARUCO_5X5_1000: cv.aruco.DICT_5X5_1000,
        FiducialMarkerType.ARUCO_6X6_50: cv.aruco.DICT_6X6_50,
        FiducialMarkerType.ARUCO_6X6_100: cv.aruco.DICT_6X6_100,
        FiducialMarkerType.ARUCO_6X6_250: cv.aruco.DICT_6X6_250,
        FiducialMarkerType.ARUCO_6X6_1000: cv.aruco.DICT_6X6_1000,
        FiducialMarkerType.ARUCO_7X7_50: cv.aruco.DICT_7X7_50,
        FiducialMarkerType.ARUCO_7X7_100: cv.aruco.DICT_7X7_100,
        FiducialMarkerType.ARUCO_7X7_250: cv.aruco.DICT_7X7_250,
        FiducialMarkerType.ARUCO_7X7_1000: cv.aruco.DICT_7X7_1000,
        FiducialMarkerType.ARUCO_ORIGINAL: cv.aruco.DICT_ARUCO_ORIGINAL,
        FiducialMarkerType.ARUCO_MIP_36H12: cv.aruco.DICT_ARUCO_MIP_36h12,
        # AprilTag
        FiducialMarkerType.APRILTAG_16H5: cv.aruco.DICT_APRILTAG_16H5,
        FiducialMarkerType.APRILTAG_25H9: cv.aruco.DICT_APRILTAG_25H9,
        FiducialMarkerType.APRILTAG_36H10: cv.aruco.DICT_APRILTAG_36H10,
        FiducialMarkerType.APRILTAG_36H11: cv.aruco.DICT_APRILTAG_36H11,
    }

    def __init__(
        self,
        cam: Camera,
        marker_size: float,
        marker_type: FiducialMarkerType = FiducialMarkerType.NONE,
    ):
        '''
        Initializes the OpenCVMarkerDetector with a camera, marker size, and OpenCV dictionary type.

        Args:
            cam (Camera): A camera object containing intrinsic/extrinsic parameters.
            marker_size (float): The side length of the marker in meters.
            marker_type (FiducialMarkerType): The specific OpenCV dictionary type to use.

        Raises:
            InvalidMarkerTypeError: If the provided marker_type is not recognized as a valid OpenCV type.
        '''
        super(OpenCVMarkerDetector, self).__init__(cam, marker_size, marker_type)

        # Check if the provided marker_type is a valid dictionary
        if marker_type not in self._VALID_OPENCV_TYPES_MAP:
            raise InvalidMarkerTypeError(
                f"Marker type {marker_type} is not supported by OpenCV detector. "
                f"Supported types: {list(self._VALID_OPENCV_TYPES_MAP.keys())}. "
                f"Please use a valid OpenCV marker dictionary type."
            )

        try:
            _opencv_dict = cv.aruco.getPredefinedDictionary(
                self._VALID_OPENCV_TYPES_MAP[marker_type]
            )
            _opencv_detect_params = cv.aruco.DetectorParameters()
            self._opencv_detector = cv.aruco.ArucoDetector(
                _opencv_dict, _opencv_detect_params
            )
        except Exception as e:
            raise MarkerDetectionError(
                f"Failed to initialize OpenCV ArUco detector: {e}. "
                f"Please check your OpenCV installation and marker type."
            ) from e

    def detect_marker(
        self, image: ArrayLike, estimate_pose: bool = True
    ) -> List[Marker]:
        '''
        Detects OpenCV ArUco markers in the input image and returns a list of Marker objects.

        Args:
            image (ArrayLike, [H,W,3] or [H,W]): An input image (RGB or grayscale).
            estimate_pose (bool): If True, estimate pose using the PnP algorithm.

        Returns:
            List[Marker]: A list of detected Marker objects with ID, corner points,
                          and pose information (if pose estimation was successful).
                          
        Raises:
            IncompatibleTypeError: If image is not an array-like object.
            MarkerDetectionError: If marker detection fails.
        '''
        if not is_array(image):
            raise IncompatibleTypeError(
                f"Image must be ArrayLike (numpy or torch), got {type(image)}. "
                f"Please provide a valid image array."
            )
        
        image = convert_numpy(image)
        
        # Detect markers
        try:
            result = self._opencv_detector.detectMarkers(image)
        except Exception as e:
            raise MarkerDetectionError(f"OpenCV marker detection failed: {e}") from e
        
        # If no markers found, return empty list
        if result is not None:
            corners, ids, _ = result
        else:
            return []
        if ids is None:
            return []

        detected_markers = []

        # For each detected marker
        for marker_corners, marker_id in zip(corners, ids):
            # Estimate pose using solvePnP Algorithm
            corner_2d = marker_corners[0]  # (1,4,2) -> (4,2)
            # Convert marker_id from numpy array to int
            marker_id_int = int(marker_id[0]) if isinstance(marker_id, np.ndarray) else int(marker_id)

            if estimate_pose:
                # SolvePnP
                try:
                    marker2cam = solve_pnp(
                        swapaxes(corner_2d, 0, 1), swapaxes(self.corner_3d, 0, 1), self._cam
                    )
                    if marker2cam is None:
                        continue
                    # Build Marker object
                    marker = Marker(marker_id_int, marker2cam, corner_2d)
                except Exception:
                    # If pose estimation fails, skip this marker
                    continue
            else:
                marker = Marker(id=marker_id_int, corners=corner_2d)
            detected_markers.append(marker)
        return detected_markers