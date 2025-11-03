"""
Module Name: apriltag_detector.py

Description:
    This module defines a detector for AprilTag markers.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: v0.3.0
License: MIT License
"""
from typing import List
import numpy as np
from dt_apriltags import Detection, Detector
import cv2 as cv

from ..camera import Camera
from ..geom import solve_pnp
from ..ops.uops import ArrayLike, convert_numpy, is_array, swapaxes
from .marker import Marker, FiducialMarkerType
from ..common.logger import LOG_INFO
from ..common.exceptions import (
    InvalidMarkerTypeError,
    MarkerDetectionError,
    IncompatibleTypeError
)
from .base import MarkerDetector


class AprilTagMarkerDetector(MarkerDetector):
    """
    A marker detector for detecting AprilTag fiducial markers.

    This class supports two detection backends:
      - dt_apriltags library for native AprilTag detection.
      - OpenCV-based detection (if the marker type is supported by OpenCV).

    Detector parameters such as nthreads, quad_decimate, etc., can be configured during initialization.
    """

    _DT_SUPPORT_APRILTAG_TYPES_STR = {
        FiducialMarkerType.APRILTAG_36H11: "tag36h11",
        FiducialMarkerType.APRILTAG_CUSTOM48H12: "tagCustom48h12",
        FiducialMarkerType.APRILTAG_STANDARD41H12: "tagStandard41h12",
        FiducialMarkerType.APRILTAG_STANDARD52H13: "tagStandard52h13",
    }

    _OPENCV_SUPPORT_APRILTAG_TYPES = {
        FiducialMarkerType.APRILTAG_16H5: cv.aruco.DICT_APRILTAG_16H5,
        FiducialMarkerType.APRILTAG_25H9: cv.aruco.DICT_APRILTAG_25H9,
        FiducialMarkerType.APRILTAG_36H10: cv.aruco.DICT_APRILTAG_36H10,
    }

    def __init__(
        self,
        cam: Camera,
        marker_size: float,
        marker_type: FiducialMarkerType = FiducialMarkerType.NONE,
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
    ):
        """
        Initializes the AprilTagMarkerDetector with a camera, marker size, and detector parameters.

        Args:
            cam (Camera): A camera object containing intrinsic/extrinsic parameters.
            marker_size (float): The side length of the marker in meters.
            marker_type (FiducialMarkerType): The specific AprilTag type to use.
            nthreads (int): Number of threads to use for detection.
            quad_decimate (float): Decimation factor for quad detection.
            quad_sigma (float): Gaussian blur sigma for quad detection.
            refine_edges (int): Whether to refine the detected edges (1 for yes, 0 for no).
            decode_sharpening (float): Sharpening factor for tag decoding.

        Raises:
            InvalidMarkerTypeError: If the marker_type is not supported for AprilTag detection.
            MarkerDetectionError: If detector initialization fails.
        """
        super().__init__(cam, marker_size, marker_type)

        if marker_type in self._DT_SUPPORT_APRILTAG_TYPES_STR:
            families = self._DT_SUPPORT_APRILTAG_TYPES_STR[marker_type]
            try:
                self._apriltag_detector = Detector(
                    searchpath=["apriltags"],
                    families=families,
                    nthreads=nthreads,
                    quad_decimate=quad_decimate,
                    quad_sigma=quad_sigma,
                    refine_edges=refine_edges,
                    decode_sharpening=decode_sharpening,
                )
                self._use_opencv_deatector = False
            except Exception as e:
                raise MarkerDetectionError(
                    f"Failed to initialize dt_apriltags detector: {e}. "
                    f"Please check your dt_apriltags installation."
                ) from e
        elif marker_type in self._OPENCV_SUPPORT_APRILTAG_TYPES:
            try:
                _opencv_dict = cv.aruco.getPredefinedDictionary(
                    self._OPENCV_SUPPORT_APRILTAG_TYPES[marker_type]
                )
                _opencv_detect_params = cv.aruco.DetectorParameters()
                self._apriltag_detector = cv.aruco.ArucoDetector(
                    _opencv_dict, _opencv_detect_params
                )
                self._use_opencv_deatector = True
                LOG_INFO("This MarkerType is processed to OpenCV.")
            except Exception as e:
                raise MarkerDetectionError(
                    f"Failed to initialize OpenCV AprilTag detector: {e}. "
                    f"Please check your OpenCV installation."
                ) from e
        else:
            raise InvalidMarkerTypeError(
                f"Marker type {marker_type} is not supported by AprilTag detector. "
                f"Supported dt_apriltags types: {list(self._DT_SUPPORT_APRILTAG_TYPES_STR.keys())}. "
                f"Supported OpenCV types: {list(self._OPENCV_SUPPORT_APRILTAG_TYPES.keys())}. "
                f"Please use a valid AprilTag marker type."
            )

    def detect_marker(
        self, image: ArrayLike, estimate_pose: bool = True
    ) -> List[Marker]:
        """
        Detects AprilTag markers in the input image and returns a list of Marker objects.

        This method supports two detection backends:
          - OpenCV-based detection (if marker_type is in _OPENCV_SUPPORT_APRILTAG_TYPES)
          - dt_apriltags library detection

        Args:
            image (ArrayLike, [H,W,3] or [H,W]): An input image (RGB or grayscale).
            estimate_pose (bool): If True, estimate pose using the PnP algorithm.

        Returns:
            List[Marker]: A list of detected Marker objects with ID, corner points,
                          and pose information (if pose estimation was successful).
                          
        Raises:
            IncompatibleTypeError: If image is not an array-like object.
            MarkerDetectionError: If marker detection fails.
        """
        if not is_array(image):
            raise IncompatibleTypeError(
                f"Image must be ArrayLike (numpy or torch), got {type(image)}. "
                f"Please provide a valid image array."
            )
        
        image = convert_numpy(image)

        try:
            if self._use_opencv_deatector:
                corners, ids, _ = self._apriltag_detector.detectMarkers(image)
                if ids is None:
                    return []
            else:
                # Convert image to grayscale if needed
                if image.ndim == 3:
                    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                detections: List[Detection] = self._apriltag_detector.detect(
                    image, estimate_tag_pose=False
                )

                if not detections:
                    return []
                corners = []
                ids = []
                for detection in detections:
                    corners.append(
                        detection.corners[::-1]
                    )  # Reverse the order of corners. See MarkerDetector Abstract Class note.
                    ids.append(detection.tag_id)
        except Exception as e:
            raise MarkerDetectionError(f"AprilTag marker detection failed: {e}") from e

        detected_markers = []

        for marker_corners, marker_id in zip(corners, ids):
            corner_2d = (
                marker_corners[0] if self._use_opencv_deatector else marker_corners
            )

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