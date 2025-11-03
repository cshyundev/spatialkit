"""
Module Name: stag_detector.py

Description:
    This module defines a detector for STag markers.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: v0.3.0
License: MIT License
"""
from typing import List
import numpy as np
import stag

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


class STagMarkerDetector(MarkerDetector):
    """
    A marker detector for detecting STag fiducial markers.

    This class uses the STag library to detect STag markers and estimates their poses
    in the camera coordinate system using a PnP algorithm.
    """

    _VALID_STAG_TYPES_HD = {
        FiducialMarkerType.STAG_HD11: 11,
        FiducialMarkerType.STAG_HD13: 13,
        FiducialMarkerType.STAG_HD15: 15,
        FiducialMarkerType.STAG_HD17: 17,
        FiducialMarkerType.STAG_HD19: 19,
        FiducialMarkerType.STAG_HD21: 21,
        FiducialMarkerType.STAG_HD23: 23,
    }

    def __init__(
        self,
        cam: Camera,
        marker_size: float,
        marker_type: FiducialMarkerType = FiducialMarkerType.NONE,
    ):
        """
        Initializes the STagMarkerDetector with a camera, marker size, and STag dictionary type.

        Args:
            cam (Camera): A camera object containing intrinsic/extrinsic parameters.
            marker_size (float): The side length of the marker in meters.
            marker_type (FiducialMarkerType): The specific STag dictionary type to use.

        Raises:
            InvalidMarkerTypeError: If the provided marker_type is not supported by STag.
        """
        super(STagMarkerDetector, self).__init__(cam, marker_size, marker_type)

        # Check if the provided marker_type is a valid dictionary for STag
        if marker_type not in self._VALID_STAG_TYPES_HD:
            raise InvalidMarkerTypeError(
                f"Marker type {marker_type} is not supported by STag detector. "
                f"Supported types: {list(self._VALID_STAG_TYPES_HD.keys())}. "
                f"Please use a valid STag marker dictionary type."
            )

        self._stag_hd = self._VALID_STAG_TYPES_HD[marker_type]

    def detect_marker(
        self, image: ArrayLike, estimate_pose: bool = True
    ) -> List[Marker]:
        """
        Detects STag markers in the input image and returns a list of Marker objects.

        Args:
            image (ArrayLike, [H,W,3] or [H,W]): An input image (RGB or grayscale).
            estimate_pose (bool): If True, estimate the marker's pose using the PnP algorithm.

        Returns:
            List[Marker]: A list of Marker objects with detected marker IDs, corner points,
                          and pose information (if pose estimation succeeded).
                          
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
        
        # Detect markers using STag
        try:
            (corners, ids, _) = stag.detectMarkers(image, self._stag_hd)
        except Exception as e:
            raise MarkerDetectionError(f"STag marker detection failed: {e}") from e
        
        # If no markers found, return an empty list
        if ids is None:
            return []

        detected_markers = []

        # For each detected marker
        for marker_corners, marker_id in zip(corners, ids):
            # Extract the 2D corner coordinates (shape: (4,2))
            corner_2d = marker_corners[0]  # (1,4,2) -> (4,2)

            # Convert marker_id from numpy array to int
            marker_id_int = int(marker_id[0]) if isinstance(marker_id, np.ndarray) else int(marker_id)

            if estimate_pose:
                # Estimate pose using solvePnP
                try:
                    marker2cam = solve_pnp(
                        swapaxes(corner_2d, 0, 1), swapaxes(self.corner_3d, 0, 1), self._cam
                    )
                    if marker2cam is None:
                        continue
                    # Create Marker object with pose
                    marker = Marker(marker_id_int, marker2cam, corner_2d)
                except Exception:
                    # If pose estimation fails, skip this marker
                    continue
            else:
                marker = Marker(id=marker_id_int, corners=corner_2d)
            detected_markers.append(marker)
        return detected_markers