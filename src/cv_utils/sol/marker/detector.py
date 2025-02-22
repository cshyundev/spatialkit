"""
Module Name: detector.py

Description:
    This module defines a generic MarkerDetector base class that can be
    inherited for detecting various fiducial markers (e.g., ArUco, AprilTag, STag).
    It also declares an enumeration FiducialMarkerType that represents
    different types of marker dictionaries.

    This module makes use of the AprilTag detection algorithm, which is 
    developed by The University of Michigan and Matt Zucker. AprilTag is 
    licensed under a BSD-style license. See the LICENSE file for details.

Main Classes:
    - FiducialMarkerType (Enum): Enumerates various marker dictionary types.
    - MarkerDetector (class): Provides an abstract interface for detecting markers
      and estimating their 6D poses in the camera frame.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: v0.2.1
License: MIT License
"""

from typing import *

import numpy as np
from dt_apriltags import Detection, Detector
import stag
import cv2 as cv

from ...geom import Camera, Transform
from ...geom import solve_pnp
from ...ops.uops import ArrayLike, convert_numpy, is_array, swapaxes
from .marker import Marker, FiducialMarkerType
from ...utils.vis import draw_polygon
from ...common.logger import LOG_INFO


# Abstract Fiducial Marker Detector
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
        """
        # Arguments Check
        assert isinstance(cam, Camera), f"cam must be Camera type, but got {type(cam)}."
        assert isinstance(
            marker_size, float
        ), f"marker_size must be float type, but got {type(marker_size)}."
        assert isinstance(
            marker_type, FiducialMarkerType
        ), f"marker_type must be FiducialMarkerType, but got {marker_type}."
        assert marker_size > 0.0, "marker size must be positive."

        self._cam: Camera = cam
        self._marker_size: float = marker_size
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

        Note:
            - This is an abstract method; it must be implemented in a subclass.
            - The typical flow involves:
                1) Marker detection (finding corner points and IDs).
                2) Pose estimation using solvePnP or a similar approach.
                3) Creating Marker instances with the results.
        """
        raise NotImplementedError

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
        """
        assert is_array(image), f"image must be ArrayLike, but got {type(image)}."
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
        """
        Initializes the OpenCVMarkerDetector with a camera, marker size, and OpenCV dictionary type.

        Args:
            cam (Camera): A camera object containing intrinsic/extrinsic parameters.
            marker_size (float): The side length of the marker in meters.
            marker_type (FiducialMarkerType): The specific OpenCV dictionary type to use.

        Raises:
            ValueError: If the provided marker_type is not recognized as a valid OpenCV type.
        """
        super(OpenCVMarkerDetector, self).__init__(cam, marker_size, marker_type)

        # Check if the provided marker_type is a valid dictionary
        if marker_type not in self._VALID_OPENCV_TYPES_MAP:
            raise ValueError(
                f"{marker_type} is not a valid dictionary type opencv supported."
            )

        _opencv_dict = cv.aruco.getPredefinedDictionary(
            self._VALID_OPENCV_TYPES_MAP[marker_type]
        )
        _opencv_detect_params = cv.aruco.DetectorParameters()
        self._opencv_detector = cv.aruco.ArucoDetector(
            _opencv_dict, _opencv_detect_params
        )

    def detect_marker(
        self, image: ArrayLike, estimate_pose: bool = True
    ) -> List[Marker]:

        assert is_array(
            image
        ), f"image must be ArrayLike(Numpy or Torch), but got {type(image)}."
        image = convert_numpy(image)
        # Detect markers
        result = self._opencv_detector.detectMarkers(image)
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

            if estimate_pose:
                # SolvePnP
                marker2cam: Transform = solve_pnp(
                    swapaxes(corner_2d, 0, 1), swapaxes(self.corner_3d, 0, 1), self._cam
                )
                if marker2cam is None:
                    continue
                # Build Marker object
                marker = Marker(marker_id, marker2cam, corner_2d)
            else:
                marker = Marker(id=marker_id, corners=corner_2d)
            detected_markers.append(marker)
        return detected_markers


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
        FiducialMarkerType.APRILTAG_16H5: cv.aruco.DICT_APRILTAG_16h5,
        FiducialMarkerType.APRILTAG_25H9: cv.aruco.DICT_APRILTAG_25h9,
        FiducialMarkerType.APRILTAG_36H10: cv.aruco.DICT_APRILTAG_36h10,
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
            ValueError: If the marker_type is not supported for AprilTag detection.
        """
        super().__init__(cam, marker_size, marker_type)

        if marker_type in self._DT_SUPPORT_APRILTAG_TYPES_STR:
            families = self._DT_SUPPORT_APRILTAG_TYPES_STR[marker_type]
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
        elif marker_type in self._OPENCV_SUPPORT_APRILTAG_TYPES:
            _opencv_dict = cv.aruco.getPredefinedDictionary(
                self._OPENCV_SUPPORT_APRILTAG_TYPES[marker_type]
            )
            _opencv_detect_params = cv.aruco.DetectorParameters()
            self._apriltag_detector = cv.aruco.ArucoDetector(
                _opencv_dict, _opencv_detect_params
            )
            self._use_opencv_deatector = True
            LOG_INFO("This MarkerType is processed to OpenCV.")
        else:
            raise ValueError(f"{marker_type} is not a AprilTag Type.")

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
        """
        assert is_array(
            image
        ), f"image must be ArrayLike(Numpy or Torch), but got {type(image)}."
        image = convert_numpy(image)

        if self._use_opencv_deatector:
            corners, ids, _ = self._apriltag_detector(image)
            if ids is None:
                return []
        else:
            # Convert image to grayscale if needed
            if image.ndim == 3:
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            detections: List[Detection] = self._apriltag_detector.detect(
                image, estimate_tag_pose=False
            )

            if len(detections) == 0:
                return []
            corners = []
            ids = []
            for detection in detections:
                corners.append(
                    detection.corners[::-1]
                )  # Reverse the order of corners. See MarkerDetector Abstract Class note.
                ids.append(detection.tag_id)

        detected_markers = []

        for marker_corners, marker_id in zip(corners, ids):
            corner_2d = (
                marker_corners[0] if self._use_opencv_deatector else marker_corners
            )

            if estimate_pose:
                # SolvePnP
                marker2cam: Transform = solve_pnp(
                    swapaxes(corner_2d, 0, 1), swapaxes(self.corner_3d, 0, 1), self._cam
                )
                if marker2cam is None:
                    continue
                # Build Marker object
                marker = Marker(marker_id, marker2cam, corner_2d)
            else:
                marker = Marker(id=marker_id, corners=corner_2d)
            detected_markers.append(marker)

        return detected_markers


class STagMarkerDetector(MarkerDetector):
    """
    A marker detector for detecting STag fiducial markers.

    This class uses the STag library to detect STag markers and estimates their poses
    in the camera coordinate system using a PnP algorithm.
    """

    _VALID_APRILTAG_TYPES_HD = {
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
            ValueError: If the provided marker_type is not supported by STag.
        """
        super(STagMarkerDetector, self).__init__(cam, marker_size, marker_type)

        # Check if the provided marker_type is a valid dictionary for STag
        if marker_type not in self._VALID_APRILTAG_TYPES_HD:
            raise ValueError(
                f"{marker_type} is not a valid dictionary type stag supported."
            )

        self._stag_hd = self._VALID_APRILTAG_TYPES_HD[marker_type]

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
        """
        assert is_array(
            image
        ), f"image must be ArrayLike(Numpy or Torch), but got {type(image)}."

        image = convert_numpy(image)
        # Detect markers using STag
        (corners, ids, _) = stag.detectMarkers(image, self._stag_hd)
        # If no markers found, return an empty list
        if ids is None:
            return []

        detected_markers = []

        # For each detected marker
        for marker_corners, marker_id in zip(corners, ids):
            # Extract the 2D corner coordinates (shape: (4,2))
            corner_2d = marker_corners[0]  # (1,4,2) -> (4,2)

            if estimate_pose:
                # Estimate pose using solvePnP
                marker2cam: Transform = solve_pnp(
                    swapaxes(corner_2d, 0, 1), swapaxes(self.corner_3d, 0, 1), self._cam
                )
                if marker2cam is None:
                    continue
                # Create Marker object with pose
                marker = Marker(marker_id, marker2cam, corner_2d)
            else:
                marker = Marker(id=marker_id, corners=corner_2d)
            detected_markers.append(marker)
        return detected_markers
