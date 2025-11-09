"""
Module Name: marker.py

Description:
    This module defines the Marker class for storing and managing
    a marker's ID and its pose relative to the camera.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.2
License: Your License Info

Usage:
    from marker import Marker
    
    marker = Marker()
    marker.id(42)
    print(marker.id)
    print(marker.marker2cam)     # Pose object
"""

from enum import Enum
import numpy as np
from ..geom import Transform
from ..common.exceptions import InvalidMarkerTypeError, InvalidShapeError


class FiducialMarkerType(Enum):
    """
    Enumerates various types of fiducial marker dictionaries.

    Attributes:
        NONE (tuple): No marker type specified.
        ARUCO_4X4_50, ARUCO_4X4_100, ... etc:
            Each entry corresponds to a particular ArUco dictionary or AprilTag variant.
            The first element in the tuple is an internal string identifier,
            and the second element is a descriptive name.
    """

    NONE = ("None", "No Fiducial Marker Type")
    # 4x4 Dict
    ARUCO_4X4_50 = ("ArUco_4X4_50", "ArUco Marker Dict 4X4 50")
    ARUCO_4X4_100 = ("ArUco_4X4_100", "ArUco Marker Dict 4X4 100")
    ARUCO_4X4_250 = ("ArUco_4X4_250", "ArUco Marker Dict 4X4 250")
    ARUCO_4X4_1000 = ("ArUco_4X4_1000", "ArUco Marker Dict 4X4 1000")

    # 5x5 Dict
    ARUCO_5X5_50 = ("ArUco_5X5_50", "ArUco Marker Dict 5X5 50")
    ARUCO_5X5_100 = ("ArUco_5X5_100", "ArUco Marker Dict 5X5 100")
    ARUCO_5X5_250 = ("ArUco_5X5_250", "ArUco Marker Dict 5X5 250")
    ARUCO_5X5_1000 = ("ArUco_5X5_1000", "ArUco Marker Dict 5X5 1000")

    # 6x6 Dict
    ARUCO_6X6_50 = ("ArUco_6X6_50", "ArUco Marker Dict 6X6 50")
    ARUCO_6X6_100 = ("ArUco_6X6_100", "ArUco Marker Dict 6X6 100")
    ARUCO_6X6_250 = ("ArUco_6X6_250", "ArUco Marker Dict 6X6 250")
    ARUCO_6X6_1000 = ("ArUco_6X6_1000", "ArUco Marker Dict 6X6 1000")

    # 7x7 Dict
    ARUCO_7X7_50 = ("ArUco_7X7_50", "ArUco Marker Dict 7X7 50")
    ARUCO_7X7_100 = ("ArUco_7X7_100", "ArUco Marker Dict 7X7 100")
    ARUCO_7X7_250 = ("ArUco_7X7_250", "ArUco Marker Dict 7X7 250")
    ARUCO_7X7_1000 = ("ArUco_7X7_1000", "ArUco Marker Dict 7X7 1000")

    # Original ArUco
    ARUCO_ORIGINAL = ("ArUco_ORIGINAL", "ArUco Marker Dict ORIGINAL")

    # AprilTag variations
    APRILTAG_16H5 = ("AprilTag_16h5", "AprilTag 16h5")
    APRILTAG_25H9 = ("AprilTag_25h9", "AprilTag 25h9")
    APRILTAG_36H10 = ("AprilTag_36h10", "AprilTag 36h10")
    APRILTAG_36H11 = ("AprilTag_36h11", "AprilTag 36h11")

    # Apriltag3
    # APRILTAG_CIRCLE21H7 = ("AprilTag_tagCircle21h7", "AprilTag Circle21h7")
    # APRILTAG_CIRCLE49H12 = ("AprilTag_tagCircle49h12", "AprilTag Circle49h12")
    APRILTAG_CUSTOM48H12 = ("AprilTag_tagCustom48h12", "AprilTag Custom48h12")
    APRILTAG_STANDARD41H12 = ("AprilTag_tagStandard41h12", "AprilTag Standard41h12")
    APRILTAG_STANDARD52H13 = ("AprilTag_tagStandard52h13", "AprilTag Standard52h13")

    # MIP 36h12
    ARUCO_MIP_36H12 = ("ArUco_MIP_36h12", "ArUco MIP 36h12")

    # STag
    STAG_HD11 = ("STag_LibrayHD11", "STag LibrayHD11")  # Size = 22309
    STAG_HD13 = ("STag_LibrayHD13", "STag LibrayHD13")  # Size = 2884
    STAG_HD15 = ("STag_LibrayHD15", "STag LibrayHD15")  # Size = 766
    STAG_HD17 = ("STag_LibrayHD17", "STag LibrayHD17")  # Size = 157
    STAG_HD19 = ("STag_LibrayHD19", "STag LibrayHD19")  # Size = 38
    STAG_HD21 = ("STag_LibrayHD21", "STag LibrayHD21")  # Size = 12
    STAG_HD23 = ("STag_LibrayHD23", "STag LibrayHD23")  # Size = 6


class Marker:
    """
    A class for storing marker ID and its relative pose to the camera.

    Attributes:
        _id (int): Internal marker ID storage. Defaults to -1.
        _marker2cam (Transform): Transform object representing the marker's
            transform relative to the camera. Defaults to Transform().
        _corners (np.ndarray, [4,2]): 2D Marker's Corner Point.
    """

    def __init__(
        self,
        id: int = -1,
        marker2cam: Transform = Transform(),
        corners: np.ndarray = np.zeros(shape=(4, 2)),
    ):
        """
        Initializes the Marker instance.
        
        Args:
            id (int): Marker ID. Defaults to -1.
            marker2cam (Transform): Transform from marker to camera coordinates.
            corners (np.ndarray): 2D corner points with shape (4, 2).
            
        Raises:
            InvalidShapeError: If corners array doesn't have the correct shape (4, 2).
        """
        self.id = id  # Use setter for validation
        self.marker2cam = marker2cam  # Use setter for validation
        self.corners = corners  # Use setter for validation

    @property
    def id(self):
        """
        Gets the marker's ID.

        Returns:
            int: The current marker ID.
        """
        return self._id

    @id.setter
    def id(self, new_id: int):
        """
        Sets the marker's ID.

        Args:
            new_id (int): The ID to be assigned to this marker.
            
        Raises:
            InvalidMarkerTypeError: If ID is not a valid integer or is negative.
        """
        if not isinstance(new_id, int):
            raise InvalidMarkerTypeError(
                f"Marker ID must be an integer, got {type(new_id)}. "
                f"Please provide a valid integer ID."
            )
        if new_id < -1:
            raise InvalidMarkerTypeError(
                f"Marker ID must be -1 (uninitialized) or positive, got {new_id}. "
                f"Please provide a valid marker ID."
            )
        self._id = new_id

    @property
    def marker2cam(self):
        """
        Gets the transform of this marker relative to the camera.

        Returns:
            Transform: The marker's transform in camera coordinates.
        """
        return self._marker2cam

    @marker2cam.setter
    def marker2cam(self, new_pose: Transform):
        """
        Sets the transform of this marker relative to the camera.

        Args:
            new_pose (Transform): The new transform of this marker in camera coordinates.
            
        Raises:
            InvalidMarkerTypeError: If new_pose is not a Transform object.
        """
        if not isinstance(new_pose, Transform):
            raise InvalidMarkerTypeError(
                f"Marker pose must be a Transform object, got {type(new_pose)}. "
                f"Please provide a valid Transform instance."
            )
        self._marker2cam = new_pose

    @property
    def corners(self):
        """
        Gets the 2D corner points of the marker in the image.

        Returns:
            np.ndarray: A (4, 2) array representing the pixel coordinates
                        of each corner in the order [corner0, corner1, corner2, corner3].

        Details:
            - The default ordering convention can be (top-left -> top-right ->
              bottom-right -> bottom-left) if consistent with your pipeline.
            - This ordering is crucial for pose estimation, as each corner corresponds
              to a specific 3D point on the marker.
        """
        return self._corners

    @corners.setter
    def corners(self, new_corners: np.ndarray):
        """
        Sets the 2D corner points of the marker in the image.

        Args:
            new_corners (np.ndarray): A (4, 2) array representing the pixel coordinates
                                      of each corner.

        Raises:
            InvalidShapeError: If corners array doesn't have the correct shape (4, 2).

        Details:
            - Ensure the shape is (4, 2) and the ordering matches your pose estimation
              pipeline (e.g., clockwise or counter-clockwise).
            - The recommended convention is top-left -> top-right -> bottom-right ->
              bottom-left for clarity.
        """
        if not isinstance(new_corners, np.ndarray):
            raise InvalidShapeError(
                f"Marker corners must be a numpy array, got {type(new_corners)}. "
                f"Please provide a valid numpy array with shape (4, 2)."
            )
        
        if new_corners.shape != (4, 2):
            raise InvalidShapeError(
                f"Marker corners must have shape (4, 2), got {new_corners.shape}. "
                f"Please provide exactly 4 corner points with 2D coordinates (x, y)."
            )
        
        self._corners = new_corners
