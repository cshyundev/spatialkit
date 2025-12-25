"""
spatialkit: Computer Vision Utilities for Research and Development

A library for computer vision and robotics research, focusing on 3D vision tasks.
Provides unified interfaces for NumPy/PyTorch operations, comprehensive geometry
primitives, camera models, and marker detection capabilities.

Package Structure:
    common: Logging, constants, and exceptions
    ops: Unified NumPy/PyTorch operations
    geom: 3D geometric primitives (rotations, poses, transforms)
    camera: Camera projection models
    imgproc: 2D image processing and transformations
    io: File I/O (images, videos, configs)
    vis2d: 2D image visualization and drawing
    markers: Fiducial marker detection (AprilTag, STag)
    vis3d: 3D visualization with Open3D
"""

__version__ = "0.3.3"

# ==============================================================================
# Core Package Modules (import as modules for organized access)
# ==============================================================================

from . import ops
from .ops import umath, uops

from . import geom
from .geom import rotation, pose, tf, epipolar, multiview, pointcloud

from . import camera
from . import imgproc

from . import io
from . import vis2d

from . import common
from .common import logger, constant

# ==============================================================================
# High-Level Classes (import directly for convenience)
# ==============================================================================

# Geometry primitives
from .geom.rotation import Rotation, RotType
from .geom.pose import Pose
from .geom.tf import Transform

# Camera models
from .camera import (
    Camera,
    CamType,
    PerspectiveCamera,
    OpenCVFisheyeCamera,
    ThinPrismFisheyeCamera,
    OmnidirectionalCamera,
    DoubleSphereCamera,
    EquirectangularCamera,
)

# Markers
from . import markers
from .markers import (
    Marker,
    FiducialMarkerType,
    MarkerDetector,
    OpenCVMarkerDetector,
    AprilTagMarkerDetector,
    STagMarkerDetector,
)

# 3D Visualization
from . import vis3d

# ==============================================================================
# Logging (commonly used)
# ==============================================================================

from .common.logger import LOG_CRITICAL, LOG_DEBUG, LOG_ERROR, LOG_INFO, LOG_WARN

# ==============================================================================
# Exception Hierarchy
# ==============================================================================

from .common.exceptions import (
    # Base exception
    SpatialKitError,
    # Math exceptions
    MathError,
    InvalidDimensionError,
    InvalidShapeError,
    IncompatibleTypeError,
    NumericalError,
    SingularMatrixError,
    # Geometry exceptions
    GeometryError,
    ConversionError,
    InvalidCoordinateError,
    ProjectionError,
    CalibrationError,
    # Camera exceptions
    CameraError,
    InvalidCameraParameterError,
    UnsupportedCameraTypeError,
    CameraModelError,
    # Visualization exceptions
    VisualizationError,
    RenderingError,
    DisplayError,
    # I/O exceptions
    IOError,
    FileNotFoundError,
    FileFormatError,
    ReadWriteError,
    # Marker exceptions
    MarkerError,
    MarkerDetectionError,
    InvalidMarkerTypeError,
)

# ==============================================================================
# Public API Definition
# ==============================================================================

__all__ = [
    # Version
    "__version__",

    # Package modules (for hierarchical access)
    "ops", "umath", "uops",
    "geom", "rotation", "pose", "tf",
    "epipolar", "multiview", "pointcloud",
    "camera", "imgproc",
    "io", "vis2d",
    "common", "logger", "constant",
    "markers",
    "vis3d",

    # Geometry classes (frequently used, direct access)
    "Rotation", "RotType",
    "Pose",
    "Transform",

    # Camera classes
    "Camera", "CamType",
    "PerspectiveCamera",
    "OpenCVFisheyeCamera",
    "ThinPrismFisheyeCamera",
    "OmnidirectionalCamera",
    "DoubleSphereCamera",
    "EquirectangularCamera",

    # Marker classes
    "Marker", "FiducialMarkerType",
    "MarkerDetector",
    "OpenCVMarkerDetector",
    "AprilTagMarkerDetector",
    "STagMarkerDetector",

    # Logging
    "LOG_CRITICAL", "LOG_DEBUG", "LOG_ERROR", "LOG_INFO", "LOG_WARN",

    # Exceptions
    "SpatialKitError",
    "MathError", "InvalidDimensionError", "InvalidShapeError",
    "IncompatibleTypeError", "NumericalError", "SingularMatrixError",
    "GeometryError", "ConversionError", "InvalidCoordinateError",
    "ProjectionError", "CalibrationError",
    "CameraError", "InvalidCameraParameterError",
    "UnsupportedCameraTypeError", "CameraModelError",
    "VisualizationError", "RenderingError", "DisplayError",
    "IOError", "FileNotFoundError", "FileFormatError", "ReadWriteError",
    "MarkerError", "MarkerDetectionError", "InvalidMarkerTypeError",
]
