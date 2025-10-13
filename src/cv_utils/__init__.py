"""
cv_utils: Computer Vision Utilities for Research and Development

A library for computer vision and robotics research, focusing on 3D vision tasks.
Provides unified interfaces for NumPy/PyTorch operations, comprehensive geometry
primitives, camera models, and solutions for marker detection and multi-view stereo.

Package Structure:
    ops: Unified NumPy/PyTorch operations
    geom: Geometric primitives (rotations, poses, cameras)
    utils: I/O, visualization, and utilities
    sol: Domain-specific solutions (markers, MVS)
    common: Logging and constants
    exceptions: Hierarchical exception system
"""

__version__ = "0.3.0-alpha"

# ==============================================================================
# Core Package Modules (import as modules for organized access)
# ==============================================================================

from . import ops
from .ops import umath, uops

from . import geom
from .geom import camera, rotation, pose, tf, img_tf, geom_utils

from . import utils
from .utils import io, vis, point_selector

from . import common
from .common import logger, constant

# ==============================================================================
# High-Level Classes (import directly for convenience)
# ==============================================================================

# Geometry primitives
from .geom.rotation import Rotation, RotType
from .geom.pose import Pose
from .geom.tf import Transform
from .geom.camera import (
    Camera,
    CamType,
    PerspectiveCamera,
    OpenCVFisheyeCamera,
    ThinPrismFisheyeCamera,
    OmnidirectionalCamera,
    DoubleSphereCamera,
    EquirectangularCamera,
)

# Solutions
from . import sol
from .sol import marker, mvs
from .sol.marker import (
    Marker,
    FiducialMarkerType,
    MarkerDetector,
    OpenCVMarkerDetector,
    AprilTagMarkerDetector,
    STagMarkerDetector,
)
from .sol.mvs import ScannetV1Manager

# Visualization (external)
from .externals import _o3d as vis3d

# ==============================================================================
# Logging (commonly used)
# ==============================================================================

from .common.logger import LOG_CRITICAL, LOG_DEBUG, LOG_ERROR, LOG_INFO, LOG_WARN

# ==============================================================================
# Exception Hierarchy
# ==============================================================================

from .exceptions import (
    # Base exception
    CVUtilsError,
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
    # MVS exceptions
    MVSError,
    DatasetError,
    ReconstructionError,
)

# ==============================================================================
# Public API Definition
# ==============================================================================

__all__ = [
    # Version
    "__version__",

    # Package modules (for hierarchical access)
    "ops", "umath", "uops",
    "geom", "camera", "rotation", "pose", "tf", "img_tf", "geom_utils",
    "utils", "io", "vis", "point_selector",
    "common", "logger", "constant",
    "sol", "marker", "mvs",
    "vis3d",

    # Geometry classes (frequently used, direct access)
    "Rotation", "RotType",
    "Pose",
    "Transform",
    "Camera", "CamType",
    "PerspectiveCamera",
    "OpenCVFisheyeCamera",
    "ThinPrismFisheyeCamera",
    "OmnidirectionalCamera",
    "DoubleSphereCamera",
    "EquirectangularCamera",

    # Solution classes
    "Marker", "FiducialMarkerType",
    "MarkerDetector",
    "OpenCVMarkerDetector",
    "AprilTagMarkerDetector",
    "STagMarkerDetector",
    "ScannetV1Manager",

    # Logging
    "LOG_CRITICAL", "LOG_DEBUG", "LOG_ERROR", "LOG_INFO", "LOG_WARN",

    # Exceptions
    "CVUtilsError",
    "MathError", "InvalidDimensionError", "InvalidShapeError",
    "IncompatibleTypeError", "NumericalError", "SingularMatrixError",
    "GeometryError", "ConversionError", "InvalidCoordinateError",
    "ProjectionError", "CalibrationError",
    "CameraError", "InvalidCameraParameterError",
    "UnsupportedCameraTypeError", "CameraModelError",
    "VisualizationError", "RenderingError", "DisplayError",
    "IOError", "FileNotFoundError", "FileFormatError", "ReadWriteError",
    "MarkerError", "MarkerDetectionError", "InvalidMarkerTypeError",
    "MVSError", "DatasetError", "ReconstructionError",
]
