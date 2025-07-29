__version__ = "0.3.0-alpha"

# Core modules
from .geom import camera, geom_utils, img_tf, pose, tf, rotation
from .geom.rotation import RotType
from .utils import io, point_selector, vis
from .common.logger import LOG_CRITICAL, LOG_DEBUG, LOG_ERROR, LOG_INFO, LOG_WARN
from .common.constant import *
from .externals import _o3d as vis3d

# Solutions
from .sol import mvs, marker
from .sol.marker.marker import FiducialMarkerType

# Operations
from .ops.uops import *
from .ops.umath import *

# Exception handling
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
