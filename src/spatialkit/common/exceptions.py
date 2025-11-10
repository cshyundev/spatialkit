"""
Module Name: exceptions.py

Description:
This module defines a hierarchical exception structure for the spatialkit library.
All custom exceptions inherit from SpatialKitError to enable unified error handling
while providing specific error types for different domains and operations.

Exception Hierarchy:
    SpatialKitError
    ├── MathError
    │   ├── InvalidDimensionError
    │   ├── InvalidShapeError
    │   ├── IncompatibleTypeError
    │   ├── NumericalError
    │   └── SingularMatrixError
    ├── GeometryError
    │   ├── ConversionError
    │   ├── InvalidCoordinateError
    │   ├── ProjectionError
    │   └── CalibrationError
    ├── CameraError
    │   ├── InvalidCameraParameterError
    │   ├── UnsupportedCameraTypeError
    │   └── CameraModelError
    ├── VisualizationError
    │   ├── RenderingError
    │   └── DisplayError
    ├── IOError
    │   ├── FileNotFoundError
    │   ├── FileFormatError
    │   └── ReadWriteError
    └── MarkerError
        ├── MarkerDetectionError
        └── InvalidMarkerTypeError

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.2
License: MIT LICENSE

Usage:
    >>> from spatialkit.exceptions import InvalidDimensionError, MathError
    >>> 
    >>> # Specific exception handling
    >>> try:
    ...     # some math operation
    ... except InvalidDimensionError as e:
    ...     print(f"Dimension error: {e}")
    >>> 
    >>> # Category-level exception handling
    >>> try:
    ...     # some math operations
    ... except MathError as e:
    ...     print(f"Math operation failed: {e}")
    >>> 
    >>> # Library-level exception handling  
    >>> try:
    ...     # any spatialkit operation
    ... except SpatialKitError as e:
    ...     print(f"spatialkit error: {e}")
"""


# =============================================================================
# Base Exception
# =============================================================================

class SpatialKitError(Exception):
    """
    Base exception for all spatialkit library errors.
    
    This is the root exception from which all other spatialkit exceptions inherit.
    It allows users to catch any error originating from the spatialkit library.
    """
    pass

class NotArrayLikeError(SpatialKitError):
    """
    Exception raised when an input is expected to be array-like but is not.
    
    Examples:
        - Providing a scalar where an array is required
        - Passing unsupported types (e.g., string, dict) to functions expecting arrays
    """
    pass

class InvalidArgumentError(SpatialKitError):
    """
    Exception raised when a function receives invalid arguments.
    
    Examples:
        - Arguments out of expected range
        - Invalid combinations of parameters
        - Missing required arguments
    """
    pass

# =============================================================================
# Mathematical Operations Exceptions
# =============================================================================

class MathError(SpatialKitError):
    """
    Base exception for mathematical operations.
    
    Raised when mathematical computations fail due to invalid inputs,
    numerical issues, or algorithmic limitations.
    """
    pass


class InvalidDimensionError(MathError):
    """
    Exception raised when an operation receives input with invalid dimensions.
    
    Examples:
        - Providing 1D array to matrix operations requiring 2D
        - Specifying dimension index that exceeds array dimensions
        - Matrix operations with incompatible dimensional requirements
    """
    pass


class InvalidShapeError(MathError):
    """
    Exception raised when operations receive input with incompatible shapes.
    
    Examples:
        - Matrix multiplication with mismatched dimensions
        - Broadcasting incompatible arrays
        - Operations requiring square matrices receiving non-square input
    """
    pass


class IncompatibleTypeError(MathError):
    """
    Exception raised when operations receive incompatible data types.
    
    Examples:
        - Mixing numpy arrays and torch tensors in operations requiring same type
        - Providing unsupported array-like objects
        - Type mismatches in unified math operations
    """
    pass


class NumericalError(MathError):
    """
    Exception raised for numerical computation issues.
    
    Examples:
        - Division by zero
        - Overflow/underflow in calculations
        - Convergence failures in iterative algorithms
        - Invalid mathematical operations (e.g., sqrt of negative)
    """
    pass


class SingularMatrixError(MathError):
    """
    Exception raised when operations encounter singular (non-invertible) matrices.
    
    Examples:
        - Matrix inversion of singular matrices
        - Linear system solving with rank-deficient matrices
        - Decompositions failing due to singularity
    """
    pass


# =============================================================================
# Geometry Operations Exceptions
# =============================================================================

class GeometryError(SpatialKitError):
    """
    Base exception for geometric operations.
    
    Raised when 3D geometry computations, transformations, or 
    coordinate operations fail.
    """
    pass


class ConversionError(GeometryError):
    """
    Exception raised for coordinate conversion failures.
    
    Examples:
        - Depth map to point cloud conversion errors
        - Point cloud to depth map projection failures
        - Invalid transformations between coordinate systems
    """
    pass


class InvalidCoordinateError(GeometryError):
    """
    Exception raised for invalid coordinate operations.
    
    Examples:
        - Homogeneous coordinates with wrong dimensions
        - Invalid coordinate ranges or values
        - Coordinate system mismatches
    """
    pass


class ProjectionError(GeometryError):
    """
    Exception raised for projection-related failures.
    
    Examples:
        - Points behind camera in projection
        - Invalid projection parameters
        - Projection onto invalid image planes
    """
    pass


class CalibrationError(GeometryError):
    """
    Exception raised for camera calibration related errors.
    
    Examples:
        - Invalid calibration parameters
        - Calibration computation failures
        - Inconsistent calibration data
    """
    pass


# =============================================================================
# Camera Operations Exceptions
# =============================================================================

class CameraError(SpatialKitError):
    """
    Base exception for camera-related operations.
    
    Raised when camera model operations, parameter validation,
    or camera-specific computations fail.
    """
    pass


class InvalidCameraParameterError(CameraError):
    """
    Exception raised for invalid camera parameters.
    
    Examples:
        - Negative focal lengths
        - Invalid distortion coefficients
        - Inconsistent image dimensions
        - Out-of-range principal point coordinates
    """
    pass


class UnsupportedCameraTypeError(CameraError):
    """
    Exception raised for unsupported camera types or models.
    
    Examples:
        - Requesting operations not supported by specific camera model
        - Using unsupported camera model parameters
        - Incompatible camera model combinations
    """
    pass


class CameraModelError(CameraError):
    """
    Exception raised for camera model computation failures.
    
    Examples:
        - Camera model creation failures
        - Invalid model state or configuration
        - Model-specific computation errors
    """
    pass


# =============================================================================
# Visualization Exceptions
# =============================================================================

class VisualizationError(SpatialKitError):
    """
    Base exception for visualization operations.
    
    Raised when rendering, display, or visualization operations fail.
    """
    pass


class RenderingError(VisualizationError):
    """
    Exception raised for rendering failures.
    
    Examples:
        - 3D rendering failures
        - Point cloud visualization errors
        - Mesh rendering issues
    """
    pass


class DisplayError(VisualizationError):
    """
    Exception raised for display-related failures.
    
    Examples:
        - Window creation failures
        - Display device issues
        - GUI interaction errors
    """
    pass


# =============================================================================
# I/O Operations Exceptions
# =============================================================================

class IOError(SpatialKitError):
    """
    Base exception for input/output operations.
    
    Raised when file operations, data loading/saving, or 
    format conversions fail.
    """
    pass


class FileNotFoundError(IOError):
    """
    Exception raised when requested files cannot be found.
    
    Examples:
        - Missing image files
        - Missing calibration files
        - Missing dataset files
    """
    pass


class FileFormatError(IOError):
    """
    Exception raised for unsupported or invalid file formats.
    
    Examples:
        - Unsupported image formats
        - Invalid file headers
        - Corrupted file data
    """
    pass


class ReadWriteError(IOError):
    """
    Exception raised for file read/write operation failures.
    
    Examples:
        - Permission denied errors
        - Disk space issues
        - Network failures for remote files
    """
    pass


# =============================================================================
# Marker Detection Exceptions
# =============================================================================

class MarkerError(SpatialKitError):
    """
    Base exception for fiducial marker operations.
    
    Raised when marker detection, pose estimation, or 
    marker-related computations fail.
    """
    pass


class MarkerDetectionError(MarkerError):
    """
    Exception raised for marker detection failures.
    
    Examples:
        - No markers found in image
        - Detection algorithm failures
        - Invalid detection parameters
    """
    pass


class InvalidMarkerTypeError(MarkerError):
    """
    Exception raised for invalid or unsupported marker types.
    
    Examples:
        - Unsupported marker family
        - Invalid marker configuration
        - Incompatible marker parameters
    """
    pass


# =============================================================================
# Exception Utilities
# =============================================================================

def get_exception_hierarchy() -> dict:
    """
    Get the complete exception hierarchy as a nested dictionary.
    
    Returns:
        dict: Nested dictionary representing the exception hierarchy.
        
    Example:
        >>> hierarchy = get_exception_hierarchy()
        >>> print(hierarchy['SpatialKitError']['MathError'])
        ['InvalidDimensionError', 'InvalidShapeError', ...]
    """
    return {
        'SpatialKitError': {
            'MathError': [
                'InvalidDimensionError',
                'InvalidShapeError', 
                'IncompatibleTypeError',
                'NumericalError',
                'SingularMatrixError'
            ],
            'GeometryError': [
                'ConversionError',
                'InvalidCoordinateError',
                'ProjectionError',
                'CalibrationError'
            ],
            'CameraError': [
                'InvalidCameraParameterError',
                'UnsupportedCameraTypeError',
                'CameraModelError'
            ],
            'VisualizationError': [
                'RenderingError',
                'DisplayError'
            ],
            'IOError': [
                'FileNotFoundError',
                'FileFormatError', 
                'ReadWriteError'
            ],
            'MarkerError': [
                'MarkerDetectionError',
                'InvalidMarkerTypeError'
            ]
        }
    }


def is_spatialkit_error(exception: Exception) -> bool:
    """
    Check if an exception is a spatialkit library exception.
    
    Args:
        exception (Exception): Exception to check.
        
    Returns:
        bool: True if exception is from spatialkit, False otherwise.
    """
    return isinstance(exception, SpatialKitError)


# Export all exceptions for convenient importing
__all__ = [
    # Base
    'SpatialKitError',
    'NotArrayLikeError',
    'InvalidArgumentError',
    
    # Math
    'MathError',
    'InvalidDimensionError',
    'InvalidShapeError',
    'IncompatibleTypeError', 
    'NumericalError',
    'SingularMatrixError',
    
    # Geometry
    'GeometryError',
    'ConversionError',
    'InvalidCoordinateError',
    'ProjectionError',
    'CalibrationError',
    
    # Camera
    'CameraError',
    'InvalidCameraParameterError',
    'UnsupportedCameraTypeError',
    'CameraModelError',
    
    # Visualization
    'VisualizationError',
    'RenderingError',
    'DisplayError',
    
    # I/O
    'IOError',
    'FileNotFoundError',
    'FileFormatError',
    'ReadWriteError',
    
    # Marker
    'MarkerError',
    'MarkerDetectionError',
    'InvalidMarkerTypeError',

    # Utilities
    'get_exception_hierarchy',
    'is_spatialkit_error'
]
