# Spatialkit Exception Handling Policy and Guidelines

**Last Updated:** 2025-11-03
**Version:** 1.0.0

This document defines the exception handling policy and usage guidelines for the spatialkit library.

## Exception Handling Policy

### 1. Core Principles

- **Fail Fast**: Detect incorrect inputs or states quickly and raise exceptions immediately
- **Clear Messages**: Provide specific error messages that help users understand and solve problems
- **Hierarchical Structure**: Classify exceptions by domain to enable selective exception handling
- **Consistency**: Use the same exception types for similar situations

### 2. Exception vs Assert Usage Guidelines

| Situation | Use | Reason |
|-----------|-----|--------|
| User input validation | `raise Exception` | Always required in production |
| Type/shape validation | `raise Exception` | Clear feedback for library users |
| Internal logic verification | `assert` (dev only) | Developer assumption checking |
| External dependency errors | `raise Exception` | User-recoverable errors |

### 3. Logging vs Exception Usage Guidelines

| Situation | Use | Example |
|-----------|-----|---------|
| Critical errors | `raise Exception` | Invalid input, computation failure |
| Warning messages | `logger.warning` | Performance degradation, recommendations |
| Informational messages | `logger.info` | Processing complete, progress updates |
| Debug information | `logger.debug` | Internal state, intermediate results |

## Exception Hierarchy

```
SpatialKitError (base exception)
├── MathError (mathematical operations)
│   ├── InvalidDimensionError (dimension errors)
│   ├── InvalidShapeError (shape mismatch)
│   ├── IncompatibleTypeError (type mismatch)
│   ├── NumericalError (numerical computation errors)
│   └── SingularMatrixError (singular matrix)
├── GeometryError (geometric operations)
│   ├── ConversionError (coordinate conversion errors)
│   ├── InvalidCoordinateError (invalid coordinates)
│   ├── ProjectionError (projection errors)
│   └── CalibrationError (calibration errors)
├── CameraError (camera-related)
│   ├── InvalidCameraParameterError (invalid camera parameters)
│   ├── UnsupportedCameraTypeError (unsupported camera type)
│   └── CameraModelError (camera model errors)
├── VisualizationError (visualization)
│   ├── RenderingError (rendering errors)
│   └── DisplayError (display errors)
├── IOError (input/output)
│   ├── FileNotFoundError (file not found)
│   ├── FileFormatError (invalid file format)
│   └── ReadWriteError (read/write errors)
└── MarkerError (marker-related)
    ├── MarkerDetectionError (marker detection failure)
    └── InvalidMarkerTypeError (invalid marker type)
```

## Usage Guidelines

### 1. For Library Developers

#### Assert Removal Rule
```python
# ❌ Before (incorrect usage)
def qr(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Expected 2D matrix, got {x.shape}"
    return np.linalg.qr(x)

# ✅ After (correct usage)
def qr(x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"QR decomposition requires 2D matrix, got {x.ndim}D array with shape {x.shape}"
        )

    try:
        return np.linalg.qr(x)
    except Exception as e:
        raise NumericalError(f"QR decomposition failed: {e}") from e
```

#### Logging Cleanup Rule
```python
# ❌ Before (LOG_ERROR misuse)
def convert_coordinates(x):
    if x.shape[0] not in [2, 3]:
        LOG_ERROR(f"Invalid shape: {x.shape}")
        raise ValueError("Invalid input")

# ✅ After (clear exception)
def convert_coordinates(x):
    if x.shape[0] not in [2, 3]:
        raise InvalidCoordinateError(
            f"Expected 2D or 3D coordinates, got shape {x.shape}"
        )
```

#### Docstring Update Rule
```python
def matrix_operation(x: ArrayLike) -> ArrayLike:
    """
    Perform matrix operation on input.

    Args:
        x (ArrayLike): Input matrix.

    Returns:
        ArrayLike: Result of operation.

    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
        InvalidShapeError: If matrix is not square (for operations requiring square matrices).
        NumericalError: If computation fails due to numerical issues.
        IncompatibleTypeError: If input type is not supported.

    Example:
        >>> result = matrix_operation(np.eye(3))
        >>> result.shape
        (3, 3)
    """
```

#### Module-Specific Refactoring Patterns

**Operations Module (ops/)**
```python
# Before
def qr(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Expected 2D matrix, got {x.shape}"
    return np.linalg.qr(x)

# After
def qr(x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """QR decomposition with proper exception handling."""
    from ..common.exceptions import InvalidDimensionError, NumericalError

    if x.ndim != 2:
        raise InvalidDimensionError(
            f"QR decomposition requires 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )

    try:
        if is_tensor(x):
            return torch.linalg.qr(x)
        return np.linalg.qr(x)
    except Exception as e:
        raise NumericalError(f"QR decomposition failed: {e}") from e
```

**Camera Module (camera/)**
```python
# Before
def __init__(self, K, image_size):
    assert K.shape == (3, 3), f"Invalid K matrix shape: {K.shape}"
    self.K = K

# After
def __init__(self, K, image_size):
    """Initialize camera with proper validation."""
    from ..common.exceptions import InvalidCameraParameterError

    if not isinstance(K, np.ndarray) or K.shape != (3, 3):
        raise InvalidCameraParameterError(
            f"Camera matrix K must be 3x3 numpy array, got {type(K)} with shape {getattr(K, 'shape', 'unknown')}. "
            f"Please provide a valid 3x3 intrinsic matrix."
        )

    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise InvalidCameraParameterError(
            f"Focal lengths must be positive, got fx={K[0,0]}, fy={K[1,1]}. "
            f"Please check your camera calibration."
        )

    self.K = K
```

**I/O Module (io/)**
```python
# Before
def read_image(path):
    if not os.path.exists(path):
        LOG_ERROR(f"File not found: {path}")
        return None

# After
def read_image(path):
    """Read image with proper exception handling."""
    from ..common.exceptions import FileNotFoundError, FileFormatError

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )

    try:
        return cv2.imread(path)
    except Exception as e:
        raise FileFormatError(
            f"Failed to read image {path}: {e}. "
            f"Please ensure the file is a valid image format."
        ) from e
```

### 2. For Library Users

#### Exception Handling Patterns

**Fine-Grained Exception Handling (Recommended)**
```python
import spatialkit as sk
from spatialkit.common.exceptions import InvalidDimensionError, NumericalError, MathError

try:
    result = sk.umath.qr(matrix)
except InvalidDimensionError as e:
    print(f"Input dimension error: {e}")
    # Correct dimension and retry
except NumericalError as e:
    print(f"Numerical computation failed: {e}")
    # Try different algorithm
except MathError as e:
    print(f"Math operation error: {e}")
    # Handle general math errors
```

**Category-Based Exception Handling**
```python
try:
    # Multiple math operations
    result1 = sk.umath.svd(matrix1)
    result2 = sk.umath.inv(matrix2)
    result3 = sk.umath.solve(A, b)
except MathError as e:
    print(f"Math operation error: {e}")
    # Handle all math operation errors uniformly
```

**Library-Wide Exception Handling**
```python
from spatialkit.common.exceptions import SpatialKitError

try:
    # Use all spatialkit features
    cam = sk.PerspectiveCamera(...)
    points = sk.geom.multiview.triangulate_points(...)
    # ... more operations
except SpatialKitError as e:
    print(f"Spatialkit library error: {e}")
    # Handle all spatialkit-related errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other library or system errors
```

## Exception Message Guidelines

### Good Exception Message Characteristics
1. **Specific**: Clearly explain what went wrong
2. **Actionable**: Users can understand how to fix it
3. **Contextual**: Include relevant values or state information
4. **Consistent**: Use similar format for similar situations

### Examples
```python
# ❌ Bad example
raise ValueError("Invalid input")

# ✅ Good example
raise InvalidDimensionError(
    f"QR decomposition requires 2D matrix, got {x.ndim}D array with shape {x.shape}. "
    f"Please ensure input is a 2D matrix."
)

# ✅ Better example (includes solution)
raise InvalidShapeError(
    f"Matrix multiplication requires compatible shapes: A{A.shape} @ B{B.shape}. "
    f"Expected A.shape[-1] == B.shape[0], but got {A.shape[-1]} != {B.shape[0]}. "
    f"Consider reshaping or transposing one of the matrices."
)
```

## Importing Exceptions

### Method 1: Direct Import from common.exceptions
```python
from spatialkit.common.exceptions import InvalidDimensionError, NumericalError
```

### Method 2: Via common Package Re-export
```python
from spatialkit.common import InvalidDimensionError, NumericalError
```

### Method 3: Via Main Package (for external users)
```python
import spatialkit as sk
raise sk.InvalidDimensionError("...")
```

## Exception Chaining

Always use exception chaining to preserve stack traces:

```python
try:
    result = complex_operation(data)
except ValueError as e:
    raise NumericalError(
        f"Complex operation failed: {e}"
    ) from e  # Preserve original exception
```

## Validation Order

Follow this standard order when validating inputs:

1. **Type validation** - Is it the right type?
2. **Dimension validation** - Does it have the right number of dimensions?
3. **Shape validation** - Does it have the right shape?
4. **Value validation** - Does it have valid values?

```python
from spatialkit.common.exceptions import (
    IncompatibleTypeError,
    InvalidDimensionError,
    InvalidShapeError,
    NumericalError
)

def validate_rotation_matrix(R: ArrayLike) -> None:
    # 1. Type validation
    if not is_array(R):
        raise IncompatibleTypeError(
            f"Expected NumPy array or PyTorch tensor, got {type(R)}"
        )

    # 2. Dimension validation
    if R.ndim != 2:
        raise InvalidDimensionError(
            f"Rotation matrix must be 2D, got {R.ndim}D array"
        )

    # 3. Shape validation
    if R.shape != (3, 3):
        raise InvalidShapeError(
            f"Rotation matrix must be 3x3, got shape {R.shape}"
        )

    # 4. Value validation
    det = np.linalg.det(R) if is_numpy(R) else torch.det(R)
    if abs(det - 1.0) > 1e-6:
        raise NumericalError(
            f"Rotation matrix must have determinant 1, got {det:.6f}"
        )
```

## Best Practices Summary

1. **Never use `assert` for user input validation** - Always use exceptions
2. **Never use logging for errors** - Use exceptions with clear messages
3. **Always document exceptions** - Include `Raises:` section in docstrings
4. **Use specific exception types** - Choose the most appropriate exception from the hierarchy
5. **Provide context in messages** - Include shapes, values, types, and suggestions
6. **Chain exceptions** - Use `raise ... from e` to preserve stack traces
7. **Follow validation order** - Type → Dimension → Shape → Value
8. **Be consistent** - Use the same patterns across the codebase

## Version History

- **v1.0.0** (2025-11-03): Updated for current spatialkit structure, renamed from CVUtilsError to SpatialKitError
- **v0.9.0** (2025-01-30): Initial comprehensive exception hierarchy
