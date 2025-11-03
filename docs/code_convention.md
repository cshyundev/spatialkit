# Project Code Convention Documentation

**Last Updated:** 2025-11-03
**Version:** 0.3.0-alpha

This document defines the coding conventions for the spatialkit project.

## Naming Conventions

### Packages
- Package names are all lowercase.
- Do not use underscores between words; use short, clear names.
- Examples: `ops`, `geom`, `camera`, `markers`, `vis2d`, `vis3d`

### Modules
- Module file names are all lowercase with underscores separating words.
- Module names should clearly indicate their functionality.
- Examples: `uops.py`, `umath.py`, `img_tf.py`, `geom_utils.py`, `point_selector.py`

### Classes
- Class names use PascalCase.
- Class names should clearly indicate the role and function of the class.
- Internal-only variables and functions are prefixed with an underscore (_).
- Examples: `Rotation`, `Pose`, `Transform`, `PerspectiveCamera`, `MarkerDetector`

### Variables
- Variable names use snake_case.
- Variable names should clearly indicate the role and function of the variable.
- Constant names are all uppercase with underscores separating words.
- Examples: `focal_length`, `image_size`, `rotation_matrix`
- Constants: `PI`, `ROTATION_SO3_THRESHOLD`, `DEFAULT_FOV`

### Functions
- Function names use snake_case.
- Function names should clearly indicate the role and function of the function.
- Private/internal functions are prefixed with an underscore (_).
- Examples: `is_SO3`, `quat_to_SO3`, `triangulate_points`, `read_image`
- Private: `_validate_shape`, `_assert_same_type`

## Documentation Style

### Functions
- Function documentation follows the format shown below.
- Documentation may be omitted if the function's behavior is simple or self-evident.
- Documentation is placed at the top of the function body.

#### Components
1. Brief description
2. Function arguments (Args)
3. Function returns (Returns)
4. Exceptions raised (Raises) - **REQUIRED** if any exceptions are raised
5. Detailed behavior explanation (Details) - Optional
6. Usage examples (Example) - Optional

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.

    Args:
        param1 (type, shape): Description of param1.
        param2 (type, shape): Description of param2.

    Returns:
        return_type (type, shape): Description of the return value.

    Raises:
        ExceptionType1: When and why this exception is raised.
        ExceptionType2: When and why this exception is raised.

    Details:
        - Explanation of the first key detail.
        - Explanation of the second key detail.

    Example:
        >>> result = function_name(value1, value2)
        >>> result.shape
        (3, 3)
    """
    ...
```

**Important Notes:**
- The `Raises:` section is **mandatory** when the function raises any exceptions
- Always document all possible exceptions that can be raised
- Include specific conditions under which each exception is raised
- Use descriptive exception types from the spatialkit exception hierarchy

### Classes
- Class documentation follows the format shown below.
- Documentation is placed at the top of the `__init__` function.

#### Components
1. Brief description
2. Class member variables (Attributes)
3. Abstract methods (Abstract Methods) - Optional

```python
class ClassName:
    """
    Brief description of what the class does.

    Attributes:
        attr1 (type, shape): Description of attr1.
        attr2 (type, shape): Description of attr2.

    Abstract Methods:
        abstract_method1: Description of abstract_method1.
        abstract_method2: Description of abstract_method2.
    """
    def __init__(self, ...):
        ...
```

**For classes with complex initialization:**
```python
class PerspectiveCamera(Camera):
    """
    Perspective (pinhole) camera model.

    Attributes:
        K (np.ndarray, (3, 3)): Camera intrinsic matrix.
        image_size (tuple[int, int]): Image size (width, height).
        dist_coeffs (np.ndarray | None): Distortion coefficients if applicable.

    Raises:
        InvalidCameraParameterError: If K matrix is invalid or focal lengths are non-positive.
    """
```

### Modules
- Module documentation follows the format shown below.
- Documentation is placed at the very top of the file.

#### Components
1. Module name
2. Brief description of the module
3. Main functions and supported types (Optional)
4. Author information
5. License information
6. Usage examples

```python
"""
Module Name: module_name.py

Description:
Brief description of what this module provides.

Main Functions:
- function1: Description
- function2: Description

Supported Types:
- NumPy arrays
- PyTorch tensors

Author: Author Name
Email: email@example.com
Version: x.x.x

License: MIT License

Usage:
    from spatialkit.package import module_name
    result = module_name.function(...)
"""
```

## Type Hints

### General Guidelines
- All function signatures should include type hints for parameters and return values
- Use standard typing module imports: `from typing import Optional, Union, Tuple, List`
- Use newer Python 3.10+ syntax when available: `list[int]` instead of `List[int]`

### Array Type Hints
For functions working with both NumPy and PyTorch:
```python
from spatialkit.ops.uops import ArrayLike

def process_array(arr: ArrayLike) -> ArrayLike:
    """Process array (NumPy or PyTorch)."""
    ...
```

### Specific Type Hints
```python
# For NumPy arrays
import numpy as np

def numpy_function(arr: np.ndarray) -> np.ndarray:
    ...

# For PyTorch tensors
import torch

def torch_function(tensor: torch.Tensor) -> torch.Tensor:
    ...

# For optional parameters
def function(param: int | None = None) -> float:
    ...

# For multiple return values
def decompose() -> tuple[np.ndarray, np.ndarray]:
    ...
```

## Code Organization

### Import Order
1. Standard library imports
2. Third-party library imports (numpy, torch, cv2, etc.)
3. Local application imports

```python
# Standard library
import os
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import torch
import cv2

# Local
from ..common.exceptions import InvalidDimensionError
from ..ops.uops import is_tensor, convert_array
```

### Module Structure
1. Module docstring
2. Imports
3. Constants/globals
4. Private helper functions
5. Public API functions/classes
6. `__all__` declaration at the end

```python
"""Module docstring."""

# Imports
import numpy as np
from ..common.exceptions import MathError

# Constants
DEFAULT_TOLERANCE = 1e-6

# Private functions
def _validate_input(x):
    ...

# Public API
def public_function(x):
    ...

class PublicClass:
    ...

# Public API declaration
__all__ = ["public_function", "PublicClass"]
```

## Exception Handling

### Never Use Assert for User Input
```python
# ❌ WRONG
def qr(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Expected 2D matrix, got {x.shape}"
    return np.linalg.qr(x)

# ✅ CORRECT
from ..common.exceptions import InvalidDimensionError, NumericalError

def qr(x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute QR decomposition.

    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
        NumericalError: If QR decomposition fails.
    """
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"QR decomposition requires 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )

    try:
        return np.linalg.qr(x) if is_numpy(x) else torch.linalg.qr(x)
    except Exception as e:
        raise NumericalError(f"QR decomposition failed: {e}") from e
```

### Use Descriptive Exception Messages
Exception messages should:
- Clearly state what went wrong
- Provide context (shapes, values, types)
- Suggest how to fix the issue
- Use consistent formatting

```python
# Good exception message
raise InvalidShapeError(
    f"Matrix multiplication requires compatible shapes: A{A.shape} @ B{B.shape}. "
    f"Expected A.shape[-1] == B.shape[0], but got {A.shape[-1]} != {B.shape[0]}. "
    f"Consider reshaping or transposing one of the matrices."
)
```

## Validation Patterns

### Standard Validation Order
1. Type validation (is it an array?)
2. Dimension validation (is it 2D?)
3. Shape validation (is it 3x3?)
4. Value validation (is det=1? is it orthogonal?)

```python
from ..common.exceptions import (
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

## Performance Considerations

### Array Operations
- Prefer vectorized operations over loops
- Use in-place operations when appropriate (with care for mutability)
- Avoid unnecessary array copies

```python
# ✅ GOOD - Vectorized
result = np.sqrt(arr ** 2 + 1)

# ❌ BAD - Loop
result = np.array([np.sqrt(x**2 + 1) for x in arr])
```

### Type Checking
- Cache type checks when performing multiple operations
- Use `isinstance` for Python types, custom functions for array types

```python
# ✅ GOOD - Check once
is_torch = is_tensor(x)
if is_torch:
    result = torch.matmul(torch.inv(A), b)
else:
    result = np.linalg.solve(A, b)

# ❌ BAD - Repeated checks
if is_tensor(x):
    A_inv = torch.inv(A)
if is_tensor(x):
    result = torch.matmul(A_inv, b)
```

## Testing Conventions

### Test File Organization
- Test files mirror the source structure: `src/spatialkit/ops/umath.py` → `tests/ops/test_umath.py`
- Use descriptive test function names: `test_qr_decomposition_square_matrix`
- Group related tests using classes or pytest marks

### Test Function Pattern
```python
import pytest
import numpy as np
from spatialkit.ops.umath import qr
from spatialkit.common.exceptions import InvalidDimensionError

def test_qr_square_matrix():
    """Test QR decomposition on square matrix."""
    A = np.random.randn(3, 3)
    Q, R = qr(A)

    # Verify Q is orthogonal
    assert np.allclose(Q @ Q.T, np.eye(3))

    # Verify reconstruction
    assert np.allclose(Q @ R, A)

def test_qr_invalid_dimension():
    """Test QR raises exception for non-2D input."""
    A = np.array([1, 2, 3])

    with pytest.raises(InvalidDimensionError):
        qr(A)
```

## Version History

- **v0.3.0-alpha** (2025-11-03): Updated for new package structure, added exception handling guidelines
- **v0.2.1-alpha** (2025-01-30): Added type hints and validation patterns
- **v0.2.0-alpha** (2024-12): Initial code conventions
