# Import Pattern Documentation

This document describes the hierarchical import pattern used in cv_utils. This pattern was designed to provide both organized access through module namespaces and convenient direct access to frequently used classes.

## Overview

The cv_utils package uses a **hierarchical import pattern** that:
- Exports modules (not wildcard contents) at package level
- Provides direct access to high-level classes for convenience
- Uses `__all__` in every module to explicitly control the public API
- Prevents namespace pollution while maintaining usability

## Import Structure

### Package-Level Imports (`cv_utils/__init__.py`)

The main package exports items in two categories:

#### 1. Package Modules (for organized access)

These are imported as modules, allowing hierarchical access:

```python
from . import ops
from .ops import umath, uops

from . import geom
from .geom import camera, rotation, pose, tf, img_tf, geom_utils

from . import utils
from .utils import io, vis, point_selector

from . import common
from .common import logger, constant

from . import sol
from .sol import marker, mvs
```

**Usage:**
```python
import cv_utils

# Access functions through module namespaces
result = cv_utils.umath.matmul(a, b)
camera = cv_utils.camera.PerspectiveCamera(...)
```

#### 2. High-Level Classes (for convenience)

Frequently used classes are exported directly from the main package:

```python
# Geometry primitives
from .geom.rotation import Rotation, RotType
from .geom.pose import Pose
from .geom.tf import Transform
from .geom.camera import (
    Camera, CamType,
    PerspectiveCamera,
    OpenCVFisheyeCamera,
    ThinPrismFisheyeCamera,
    OmnidirectionalCamera,
    DoubleSphereCamera,
    EquirectangularCamera,
)

# Solutions
from .sol.marker import (
    Marker,
    FiducialMarkerType,
    MarkerDetector,
    OpenCVMarkerDetector,
    AprilTagMarkerDetector,
    STagMarkerDetector,
)
from .sol.mvs import ScannetV1Manager

# Visualization
from .externals import _o3d as vis3d

# Logging
from .common.logger import LOG_CRITICAL, LOG_DEBUG, LOG_ERROR, LOG_INFO, LOG_WARN

# Exception hierarchy
from .exceptions import (
    CVUtilsError,
    MathError, InvalidDimensionError, InvalidShapeError,
    IncompatibleTypeError, NumericalError, SingularMatrixError,
    GeometryError, ConversionError, InvalidCoordinateError,
    ProjectionError, CalibrationError,
    CameraError, InvalidCameraParameterError,
    UnsupportedCameraTypeError, CameraModelError,
    VisualizationError, RenderingError, DisplayError,
    IOError, FileNotFoundError, FileFormatError, ReadWriteError,
    MarkerError, MarkerDetectionError, InvalidMarkerTypeError,
    MVSError, DatasetError, ReconstructionError,
)
```

**Usage:**
```python
import cv_utils

# Direct access to frequently used classes
rotation = cv_utils.Rotation.from_quat_wxyz(quat)
camera = cv_utils.PerspectiveCamera.from_fov([1920, 1080], 90)
```

### Subpackage-Level Imports

Each subpackage exports its modules (not contents):

**`cv_utils/ops/__init__.py`:**
```python
from . import uops
from . import umath

__all__ = ["uops", "umath"]
```

**`cv_utils/geom/__init__.py`:**
```python
from . import camera
from . import rotation
from . import pose
from . import tf
from . import img_tf
from . import geom_utils

__all__ = ["camera", "rotation", "pose", "tf", "img_tf", "geom_utils"]
```

**`cv_utils/utils/__init__.py`:**
```python
from . import io
from . import vis
from . import point_selector

__all__ = ["io", "vis", "point_selector"]
```

### Module-Level `__all__`

Every module defines `__all__` to explicitly control its public API:

**Example from `uops.py`:**
```python
__all__ = [
    # Type alias
    "ArrayLike",
    # Type checking
    "is_tensor",
    "is_numpy",
    "is_array",
    # Type conversion
    "convert_tensor",
    "convert_numpy",
    "convert_array",
    # ... (organized by category)
]
```

**Example from `camera.py`:**
```python
__all__ = [
    "CamType",
    "Camera",
    "RadialCamera",
    "PerspectiveCamera",
    "OpenCVFisheyeCamera",
    "ThinPrismFisheyeCamera",
    "OmnidirectionalCamera",
    "DoubleSphereCamera",
    "EquirectangularCamera",
]
```

## Import Patterns

### Recommended Usage

#### 1. For unified operations (most common)

```python
import cv_utils
import numpy as np

# Use unified operations
arr = np.array([1, 2, 3])
result = cv_utils.umath.sqrt(arr)
normalized = cv_utils.umath.normalize(result, dim=0)
```

#### 2. For high-level classes

```python
import cv_utils

# Direct class access
camera = cv_utils.PerspectiveCamera.from_fov([1920, 1080], 90)
rotation = cv_utils.Rotation.from_quat_wxyz(quat)
pose = cv_utils.Pose(rotation, translation)
```

#### 3. For specific module functionality

```python
from cv_utils import camera, rotation

# Module-level access
cam = camera.PerspectiveCamera.from_K(K, image_size)
rot = rotation.quat_to_SO3(quat, is_xyzw=True)
```

#### 4. For direct imports (when needed frequently)

```python
from cv_utils.umath import matmul, normalize, svd
from cv_utils.uops import concat, stack
from cv_utils import Rotation, Pose, PerspectiveCamera

# Direct usage
R = matmul(A, B)
pts = concat([x, y, z], dim=0)
pose = Pose(Rotation.from_mat3(R), t)
```

### Anti-Patterns (Avoid)

❌ **Wildcard imports from subpackages:**
```python
from cv_utils.ops import *  # Don't do this - unclear what's imported
```

❌ **Deep nested imports for common classes:**
```python
from cv_utils.geom.rotation import Rotation  # Unnecessary - use cv_utils.Rotation
```

❌ **Importing internal functions:**
```python
from cv_utils.uops import _assert_same_array_type  # Private function, don't import
```

## Benefits of This Pattern

1. **Namespace Organization**: Functions are organized into logical modules
2. **Convenient Access**: Frequently used classes are directly accessible
3. **Explicit API Control**: `__all__` prevents accidental exports
4. **IDE Support**: Better autocomplete and type hints
5. **Backward Compatibility**: Old import patterns still work
6. **Clear Dependencies**: Easy to see what's public vs internal

## Design Principles

1. **Modules Over Wildcards**: Export modules, not wildcard contents
2. **Hierarchical Organization**: ops → geom → utils → sol reflects increasing abstraction
3. **Convenience for Common Cases**: Direct exports for frequently used classes
4. **Explicit is Better**: Use `__all__` everywhere to be explicit about public API
5. **Minimal Pollution**: Only export what users need

## Migration Notes

If you have existing code using the old patterns, the new pattern is backward compatible:

**Old pattern (still works):**
```python
from cv_utils.geom.camera import PerspectiveCamera
from cv_utils.ops.umath import matmul
```

**New recommended pattern:**
```python
from cv_utils import PerspectiveCamera
import cv_utils

result = cv_utils.umath.matmul(a, b)
```

## Module Reference

### Core Operations (`cv_utils.ops`)
- `uops`: Unified basic operations (type checking, conversion, array construction)
- `umath`: Unified mathematical operations (linear algebra, trigonometry, geometry)

### Geometry (`cv_utils.geom`)
- `camera`: Camera models and projections
- `rotation`: 3D rotation representations and conversions
- `pose`: 6-DOF pose (rotation + translation)
- `tf`: 6-DOF transformation
- `img_tf`: 2D image transformations
- `geom_utils`: Geometric algorithms (PnP, triangulation, etc.)

### Utilities (`cv_utils.utils`)
- `io`: File I/O operations
- `vis`: Visualization utilities
- `point_selector`: Interactive point selection GUI

### Solutions (`cv_utils.sol`)
- `marker`: Fiducial marker detection
- `mvs`: Multi-view stereo utilities

### Common (`cv_utils.common`)
- `logger`: Logging utilities
- `constant`: Mathematical and physical constants

### External (`cv_utils.externals`)
- `_o3d` (exported as `vis3d`): Open3D visualization wrappers
