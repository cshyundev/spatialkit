# Import Pattern Documentation

**Last Updated:** 2025-11-03
**Current Version:** 0.3.2

This document describes the hierarchical import pattern used in spatialkit. This pattern was designed to provide both organized access through module namespaces and convenient direct access to frequently used classes.

## Overview

The spatialkit package uses a **hierarchical import pattern** that:
- Exports modules (not wildcard contents) at package level
- Provides direct access to high-level classes for convenience
- Uses `__all__` in every module to explicitly control the public API
- Prevents namespace pollution while maintaining usability

## Import Structure

### Package-Level Imports (`spatialkit/__init__.py`)

The main package exports items in two categories:

#### 1. Package Modules (for organized access)

These are imported as modules, allowing hierarchical access:

```python
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

from . import markers
from . import vis3d
```

**Usage:**
```python
import spatialkit as sk

# Access functions through module namespaces
result = sk.umath.matmul(a, b)
cam = sk.camera.PerspectiveCamera(...)
points = sk.multiview.triangulate_points(...)
```

#### 2. High-Level Classes (for convenience)

Frequently used classes are exported directly from the main package:

```python
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

# Marker detection
from .markers import (
    Marker,
    FiducialMarkerType,
    MarkerDetector,
    OpenCVMarkerDetector,
    AprilTagMarkerDetector,
    STagMarkerDetector,
)

# Logging
from .common.logger import LOG_CRITICAL, LOG_DEBUG, LOG_ERROR, LOG_INFO, LOG_WARN

# Exception hierarchy
from .common.exceptions import (
    SpatialKitError,
    MathError, InvalidDimensionError, InvalidShapeError,
    IncompatibleTypeError, NumericalError, SingularMatrixError,
    GeometryError, ConversionError, InvalidCoordinateError,
    ProjectionError, CalibrationError,
    CameraError, InvalidCameraParameterError,
    UnsupportedCameraTypeError, CameraModelError,
    VisualizationError, RenderingError, DisplayError,
    IOError, FileNotFoundError, FileFormatError, ReadWriteError,
    MarkerError, MarkerDetectionError, InvalidMarkerTypeError,
)
```

**Usage:**
```python
import spatialkit as sk

# Direct access to frequently used classes
rotation = sk.Rotation.from_quat_wxyz(quat)
camera = sk.PerspectiveCamera.from_fov([1920, 1080], 90)
detector = sk.AprilTagMarkerDetector()
```

### Subpackage-Level Imports

Each subpackage exports its modules (not contents):

**`spatialkit/ops/__init__.py`:**
```python
from . import uops
from . import umath

__all__ = ["uops", "umath"]
```

**`spatialkit/geom/__init__.py`:**
```python
from . import rotation
from . import pose
from . import tf
from . import epipolar
from . import multiview
from . import pointcloud

__all__ = ["rotation", "pose", "tf", "epipolar", "multiview", "pointcloud"]
```

**`spatialkit/camera/__init__.py`:**
```python
from .base import Camera, CamType
from .perspective import PerspectiveCamera
from .fisheye import OpenCVFisheyeCamera, ThinPrismFisheyeCamera
from .omnidirectional import OmnidirectionalCamera
from .doublesphere import DoubleSphereCamera
from .equirectangular import EquirectangularCamera

__all__ = [
    "Camera", "CamType",
    "PerspectiveCamera",
    "OpenCVFisheyeCamera",
    "ThinPrismFisheyeCamera",
    "OmnidirectionalCamera",
    "DoubleSphereCamera",
    "EquirectangularCamera",
]
```

**`spatialkit/io/__init__.py`:**
```python
from . import image
from . import video
from . import config

__all__ = ["image", "video", "config"]
```

**`spatialkit/vis2d/__init__.py`:**
```python
from . import convert
from . import draw
from . import display

__all__ = ["convert", "draw", "display"]
```

**`spatialkit/markers/__init__.py`:**
```python
from .marker import Marker, FiducialMarkerType
from .base import MarkerDetector
from .opencv_detector import OpenCVMarkerDetector
from .apriltag_detector import AprilTagMarkerDetector
from .stag_detector import STagMarkerDetector

__all__ = [
    "Marker",
    "FiducialMarkerType",
    "MarkerDetector",
    "OpenCVMarkerDetector",
    "AprilTagMarkerDetector",
    "STagMarkerDetector",
]
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
    # Array construction
    "zeros", "ones", "eye", "arange", "linspace",
    # Shape operations
    "reshape", "transpose", "squeeze", "unsqueeze",
    # Concatenation
    "concat", "stack",
    # ... (organized by category)
]
```

**Example from `camera/perspective.py`:**
```python
__all__ = [
    "PerspectiveCamera",
]
```

**Example from `io/image.py`:**
```python
__all__ = [
    "read_image",
    "write_image",
    "read_tiff",
    "write_tiff",
    "read_pgm",
    "write_pgm",
]
```

## Import Patterns

### Recommended Usage

#### 1. For unified operations (most common)

```python
import spatialkit as sk
import numpy as np

# Use unified operations
arr = np.array([1, 2, 3])
result = sk.umath.sqrt(arr)
normalized = sk.umath.normalize(result, dim=0)

# Also works with PyTorch
import torch
tensor = torch.tensor([1.0, 2.0, 3.0])
result = sk.umath.sqrt(tensor)  # Returns PyTorch tensor
```

#### 2. For high-level classes

```python
import spatialkit as sk

# Direct class access
camera = sk.PerspectiveCamera.from_fov([1920, 1080], 90)
rotation = sk.Rotation.from_quat_wxyz(quat)
pose = sk.Pose(rotation, translation)

# Marker detection
detector = sk.AprilTagMarkerDetector()
markers = detector.detect(image)
```

#### 3. For specific module functionality

```python
from spatialkit import camera, rotation, multiview

# Module-level access
cam = camera.PerspectiveCamera.from_K(K, image_size)
rot = rotation.quat_to_SO3(quat, is_xyzw=True)
points_3d = multiview.triangulate_points(pts1, pts2, P1, P2)
```

#### 4. For I/O operations

```python
from spatialkit import io

# Image I/O
image = io.image.read_image("photo.jpg")
io.image.write_image("output.png", image)

# Video I/O
video = io.video.VideoReader("video.mp4")
for frame in video:
    process(frame)

# Config I/O
config = io.config.read_json("config.json")
io.config.write_yaml("output.yaml", config)
```

#### 5. For visualization

```python
from spatialkit import vis2d, vis3d

# 2D visualization
vis2d.display.show_image(image)
vis2d.draw.draw_circle(image, center, radius, color)

# 3D visualization
pcd = vis3d.components.create_point_cloud(points, colors)
vis3d.o3dutils.visualize([pcd])
```

#### 6. For direct imports (when needed frequently)

```python
from spatialkit.umath import matmul, normalize, svd
from spatialkit.uops import concat, stack
from spatialkit import Rotation, Pose, PerspectiveCamera

# Direct usage
R = matmul(A, B)
pts = concat([x, y, z], dim=0)
pose = Pose(Rotation.from_mat3(R), t)
```

### Anti-Patterns (Avoid)

❌ **Wildcard imports from subpackages:**
```python
from spatialkit.ops import *  # Don't do this - unclear what's imported
```

❌ **Deep nested imports for common classes:**
```python
from spatialkit.geom.rotation import Rotation  # Unnecessary - use spatialkit.Rotation
from spatialkit.camera.perspective import PerspectiveCamera  # Use sk.PerspectiveCamera
```

❌ **Importing internal functions:**
```python
from spatialkit.ops.uops import _assert_same_array_type  # Private function, don't import
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
2. **Hierarchical Organization**: common → ops → geom → camera → imgproc/io/vis2d → markers reflects increasing abstraction
3. **Convenience for Common Cases**: Direct exports for frequently used classes
4. **Explicit is Better**: Use `__all__` everywhere to be explicit about public API
5. **Minimal Pollution**: Only export what users need

## Migration Notes

If you have existing code using the old patterns, the new pattern is backward compatible:

**Old pattern (still works):**
```python
from spatialkit.geom.camera import PerspectiveCamera  # Old location
from spatialkit.camera import PerspectiveCamera  # New location
from spatialkit import PerspectiveCamera  # Recommended
```

**New recommended pattern:**
```python
import spatialkit as sk

# Classes
camera = sk.PerspectiveCamera(...)

# Operations
result = sk.umath.matmul(a, b)
```

## Module Reference

### Core Operations (`spatialkit.ops`)
- `uops`: Unified basic operations (type checking, conversion, array construction, shape manipulation)
- `umath`: Unified mathematical operations (linear algebra, trigonometry, statistics)

### Geometry (`spatialkit.geom`)
- `rotation`: 3D rotation representations and conversions
- `pose`: 6-DOF pose (rotation + translation)
- `tf`: 6-DOF transformation
- `epipolar`: Epipolar geometry (fundamental/essential matrices)
- `multiview`: Multi-view geometry algorithms (PnP, triangulation, homography)
- `pointcloud`: Point cloud processing utilities

### Camera (`spatialkit.camera`)
Camera models and projections:
- `base`: Abstract Camera base class
- `perspective`: Perspective (pinhole) camera
- `fisheye`: Fisheye cameras (OpenCV, ThinPrism)
- `omnidirectional`: Omnidirectional camera model
- `doublesphere`: Double sphere camera for wide FOV
- `equirectangular`: Equirectangular projection

### Image Processing (`spatialkit.imgproc`)
- `transform`: 2D image transformations (homography-based warping)
- `synthesis`: Image synthesis utilities

### I/O Operations (`spatialkit.io`)
- `image`: Image I/O (PNG, JPG, TIFF, PGM)
- `video`: Video I/O with lazy loading (VideoReader class)
- `config`: Configuration file I/O (JSON, YAML)

### 2D Visualization (`spatialkit.vis2d`)
- `convert`: Image conversion utilities (float_to_image, normal_to_image, concat_images)
- `draw`: Drawing functions (circles, lines, polygons, correspondences)
- `display`: Display utilities (show_image, show_two_images)

### 3D Visualization (`spatialkit.vis3d`)
- `common`: Open3D type aliases
- `components`: 3D geometry creation (coordinate frames, camera frustums)
- `o3dutils`: Open3D utility functions

### Markers (`spatialkit.markers`)
- `marker`: Marker data classes (Marker, FiducialMarkerType)
- `base`: Abstract MarkerDetector base class
- `opencv_detector`: OpenCV ArUco detector
- `apriltag_detector`: AprilTag detector
- `stag_detector`: STag detector

### Common (`spatialkit.common`)
- `logger`: Logging utilities (LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL)
- `constant`: Mathematical and physical constants
- `exceptions`: Hierarchical exception system

## Quick Reference

### Most Common Imports

```python
import spatialkit as sk
import numpy as np

# Geometry classes
rot = sk.Rotation.from_mat3(R)
pose = sk.Pose(rot, t)
cam = sk.PerspectiveCamera.from_fov([1920, 1080], 90)

# Unified operations
result = sk.umath.matmul(A, b)
normalized = sk.umath.normalize(v)

# Multi-view geometry
pts_3d = sk.multiview.triangulate_points(pts1, pts2, P1, P2)

# I/O
image = sk.io.image.read_image("file.jpg")
config = sk.io.config.read_json("config.json")

# Visualization
sk.vis2d.display.show_image(image)

# Marker detection
detector = sk.AprilTagMarkerDetector()
markers = detector.detect(image)

# Exceptions
from spatialkit.common.exceptions import InvalidDimensionError
```

## Version History

- **v0.3.0-alpha** (2025-11-03): Updated for new package structure (separated camera, added imgproc/vis3d)
- **v0.2.1-alpha** (2025-01-30): Added exception hierarchy exports
- **v0.2.0-alpha** (2024-12): Initial hierarchical import pattern
