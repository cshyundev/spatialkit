"""
Camera models for computer vision applications.

This package provides various camera projection models including:
- Perspective cameras
- Fisheye cameras (OpenCV, ThinPrism)
- Wide FOV cameras (DoubleSphere, Omnidirectional, Equirectangular)
"""

from .base import Camera, CamType
from .perspective import PerspectiveCamera
from .fisheye import OpenCVFisheyeCamera, ThinPrismFisheyeCamera
from .omnidirectional import OmnidirectionalCamera
from .doublesphere import DoubleSphereCamera
from .equirectangular import EquirectangularCamera

__all__ = [
    "Camera",
    "CamType",
    "PerspectiveCamera",
    "OpenCVFisheyeCamera",
    "ThinPrismFisheyeCamera",
    "OmnidirectionalCamera",
    "DoubleSphereCamera",
    "EquirectangularCamera",
]
