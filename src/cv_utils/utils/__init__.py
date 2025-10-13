"""
Utilities package for I/O, visualization, and helper functions.

This package provides general-purpose utilities for file I/O, image visualization,
and interactive point selection.

Modules:
    io: File I/O operations (images, point clouds, etc.)
    vis: Visualization utilities (drawing, displaying images)
    point_selector: Interactive GUI for point selection
"""

from . import io
from . import vis
from . import point_selector

__all__ = ["io", "vis", "point_selector"]