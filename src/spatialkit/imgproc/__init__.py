"""
Image processing operations for 2D transformations.

This package provides 2D image transformation functions including:
- Homography computation and application
- Affine transformations
- Similarity transformations
- Translation, rotation, scaling, and shear
- View synthesis and image warping
"""

from .img_tf import (
    translation,
    rotation,
    shear,
    scaling,
    similarity,
    affine,
    compute_homography,
    apply_transform,
)

from .synthesis import (
    transition_camera_view,
)

__all__ = [
    "translation",
    "rotation",
    "shear",
    "scaling",
    "similarity",
    "affine",
    "compute_homography",
    "apply_transform",
    "transition_camera_view",
]
