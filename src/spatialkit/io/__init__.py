"""
I/O Package for spatialkit.

This package provides file I/O operations for images, videos, and configuration files.

Modules:
    image: Image I/O (PNG, JPG, TIFF, PGM)
    video: Video I/O with sampling and lazy loading
    config: Configuration files (JSON, YAML)

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.0
License: MIT License
"""

# Image I/O
from .image import (
    read_tiff,
    write_tiff,
    read_all_images,
    read_image,
    write_image,
    read_pgm,
    write_pgm,
)

# Video I/O
from .video import (
    write_video_from_image_paths,
    write_video_from_images,
    read_video_with_sampling_ratio,
    read_video_with_num_samples,
    VideoReader,
)

# Config I/O
from .config import (
    read_json,
    write_json,
    read_yaml,
    write_yaml,
)

__all__ = [
    # Image
    "read_tiff",
    "write_tiff",
    "read_all_images",
    "read_image",
    "write_image",
    "read_pgm",
    "write_pgm",
    # Video
    "write_video_from_image_paths",
    "write_video_from_images",
    "read_video_with_sampling_ratio",
    "read_video_with_num_samples",
    "VideoReader",
    # Config
    "read_json",
    "write_json",
    "read_yaml",
    "write_yaml",
]
