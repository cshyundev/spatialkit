"""
Module Name: constant.py

Description:
This module defines constant values used across various computer vision utilities. 
These constants are crucial for ensuring numerical stability and precision in computations.

Constants:
- PI: The mathematical constant π, used in calculations involving circles and angles.
- EPSILON: A small value used to prevent division by zero and ensure numerical stability.
- NORM_PIXEL_THRESHOLD: A threshold value for normalizing pixel intensities.
- ROTATION_SO3_THRESHOLD: A threshold value for validating rotations in the SO3.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT LICENSE
"""

import numpy as np

## CONSTANT VALUES
PI = 3.141592
EPSILON = 1e-9
NORM_PIXEL_THRESHOLD = 1e-4
ROTATION_SO3_THRESHOLD = 1e-5

## COORDINATE FRAME TRANSFORMATION MATRICES (4x4)
# OpenCV (Z-forward, Y-down) → ROS (X-forward, Z-up)
# ROS: x=opencv_z, y=-opencv_x, z=-opencv_y
OPENCV_TO_ROS = np.array(
    [
        [0, 0, 1, 0],  # ros_x = opencv_z
        [-1, 0, 0, 0],  # ros_y = -opencv_x
        [0, -1, 0, 0],  # ros_z = -opencv_y
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

# OpenCV (Z-forward, Y-down) → OpenGL (Z-backward, Y-up)
OPENCV_TO_OPENGL = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
