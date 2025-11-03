"""
Module Name: synthesis.py

Description:
This module provides functions for view synthesis and image warping between different
camera viewpoints. Supports arbitrary camera models and homographic transformations.

Supported Functions:
    - Transition camera view with optional homographic transformation

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.3.0

License: MIT LICENSE
"""

from typing import Optional
from scipy.ndimage import map_coordinates
import numpy as np
from ..ops.uops import logical_and
from ..ops.umath import inv
from ..camera import Camera


def transition_camera_view(
    src_image: np.ndarray,
    src_cam: Camera,
    dst_cam: Camera,
    img_tf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transition the view from one camera to another with a specified transformation.

    Args:
        src_image (np.ndarray, [H,W] or [H,W,3]): The input image array from the source camera.
        src_cam (Camera): Source camera instance, camera size = [W,H]
        dst_cam (Camera): Destination camera instance, camera size = [out_W,out_H]
        img_tf (np.ndarray, [3,3], optional): Transform matrix applied in normalized coordinates.

    Returns:
        output_image (np.ndarray, [out_H,out_W] or [out_H,out_W,3]): The output image transformed and projected onto the destination camera's resolution.

    Details:
    - Maps each pixel in the destination camera to the source camera
    - Applies optional homographic transformation in normalized coordinates
    - Uses bilinear interpolation for sub-pixel accuracy
    - Invalid regions are set to 0
    """
    out_height, out_width = dst_cam.hw
    # Prepare the output image array
    if src_image.ndim == 3:
        output_image = np.zeros(
            (out_height, out_width, src_image.shape[2]), dtype=src_image.dtype
        )
    else:
        output_image = np.zeros((out_height, out_width), dtype=src_image.dtype)

    output_rays, dst_valid_mask = dst_cam.convert_to_rays()  # 3 * N
    # Apply inverse transform
    if img_tf is not None:
        inverse_img_tf = inv(img_tf)
        output_rays = inverse_img_tf @ output_rays  # 3 * N

    # Project ray onto source camera
    input_coords, src_valid_mask = src_cam.convert_to_pixels(
        output_rays, out_subpixel=True
    )  # 2 * N
    # input_coords = input_coords[:,src_valid_mask]
    # Split coordinates
    input_x, input_y = input_coords[0, :], input_coords[1, :]
    if src_image.ndim == 3:
        # For multi-channel images, handle each channel separately
        for c in range(src_image.shape[2]):
            output_image[..., c] = map_coordinates(
                src_image[..., c], [input_y, input_x], order=1, mode="constant"
            ).reshape((out_height, out_width))
    else:
        # For single-channel images
        output_image = map_coordinates(
            src_image, [input_y, input_x], order=1, mode="constant"
        ).reshape((out_height, out_width))
    mask = logical_and(src_valid_mask, dst_valid_mask).reshape((out_height, out_width))
    output_image[~mask] = 0

    return output_image
