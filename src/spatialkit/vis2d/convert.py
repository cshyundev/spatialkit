"""
Image conversion and manipulation utilities.

This module provides functions for converting images between different formats
and manipulating image data for visualization purposes.

Functions:
    - convert_to_uint8_image: Convert image to uint8 format
    - float_to_image: Convert float image to uint8
    - normal_to_image: Convert normal map to RGB image
    - concat_images: Concatenate multiple images
    - generate_rainbow_colors: Generate rainbow color palette
    - _is_image: Internal helper to validate image arrays

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT License
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from ..common.exceptions import (
    InvalidShapeError,
    InvalidDimensionError,
    ConversionError,
    RenderingError
)
def _is_image(image: np.ndarray) -> bool:
    """
    Checks if the given array is an image with 3 channels.

    Args:
        image (np.ndarray): Array to check.

    Returns:
        bool: True if the array is an image with 3 channels, False otherwise.

    Example:
        result = is_image(np.random.rand(100, 100, 3))
    """
    if len(image.shape) != 3:
        return False
    if image.shape[0] == 3 or image.shape[-1] == 3:
        return True
    return False


def convert_to_uint8_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes data to the range [0, 255] and converts it to uint8 type.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Converted image array.

    Raises:
        InvalidShapeError: If input image doesn't have valid shape (H,W,3) or (3,H,W).
        ConversionError: If image conversion fails.

    Details:
        - Checks if the input image has a valid shape: (H,W,3) or (3,H,W).
        - Transposes the image if its shape is (3,H,W) to (H,W,3).
        - If the image data type is not uint8, normalizes the data to the range [0, 255] and converts it to uint8.

    Example:
        converted_image = convert_to_uint8_image(np.random.rand(3, 100, 100))
    """
    if not _is_image(image):
        raise InvalidShapeError(
            f"Invalid image shape. Expected (H,W,3) or (3,H,W), got {image.shape}. "
            f"Please ensure input is a valid 3-channel image array."
        )
    
    try:
        if image.shape[0] == 3:  # (3,H,W)
            image = np.transpose(image, (1, 2, 0))  # (H,W,3)
        if image.dtype != np.uint8:
            image = image * 255
            image = image.astype(np.uint8)
        return image
    except Exception as e:
        raise ConversionError(
            f"Failed to convert image to uint8 format: {e}. "
            f"Please check input image data type and value range."
        ) from e


def float_to_image(
    v: np.ndarray,
    min_v: Optional[float] = None,
    max_v: Optional[float] = None,
    color_map: str = "magma",
) -> np.ndarray:
    """
    Converts a float array to a color image using a color map.

    Args:
        v (np.ndarray): Input float array.
        min_v (Optional[float]): Minimum value for normalization. Default is None.
        max_v (Optional[float]): Maximum value for normalization. Default is None.
        color_map (str): Color map to use. Default is 'magma'.

    Returns:
        np.ndarray: Color mapped image.
        
    Raises:
        InvalidDimensionError: If input array has invalid dimensions.
        RenderingError: If color mapping fails.
        
    Example:
        color_image = float_to_image(depth_array, color_map='viridis')
    """
    if not isinstance(v, np.ndarray):
        raise InvalidDimensionError(
            f"Expected numpy array, got {type(v)}. "
            f"Please provide a valid numpy array."
        )
    
    if len(v.shape) == 0 or len(v.shape) > 3:
        raise InvalidDimensionError(
            f"Expected 1D, 2D, or 3D array, got {len(v.shape)}D array with shape {v.shape}. "
            f"Please ensure input is a valid array for color mapping."
        )
    
    try:
        if len(v.shape) == 3:
            v = np.squeeze(v)
        if min_v is None:
            min_v = v.min()
        if max_v is None:
            max_v = v.max()
        
        if max_v == min_v:
            # Handle constant array case
            normalized_v = np.zeros_like(v)
        else:
            v = np.clip(v, min_v, max_v)
            normalized_v = (v - min_v) / (max_v - min_v)
        
        color_mapped = plt.cm.get_cmap(color_map)(normalized_v)
        color_mapped = (color_mapped[:, :, :3] * 255).astype(np.uint8)
        return color_mapped
    except Exception as e:
        raise RenderingError(
            f"Failed to apply color map '{color_map}' to array: {e}. "
            f"Please check color map name and array values."
        ) from e


def normal_to_image(normal: np.ndarray) -> np.ndarray:
    """
    Converts normal map to an image.

    Args:
        normal (np.ndarray): Input normal map array.

    Returns:
        np.ndarray: Converted image.
        
    Raises:
        InvalidShapeError: If normal map doesn't have 3 channels.
        ConversionError: If normal map conversion fails.

    Details:
        - Converts the normal map coordinates to OpenGL coordinates.
        - The normal map is expected to have 3 channels.
        - The red channel is mapped from (x + 1) / 2.
        - The green channel is mapped from (-y + 1) / 2.
        - The blue channel is mapped from (-z + 1) / 2.
        
    Example:
        normal_image = normal_to_image(normal_map)
    """
    if len(normal.shape) != 3 or normal.shape[-1] != 3:
        raise InvalidShapeError(
            f"Normal map must be 3D array with 3 channels, got shape {normal.shape}. "
            f"Expected shape (H, W, 3). Please ensure input is a valid normal map."
        )
    
    try:
        r = (normal[:, :, 0] + 1) / 2.0  # (H,W,3)
        g = (-normal[:, :, 1] + 1) / 2.0
        b = (-normal[:, :, 2] + 1) / 2.0
        color_mapped = convert_to_uint8_image(np.stack((r, g, b), -1))
        return color_mapped
    except Exception as e:
        raise ConversionError(
            f"Failed to convert normal map to image: {e}. "
            f"Please check normal map values are in valid range."
        ) from e


def concat_images(images: List[np.ndarray], vertical: bool = False):
    """
    Concatenates a list of images either vertically or horizontally.

    Args:
        images (List[np.ndarray]): List of images to concatenate.
        vertical (bool): Flag to concatenate vertically. Default is False.

    Returns:
        np.ndarray: Concatenated image.
        
    Raises:
        InvalidDimensionError: If images list is empty.
        InvalidShapeError: If images have incompatible shapes for concatenation.
        RenderingError: If concatenation fails.

    Example:
        combined_image = concat_images([image1, image2], vertical=True)
    """
    if not images:
        raise InvalidDimensionError(
            "Images list cannot be empty. "
            "Please provide at least one image to concatenate."
        )
    
    if len(images) == 1:
        return images[0]
    
    try:
        concated_image = images[0]
        for i, image in enumerate(images[1:], 1):
            if vertical:
                if concated_image.shape[1] != image.shape[1]:
                    raise InvalidShapeError(
                        f"Images must have same width for vertical concatenation. "
                        f"Image 0 width: {concated_image.shape[1]}, Image {i} width: {image.shape[1]}. "
                        f"Please resize images to have compatible dimensions."
                    )
                concated_image = np.concatenate([concated_image, image], 0)
            else:
                if concated_image.shape[0] != image.shape[0]:
                    raise InvalidShapeError(
                        f"Images must have same height for horizontal concatenation. "
                        f"Image 0 height: {concated_image.shape[0]}, Image {i} height: {image.shape[0]}. "
                        f"Please resize images to have compatible dimensions."
                    )
                concated_image = np.concatenate([concated_image, image], 1)
        return concated_image
    except InvalidShapeError:
        raise  # Re-raise our own exceptions
    except Exception as e:
        raise RenderingError(
            f"Failed to concatenate images: {e}. "
            f"Please check image formats and dimensions."
        ) from e


def generate_rainbow_colors(num_colors: int):
    """
    Generates a list of RGB colors transitioning smoothly through the rainbow colors.

    Args:
        num_colors (int): The number of colors to generate.

    Returns:
        list: A list of RGB colors, where each color is represented by a list of three integers [R, G, B].
        
    Raises:
        ValueError: If num_colors is not a positive integer.
        RenderingError: If color generation fails.

    Details:
    - The function interpolates colors between red, orange, yellow, green, blue, and violet.
    - The colors are generated in the RGB color space.

    Example:
        example_colors = generate_rainbow_colors(14)
        # example_colors might output:
        # [[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0], [255, 255, 0], [204, 255, 0], [153, 255, 0], [0, 255, 0], [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 51, 255], [148, 0, 211]]
    """
    if not isinstance(num_colors, int) or num_colors <= 0:
        raise ValueError(
            f"num_colors must be a positive integer, got {num_colors} of type {type(num_colors)}. "
            f"Please provide a positive number of colors to generate."
        )
    
    try:
        def interpolate_color(start_color, end_color, t):
            return start_color + (end_color - start_color) * t

        rainbow_colors = [
            np.array([255, 0, 0]),  # Red
            np.array([255, 127, 0]),  # Orange
            np.array([255, 255, 0]),  # Yellow
            np.array([0, 255, 0]),  # Green
            np.array([0, 0, 255]),  # Blue
            np.array([75, 0, 130]),  # Indigo
            np.array([148, 0, 211]),  # Violet
        ]

        total_segments = len(rainbow_colors) - 1
        colors = []

        for i in range(total_segments):
            start_color = rainbow_colors[i]
            end_color = rainbow_colors[i + 1]
            segment_colors = int(np.ceil(num_colors / total_segments))

            for t in np.linspace(0, 1, segment_colors, endpoint=False):
                colors.append(interpolate_color(start_color, end_color, t))

        if len(colors) < num_colors:
            colors.append(rainbow_colors[-1])
        elif len(colors) > num_colors:
            colors = colors[:num_colors]

        colors = [[int(c) for c in color] for color in colors]
        return colors
    except Exception as e:
        raise RenderingError(
            f"Failed to generate {num_colors} rainbow colors: {e}. "
            f"Please check the requested number of colors."
        ) from e


