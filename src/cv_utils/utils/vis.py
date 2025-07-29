"""
Module Name: vis.py

Description:
This module provides a collection of utility functions for image processing using numpy and OpenCV.
It includes functions for image conversion, drawing shapes, and visualizing images.

Author: Sehyun Cha
Email: cshyundev@gmail.com
Version: 0.2.1

License: MIT License

Usage:

import cv2 as cv
import numpy as np
from vis import convert_to_uint8_image, draw_circle

image = np.random.rand(100, 100, 3)
converted_image = convert_to_uint8_image(image)
image_with_circle = draw_circle(converted_image, (50, 50), radius=5)
"""

import random
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from ..exceptions import (
    VisualizationError,
    DisplayError,
    RenderingError,
    InvalidShapeError,
    InvalidDimensionError,
    ConversionError
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


def draw_circle(
    image: np.ndarray,
    pt2d: Tuple[int, int],
    radius: int = 1,
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
):
    """
    Draws a circle on the image.

    Args:
        image (np.ndarray, [H,W,3] or [H,W]): Input image array.
        pt2d (Tuple[int, int]): Center of the circle.
        radius (int): Radius of the circle. Default is 1.
        rgb (Optional[Tuple[int, int, int]]): Color of the circle in RGB. Default is random.
        thickness (int): Thickness of the circle outline. Default is 2.

    Returns:
        np.ndarray: Image with the drawn circle.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        RenderingError: If circle drawing fails.

    Example:
        image_with_circle = draw_circle(image, (50, 50), radius=5)
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if radius <= 0:
        raise ValueError(
            f"Radius must be positive, got {radius}. "
            f"Please provide a positive radius value."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        pt2d = (int(pt2d[0]), int(pt2d[1]))
        return cv.circle(image, pt2d, radius, rgb, thickness)
    except Exception as e:
        raise RenderingError(
            f"Failed to draw circle at {pt2d} with radius {radius}: {e}. "
            f"Please check point coordinates are within image bounds."
        ) from e


def draw_line_by_points(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[float, float],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draws a line between two points on the image.

    Args:
        image (np.ndarray): Input image array.
        pt1 (Tuple[int, int]): Starting point of the line.
        pt2 (Tuple[float, float]): Ending point of the line.
        rgb (Optional[Tuple[int, int, int]]): Color of the line in RGB. Default is random.
        thickness (int): Thickness of the line. Default is 2.

    Returns:
        np.ndarray: Image with the drawn line.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        RenderingError: If line drawing fails.

    Example:
        image_with_line = draw_line_by_points(image, (10, 10), (100, 100))
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        pt1_int = (int(pt1[0]), int(pt1[1]))
        pt2_int = (int(pt2[0]), int(pt2[1]))
        return cv.line(image, pt1_int, pt2_int, rgb, thickness)
    except Exception as e:
        raise RenderingError(
            f"Failed to draw line from {pt1} to {pt2}: {e}. "
            f"Please check point coordinates are valid."
        ) from e


def draw_line_by_line(
    image: np.ndarray,
    line: Tuple[float, float, float],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw Line by (a,b,c), which (a,b,c) means a line: ax + by + c = 0

    Args:
        image (np.ndarray, [H,W,3] or [H,W]): Input image array.
        line (Tuple[float], [3,]): line parameter (a,b,c)
        rgb (Tuple[int], [3,] ): RGB color
        thickness (int): thickness of the line

    Returns:
        np.ndarray: Image with the drawn line.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        ValueError: If line parameters are invalid.
        RenderingError: If line drawing fails.
        
    Example:
        image_with_line = draw_line_by_line(image, (1.0, -1.0, 50.0))
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if len(line) != 3:
        raise ValueError(
            f"Line parameters must be tuple of 3 values (a, b, c), got {len(line)} values. "
            f"Please provide line equation coefficients ax + by + c = 0."
        )
    
    a, b, c = line
    if a == 0 and b == 0:
        raise ValueError(
            f"Invalid line parameters: both a and b cannot be zero. "
            f"Please provide valid line equation coefficients."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        h, w = image.shape[:2]
        
        if b == 0.0:
            # Vertical line: x = -c/a
            x0 = x1 = int(-c / a)
            y0 = 0
            y1 = h
        else:
            # General line
            x0, y0 = map(int, [0, -c / b])
            x1, y1 = map(int, [w, -(c + a * w) / b])
        
        return cv.line(image, (x0, y0), (x1, y1), rgb, thickness)
    except Exception as e:
        raise RenderingError(
            f"Failed to draw line with parameters {line}: {e}. "
            f"Please check line parameters and image dimensions."
        ) from e


def draw_polygon(
    image: np.ndarray,
    pts: Union[List[Tuple[int, int]], np.ndarray],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 3,
) -> np.ndarray:
    """
    Draws a polygon on the image based on provided points using OpenCV.

    Args:
        image (np.ndarray): The input image on which the polygon will be drawn.
        pts (Union[List[Tuple[int, int]], np.ndarray], [N,2]): List of points (x, y) that define the vertices of the polygon.
        rgb (Optional[Tuple[int, int, int]]): Color of the polygon in RGB. Default is green.
        thickness (int): Thickness of the polygon lines. Default is 3.

    Returns:
        np.ndarray: The image with the drawn polygon.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        ValueError: If insufficient points or invalid point format.
        RenderingError: If polygon drawing fails.

    Example:
        image_with_polygon = draw_polygon(image, [(10, 10), (20, 20), (30, 10)])
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if len(pts) < 3:
        raise ValueError(
            f"Polygon requires at least 3 points, got {len(pts)} points. "
            f"Please provide at least 3 vertices to form a polygon."
        )
    
    if isinstance(pts, list):
        if not all(isinstance(pt, tuple) and len(pt) == 2 for pt in pts):
            raise ValueError(
                f"Each point must be a tuple of two numbers (x, y). "
                f"Please ensure all points are in format (x, y)."
            )
    
    try:
        points_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        
        if rgb is None:
            rgb = (0, 255, 0)  # default color = green
        cv.polylines(image, [points_array], isClosed=True, color=rgb, thickness=thickness)
        
        return image
    except Exception as e:
        raise RenderingError(
            f"Failed to draw polygon with {len(pts)} points: {e}. "
            f"Please check point coordinates and image dimensions."
        ) from e


def draw_lines(
    image,
    pts: List[Tuple[int, int]],
    rgb: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
):
    """
    Draws lines connecting a series of points on an image in the order.

    Args:
        image (np.ndarray): The image on which to draw the lines.
        pts (List[Tuple[int, int]]): List of (x, y) tuples representing the points.
        rgb (Optional[Tuple[int, int, int]]): Color of the line in RGB format. Default is random.
        thickness (int): Thickness of the lines. Default is 2.

    Returns:
        np.ndarray: The image with lines drawn on it.
        
    Raises:
        InvalidDimensionError: If image is not 2D or 3D array.
        ValueError: If insufficient points provided.
        RenderingError: If line drawing fails.

    Example:
        image_with_lines = draw_lines(image, [(10, 10), (20, 20), (30, 10)])
    """
    if not isinstance(image, np.ndarray) or len(image.shape) not in [2, 3]:
        raise InvalidDimensionError(
            f"Image must be 2D or 3D numpy array, got {type(image)} with shape {getattr(image, 'shape', 'unknown')}. "
            f"Please provide a valid image array."
        )
    
    if len(pts) < 2:
        raise ValueError(
            f"Need at least 2 points to draw lines, got {len(pts)} points. "
            f"Please provide at least 2 points to connect with lines."
        )
    
    try:
        if rgb is None:
            rgb = tuple(np.random.randint(0, 255, 3).tolist())
        for i in range(len(pts) - 1):
            pt1 = (int(pts[i][0]), int(pts[i][1]))
            pt2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
            cv.line(image, pt1, pt2, rgb, thickness)
        return image
    except Exception as e:
        raise RenderingError(
            f"Failed to draw lines connecting {len(pts)} points: {e}. "
            f"Please check point coordinates are valid."
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


def show_image(image: np.ndarray, title: str = "image"):
    """
    Displays an image with a title.

    Args:
        image (np.ndarray): Image to display.
        title (str): Title of the image. Default is "image".
        
    Raises:
        InvalidDimensionError: If image is not a valid array.
        DisplayError: If image display fails.

    Example:
        show_image(image, title="Example Image")
    """
    if not isinstance(image, np.ndarray):
        raise InvalidDimensionError(
            f"Image must be numpy array, got {type(image)}. "
            f"Please provide a valid image array."
        )
    
    try:
        plt.imshow(image)
        plt.title(title)
        plt.show()
    except Exception as e:
        raise DisplayError(
            f"Failed to display image '{title}': {e}. "
            f"Please check image format and display environment."
        ) from e


def show_two_images(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Left image",
    title2: str = "Right image",
):
    """
    Displays two images side by side.

    Args:
        image1 (np.ndarray): First image to display.
        image2 (np.ndarray): Second image to display.
        title1 (str): Title for the first image. Default is "Left image".
        title2 (str): Title for the second image. Default is "Right image".
        
    Raises:
        InvalidDimensionError: If images are not valid arrays.
        DisplayError: If image display fails.

    Details:
    - Each images can have different size.

    Example:
        show_two_images(image1, image2, title1="Image 1", title2="Image 2")
    """
    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        raise InvalidDimensionError(
            f"Both images must be numpy arrays, got {type(image1)} and {type(image2)}. "
            f"Please provide valid image arrays."
        )
    
    try:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image1)
        axes[0].set_title(title1)
        axes[0].axis("off")

        axes[1].imshow(image2)
        axes[1].set_title(title2)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise DisplayError(
            f"Failed to display two images '{title1}' and '{title2}': {e}. "
            f"Please check image formats and display environment."
        ) from e


def show_correspondences(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: List[Tuple[float, float]],
    pts2: List[Tuple[float, float]],
    margin_width: int = 20,
):
    """
    Plots corresponding points between two images with an optional white margin between them.

    Args:
        image1 (np.ndarray): First input image.
        image2 (np.ndarray): Second input image.
        pts1 (List[Tuple[float, float]]): Points in the first image.
        pts2 (List[Tuple[float, float]]): Points in the second image.
        margin_width (int): Width of the white margin between the images. Default is 20.
        
    Raises:
        InvalidDimensionError: If images are not valid arrays.
        ValueError: If point lists have different lengths.
        DisplayError: If correspondence display fails.

    Example:
        show_correspondences(image1, image2, pts1, pts2, margin_width=30)
    """
    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        raise InvalidDimensionError(
            f"Both images must be numpy arrays, got {type(image1)} and {type(image2)}. "
            f"Please provide valid image arrays."
        )
    
    if len(pts1) != len(pts2):
        raise ValueError(
            f"Point lists must have same length, got {len(pts1)} and {len(pts2)} points. "
            f"Please provide equal number of corresponding points."
        )
    
    if margin_width < 0:
        raise ValueError(
            f"Margin width must be non-negative, got {margin_width}. "
            f"Please provide a valid margin width."
        )
    
    try:
        height = image1.shape[0]
        white_margin = float_to_image(
            np.ones((height, margin_width)), 0.0, 1.0, color_map="gray"
        )

        combined_image = concat_images([image1, white_margin, image2], vertical=False)

        _, ax = plt.subplots()
        ax.imshow(combined_image, cmap="gray")
        ax.set_axis_off()

        offset = image1.shape[1] + margin_width

        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            color = "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            ax.plot([x1, x2 + offset], [y1, y2], linestyle="-", color=color)
            ax.plot(x1, y1, "o", mfc="none", mec=color, mew=2)
            ax.plot(x2 + offset, y2, "o", mfc="none", mec=color, mew=2)

        plt.show()
    except Exception as e:
        raise DisplayError(
            f"Failed to display correspondences between {len(pts1)} point pairs: {e}. "
            f"Please check image formats and point coordinates."
        ) from e
